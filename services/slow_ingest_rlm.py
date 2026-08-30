"""Slow and hybrid ingestion via RLM loops.

Provides:
- run_slow_ingest_chunk(): Mini-RLM loop for consciously reading one chunk
- run_slow_ingest(): Full-document slow ingestion orchestrator
- run_hybrid_ingest(): Fast first pass then slow on high-signal chunks
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from core.cognitive_memory_api import RelationshipType
from core.memory_repo import MemoryRepo
from services.hexis_rlm import (
    RlmLoopFailure,
    _make_sync_llm_query,
    _run_loop,
    find_final_answer,
    normalize_llm_config,
)
from services.ingest import (
    Appraisal,
    Config,
    DocumentInfo,
    Extraction,
    IngestionMetrics,
    IngestionMode,
    IngestionPipeline,
    Section,
    _hash_text,
    _word_count,
)
from services.prompt_resources import compose_compact_personhood_prompt, load_rlm_slow_ingest_prompt
from services.rlm_memory_env import RLMMemoryEnv, RLMWorkspace, WorkspaceBudgets
from services.rlm_repl import HexisLocalREPL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Assessment dataclass
# ---------------------------------------------------------------------------

_DEFAULT_ASSESSMENT: dict[str, Any] = {
    "acceptance": "question",
    "analysis": "Assessment could not be completed.",
    "emotional_reaction": {"valence": 0.0, "arousal": 0.2, "primary_emotion": "uncertain"},
    "worldview_impact": "neutral",
    "importance": 0.5,
    "trust_assessment": 0.5,
    "extracted_facts": [],
    "connections": [],
    "rejection_reasons": [],
}

_ASSESSMENT_KEYS = set(_DEFAULT_ASSESSMENT.keys())


def _require_source_document(doc: DocumentInfo) -> None:
    document_id = getattr(doc, "document_id", None)
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("slow/hybrid ingestion requires doc.document_id; call IngestionPipeline._ingest_content first")


def _safe_assessment(raw: Any) -> dict[str, Any]:
    """Ensure assessment has all required keys with safe defaults."""
    if not isinstance(raw, dict):
        return dict(_DEFAULT_ASSESSMENT)
    result = dict(_DEFAULT_ASSESSMENT)
    result.update({k: v for k, v in raw.items() if k in _ASSESSMENT_KEYS})
    # Validate acceptance enum
    if result["acceptance"] not in ("accept", "contest", "question"):
        result["acceptance"] = "question"
    # Clamp numeric fields
    result["importance"] = max(0.0, min(1.0, float(result.get("importance", 0.5))))
    result["trust_assessment"] = max(0.0, min(1.0, float(result.get("trust_assessment", 0.5))))
    return result


# ---------------------------------------------------------------------------
# Trust multipliers for acceptance levels
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-chunk RLM runner
# ---------------------------------------------------------------------------


async def run_slow_ingest_chunk(
    *,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    source_info: dict[str, Any],
    worldview_stubs: list[dict[str, Any]],
    emotional_state: dict[str, Any],
    goal_stubs: list[dict[str, Any]],
    llm_config: dict[str, Any],
    dsn: str,
    max_iterations: int = 6,
    timeout_seconds: int = 120,
    workspace_budgets: WorkspaceBudgets | None = None,
) -> dict[str, Any]:
    """Run a mini-RLM loop to consciously read and assess one chunk.

    Returns an assessment dict with keys: acceptance, analysis,
    emotional_reaction, worldview_impact, importance, trust_assessment,
    extracted_facts, connections, rejection_reasons.
    """
    time_start = time.perf_counter()
    llm_cfg = normalize_llm_config(llm_config)

    repo = MemoryRepo(dsn)
    budgets = workspace_budgets or WorkspaceBudgets(
        max_loaded_memories=15,
        max_loaded_chars=10_000,
    )

    context_payload = {
        "chunk_text": chunk_text,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "source": source_info,
        "worldview": worldview_stubs,
        "emotional_state": emotional_state,
        "goals": goal_stubs,
    }

    workspace = RLMWorkspace(
        task="slow_ingest_chunk",
        turn_snapshot=context_payload,
        budgets=budgets,
    )

    loop = asyncio.get_running_loop()
    llm_query_fn = _make_sync_llm_query(llm_cfg, loop)
    memory_env = RLMMemoryEnv(repo, workspace, llm_query_fn=llm_query_fn)

    repl = HexisLocalREPL()
    repl.setup(
        context_payload=context_payload,
        memory_env=memory_env,
        llm_query_fn=llm_query_fn,
    )

    system_prompt = load_rlm_slow_ingest_prompt()
    personhood_addendum = compose_compact_personhood_prompt("ingest")
    if personhood_addendum:
        system_prompt = system_prompt + "\n\n---\n\n" + personhood_addendum

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                _run_loop,
                repl,
                llm_cfg,
                loop,
                system_prompt,
                max_iterations,
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "slow_ingest_chunk timed out chunk=%d/%d after %ds",
            chunk_index + 1,
            total_chunks,
            timeout_seconds,
        )
        duration = time.perf_counter() - time_start
        return {
            "assessment": dict(_DEFAULT_ASSESSMENT),
            "metrics": {
                "iterations": 0,
                "message_count": 0,
                "timed_out": True,
                "duration_seconds": round(duration, 2),
            },
        }
    except RlmLoopFailure as exc:
        # #111: a total LLM failure -- same honest degrade as the timeout
        # branch above (a default, conservative assessment) rather than
        # letting a fabricated decision or an unhandled crash reach the
        # ingestion pipeline.
        logger.error(
            "slow_ingest_chunk got no usable LLM response chunk=%d/%d: %s",
            chunk_index + 1,
            total_chunks,
            exc,
        )
        duration = time.perf_counter() - time_start
        return {
            "assessment": dict(_DEFAULT_ASSESSMENT),
            "metrics": {
                "iterations": 0,
                "message_count": 0,
                "llm_error": str(exc),
                "duration_seconds": round(duration, 2),
            },
        }
    finally:
        repl.cleanup()
        repo.close()

    raw_answer = result.get("final_answer", "")

    # Parse JSON assessment from FINAL answer
    try:
        parsed = json.loads(raw_answer) if raw_answer else {}
    except (json.JSONDecodeError, TypeError):
        parsed = {}

    assessment = _safe_assessment(parsed)
    duration = time.perf_counter() - time_start

    logger.info(
        "slow_ingest_chunk complete chunk=%d/%d acceptance=%s importance=%.2f "
        "iterations=%d duration=%.1fs",
        chunk_index + 1,
        total_chunks,
        assessment["acceptance"],
        assessment["importance"],
        result.get("iterations", 0),
        duration,
    )

    return {
        "assessment": assessment,
        "metrics": {
            "iterations": result.get("iterations", 0),
            "message_count": result.get("message_count", 0),
            "search_count": workspace.metrics.search_count,
            "fetch_count": workspace.metrics.fetch_count,
            "fetched_chars_total": workspace.metrics.fetched_chars_total,
            "timed_out": False,
            "duration_seconds": round(duration, 2),
        },
    }


# ---------------------------------------------------------------------------
# Full-document slow ingest orchestrator
# ---------------------------------------------------------------------------


async def run_slow_ingest(
    *,
    pipeline: IngestionPipeline,
    doc: DocumentInfo,
    sections: list[Section],
    llm_config: dict[str, Any],
    dsn: str,
    workspace_budgets: WorkspaceBudgets | None = None,
) -> dict[str, Any]:
    """Orchestrate slow ingestion of a full document.

    Each section gets a mini-RLM loop for conscious reading and assessment.
    Returns summary dict with memories_created, chunks_processed, assessments.
    """
    time_start = time.perf_counter()

    # Ensure store is connected
    if pipeline.store.client is None:
        await pipeline.store.connect()
    _require_source_document(doc)

    # Fetch appraisal context (worldview, emotional state, goals)
    ctx = await pipeline.store.fetch_appraisal_context()
    worldview_stubs = ctx.get("worldview", [])
    emotional_state = ctx.get("emotional_state", {})
    goal_stubs = ctx.get("goals", [])

    source_info = pipeline._source_payload(doc)

    # Receipt gate (#85): doc-complete skips; receipted chunks resume; the
    # enc: sentinel reuses the encounter across attempts.
    doc_ref = doc.content_hash
    chunk_hashes = {section.index: _hash_text(section.content) for section in sections}
    receipts = await pipeline.store.get_receipts(
        doc_ref, [doc_ref, f"enc:{doc_ref}"] + list(chunk_hashes.values())
    )
    if doc_ref in receipts:
        return {"memories_created": 0, "chunks_processed": 0, "assessments": [], "skipped": True}

    encounter_id = receipts.get(f"enc:{doc_ref}")
    if encounter_id is None:
        encounter_appraisal = Appraisal(
            primary_emotion="curious",
            intensity=0.4,
            curiosity=0.6,
            summary=f"Beginning conscious reading of '{doc.title}'.",
        )
        encounter_id = await pipeline._create_encounter_memory(doc, encounter_appraisal, IngestionMode.SLOW)
        await pipeline.store.record_receipt(
            doc_ref, f"enc:{doc_ref}", memory_id=encounter_id, source_path=doc.path
        )

    memories_created: list[str] = []
    assessments: list[dict[str, Any]] = []
    chunks_processed = 0
    total_chunks = len(sections)

    for section in sections:
        if pipeline._skip_section(section.title):
            continue
        if chunk_hashes[section.index] in receipts:
            continue  # resumed: this chunk already persisted

        chunk_result = await run_slow_ingest_chunk(
            chunk_text=section.extraction_view(),
            chunk_index=section.index,
            total_chunks=total_chunks,
            source_info=source_info,
            worldview_stubs=worldview_stubs,
            emotional_state=emotional_state,
            goal_stubs=goal_stubs,
            llm_config=llm_config,
            dsn=dsn,
            workspace_budgets=workspace_budgets,
        )

        assessment = chunk_result["assessment"]
        assessments.append(assessment)
        chunks_processed += 1

        # Update emotional state for subsequent chunks
        emo = assessment.get("emotional_reaction", {})
        if isinstance(emo, dict) and "valence" in emo:
            emotional_state = emo

        # Create semantic memories from extracted facts
        acceptance = assessment["acceptance"]

        # Build enriched source payload
        chunk_source = dict(source_info)
        chunk_source["section_hash"] = chunk_hashes[section.index]
        chunk_id = doc.chunk_ids.get(section.index)
        if chunk_id:
            chunk_source["chunk_id"] = chunk_id
            chunk_source["chunk_index"] = section.index
        chunk_source["conscious_analysis"] = assessment.get("analysis", "")
        chunk_source["acceptance"] = acceptance
        if acceptance == "contest":
            chunk_source["contested"] = True
        elif acceptance == "question":
            chunk_source["questioned"] = True

        facts = assessment.get("extracted_facts", [])
        connection_ids = assessment.get("connections", [])
        rejection_reason_ids = assessment.get("rejection_reasons", [])

        valid_facts = [
            fact for fact in facts
            if fact and isinstance(fact, str) and len(fact.strip()) >= 10
        ]
        # The whole fact-persistence pass is atomic in the DB (db/66
        # slow_ingest_persist_facts): routing, corroboration via the audited
        # belief-revision policy, creation, and every edge kind. Trust
        # derives from source attributions (#83); the acceptance stance is
        # recorded as edges and metadata.
        worldview_edge_ids = [
            wv.get("memory_id") or wv.get("id") for wv in worldview_stubs[:3]
        ]
        result = await pipeline.store.persist_slow_facts(
            valid_facts,
            assessment,
            chunk_source,
            encounter_id=encounter_id,
            connection_ids=connection_ids,
            worldview_ids=worldview_edge_ids,
            rejection_reason_ids=rejection_reason_ids,
            context="slow_ingest",
        )
        memories_created.extend(str(m) for m in (result.get("created") or []))

    # Doc-complete: the final receipt — every earlier crash point resumes.
    await pipeline.store.record_receipt(
        doc_ref, doc_ref, memories_created=len(memories_created), source_path=doc.path
    )

    duration = time.perf_counter() - time_start
    logger.info(
        "slow_ingest complete doc=%s memories=%d chunks=%d duration=%.1fs",
        doc.title,
        len(memories_created),
        chunks_processed,
        duration,
    )

    return {
        "memories_created": len(memories_created),
        "memory_ids": memories_created,
        "chunks_processed": chunks_processed,
        "assessments": assessments,
        "encounter_id": encounter_id,
        "duration_seconds": round(duration, 2),
    }


# ---------------------------------------------------------------------------
# Hybrid ingest orchestrator
# ---------------------------------------------------------------------------


async def run_hybrid_ingest(
    *,
    pipeline: IngestionPipeline,
    doc: DocumentInfo,
    sections: list[Section],
    llm_config: dict[str, Any],
    dsn: str,
    workspace_budgets: WorkspaceBudgets | None = None,
) -> dict[str, Any]:
    """Hybrid ingestion: fast first pass to score chunks, slow only for high-signal.

    A chunk is "high-signal" if any of:
    - Importance > 0.7 from fast extraction
    - Vector similarity to worldview memory > 0.6
    - Vector similarity to active goal > 0.6

    Returns summary dict with memories_created, chunks_processed, slow_chunks, fast_chunks.
    """
    time_start = time.perf_counter()

    if pipeline.store.client is None:
        await pipeline.store.connect()
    _require_source_document(doc)

    ctx = await pipeline.store.fetch_appraisal_context()
    worldview_stubs = ctx.get("worldview", [])
    emotional_state = ctx.get("emotional_state", {})
    goal_stubs = ctx.get("goals", [])

    source_info = pipeline._source_payload(doc)

    # Receipt gate (#85), same shape as the slow path.
    doc_ref = doc.content_hash
    chunk_hashes = {section.index: _hash_text(section.content) for section in sections}
    receipts = await pipeline.store.get_receipts(
        doc_ref, [doc_ref, f"enc:{doc_ref}"] + list(chunk_hashes.values())
    )
    if doc_ref in receipts:
        return {
            "memories_created": 0, "chunks_processed": 0, "slow_chunks": 0,
            "fast_chunks": 0, "assessments": [], "skipped": True,
        }

    encounter_id = receipts.get(f"enc:{doc_ref}")
    if encounter_id is None:
        encounter_appraisal = Appraisal(
            primary_emotion="curious",
            intensity=0.4,
            curiosity=0.6,
            summary=f"Beginning hybrid reading of '{doc.title}'.",
        )
        encounter_id = await pipeline._create_encounter_memory(doc, encounter_appraisal, IngestionMode.HYBRID)
        await pipeline.store.record_receipt(
            doc_ref, f"enc:{doc_ref}", memory_id=encounter_id, source_path=doc.path
        )

    # Phase 1: Fast pass -- use the pipeline's KnowledgeExtractor to score all chunks
    fast_extractions: dict[int, list[Extraction]] = {}
    chunk_importance: dict[int, float] = {}

    appraisal = Appraisal(primary_emotion="neutral", intensity=0.2)

    for section in sections:
        if pipeline._skip_section(section.title):
            continue

        try:
            extractions = await pipeline.extractor.extract(
                section=section,
                doc=doc,
                appraisal=appraisal,
                mode=IngestionMode.FAST,
                max_items=pipeline.config.max_facts_per_section,
            )
            fast_extractions[section.index] = extractions
            max_imp = max((e.importance for e in extractions), default=0.0)
            chunk_importance[section.index] = max_imp
        except Exception as e:
            logger.warning("Fast extraction failed for section %d: %s", section.index, e)
            fast_extractions[section.index] = []
            chunk_importance[section.index] = 0.0

    # Phase 2: Identify high-signal chunks
    high_signal_indices: set[int] = set()

    for section in sections:
        if section.index not in chunk_importance:
            continue
        # Criterion 1: High importance from extraction
        if chunk_importance[section.index] > 0.7:
            high_signal_indices.add(section.index)
            continue

        # Criterion 2/3: Similarity to worldview or goals. Query those types
        # directly — the old semantic-only recall could never return them.
        try:
            similar = await pipeline.store.recall_similar(
                section.content[:500], memory_types=["worldview", "goal"], limit=3
            )
            if any(
                mem.similarity is not None and mem.similarity >= 0.6
                for mem in similar
            ):
                high_signal_indices.add(section.index)
        except Exception:
            pass

    logger.info(
        "hybrid_ingest triage: %d/%d chunks flagged as high-signal",
        len(high_signal_indices),
        len(sections),
    )

    # Phase 3: Process chunks
    memories_created: list[str] = []
    slow_chunk_count = 0
    fast_chunk_count = 0
    assessments: list[dict[str, Any]] = []
    total_chunks = len(sections)

    for section in sections:
        if pipeline._skip_section(section.title):
            continue

        if section.index in high_signal_indices:
            # Slow path: RLM loop
            chunk_result = await run_slow_ingest_chunk(
                chunk_text=section.extraction_view(),
                chunk_index=section.index,
                total_chunks=total_chunks,
                source_info=source_info,
                worldview_stubs=worldview_stubs,
                emotional_state=emotional_state,
                goal_stubs=goal_stubs,
                llm_config=llm_config,
                dsn=dsn,
                workspace_budgets=workspace_budgets,
            )

            assessment = chunk_result["assessment"]
            assessments.append(assessment)
            slow_chunk_count += 1

            # Update emotional state
            emo = assessment.get("emotional_reaction", {})
            if isinstance(emo, dict) and "valence" in emo:
                emotional_state = emo

            # Create memories from slow assessment (same logic as run_slow_ingest)
            acceptance = assessment["acceptance"]

            chunk_source = dict(source_info)
            chunk_source["section_hash"] = chunk_hashes[section.index]
            chunk_id = doc.chunk_ids.get(section.index)
            if chunk_id:
                chunk_source["chunk_id"] = chunk_id
                chunk_source["chunk_index"] = section.index
            chunk_source["conscious_analysis"] = assessment.get("analysis", "")
            chunk_source["acceptance"] = acceptance
            if acceptance == "contest":
                chunk_source["contested"] = True
            elif acceptance == "question":
                chunk_source["questioned"] = True

            hybrid_facts = [
                fact for fact in assessment.get("extracted_facts", [])
                if fact and isinstance(fact, str) and len(fact.strip()) >= 10
            ]
            hybrid_result = await pipeline.store.persist_slow_facts(
                hybrid_facts,
                assessment,
                chunk_source,
                encounter_id=encounter_id,
                connection_ids=None,
                worldview_ids=None,
                rejection_reason_ids=assessment.get("rejection_reasons", []),
                context="hybrid_ingest",
            )
            memories_created.extend(str(m) for m in (hybrid_result.get("created") or []))

        else:
            # Fast path: use pre-extracted facts from Phase 1
            fast_chunk_count += 1
            extractions = fast_extractions.get(section.index, [])
            created = await pipeline._create_semantic_memories(
                doc, encounter_id, appraisal, extractions,
                section_hash=chunk_hashes[section.index],
                chunk_index=section.index,
            )
            memories_created.extend(created)

    await pipeline.store.record_receipt(
        doc_ref, doc_ref, memories_created=len(memories_created), source_path=doc.path
    )

    duration = time.perf_counter() - time_start
    logger.info(
        "hybrid_ingest complete doc=%s memories=%d slow=%d fast=%d duration=%.1fs",
        doc.title,
        len(memories_created),
        slow_chunk_count,
        fast_chunk_count,
        duration,
    )

    return {
        "memories_created": len(memories_created),
        "memory_ids": memories_created,
        "chunks_processed": slow_chunk_count + fast_chunk_count,
        "slow_chunks": slow_chunk_count,
        "fast_chunks": fast_chunk_count,
        "assessments": assessments,
        "encounter_id": encounter_id,
        "duration_seconds": round(duration, 2),
    }
