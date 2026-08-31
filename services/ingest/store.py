"""Hexis ingestion — split from the former services/ingest.py (#89).
Module: store.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import UUID

from core.cognitive_memory_api import (
    CognitiveMemory,
    MemoryInput as ApiMemoryInput,
    MemoryType as ApiMemoryType,
    RelationshipType,
)

from .config import Config, IngestionMetrics

# =========================================================================
# STORAGE
# =========================================================================


class MemoryStore:
    """Async-native store (#88): one event loop — the caller's — over an
    asyncpg pool and the async CognitiveMemory client. The former sync
    wrapper drove a private loop per call through CognitiveMemorySync's
    internals; every method is now a plain coroutine."""

    def __init__(self, config: Config):
        self.config = config
        self.client: CognitiveMemory | None = None
        self._pool: Any = None

    async def connect(self) -> None:
        if self.client is not None:
            return
        import asyncpg

        if self.config.dsn:
            dsn = self.config.dsn
        else:
            dsn = (
                f"postgresql://{self.config.db_user}:{self.config.db_password}"
                f"@{self.config.db_host}:{self.config.db_port}/{self.config.db_name}"
            )
        self._pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
        self.client = CognitiveMemory(self._pool)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
        self._pool = None
        self.client = None

    async def _exec(self, sql: str, *params: Any) -> Any:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            return await conn.execute(sql, *params)

    async def _fetchval(self, sql: str, *params: Any) -> Any:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            return await conn.fetchval(sql, *params)

    async def get_receipts(self, doc_ref: str, hashes: list[str]) -> dict[str, Any]:
        """Receipt lookup (#85/#90): the ingestion_receipts table UNION'd with
        legacy whole-document attributions, via SQL get_ingestion_receipts."""
        if self.client is None:
            await self.connect()
        raw = await self._fetchval(
            "SELECT get_ingestion_receipts($1, $2::text[])", doc_ref, hashes
        )
        doc = json.loads(raw) if isinstance(raw, str) else (raw or {})
        return doc if isinstance(doc, dict) else {}

    async def record_receipt(
        self,
        doc_ref: str,
        section_hash: str,
        *,
        memory_id: str | None = None,
        memories_created: int = 0,
        source_path: str | None = None,
    ) -> None:
        if self.client is None:
            await self.connect()
        await self._exec(
            "SELECT record_ingestion_receipt($1, $2, $3::uuid, $4, $5)",
            doc_ref, section_hash, memory_id, memories_created, source_path,
        )

    async def store_source_document(
        self,
        *,
        title: str,
        source_type: str,
        content_hash: str,
        path: str | None,
        file_type: str | None,
        content: str,
        word_count: int,
        source_attribution: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist the exact raw source text before chunk-level ingestion."""
        if self.client is None:
            await self.connect()
        raw = await self._fetchval(
            """
            SELECT upsert_source_document(
                $1::text, $2::text, $3::text, $4::text, $5::text,
                $6::text, $7::int, $8::jsonb, $9::jsonb
            )
            """,
            title,
            source_type,
            content_hash,
            path,
            file_type,
            content,
            int(word_count),
            json.dumps(source_attribution),
            json.dumps(metadata or {}),
        )
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        return parsed if isinstance(parsed, dict) else {}

    async def upsert_source_document_chunks(
        self,
        document_id: str,
        sections: list[Any],
        chunker_version: str,
    ) -> dict[str, Any]:
        """Persist the durable chunk set for a document (db/83
        upsert_source_document_chunks): keep-if-unchanged ids/embeddings,
        trailing trim, locators for citation."""
        if self.client is None:
            await self.connect()
        payload = []
        for s in sections:
            entry: dict[str, Any] = {
                "chunk_index": s.index,
                "locator_kind": s.locator_kind,
                "locator": s.locator(),
                "heading_path": list(s.heading_path or []),
                "content": s.content,
                "char_start": s.char_start,
                "char_end": s.char_end,
                "metadata": {"title": s.title} if s.title else {},
            }
            if s.page_start is not None:
                entry["page_start"] = s.page_start
            if s.page_end is not None:
                entry["page_end"] = s.page_end
            if s.sheet_name is not None:
                entry["sheet_name"] = s.sheet_name
            if s.row_start is not None:
                entry["row_start"] = s.row_start
            if s.row_end is not None:
                entry["row_end"] = s.row_end
            payload.append(entry)
        raw = await self._fetchval(
            "SELECT upsert_source_document_chunks($1::uuid, $2::jsonb, $3::text)",
            document_id,
            json.dumps(payload),
            chunker_version,
        )
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        result = parsed if isinstance(parsed, dict) else {}
        if result.get("error"):
            raise RuntimeError(f"upsert_source_document_chunks: {result['error']}")
        return result

    async def load_source_documents_to_recmem(
        self,
        document_ids: list[str],
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Place preserved source documents on the RecMem desk.

        This delegates to the DB-owned loader so chunking, idempotency, access
        stamps, and cleanup metadata stay in Postgres.
        """
        if self.client is None:
            await self.connect()
        raw = await self._fetchval(
            """
            SELECT load_source_documents_to_recmem(
                $1::uuid[], NULL::text[], NULL::text[], 0,
                NULL::int, NULL::int, NULL::int, FALSE, $2::text
            )
            """,
            document_ids,
            reason,
        )
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        result = parsed if isinstance(parsed, dict) else {}
        if result.get("error"):
            raise RuntimeError(f"load_source_documents_to_recmem: {result['error']}")
        return result

    async def upsert_source_artifact(
        self,
        *,
        sha256: str,
        storage_kind: str,
        data: bytes | None = None,
        storage_ref: str | None = None,
        source_document_id: str | None = None,
        original_filename: str | None = None,
        mime_type: str | None = None,
        byte_size: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Preserve the original artifact (db/74 upsert_source_artifact):
        sha256-deduped, never rewrites bytes, links the document once known."""
        if self.client is None:
            await self.connect()
        raw = await self._fetchval(
            """
            SELECT upsert_source_artifact(
                $1::text, $2::text, $3::bytea, $4::text, $5::uuid,
                $6::text, $7::text, $8::bigint, $9::jsonb
            )
            """,
            sha256,
            storage_kind,
            data,
            storage_ref,
            source_document_id,
            original_filename,
            mime_type,
            byte_size,
            json.dumps(metadata or {}),
        )
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        return parsed if isinstance(parsed, dict) else {}

    async def record_extraction_run(
        self,
        *,
        document_id: str | None,
        artifact_id: str | None,
        extractor_name: str,
        extractor_version: str = "",
        status: str = "completed",
        warnings: list[dict[str, Any]] | None = None,
        errors: list[dict[str, Any]] | None = None,
        started_at: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record one extractor run with structured warnings (db/74
        record_source_extraction_run)."""
        if self.client is None:
            await self.connect()
        raw = await self._fetchval(
            """
            SELECT record_source_extraction_run(
                $1::uuid, $2::uuid, $3::text, $4::text, $5::text,
                $6::jsonb, $7::jsonb, $8::timestamptz, $9::jsonb
            )
            """,
            document_id,
            artifact_id,
            extractor_name,
            extractor_version,
            status,
            json.dumps(warnings or []),
            json.dumps(errors or []),
            started_at,
            json.dumps(metadata or {}),
        )
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        return parsed if isinstance(parsed, dict) else {}

    async def set_original_hash(self, document_id: str, original_hash: str) -> None:
        """For sources with no separate artifact (pasted text), the content
        itself is the original — record its hash once."""
        if self.client is None:
            await self.connect()
        await self._exec(
            """
            UPDATE source_documents
            SET original_hash = $2
            WHERE id = $1::uuid AND status <> 'redacted' AND original_hash IS NULL
            """,
            document_id,
            original_hash,
        )

    async def create_encounter_memory(
        self,
        *,
        text: str,
        source: dict[str, Any],
        emotional_valence: float,
        context: dict[str, Any] | None,
        importance: float,
    ) -> str:
        if self.client is None:
            await self.connect()
        assert self.client is not None
        memory_id = await self.client.remember(
            text,
            type=ApiMemoryType.EPISODIC,
            importance=importance,
            emotional_valence=emotional_valence,
            context=context,
            source_attribution=source,
        )
        return str(memory_id)

    async def create_semantic_memory(
        self,
        *,
        content: str,
        confidence: float,
        category: str,
        related_concepts: list[str],
        source: dict[str, Any],
        importance: float,
        trust: float | None,
    ) -> str:
        if self.client is None:
            await self.connect()
        payload_sources = json.dumps([source])
        return str(
            await self._fetchval(
                "SELECT create_semantic_memory($1::text,$2::float,$3::text[],$4::text[],$5::jsonb,$6::float,$7::jsonb,$8::float)",
                content,
                confidence,
                [category],
                related_concepts,
                payload_sources,
                importance,
                json.dumps(source),
                trust,
            )
        )

    async def add_source(self, memory_id: str, source: dict[str, Any]) -> None:
        if self.client is None:
            await self.connect()
        assert self.client is not None
        await self.client.add_source(UUID(memory_id), source)

    async def add_evidence(
        self,
        memory_id: str,
        stance: str,
        source: dict[str, Any],
        note: str | None = None,
        evidence_memory_id: str | None = None,
        context: str = "ingest",
    ) -> dict[str, Any]:
        """Attach evidence to an existing memory through the DB-owned belief
        revision policy (db/59): source merge, SUPPORTS/CONTRADICTS edge,
        calibrated confidence update, and an audit row. Returns the revision
        result ({prior, posterior, applied, reason, ...})."""
        if self.client is None:
            await self.connect()
        raw = await self._fetchval(
            "SELECT add_memory_evidence($1::uuid, $2::text, $3::jsonb, $4::text, $5::uuid, $6::text)",
            memory_id,
            stance,
            json.dumps(source),
            note,
            evidence_memory_id,
            context,
        )
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        return parsed if isinstance(parsed, dict) else {}

    async def link_concept(self, memory_id: str, concept: str, strength: float = 1.0) -> None:
        """Link a memory to a concept in the knowledge graph."""
        if self.client is None:
            await self.connect()
        await self._fetchval(
            "SELECT link_memory_to_concept($1::uuid, $2::text, $3::float)",
            memory_id,
            concept,
            strength,
        )

    async def link_concepts_batch(self, pairs: list[tuple[str, str]], strength: float = 1.0) -> None:
        """Link multiple (memory_id, concept) pairs in a single batch call."""
        if not pairs:
            return
        if self.client is None:
            await self.connect()
        assert self.client is not None

        async with self._pool.acquire() as conn:
            await conn.executemany(
                "SELECT link_memory_to_concept($1::uuid, $2::text, $3::float)",
                [(mid, concept, strength) for mid, concept in pairs],
            )

    async def connect_memories_batch(self, edges: list[tuple[str, str, "RelationshipType", float]]) -> None:
        """Create multiple memory relationships in a single batch call."""
        if not edges:
            return
        if self.client is None:
            await self.connect()
        assert self.client is not None
        from core.cognitive_memory_api import RelationshipInput
        rels = [
            RelationshipInput(
                from_id=UUID(from_id),
                to_id=UUID(to_id),
                relationship_type=rel_type,
                confidence=conf,
            )
            for from_id, to_id, rel_type, conf in edges
        ]
        await self.client.connect_batch(rels)

    async def prefetch_embeddings(self, texts: list[str]) -> int:
        """Pre-warm embedding cache for a batch of texts.

        Calls the SQL ``prefetch_embeddings()`` function which batches HTTP
        requests to the embedding service (default batch size 8) and caches
        results.  Subsequent ``recall_similar_semantic`` / ``create_semantic_memory``
        calls for the same content become cache hits.
        """
        if not texts:
            return 0
        if self.client is None:
            await self.connect()
        return await self._fetchval("SELECT prefetch_embeddings($1::text[])", texts) or 0

    async def recall_similar_semantic(self, query: str, limit: int = 5):
        if self.client is None:
            await self.connect()
        assert self.client is not None
        return await self.client.recall(
            query,
            limit=limit,
            memory_types=[ApiMemoryType.SEMANTIC],
        ).memories

    async def recall_similar(self, query: str, memory_types: list[str], limit: int = 5):
        """Recall nearest memories of the given types (list result)."""
        if self.client is None:
            await self.connect()
        assert self.client is not None
        return await self.client.recall(
            query,
            limit=limit,
            memory_types=[ApiMemoryType(t) for t in memory_types],
        ).memories

    async def route_extractions(self, extractions: list, min_confidence: float) -> list:
        """Route extractions through the DB dedup/related/create policy
        (db/41 ingest_route_extractions): config-driven thresholds + one batched
        nearest-neighbor search. Returns a per-extraction plan with 'index',
        'decision' (duplicate|related|create) and 'matched_memory_id'."""
        if self.client is None:
            await self.connect()
        payload = json.dumps([
            {"content": ext.content, "confidence": ext.confidence}
            for ext in extractions
        ])
        raw = await self._fetchval(
            "SELECT ingest_route_extractions($1::jsonb, $2::float)", payload, min_confidence
        )
        plan = json.loads(raw) if isinstance(raw, str) else raw
        return plan or []

    async def persist_extractions(
        self,
        extractions: list,
        source: dict[str, Any],
        *,
        encounter_id: str | None,
        intensity: float,
        min_confidence: float,
        min_importance_floor: float | None,
        base_trust: float | None,
        permanent: bool,
    ) -> dict[str, Any]:
        """The whole post-LLM persistence pass, atomic in the DB (db/66
        ingest_persist_extractions): route -> corroborate/create -> concept
        links -> worldview edges -> provenance edges -> decay."""
        if self.client is None:
            await self.connect()
        payload = json.dumps([
            {
                "content": ext.content,
                "confidence": ext.confidence,
                "importance": ext.importance,
                "category": ext.category,
                "concepts": list(ext.concepts or []),
                "connections": list(ext.connections or []),
                "supports": ext.supports,
                "contradicts": ext.contradicts,
            }
            for ext in extractions
        ])
        raw = await self._fetchval(
            "SELECT ingest_persist_extractions($1::jsonb, $2::jsonb, $3::uuid, $4::float, $5::jsonb)",
            payload,
            json.dumps(source),
            encounter_id,
            float(intensity),
            json.dumps({
                "min_confidence": min_confidence,
                "min_importance_floor": min_importance_floor,
                "base_trust": base_trust,
                "permanent": permanent,
            }),
        )
        result = json.loads(raw) if isinstance(raw, str) else raw
        return result if isinstance(result, dict) else {}

    async def persist_slow_facts(
        self,
        facts: list[str],
        assessment: dict[str, Any],
        source: dict[str, Any],
        *,
        encounter_id: str | None,
        connection_ids: list[str] | None = None,
        worldview_ids: list[str] | None = None,
        rejection_reason_ids: list[str] | None = None,
        context: str = "slow_ingest",
    ) -> dict[str, Any]:
        """Atomic slow/hybrid ingest fact persistence (db/66
        slow_ingest_persist_facts): route -> corroborate/create -> provenance,
        association, worldview, and contested edges."""
        if self.client is None:
            await self.connect()

        def _uuids(vals):
            out = []
            for v in vals or []:
                try:
                    out.append(str(UUID(str(v))))
                except (ValueError, AttributeError, TypeError):
                    continue
            return out

        raw = await self._fetchval(
            "SELECT slow_ingest_persist_facts($1::jsonb, $2::jsonb, $3::jsonb,"
            " $4::uuid, $5::uuid[], $6::uuid[], $7::uuid[], $8::text)",
            json.dumps(list(facts)),
            json.dumps(assessment),
            json.dumps(source),
            encounter_id,
            _uuids(connection_ids),
            _uuids(worldview_ids),
            _uuids(rejection_reason_ids),
            context,
        )
        result = json.loads(raw) if isinstance(raw, str) else raw
        return result if isinstance(result, dict) else {}

    async def apply_ingest_decay(self, memory_id: str, intensity: float, permanent: bool) -> None:
        """Decay policy is DB-owned (db/66 decay_rate_for_intensity)."""
        if self.client is None:
            await self.connect()
        try:
            await self._exec(
                "UPDATE memories SET decay_rate = CASE WHEN $3 THEN 0.0"
                " ELSE decay_rate_for_intensity($2) END WHERE id = $1::uuid",
                memory_id,
                float(intensity),
                bool(permanent),
            )
        except Exception:
            pass

    async def route_texts(self, items: list[tuple[str, float]], min_confidence: float = 0.0) -> list:
        """Route bare (content, confidence) pairs through the same DB
        dedup/related/create policy as route_extractions."""
        if not items:
            return []
        if self.client is None:
            await self.connect()
        payload = json.dumps([
            {"content": content, "confidence": confidence}
            for content, confidence in items
        ])
        raw = await self._fetchval(
            "SELECT ingest_route_extractions($1::jsonb, $2::float)", payload, min_confidence
        )
        plan = json.loads(raw) if isinstance(raw, str) else raw
        return plan or []

    async def connect_memories(self, from_id: str, to_id: str, relationship: RelationshipType, confidence: float = 0.8) -> None:
        if self.client is None:
            await self.connect()
        assert self.client is not None
        await self.client.connect_memories(
            from_id,
            to_id,
            relationship,
            confidence=confidence,
        )

    async def update_decay_rate(self, memory_id: str, decay_rate: float) -> None:
        if self.client is None:
            await self.connect()
        try:
            await self._exec("UPDATE memories SET decay_rate = $1 WHERE id = $2::uuid", decay_rate, memory_id)
        except Exception:
            pass

    async def fetch_appraisal_context(self) -> dict[str, Any]:
        if self.client is None:
            await self.connect()
        try:
            raw = await self._fetchval(
                """
                SELECT jsonb_build_object(
                    'emotional_state', get_current_affective_state(),
                    'goals', get_goals_snapshot(),
                    'worldview', get_worldview_context(),
                    'recent_memories', get_recent_context(5)
                )
                """
            )
            if isinstance(raw, str):
                return json.loads(raw)
            if isinstance(raw, dict):
                return raw
        except Exception:
            return {}
        return {}

    async def store_metrics(self, metrics: "IngestionMetrics") -> None:
        """Store ingestion metrics for observability."""
        if self.client is None:
            await self.connect()
        try:
            await self._exec(
                """
                INSERT INTO ingestion_metrics (
                    source_type, source_size_bytes, word_count, mode,
                    appraisal_valence, appraisal_arousal, appraisal_emotion, appraisal_intensity,
                    extraction_count, dedup_count, memory_count, llm_calls,
                    duration_seconds, errors
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb
                )
                """,
                metrics.source_type,
                metrics.source_size_bytes,
                metrics.word_count,
                metrics.mode,
                metrics.appraisal_valence,
                metrics.appraisal_arousal,
                metrics.appraisal_emotion,
                metrics.appraisal_intensity,
                metrics.extraction_count,
                metrics.dedup_count,
                metrics.memory_count,
                metrics.llm_calls,
                metrics.duration_seconds,
                json.dumps(metrics.errors),
            )
        except Exception:
            pass  # Don't fail ingestion due to metrics storage

    async def check_archived_for_query(self, query: str, threshold: float = 0.75, limit: int = 5) -> list[dict[str, Any]]:
        """Check if archived content matches a query."""
        if self.client is None:
            await self.connect()
        try:
            rows = await self._fetchval(
                """
                SELECT jsonb_agg(jsonb_build_object(
                    'memory_id', memory_id,
                    'content_hash', content_hash,
                    'title', title,
                    'similarity', similarity,
                    'source_path', source_path
                ))
                FROM check_archived_for_query($1, $2, $3)
                """,
                query,
                threshold,
                limit,
            )
            if not rows:
                return []
            result = json.loads(rows) if isinstance(rows, str) else rows
            return result if result else []
        except Exception:
            return []

    async def mark_archived_processed(self, memory_id: str) -> bool:
        """Mark an archived memory as processed."""
        if self.client is None:
            await self.connect()
        try:
            result = await self._fetchval(
                "SELECT mark_archived_as_processed($1::uuid)",
                memory_id,
            )
            return bool(result)
        except Exception:
            return False
