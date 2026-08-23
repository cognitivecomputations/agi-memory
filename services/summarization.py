"""Memory consolidation summarization worker.

Drains `memory_summarization_queue`: for each consolidated GIST, the LLM compacts
its full content into a concise recollection AND distills its durable lessons
upward into the schema. Mirrors `services/recmem.py` (claim → chat_json →
apply_*/fail_*). Gated by config `retention.enabled`.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from core.llm_config import load_llm_config
from core.llm_json import chat_json
from services.prompt_resources import load_memory_summarization_prompt

logger = logging.getLogger("summarization")


async def run_memory_summarization_step(conn) -> dict[str, Any]:
    """Claim and summarize one batch of consolidated gists."""
    batch = await conn.fetchval("SELECT COALESCE(get_config_int('retention.summarize_batch_size'), 8)")
    rows = await conn.fetch("SELECT * FROM claim_memory_summarization_batch($1::int)", int(batch or 8))
    if not rows:
        return {"skipped": True, "reason": "no_pending_summaries"}

    llm_config = await load_llm_config(conn, "llm.summarization", fallback_key="llm.subconscious")
    # Read once per batch: this loader is not cached, so it was re-reading the
    # prompt file on every row.
    system_prompt = load_memory_summarization_prompt().strip()
    done = 0
    failed = 0
    # Deliberately one call per memory rather than a batch (cf. core.llm_batch,
    # PLAN.md §13.3·B2). Summarization is generation, not classification: each
    # input runs to 24k characters and each output is a substantial recollection,
    # so a chunk budget would yield one item per call anyway — and asking for
    # eight long summaries in one response trades quality for a saving that is
    # not there. The batching win was in connector_cognition, where eighty short
    # items were being classified one at a time.
    for row in rows:
        memory_id = row["memory_id"]
        content = row["content"] or ""
        try:
            doc, _raw = await chat_json(
                llm_config=llm_config,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content[:24000]},
                ],
                max_tokens=1800,
                temperature=0.2,
                response_format={"type": "json_object"},
                fallback={"summary": "", "lessons": []},
            )
            summary = (doc or {}).get("summary") if isinstance(doc, dict) else None
            lessons = (doc or {}).get("lessons") if isinstance(doc, dict) else []
            if not summary or not str(summary).strip():
                # Never wipe a memory to empty; retry/backoff and keep the full content.
                await conn.fetchval("SELECT fail_memory_summarization($1::uuid, $2::text)", str(memory_id), "empty summary")
                failed += 1
                continue
            await conn.fetchval(
                "SELECT apply_memory_summary($1::uuid, $2::text, $3::jsonb)",
                str(memory_id),
                str(summary),
                json.dumps(lessons if isinstance(lessons, list) else []),
            )
            done += 1
        except Exception as exc:
            logger.error("summarization failed: memory=%s error=%s", memory_id, exc)
            try:
                await conn.fetchval("SELECT fail_memory_summarization($1::uuid, $2::text)", str(memory_id), str(exc))
            except Exception:
                pass
            failed += 1
    return {"summarized": done, "failed": failed}
