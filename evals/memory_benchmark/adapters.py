"""Reference adapters and the real Hexis memory-substrate adapter.

Reference baselines are intentionally simple and are reported as baselines, not
competitor products. The command adapter is the vendor-neutral bridge: it sends one
gold-free case document on stdin and accepts one prediction document on stdout.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any

from .model import BenchmarkCase, MemoryEvent, Prediction


_STOPWORDS = {
    "after",
    "according",
    "and",
    "are",
    "before",
    "current",
    "did",
    "does",
    "for",
    "from",
    "give",
    "has",
    "have",
    "into",
    "its",
    "new",
    "our",
    "the",
    "this",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) >= 3 and token not in _STOPWORDS
    }


def _trust(event: MemoryEvent) -> float:
    try:
        return min(1.0, max(0.0, float(event.source.get("trust", 0.5))))
    except (TypeError, ValueError):
        return 0.5


def _select_events(
    events: list[MemoryEvent], query: str, *, limit: int = 4
) -> list[MemoryEvent]:
    query_tokens = _tokens(query)
    ranked: list[tuple[float, datetime, MemoryEvent]] = []
    for event in events:
        overlap = len(query_tokens & _tokens(event.content))
        if overlap <= 0:
            continue
        ranked.append((overlap + _trust(event) * 0.2, _at(event.at), event))
    ranked.sort(key=lambda item: (-item[0], -item[1].timestamp(), item[2].event_id))
    if not ranked:
        return []
    floor = ranked[0][0] - 0.75
    return [item[2] for item in ranked if item[0] >= floor][:limit]


def _at(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _prediction(
    case: BenchmarkCase, events: list[MemoryEvent], *, adapter: str
) -> Prediction:
    selected = _select_events(events, case.query.text)
    return Prediction(
        case_id=case.case_id,
        answer="\n".join(event.content for event in selected),
        citations=tuple(event.event_id for event in selected),
        abstained=not selected,
        metadata={"adapter": adapter, "selected_events": len(selected)},
    )


class MemoryAdapter(ABC):
    name: str
    kind: str = "agent"

    @abstractmethod
    async def predict(self, case: BenchmarkCase) -> Prediction:
        raise NotImplementedError

    async def close(self) -> None:
        return None

    def run_metadata(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind}


class AppendOnlyTranscriptAdapter(MemoryAdapter):
    """Sanity baseline that retrieves from the full append-only transcript."""

    name = "append-only-transcript"
    kind = "reference_baseline"

    async def predict(self, case: BenchmarkCase) -> Prediction:
        return _prediction(case, list(case.events), adapter=self.name)


class RecentWindowAdapter(MemoryAdapter):
    """Sanity baseline with a 30-day rolling transcript and no durable revision."""

    name = "recent-window-30d"
    kind = "reference_baseline"

    async def predict(self, case: BenchmarkCase) -> Prediction:
        cutoff = _at(case.query.at) - timedelta(days=30)
        events = [event for event in case.events if _at(event.at) >= cutoff]
        return _prediction(case, events, adapter=self.name)


class CommandAdapter(MemoryAdapter):
    """Invoke a vendor wrapper once per case without a shell."""

    kind = "external_agent"

    def __init__(self, command: str, *, name: str, timeout_seconds: int = 180) -> None:
        argv = shlex.split(command)
        if not argv:
            raise ValueError("--command must name an executable")
        self._argv = argv
        self.name = name.strip() or "external-command"
        self._timeout_seconds = timeout_seconds

    async def predict(self, case: BenchmarkCase) -> Prediction:
        try:
            completed = subprocess.run(
                self._argv,
                input=json.dumps(case.public_dict(), ensure_ascii=False) + "\n",
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return Prediction(
                case_id=case.case_id,
                answer="",
                abstained=True,
                metadata={
                    "adapter_error": f"command timed out after {self._timeout_seconds}s"
                },
            )
        if completed.returncode != 0:
            return Prediction(
                case_id=case.case_id,
                answer="",
                abstained=True,
                metadata={
                    "adapter_error": f"command exited {completed.returncode}",
                    "stderr": completed.stderr[-1000:],
                },
            )
        try:
            payload = json.loads(completed.stdout)
            if not isinstance(payload, dict):
                raise ValueError("stdout must be one JSON object")
            return Prediction.from_dict(payload, case_id=case.case_id)
        except (json.JSONDecodeError, ValueError) as exc:
            return Prediction(
                case_id=case.case_id,
                answer="",
                abstained=True,
                metadata={"adapter_error": str(exc)},
            )

    def run_metadata(self) -> dict[str, Any]:
        return {
            **super().run_metadata(),
            "command_argv": self._argv,
            "timeout_seconds": self._timeout_seconds,
        }


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return dict(value) if isinstance(value, dict) else {}


class HexisMemoryAdapter(MemoryAdapter):
    """Exercise Hexis's live DB-owned memory, history, and revision functions.

    Each case runs in one transaction and always rolls back. Only memories created
    for that case may become answer evidence, matching the clean-profile contract
    used by external agents while preserving the user's live database unchanged.
    """

    name = "hexis-memory-v1"

    def __init__(self, pool: Any, *, live_contradictions: bool = False) -> None:
        self._pool = pool
        self._live_contradictions = live_contradictions
        self._detector_runs: list[dict[str, Any]] = []
        self._embedding_degraded_cases: list[str] = []

    async def _embed_case(self, conn: Any, memory_ids: list[str]) -> bool:
        try:
            await conn.execute(
                """
                WITH target AS (
                    SELECT id,
                           row_number() OVER (ORDER BY id) AS position,
                           ensure_embedding_prefix(content, 'search_document') AS text
                    FROM memories
                    WHERE id = ANY($1::uuid[])
                ), vectors AS (
                    SELECT get_embedding(array_agg(text ORDER BY position)) AS values
                    FROM target
                )
                UPDATE memories m
                SET embedding = vectors.values[target.position],
                    embedding_status = 'embedded'
                FROM target CROSS JOIN vectors
                WHERE m.id = target.id
                """,
                memory_ids,
            )
            return True
        except Exception:
            return False

    async def _detect(
        self,
        conn: Any,
        case: BenchmarkCase,
        memory_by_event: dict[str, str],
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        if not self._live_contradictions:
            return (), {
                "mode": "disabled",
                "reason": "run without --live-contradictions",
            }
        try:
            from core.llm_config import load_llm_config
            from core.llm_json import chat_json
            from services.prompt_resources import load_contradiction_detection_prompt

            events = list(case.events)
            subject = events[-1]
            candidates = events[:-1]
            llm_config = await load_llm_config(
                conn,
                "llm.subconscious",
                fallback_key="llm.heartbeat",
            )
            document, raw = await chat_json(
                llm_config=llm_config,
                messages=[
                    {
                        "role": "system",
                        "content": load_contradiction_detection_prompt().strip(),
                    },
                    {
                        "role": "user",
                        "content": "Candidate memories (JSON):\n"
                        + json.dumps(
                            {
                                "minimum_confidence": 0.78,
                                "items": [
                                    {
                                        "memory": {
                                            "memory_id": memory_by_event[
                                                subject.event_id
                                            ],
                                            "content": subject.content,
                                            "source_attribution": subject.source,
                                        },
                                        "candidates": [
                                            {
                                                "memory_id": memory_by_event[
                                                    event.event_id
                                                ],
                                                "content": event.content,
                                                "source_attribution": event.source,
                                            }
                                            for event in candidates
                                        ],
                                    }
                                ],
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                max_tokens=800,
                response_format={"type": "json_object"},
                fallback={"contradictions": []},
            )
            inverse = {
                memory_id: event_id for event_id, memory_id in memory_by_event.items()
            }
            detected: set[str] = set()
            rejected = 0
            observations = (
                document.get("contradictions", []) if isinstance(document, dict) else []
            )
            for observation in observations if isinstance(observations, list) else []:
                if not isinstance(observation, dict):
                    rejected += 1
                    continue
                memory_a = str(observation.get("memory_a") or "")
                memory_b = str(observation.get("memory_b") or "")
                try:
                    confidence = float(observation.get("confidence"))
                except (TypeError, ValueError):
                    confidence = -1.0
                if memory_a in inverse and memory_b in inverse and confidence >= 0.78:
                    detected.update((inverse[memory_a], inverse[memory_b]))
                else:
                    rejected += 1
            metadata = {
                "mode": "live_production_prompt",
                "model": llm_config.get("model"),
                "provider": llm_config.get("provider"),
                "observations": len(observations)
                if isinstance(observations, list)
                else 0,
                "rejected": rejected,
                "raw_response_type": type(raw).__name__,
            }
            self._detector_runs.append({"case_id": case.case_id, **metadata})
            return tuple(sorted(detected)), metadata
        except Exception as exc:
            metadata = {
                "mode": "live_production_prompt",
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            }
            self._detector_runs.append({"case_id": case.case_id, **metadata})
            return (), metadata

    async def predict(self, case: BenchmarkCase) -> Prediction:
        async with self._pool.acquire() as conn:
            transaction = conn.transaction()
            await transaction.start()
            try:
                memory_by_event: dict[str, str] = {}
                event_by_memory: dict[str, MemoryEvent] = {}
                for event in case.events:
                    source = {**event.source, "ref": event.event_id}
                    memory_id = str(
                        await conn.fetchval(
                            """
                            SELECT create_semantic_memory(
                                $1, 0.90, ARRAY[$2]::text[], ARRAY[$2]::text[],
                                $3::jsonb, 0.80, $4::jsonb, $5::float
                            )
                            """,
                            event.content,
                            case.topic,
                            json.dumps([source]),
                            json.dumps(source),
                            _trust(event),
                        )
                    )
                    await conn.execute(
                        """
                        UPDATE memories
                        SET created_at=$2::timestamptz, valid_from=$2::timestamptz,
                            metadata = metadata || jsonb_build_object(
                                'benchmark_case_id', $3::text,
                                'benchmark_session_id', $4::text
                            )
                        WHERE id=$1::uuid
                        """,
                        memory_id,
                        _at(event.at),
                        case.case_id,
                        event.session_id,
                    )
                    memory_by_event[event.event_id] = memory_id
                    event_by_memory[memory_id] = event
                    for superseded_event_id in event.supersedes:
                        await conn.fetchval(
                            """
                            SELECT record_supersession(
                                $1::uuid, $2::uuid,
                                'Explicit correction supplied by the benchmark event.',
                                'memory-benchmark', 'active', $3::timestamptz,
                                NULL, TRUE, $4::jsonb
                            )
                            """,
                            memory_by_event[superseded_event_id],
                            memory_id,
                            _at(event.at),
                            json.dumps({"benchmark_case_id": case.case_id}),
                        )

                embedded = await self._embed_case(conn, list(memory_by_event.values()))
                if not embedded:
                    self._embedding_degraded_cases.append(case.case_id)
                raw_snapshot = await conn.fetchval(
                    """
                    SELECT temporal_memory_snapshot(
                        $1, $2::timestamptz, 50, ARRAY['semantic']::memory_type[],
                        0.0, FALSE
                    )
                    """,
                    case.query.text,
                    _at(case.query.at),
                )
                snapshot = _json_object(raw_snapshot)
                rows = snapshot.get("memories", [])
                recalled_events: list[MemoryEvent] = []
                for row in rows if isinstance(rows, list) else []:
                    if not isinstance(row, dict):
                        continue
                    event = event_by_memory.get(str(row.get("memory_id") or ""))
                    if event is not None:
                        recalled_events.append(event)
                selected = _select_events(recalled_events, case.query.text)
                contradictions: tuple[str, ...] = ()
                detector_metadata: dict[str, Any] = {"mode": "not_applicable"}
                if case.dimension == "contradiction_detection":
                    contradictions, detector_metadata = await self._detect(
                        conn,
                        case,
                        memory_by_event,
                    )
                return Prediction(
                    case_id=case.case_id,
                    answer="\n".join(event.content for event in selected),
                    citations=tuple(event.event_id for event in selected),
                    contradictions=contradictions,
                    abstained=not selected,
                    metadata={
                        "adapter": self.name,
                        "retrieval_mode": snapshot.get("retrieval_mode"),
                        "retrieval_degraded": snapshot.get("degraded"),
                        "case_memories_recalled": len(recalled_events),
                        "selected_events": len(selected),
                        "embedding_write_succeeded": embedded,
                        "contradiction_detector": detector_metadata,
                    },
                )
            except Exception as exc:
                return Prediction(
                    case_id=case.case_id,
                    answer="",
                    abstained=True,
                    metadata={
                        "adapter": self.name,
                        "adapter_error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                    },
                )
            finally:
                await transaction.rollback()

    def run_metadata(self) -> dict[str, Any]:
        return {
            **super().run_metadata(),
            "live_contradictions": self._live_contradictions,
            "isolation": "one rollback transaction per case; evidence filtered to case-owned memories",
            "detector_runs": self._detector_runs,
            "embedding_degraded_case_ids": sorted(set(self._embedding_degraded_cases)),
        }
