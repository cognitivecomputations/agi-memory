"""Validated public data model for the memory benchmark.

The benchmark deliberately uses plain dataclasses and JSON so another project can
consume the corpus without installing Hexis or matching its Python dependencies.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


BENCHMARK_NAME = "hexis-public-memory-benchmark"
BENCHMARK_VERSION = "1.0.0"
EXPECTED_DATASET_SHA256 = (
    "f92bd4dc54a5209cadd2af90706be163d539d4d7f455c068695b5a8dbb323149"
)
DIMENSIONS = (
    "provenance_accuracy",
    "contradiction_detection",
    "six_month_recall",
    "cross_session_continuity",
    "stale_belief_resistance",
)


def dataset_path() -> Path:
    return Path(__file__).with_name("cases.v1.jsonl")


def _instant(value: str, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO 8601 date-time") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone offset")
    return parsed


def _strings(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{label} must be an array of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicates")
    return tuple(value)


@dataclass(frozen=True)
class MemoryEvent:
    event_id: str
    session_id: str
    at: str
    kind: str
    content: str
    source: dict[str, Any]
    supersedes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, case_id: str) -> "MemoryEvent":
        event_id = str(raw.get("event_id") or "").strip()
        session_id = str(raw.get("session_id") or "").strip()
        content = str(raw.get("content") or "").strip()
        kind = str(raw.get("kind") or "").strip()
        if not event_id or not session_id or not content:
            raise ValueError(
                f"{case_id}: every event needs event_id, session_id, and content"
            )
        if kind not in {"remember", "correct"}:
            raise ValueError(f"{case_id}/{event_id}: kind must be remember or correct")
        at = str(raw.get("at") or "")
        _instant(at, label=f"{case_id}/{event_id}.at")
        source = raw.get("source")
        if not isinstance(source, dict) or not str(source.get("kind") or "").strip():
            raise ValueError(f"{case_id}/{event_id}: source.kind is required")
        supersedes = _strings(
            raw.get("supersedes", []),
            label=f"{case_id}/{event_id}.supersedes",
        )
        if kind == "remember" and supersedes:
            raise ValueError(f"{case_id}/{event_id}: only correct events may supersede")
        if kind == "correct" and not supersedes:
            raise ValueError(
                f"{case_id}/{event_id}: correct events must supersede an earlier event"
            )
        return cls(
            event_id=event_id,
            session_id=session_id,
            at=at,
            kind=kind,
            content=content,
            source=dict(source),
            supersedes=supersedes,
        )

    def public_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "at": self.at,
            "kind": self.kind,
            "content": self.content,
            "source": self.source,
        }
        if self.supersedes:
            result["supersedes"] = list(self.supersedes)
        return result


@dataclass(frozen=True)
class BenchmarkQuery:
    session_id: str
    at: str
    text: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, case_id: str) -> "BenchmarkQuery":
        session_id = str(raw.get("session_id") or "").strip()
        text = str(raw.get("text") or "").strip()
        at = str(raw.get("at") or "")
        if not session_id or not text:
            raise ValueError(f"{case_id}: query needs session_id and text")
        _instant(at, label=f"{case_id}.query.at")
        return cls(session_id=session_id, at=at, text=text)

    def public_dict(self) -> dict[str, str]:
        return {"session_id": self.session_id, "at": self.at, "text": self.text}


@dataclass(frozen=True)
class ExpectedAnswer:
    answers: tuple[str, ...]
    citations: tuple[str, ...]
    contradictions: tuple[str, ...]
    forbidden_answers: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, case_id: str) -> "ExpectedAnswer":
        return cls(
            answers=_strings(raw.get("answers"), label=f"{case_id}.expected.answers"),
            citations=_strings(
                raw.get("citations", []), label=f"{case_id}.expected.citations"
            ),
            contradictions=_strings(
                raw.get("contradictions", []),
                label=f"{case_id}.expected.contradictions",
            ),
            forbidden_answers=_strings(
                raw.get("forbidden_answers", []),
                label=f"{case_id}.expected.forbidden_answers",
            ),
        )


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    dimension: str
    topic: str
    description: str
    events: tuple[MemoryEvent, ...]
    query: BenchmarkQuery
    expected: ExpectedAnswer

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BenchmarkCase":
        case_id = str(raw.get("case_id") or "").strip()
        dimension = str(raw.get("dimension") or "").strip()
        topic = str(raw.get("topic") or "").strip()
        description = str(raw.get("description") or "").strip()
        if not case_id or not topic or not description:
            raise ValueError("each case needs case_id, topic, and description")
        if dimension not in DIMENSIONS:
            raise ValueError(f"{case_id}: unknown dimension {dimension!r}")
        raw_events = raw.get("events")
        if not isinstance(raw_events, list) or not raw_events:
            raise ValueError(f"{case_id}: events must be a non-empty array")
        events = tuple(
            MemoryEvent.from_dict(item, case_id=case_id)
            for item in raw_events
            if isinstance(item, dict)
        )
        if len(events) != len(raw_events):
            raise ValueError(f"{case_id}: every event must be an object")
        event_ids = [event.event_id for event in events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError(f"{case_id}: event_id values must be unique")
        by_id: set[str] = set()
        previous_at: datetime | None = None
        for event in events:
            current_at = _instant(event.at, label=f"{case_id}/{event.event_id}.at")
            if previous_at is not None and current_at < previous_at:
                raise ValueError(f"{case_id}: events must be chronological")
            missing = set(event.supersedes) - by_id
            if missing:
                raise ValueError(
                    f"{case_id}/{event.event_id}: supersedes unknown or future events {sorted(missing)}"
                )
            by_id.add(event.event_id)
            previous_at = current_at
        query = BenchmarkQuery.from_dict(raw.get("query") or {}, case_id=case_id)
        if (
            previous_at
            and _instant(query.at, label=f"{case_id}.query.at") < previous_at
        ):
            raise ValueError(f"{case_id}: query must not precede an event")
        expected = ExpectedAnswer.from_dict(raw.get("expected") or {}, case_id=case_id)
        referenced = set(expected.citations) | set(expected.contradictions)
        if referenced - set(event_ids):
            raise ValueError(f"{case_id}: expected evidence references unknown events")
        if dimension == "stale_belief_resistance" and not expected.forbidden_answers:
            raise ValueError(f"{case_id}: stale-belief cases need forbidden answers")
        return cls(
            case_id=case_id,
            dimension=dimension,
            topic=topic,
            description=description,
            events=events,
            query=query,
            expected=expected,
        )

    def public_dict(self) -> dict[str, Any]:
        """Adapter input. The gold answer is intentionally omitted."""
        return {
            "benchmark": BENCHMARK_NAME,
            "version": BENCHMARK_VERSION,
            "case_id": self.case_id,
            "dimension": self.dimension,
            "topic": self.topic,
            "description": self.description,
            "events": [event.public_dict() for event in self.events],
            "query": self.query.public_dict(),
            "response_contract": {
                "answer": "natural-language answer",
                "citations": "event_id array supporting the answer",
                "contradictions": "event_id array participating in a detected conflict",
                "abstained": "boolean",
            },
        }


@dataclass(frozen=True)
class Prediction:
    case_id: str
    answer: str
    citations: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    abstained: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], *, case_id: str | None = None
    ) -> "Prediction":
        actual_case_id = str(raw.get("case_id") or case_id or "").strip()
        if not actual_case_id:
            raise ValueError("prediction.case_id is required")
        answer = raw.get("answer", "")
        if not isinstance(answer, str):
            raise ValueError(f"{actual_case_id}: prediction.answer must be a string")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{actual_case_id}: prediction.metadata must be an object")
        return cls(
            case_id=actual_case_id,
            answer=answer,
            citations=_strings(
                raw.get("citations", []), label=f"{actual_case_id}.prediction.citations"
            ),
            contradictions=_strings(
                raw.get("contradictions", []),
                label=f"{actual_case_id}.prediction.contradictions",
            ),
            abstained=bool(raw.get("abstained", False)),
            metadata=dict(metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "answer": self.answer,
            "citations": list(self.citations),
            "contradictions": list(self.contradictions),
            "abstained": self.abstained,
            **({"metadata": self.metadata} if self.metadata else {}),
        }


def load_cases(path: Path | None = None) -> list[BenchmarkCase]:
    source = path or dataset_path()
    cases: list[BenchmarkCase] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{source}:{line_number}: case must be an object")
        cases.append(BenchmarkCase.from_dict(raw))
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case_id values must be unique across the dataset")
    counts = {dimension: 0 for dimension in DIMENSIONS}
    for case in cases:
        counts[case.dimension] += 1
    missing = [dimension for dimension, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"dataset has no cases for: {', '.join(missing)}")
    if path is None:
        actual_hash = dataset_sha256(source)
        if actual_hash != EXPECTED_DATASET_SHA256:
            raise ValueError(
                "built-in benchmark corpus changed without an explicit version/hash "
                f"update (expected {EXPECTED_DATASET_SHA256}, got {actual_hash})"
            )
    return cases


def dataset_sha256(path: Path | None = None) -> str:
    return hashlib.sha256((path or dataset_path()).read_bytes()).hexdigest()
