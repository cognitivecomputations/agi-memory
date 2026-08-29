"""HMX (Hexis Memory Exchange) v1.7 — intent policy and envelope construction.

HMX is one wire format used by four operations with different safety rules
(plans/hmx.md): ``port`` and ``duplicate`` may carry deep self-structure
because source and target are the same agent or an intended clone; ``telepathy``
and ``analysis`` must not silently graft one instance's identity, worldview,
drives, emotional triggers, narrative, or active goals into another.

This module is the policy layer (which sections an export intent includes by
default, what explicit opt-in unlocks), the envelope every export carries, and
the export pipeline: section data comes from the SQL functions in
db/48_functions_memory_exchange.sql with local UUIDs, and this layer applies
export-scoped refs, content hashes, provenance enrichment, protected-section
digests, statistics, and JSON/JSONL serialization.
"""

from __future__ import annotations

import copy
import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from importlib.resources import files
from typing import Any, Iterable

HMX_VERSION = "1.7"

EXPORT_INTENTS = ("port", "duplicate", "telepathy", "analysis")
SUPPORTED_IMPORT_STRATEGIES = (
    "additive",
    "authoritative",
    "deliberative",
    "analysis_only",
)
REDACTION_POLICIES = ("none", "basic", "strict", "custom")

# Sections that affect self-constitution, motivational state, value structure,
# or long-range narrative identity. Excluded by default for telepathy/analysis;
# replacement in an active target requires the Protected Section Replacement
# Protocol regardless of intent.
PROTECTED_SECTIONS = frozenset(
    {
        "identity",
        "worldview",
        "drives",
        "emotional_triggers",
        "narrative",
        "goals",
    }
)

# Every section HMX can carry, in canonical envelope order.
ALL_SECTIONS = (
    "memories",
    "episodes",
    "relationships",
    "narrative",
    "identity",
    "worldview",
    "goals",
    "drives",
    "emotional_triggers",
    "clusters",
    "in_flight_work",
    "audit_records",
)

# Optional sections gated by explicit flags rather than intent.
FLAG_GATED_SECTIONS = frozenset({"raw_units", "config"})

EXCLUDED_SECRET_PATTERNS = (
    "key",
    "secret",
    "token",
    "password",
    "signature",
    "credential",
    "auth",
    "trust",
    "anchor",
    "certificate",
)

DEFAULT_IMPORT_STRATEGY = {
    "port": "authoritative",
    "duplicate": "authoritative",
    "telepathy": "deliberative",
    "analysis": "analysis_only",
}

HASH_ALGORITHMS = (
    "content_hash_v1",
    "protected_section_digest_v1",
    "audit_record_digest_v1",
)

OPTIONAL_FEATURES = (
    "raw_units",
    "config",
    "jsonl_streaming",
    "protected_replacement_protocol_v1",
    "fast_path_verification",
)


class HmxPolicyError(ValueError):
    """An export/import request violates HMX intent policy."""


class HmxSchemaError(ValueError):
    """An HMX document does not satisfy the canonical wire schema."""


@dataclass(frozen=True)
class HmxImportResult:
    """Structured outcome of one transactional additive import."""

    export_id: str
    intent: str
    strategy: str
    target_state: dict[str, Any]
    inserted: dict[str, int]
    duplicate_refs: tuple[str, ...]
    ref_map: dict[str, str]
    conflicts: tuple[dict[str, Any], ...]
    warnings: tuple[dict[str, Any], ...]
    work_summary: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class HmxAuthoritativeResult:
    """Outcome of an authoritative import request.

    Ordinary sections may commit immediately. Protected operations report
    verified no-ops, durable requests, or a reused terminal decision.
    """

    export_id: str
    intent: str
    strategy: str
    target_state: dict[str, Any]
    inserted: dict[str, int]
    protected_operations: tuple[dict[str, Any], ...]
    ref_map: dict[str, str]
    conflicts: tuple[dict[str, Any], ...]
    warnings: tuple[dict[str, Any], ...]
    work_summary: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class HmxDryRunResult:
    """Database-aware import forecast produced without changing state."""

    export_id: str
    intent: str
    strategy: str
    target_state: dict[str, Any]
    can_import: bool
    counts: dict[str, int]
    duplicate_refs: tuple[str, ...]
    conflicts: tuple[dict[str, Any], ...]
    warnings: tuple[dict[str, Any], ...]
    protected_policy: dict[str, Any]
    privacy: dict[str, Any]
    estimated_embedding_items: int


@dataclass(frozen=True)
class HmxStagingResult:
    export_id: str
    intent: str
    strategy: str
    batch_id: str
    staged: dict[str, int]
    staging_ids: tuple[str, ...]
    conflicts: tuple[dict[str, Any], ...]
    warnings: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class HmxAnalysisResult:
    export_id: str
    intent: str
    strategy: str
    batch_id: str
    loaded: dict[str, int]
    analysis_ids: tuple[str, ...]
    warnings: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class HmxReviewResult:
    staging_id: str
    decision: str
    section: str
    local_ref: str | None = None
    warnings: tuple[dict[str, Any], ...] = ()


@lru_cache(maxsize=1)
def load_hmx_schema() -> dict[str, Any]:
    """Load the packaged canonical schema for the supported HMX version."""

    import json

    schema_file = files("schemas").joinpath(f"hmx-{HMX_VERSION}.schema.json")
    with schema_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_hmx_document(document: dict[str, Any]) -> None:
    """Validate a complete single-document HMX export.

    The first error includes its JSON path so callers get a useful cause and
    next debugging location instead of a raw validator traceback.
    """

    from jsonschema import FormatChecker
    from jsonschema.validators import validator_for

    schema = load_hmx_schema()
    validator_class = validator_for(schema)
    validator_class.check_schema(schema)
    validator = validator_class(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.path))
    if not errors:
        return

    error = errors[0]
    path = "$" + "".join(
        f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.path
    )
    raise HmxSchemaError(
        f"invalid HMX {HMX_VERSION} document at {path}: {error.message}"
    )


def _coerce_json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _document_probe(
    document: dict[str, Any], sections: dict[str, Any]
) -> dict[str, Any]:
    """Build a schema probe that accepts a newer 1.x minor during import."""

    probe = copy.deepcopy(document)
    probe["hmx_version"] = HMX_VERSION
    probe["sections"] = sections
    return probe


def _validate_import_header(document: dict[str, Any]) -> None:
    version = str(document.get("hmx_version", ""))
    try:
        major = int(version.split(".", 1)[0])
    except (TypeError, ValueError):
        raise HmxSchemaError(
            f"invalid HMX version {version!r}; expected major version 1"
        )
    if major != int(HMX_VERSION.split(".", 1)[0]):
        raise HmxSchemaError(
            f"unsupported HMX major version {version!r}; this importer supports 1.x"
        )
    validate_hmx_document(_document_probe(document, {}))


def _validated_records(
    document: dict[str, Any], section: str, records: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate records independently so one malformed record is skippable."""

    valid: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not isinstance(records, list):
        return valid, [
            {
                "code": "schema_validation_error",
                "section": section,
                "error": "section must be an array",
            }
        ]
    for index, record in enumerate(records):
        try:
            validate_hmx_document(_document_probe(document, {section: [record]}))
        except HmxSchemaError as exc:
            warnings.append(
                {
                    "code": "schema_validation_error",
                    "section": section,
                    "index": index,
                    "error": str(exc),
                }
            )
        else:
            valid.append(copy.deepcopy(record))
    return valid, warnings


def _validated_in_flight_work(
    document: dict[str, Any], work: Any
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Validate task records independently while preserving group identity."""

    groups = {
        "consolidation_tasks": [],
        "reconsolidation_tasks": [],
    }
    warnings: list[dict[str, Any]] = []
    if not isinstance(work, dict):
        return groups, [
            {
                "code": "schema_validation_error",
                "section": "in_flight_work",
                "error": "section must be an object",
            }
        ]
    for group in groups:
        records = work.get(group, [])
        if not isinstance(records, list):
            warnings.append(
                {
                    "code": "schema_validation_error",
                    "section": "in_flight_work",
                    "task_group": group,
                    "error": "task group must be an array",
                }
            )
            continue
        for index, record in enumerate(records):
            try:
                validate_hmx_document(
                    _document_probe(
                        document,
                        {"in_flight_work": {group: [record]}},
                    )
                )
            except HmxSchemaError as exc:
                warnings.append(
                    {
                        "code": "schema_validation_error",
                        "section": "in_flight_work",
                        "task_group": group,
                        "index": index,
                        "error": str(exc),
                    }
                )
            else:
                groups[group].append(copy.deepcopy(record))
    return groups, warnings


def validate_intent(intent: str) -> str:
    if intent not in EXPORT_INTENTS:
        raise HmxPolicyError(
            f"export_intent must be one of {', '.join(EXPORT_INTENTS)}; got {intent!r}"
        )
    return intent


def new_export_id() -> str:
    return f"exp_{secrets.token_hex(8)}"


@dataclass(frozen=True)
class SectionPlan:
    """Resolved section policy for one export."""

    intent: str
    sections: tuple[str, ...]
    include_raw_units: bool = False
    include_config: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def protected(self) -> tuple[str, ...]:
        return tuple(s for s in self.sections if s in PROTECTED_SECTIONS)


def resolve_export_sections(
    intent: str,
    *,
    include_protected: list[str] | None = None,
    include_raw_units: bool = False,
    include_config: bool = False,
    include_in_flight_work: bool | None = None,
    include_audit_records: bool | None = None,
    sections: list[str] | None = None,
) -> SectionPlan:
    """Apply intent-specific default section policy (spec: "Export Intents").

    - port/duplicate: all portable sections, including protected structure,
      in-flight work, and audit records.
    - telepathy: memories, episodes, relationships, clusters. Protected
      sections require explicit ``include_protected`` opt-in; in-flight work
      and audit records are excluded (audit opt-in is foreign diagnostic).
    - analysis: caller-selected read-only sections (default: the telepathy
      base set); protected/in-flight/audit only if explicitly requested.
    """
    validate_intent(intent)
    include_protected = list(include_protected or [])
    warnings: list[str] = []

    unknown = sorted(set(include_protected) - PROTECTED_SECTIONS)
    if unknown:
        raise HmxPolicyError(
            "include_protected accepts only protected sections "
            f"({', '.join(sorted(PROTECTED_SECTIONS))}); got: {', '.join(unknown)}"
        )

    if intent in ("port", "duplicate"):
        selected = set(ALL_SECTIONS)
        if include_in_flight_work is False:
            selected.discard("in_flight_work")
        if include_audit_records is False:
            # Audit history is how an instance accounts for past rewrites of
            # its protected state; dropping it from a port is legal but loud.
            selected.discard("audit_records")
            warnings.append(
                "audit_records excluded from a port/duplicate export: the "
                "instance's protected-section history will not travel with it"
            )
    else:
        selected = {"memories", "episodes", "relationships", "clusters"}
        if sections is not None:
            requested = set(sections)
            unknown_sections = sorted(requested - set(ALL_SECTIONS))
            if unknown_sections:
                raise HmxPolicyError(f"unknown sections: {', '.join(unknown_sections)}")
            protected_requested = sorted(
                (requested & PROTECTED_SECTIONS) - set(include_protected)
            )
            if protected_requested:
                raise HmxPolicyError(
                    f"{intent} exports exclude protected sections by default; "
                    f"pass include_protected for: {', '.join(protected_requested)}"
                )
            selected = requested - PROTECTED_SECTIONS
        for section in include_protected:
            selected.add(section)
            warnings.append(
                f"protected section '{section}' included in a {intent} export by "
                "explicit opt-in; importers must route it through deliberative "
                "review or analysis-only handling"
            )
        if include_in_flight_work is True:
            if intent == "analysis":
                selected.add("in_flight_work")
            else:
                warnings.append("in_flight_work is excluded for telepathy exports")
        if include_audit_records is True:
            selected.add("audit_records")
            if intent == "telepathy":
                warnings.append(
                    "audit_records in a telepathy export are foreign diagnostics, "
                    "not local history"
                )

    ordered = tuple(s for s in ALL_SECTIONS if s in selected)
    if include_raw_units:
        ordered += ("raw_units",)
    if include_config:
        ordered += ("config",)
    return SectionPlan(
        intent=intent,
        sections=ordered,
        include_raw_units=include_raw_units,
        include_config=include_config,
        warnings=tuple(warnings),
    )


def default_import_strategy(intent: str) -> str:
    validate_intent(intent)
    return DEFAULT_IMPORT_STRATEGY[intent]


def normalize_replace_sections(sections: Iterable[str] | None) -> tuple[str, ...]:
    normalized = tuple(
        dict.fromkeys(
            str(section).strip().replace("-", "_")
            for section in (sections or ())
            if str(section).strip()
        )
    )
    invalid = sorted(set(normalized) - PROTECTED_SECTIONS)
    if invalid:
        raise HmxPolicyError(
            "--replace accepts protected sections only; invalid: " + ", ".join(invalid)
        )
    return normalized


def build_capabilities(relationship_edge_types: list[str]) -> dict[str, Any]:
    """Capabilities block. Edge types come from the live ``graph_edge_type``
    enum (derive from truth), not a hardcoded list."""
    return {
        "formats": ["json", "jsonl"],
        "sections": list(ALL_SECTIONS),
        "hash_algorithms": list(HASH_ALGORITHMS),
        "relationship_edge_types": sorted(relationship_edge_types),
        "optional_features": list(OPTIONAL_FEATURES),
    }


def build_envelope(
    *,
    intent: str,
    plan: SectionPlan,
    instance_id: str,
    schema_version: str,
    embedding_model: str,
    embedding_dimension: int,
    lineage_id: str,
    relationship_edge_types: list[str],
    redaction_policy: str = "none",
    consent_scope: str = "unspecified",
    types: list[str] | None = None,
    time_range: tuple[str | None, str | None] = (None, None),
    export_filter: str | None = None,
    export_id: str | None = None,
    exported_at: datetime | None = None,
) -> dict[str, Any]:
    """Construct the HMX envelope (spec: "Format Specification"). Sections and
    statistics start empty; the exporter fills them as it serializes."""
    validate_intent(intent)
    if redaction_policy not in REDACTION_POLICIES:
        raise HmxPolicyError(f"unknown redaction_policy: {redaction_policy!r}")

    stamp = (exported_at or datetime.now(timezone.utc)).isoformat().replace("+00:00", "Z")
    return {
        "hmx_version": HMX_VERSION,
        "export_id": export_id or new_export_id(),
        "export_intent": intent,
        "exported_at": stamp,
        "source": {
            "instance_id": instance_id,
            "schema_version": schema_version,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "hexis_lineage_id": lineage_id,
        },
        "capabilities": build_capabilities(relationship_edge_types),
        "privacy": {
            "redaction_policy": redaction_policy,
            "contains_sensitive_content": True,
            "consent_scope": consent_scope,
            "excluded_secret_patterns": list(EXCLUDED_SECRET_PATTERNS),
        },
        "export_scope": {
            "types": types or [],
            "time_range": list(time_range),
            "include_protected": list(plan.protected),
            "include_raw_units": plan.include_raw_units,
            "include_config": plan.include_config,
            "include_in_flight_work": "in_flight_work" in plan.sections,
            "include_audit_records": "audit_records" in plan.sections,
            "filter": export_filter,
        },
        "sections": {},
        "statistics": {
            "total_memories": 0,
            "total_relationships": 0,
            "total_episodes": 0,
            "total_raw_units": 0,
            "total_narrative_nodes": 0,
            "total_in_flight_tasks": 0,
            "total_audit_records": 0,
            "total_conflicts_predicted": 0,
            "estimated_embedding_items": 0,
            "estimated_embedding_tokens": 0,
            "estimated_embedding_cost_units": None,
            "estimated_uncompressed_bytes": 0,
        },
    }


# ---------------------------------------------------------------------------
# Export pipeline
# ---------------------------------------------------------------------------

_SECTION_FN = {
    "memories": "hmx_export_memories($1::text[], $2::timestamptz, $3::timestamptz, $4::boolean)",
    "episodes": "hmx_export_episodes()",
    "relationships": "hmx_export_relationships()",
    "narrative": "hmx_export_narrative()",
    "identity": "hmx_export_identity()",
    "worldview": "hmx_export_worldview()",
    "goals": "hmx_export_goals()",
    "drives": "hmx_export_drives()",
    "emotional_triggers": "hmx_export_emotional_triggers()",
    "clusters": "hmx_export_clusters()",
    "in_flight_work": "hmx_export_in_flight_work()",
    "audit_records": "hmx_export_audit_records()",
    "raw_units": "hmx_export_raw_units($1::boolean)",
    "config": "hmx_export_config()",
}

# record_type values for JSONL streaming, per section.
_JSONL_RECORD_TYPE = {
    "memories": "memory",
    "episodes": "episode",
    "relationships": "relationship",
    "worldview": "worldview",
    "goals": "goal",
    "drives": "drive",
    "emotional_triggers": "emotional_trigger",
    "clusters": "cluster",
    "identity": "identity",
    "raw_units": "raw_unit",
}

_JSONL_SECTION = {
    record_type: section for section, record_type in _JSONL_RECORD_TYPE.items()
}


def _ref(export_id: str, local_id: Any) -> str:
    return f"{export_id}:{local_id}"


def _refs(export_id: str, local_ids: Any) -> list[str]:
    return [_ref(export_id, i) for i in (local_ids or [])]


def _take_refs(record: dict[str, Any], export_id: str, wire_name: str) -> None:
    """Scope a raw AGE/SQL reference list under its HMX wire name."""

    raw_name = wire_name.replace("_refs", "_ids")
    local_ids = record.pop(raw_name, record.pop(wire_name, []))
    record[wire_name] = _refs(export_id, local_ids)


def _enrich_provenance(
    record: dict[str, Any], *, instance_id: str, local_id: Any
) -> None:
    """Move provenance out of metadata to the wire position and fill defaults
    (spec field-default table: acquisition_mode=experienced on export)."""
    metadata = record.get("metadata") or {}
    provenance = metadata.pop("provenance", None) or {}
    provenance.setdefault("acquisition_mode", "experienced")
    provenance.setdefault("origin_instance", instance_id)
    provenance.setdefault("origin_id", str(local_id))
    provenance.setdefault("import_chain", [])
    provenance.setdefault("modification_chain", [])
    record["metadata"] = metadata
    record["provenance"] = provenance


def _postprocess_section(
    section: str,
    data: Any,
    *,
    export_id: str,
    instance_id: str,
) -> Any:
    """Local UUIDs -> export-scoped refs; content hashes; wire field names."""
    from core.digest import content_hash_v1

    if section in ("memories", "worldview"):
        for record in data:
            local_id = record.pop("id")
            record["ref"] = _ref(export_id, local_id)
            record["content_hash_v1"] = content_hash_v1(record.get("content") or "")
            record.pop("superseded_by", None)  # normalized into SUPERSEDES edges
            _enrich_provenance(record, instance_id=instance_id, local_id=local_id)
            if section == "worldview":
                record["supporting_refs"] = _refs(
                    export_id, record.pop("supporting_ids", [])
                )
                record["contesting_refs"] = _refs(
                    export_id, record.pop("contesting_ids", [])
                )
    elif section == "episodes":
        for record in data:
            record["ref"] = _ref(export_id, record.pop("id"))
            record["memory_refs"] = _refs(export_id, record.pop("memory_ids", []))
    elif section == "relationships":
        for record in data:
            props = record.get("properties") or {}
            props["source_type"] = record.pop("source_type", None)
            props["target_type"] = record.pop("target_type", None)
            record["properties"] = props
            record["source_ref"] = _ref(export_id, record.pop("source_id"))
            record["target_ref"] = _ref(export_id, record.pop("target_id"))
    elif section == "narrative":
        for group, title_field in (
            ("life_chapters", "title"),
            ("turning_points", "title"),
            ("narrative_threads", "name"),
            ("value_conflicts", "summary"),
        ):
            for record in data.get(group, []):
                local_id = record.pop("id")
                record["ref"] = _ref(export_id, local_id)
                if not record.get(title_field):
                    record[title_field] = str(
                        record.get("name")
                        or record.get("title")
                        or record.get("description")
                        or record.get("key")
                        or ""
                    )
                for wire_name in (
                    "memory_refs",
                    "chapter_refs",
                    "supporting_refs",
                    "contesting_refs",
                ):
                    if (
                        wire_name in record
                        or wire_name.replace("_refs", "_ids") in record
                    ):
                        _take_refs(record, export_id, wire_name)
    elif section == "identity":
        for record in data:
            local_id = f"identity:{record.get('key', 'self')}"
            _enrich_provenance(record, instance_id=instance_id, local_id=local_id)
            for facet in record.get("facets", []):
                evidence_id = facet.pop("evidence_memory_id", None)
                if evidence_id:
                    facet["evidence_memory_ref"] = _ref(export_id, evidence_id)
    elif section == "drives":
        for record in data:
            _enrich_provenance(
                record,
                instance_id=instance_id,
                local_id=f"drive:{record.get('name', 'unknown')}",
            )
    elif section == "goals":
        for record in data:
            local_id = record.pop("id")
            record["ref"] = _ref(export_id, local_id)
            parent = record.pop("parent_goal_id", None)
            record["parent_ref"] = _ref(export_id, parent) if parent else None
            blocked_by = record.get("blocked_by")
            if isinstance(blocked_by, list):
                record["blocked_by"] = _refs(export_id, blocked_by)
            _enrich_provenance(record, instance_id=instance_id, local_id=local_id)
    elif section == "emotional_triggers":
        for record in data:
            local_id = record.pop("id", None)
            record["content_hash_v1"] = content_hash_v1(
                record.get("trigger_pattern") or ""
            )
            record["source_memory_refs"] = _refs(
                export_id, record.pop("source_memory_ids", [])
            )
            _enrich_provenance(
                record, instance_id=instance_id, local_id=local_id or "trigger"
            )
    elif section == "clusters":
        for record in data:
            record["ref"] = _ref(export_id, record.pop("id"))
            record["member_refs"] = _refs(export_id, record.pop("member_ids", []))
    elif section == "raw_units":
        for record in data:
            record["ref"] = _ref(export_id, record.pop("id"))
            record["derived_memory_refs"] = _refs(
                export_id, record.pop("derived_memory_ids", [])
            )
    elif section == "in_flight_work":
        for task in data.get("consolidation_tasks", []):
            task["ref"] = _ref(export_id, task.pop("id"))
            task["input_refs"] = _refs(export_id, task.pop("input_ids", []))
            trigger = task.pop("trigger_unit_id", None)
            if trigger:
                task["trigger_ref"] = _ref(export_id, trigger)
            target = task.pop("target_memory_id", None)
            task["output_refs"] = [_ref(export_id, target)] if target else []
        for task in data.get("reconsolidation_tasks", []):
            task["ref"] = _ref(export_id, task.pop("id"))
            task["memory_refs"] = _refs(export_id, task.pop("memory_ids", []))
    return data


def _fill_statistics(envelope: dict[str, Any]) -> None:
    import json as _json

    sections = envelope["sections"]
    stats = envelope["statistics"]
    stats["total_memories"] = len(sections.get("memories", []))
    stats["total_relationships"] = len(sections.get("relationships", []))
    stats["total_episodes"] = len(sections.get("episodes", []))
    stats["total_raw_units"] = len(sections.get("raw_units", []))
    narrative = sections.get("narrative") or {}
    stats["total_narrative_nodes"] = sum(len(v) for v in narrative.values())
    in_flight = sections.get("in_flight_work") or {}
    stats["total_in_flight_tasks"] = sum(len(v) for v in in_flight.values())
    audit = sections.get("audit_records") or {}
    stats["total_audit_records"] = sum(len(v) for v in audit.values())

    embeddable: list[str] = []
    embeddable += [m.get("content") or "" for m in sections.get("memories", [])]
    embeddable += [w.get("content") or "" for w in sections.get("worldview", [])]
    embeddable += [g.get("title") or "" for g in sections.get("goals", [])]
    embeddable += [
        t.get("trigger_pattern") or "" for t in sections.get("emotional_triggers", [])
    ]
    embeddable += [
        e.get("summary") or "" for e in sections.get("episodes", []) if e.get("summary")
    ]
    stats["estimated_embedding_items"] = len(embeddable)
    stats["estimated_embedding_tokens"] = sum(len(t) for t in embeddable) // 4
    stats["estimated_uncompressed_bytes"] = len(_json.dumps(envelope, default=str))


async def export_hmx(
    conn,
    *,
    intent: str,
    include_protected: list[str] | None = None,
    include_raw_units: bool = False,
    include_config: bool = False,
    include_in_flight_work: bool | None = None,
    include_audit_records: bool | None = None,
    sections: list[str] | None = None,
    types: list[str] | None = None,
    since: Any = None,
    until: Any = None,
    redaction_policy: str = "none",
    consent_scope: str = "unspecified",
    include_sensitive: bool = False,
) -> dict[str, Any]:
    """Produce a complete HMX envelope from the live database.

    Embeddings never leave the database; refs are export-scoped; protected
    sections ride only where intent policy allows and, for port/duplicate,
    carry ``protected_section_digest_v1`` under ``section_digests`` for the
    replacement protocol's Phase 0 fast path.
    """
    import json as _json

    from core.digest import protected_section_digest_v1

    if redaction_policy == "strict" and include_raw_units:
        raise HmxPolicyError(
            "strict redaction excludes raw units; remove include_raw_units or choose another policy"
        )

    plan = resolve_export_sections(
        intent,
        include_protected=include_protected,
        include_raw_units=include_raw_units,
        include_config=include_config,
        include_in_flight_work=include_in_flight_work,
        include_audit_records=include_audit_records,
        sections=sections,
    )
    ctx = await load_source_context(conn)
    envelope = build_envelope(
        intent=intent,
        plan=plan,
        redaction_policy=redaction_policy,
        consent_scope=consent_scope,
        types=types,
        **ctx,
    )
    export_id = envelope["export_id"]

    # Sensitivity policy (#92), mirroring the protected-section intent split:
    # port/duplicate move the whole brain, so private-marked rows always
    # travel; telepathy/analysis are egress to someone else, so they require
    # an explicit opt-in.
    sensitive_included = include_sensitive or intent in ("port", "duplicate")

    for section in plan.sections:
        if section == "memories":
            raw = await conn.fetchval(
                f"SELECT {_SECTION_FN['memories']}",
                types,
                since,
                until,
                sensitive_included,
            )
        elif section == "raw_units":
            raw = await conn.fetchval(
                f"SELECT {_SECTION_FN['raw_units']}", sensitive_included
            )
        else:
            raw = await conn.fetchval(f"SELECT {_SECTION_FN[section]}")
        data = _json.loads(raw) if isinstance(raw, str) else raw
        envelope["sections"][section] = _postprocess_section(
            section, data, export_id=export_id, instance_id=ctx["instance_id"]
        )

    if intent in ("port", "duplicate"):
        digests = {
            section: protected_section_digest_v1(section, envelope["sections"][section])
            for section in plan.protected
            if section in envelope["sections"]
        }
        envelope["section_digests"] = digests

    export_warnings = list(plan.warnings)
    if include_sensitive and intent not in ("port", "duplicate"):
        export_warnings.append(
            f"private-marked memories included in a {intent} export by explicit "
            "opt-in; the receiver sees content the owner marked private"
        )
    in_flight = envelope["sections"].get("in_flight_work") or {}
    if in_flight.get("consolidation_tasks") and "raw_units" not in envelope["sections"]:
        export_warnings.append(
            "consolidation tasks reference raw units that are excluded from this "
            "exchange and will be dropped on import; re-export with "
            "include_raw_units=true (CLI: --include-raw) to carry their inputs"
        )
    if export_warnings:
        envelope["export_warnings"] = export_warnings

    _fill_statistics(envelope)
    validate_hmx_document(envelope)
    return envelope


async def verify_mind_continuity(conn, document: dict[str, Any]) -> dict[str, Any]:
    """Compare imported constitutional state with the source mind envelope.

    HMX refs are deliberately export-scoped, so identity continuity is proven
    with the lineage label and constitutional-state projections rather than
    local UUID equality.  Exact canonical transport digests remain available
    for audit, while the projections exclude only destination-local encoding
    metadata.  Ordinary-record import counts remain in the import result; this
    check answers the stronger "did the same self wake up?" claim.
    """

    if str(document.get("export_intent") or "") != "port":
        raise HmxPolicyError("mind continuity verification requires export_intent 'port'")
    source_canonical_digests = document.get("section_digests")
    if not isinstance(source_canonical_digests, dict):
        raise HmxSchemaError(
            "mind file is missing canonical protected-section digests"
        )

    from core.digest import protected_section_digest_v1

    def continuity_projection(section: str, value: Any) -> Any:
        projected = copy.deepcopy(value)
        records: list[dict[str, Any]] = []
        if section == "narrative" and isinstance(projected, dict):
            records = [
                record
                for group in projected.values()
                if isinstance(group, list)
                for record in group
                if isinstance(record, dict)
            ]
        elif isinstance(projected, list):
            records = [record for record in projected if isinstance(record, dict)]
        elif isinstance(projected, dict):
            records = [projected]

        duplicated_metadata = {
            "worldview": {"category", "confidence", "stability"},
            "goals": {
                "title",
                "description",
                "priority",
                "source",
                "due_at",
                "progress",
                "blocked_by",
                "parent_ref",
                "parent_goal_id",
            },
        }
        for record in records:
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                continue
            # These are regenerated by the destination's local encoding
            # pipeline.  They are neither transported self-constitution nor
            # evidence that the imported mind changed.
            metadata.pop("dopamine_at_encoding", None)
            metadata.pop("emotional_context", None)
            for key in duplicated_metadata.get(section, set()):
                metadata.pop(key, None)
        return projected

    current = await export_hmx(conn, intent="port")
    target_canonical_digests = current.get("section_digests") or {}
    section_checks: dict[str, dict[str, Any]] = {}
    for section in sorted(PROTECTED_SECTIONS):
        source_continuity_digest = protected_section_digest_v1(
            section,
            continuity_projection(section, (document.get("sections") or {}).get(section)),
        )
        target_continuity_digest = protected_section_digest_v1(
            section,
            continuity_projection(section, (current.get("sections") or {}).get(section)),
        )
        section_checks[section] = {
            "matches": source_continuity_digest == target_continuity_digest,
            "source_continuity_digest_v1": source_continuity_digest,
            "target_continuity_digest_v1": target_continuity_digest,
            "canonical_transport_digest_matches": (
                bool(source_canonical_digests.get(section))
                and source_canonical_digests.get(section)
                == target_canonical_digests.get(section)
            ),
        }
    source_lineage = str(
        (document.get("source") or {}).get("hexis_lineage_id") or ""
    )
    target_lineage = str((current.get("source") or {}).get("hexis_lineage_id") or "")
    lineage_matches = bool(source_lineage) and source_lineage == target_lineage
    verified = lineage_matches and all(
        check["matches"] for check in section_checks.values()
    )
    return {
        "verified": verified,
        "source_export_id": document.get("export_id"),
        "hmx_version": document.get("hmx_version"),
        "lineage": {
            "matches": lineage_matches,
            "source": source_lineage,
            "target": target_lineage,
        },
        "protected_sections": section_checks,
        "next_step": (
            None
            if verified
            else "Keep the source mind file, inspect the mismatched constitutional section, and rerun the import dry-run before starting this target."
        ),
    }


def iter_hmx_jsonl(envelope: dict[str, Any]):
    """Yield JSONL lines for an envelope (spec: "Streaming Format"): header,
    typed records, footer with statistics."""
    import json as _json

    header = {k: v for k, v in envelope.items() if k not in ("sections", "statistics")}
    yield _json.dumps({"record_type": "envelope", "data": header}, default=str)
    for section, data in envelope["sections"].items():
        if not _section_has_records(section, data):
            yield _json.dumps(
                {"record_type": "section", "section": section, "data": data},
                default=str,
            )
            continue
        record_type = _JSONL_RECORD_TYPE.get(section)
        if record_type:
            for record in data:
                yield _json.dumps(
                    {"record_type": record_type, "data": record}, default=str
                )
        elif section == "narrative":
            yield _json.dumps({"record_type": "narrative", "data": data}, default=str)
        elif section == "in_flight_work":
            for group in ("consolidation_tasks", "reconsolidation_tasks"):
                for task in data.get(group, []):
                    yield _json.dumps(
                        {
                            "record_type": "in_flight_task",
                            "data": dict(task, task_group=group),
                        },
                        default=str,
                    )
        elif section == "audit_records":
            yield _json.dumps(
                {"record_type": "audit_record", "data": data}, default=str
            )
        elif section == "config":
            yield _json.dumps({"record_type": "config", "data": data}, default=str)
        else:
            # Preserve forward-compatible sections instead of dropping them
            # when an HMX document is transported as JSONL.
            yield _json.dumps(
                {"record_type": "section", "section": section, "data": data},
                default=str,
            )
    yield _json.dumps(
        {"record_type": "footer", "statistics": envelope["statistics"]}, default=str
    )


def parse_hmx_jsonl(lines: Iterable[str]) -> dict[str, Any]:
    """Reconstruct a single HMX document from typed JSONL records.

    Errors include the source line so malformed exchange files are actionable.
    Unknown record types fail rather than disappearing silently; newer sections
    can travel through the generic ``section`` record type.
    """

    envelope: dict[str, Any] | None = None
    sections: dict[str, Any] = {}
    statistics: dict[str, Any] | None = None
    saw_footer = False

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if saw_footer:
            raise HmxSchemaError(
                f"invalid HMX JSONL at line {line_number}: records follow the footer"
            )
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HmxSchemaError(
                f"invalid HMX JSONL at line {line_number}, column {exc.colno}: {exc.msg}"
            ) from exc
        if not isinstance(item, dict):
            raise HmxSchemaError(
                f"invalid HMX JSONL at line {line_number}: record must be an object"
            )

        record_type = item.get("record_type")
        if envelope is None and record_type != "envelope":
            raise HmxSchemaError(
                f"invalid HMX JSONL at line {line_number}: first record must be envelope"
            )
        if record_type == "envelope":
            if envelope is not None:
                raise HmxSchemaError(
                    f"invalid HMX JSONL at line {line_number}: duplicate envelope"
                )
            data = item.get("data")
            if not isinstance(data, dict):
                raise HmxSchemaError(
                    f"invalid HMX JSONL at line {line_number}: envelope data must be an object"
                )
            envelope = copy.deepcopy(data)
            continue
        if record_type == "footer":
            stats = item.get("statistics", {})
            if not isinstance(stats, dict):
                raise HmxSchemaError(
                    f"invalid HMX JSONL at line {line_number}: footer statistics must be an object"
                )
            statistics = copy.deepcopy(stats)
            saw_footer = True
            continue

        data = copy.deepcopy(item.get("data"))
        section = _JSONL_SECTION.get(str(record_type))
        if section is not None:
            sections.setdefault(section, []).append(data)
        elif record_type == "narrative":
            if "narrative" in sections:
                raise HmxSchemaError(
                    f"invalid HMX JSONL at line {line_number}: duplicate narrative record"
                )
            sections["narrative"] = data
        elif record_type == "in_flight_task":
            if not isinstance(data, dict):
                raise HmxSchemaError(
                    f"invalid HMX JSONL at line {line_number}: in-flight task data must be an object"
                )
            group = data.pop("task_group", None)
            if group not in ("consolidation_tasks", "reconsolidation_tasks"):
                raise HmxSchemaError(
                    f"invalid HMX JSONL at line {line_number}: invalid task_group {group!r}"
                )
            sections.setdefault("in_flight_work", {}).setdefault(group, []).append(data)
        elif record_type == "audit_record":
            if "audit_records" in sections:
                raise HmxSchemaError(
                    f"invalid HMX JSONL at line {line_number}: duplicate audit_record"
                )
            sections["audit_records"] = data
        elif record_type == "config":
            if "config" in sections:
                raise HmxSchemaError(
                    f"invalid HMX JSONL at line {line_number}: duplicate config record"
                )
            sections["config"] = data
        elif record_type == "section":
            section_name = item.get("section")
            if not isinstance(section_name, str) or not section_name:
                raise HmxSchemaError(
                    f"invalid HMX JSONL at line {line_number}: generic section name is required"
                )
            if section_name in sections:
                raise HmxSchemaError(
                    f"invalid HMX JSONL at line {line_number}: duplicate section {section_name!r}"
                )
            sections[section_name] = data
        else:
            raise HmxSchemaError(
                f"invalid HMX JSONL at line {line_number}: unknown record_type {record_type!r}"
            )

    if envelope is None:
        raise HmxSchemaError("invalid HMX JSONL: file contains no envelope record")
    envelope["sections"] = sections
    envelope["statistics"] = statistics or envelope.get("statistics") or {}
    return envelope


async def load_source_context(conn) -> dict[str, Any]:
    """Read the envelope's source-block facts from the live database."""
    import json as _json

    async def _config(key: str) -> Any:
        raw = await conn.fetchval("SELECT value FROM config WHERE key = $1", key)
        if raw is None:
            return None
        return _json.loads(raw) if isinstance(raw, str) else raw

    edge_types = [
        r["enumlabel"]
        for r in await conn.fetch(
            "SELECT enumlabel FROM pg_enum e JOIN pg_type t ON t.oid = e.enumtypid "
            "WHERE t.typname = 'graph_edge_type' ORDER BY enumlabel"
        )
    ]
    from core.migrations import migrations_table_name

    migrations_table = await migrations_table_name(conn)
    schema_version = await conn.fetchval(
        f"SELECT COALESCE(max(version), 'baseline') FROM {migrations_table}"
    )
    return {
        "instance_id": str(
            await _config("agent.instance_id") or await _config("agent.name") or "hexis"
        ),
        "schema_version": str(schema_version),
        "embedding_model": str(await _config("embedding.model_id") or "unknown"),
        "embedding_dimension": int(await _config("embedding.dimension") or 768),
        "lineage_id": str(await _config("agent.lineage_id") or ""),
        "relationship_edge_types": edge_types,
    }


# ---------------------------------------------------------------------------
# Additive import (Slice 2)
# ---------------------------------------------------------------------------

_KNOWN_MODIFICATION_KINDS = frozenset(
    {
        "trivial_edit",
        "formatting",
        "clarification",
        "reflection_revision",
        "contradiction_resolution",
        "temporal_update",
        "correction",
        "supersession",
        "integration",
    }
)
_NON_MATERIAL_MODIFICATION_KINDS = frozenset(
    {"trivial_edit", "formatting", "clarification"}
)


def _modification_is_material(kind: str | None) -> bool:
    """Treat unknown modification kinds conservatively, per the HMX contract."""

    return bool(kind) and kind not in _NON_MATERIAL_MODIFICATION_KINDS


def _section_has_records(section: str, value: Any) -> bool:
    if isinstance(value, dict):
        return any(bool(records) for records in value.values())
    return bool(value)


def _unknown_section_warnings(
    sections: dict[str, Any], *, isolated: bool
) -> list[dict[str, Any]]:
    known = set(ALL_SECTIONS) | set(FLAG_GATED_SECTIONS)
    return [
        {
            "code": "unsupported_section",
            "section": section,
            "error": (
                "unknown future section was retained in isolated review storage"
                if isolated
                else "unknown future section was not applied"
            ),
        }
        for section in sorted(set(sections) - known)
        if _section_has_records(section, sections[section])
    ]


def _import_sections(data: dict[str, Any]) -> dict[str, Any]:
    sections = data.get("sections", {})
    if not isinstance(sections, dict):
        raise HmxSchemaError("invalid HMX document at $.sections: expected an object")
    return sections


def _append_import_provenance(
    record: dict[str, Any],
    *,
    origin_id: str,
    intent: str,
    source: dict[str, Any],
    export_id: str,
    local_instance_id: str,
    imported_at: str,
    acquisition_mode: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prepared = copy.deepcopy(record)
    provenance = copy.deepcopy(prepared.get("provenance") or {})
    provenance.setdefault(
        "origin_instance", str(source.get("instance_id") or "unknown")
    )
    provenance.setdefault("origin_id", origin_id)
    provenance.setdefault("modification_chain", [])
    if acquisition_mode is not None:
        provenance["acquisition_mode"] = acquisition_mode
    elif intent not in ("port", "duplicate"):
        if provenance.get("acquisition_mode") not in {
            "derived_from_import",
            "imported_and_archived",
        }:
            provenance["acquisition_mode"] = "imported_and_accepted"
    else:
        provenance.setdefault("acquisition_mode", "experienced")

    import_chain = list(provenance.get("import_chain") or [])
    if not import_chain or not (
        import_chain[-1].get("instance_id") == local_instance_id
        and import_chain[-1].get("export_id") == export_id
    ):
        import_chain.append(
            {
                "instance_id": local_instance_id,
                "imported_at": imported_at,
                "export_id": export_id,
            }
        )
    provenance["import_chain"] = import_chain
    prepared["provenance"] = provenance

    warnings: list[dict[str, Any]] = []
    for modification in provenance.get("modification_chain") or []:
        kind = (
            modification.get("modification_kind")
            if isinstance(modification, dict)
            else None
        )
        if kind not in _KNOWN_MODIFICATION_KINDS:
            warnings.append(
                {
                    "code": "material_change_unverifiable",
                    "ref": record.get("ref") or origin_id,
                    "modification_kind": kind,
                }
            )
    return prepared, warnings


def _prepare_import_memory(
    record: dict[str, Any],
    *,
    intent: str,
    source: dict[str, Any],
    export_id: str,
    local_instance_id: str,
    imported_at: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from core.digest import content_hash_v1, normalize_v1

    prepared, warnings = _append_import_provenance(
        record,
        origin_id=record["ref"].split(":", 1)[-1],
        intent=intent,
        source=source,
        export_id=export_id,
        local_instance_id=local_instance_id,
        imported_at=imported_at,
    )
    if intent not in ("port", "duplicate"):
        prepared["access_count"] = 0
        prepared["last_accessed"] = None
    prepared.setdefault("content_hash_v1", content_hash_v1(prepared["content"]))
    prepared["_transient_normalized_content"] = normalize_v1(prepared["content"])
    return prepared, warnings


def _protected_memory_record(section: str, record: dict[str, Any]) -> dict[str, Any]:
    converted = copy.deepcopy(record)
    metadata = copy.deepcopy(converted.get("metadata") or {})
    if section == "worldview":
        metadata.update(
            {
                key: converted.get(key)
                for key in ("category", "confidence", "stability")
                if key in converted
            }
        )
        converted["type"] = "worldview"
    elif section == "goals":
        metadata.update(
            {
                key: converted.get(key)
                for key in (
                    "title",
                    "description",
                    "priority",
                    "source",
                    "due_at",
                    "progress",
                    "blocked_by",
                    "parent_ref",
                )
                if key in converted
            }
        )
        converted["type"] = "goal"
        converted["content"] = converted.get("description") or converted["title"]
    converted["metadata"] = metadata
    return converted


def prepare_protected_section_import(
    section: str,
    section_data: Any,
    *,
    intent: str,
    source: dict[str, Any],
    export_id: str,
    local_instance_id: str,
    imported_at: str,
) -> tuple[Any, list[dict[str, Any]]]:
    """Prepare one validated protected section for DB-owned replacement."""

    warnings: list[dict[str, Any]] = []
    if section == "narrative":
        prepared_narrative = copy.deepcopy(section_data or {})
        for group, records in prepared_narrative.items():
            for index, record in enumerate(records):
                prepared, record_warnings = _append_import_provenance(
                    record,
                    origin_id=str(record.get("ref") or f"{group}:{index}"),
                    intent=intent,
                    source=source,
                    export_id=export_id,
                    local_instance_id=local_instance_id,
                    imported_at=imported_at,
                )
                records[index] = prepared
                warnings.extend(record_warnings)
        return prepared_narrative, warnings

    prepared_records: list[dict[str, Any]] = []
    for index, record in enumerate(copy.deepcopy(section_data or [])):
        if section in {"worldview", "goals"}:
            prepared, record_warnings = _prepare_import_memory(
                _protected_memory_record(section, record),
                intent=intent,
                source=source,
                export_id=export_id,
                local_instance_id=local_instance_id,
                imported_at=imported_at,
            )
        else:
            origin_id = str(
                record.get("ref")
                or record.get("key")
                or record.get("name")
                or record.get("trigger_pattern")
                or index
            )
            prepared, record_warnings = _append_import_provenance(
                record,
                origin_id=origin_id,
                intent=intent,
                source=source,
                export_id=export_id,
                local_instance_id=local_instance_id,
                imported_at=imported_at,
            )
            if section == "emotional_triggers":
                from core.digest import normalize_v1

                prepared["_transient_normalized_content"] = normalize_v1(
                    prepared["trigger_pattern"]
                )
        prepared_records.append(prepared)
        warnings.extend(record_warnings)
    return prepared_records, warnings


def prepare_protected_section_restore(section: str, section_data: Any) -> Any:
    """Convert a local HMX snapshot back to storage input without new provenance."""

    from core.digest import content_hash_v1, normalize_v1

    if section == "narrative":
        return copy.deepcopy(section_data or {})

    prepared_records: list[dict[str, Any]] = []
    for record in copy.deepcopy(section_data or []):
        prepared = (
            _protected_memory_record(section, record)
            if section in {"worldview", "goals"}
            else record
        )
        if section in {"worldview", "goals"}:
            prepared.setdefault("content_hash_v1", content_hash_v1(prepared["content"]))
            prepared["_transient_normalized_content"] = normalize_v1(
                prepared["content"]
            )
        elif section == "emotional_triggers":
            prepared["_transient_normalized_content"] = normalize_v1(
                prepared["trigger_pattern"]
            )
        prepared_records.append(prepared)
    return prepared_records


async def dry_run_hmx(
    conn,
    data: dict[str, Any],
    *,
    strategy: str = "additive",
    retry_failed_work: bool = False,
    replace_sections: Iterable[str] | None = None,
    allow_locally_trusted_lineage: bool = False,
) -> HmxDryRunResult:
    """Validate and forecast an HMX import without opening a write transaction."""

    from core.digest import normalize_v1

    if not isinstance(data, dict):
        raise HmxSchemaError("HMX input must be a JSON object")
    _validate_import_header(data)
    intent = validate_intent(str(data["export_intent"]))
    requested_replacements = normalize_replace_sections(replace_sections)
    if requested_replacements and strategy != "authoritative":
        raise HmxPolicyError("replace_sections requires strategy='authoritative'")
    sections = _import_sections(data)
    export_id = str(data["export_id"])
    warnings: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    invalid_records = 0
    invalid_by_section: dict[str, int] = {}
    warnings.extend(
        _unknown_section_warnings(
            sections, isolated=strategy in {"deliberative", "analysis_only"}
        )
    )

    validated: dict[str, list[dict[str, Any]]] = {}
    for section in (
        "memories",
        "episodes",
        "relationships",
        "identity",
        "worldview",
        "goals",
        "drives",
        "emotional_triggers",
        "clusters",
        "raw_units",
    ):
        records, section_warnings = _validated_records(
            data, section, sections.get(section, [])
        )
        validated[section] = records
        counts[section] = len(records)
        invalid_by_section[section] = len(section_warnings)
        invalid_records += len(section_warnings)
        warnings.extend(section_warnings)

    narrative = copy.deepcopy(sections.get("narrative") or {})
    if narrative:
        try:
            validate_hmx_document(_document_probe(data, {"narrative": narrative}))
        except HmxSchemaError as exc:
            invalid_records += 1
            invalid_by_section["narrative"] = 1
            counts["narrative"] = 0
            warnings.append(
                {
                    "code": "schema_validation_error",
                    "section": "narrative",
                    "error": str(exc),
                }
            )
        else:
            counts["narrative"] = sum(len(records) for records in narrative.values())
    else:
        counts["narrative"] = 0
        invalid_by_section["narrative"] = 0

    in_flight, in_flight_warnings = _validated_in_flight_work(
        data, sections.get("in_flight_work", {})
    )
    if _section_has_records("in_flight_work", sections.get("in_flight_work")):
        invalid_records += len(in_flight_warnings)
        warnings.extend(in_flight_warnings)

    available_refs = {
        str(record.get("ref"))
        for section in ("memories", "worldview", "goals", "raw_units")
        for record in validated[section]
        if record.get("ref")
    }
    eligible_in_flight = 0
    failed_in_flight = 0
    if intent in ("port", "duplicate") and strategy in {
        "additive",
        "authoritative",
    }:
        for group, records in in_flight.items():
            for record in records:
                required_refs = (
                    list(record.get("input_refs") or [])
                    + list(record.get("output_refs") or [])
                    if group == "consolidation_tasks"
                    else list(record.get("memory_refs") or [])
                )
                if group == "consolidation_tasks" and record.get("trigger_ref"):
                    required_refs.append(str(record["trigger_ref"]))
                if (
                    group == "consolidation_tasks"
                    and record.get("task_type") == "episode_merge"
                    and not record.get("output_refs")
                ):
                    required_refs.append("<target_memory_ref>")
                missing_refs = sorted(set(required_refs) - available_refs)
                if missing_refs:
                    warnings.append(
                        {
                            "code": "dropped_in_flight_task",
                            "section": "in_flight_work",
                            "task_group": group,
                            "ref": record.get("ref"),
                            "missing_refs": missing_refs,
                            "error": "required inputs are absent from this exchange",
                        }
                    )
                    continue
                eligible_in_flight += 1
                if record.get("status") == "failed":
                    failed_in_flight += 1
        counts["in_flight_work"] = eligible_in_flight
        if failed_in_flight:
            warnings.append(
                {
                    "code": (
                        "failed_in_flight_retry_requested"
                        if retry_failed_work
                        else "failed_in_flight_preserved"
                    ),
                    "section": "in_flight_work",
                    "count": failed_in_flight,
                    "error": (
                        "failed tasks will be reset to pending by explicit request"
                        if retry_failed_work
                        else (
                            "failed tasks will remain non-runnable diagnostics; "
                            "set retry_failed_work=true (CLI: --retry-failed-work) "
                            "only to rerun them"
                        )
                    ),
                }
            )

    if _section_has_records("audit_records", sections.get("audit_records")):
        from core.protected_replacement import forecast_audit_records

        audit_forecast = await forecast_audit_records(conn, data)
        counts["audit_records"] = audit_forecast.inserted
        warnings.extend(audit_forecast.warnings)
        conflicts.extend(audit_forecast.conflicts)

    for deferred_section in ("raw_units", "config", "in_flight_work"):
        if _section_has_records(deferred_section, sections.get(deferred_section)):
            supported_additive = (
                deferred_section in {"raw_units", "in_flight_work"}
                and intent in ("port", "duplicate")
                and strategy in {"additive", "authoritative"}
            )
            if supported_additive:
                continue
            isolated = strategy in {"deliberative", "analysis_only"}
            warnings.append(
                {
                    "code": "unsupported_section",
                    "section": deferred_section,
                    "error": (
                        "section will remain isolated and cannot be admitted to active state"
                        if isolated
                        else "section import is assigned to a later HMX slice and would not be applied"
                    ),
                }
            )
            if strategy == "additive":
                counts[deferred_section] = 0
            elif deferred_section != "raw_units":
                counts[deferred_section] = 1

    memory_candidates: list[tuple[str, str, str]] = []
    candidate_sections = (
        ("memories",)
        if strategy == "authoritative"
        else ("memories", "worldview", "goals")
    )
    for section in candidate_sections:
        for record in validated[section]:
            content = record.get("content")
            if section == "goals":
                content = record.get("description") or record.get("title")
            memory_candidates.append(
                (section, str(record.get("ref") or ""), normalize_v1(content or ""))
            )

    normalized_values = sorted({normalized for _, _, normalized in memory_candidates})
    existing_normalized: set[str] = set()
    if normalized_values:
        rows = await conn.fetch(
            "SELECT DISTINCT regexp_replace(lower(btrim(content)), '\\s+', ' ', 'g') "
            "AS normalized_content FROM memories "
            "WHERE regexp_replace(lower(btrim(content)), '\\s+', ' ', 'g') = ANY($1::text[])",
            normalized_values,
        )
        existing_normalized = {str(row["normalized_content"]) for row in rows}

    duplicate_refs: list[str] = []
    seen = set(existing_normalized)
    new_by_section = {section: 0 for section in ("memories", "worldview", "goals")}
    for section, ref, normalized in memory_candidates:
        if normalized in seen:
            duplicate_refs.append(ref)
            conflicts.append({"code": "duplicate_content", "ref": ref})
        else:
            seen.add(normalized)
            new_by_section[section] += 1
    if strategy in {"additive", "authoritative"}:
        counts.update(new_by_section)
    if strategy == "authoritative":
        counts["worldview"] = (
            len(validated["worldview"]) if "worldview" in requested_replacements else 0
        )
        counts["goals"] = (
            len(validated["goals"]) if "goals" in requested_replacements else 0
        )
        for section in PROTECTED_SECTIONS:
            if section not in requested_replacements:
                counts[section] = 0
    counts["duplicate_memories"] = len(duplicate_refs)
    counts["invalid_records"] = invalid_records
    counts["total_records"] = sum(
        counts.get(section, 0)
        for section in (
            "memories",
            "episodes",
            "relationships",
            "identity",
            "worldview",
            "goals",
            "drives",
            "emotional_triggers",
            "clusters",
            "narrative",
            "raw_units",
            "audit_records",
        )
    )

    target_state = _coerce_json(await conn.fetchval("SELECT hexis_instance_is_empty()"))
    protected_present = sorted(
        section
        for section in PROTECTED_SECTIONS
        if _section_has_records(section, sections.get(section))
    )
    protected_available = sorted(
        section for section in PROTECTED_SECTIONS if section in sections
    )
    direct_protected_allowed = not protected_present or (
        intent in ("port", "duplicate") and bool(target_state.get("is_empty"))
    )
    if strategy == "additive" and protected_present and not direct_protected_allowed:
        conflicts.append(
            {
                "code": "bootstrap_state_violation",
                "sections": protected_present,
                "reason": "protected state requires port/duplicate intent and an empty target",
                "recommended_action": (
                    "use authoritative import with explicit replacement sections to "
                    "run the MVP-PR Protected Section Replacement Protocol"
                ),
            }
        )

    source = data.get("source") or {}
    source_lineage = str(source.get("hexis_lineage_id") or "")
    local_lineage = str(
        await conn.fetchval(
            "SELECT value #>> '{}' FROM config WHERE key='agent.lineage_id'"
        )
        or ""
    )
    direct_lineage_allowed = not (
        intent in ("port", "duplicate")
        and not target_state.get("is_empty", False)
        and source_lineage
        and source_lineage != local_lineage
    )
    if strategy == "additive" and not direct_lineage_allowed:
        conflicts.append(
            {
                "code": "lineage_mismatch",
                "source_lineage": source_lineage,
                "target_lineage": local_lineage,
            }
        )

    authoritative_allowed = True
    authoritative_operations: list[dict[str, Any]] = []
    if strategy == "authoritative":
        if intent not in {"port", "duplicate"}:
            authoritative_allowed = False
            conflicts.append(
                {
                    "code": "invalid_authoritative_intent",
                    "intent": intent,
                    "error": "authoritative replacement requires export_intent port or duplicate",
                }
            )
        if not requested_replacements:
            authoritative_allowed = False
            conflicts.append(
                {
                    "code": "replacement_sections_missing",
                    "error": "authoritative import requires at least one explicit --replace section",
                }
            )
        features = set((data.get("capabilities") or {}).get("optional_features") or [])
        if "protected_replacement_protocol_v1" not in features:
            authoritative_allowed = False
            conflicts.append(
                {
                    "code": "protected_replacement_capability_missing",
                    "error": "the HMX exchange does not advertise protected_replacement_protocol_v1",
                }
            )

        for section in sorted(set(protected_available) - set(requested_replacements)):
            warnings.append(
                {
                    "code": "protected_section_not_selected",
                    "section": section,
                    "error": "section will remain unchanged; add --replace to request whole-section replacement",
                }
            )

        local_envelope = None
        if requested_replacements:
            local_envelope = await export_hmx(
                conn,
                intent="port",
                include_in_flight_work=False,
                include_audit_records=False,
            )
        from core.digest import protected_section_digest_v1

        for section in requested_replacements:
            if section not in sections:
                authoritative_allowed = False
                conflicts.append(
                    {
                        "code": "protected_section_missing",
                        "section": section,
                        "error": "the HMX exchange does not contain the requested section",
                    }
                )
                continue
            if invalid_by_section.get(section, 0):
                authoritative_allowed = False
                conflicts.append(
                    {
                        "code": "schema_validation_error",
                        "section": section,
                        "invalid_records": invalid_by_section[section],
                        "error": "authoritative replacement cannot skip invalid protected records",
                    }
                )
                continue
            assert local_envelope is not None
            declared_digest = str(
                (data.get("section_digests") or {}).get(section) or ""
            )
            actual_digest = protected_section_digest_v1(section, sections[section])
            if not declared_digest or actual_digest != declared_digest:
                authoritative_allowed = False
                conflicts.append(
                    {
                        "code": (
                            "protected_section_digest_missing"
                            if not declared_digest
                            else "protected_section_digest_invalid"
                        ),
                        "section": section,
                        "declared_digest_v1": declared_digest or None,
                        "actual_digest_v1": actual_digest,
                    }
                )
                continue
            local_digest = str(local_envelope["section_digests"][section])
            labels_equal = bool(source_lineage) and source_lineage == local_lineage
            content_identical = local_digest == declared_digest
            fast_path_candidate = (
                content_identical and labels_equal and allow_locally_trusted_lineage
            )
            operation = {
                "section": section,
                "local_digest_v1": local_digest,
                "imported_digest_v1": declared_digest,
                "disposition": (
                    "verified_noop_candidate"
                    if fast_path_candidate
                    else "pending_acknowledgement"
                ),
                "agent_acknowledgement_required": not fast_path_candidate,
            }
            authoritative_operations.append(operation)
            if not content_identical:
                conflicts.append(
                    {
                        "code": "protected_section_digest_mismatch",
                        "section": section,
                        "local_digest_v1": local_digest,
                        "imported_digest_v1": declared_digest,
                    }
                )
            if not labels_equal:
                conflicts.append(
                    {
                        "code": "lineage_mismatch",
                        "section": section,
                        "source_lineage": source_lineage,
                        "target_lineage": local_lineage,
                    }
                )
            elif content_identical and not allow_locally_trusted_lineage:
                conflicts.append(
                    {
                        "code": "unverified_lineage",
                        "section": section,
                        "error": "matching lineage labels are not trusted automatically; acknowledgement will be required",
                    }
                )

    strategy_available = strategy in set(SUPPORTED_IMPORT_STRATEGIES)
    if not strategy_available:
        warnings.append(
            {
                "code": "strategy_not_available",
                "strategy": strategy,
                "error": "unknown HMX import strategy",
            }
        )

    privacy = copy.deepcopy(data.get("privacy") or {})
    if privacy.get("contains_sensitive_content"):
        warnings.append(
            {
                "code": "sensitive_content",
                "redaction_policy": privacy.get("redaction_policy", "unknown"),
                "error": "treat this exchange file as sensitive data",
            }
        )

    if strategy == "deliberative":
        protected_decision = "staged_review"
        policy_allowed = True
    elif strategy == "analysis_only":
        protected_decision = "analysis_only"
        policy_allowed = True
    elif strategy == "authoritative":
        protected_decision = "protected_replacement_protocol"
        policy_allowed = authoritative_allowed
    else:
        protected_decision = "direct_import" if direct_protected_allowed else "blocked"
        policy_allowed = direct_protected_allowed and direct_lineage_allowed

    return HmxDryRunResult(
        export_id=export_id,
        intent=intent,
        strategy=strategy,
        target_state=target_state,
        can_import=policy_allowed and strategy_available,
        counts=counts,
        duplicate_refs=tuple(duplicate_refs),
        conflicts=tuple(conflicts),
        warnings=tuple(warnings),
        protected_policy={
            "sections": (
                list(requested_replacements)
                if strategy == "authoritative"
                else protected_present
            ),
            "allowed": policy_allowed,
            "decision": protected_decision,
            "requires_empty_target": strategy == "additive" and bool(protected_present),
            "operations": authoritative_operations,
        },
        privacy=privacy,
        estimated_embedding_items=(
            (
                sum(new_by_section.values())
                + (counts.get("raw_units", 0) if intent in ("port", "duplicate") else 0)
            )
            if strategy in {"additive", "authoritative"}
            else 0
        ),
    )


_LIST_SECTIONS = frozenset(
    {
        "memories",
        "episodes",
        "relationships",
        "identity",
        "worldview",
        "goals",
        "drives",
        "emotional_triggers",
        "clusters",
        "raw_units",
    }
)


def _exchange_records(
    data: dict[str, Any],
) -> list[tuple[str, str | None, dict[str, Any]]]:
    """Return independently reviewable valid records from an HMX document."""

    sections = _import_sections(data)
    records: list[tuple[str, str | None, dict[str, Any]]] = []
    for section, value in sections.items():
        if not _section_has_records(section, value):
            continue
        if section in _LIST_SECTIONS:
            valid, _ = _validated_records(data, section, value)
            for index, record in enumerate(valid):
                source_ref = str(
                    record.get("ref")
                    or record.get("source_ref")
                    or record.get("name")
                    or record.get("key")
                    or f"{section}:{index}"
                )
                records.append((section, source_ref, record))
        elif section == "narrative":
            try:
                validate_hmx_document(_document_probe(data, {section: value}))
            except HmxSchemaError:
                continue
            records.append((section, f"{section}:bundle", copy.deepcopy(value)))
        else:
            # Object-shaped and future sections remain inspectable even when
            # this importer cannot admit them into active state yet.
            records.append((section, f"{section}:bundle", copy.deepcopy(value)))
    return records


def _with_isolated_provenance(
    record: Any,
    *,
    source_ref: str,
    mode: str,
    intent: str,
    source: dict[str, Any],
    export_id: str,
    local_instance_id: str,
    imported_at: str,
) -> Any:
    if not isinstance(record, dict):
        return copy.deepcopy(record)
    prepared, _ = _append_import_provenance(
        record,
        origin_id=source_ref.split(":", 1)[-1],
        intent=intent,
        source=source,
        export_id=export_id,
        local_instance_id=local_instance_id,
        imported_at=imported_at,
        acquisition_mode=mode,
    )
    return prepared


async def stage_hmx(conn, data: dict[str, Any]) -> HmxStagingResult:
    """Load valid HMX records into the deliberative review store only."""

    forecast = await dry_run_hmx(conn, data, strategy="deliberative")
    records = _exchange_records(data)
    source = data.get("source") or {}
    export_id = str(data["export_id"])
    imported_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    local_context = await load_source_context(conn)
    duplicate_refs = set(forecast.duplicate_refs)
    unsupported = {"raw_units", "config", "in_flight_work", "audit_records"}
    counts: dict[str, int] = {}
    staging_ids: list[str] = []

    header = copy.deepcopy(data)
    header.pop("sections", None)
    async with conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock(hashtext('hmx_import'))")
        batch_id = await conn.fetchval(
            "INSERT INTO hmx_import_batches "
            "(export_id, export_intent, strategy, source, privacy, envelope) "
            "VALUES ($1, $2, 'deliberative', $3::jsonb, $4::jsonb, $5::jsonb) RETURNING id",
            export_id,
            forecast.intent,
            json.dumps(source),
            json.dumps(data.get("privacy") or {}),
            json.dumps(header),
        )
        for section, source_ref, record in records:
            prepared = _with_isolated_provenance(
                record,
                source_ref=source_ref or section,
                mode="imported_staged",
                intent=forecast.intent,
                source=source,
                export_id=export_id,
                local_instance_id=local_context["instance_id"],
                imported_at=imported_at,
            )
            record_conflicts: list[dict[str, Any]] = []
            if source_ref in duplicate_refs:
                record_conflicts.append(
                    {"code": "duplicate_content", "ref": source_ref}
                )
            if section in PROTECTED_SECTIONS:
                record_conflicts.append(
                    {
                        "code": "protected_replacement_requested",
                        "section": section,
                        "target_state": forecast.target_state.get("state"),
                    }
                )
            if section in unsupported:
                record_conflicts.append(
                    {"code": "unsupported_section", "section": section}
                )
            staging_id = await conn.fetchval(
                "INSERT INTO hmx_import_staging "
                "(batch_id, section, source_ref, record, conflicts) "
                "VALUES ($1, $2, $3, $4::jsonb, $5::jsonb) RETURNING id",
                batch_id,
                section,
                source_ref,
                json.dumps(prepared),
                json.dumps(record_conflicts),
            )
            staging_ids.append(str(staging_id))
            counts[section] = counts.get(section, 0) + 1

    return HmxStagingResult(
        export_id=export_id,
        intent=forecast.intent,
        strategy="deliberative",
        batch_id=str(batch_id),
        staged=counts,
        staging_ids=tuple(staging_ids),
        conflicts=forecast.conflicts,
        warnings=forecast.warnings,
    )


async def load_analysis_hmx(conn, data: dict[str, Any]) -> HmxAnalysisResult:
    """Load valid HMX records into physically isolated analysis storage."""

    forecast = await dry_run_hmx(conn, data, strategy="analysis_only")
    records = _exchange_records(data)
    source = data.get("source") or {}
    export_id = str(data["export_id"])
    imported_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    local_context = await load_source_context(conn)
    counts: dict[str, int] = {}
    analysis_ids: list[str] = []
    header = copy.deepcopy(data)
    header.pop("sections", None)

    async with conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock(hashtext('hmx_import'))")
        batch_id = await conn.fetchval(
            "INSERT INTO hmx_analysis_batches "
            "(export_id, export_intent, source, privacy, envelope) "
            "VALUES ($1, $2, $3::jsonb, $4::jsonb, $5::jsonb) RETURNING id",
            export_id,
            forecast.intent,
            json.dumps(source),
            json.dumps(data.get("privacy") or {}),
            json.dumps(header),
        )
        for section, source_ref, record in records:
            prepared = _with_isolated_provenance(
                record,
                source_ref=source_ref or section,
                mode="analysis_only",
                intent=forecast.intent,
                source=source,
                export_id=export_id,
                local_instance_id=local_context["instance_id"],
                imported_at=imported_at,
            )
            analysis_id = await conn.fetchval(
                "INSERT INTO hmx_analysis_records "
                "(batch_id, section, source_ref, record, metadata) "
                "VALUES ($1, $2, $3, $4::jsonb, $5::jsonb) RETURNING id",
                batch_id,
                section,
                source_ref,
                json.dumps(prepared),
                json.dumps(
                    {
                        "conflicts": [
                            conflict
                            for conflict in forecast.conflicts
                            if conflict.get("ref") == source_ref
                        ]
                    }
                ),
            )
            analysis_ids.append(str(analysis_id))
            counts[section] = counts.get(section, 0) + 1

    return HmxAnalysisResult(
        export_id=export_id,
        intent=forecast.intent,
        strategy="analysis_only",
        batch_id=str(batch_id),
        loaded=counts,
        analysis_ids=tuple(analysis_ids),
        warnings=forecast.warnings,
    )


async def import_hmx(
    conn,
    data: dict[str, Any],
    *,
    strategy: str = "additive",
    initial_ref_map: dict[str, str] | None = None,
    reviewed: bool = False,
    retry_failed_work: bool = False,
    replace_sections: Iterable[str] | None = None,
    replacement_rationale: str | None = None,
    allow_locally_trusted_lineage: bool = False,
    verifier: Any = None,
    operator_signature: str | None = None,
    operator_identity: str | None = None,
    force_replace: bool = False,
    override_acknowledgement: str | None = None,
    override_reason_code: str | None = None,
    override_evidence_ref: str | None = None,
) -> HmxImportResult | HmxAuthoritativeResult | HmxStagingResult | HmxAnalysisResult:
    """Import HMX using additive, authoritative, or isolated storage.

    Protected state is admitted directly only for a port/duplicate into a
    target that is empty by HMX's diagnostic predicate. Authoritative requests
    use the durable Protected Section Replacement Protocol.
    """

    if not isinstance(data, dict):
        raise HmxSchemaError("HMX input must be a JSON object")
    _validate_import_header(data)
    intent = validate_intent(str(data["export_intent"]))
    if strategy == "authoritative":
        from core.protected_replacement import import_authoritative_hmx

        return await import_authoritative_hmx(
            conn,
            data,
            replace_sections=normalize_replace_sections(replace_sections),
            rationale=replacement_rationale,
            verifier=verifier,
            allow_locally_trusted_lineage=allow_locally_trusted_lineage,
            retry_failed_work=retry_failed_work,
            operator_signature=operator_signature,
            operator_identity=operator_identity,
            force_replace=force_replace,
            override_acknowledgement=override_acknowledgement,
            override_reason_code=override_reason_code,
            override_evidence_ref=override_evidence_ref,
        )
    if strategy == "deliberative":
        return await stage_hmx(conn, data)
    if strategy == "analysis_only":
        return await load_analysis_hmx(conn, data)
    if strategy != "additive":
        raise HmxPolicyError(f"unsupported import strategy {strategy!r}")

    sections = _import_sections(data)
    warnings = _unknown_section_warnings(sections, isolated=False)
    for deferred_section in ("config",):
        if _section_has_records(deferred_section, sections.get(deferred_section)):
            warnings.append(
                {
                    "code": "unsupported_section",
                    "section": deferred_section,
                    "error": "section import is assigned to a later HMX slice and was not applied",
                }
            )
    protected_present = {
        section
        for section in PROTECTED_SECTIONS
        if _section_has_records(section, sections.get(section))
    }
    source = data.get("source") or {}
    export_id = str(data["export_id"])
    imported_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    local_context = await load_source_context(conn)

    memory_records, record_warnings = _validated_records(
        data, "memories", sections.get("memories", [])
    )
    warnings.extend(record_warnings)
    raw_records: list[dict[str, Any]] = []
    if _section_has_records("raw_units", sections.get("raw_units")):
        if intent in ("port", "duplicate"):
            raw_records, raw_warnings = _validated_records(
                data, "raw_units", sections.get("raw_units", [])
            )
            warnings.extend(raw_warnings)
        else:
            warnings.append(
                {
                    "code": "unsupported_section",
                    "section": "raw_units",
                    "error": "raw units enter active RecMem only for port or duplicate intent",
                }
            )
    in_flight_work: dict[str, list[dict[str, Any]]] = {
        "consolidation_tasks": [],
        "reconsolidation_tasks": [],
    }
    if _section_has_records("in_flight_work", sections.get("in_flight_work")):
        if intent in ("port", "duplicate"):
            in_flight_work, in_flight_warnings = _validated_in_flight_work(
                data, sections.get("in_flight_work")
            )
            warnings.extend(in_flight_warnings)
        else:
            warnings.append(
                {
                    "code": "unsupported_section",
                    "section": "in_flight_work",
                    "error": "in-flight work enters active queues only for port or duplicate intent",
                }
            )
    goal_records: list[dict[str, Any]] = []
    for section in ("worldview", "goals"):
        records, section_warnings = _validated_records(
            data, section, sections.get(section, [])
        )
        warnings.extend(section_warnings)
        if section == "goals":
            goal_records = records
        memory_records.extend(
            _protected_memory_record(section, record) for record in records
        )

    prepared_memories: list[dict[str, Any]] = []
    for record in memory_records:
        prepared, provenance_warnings = _prepare_import_memory(
            record,
            intent=intent,
            source=source,
            export_id=export_id,
            local_instance_id=local_context["instance_id"],
            imported_at=imported_at,
        )
        prepared_memories.append(prepared)
        warnings.extend(provenance_warnings)

    episode_records, episode_warnings = _validated_records(
        data, "episodes", sections.get("episodes", [])
    )
    cluster_records, cluster_warnings = _validated_records(
        data, "clusters", sections.get("clusters", [])
    )
    relationship_records, relationship_warnings = _validated_records(
        data, "relationships", sections.get("relationships", [])
    )
    for record in memory_records:
        superseded_by = record.get("superseded_by")
        if superseded_by:
            relationship_records.append(
                {
                    "source_ref": record["ref"],
                    "target_ref": superseded_by,
                    "edge_type": "SUPERSEDES",
                    "properties": {
                        "source_type": "memory",
                        "target_type": "memory",
                        "reason": "legacy_superseded_by",
                    },
                }
            )
    warnings.extend(episode_warnings + cluster_warnings + relationship_warnings)

    protected_records: dict[str, list[dict[str, Any]]] = {}
    for section in ("identity", "drives", "emotional_triggers"):
        records, section_warnings = _validated_records(
            data, section, sections.get(section, [])
        )
        warnings.extend(section_warnings)
        prepared_records: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            origin_id = str(
                record.get("ref")
                or record.get("key")
                or record.get("name")
                or record.get("trigger_pattern")
                or index
            )
            prepared, provenance_warnings = _append_import_provenance(
                record,
                origin_id=origin_id,
                intent=intent,
                source=source,
                export_id=export_id,
                local_instance_id=local_context["instance_id"],
                imported_at=imported_at,
            )
            if section == "emotional_triggers":
                from core.digest import normalize_v1

                prepared["_transient_normalized_content"] = normalize_v1(
                    prepared["trigger_pattern"]
                )
            prepared_records.append(prepared)
            warnings.extend(provenance_warnings)
        protected_records[section] = prepared_records

    narrative = copy.deepcopy(sections.get("narrative") or {})
    if narrative:
        try:
            validate_hmx_document(_document_probe(data, {"narrative": narrative}))
        except HmxSchemaError as exc:
            warnings.append(
                {
                    "code": "schema_validation_error",
                    "section": "narrative",
                    "error": str(exc),
                }
            )
            narrative = {}
        else:
            for group, records in narrative.items():
                for index, record in enumerate(records):
                    prepared, provenance_warnings = _append_import_provenance(
                        record,
                        origin_id=str(record.get("ref") or f"{group}:{index}"),
                        intent=intent,
                        source=source,
                        export_id=export_id,
                        local_instance_id=local_context["instance_id"],
                        imported_at=imported_at,
                    )
                    records[index] = prepared
                    warnings.extend(provenance_warnings)

    async with conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock(hashtext('hmx_import'))")
        target_state = _coerce_json(
            await conn.fetchval("SELECT hexis_instance_is_empty()")
        )
        protected_direct = intent in ("port", "duplicate") and target_state.get(
            "is_empty", False
        )
        protected_reviewed = reviewed and target_state.get("is_empty", False)
        if protected_present and not (protected_direct or protected_reviewed):
            reason = (
                "protected sections may be imported directly only into an empty target "
                "with export_intent port or duplicate; use deliberative handling or the "
                "Protected Section Replacement Protocol"
            )
            raise HmxPolicyError(f"bootstrap_state_violation: {reason}")

        source_lineage = str(source.get("hexis_lineage_id") or "")
        local_lineage = str(
            await conn.fetchval(
                "SELECT value #>> '{}' FROM config WHERE key='agent.lineage_id'"
            )
            or ""
        )
        if (
            intent in ("port", "duplicate")
            and not target_state.get("is_empty", False)
            and not reviewed
            and source_lineage
            and source_lineage != local_lineage
        ):
            raise HmxPolicyError(
                "lineage_mismatch: port/duplicate source lineage differs from the "
                "active target; use deliberative additive import instead"
            )

        memory_result = _coerce_json(
            await conn.fetchval(
                "SELECT hmx_import_memories($1::jsonb)", json.dumps(prepared_memories)
            )
        )
        if memory_result.get("errors"):
            warnings.extend(
                {
                    "code": "schema_validation_error",
                    "section": "memories",
                    **error,
                }
                for error in memory_result["errors"]
            )

        duplicate_memory_refs = set(memory_result.get("duplicate_refs") or [])
        memory_ref_map = dict(memory_result.get("ref_map") or {})
        queued_memory_ids = [
            memory_ref_map[record["ref"]]
            for record in prepared_memories
            if record.get("ref") in memory_ref_map
            and record.get("ref") not in duplicate_memory_refs
        ]
        if queued_memory_ids:
            await conn.fetchval(
                "SELECT hmx_queue_reembed($1::uuid[])", queued_memory_ids
            )

        ref_map = dict(initial_ref_map or {})
        ref_map.update(memory_result.get("ref_map") or {})
        goal_result = _coerce_json(
            await conn.fetchval(
                "SELECT hmx_remap_goal_references($1::jsonb, $2::jsonb)",
                json.dumps(goal_records),
                json.dumps(ref_map),
            )
        )
        identity_result = _coerce_json(
            await conn.fetchval(
                "SELECT hmx_import_identity($1::jsonb, $2::jsonb)",
                json.dumps(protected_records["identity"]),
                json.dumps(ref_map),
            )
        )
        drive_result = _coerce_json(
            await conn.fetchval(
                "SELECT hmx_import_drives($1::jsonb)",
                json.dumps(protected_records["drives"]),
            )
        )
        narrative_result = _coerce_json(
            await conn.fetchval(
                "SELECT hmx_import_narrative($1::jsonb, $2::jsonb)",
                json.dumps(narrative),
                json.dumps(ref_map),
            )
        )
        ref_map.update(narrative_result.get("ref_map") or {})
        trigger_result = _coerce_json(
            await conn.fetchval(
                "SELECT hmx_import_emotional_triggers($1::jsonb, $2::jsonb)",
                json.dumps(protected_records["emotional_triggers"]),
                json.dumps(ref_map),
            )
        )
        episode_result = _coerce_json(
            await conn.fetchval(
                "SELECT hmx_import_episodes($1::jsonb, $2::jsonb)",
                json.dumps(episode_records),
                json.dumps(ref_map),
            )
        )
        ref_map = dict(episode_result.get("ref_map") or ref_map)
        cluster_result = _coerce_json(
            await conn.fetchval(
                "SELECT hmx_import_clusters($1::jsonb, $2::jsonb)",
                json.dumps(cluster_records),
                json.dumps(ref_map),
            )
        )
        ref_map = dict(cluster_result.get("ref_map") or ref_map)
        relationship_result = _coerce_json(
            await conn.fetchval(
                "SELECT hmx_import_relationships($1::jsonb, $2::jsonb)",
                json.dumps(relationship_records),
                json.dumps(ref_map),
            )
        )
        raw_result = _coerce_json(
            await conn.fetchval(
                "SELECT hmx_import_raw_units($1::jsonb, $2::text, $3::jsonb)",
                json.dumps(raw_records),
                export_id,
                json.dumps(ref_map),
            )
        )
        ref_map.update(raw_result.get("ref_map") or {})
        in_flight_result = _coerce_json(
            await conn.fetchval(
                "SELECT hmx_import_in_flight_work($1::jsonb, $2::text, "
                "$3::jsonb, $4::boolean)",
                json.dumps(in_flight_work),
                export_id,
                json.dumps(ref_map),
                retry_failed_work,
            )
        )
        ref_map.update(in_flight_result.get("ref_map") or {})
        from core.protected_replacement import import_audit_records

        audit_result = await import_audit_records(conn, data)
        warnings.extend(episode_result.get("warnings") or [])
        warnings.extend(cluster_result.get("warnings") or [])
        warnings.extend(relationship_result.get("warnings") or [])
        warnings.extend(goal_result.get("warnings") or [])
        warnings.extend(identity_result.get("warnings") or [])
        warnings.extend(drive_result.get("warnings") or [])
        warnings.extend(narrative_result.get("warnings") or [])
        warnings.extend(trigger_result.get("warnings") or [])
        warnings.extend(raw_result.get("warnings") or [])
        warnings.extend(in_flight_result.get("warnings") or [])
        warnings.extend(audit_result.warnings)

        if target_state.get("is_empty") and intent in ("port", "duplicate"):
            if source_lineage:
                await conn.execute(
                    "SELECT set_config('agent.lineage_id', to_jsonb($1::text))",
                    source_lineage,
                )

    duplicate_refs = tuple(memory_result.get("duplicate_refs") or [])
    conflicts = tuple(
        [{"code": "duplicate_content", "ref": ref} for ref in duplicate_refs]
        + list(audit_result.conflicts)
    )
    inserted = {
        "memories": int(memory_result.get("inserted", 0)),
        "episodes": int(episode_result.get("inserted", 0)),
        "clusters": int(cluster_result.get("inserted", 0)),
        "relationships": int(relationship_result.get("inserted", 0)),
        "identity": int(identity_result.get("imported", 0)),
        "drives": int(drive_result.get("imported", 0)),
        "emotional_triggers": int(trigger_result.get("inserted", 0)),
        "narrative": int(narrative_result.get("imported", 0)),
    }
    if raw_records:
        inserted["raw_units"] = int(raw_result.get("inserted", 0))
    if _section_has_records("in_flight_work", in_flight_work):
        inserted["in_flight_work"] = int(in_flight_result.get("inserted", 0))
    if _section_has_records("audit_records", sections.get("audit_records")):
        inserted["audit_records"] = int(audit_result.inserted)

    return HmxImportResult(
        export_id=export_id,
        intent=intent,
        strategy=strategy,
        target_state=target_state,
        inserted=inserted,
        duplicate_refs=duplicate_refs,
        ref_map=ref_map,
        conflicts=conflicts,
        warnings=tuple(warnings),
        work_summary=(
            {
                key: int(in_flight_result.get(key, 0))
                for key in (
                    "inserted",
                    "duplicates",
                    "dropped",
                    "failed_preserved",
                    "requeued",
                    "retried",
                )
            }
            if _section_has_records("in_flight_work", in_flight_work)
            else {}
        ),
    )


def _review_document(
    envelope: Any, section: str, record: dict[str, Any]
) -> dict[str, Any]:
    document = copy.deepcopy(_coerce_json(envelope) or {})
    document["sections"] = {
        section: (
            record
            if section in {"narrative", "config", "in_flight_work", "audit_records"}
            else [record]
        )
    }
    return document


async def pending_hmx_reviews(conn) -> dict[str, Any]:
    return _coerce_json(await conn.fetchval("SELECT hmx_pending_review()"))


async def accept_staged_import(
    conn, staging_id: str, *, rationale: str | None = None
) -> HmxReviewResult:
    """Accept one reviewed record, preserving the batch reference map."""

    unsupported = {"raw_units", "config", "in_flight_work", "audit_records"}
    async with conn.transaction():
        row = await conn.fetchrow(
            "SELECT s.*, b.envelope, b.export_intent FROM hmx_import_staging s "
            "JOIN hmx_import_batches b ON b.id=s.batch_id "
            "WHERE s.id=$1::uuid FOR UPDATE OF s",
            staging_id,
        )
        if row is None:
            raise HmxPolicyError(f"staged import not found: {staging_id}")
        if row["status"] != "pending":
            raise HmxPolicyError(
                f"staged import {staging_id} is already {row['status']}"
            )
        section = str(row["section"])
        if section in unsupported or section not in set(ALL_SECTIONS):
            raise HmxPolicyError(
                f"staged section {section!r} cannot enter active state in this HMX slice"
            )

        record = copy.deepcopy(_coerce_json(row["record"]))
        provenance = copy.deepcopy(record.get("provenance") or {})
        provenance["acquisition_mode"] = (
            "derived_from_import"
            if _modification_is_material(row["modification_kind"])
            else "imported_and_accepted"
        )
        record["provenance"] = provenance
        document = _review_document(row["envelope"], section, record)
        ref_rows = await conn.fetch(
            "SELECT source_ref, local_ref FROM hmx_import_ref_map WHERE batch_id=$1",
            row["batch_id"],
        )
        ref_map = {str(item["source_ref"]): str(item["local_ref"]) for item in ref_rows}
        if section == "relationships":
            missing = [
                ref
                for ref in (record.get("source_ref"), record.get("target_ref"))
                if ref and ref not in ref_map
            ]
            if missing:
                raise HmxPolicyError(
                    "unresolved_dependencies: accept referenced records first: "
                    + ", ".join(missing)
                )

        result = await import_hmx(
            conn,
            document,
            strategy="additive",
            initial_ref_map=ref_map,
            reviewed=True,
        )
        assert isinstance(result, HmxImportResult)
        for source_ref, local_ref in result.ref_map.items():
            await conn.execute(
                "INSERT INTO hmx_import_ref_map (batch_id, source_ref, local_ref) "
                "VALUES ($1, $2, $3) ON CONFLICT (batch_id, source_ref) "
                "DO UPDATE SET local_ref=EXCLUDED.local_ref",
                row["batch_id"],
                source_ref,
                local_ref,
            )
        local_ref = result.ref_map.get(str(row["source_ref"]))
        await conn.execute(
            "UPDATE hmx_import_staging SET status='accepted', local_ref=$2, "
            "decision_rationale=$3, reviewed_at=CURRENT_TIMESTAMP, "
            "updated_at=CURRENT_TIMESTAMP WHERE id=$1::uuid",
            staging_id,
            local_ref,
            rationale,
        )
        await conn.execute(
            "UPDATE hmx_import_batches SET status=CASE WHEN EXISTS "
            "(SELECT 1 FROM hmx_import_staging WHERE batch_id=$1 AND status='pending') "
            "THEN 'pending' ELSE 'reviewed' END, updated_at=CURRENT_TIMESTAMP WHERE id=$1",
            row["batch_id"],
        )
    return HmxReviewResult(
        staging_id=staging_id,
        decision="accepted",
        section=section,
        local_ref=local_ref,
        warnings=result.warnings,
    )


async def reject_staged_import(
    conn, staging_id: str, *, rationale: str
) -> HmxReviewResult:
    if not rationale.strip():
        raise HmxPolicyError("rejection rationale is required")
    async with conn.transaction():
        row = await conn.fetchrow(
            "UPDATE hmx_import_staging SET status='rejected', decision_rationale=$2, "
            "reviewed_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP "
            "WHERE id=$1::uuid AND status='pending' RETURNING section, batch_id",
            staging_id,
            rationale,
        )
        if row is None:
            raise HmxPolicyError(f"pending staged import not found: {staging_id}")
        await conn.execute(
            "UPDATE hmx_import_batches SET status=CASE WHEN EXISTS "
            "(SELECT 1 FROM hmx_import_staging WHERE batch_id=$1 AND status='pending') "
            "THEN 'pending' ELSE 'reviewed' END, updated_at=CURRENT_TIMESTAMP WHERE id=$1",
            row["batch_id"],
        )
    return HmxReviewResult(
        staging_id=staging_id, decision="rejected", section=row["section"]
    )


async def modify_staged_import(
    conn,
    staging_id: str,
    changes: dict[str, Any],
    *,
    modification_kind: str,
    rationale: str,
) -> HmxReviewResult:
    from core.digest import content_hash_v1

    if modification_kind not in _KNOWN_MODIFICATION_KINDS:
        raise HmxPolicyError(f"unknown modification_kind: {modification_kind!r}")
    if not changes:
        raise HmxPolicyError("at least one record change is required")
    async with conn.transaction():
        row = await conn.fetchrow(
            "SELECT s.*, b.envelope FROM hmx_import_staging s "
            "JOIN hmx_import_batches b ON b.id=s.batch_id "
            "WHERE s.id=$1::uuid FOR UPDATE OF s",
            staging_id,
        )
        if row is None or row["status"] != "pending":
            raise HmxPolicyError(f"pending staged import not found: {staging_id}")
        record = copy.deepcopy(_coerce_json(row["record"]))
        previous_hash = record.get("content_hash_v1")
        record.update(copy.deepcopy(changes))
        if "content" in changes:
            record["content_hash_v1"] = content_hash_v1(str(record["content"]))
        provenance = copy.deepcopy(record.get("provenance") or {})
        chain = list(provenance.get("modification_chain") or [])
        local_instance_id = str(
            await conn.fetchval(
                "SELECT COALESCE("
                "(SELECT value #>> '{}' FROM config WHERE key='agent.instance_id'), "
                "(SELECT value #>> '{}' FROM config WHERE key='agent.name'), "
                "'hexis')"
            )
        )
        modification = {
            "instance_id": local_instance_id,
            "modified_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "modification_kind": modification_kind,
            "rationale": rationale,
        }
        if previous_hash:
            modification["previous_content_hash_v1"] = previous_hash
        if record.get("content_hash_v1"):
            modification["new_content_hash_v1"] = record["content_hash_v1"]
        chain.append(modification)
        provenance["modification_chain"] = chain
        provenance["acquisition_mode"] = "imported_staged"
        record["provenance"] = provenance
        review_document = _review_document(row["envelope"], row["section"], record)
        validate_hmx_document(
            _document_probe(review_document, review_document["sections"])
        )
        await conn.execute(
            "UPDATE hmx_import_staging SET record=$2::jsonb, modification_kind=$3, "
            "decision_rationale=$4, updated_at=CURRENT_TIMESTAMP WHERE id=$1::uuid",
            staging_id,
            json.dumps(record),
            modification_kind,
            rationale,
        )
    return HmxReviewResult(
        staging_id=staging_id, decision="modified", section=row["section"]
    )


async def quote_staged_import(
    conn, staging_id: str, *, rationale: str
) -> HmxReviewResult:
    """Retain a staged record as archived foreign context, never active recall."""

    if not rationale.strip():
        raise HmxPolicyError("quote rationale is required")
    async with conn.transaction():
        row = await conn.fetchrow(
            "SELECT s.*, b.envelope FROM hmx_import_staging s "
            "JOIN hmx_import_batches b ON b.id=s.batch_id "
            "WHERE s.id=$1::uuid FOR UPDATE OF s",
            staging_id,
        )
        if row is None or row["status"] != "pending":
            raise HmxPolicyError(f"pending staged import not found: {staging_id}")
        record = _coerce_json(row["record"])
        original_content = str(
            record.get("content")
            or record.get("title")
            or record.get("summary")
            or json.dumps(record, sort_keys=True)
        )
        content = (
            f"Quoted HMX {row['section']} from {row['source_ref']}:\n{original_content}"
        )
        quote = {
            "ref": str(row["source_ref"] or f"quote:{staging_id}"),
            "type": "semantic",
            "status": "archived",
            "content": content,
            "importance": float(record.get("importance", 0.3)),
            "trust_level": float(record.get("trust_level", 0.5)),
            "metadata": {
                "hmx": {
                    "quoted": True,
                    "source_section": row["section"],
                    "staging_id": staging_id,
                    "rationale": rationale,
                }
            },
            "provenance": copy.deepcopy(record.get("provenance") or {}),
        }
        quote["provenance"]["acquisition_mode"] = "imported_and_archived"
        document = _review_document(row["envelope"], "memories", quote)
        result = await import_hmx(conn, document, strategy="additive", reviewed=True)
        assert isinstance(result, HmxImportResult)
        local_ref = result.ref_map.get(quote["ref"])
        await conn.execute(
            "UPDATE hmx_import_staging SET status='quoted', local_ref=$2, "
            "decision_rationale=$3, reviewed_at=CURRENT_TIMESTAMP, "
            "updated_at=CURRENT_TIMESTAMP WHERE id=$1::uuid",
            staging_id,
            local_ref,
            rationale,
        )
        await conn.execute(
            "UPDATE hmx_import_batches SET status=CASE WHEN EXISTS "
            "(SELECT 1 FROM hmx_import_staging WHERE batch_id=$1 AND status='pending') "
            "THEN 'pending' ELSE 'reviewed' END, updated_at=CURRENT_TIMESTAMP WHERE id=$1",
            row["batch_id"],
        )
    return HmxReviewResult(
        staging_id=staging_id,
        decision="quoted",
        section=row["section"],
        local_ref=local_ref,
        warnings=result.warnings,
    )


async def promote_analysis_to_staged(conn, analysis_id: str, *, rationale: str) -> str:
    section = await conn.fetchval(
        "SELECT section FROM hmx_analysis_records WHERE id=$1::uuid", analysis_id
    )
    if section is None:
        raise HmxPolicyError(f"analysis record not found: {analysis_id}")
    promotable = set(ALL_SECTIONS) - {
        "raw_units",
        "config",
        "in_flight_work",
        "audit_records",
    }
    if section not in promotable:
        raise HmxPolicyError(
            f"analysis section {section!r} cannot be promoted in this HMX slice"
        )
    return str(
        await conn.fetchval(
            "SELECT hmx_promote_to_staged($1::uuid, $2)", analysis_id, rationale
        )
    )


async def demote_staged_to_analysis(conn, staging_id: str, *, rationale: str) -> str:
    return str(
        await conn.fetchval(
            "SELECT hmx_demote_to_analysis($1::uuid, $2)", staging_id, rationale
        )
    )
