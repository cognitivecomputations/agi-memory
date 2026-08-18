"""HMX Protected Section Replacement Protocol and authoritative execution."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from core.digest import audit_record_digest_v1, protected_section_digest_v1
from core.memory_exchange import (
    PROTECTED_SECTIONS,
    HmxAuthoritativeResult,
    HmxImportResult,
    HmxPolicyError,
    HmxSchemaError,
    dry_run_hmx,
    export_hmx,
    import_hmx,
    load_source_context,
    prepare_protected_section_import,
    prepare_protected_section_restore,
    validate_hmx_document,
)
from core.trust_anchors import (
    TrustAnchorVerifier,
    TrustStatus,
    TrustVerification,
    UnconfiguredTrustAnchors,
)

logger = logging.getLogger(__name__)

PROTECTED_REPLACEMENT_CAPABILITY = "protected_replacement_protocol_v1"
FAST_PATH_CAPABILITY = "fast_path_verification"
ACKNOWLEDGEMENT_DECISIONS = (
    "accept",
    "refuse",
    "request_modification",
    "defer",
)
OPERATOR_OVERRIDE_ACKNOWLEDGEMENT = (
    "I accept responsibility for replacing this Hexis instance's protected state "
    "without its acknowledgement"
)
OPERATOR_OVERRIDE_REASON_CODES = (
    "agent_paused",
    "agent_terminated",
    "agent_unresponsive",
    "state_corruption",
    "emergency_recovery",
    "lineage_integrity_failure",
)

_EVIDENCE_REF_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:\S(?:.*\S)?$")

_AUDIT_GROUPS = {
    "protected_replacement_audit": "protected_section_replacement",
    "protected_section_verified_audit": "protected_section_verified",
    "protected_replacement_reversion_audit": "protected_section_reverted",
}


class ProtectedReplacementError(HmxPolicyError):
    """A stable protocol error with a machine-readable HMX conflict code."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class ProtectedReplacementResult:
    disposition: str
    section: str
    local_digest_v1: str
    imported_digest_v1: str
    lineage_verification: dict[str, Any]
    agent_acknowledgement_required: bool
    audit_ids: tuple[str, ...] = ()
    replacement_id: str | None = None
    consent_id: str | None = None
    status: str | None = None
    replacement_executor: str | None = None
    reused: bool = False
    conflicts: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class AuditImportResult:
    inserted: int = 0
    duplicates: int = 0
    conflicts: tuple[dict[str, Any], ...] = ()
    warnings: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class _OperatorOverrideAuthorization:
    acknowledgement: str
    reason_code: str
    evidence_ref: str
    signature: str
    operator_identity: str | None
    verification: TrustVerification
    payload_sha256: str


def _coerce_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _audit_source(envelope: Mapping[str, Any]) -> dict[str, Any]:
    source = envelope.get("source") or {}
    return {
        "export_id": str(envelope.get("export_id") or ""),
        "origin_instance": str(source.get("instance_id") or ""),
        "hexis_lineage_id": str(source.get("hexis_lineage_id") or ""),
        "export_intent": str(envelope.get("export_intent") or ""),
    }


def _trust_payload(
    verification: TrustVerification, *, locally_trusted_label: bool
) -> dict[str, Any]:
    return {
        "status": verification.status.value,
        "reason": verification.reason,
        "anchor_id": verification.anchor_id,
        "metadata": dict(verification.metadata),
        "locally_trusted_label": locally_trusted_label,
    }


def _required_capabilities(envelope: Mapping[str, Any]) -> set[str]:
    capabilities = envelope.get("capabilities") or {}
    features = capabilities.get("optional_features") or []
    return {str(feature) for feature in features}


def _validate_request(
    envelope: dict[str, Any], section: str, scope_mode: str
) -> tuple[Any, str]:
    if section not in PROTECTED_SECTIONS:
        raise ProtectedReplacementError(
            "invalid_replacement_section",
            f"section must be one of {', '.join(sorted(PROTECTED_SECTIONS))}",
        )
    if scope_mode != "whole_section":
        raise ProtectedReplacementError(
            "unsupported_replacement_scope",
            "MVP protected replacement supports whole_section only; use "
            "whole_section or deliberative import",
        )
    if envelope.get("export_intent") not in ("port", "duplicate"):
        raise ProtectedReplacementError(
            "invalid_replacement_intent",
            "protected replacement requires export_intent port or duplicate",
        )
    features = _required_capabilities(envelope)
    if PROTECTED_REPLACEMENT_CAPABILITY not in features:
        raise ProtectedReplacementError(
            "protected_replacement_capability_missing",
            "the HMX exchange does not advertise protected_replacement_protocol_v1",
        )
    sections = envelope.get("sections") or {}
    if section not in sections:
        raise ProtectedReplacementError(
            "protected_section_missing", f"the exchange does not contain {section}"
        )
    declared = str((envelope.get("section_digests") or {}).get(section) or "")
    if not declared:
        raise ProtectedReplacementError(
            "protected_section_digest_missing",
            f"the exchange does not declare a protected digest for {section}",
        )
    actual = protected_section_digest_v1(section, sections[section])
    if actual != declared:
        raise ProtectedReplacementError(
            "protected_section_digest_invalid",
            f"declared digest for {section} does not match the imported content",
        )
    return sections[section], declared


async def store_audit_record(
    conn,
    record: dict[str, Any],
    *,
    is_foreign_diagnostic: bool = False,
    imported_at: datetime | None = None,
) -> dict[str, Any]:
    """Insert or compare one immutable audit record by stable ``audit_id``."""

    digest = audit_record_digest_v1(record)
    result = await conn.fetchval(
        "SELECT hmx_store_audit_record($1::jsonb, $2::text, $3::boolean, $4::timestamptz)",
        json.dumps(record),
        digest,
        is_foreign_diagnostic,
        imported_at,
    )
    parsed = _coerce_json(result)
    if not isinstance(parsed, dict):
        raise RuntimeError("hmx_store_audit_record returned a non-object result")
    return parsed


async def _write_verified_audit(
    conn,
    *,
    envelope: dict[str, Any],
    section: str,
    local_digest: str,
    imported_digest: str,
    trust_payload: dict[str, Any],
) -> str:
    audit_id = str(uuid.uuid4())
    record = {
        "audit_id": audit_id,
        "event_type": "protected_section_verified",
        "event_time": _iso_now(),
        "sections_verified": [section],
        "source": _audit_source(envelope),
        "local_digest_v1": local_digest,
        "imported_digest_v1": imported_digest,
        "lineage_verification": trust_payload,
    }
    result = await store_audit_record(conn, record)
    if result.get("status") != "inserted":
        raise RuntimeError(f"verified audit insert returned {result.get('status')}")
    return audit_id


def _request_key(
    *,
    envelope: Mapping[str, Any],
    section: str,
    imported_digest: str,
    local_digest: str,
    rationale: str,
) -> str:
    payload = json.dumps(
        {
            "export_id": envelope.get("export_id"),
            "section": section,
            "imported_digest_v1": imported_digest,
            "local_digest_v1": local_digest,
            "rationale": rationale.strip(),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _attempt_request_key(request_fingerprint: str, attempt: int) -> str:
    if attempt == 1:
        return request_fingerprint
    return hashlib.sha256(
        f"{request_fingerprint}:{attempt}".encode("ascii")
    ).hexdigest()


def _operator_signature_payload(
    *,
    envelope: Mapping[str, Any],
    section: str,
    imported_digest: str,
    local_digest: str,
    rationale: str,
) -> bytes:
    return json.dumps(
        {
            "source": _audit_source(envelope),
            "replacement_scope": {
                "section": section,
                "mode": "whole_section",
                "selector": None,
            },
            "imported_digest_v1": imported_digest,
            "local_digest_v1": local_digest,
            "rationale": rationale,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def operator_override_signing_payload(
    *,
    envelope: Mapping[str, Any],
    operations: Sequence[Mapping[str, Any]],
    acknowledgement: str,
    reason_code: str,
    evidence_ref: str,
    rationale: str,
    operator_identity: str | None = None,
) -> bytes:
    """Return the canonical bytes an operator must sign for one override bundle."""

    normalized_operations = sorted(
        (
            {
                "section": str(operation.get("section") or ""),
                "local_digest_v1": str(operation.get("local_digest_v1") or ""),
                "imported_digest_v1": str(operation.get("imported_digest_v1") or ""),
            }
            for operation in operations
        ),
        key=lambda operation: operation["section"],
    )
    if not normalized_operations or any(
        operation["section"] not in PROTECTED_SECTIONS
        or not re.fullmatch(r"[0-9a-f]{64}", operation["local_digest_v1"])
        or not re.fullmatch(r"[0-9a-f]{64}", operation["imported_digest_v1"])
        for operation in normalized_operations
    ):
        raise ProtectedReplacementError(
            "override_signing_scope_invalid",
            "override signing requires protected sections with valid local and imported digests",
        )
    return json.dumps(
        {
            "action": "hmx_operator_override_v1",
            "source": _audit_source(envelope),
            "replacement_scope": normalized_operations,
            "acknowledgement": acknowledgement,
            "override_reason_code": reason_code,
            "override_evidence_ref": evidence_ref,
            "rationale": rationale.strip(),
            "operator_identity": (
                operator_identity.strip() if operator_identity else None
            ),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def operator_override_signing_material(
    *,
    envelope: Mapping[str, Any],
    operations: Sequence[Mapping[str, Any]],
    acknowledgement: str,
    reason_code: str,
    evidence_ref: str,
    rationale: str,
    operator_identity: str | None = None,
) -> dict[str, Any]:
    """Describe the exact payload used by external Ed25519 signing tools."""

    payload = operator_override_signing_payload(
        envelope=envelope,
        operations=operations,
        acknowledgement=acknowledgement,
        reason_code=reason_code,
        evidence_ref=evidence_ref,
        rationale=rationale,
        operator_identity=operator_identity,
    )
    return {
        "algorithm": "ed25519",
        "payload_encoding": "base64",
        "payload_base64": base64.b64encode(payload).decode("ascii"),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }


def validate_operator_override_fields(
    *,
    acknowledgement: str | None,
    reason_code: str | None,
    evidence_ref: str | None,
    rationale: str,
) -> tuple[str, str]:
    if acknowledgement != OPERATOR_OVERRIDE_ACKNOWLEDGEMENT:
        raise ProtectedReplacementError(
            "override_acknowledgement_mismatch",
            "operator override requires the exact responsibility acknowledgement phrase",
        )
    normalized_reason = str(reason_code or "").strip()
    if normalized_reason not in OPERATOR_OVERRIDE_REASON_CODES:
        raise ProtectedReplacementError(
            "invalid_override_reason_code",
            "override_reason_code must be one of "
            + ", ".join(OPERATOR_OVERRIDE_REASON_CODES),
        )
    normalized_evidence = str(evidence_ref or "").strip()
    if not _EVIDENCE_REF_PATTERN.fullmatch(normalized_evidence):
        raise ProtectedReplacementError(
            "invalid_override_evidence_ref",
            "override_evidence_ref must be an independently recorded scheme:value reference",
        )
    if not rationale.strip():
        raise ProtectedReplacementError(
            "replacement_rationale_missing",
            "operator override requires a free-text replacement rationale",
        )
    return normalized_reason, normalized_evidence


async def validate_operator_override_reason_state(conn, reason_code: str) -> None:
    paused = bool(
        await conn.fetchval(
            "SELECT COALESCE(is_paused, FALSE) FROM heartbeat_state WHERE id=1"
        )
    )
    terminated = bool(await conn.fetchval("SELECT is_agent_terminated()"))
    if reason_code == "agent_paused" and not paused:
        raise ProtectedReplacementError(
            "override_reason_not_observed",
            "agent_paused requires the live heartbeat state to be paused",
        )
    if reason_code == "agent_terminated" and not terminated:
        raise ProtectedReplacementError(
            "override_reason_not_observed",
            "agent_terminated requires the live agent state to be terminated",
        )
    if reason_code == "agent_unresponsive" and (paused or terminated):
        observed = "terminated" if terminated else "paused"
        raise ProtectedReplacementError(
            "override_reason_not_observed",
            f"agent_unresponsive requires a running, unpaused agent; live state is {observed}",
        )


async def _authorize_operator_override(
    conn,
    *,
    envelope: Mapping[str, Any],
    operations: Sequence[Mapping[str, Any]],
    acknowledgement: str | None,
    reason_code: str | None,
    evidence_ref: str | None,
    rationale: str,
    operator_signature: str | None,
    operator_identity: str | None,
    verifier: TrustAnchorVerifier | None,
) -> _OperatorOverrideAuthorization:
    normalized_reason, normalized_evidence = validate_operator_override_fields(
        acknowledgement=acknowledgement,
        reason_code=reason_code,
        evidence_ref=evidence_ref,
        rationale=rationale,
    )
    if not operator_signature:
        raise ProtectedReplacementError(
            "unverified_signature",
            "operator override requires --operator-signature verified against a configured trust anchor",
        )
    await validate_operator_override_reason_state(conn, normalized_reason)
    payload = operator_override_signing_payload(
        envelope=envelope,
        operations=operations,
        acknowledgement=acknowledgement or "",
        reason_code=normalized_reason,
        evidence_ref=normalized_evidence,
        rationale=rationale,
        operator_identity=operator_identity,
    )
    active_verifier = verifier or UnconfiguredTrustAnchors()
    verification = active_verifier.verify_operator_signature(
        signature=operator_signature,
        payload=payload,
        operator_identity=operator_identity,
    )
    if not verification.verified:
        raise ProtectedReplacementError(
            "unverified_signature",
            "operator override signature was treated as absent: " + verification.reason,
        )
    return _OperatorOverrideAuthorization(
        acknowledgement=acknowledgement or "",
        reason_code=normalized_reason,
        evidence_ref=normalized_evidence,
        signature=operator_signature,
        operator_identity=operator_identity.strip() if operator_identity else None,
        verification=verification,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
    )


async def _enqueue_replacement(
    conn,
    *,
    envelope: dict[str, Any],
    section: str,
    imported_section: Any,
    imported_digest: str,
    local_digest: str,
    rationale: str,
    trust_payload: dict[str, Any],
    operator_signature: str | None,
    operator_identity: str | None,
) -> tuple[str, str, str, bool]:
    request_fingerprint = _request_key(
        envelope=envelope,
        section=section,
        imported_digest=imported_digest,
        local_digest=local_digest,
        rationale=rationale,
    )
    scope = {"section": section, "mode": "whole_section", "selector": None}
    source = _audit_source(envelope)

    async with conn.transaction():
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtext('hmx_protected_replacement'))"
        )
        existing = await conn.fetchrow(
            "SELECT replacement_id, consent_id, status, request_attempt "
            "FROM hmx_pending_replacements WHERE request_fingerprint=$1 "
            "ORDER BY request_attempt DESC LIMIT 1",
            request_fingerprint,
        )
        if existing and existing["status"] != "timed_out":
            return (
                str(existing["replacement_id"]),
                str(existing["consent_id"]),
                str(existing["status"]),
                True,
            )

        request_attempt = int(existing["request_attempt"]) + 1 if existing else 1
        request_key = _attempt_request_key(request_fingerprint, request_attempt)

        heartbeat_count = int(
            await conn.fetchval(
                "SELECT COALESCE(heartbeat_count, 0) FROM heartbeat_state WHERE id=1"
            )
            or 0
        )
        consent_id = await conn.fetchval(
            "INSERT INTO hmx_consent "
            "(sections, source, replacement_scope, rationale, operator_signature, "
            "operator_identity, agent_acknowledgement_required, trust_verification) "
            "VALUES ($1::text[], $2::jsonb, $3::jsonb, $4, $5, $6, TRUE, $7::jsonb) "
            "RETURNING consent_id",
            [section],
            json.dumps(source),
            json.dumps(scope),
            rationale.strip(),
            operator_signature,
            operator_identity,
            json.dumps(trust_payload),
        )
        replacement_id = await conn.fetchval(
            "INSERT INTO hmx_pending_replacements "
            "(request_key, request_fingerprint, request_attempt, consent_id, "
            "export_id, section, source, imported_section, "
            "imported_digest_v1, local_digest_v1, replacement_scope, rationale, "
            "created_heartbeat_count, timeout_heartbeat_count) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9, $10, "
            "$11::jsonb, $12, $13, $14) RETURNING replacement_id",
            request_key,
            request_fingerprint,
            request_attempt,
            consent_id,
            str(envelope["export_id"]),
            section,
            json.dumps(source),
            json.dumps(imported_section),
            imported_digest,
            local_digest,
            json.dumps(scope),
            rationale.strip(),
            heartbeat_count,
            heartbeat_count + 10,
        )
    return str(replacement_id), str(consent_id), "pending", False


async def evaluate_protected_replacement(
    conn,
    envelope: dict[str, Any],
    *,
    section: str,
    rationale: str,
    verifier: TrustAnchorVerifier | None = None,
    allow_locally_trusted_lineage: bool = False,
    scope_mode: str = "whole_section",
    operator_signature: str | None = None,
    operator_identity: str | None = None,
) -> ProtectedReplacementResult:
    """Run Phase 0 or create an acknowledgement-gated pending replacement.

    This function never mutates protected state. A successful Phase 0 result
    means only that the requested replacement was content-identical and its
    immutable verification audit was durably written.
    """

    if not isinstance(envelope, dict):
        raise HmxSchemaError("HMX input must be a JSON object")
    validate_hmx_document(envelope)
    imported_section, imported_digest = _validate_request(envelope, section, scope_mode)
    rationale = rationale.strip()
    if not rationale:
        raise ProtectedReplacementError(
            "replacement_rationale_missing", "protected replacement requires rationale"
        )

    local_envelope = await export_hmx(
        conn,
        intent="port",
        include_in_flight_work=False,
        include_audit_records=False,
    )
    local_digest = str(local_envelope["section_digests"][section])
    local_context = await load_source_context(conn)
    source = envelope.get("source") or {}
    source_lineage = str(source.get("hexis_lineage_id") or "")
    local_lineage = str(local_context.get("lineage_id") or "")
    labels_equal = bool(source_lineage) and source_lineage == local_lineage

    active_verifier = verifier or UnconfiguredTrustAnchors()
    verification = active_verifier.verify_lineage(
        source=source,
        local_instance_id=str(local_context.get("instance_id") or ""),
        local_lineage_id=local_lineage,
    )
    locally_trusted = (
        labels_equal
        and allow_locally_trusted_lineage
        and verification.status is TrustStatus.UNVERIFIED
    )
    lineage_matches = labels_equal and (verification.verified or locally_trusted)
    trust_payload = _trust_payload(verification, locally_trusted_label=locally_trusted)

    if local_digest == imported_digest and lineage_matches:
        try:
            async with conn.transaction():
                audit_id = await _write_verified_audit(
                    conn,
                    envelope=envelope,
                    section=section,
                    local_digest=local_digest,
                    imported_digest=imported_digest,
                    trust_payload=trust_payload,
                )
        except Exception as exc:
            raise ProtectedReplacementError(
                "verified_audit_write_failure",
                "content-identical state was not reported as verified because the "
                f"required audit write failed: {exc}",
            ) from exc
        return ProtectedReplacementResult(
            disposition="verified_noop",
            section=section,
            local_digest_v1=local_digest,
            imported_digest_v1=imported_digest,
            lineage_verification=trust_payload,
            agent_acknowledgement_required=False,
            audit_ids=(audit_id,),
            status="verified",
        )

    conflicts: list[dict[str, Any]] = []
    if local_digest != imported_digest:
        conflicts.append(
            {
                "code": "protected_section_digest_mismatch",
                "section": section,
                "local_digest_v1": local_digest,
                "imported_digest_v1": imported_digest,
            }
        )
    if not lineage_matches:
        code = (
            "lineage_integrity_failure"
            if labels_equal and verification.status is TrustStatus.INVALID
            else "unverified_lineage" if labels_equal else "lineage_mismatch"
        )
        conflicts.append(
            {
                "code": code,
                "section": section,
                "source_lineage": source_lineage,
                "local_lineage": local_lineage,
                "verification": trust_payload,
            }
        )

    effective_operator_signature = operator_signature
    if operator_signature:
        signature_verification = active_verifier.verify_operator_signature(
            signature=operator_signature,
            payload=_operator_signature_payload(
                envelope=envelope,
                section=section,
                imported_digest=imported_digest,
                local_digest=local_digest,
                rationale=rationale,
            ),
            operator_identity=operator_identity,
        )
        trust_payload = dict(trust_payload)
        trust_payload["operator_signature_verification"] = _trust_payload(
            signature_verification, locally_trusted_label=False
        )
        if not signature_verification.verified:
            effective_operator_signature = None
            conflicts.append(
                {
                    "code": "unverified_signature",
                    "section": section,
                    "verification": trust_payload["operator_signature_verification"],
                }
            )

    replacement_id, consent_id, status, reused = await _enqueue_replacement(
        conn,
        envelope=envelope,
        section=section,
        imported_section=imported_section,
        imported_digest=imported_digest,
        local_digest=local_digest,
        rationale=rationale,
        trust_payload=trust_payload,
        operator_signature=effective_operator_signature,
        operator_identity=operator_identity,
    )
    disposition = {
        "pending": "pending_acknowledgement",
        "deferred": "pending_acknowledgement",
        "accepted": "accepted_awaiting_execution",
        "refused": "refused",
        "modification_requested": "modification_requested",
        "executed": "executed",
        "reverted": "reverted",
        "cancelled": "cancelled",
    }.get(status, status)
    return ProtectedReplacementResult(
        disposition=disposition,
        section=section,
        local_digest_v1=local_digest,
        imported_digest_v1=imported_digest,
        lineage_verification=trust_payload,
        agent_acknowledgement_required=status in {"pending", "deferred"},
        replacement_id=replacement_id,
        consent_id=consent_id,
        status=status,
        reused=reused,
        conflicts=tuple(conflicts),
    )


async def pending_protected_replacements(conn) -> dict[str, Any]:
    result = _coerce_json(await conn.fetchval("SELECT hmx_pending_replacements()"))
    if not isinstance(result, dict):
        raise RuntimeError("hmx_pending_replacements returned a non-object result")
    return result


async def list_protected_replacement_audit(
    conn,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """List immutable local replacement, verification, and reversion history."""

    if since and until and since > until:
        raise ProtectedReplacementError(
            "invalid_audit_time_range",
            "since must be earlier than or equal to until",
        )
    if limit < 1 or limit > 500:
        raise ProtectedReplacementError(
            "invalid_audit_limit", "limit must be between 1 and 500"
        )
    total = int(
        await conn.fetchval(
            "SELECT count(*) FROM protected_replacement_audit "
            "WHERE NOT is_foreign_diagnostic "
            "AND ($1::timestamptz IS NULL OR event_time >= $1) "
            "AND ($2::timestamptz IS NULL OR event_time <= $2)",
            since,
            until,
        )
        or 0
    )
    rows = await conn.fetch(
        "SELECT record FROM protected_replacement_audit "
        "WHERE NOT is_foreign_diagnostic "
        "AND ($1::timestamptz IS NULL OR event_time >= $1) "
        "AND ($2::timestamptz IS NULL OR event_time <= $2) "
        "ORDER BY event_time DESC, audit_id DESC LIMIT $3",
        since,
        until,
        limit,
    )
    records = [_coerce_json(row["record"]) for row in rows]
    return {
        "total": total,
        "returned": len(records),
        "truncated": total > len(records),
        "records": records,
    }


async def inspect_protected_replacement(conn, replacement_id: str) -> dict[str, Any]:
    row = await conn.fetchrow(
        "SELECT replacement_id, export_id, section, source, imported_section, "
        "imported_digest_v1, local_digest_v1, replacement_scope, rationale, "
        "status, acknowledgement, snapshot_id, execution_audit_id, "
        "reversion_audit_id, reverted_at, created_at, timeout_at "
        "FROM hmx_pending_replacements WHERE replacement_id=$1::uuid",
        replacement_id,
    )
    if row is None:
        raise ProtectedReplacementError(
            "protected_replacement_not_found",
            f"protected replacement not found: {replacement_id}",
        )
    section = str(row["section"])
    local = await export_hmx(
        conn,
        intent="port",
        include_in_flight_work=False,
        include_audit_records=False,
    )
    current_digest = str(local["section_digests"][section])
    reversion_window = None
    if row["snapshot_id"]:
        reversion_window = _coerce_json(
            await conn.fetchval(
                "SELECT hmx_snapshot_window($1::uuid)", str(row["snapshot_id"])
            )
        )
    status = str(row["status"])
    current_matches_executed_state = current_digest == str(row["imported_digest_v1"])
    return {
        "replacement_id": replacement_id,
        "export_id": str(row["export_id"]),
        "section": section,
        "source": _coerce_json(row["source"]),
        "replacement_scope": _coerce_json(row["replacement_scope"]),
        "rationale": str(row["rationale"]),
        "status": status,
        "acknowledgement": _coerce_json(row["acknowledgement"]),
        "snapshot_id": str(row["snapshot_id"]) if row["snapshot_id"] else None,
        "execution_audit_id": row["execution_audit_id"],
        "reversion_audit_id": row["reversion_audit_id"],
        "reverted_at": _iso_value(row["reverted_at"]),
        "reversion_window": reversion_window,
        "created_at": _iso_value(row["created_at"]),
        "timeout_at": _iso_value(row["timeout_at"]),
        "local_digest_v1_at_request": str(row["local_digest_v1"]),
        "current_local_digest_v1": current_digest,
        "imported_digest_v1": str(row["imported_digest_v1"]),
        "local_state_changed_since_request": current_digest
        != str(row["local_digest_v1"]),
        "current_matches_executed_state": current_matches_executed_state,
        "reversion_state_eligible": status == "executed"
        and current_matches_executed_state
        and isinstance(reversion_window, dict)
        and bool(reversion_window.get("window_open")),
        "current_local_section": local["sections"][section],
        "imported_section": _coerce_json(row["imported_section"]),
    }


def _iso_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return str(value)


async def execute_protected_replacement(
    conn,
    replacement_id: str,
    *,
    executor: str = "agent_tool",
    _operator_override: _OperatorOverrideAuthorization | None = None,
) -> dict[str, Any]:
    """Execute one accepted replacement with snapshot/audit/write atomicity."""

    if _operator_override is not None:
        executor = "operator_override"
    if executor not in {"user", "cli", "agent_tool", "system", "operator_override"}:
        raise ProtectedReplacementError(
            "invalid_replacement_executor",
            "executor must be user, cli, agent_tool, system, or operator_override",
        )
    if executor == "operator_override" and _operator_override is None:
        raise ProtectedReplacementError(
            "unverified_signature",
            "operator_override execution requires a verified override authorization",
        )

    async with conn.transaction():
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtext('hmx_protected_replacement'))"
        )
        row = await conn.fetchrow(
            "SELECT p.*, c.consent_kind, c.consent_at, c.operator_signature, "
            "c.operator_identity FROM hmx_pending_replacements p "
            "JOIN hmx_consent c ON c.consent_id=p.consent_id "
            "WHERE p.replacement_id=$1::uuid FOR UPDATE OF p",
            replacement_id,
        )
        if row is None:
            raise ProtectedReplacementError(
                "protected_replacement_not_found",
                f"protected replacement not found: {replacement_id}",
            )
        if row["status"] == "executed":
            if _operator_override is not None:
                raise ProtectedReplacementError(
                    "operator_override_not_available",
                    f"replacement {replacement_id} is already executed; no override was applied",
                )
            return {
                "replacement_id": replacement_id,
                "section": str(row["section"]),
                "status": "executed",
                "snapshot_id": (
                    str(row["snapshot_id"]) if row["snapshot_id"] else None
                ),
                "audit_id": row["execution_audit_id"],
                "reused": True,
                "warnings": [],
            }
        override_statuses = {"pending", "deferred"}
        expected_statuses = override_statuses if _operator_override else {"accepted"}
        if row["status"] not in expected_statuses:
            if _operator_override and row["status"] in {
                "refused",
                "modification_requested",
            }:
                raise ProtectedReplacementError(
                    "agent_decision_cannot_be_bypassed",
                    f"replacement {replacement_id} is {row['status']}; operator override cannot bypass an agent decision",
                )
            raise ProtectedReplacementError(
                (
                    "operator_override_not_available"
                    if _operator_override
                    else "protected_replacement_not_accepted"
                ),
                (
                    f"replacement {replacement_id} is {row['status']}; override applies only while acknowledgement is pending"
                    if _operator_override
                    else f"replacement {replacement_id} is {row['status']}; agent acceptance is required before execution"
                ),
            )

        section = str(row["section"])
        before = await export_hmx(
            conn,
            intent="port",
            include_in_flight_work=False,
            include_audit_records=False,
        )
        current_digest = str(before["section_digests"][section])
        expected_digest = str(row["local_digest_v1"])
        if current_digest != expected_digest:
            raise ProtectedReplacementError(
                "protected_state_changed_while_pending",
                f"{section} changed after this request was created; refuse this request and submit a fresh authoritative import",
            )

        local_context = await load_source_context(conn)
        source = _coerce_json(row["source"]) or {}
        import_source = {
            "instance_id": source.get("origin_instance"),
            "hexis_lineage_id": source.get("hexis_lineage_id"),
        }
        imported_at = _iso_now()
        prepared_section, preparation_warnings = prepare_protected_section_import(
            section,
            _coerce_json(row["imported_section"]),
            intent=str(source.get("export_intent") or "port"),
            source=import_source,
            export_id=str(row["export_id"]),
            local_instance_id=str(local_context["instance_id"]),
            imported_at=imported_at,
        )
        snapshot_id = await create_protected_snapshot(conn, [section])
        window = _coerce_json(
            await conn.fetchval("SELECT hmx_snapshot_window($1::uuid)", snapshot_id)
        )
        if not isinstance(window, dict) or not window.get("window_open"):
            raise ProtectedReplacementError(
                "snapshot_window_unavailable",
                "the protected-state snapshot was created without an open rollback window",
            )

        audit_id = str(uuid.uuid4())
        event_time = _iso_now()
        operator_identity = row["operator_identity"]
        operator_signature = row["operator_signature"]
        acknowledgement = "accepted"
        acknowledgement_at = _iso_value(row["acknowledgement_at"])
        override_reason_code = None
        override_evidence_ref = None
        override_verification = None
        if _operator_override is not None:
            acknowledgement = "bypassed"
            acknowledgement_at = event_time
            operator_identity = (
                _operator_override.operator_identity
                or _operator_override.verification.anchor_id
            )
            operator_signature = _operator_override.signature
            override_reason_code = _operator_override.reason_code
            override_evidence_ref = _operator_override.evidence_ref
            override_verification = _trust_payload(
                _operator_override.verification, locally_trusted_label=False
            )
        audit_record = {
            "audit_id": audit_id,
            "event_type": "protected_section_replacement",
            "event_time": event_time,
            "consent": {
                "consent_id": str(row["consent_id"]),
                "consent_kind": str(row["consent_kind"]),
                "consent_at": _iso_value(row["consent_at"]),
                "rationale": str(row["rationale"]),
                "operator_signature": operator_signature,
                "operator_identity": operator_identity,
            },
            "replacement_scope": _coerce_json(row["replacement_scope"]),
            "sections_replaced": [section],
            "source": source,
            "previous_state_snapshot_ref": snapshot_id,
            "previous_state_digest_v1": current_digest,
            "new_state_digest_v1": str(row["imported_digest_v1"]),
            "agent_acknowledgement": acknowledgement,
            "agent_acknowledgement_at": acknowledgement_at,
            "replacement_executor": executor,
            "override_reason_code": override_reason_code,
            "override_evidence_ref": override_evidence_ref,
            "reversibility_window": window,
        }
        if _operator_override is not None:
            audit_record["operator_override"] = {
                "acknowledgement": _operator_override.acknowledgement,
                "signature_verification": override_verification,
                "signing_payload_sha256": _operator_override.payload_sha256,
            }
        stored = await store_audit_record(conn, audit_record)
        if stored.get("status") != "inserted":
            raise ProtectedReplacementError(
                "replacement_audit_write_failure",
                f"required replacement audit returned {stored.get('status')}",
            )

        reference_map = _coerce_json(row["reference_map"]) or {}
        sql_result = _coerce_json(
            await conn.fetchval(
                "SELECT hmx_import_authoritative($1::jsonb, $2::text[], $3::jsonb)",
                json.dumps({section: prepared_section}),
                [section],
                json.dumps(reference_map),
            )
        )
        if not isinstance(sql_result, dict):
            raise RuntimeError("hmx_import_authoritative returned a non-object result")

        after = await export_hmx(
            conn,
            intent="port",
            include_in_flight_work=False,
            include_audit_records=True,
        )
        resulting_digest = str(after["section_digests"][section])
        imported_digest = str(row["imported_digest_v1"])
        if resulting_digest != imported_digest:
            raise ProtectedReplacementError(
                "replacement_digest_verification_failure",
                f"authoritative {section} write produced {resulting_digest}, expected {imported_digest}; all changes were rolled back",
            )
        collateral_changes = sorted(
            protected_section
            for protected_section, previous_digest in before["section_digests"].items()
            if protected_section != section
            and after["section_digests"].get(protected_section) != previous_digest
        )
        if collateral_changes:
            raise ProtectedReplacementError(
                "replacement_scope_violation",
                "whole-section replacement also changed protected section(s) "
                + ", ".join(collateral_changes)
                + "; all changes were rolled back",
            )

        merged_ref_map = dict(reference_map)
        merged_ref_map.update(_coerce_json(sql_result.get("ref_map")) or {})
        await conn.execute(
            "UPDATE hmx_pending_replacements SET status='executed', "
            "acknowledgement=COALESCE($5::jsonb, acknowledgement), "
            "acknowledgement_at=CASE WHEN $5::jsonb IS NULL THEN acknowledgement_at "
            "ELSE CURRENT_TIMESTAMP END, "
            "reference_map=$2::jsonb, snapshot_id=$3::uuid, execution_audit_id=$4, "
            "executed_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP "
            "WHERE replacement_id=$1::uuid",
            replacement_id,
            json.dumps(merged_ref_map),
            snapshot_id,
            audit_id,
            (
                json.dumps(
                    {
                        "decision": "bypassed",
                        "reason_code": _operator_override.reason_code,
                        "evidence_ref": _operator_override.evidence_ref,
                        "operator_identity": operator_identity,
                        "trust_anchor_id": _operator_override.verification.anchor_id,
                        "signing_payload_sha256": _operator_override.payload_sha256,
                    }
                )
                if _operator_override
                else None
            ),
        )

        warnings = list(preparation_warnings)
        warnings.extend(sql_result.get("warnings") or [])
        return {
            "replacement_id": replacement_id,
            "section": section,
            "status": "executed",
            "snapshot_id": snapshot_id,
            "audit_id": audit_id,
            "resulting_digest_v1": resulting_digest,
            "reversibility_window": window,
            "replacement_executor": executor,
            "reused": False,
            "warnings": warnings,
        }


async def acknowledge_protected_replacement(
    conn,
    replacement_id: str,
    *,
    decision: str,
    rationale: str | None = None,
    proposed_changes: dict[str, Any] | None = None,
    executor: str = "agent_tool",
) -> dict[str, Any]:
    if decision not in ACKNOWLEDGEMENT_DECISIONS:
        raise ProtectedReplacementError(
            "invalid_acknowledgement",
            f"decision must be one of {', '.join(ACKNOWLEDGEMENT_DECISIONS)}",
        )
    async with conn.transaction():
        if decision == "accept":
            lineage_status = await conn.fetchval(
                "SELECT c.trust_verification->>'status' "
                "FROM hmx_pending_replacements p "
                "JOIN hmx_consent c ON c.consent_id=p.consent_id "
                "WHERE p.replacement_id=$1::uuid",
                replacement_id,
            )
            if lineage_status == "invalid":
                raise ProtectedReplacementError(
                    "lineage_integrity_failure_requires_operator_override",
                    "a configured trust anchor rejected the matching lineage claim; "
                    "normal agent acceptance cannot authorize this replacement",
                )
        result = await conn.fetchval(
            "SELECT hmx_acknowledge_protected_replacement($1::uuid, $2, $3, $4::jsonb)",
            replacement_id,
            decision,
            rationale,
            json.dumps(proposed_changes) if proposed_changes is not None else None,
        )
        parsed = _coerce_json(result)
        if not isinstance(parsed, dict):
            raise RuntimeError(
                "hmx_acknowledge_protected_replacement returned a non-object result"
            )
        if decision == "accept":
            execution = await execute_protected_replacement(
                conn,
                replacement_id,
                executor=executor,
            )
            parsed.update(execution)
        return parsed


async def open_protected_reversion_windows(conn) -> dict[str, Any]:
    result = _coerce_json(await conn.fetchval("SELECT hmx_open_reversion_windows()"))
    if not isinstance(result, dict):
        raise RuntimeError("hmx_open_reversion_windows returned a non-object result")
    return result


async def revert_protected_replacement(
    conn,
    audit_id: str,
    *,
    rationale: str,
    actor_identity: str = "agent_tool",
) -> dict[str, Any]:
    """Restore one replacement snapshot within its bounded reversion window."""

    normalized_rationale = str(rationale or "").strip()
    if not normalized_rationale:
        raise ProtectedReplacementError(
            "reversion_rationale_missing",
            "protected replacement reversion requires a rationale",
        )
    if actor_identity not in {"user", "cli", "agent_tool", "system"}:
        raise ProtectedReplacementError(
            "invalid_reversion_actor",
            "actor_identity must be user, cli, agent_tool, or system",
        )

    # Expiry cleanup must survive the expected reversion-window rejection.
    await conn.execute("SELECT hmx_purge_expired_protected_snapshots()")
    async with conn.transaction():
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtext('hmx_protected_replacement'))"
        )

        audit_row = await conn.fetchrow(
            "SELECT event_type, record, is_foreign_diagnostic "
            "FROM protected_replacement_audit WHERE audit_id=$1",
            audit_id,
        )
        if audit_row is None:
            raise ProtectedReplacementError(
                "replacement_audit_not_found",
                f"protected replacement audit not found: {audit_id}",
            )
        if audit_row["event_type"] != "protected_section_replacement":
            raise ProtectedReplacementError(
                "replacement_audit_not_revertible",
                f"audit {audit_id} records {audit_row['event_type']}, not a replacement",
            )
        if audit_row["is_foreign_diagnostic"]:
            raise ProtectedReplacementError(
                "foreign_replacement_not_revertible",
                "imported diagnostic audit history has no local snapshot to restore",
            )

        pending = await conn.fetchrow(
            "SELECT replacement_id, section, status, snapshot_id, "
            "execution_audit_id, reversion_audit_id, reverted_at "
            "FROM hmx_pending_replacements WHERE execution_audit_id=$1 FOR UPDATE",
            audit_id,
        )
        if pending is None or pending["snapshot_id"] is None:
            raise ProtectedReplacementError(
                "reversion_snapshot_unavailable",
                "this replacement audit has no local snapshot; imported audit history cannot be executed",
            )
        if pending["reversion_audit_id"]:
            return {
                "status": "already_reverted",
                "replacement_id": str(pending["replacement_id"]),
                "reverts_audit_id": audit_id,
                "audit_id": str(pending["reversion_audit_id"]),
                "snapshot_id": str(pending["snapshot_id"]),
                "reverted_at": _iso_value(pending["reverted_at"]),
                "reused": True,
                "warnings": [],
            }
        if pending["status"] != "executed":
            raise ProtectedReplacementError(
                "replacement_not_revertible",
                f"replacement {pending['replacement_id']} is {pending['status']}, not executed",
            )

        snapshot = await conn.fetchrow(
            "SELECT sections, snapshot_state, section_digests, reference_map, "
            "consumed_at, consumed_by_audit_id, purged_at, purge_reason "
            "FROM protected_replacement_snapshots WHERE snapshot_id=$1::uuid "
            "FOR UPDATE",
            str(pending["snapshot_id"]),
        )
        if snapshot is None:
            raise ProtectedReplacementError(
                "reversion_snapshot_unavailable",
                "the local replacement snapshot no longer exists",
            )
        window = _coerce_json(
            await conn.fetchval(
                "SELECT hmx_snapshot_window($1::uuid)", str(pending["snapshot_id"])
            )
        )
        if not isinstance(window, dict) or not window.get("window_open"):
            reason = snapshot["purge_reason"] or "reversion_window_closed"
            raise ProtectedReplacementError(
                "reversion_window_closed",
                f"the replacement can no longer be reverted ({reason}); submit a new authoritative request if a state change is still needed",
            )

        sections = [str(section) for section in snapshot["sections"]]
        section = str(pending["section"])
        if sections != [section]:
            raise ProtectedReplacementError(
                "reversion_snapshot_scope_invalid",
                "the replacement snapshot does not match its single-section audit scope",
            )
        snapshot_state = _coerce_json(snapshot["snapshot_state"])
        snapshot_digests = _coerce_json(snapshot["section_digests"])
        if not isinstance(snapshot_state, dict) or not isinstance(
            snapshot_digests, dict
        ):
            raise ProtectedReplacementError(
                "reversion_snapshot_integrity_failure",
                "the replacement snapshot payload or digest map is invalid",
            )
        restored_section = snapshot_state.get(section)
        restored_digest = str(snapshot_digests.get(section) or "")
        if (
            not restored_digest
            or protected_section_digest_v1(section, restored_section) != restored_digest
        ):
            raise ProtectedReplacementError(
                "reversion_snapshot_integrity_failure",
                "the stored snapshot content does not match its protected digest",
            )

        replacement_record = _coerce_json(audit_row["record"])
        if not isinstance(replacement_record, dict):
            raise ProtectedReplacementError(
                "replacement_audit_integrity_failure",
                "the replacement audit record is not a JSON object",
            )
        if replacement_record.get("previous_state_digest_v1") != restored_digest:
            raise ProtectedReplacementError(
                "reversion_snapshot_integrity_failure",
                "the snapshot digest does not match the replacement audit's previous state",
            )
        expected_current_digest = str(
            replacement_record.get("new_state_digest_v1") or ""
        )
        if not expected_current_digest:
            raise ProtectedReplacementError(
                "replacement_audit_integrity_failure",
                "the replacement audit does not declare its new protected-state digest",
            )

        before = await export_hmx(
            conn,
            intent="port",
            include_in_flight_work=False,
            include_audit_records=False,
        )
        current_digest = str(before["section_digests"][section])
        if current_digest != expected_current_digest:
            raise ProtectedReplacementError(
                "protected_state_changed_since_replacement",
                f"{section} changed after replacement {audit_id}; reversion will not overwrite newer state. Inspect the current section and submit a new authoritative request if needed",
            )

        reversion_audit_id = str(uuid.uuid4())
        reversion_record = {
            "audit_id": reversion_audit_id,
            "event_type": "protected_section_reverted",
            "event_time": _iso_now(),
            "reverts_audit_id": audit_id,
            "rationale": normalized_rationale,
            "sections_reverted": [section],
            "restored_state_digest_v1": restored_digest,
            "post_reversion_digest_v1": restored_digest,
            "pre_reversion_digest_v1": current_digest,
            "replacement_snapshot_ref": str(pending["snapshot_id"]),
            "actor_identity": actor_identity,
            "agent_initiated": actor_identity == "agent_tool",
        }
        stored = await store_audit_record(conn, reversion_record)
        if stored.get("status") != "inserted":
            raise ProtectedReplacementError(
                "reversion_audit_write_failure",
                f"required reversion audit returned {stored.get('status')}",
            )

        sql_result = _coerce_json(
            await conn.fetchval(
                "SELECT hmx_import_authoritative($1::jsonb, $2::text[], $3::jsonb)",
                json.dumps(
                    {
                        section: prepare_protected_section_restore(
                            section, restored_section
                        )
                    }
                ),
                sections,
                json.dumps(_coerce_json(snapshot["reference_map"]) or {}),
            )
        )
        if not isinstance(sql_result, dict):
            raise RuntimeError("hmx_import_authoritative returned a non-object result")

        after = await export_hmx(
            conn,
            intent="port",
            include_in_flight_work=False,
            include_audit_records=True,
        )
        resulting_digest = str(after["section_digests"][section])
        if resulting_digest != restored_digest:
            raise ProtectedReplacementError(
                "reversion_digest_verification_failure",
                f"reverting {section} produced {resulting_digest}, expected {restored_digest}; all changes were rolled back",
            )
        collateral_changes = sorted(
            protected_section
            for protected_section, previous_digest in before["section_digests"].items()
            if protected_section != section
            and after["section_digests"].get(protected_section) != previous_digest
        )
        if collateral_changes:
            raise ProtectedReplacementError(
                "reversion_scope_violation",
                "reversion also changed protected section(s) "
                + ", ".join(collateral_changes)
                + "; all changes were rolled back",
            )

        await conn.execute(
            "UPDATE protected_replacement_snapshots SET snapshot_state=NULL, "
            "consumed_at=CURRENT_TIMESTAMP, consumed_by_audit_id=$2, "
            "purged_at=CURRENT_TIMESTAMP, purge_reason='consumed_by_reversion' "
            "WHERE snapshot_id=$1::uuid",
            str(pending["snapshot_id"]),
            reversion_audit_id,
        )
        await conn.execute(
            "UPDATE hmx_pending_replacements SET status='reverted', "
            "reversion_audit_id=$2, reverted_at=CURRENT_TIMESTAMP, "
            "updated_at=CURRENT_TIMESTAMP WHERE replacement_id=$1::uuid",
            str(pending["replacement_id"]),
            reversion_audit_id,
        )

        return {
            "status": "reverted",
            "replacement_id": str(pending["replacement_id"]),
            "reverts_audit_id": audit_id,
            "audit_id": reversion_audit_id,
            "snapshot_id": str(pending["snapshot_id"]),
            "section": section,
            "resulting_digest_v1": resulting_digest,
            "reused": False,
            "warnings": list(sql_result.get("warnings") or []),
        }


async def import_authoritative_hmx(
    conn,
    envelope: dict[str, Any],
    *,
    replace_sections: tuple[str, ...],
    rationale: str | None,
    verifier: TrustAnchorVerifier | None = None,
    allow_locally_trusted_lineage: bool = False,
    retry_failed_work: bool = False,
    operator_signature: str | None = None,
    operator_identity: str | None = None,
    force_replace: bool = False,
    override_acknowledgement: str | None = None,
    override_reason_code: str | None = None,
    override_evidence_ref: str | None = None,
) -> HmxAuthoritativeResult:
    """Import ordinary data and submit explicit protected-section operations."""

    intent = str(envelope.get("export_intent") or "")
    if intent not in {"port", "duplicate"}:
        raise ProtectedReplacementError(
            "invalid_authoritative_intent",
            "authoritative replacement requires export_intent port or duplicate",
        )
    if not replace_sections:
        raise ProtectedReplacementError(
            "replacement_sections_missing",
            "authoritative import requires at least one explicit protected section",
        )
    normalized_rationale = str(rationale or "").strip()
    if not normalized_rationale:
        raise ProtectedReplacementError(
            "replacement_rationale_missing",
            "authoritative import requires replacement_rationale (CLI: --replacement-rationale)",
        )
    override_fields_present = any(
        value
        for value in (
            override_acknowledgement,
            override_reason_code,
            override_evidence_ref,
        )
    )
    if override_fields_present and not force_replace:
        raise ProtectedReplacementError(
            "operator_override_not_requested",
            "operator override arguments require force_replace (CLI: --force-replace)",
        )

    forecast = await dry_run_hmx(
        conn,
        envelope,
        strategy="authoritative",
        retry_failed_work=retry_failed_work,
        replace_sections=replace_sections,
        allow_locally_trusted_lineage=allow_locally_trusted_lineage,
    )
    if not forecast.can_import:
        blocking_codes = ", ".join(
            sorted({str(item.get("code")) for item in forecast.conflicts})
        )
        raise ProtectedReplacementError(
            "authoritative_preflight_blocked",
            f"resolve these preflight conflicts before retrying: {blocking_codes}",
        )

    ordinary_document = copy.deepcopy(envelope)
    ordinary_sections = ordinary_document.get("sections") or {}
    for protected_section in PROTECTED_SECTIONS:
        ordinary_sections.pop(protected_section, None)

    operation_results: list[ProtectedReplacementResult] = []
    override_authorization: _OperatorOverrideAuthorization | None = None
    async with conn.transaction():
        if force_replace:
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext('hmx_protected_replacement'))"
            )
            current = await export_hmx(
                conn,
                intent="port",
                include_in_flight_work=False,
                include_audit_records=False,
            )
            signing_operations = [
                {
                    "section": section,
                    "local_digest_v1": str(current["section_digests"][section]),
                    "imported_digest_v1": str(envelope["section_digests"][section]),
                }
                for section in replace_sections
            ]
            override_authorization = await _authorize_operator_override(
                conn,
                envelope=envelope,
                operations=signing_operations,
                acknowledgement=override_acknowledgement,
                reason_code=override_reason_code,
                evidence_ref=override_evidence_ref,
                rationale=normalized_rationale,
                operator_signature=operator_signature,
                operator_identity=operator_identity,
                verifier=verifier,
            )

        for section in replace_sections:
            operation_results.append(
                await evaluate_protected_replacement(
                    conn,
                    envelope,
                    section=section,
                    rationale=normalized_rationale,
                    verifier=verifier,
                    allow_locally_trusted_lineage=allow_locally_trusted_lineage,
                    operator_signature=(
                        None if override_authorization else operator_signature
                    ),
                    operator_identity=operator_identity,
                )
            )

        ordinary_result = await import_hmx(
            conn,
            ordinary_document,
            strategy="additive",
            reviewed=True,
            retry_failed_work=retry_failed_work,
        )
        if not isinstance(ordinary_result, HmxImportResult):
            raise RuntimeError(
                "authoritative ordinary import returned wrong result type"
            )

        for operation in operation_results:
            if operation.replacement_id and operation.status in {"pending", "deferred"}:
                await conn.execute(
                    "UPDATE hmx_pending_replacements SET "
                    "reference_map=reference_map || $2::jsonb, "
                    "updated_at=CURRENT_TIMESTAMP WHERE replacement_id=$1::uuid",
                    operation.replacement_id,
                    json.dumps(ordinary_result.ref_map),
                )

        if override_authorization is not None:
            if all(
                operation.disposition == "verified_noop"
                for operation in operation_results
            ):
                raise ProtectedReplacementError(
                    "operator_override_not_required",
                    "all selected protected sections were verified no-ops; no acknowledgement was bypassed",
                )
            for index, operation in enumerate(operation_results):
                if operation.disposition == "verified_noop":
                    continue
                if not operation.replacement_id:
                    raise ProtectedReplacementError(
                        "operator_override_not_available",
                        f"{operation.section} did not produce an override-eligible replacement request",
                    )
                execution = await execute_protected_replacement(
                    conn,
                    operation.replacement_id,
                    _operator_override=override_authorization,
                )
                operation_results[index] = replace(
                    operation,
                    disposition="executed",
                    agent_acknowledgement_required=False,
                    audit_ids=(str(execution["audit_id"]),),
                    status="executed",
                    replacement_executor="operator_override",
                )

    operation_conflicts = [
        conflict for operation in operation_results for conflict in operation.conflicts
    ]
    warnings = list(forecast.warnings)
    warnings.extend(ordinary_result.warnings)
    if override_authorization is not None:
        logger.warning(
            "HMX operator override executed: sections=%s reason=%s evidence=%s anchor=%s",
            ",".join(operation.section for operation in operation_results),
            override_authorization.reason_code,
            override_authorization.evidence_ref,
            override_authorization.verification.anchor_id,
        )
        warnings.append(
            {
                "code": "operator_override_executed",
                "reason_code": override_authorization.reason_code,
                "evidence_ref": override_authorization.evidence_ref,
                "trust_anchor_id": override_authorization.verification.anchor_id,
                "error": "protected acknowledgement was bypassed by a verified operator override",
            }
        )
    return HmxAuthoritativeResult(
        export_id=str(envelope["export_id"]),
        intent=intent,
        strategy="authoritative",
        target_state=forecast.target_state,
        inserted=dict(ordinary_result.inserted),
        protected_operations=tuple(
            asdict(operation) for operation in operation_results
        ),
        ref_map=dict(ordinary_result.ref_map),
        conflicts=tuple(list(ordinary_result.conflicts) + operation_conflicts),
        warnings=tuple(warnings),
        work_summary=dict(ordinary_result.work_summary),
    )


def _snapshot_reference_map(envelope: Mapping[str, Any]) -> dict[str, str]:
    """Map refs from a fresh local export back to their current local IDs."""

    prefix = f"{envelope.get('export_id')}:"
    result: dict[str, str] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str) and value.startswith(prefix):
            local_id = value[len(prefix) :]
            if local_id:
                result[value] = local_id

    visit(envelope.get("sections") or {})
    return result


async def create_protected_snapshot(
    conn,
    sections: list[str],
    *,
    heartbeat_window: int = 7,
    wall_clock_expires_at: datetime | None = None,
) -> str:
    selected = sorted(set(sections))
    invalid = sorted(set(selected) - PROTECTED_SECTIONS)
    if not selected or invalid:
        raise ProtectedReplacementError(
            "invalid_snapshot_sections",
            "snapshot sections must be a non-empty protected-section list"
            + (f"; invalid: {', '.join(invalid)}" if invalid else ""),
        )
    local = await export_hmx(
        conn,
        intent="port",
        include_in_flight_work=False,
        include_audit_records=False,
    )
    state = {section: local["sections"][section] for section in selected}
    digests = {section: local["section_digests"][section] for section in selected}
    reference_map = _snapshot_reference_map(local)
    if wall_clock_expires_at is None:
        snapshot_id = await conn.fetchval(
            "SELECT hmx_create_protected_snapshot($1::text[], $2::jsonb, "
            "$3::jsonb, p_heartbeat_window => $4::integer, "
            "p_reference_map => $5::jsonb)",
            selected,
            json.dumps(state),
            json.dumps(digests),
            heartbeat_window,
            json.dumps(reference_map),
        )
    else:
        snapshot_id = await conn.fetchval(
            "SELECT hmx_create_protected_snapshot($1::text[], $2::jsonb, "
            "$3::jsonb, $4::integer, $5::timestamptz, $6::jsonb)",
            selected,
            json.dumps(state),
            json.dumps(digests),
            heartbeat_window,
            wall_clock_expires_at,
            json.dumps(reference_map),
        )
    return str(snapshot_id)


async def import_audit_records(
    conn,
    envelope: dict[str, Any],
) -> AuditImportResult:
    """Preserve protected audit history with digest-based conflict detection."""

    records, warnings = _validated_audit_records(envelope)
    intent = str(envelope.get("export_intent") or "")
    foreign = intent not in ("port", "duplicate")
    inserted = 0
    duplicates = 0
    conflicts: list[dict[str, Any]] = []
    imported_at = datetime.now(timezone.utc)

    for record in records:
        result = await store_audit_record(
            conn,
            record,
            is_foreign_diagnostic=foreign,
            imported_at=imported_at,
        )
        if result.get("status") == "inserted":
            inserted += 1
        elif result.get("status") == "duplicate":
            duplicates += 1
        else:
            conflicts.append(dict(result))

    if foreign and (inserted or duplicates):
        warnings.append(
            {
                "code": "foreign_audit_diagnostic",
                "section": "audit_records",
                "count": inserted + duplicates,
                "error": "non-port audit history was retained as a foreign diagnostic and will not be re-exported as local history",
            }
        )
    return AuditImportResult(
        inserted=inserted,
        duplicates=duplicates,
        conflicts=tuple(conflicts),
        warnings=tuple(warnings),
    )


def _validated_audit_records(
    envelope: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate protected audit records independently so one bad record is isolated."""

    audit_section = (envelope.get("sections") or {}).get("audit_records") or {}
    if not isinstance(audit_section, dict):
        return [], [
            {
                "code": "schema_validation_error",
                "section": "audit_records",
                "error": "audit_records must be an object",
            }
        ]
    valid: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for group, expected_type in _AUDIT_GROUPS.items():
        records = audit_section.get(group, [])
        if not isinstance(records, list):
            warnings.append(
                {
                    "code": "schema_validation_error",
                    "section": "audit_records",
                    "audit_group": group,
                    "error": "audit group must be an array",
                }
            )
            continue
        for index, record in enumerate(records):
            if (
                not isinstance(record, dict)
                or record.get("event_type") != expected_type
            ):
                warnings.append(
                    {
                        "code": "schema_validation_error",
                        "section": "audit_records",
                        "audit_group": group,
                        "index": index,
                        "error": f"record must have event_type {expected_type}",
                    }
                )
                continue
            probe = dict(envelope)
            probe["sections"] = {"audit_records": {group: [record]}}
            try:
                validate_hmx_document(probe)
            except HmxSchemaError as exc:
                warnings.append(
                    {
                        "code": "schema_validation_error",
                        "section": "audit_records",
                        "audit_group": group,
                        "index": index,
                        "error": str(exc),
                    }
                )
            else:
                valid.append(dict(record))

    unknown_groups = sorted(
        set(audit_section) - set(_AUDIT_GROUPS) - {"transformation_history"}
    )
    for group in unknown_groups:
        warnings.append(
            {
                "code": "unsupported_audit_event",
                "section": "audit_records",
                "audit_group": group,
                "error": "unknown audit group was skipped for operator review",
            }
        )
    return valid, warnings


async def forecast_audit_records(conn, envelope: dict[str, Any]) -> AuditImportResult:
    """Side-effect-free audit dedupe forecast for HMX dry-run."""

    records, warnings = _validated_audit_records(envelope)
    if not records:
        return AuditImportResult(warnings=tuple(warnings))
    ids = [str(record["audit_id"]) for record in records]
    rows = await conn.fetch(
        "SELECT audit_id, record_digest_v1 FROM protected_replacement_audit "
        "WHERE audit_id = ANY($1::text[])",
        ids,
    )
    existing = {str(row["audit_id"]): str(row["record_digest_v1"]) for row in rows}
    inserted = 0
    duplicates = 0
    conflicts: list[dict[str, Any]] = []
    for record in records:
        audit_id = str(record["audit_id"])
        digest = audit_record_digest_v1(record)
        if audit_id not in existing:
            inserted += 1
        elif existing[audit_id] == digest:
            duplicates += 1
        else:
            conflicts.append(
                {
                    "status": "conflict",
                    "code": "audit_integrity_conflict",
                    "audit_id": audit_id,
                    "existing_digest_v1": existing[audit_id],
                    "imported_digest_v1": digest,
                }
            )
    return AuditImportResult(
        inserted=inserted,
        duplicates=duplicates,
        conflicts=tuple(conflicts),
        warnings=tuple(warnings),
    )
