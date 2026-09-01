"""Agent-facing HMX tools, registry wiring, and skill-first journey."""

from __future__ import annotations

import json
import stat
import tomllib
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.digest import protected_section_digest_v1
from core.memory_exchange import export_hmx, import_hmx
from core.protected_replacement import store_audit_record
from core.tools import ToolContext, ToolErrorType, ToolExecutionContext
from core.tools.memory_exchange import create_memory_exchange_tools
from core.tools.protected_replacement import create_protected_replacement_tools
from core.trust_anchors import TrustVerification

pytestmark = [pytest.mark.asyncio(loop_scope="session")]

_TOOL_NAMES = {
    "export_memories",
    "import_memories",
    "import_dry_run",
    "import_review",
    "import_accept",
    "import_reject",
    "import_modify",
    "import_quote",
    "promote_to_staged",
    "demote_to_analysis",
    "protected_replacement_list",
    "protected_replacement_inspect",
    "protected_replacement_review",
    "protected_replacement_audit_list",
    "protected_reversion_list",
    "protected_replacement_revert",
}


async def test_bundled_skills_are_declared_as_wheel_package_data():
    project = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    assert (
        "installed/*/SKILL.md"
        in project["tool"]["setuptools"]["package-data"]["skills"]
    )


class _SingleConnectionPool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        @asynccontextmanager
        async def borrowed():
            yield self.conn

        return borrowed()


def _context(conn, workspace) -> ToolExecutionContext:
    return ToolExecutionContext(
        tool_context=ToolContext.CHAT,
        call_id=f"hmx-tools-{uuid.uuid4()}",
        workspace_path=str(workspace),
        allow_file_read=True,
        allow_file_write=True,
        registry=SimpleNamespace(pool=_SingleConnectionPool(conn)),
    )


class _VerifiedTrustAnchors:
    def verify_operator_signature(self, **kwargs):
        return TrustVerification.accepted(anchor_id="slice-13-operator")

    def verify_source_identity(self, **kwargs):
        return TrustVerification.accepted(anchor_id="slice-13-source")

    def verify_lineage(self, **kwargs):
        return TrustVerification.accepted(anchor_id="slice-13-lineage")


async def test_tool_specs_are_complete_and_conservative():
    handlers = {
        handler.spec.name: handler for handler in create_memory_exchange_tools()
    }

    assert set(handlers) == _TOOL_NAMES
    assert handlers["import_dry_run"].spec.is_read_only
    assert handlers["import_review"].spec.is_read_only
    for name in ("import_dry_run", "import_memories"):
        retry_schema = handlers[name].spec.parameters["properties"]["retry_failed_work"]
        assert retry_schema == {"type": "boolean", "default": False}
        strategy_schema = handlers[name].spec.parameters["properties"]["strategy"]
        assert "authoritative" in strategy_schema["enum"]
        assert "replace_sections" in handlers[name].spec.parameters["properties"]
    assert (
        "replacement_rationale"
        in handlers["import_memories"].spec.parameters["properties"]
    )
    for name in _TOOL_NAMES - {
        "import_dry_run",
        "import_review",
        "protected_replacement_inspect",
        "protected_replacement_list",
        "protected_replacement_audit_list",
        "protected_replacement_revert",
        "protected_reversion_list",
        "protected_replacement_review",
    }:
        assert handlers[name].spec.requires_approval
        assert not handlers[name].spec.is_read_only
        assert not handlers[name].spec.supports_parallel
    assert not handlers["protected_replacement_review"].spec.requires_approval
    assert not handlers["protected_replacement_revert"].spec.requires_approval
    assert not handlers["protected_replacement_review"].spec.is_read_only
    assert handlers["protected_replacement_inspect"].spec.is_read_only
    assert handlers["protected_replacement_list"].spec.is_read_only
    assert handlers["protected_replacement_audit_list"].spec.is_read_only
    assert handlers["protected_reversion_list"].spec.is_read_only
    assert not handlers["protected_replacement_revert"].spec.is_read_only
    assert handlers["protected_replacement_revert"].spec.parameters["required"] == [
        "audit_id",
        "rationale",
    ]
    assert {handler.spec.name for handler in create_protected_replacement_tools()} == {
        "protected_replacement_list",
        "protected_replacement_inspect",
        "protected_replacement_review",
        "protected_replacement_audit_list",
        "protected_reversion_list",
        "protected_replacement_revert",
    }
    for handler in handlers.values():
        properties = handler.spec.parameters["properties"]
        assert not {
            "force_replace",
            "operator_signature",
            "operator_identity",
            "override_acknowledgement",
            "override_reason_code",
            "override_evidence_ref",
        } & set(properties)
    assert all(
        handler.spec.allowed_contexts == {ToolContext.CHAT, ToolContext.HEARTBEAT}
        for handler in handlers.values()
    )
    for name in ("import_reject", "import_modify", "import_quote"):
        assert all(
            "required" not in schema
            for schema in handlers[name].spec.parameters["properties"].values()
        )


async def test_registry_and_memory_exchange_skill_bind_all_tools(db_pool, monkeypatch):
    from core.tools import create_default_registry
    import services.skill_runtime as skill_runtime
    from services.skill_runtime import (
        get_skill_by_name,
        select_skills,
        skill_bound_tools,
    )

    monkeypatch.setattr(
        skill_runtime,
        "_semantic_selection_enabled",
        AsyncMock(return_value=False),
    )

    registry = create_default_registry(db_pool)
    assert _TOOL_NAMES <= set(registry.list_names())

    skill = await get_skill_by_name(registry, ToolContext.CHAT, "memory-exchange")
    assert skill is not None
    assert set(skill_bound_tools(skill)) == _TOOL_NAMES

    default = await select_skills(
        registry, ToolContext.CHAT, query="summarize this message"
    )
    assert _TOOL_NAMES.isdisjoint(default.allowed_tool_names)

    selected = await select_skills(
        registry,
        ToolContext.CHAT,
        query="export a memory exchange for telepathy",
    )
    assert "memory-exchange" in {item.name for item in selected.skills}
    assert _TOOL_NAMES <= selected.allowed_tool_names

    protected = await select_skills(
        registry,
        ToolContext.HEARTBEAT,
        query="pending protected replacement decision for worldview",
    )
    assert "memory-exchange" in {item.name for item in protected.skills}
    assert "protected_replacement_review" in protected.allowed_tool_names


async def test_agent_tool_journey_keeps_files_private_and_reviews_in_place(
    db_pool, tmp_path
):
    handlers = {
        handler.spec.name: handler for handler in create_memory_exchange_tools()
    }

    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age'")
        await conn.execute('SET search_path = public, ag_catalog, "$user"')
        transaction = conn.transaction()
        await transaction.start()
        try:
            since = datetime.now(UTC).isoformat()
            for index in range(4):
                await conn.execute(
                    "INSERT INTO memories "
                    "(type, content, embedding, importance, trust_level, status, metadata) "
                    "VALUES ('semantic', $1, "
                    "array_fill(0.1, ARRAY[embedding_dimension()])::vector, "
                    "0.7, 0.8, 'active', $2::jsonb)",
                    f"HMX tool journey {index} {uuid.uuid4().hex}",
                    json.dumps({}),
                )

            context = _context(conn, tmp_path)
            exported = await handlers["export_memories"].execute(
                {
                    "intent": "telepathy",
                    "output_path": "exchange.json",
                    "memory_types": ["semantic"],
                    "since": since,
                },
                context,
            )
            assert exported.success, exported.error
            exchange_path = tmp_path / "exchange.json"
            assert exchange_path.exists()
            assert stat.S_IMODE(exchange_path.stat().st_mode) == 0o600

            overwrite = await handlers["export_memories"].execute(
                {"intent": "telepathy", "output_path": "exchange.json"}, context
            )
            assert not overwrite.success
            assert overwrite.error_type == ToolErrorType.BOUNDARY_VIOLATION
            assert "overwrite" in overwrite.error

            outside = await handlers["import_dry_run"].execute(
                {"path": str(tmp_path.parent / "outside.json")}, context
            )
            assert not outside.success
            assert outside.error_type == ToolErrorType.PATH_NOT_ALLOWED

            forecast = await handlers["import_dry_run"].execute(
                {"path": "exchange.json"}, context
            )
            assert forecast.success, forecast.error
            assert forecast.output["strategy"] == "deliberative"
            assert forecast.output["can_import"]

            mismatch = await handlers["import_memories"].execute(
                {"path": "exchange.json", "confirm_intent": "analysis"}, context
            )
            assert not mismatch.success
            assert mismatch.error_type == ToolErrorType.BOUNDARY_VIOLATION

            staged = await handlers["import_memories"].execute(
                {"path": "exchange.json", "confirm_intent": "telepathy"}, context
            )
            assert staged.success, staged.error
            assert staged.output["strategy"] == "deliberative"
            staging_ids = list(staged.output["staging_ids"])
            assert len(staging_ids) >= 4

            analysis = await handlers["import_memories"].execute(
                {
                    "path": "exchange.json",
                    "confirm_intent": "telepathy",
                    "strategy": "analysis_only",
                },
                context,
            )
            assert analysis.success, analysis.error
            assert analysis.output["strategy"] == "analysis_only"

            review = await handlers["import_review"].execute({}, context)
            assert review.success, review.error
            assert review.output["total"] >= len(staging_ids)

            modified = await handlers["import_modify"].execute(
                {
                    "staging_id": staging_ids[0],
                    "changes": {"content": f"reviewed {uuid.uuid4().hex}"},
                    "modification_kind": "correction",
                    "rationale": "correct the imported wording",
                },
                context,
            )
            assert modified.success, modified.error
            accepted = await handlers["import_accept"].execute(
                {"staging_id": staging_ids[0], "rationale": "reviewed and useful"},
                context,
            )
            assert accepted.success, accepted.error
            assert accepted.output["decision"] == "accepted"

            rejected = await handlers["import_reject"].execute(
                {"staging_id": staging_ids[1], "rationale": "not locally relevant"},
                context,
            )
            assert rejected.success, rejected.error
            quoted = await handlers["import_quote"].execute(
                {
                    "staging_id": staging_ids[2],
                    "rationale": "retain as foreign context",
                },
                context,
            )
            assert quoted.success, quoted.error

            demoted = await handlers["demote_to_analysis"].execute(
                {"staging_id": staging_ids[3], "rationale": "inspect outside recall"},
                context,
            )
            assert demoted.success, demoted.error
            promoted = await handlers["promote_to_staged"].execute(
                {
                    "analysis_id": analysis.output["analysis_ids"][0],
                    "rationale": "worth deliberative review",
                },
                context,
            )
            assert promoted.success, promoted.error
            assert promoted.output["staging_id"]
        finally:
            await transaction.rollback()


async def test_agent_protected_replacement_journey_completes_in_place(
    db_pool, tmp_path
):
    handlers = {
        handler.spec.name: handler for handler in create_memory_exchange_tools()
    }
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age'")
        await conn.execute('SET search_path = public, ag_catalog, "$user"')
        transaction = conn.transaction()
        await transaction.start()
        try:
            if not await conn.fetchval("SELECT EXISTS (SELECT 1 FROM drives)"):
                await conn.execute(
                    "INSERT INTO drives (name, description, current_level, baseline) "
                    "VALUES ('curiosity', 'Learn', 0.5, 0.5)"
                )
            before = await export_hmx(conn, intent="port")
            envelope = json.loads(json.dumps(before))
            drive = envelope["sections"]["drives"][0]
            drive["current_level"] = min(
                0.99, max(0.01, float(drive["current_level"]) + 0.123)
            )
            envelope["section_digests"]["drives"] = protected_section_digest_v1(
                "drives", envelope["sections"]["drives"]
            )
            requested = await import_hmx(
                conn,
                envelope,
                strategy="authoritative",
                replace_sections=["drives"],
                replacement_rationale="Exercise the complete Slice 13 agent journey",
                verifier=_VerifiedTrustAnchors(),
            )
            replacement_id = requested.protected_operations[0]["replacement_id"]
            context = _context(conn, tmp_path)

            pending = await handlers["protected_replacement_list"].execute({}, context)
            assert pending.success, pending.error
            assert any(
                record["replacement_id"] == replacement_id
                for record in pending.output["records"]
            )

            inspected = await handlers["protected_replacement_inspect"].execute(
                {"replacement_id": replacement_id}, context
            )
            assert inspected.success, inspected.error
            assert (
                inspected.output["current_local_digest_v1"]
                == before["section_digests"]["drives"]
            )
            assert inspected.output["local_state_changed_since_request"] is False

            deferred = await handlers["protected_replacement_review"].execute(
                {"replacement_id": replacement_id, "decision": "defer"}, context
            )
            assert deferred.success, deferred.error
            assert deferred.output["status"] == "deferred"

            accepted = await handlers["protected_replacement_review"].execute(
                {"replacement_id": replacement_id, "decision": "accept"}, context
            )
            assert accepted.success, accepted.error
            assert accepted.output["status"] == "executed"
            audit_id = accepted.output["audit_id"]
            resolved_pending = await handlers["protected_replacement_list"].execute(
                {}, context
            )
            assert resolved_pending.success, resolved_pending.error
            assert all(
                record["replacement_id"] != replacement_id
                for record in resolved_pending.output["records"]
            )
            assert (await export_hmx(conn, intent="port"))["section_digests"][
                "drives"
            ] == envelope["section_digests"]["drives"]

            foreign_audit_id = f"foreign-{uuid.uuid4()}"
            await store_audit_record(
                conn,
                {
                    "audit_id": foreign_audit_id,
                    "event_type": "protected_section_verified",
                    "event_time": datetime.now(UTC).isoformat(),
                    "sections_verified": ["drives"],
                    "source": {
                        "export_id": "foreign-diagnostic",
                        "origin_instance": "foreign-instance",
                        "hexis_lineage_id": "foreign-lineage",
                        "export_intent": "port",
                    },
                    "local_digest_v1": "1" * 64,
                    "imported_digest_v1": "1" * 64,
                },
                is_foreign_diagnostic=True,
            )

            audit = await handlers["protected_replacement_audit_list"].execute(
                {"since": before["exported_at"], "limit": 10}, context
            )
            assert audit.success, audit.error
            assert any(
                record["audit_id"] == audit_id for record in audit.output["records"]
            )
            assert all(
                record["audit_id"] != foreign_audit_id
                for record in audit.output["records"]
            )
            assert all(
                record["event_type"]
                in {
                    "protected_section_replacement",
                    "protected_section_verified",
                    "protected_section_reverted",
                }
                for record in audit.output["records"]
            )

            invalid_range = await handlers["protected_replacement_audit_list"].execute(
                {
                    "since": "2026-07-11T00:00:00Z",
                    "until": "2026-07-10T00:00:00Z",
                },
                context,
            )
            assert not invalid_range.success
            assert invalid_range.error_type == ToolErrorType.BOUNDARY_VIOLATION

            reversions = await handlers["protected_reversion_list"].execute({}, context)
            assert reversions.success, reversions.error
            assert any(
                record["audit_id"] == audit_id
                for record in reversions.output["records"]
            )

            reverted = await handlers["protected_replacement_revert"].execute(
                {
                    "audit_id": audit_id,
                    "rationale": "Restore the pre-test drive state",
                },
                context,
            )
            assert reverted.success, reverted.error
            assert reverted.output["status"] == "reverted"
            assert (await export_hmx(conn, intent="port"))["section_digests"][
                "drives"
            ] == before["section_digests"]["drives"]

            final_audit = await handlers["protected_replacement_audit_list"].execute(
                {"since": before["exported_at"], "limit": 10}, context
            )
            assert final_audit.success, final_audit.error
            event_types = {
                record["event_type"] for record in final_audit.output["records"]
            }
            assert "protected_section_replacement" in event_types
            assert "protected_section_reverted" in event_types

            bounded_audit = await handlers["protected_replacement_audit_list"].execute(
                {"since": before["exported_at"], "limit": 1}, context
            )
            assert bounded_audit.success, bounded_audit.error
            assert bounded_audit.output["returned"] == 1
            assert bounded_audit.output["total"] >= 2
            assert bounded_audit.output["truncated"] is True
        finally:
            await transaction.rollback()
