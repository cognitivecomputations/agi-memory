"""End-to-end command coverage for HMX Slice 3."""

from __future__ import annotations

import base64
import json
import os
import stat
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from apps.cli_exchange import _print_import_result
from core.digest import content_hash_v1, protected_section_digest_v1
from core.memory_exchange import (
    HmxAuthoritativeResult,
    HmxImportResult,
    PROTECTED_SECTIONS,
    build_envelope,
    resolve_export_sections,
)
from core.protected_replacement import (
    OPERATOR_OVERRIDE_ACKNOWLEDGEMENT,
    revert_protected_replacement,
)
from core.trust_anchors import HMX_OPERATOR_PUBLIC_KEY_ENV

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.cli]

_ROOT = Path(__file__).resolve().parents[2]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "apps.hexis_cli", *args],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        cwd=_ROOT,
    )


def _run_with_env(
    env_updates: dict[str, str], *args: str
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(env_updates)
    return subprocess.run(
        [sys.executable, "-m", "apps.hexis_cli", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=_ROOT,
    )


def _single_memory_document(intent: str, content: str) -> dict:
    document = build_envelope(
        intent=intent,
        plan=resolve_export_sections(intent),
        instance_id="cli-test-source",
        schema_version="0009_hmx_deliberative_analysis",
        embedding_model="embeddinggemma:300m",
        embedding_dimension=768,
        lineage_id=str(uuid.uuid4()),
        relationship_edge_types=["SUPPORTS"],
    )
    document["sections"] = {
        "memories": [
            {
                "ref": f"{document['export_id']}:{uuid.uuid4()}",
                "type": "semantic",
                "status": "active",
                "content": content,
                "content_hash_v1": content_hash_v1(content),
                "importance": 0.6,
                "trust_level": 0.7,
                "metadata": {},
                "provenance": {
                    "acquisition_mode": "experienced",
                    "origin_instance": "cli-test-source",
                    "origin_id": str(uuid.uuid4()),
                    "import_chain": [],
                    "modification_chain": [],
                },
            }
        ]
    }
    return document


async def test_import_help_exposes_explicit_failed_work_retry():
    result = _run("import", "--help")
    assert result.returncode == 0
    assert "--retry-failed-work" in result.stdout
    assert "--force-replace" in result.stdout
    assert "--operator-signature" in result.stdout
    assert "--mind" in result.stdout


async def test_mind_export_creates_a_private_complete_port_file(db_pool, tmp_path):
    hexis_home = tmp_path / "hexis-home"

    exported = _run_with_env(
        {"HEXIS_HOME": str(hexis_home)},
        "export",
        "--mind",
        "--wait-seconds",
        "60",
    )

    assert exported.returncode == 0, exported.stderr
    files = list((hexis_home / "exports").glob("*.hmx.json"))
    assert len(files) == 1
    output = files[0]
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["export_intent"] == "port"
    assert set(document["export_scope"]["include_protected"]) == set(
        PROTECTED_SECTIONS
    )
    assert document["export_scope"]["include_in_flight_work"] is True
    assert document["export_scope"]["include_audit_records"] is True
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    human_output = " ".join(exported.stdout.split())
    assert f"hexis import {output.name} --mind --dry-run --json" in human_output
    assert f"hexis import {output.name} --mind --confirm-intent port" in human_output


async def test_mind_export_refuses_conflicting_or_partial_policy():
    conflicting = _run("export", "--mind", "--intent", "telepathy")
    partial = _run("export", "--mind", "--types", "semantic")

    assert conflicting.returncode != 0
    assert "cannot be combined with another intent" in conflicting.stderr
    assert partial.returncode != 0
    assert "mind export must be complete" in partial.stderr


async def test_mind_import_refuses_non_port_and_non_additive_files(tmp_path):
    telepathy_path = tmp_path / "telepathy.hmx.json"
    telepathy_path.write_text(
        json.dumps(_single_memory_document("telepathy", "A shared observation")),
        encoding="utf-8",
    )
    non_port = _run("import", str(telepathy_path), "--mind", "--dry-run")

    port_path = tmp_path / "mind.hmx.json"
    port_path.write_text(
        json.dumps(
            build_envelope(
                intent="port",
                plan=resolve_export_sections("port"),
                instance_id="cli-test-source",
                schema_version="0009_hmx_deliberative_analysis",
                embedding_model="embeddinggemma:300m",
                embedding_dimension=768,
                lineage_id=str(uuid.uuid4()),
                relationship_edge_types=["SUPPORTS"],
            )
        ),
        encoding="utf-8",
    )
    non_additive = _run(
        "import",
        str(port_path),
        "--mind",
        "--strategy",
        "authoritative",
        "--dry-run",
    )

    assert non_port.returncode != 0
    assert "accepts only a port-intent HMX file" in non_port.stderr
    assert non_additive.returncode != 0
    assert "empty-target additive path" in non_additive.stderr


async def test_mind_import_result_reports_continuity_in_human_and_json(capsys):
    result = HmxImportResult(
        export_id="mind-continuity-output",
        intent="port",
        strategy="additive",
        target_state={},
        inserted={},
        duplicate_refs=(),
        ref_map={},
        conflicts=(),
        warnings=(),
    )
    continuity = {
        "verified": True,
        "lineage": {"matches": True},
        "protected_sections": {},
    }

    _print_import_result(
        result,
        as_json=False,
        skipped=[],
        continuity=continuity,
    )
    human_output = " ".join(capsys.readouterr().out.split())
    assert "Mind continuity verified" in human_output
    assert "constitutional section projections match" in human_output

    _print_import_result(
        result,
        as_json=True,
        skipped=[],
        continuity=continuity,
    )
    json_output = json.loads(capsys.readouterr().out)
    assert json_output["continuity"] == continuity


async def test_export_jsonl_and_database_aware_dry_run(db_pool, tmp_path):
    output = tmp_path / "memory.hmx.jsonl"
    async with db_pool.acquire() as conn:
        memory_id = await conn.fetchval(
            "INSERT INTO memories (type, content, embedding) "
            "VALUES ('semantic', $1, array_fill(0.1, ARRAY[embedding_dimension()])::vector) "
            "RETURNING id",
            f"HMX CLI round trip {uuid.uuid4().hex}",
        )
    try:
        exported = _run(
            "export",
            "--intent",
            "telepathy",
            "--format",
            "jsonl",
            "--output",
            str(output),
            "--wait-seconds",
            "60",
        )
        assert exported.returncode == 0, exported.stderr
        assert (
            output.read_text(encoding="utf-8")
            .splitlines()[0]
            .startswith('{"record_type": "envelope"')
        )
        assert stat.S_IMODE(output.stat().st_mode) == 0o600

        dry_run = _run(
            "import",
            str(output),
            "--strategy",
            "additive",
            "--dry-run",
            "--json",
            "--wait-seconds",
            "60",
        )
        assert dry_run.returncode == 0, dry_run.stderr
        report = json.loads(dry_run.stdout)
        assert report["intent"] == "telepathy"
        assert report["strategy"] == "additive"
        assert report["can_import"] is True
        assert report["counts"]["duplicate_memories"] >= 1
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute("DELETE FROM memories WHERE id = $1", memory_id)


async def test_export_refuses_silent_overwrite(db_pool, tmp_path):
    output = tmp_path / "memory.hmx.json"
    first = _run("export", "--intent", "telepathy", "--output", str(output))
    assert first.returncode == 0, first.stderr
    original = output.read_bytes()

    second = _run("export", "--intent", "telepathy", "--output", str(output))
    assert second.returncode != 0
    assert "--overwrite" in second.stderr
    assert output.read_bytes() == original


async def test_authoritative_dry_run_reports_explicit_replacement_protocol(
    db_pool, tmp_path
):
    output = tmp_path / "authoritative.hmx.json"
    exported = _run("export", "--intent", "port", "--output", str(output))
    assert exported.returncode == 0, exported.stderr

    dry_run = _run(
        "import",
        str(output),
        "--strategy",
        "authoritative",
        "--replace",
        "worldview",
        "--trust-matching-lineage-label",
        "--dry-run",
        "--json",
        "--wait-seconds",
        "60",
    )
    assert dry_run.returncode == 0, dry_run.stderr
    report = json.loads(dry_run.stdout)
    assert report["can_import"] is True
    assert report["strategy"] == "authoritative"
    assert report["protected_policy"]["sections"] == ["worldview"]
    assert report["protected_policy"]["operations"][0]["disposition"] == (
        "verified_noop_candidate"
    )


async def test_operator_override_dry_run_payload_executes_with_matching_signature(
    db_pool, tmp_path
):
    source = tmp_path / "operator-override.hmx.json"
    rationale = "CLI recovery while the acknowledgement channel is paused"
    private_key = Ed25519PrivateKey.generate()
    raw_public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    env = {
        HMX_OPERATOR_PUBLIC_KEY_ENV: base64.b64encode(raw_public_key).decode("ascii")
    }
    baseline_content = f"CLI override baseline {uuid.uuid4().hex}"
    memory_id = None
    audit_id = None
    async with db_pool.acquire() as conn:
        memory_id = await conn.fetchval(
            "INSERT INTO memories "
            "(type, content, embedding, importance, trust_level, status, metadata) "
            "VALUES ('worldview', $1, "
            "array_fill(0.1, ARRAY[embedding_dimension()])::vector, 0.8, 0.9, "
            "'active', $2::jsonb) RETURNING id",
            baseline_content,
            json.dumps(
                {
                    "category": "value",
                    "confidence": 0.9,
                    "stability": 0.9,
                    "provenance": {"acquisition_mode": "experienced"},
                }
            ),
        )
    try:
        exported = _run("export", "--intent", "port", "--output", str(source))
        assert exported.returncode == 0, exported.stderr
        document = json.loads(source.read_text(encoding="utf-8"))
        record = document["sections"]["worldview"][0]
        record["content"] = f"CLI override replacement {uuid.uuid4().hex}"
        record["content_hash_v1"] = content_hash_v1(record["content"])
        document["section_digests"]["worldview"] = protected_section_digest_v1(
            "worldview", document["sections"]["worldview"]
        )
        source.write_text(json.dumps(document), encoding="utf-8")
        async with db_pool.acquire() as conn:
            await conn.execute("UPDATE heartbeat_state SET is_paused=TRUE WHERE id=1")

        override_args = (
            "import",
            str(source),
            "--strategy",
            "authoritative",
            "--replace",
            "worldview",
            "--replacement-rationale",
            rationale,
            "--force-replace",
            "--operator-identity",
            "cli-test-operator",
            "--override-acknowledgement",
            OPERATOR_OVERRIDE_ACKNOWLEDGEMENT,
            "--override-reason-code",
            "agent_paused",
            "--override-evidence-ref",
            "report:cli-override-test",
            "--json",
            "--wait-seconds",
            "60",
        )
        dry_run = _run_with_env(env, *override_args, "--dry-run")
        assert dry_run.returncode == 0, dry_run.stderr
        dry_report = json.loads(dry_run.stdout)
        override = dry_report["operator_override"]
        assert override["signature_supplied"] is False
        assert override["trust_anchor_id"].startswith("ed25519:sha256:")
        payload = base64.b64decode(override["payload_base64"], validate=True)
        signature = base64.b64encode(private_key.sign(payload)).decode("ascii")

        imported = _run_with_env(
            env,
            *override_args,
            "--operator-signature",
            signature,
            "--confirm-intent",
            "port",
        )
        assert imported.returncode == 0, imported.stderr
        report = json.loads(imported.stdout)
        operation = report["protected_operations"][0]
        assert operation["replacement_executor"] == "operator_override"
        assert operation["status"] == "executed"
        audit_id = operation["audit_ids"][0]
        async with db_pool.acquire() as conn:
            audit = await conn.fetchval(
                "SELECT record FROM protected_replacement_audit WHERE audit_id=$1",
                audit_id,
            )
            audit = json.loads(audit) if isinstance(audit, str) else audit
            assert audit["agent_acknowledgement"] == "bypassed"
            assert audit["override_evidence_ref"] == "report:cli-override-test"
    finally:
        async with db_pool.acquire() as conn:
            if audit_id:
                await revert_protected_replacement(
                    conn,
                    audit_id,
                    rationale="Restore state after the CLI override journey test",
                    actor_identity="cli",
                )
            await conn.execute("UPDATE heartbeat_state SET is_paused=FALSE WHERE id=1")
            if memory_id:
                await conn.execute(
                    "DELETE FROM memories WHERE id=$1 OR content=$2",
                    memory_id,
                    baseline_content,
                )


async def test_authoritative_human_output_surfaces_reused_refusal(capsys):
    result = HmxAuthoritativeResult(
        export_id="hmx-refused-test",
        intent="port",
        strategy="authoritative",
        target_state={},
        inserted={},
        protected_operations=(
            {
                "disposition": "refused",
                "status": "refused",
                "replacement_id": "replacement-refused-test",
                "section": "worldview",
                "agent_acknowledgement_required": False,
            },
        ),
        ref_map={},
        conflicts=(),
        warnings=(),
    )

    _print_import_result(result, as_json=False, skipped=[])

    output = " ".join(capsys.readouterr().out.split())
    assert "refused" in output
    assert "prior agent refusal remains in force" in output
    assert "resolved: 1" in output


async def test_authoritative_human_output_surfaces_reverted_request(capsys):
    result = HmxAuthoritativeResult(
        export_id="hmx-reverted-test",
        intent="port",
        strategy="authoritative",
        target_state={},
        inserted={},
        protected_operations=(
            {
                "disposition": "reverted",
                "status": "reverted",
                "replacement_id": "replacement-reverted-test",
                "section": "identity",
                "agent_acknowledgement_required": False,
            },
        ),
        ref_map={},
        conflicts=(),
        warnings=(),
    )

    _print_import_result(result, as_json=False, skipped=[])

    output = " ".join(capsys.readouterr().out.split())
    assert "executed and later reverted" in output
    assert "resolved: 1" in output


async def test_authoritative_human_output_prominently_surfaces_override(capsys):
    result = HmxAuthoritativeResult(
        export_id="hmx-override-output-test",
        intent="port",
        strategy="authoritative",
        target_state={},
        inserted={},
        protected_operations=(
            {
                "disposition": "executed",
                "status": "executed",
                "replacement_id": "replacement-override-test",
                "section": "worldview",
                "agent_acknowledgement_required": False,
                "replacement_executor": "operator_override",
            },
        ),
        ref_map={},
        conflicts=(),
        warnings=(),
    )

    _print_import_result(result, as_json=False, skipped=[])

    output = " ".join(capsys.readouterr().out.split())
    assert "HMX OPERATOR OVERRIDE EXECUTED" in output
    assert "agent acknowledgement was bypassed" in output
    assert "reversion window remains open" in output


async def test_export_rejects_inverted_time_range_before_connecting():
    result = _run(
        "export",
        "--intent",
        "telepathy",
        "--since",
        "2026-07-10",
        "--until",
        "2026-07-09",
    )
    assert result.returncode != 0
    assert "--since must be earlier" in result.stderr


async def test_import_requires_exact_intent_confirmation(db_pool, tmp_path):
    output = tmp_path / "memory.hmx.json"
    exported = _run("export", "--intent", "telepathy", "--output", str(output))
    assert exported.returncode == 0, exported.stderr

    missing = _run("import", str(output), "--strategy", "additive")
    assert missing.returncode != 0
    assert "--confirm-intent telepathy" in missing.stderr

    mismatch = _run(
        "import",
        str(output),
        "--strategy",
        "additive",
        "--confirm-intent",
        "analysis",
    )
    assert mismatch.returncode != 0
    assert "intent confirmation mismatch" in mismatch.stderr


async def test_confirmed_additive_import_completes_in_place(db_pool, tmp_path):
    content = f"HMX CLI accepted import {uuid.uuid4().hex}"
    document = _single_memory_document("telepathy", content)
    source = tmp_path / "accepted.hmx.json"
    source.write_text(json.dumps(document), encoding="utf-8")

    imported = _run(
        "import",
        str(source),
        "--strategy",
        "additive",
        "--confirm-intent",
        "telepathy",
        "--json",
        "--wait-seconds",
        "60",
    )
    assert imported.returncode == 0, imported.stderr
    report = json.loads(imported.stdout)
    assert report["inserted"]["memories"] == 1

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, metadata FROM memories WHERE content = $1", content
        )
        assert row is not None
        try:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            assert metadata["embedding_status"] == "pending_import"
        finally:
            await conn.execute("DELETE FROM memories WHERE id = $1", row["id"])


@pytest.mark.parametrize(
    ("intent", "result_key", "table"),
    [
        ("telepathy", "staging_ids", "hmx_import_staging"),
        ("analysis", "analysis_ids", "hmx_analysis_records"),
    ],
)
async def test_import_uses_intent_derived_isolated_strategy(
    db_pool, tmp_path, intent, result_key, table
):
    document = _single_memory_document(
        intent, f"CLI isolated default {intent} {uuid.uuid4().hex}"
    )
    source = tmp_path / f"{intent}.hmx.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    async with db_pool.acquire() as conn:
        before_memories = await conn.fetchval("SELECT count(*) FROM memories")

    imported = _run(
        "import",
        str(source),
        "--confirm-intent",
        intent,
        "--json",
        "--wait-seconds",
        "60",
    )
    assert imported.returncode == 0, imported.stderr
    report = json.loads(imported.stdout)
    assert len(report[result_key]) == 1
    record_id = report[result_key][0]
    if intent == "telepathy":
        pending = _run("import-review", "list", "--json", "--wait-seconds", "60")
        assert pending.returncode == 0, pending.stderr
        assert any(
            item["id"] == record_id for item in json.loads(pending.stdout)["records"]
        )
        rejected = _run(
            "import-review",
            "reject",
            record_id,
            "--rationale",
            "CLI lifecycle test",
            "--json",
            "--wait-seconds",
            "60",
        )
        assert rejected.returncode == 0, rejected.stderr
        assert json.loads(rejected.stdout)["decision"] == "rejected"
    async with db_pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM memories") == before_memories
        assert (
            await conn.fetchval(
                f"SELECT count(*) FROM {table} WHERE id=$1::uuid", record_id
            )
            == 1
        )
        batch_table = (
            "hmx_import_batches" if intent == "telepathy" else "hmx_analysis_batches"
        )
        batch_id = report["batch_id"]
        await conn.execute(f"DELETE FROM {batch_table} WHERE id=$1::uuid", batch_id)
