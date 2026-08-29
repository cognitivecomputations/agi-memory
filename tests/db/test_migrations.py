"""Tests for the schema-migration runner (core/migrations.py) + the inaugural HMX
Slice 0 migrations. The load-bearing test is the non-destructive proof: an EXISTING
database with real data evolves to the new schema WITHOUT a wipe."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session")]

_DB_ROOT = Path(__file__).resolve().parents[2] / "db"


def _admin_dsn(dbname: str) -> str:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "43815")
    user = os.getenv("POSTGRES_USER", "hexis_user")
    pw = os.getenv("POSTGRES_PASSWORD", "hexis_password")
    return f"postgresql://{user}:{pw}@{host}:{port}/{dbname}"


async def test_migrations_recorded_and_idempotent(db_pool):
    """conftest already migrated this DB; the deltas are recorded and re-runs no-op."""
    from core.migrations import apply_pending_migrations, migration_status

    async with db_pool.acquire() as conn:
        st = await migration_status(conn)
        assert "0001_hmx_enum_values" in st["applied"]
        assert "0002_hmx_supersedes_lineage" in st["applied"]
        assert "0004_hmx_export_functions" in st["applied"]
        assert "0005_hmx_narrative_export_ids" in st["applied"]
        assert "0006_hmx_optional_export_sections" in st["applied"]
        assert "0007_hmx_additive_import" in st["applied"]
        assert "0008_hmx_protected_import" in st["applied"]
        assert "0009_hmx_deliberative_analysis" in st["applied"]
        assert "0010_hmx_reembedding" in st["applied"]
        assert "0011_hmx_in_flight_work" in st["applied"]
        assert "0012_hmx_protected_replacement" in st["applied"]
        assert "0013_hmx_authoritative_import" in st["applied"]
        assert "0014_hmx_reversion" in st["applied"]
        assert "0015_hmx_acceptance_diagnostics" in st["applied"]
        assert "0016_cross_session_fts" in st["applied"]
        assert st["pending"] == []
        assert await apply_pending_migrations(conn) == []  # nothing left to do
        # the deltas are live
        assert await conn.fetchval("SELECT 'staged'::memory_status::text") == "staged"
        assert (
            await conn.fetchval("SELECT 'SUPERSEDES'::graph_edge_type::text")
            == "SUPERSEDES"
        )
        assert await conn.fetchval(
            "SELECT value IS NOT NULL FROM config WHERE key='agent.lineage_id'"
        )


async def test_applied_migration_checksum_drift_fails_loudly(db_pool):
    from core.migrations import (
        MigrationChecksumError,
        apply_pending_migrations,
        migration_status,
    )

    async with db_pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            await conn.execute(
                "UPDATE public.schema_migrations SET checksum = $1 WHERE version = $2",
                "0" * 64,
                "0001_hmx_enum_values",
            )

            status = await migration_status(conn)
            assert len(status["drifted"]) == 1
            drift = status["drifted"][0]
            assert drift["version"] == "0001_hmx_enum_values"
            assert drift["recorded_checksum"] == "0" * 64
            assert drift["current_checksum"] != "0" * 64
            with pytest.raises(
                MigrationChecksumError,
                match="Applied migrations are immutable",
            ):
                await apply_pending_migrations(conn)
        finally:
            await transaction.rollback()


async def test_action_receipt_migration_upgrades_existing_memory_dispatch(db_pool):
    """0199 must preserve the old dispatcher while installing remember v2."""
    migration = (
        _DB_ROOT / "migrations" / "0199_persist_action_receipts.sql"
    ).read_text(encoding="utf-8")

    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            # Recreate the function layout present before 0199.
            await conn.execute("DROP FUNCTION execute_memory_tool(TEXT, JSONB)")
            await conn.execute(
                "ALTER FUNCTION _execute_memory_tool_dispatch(TEXT, JSONB) "
                "RENAME TO execute_memory_tool"
            )
            legacy_prompt = """# Conversation System Prompt

- Tool results, conversation history

Your words about your own actions must match what actually happened this turn.

- **Inspected** means you read content into this conversation only — nothing was retained.
- **Ingested** means a durable ingestion tool (`slow_ingest`, `fast_ingest`, ...) succeeded and wrote provenanced memories.
- **Remembered** means an explicit `remember` call succeeded.

Never say you stored, saved, created, filed, scheduled, or sent something unless the matching tool call succeeded in this turn. Never cite file contents or line numbers you did not read with `inspect_source` this turn. Unsupported action claims are detected and corrected publicly — check before claiming.
"""
            await conn.execute(
                "UPDATE prompt_modules SET content = $1 WHERE key = 'conversation'",
                legacy_prompt,
            )
            await conn.execute(
                "UPDATE prompt_modules SET content = 'legacy current-turn verifier' "
                "WHERE key = 'action_claim_verify'"
            )

            await conn.execute(migration)

            assert await conn.fetchval(
                "SELECT to_regprocedure('public.execute_memory_tool(text,jsonb)') IS NOT NULL"
            )
            assert await conn.fetchval(
                "SELECT to_regprocedure('public._execute_memory_tool_dispatch(text,jsonb)') IS NOT NULL"
            )
            session_id = str(uuid4())
            memory_content = f"migration routing {uuid4().hex}"
            existing_id = await conn.fetchval(
                """
                INSERT INTO memories (
                    type, content, embedding, importance, trust_level, status, metadata
                )
                VALUES (
                    'episodic', $1,
                    array_fill(0.1, ARRAY[embedding_dimension()])::vector,
                    0.5, 0.8, 'active',
                    jsonb_build_object(
                        'tool_write', jsonb_build_object('session_id', $2::text)
                    )
                )
                RETURNING id
                """,
                memory_content,
                session_id,
            )
            routed = await conn.fetchval(
                "SELECT execute_memory_tool('remember', $1::jsonb)",
                json.dumps(
                    {
                        "content": memory_content,
                        "type": "episodic",
                        "_execution_context": {"session_id": session_id},
                    }
                ),
            )
            routed = json.loads(routed) if isinstance(routed, str) else routed
            assert routed["success"] is True, routed
            assert routed["output"]["reused"] is True
            assert routed["output"]["memory_id"] == str(existing_id)
            conversation_prompt = await conn.fetchval(
                "SELECT content FROM prompt_modules WHERE key = 'conversation'"
            )
            assert (
                "durable prior-action receipts as the authority" in conversation_prompt
            )
            verifier_prompt = await conn.fetchval(
                "SELECT content FROM prompt_modules WHERE key = 'action_claim_verify'"
            )
            assert "prior_action_receipts" in verifier_prompt
        finally:
            await tr.rollback()


async def test_migrate_existing_database_preserves_data():
    """Build a DB from the BASELINE only (an 'old' deployment), give it real data,
    then run the migrator: the data survives and the new schema is present."""
    admin_db = os.getenv("POSTGRES_ADMIN_DB", "postgres")
    scratch = f"tmp_mig_{uuid4().hex}"

    admin = await asyncpg.connect(_admin_dsn(admin_db))
    try:
        await admin.execute(f'CREATE DATABASE "{scratch}"')
    finally:
        await admin.close()

    try:
        conn = await asyncpg.connect(_admin_dsn(scratch))
        try:
            # baseline only — NO migrations (simulates a pre-migration instance)
            for path in sorted(_DB_ROOT.glob("*.sql"), key=lambda p: p.name):
                await conn.execute(path.read_text(encoding="utf-8"))
            await conn.execute("LOAD 'age'")
            await conn.execute("SET search_path = public, ag_catalog")

            await conn.execute(
                "INSERT INTO memories (type, content, embedding, importance, trust_level, status) "
                "VALUES ('episodic','precious pre-migration data', "
                "        array_fill(0.1, ARRAY[embedding_dimension()])::vector, 0.5, 0.9, 'active')"
            )

            # NOTE: deltas are mirrored into the baseline (db/migrations/README.md),
            # so a current-baseline DB may already contain them. The load-bearing
            # invariant is that the runner applies every migration on a DB with an
            # empty schema_migrations table, the data survives, and re-runs no-op.
            from core.migrations import apply_pending_migrations

            applied = await apply_pending_migrations(conn)
            assert "0001_hmx_enum_values" in applied
            assert "0002_hmx_supersedes_lineage" in applied
            assert "0003_hmx_bootstrap_provenance" in applied
            assert "0004_hmx_export_functions" in applied
            assert "0005_hmx_narrative_export_ids" in applied
            assert "0006_hmx_optional_export_sections" in applied
            assert "0007_hmx_additive_import" in applied
            assert "0008_hmx_protected_import" in applied
            assert "0009_hmx_deliberative_analysis" in applied
            assert "0010_hmx_reembedding" in applied
            assert "0011_hmx_in_flight_work" in applied
            assert "0012_hmx_protected_replacement" in applied
            assert "0013_hmx_authoritative_import" in applied
            assert "0014_hmx_reversion" in applied
            assert "0015_hmx_acceptance_diagnostics" in applied
            assert "0016_cross_session_fts" in applied
            assert "0017_skill_improvement_proposals" in applied
            assert "0018_full_day_active_hours" in applied

            # AFTER: the data is intact AND the schema evolved
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM memories WHERE content='precious pre-migration data'"
                )
                == 1
            )
            # the enum value is now usable on the surviving row (proves the delta landed)
            await conn.execute(
                "UPDATE memories SET status='staged' WHERE content='precious pre-migration data'"
            )
            assert await conn.fetchval(
                "SELECT value IS NOT NULL FROM config WHERE key='agent.lineage_id'"
            )
            assert await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM ag_catalog.ag_label WHERE name='SUPERSEDES')"
            )
            assert await conn.fetchval(
                "SELECT to_regclass('public.hmx_import_staging') IS NOT NULL "
                "AND to_regclass('public.hmx_analysis_records') IS NOT NULL"
            )
            assert await conn.fetchval(
                "SELECT to_regprocedure('public.hmx_queue_reembed(uuid[])') IS NOT NULL"
            )
            assert await conn.fetchval(
                "SELECT to_regclass('public.hmx_imported_work_refs') IS NOT NULL"
            )
            assert await conn.fetchval(
                "SELECT to_regclass('public.hmx_consent') IS NOT NULL "
                "AND to_regclass('public.protected_replacement_audit') IS NOT NULL "
                "AND to_regclass('public.hmx_pending_replacements') IS NOT NULL"
            )
            assert await conn.fetchval(
                "SELECT to_regprocedure("
                "'public.hmx_import_authoritative(jsonb,text[],jsonb)') IS NOT NULL"
            )
            assert await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
                "WHERE table_schema='public' "
                "AND table_name='hmx_pending_replacements' "
                "AND column_name='reference_map')"
            )
            assert await conn.fetchval(
                "SELECT to_regprocedure("
                "'public.hmx_open_reversion_windows()') IS NOT NULL"
            )
            assert await conn.fetchval(
                "SELECT to_regprocedure("
                "'public.search_cross_session_history(text,integer,text[],timestamp with time zone,timestamp with time zone,uuid,boolean)'"
                ") IS NOT NULL"
            )
            assert await conn.fetchval(
                "SELECT to_regclass('public.idx_subconscious_units_content_fts') IS NOT NULL"
            )
            assert await conn.fetchval(
                "SELECT to_regclass('public.skill_improvement_proposals') IS NOT NULL "
                "AND to_regprocedure('public.claim_skill_improvement_review()') IS NOT NULL "
                "AND to_regprocedure('public.skill_improvement_pending_summary()') IS NOT NULL"
            )
            assert await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
                "WHERE table_schema='public' "
                "AND table_name='protected_replacement_snapshots' "
                "AND column_name='consumed_by_audit_id')"
            )
            # 0003's backfill classified the pre-migration row as lived experience
            assert (
                await conn.fetchval(
                    "SELECT metadata->'provenance'->>'acquisition_mode' FROM memories "
                    "WHERE content='precious pre-migration data'"
                )
                == "experienced"
            )

            # idempotent: a second run does nothing
            assert await apply_pending_migrations(conn) == []
        finally:
            await conn.close()
    finally:
        admin = await asyncpg.connect(_admin_dsn(admin_db))
        try:
            await admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                f"WHERE datname='{scratch}' AND pid <> pg_backend_pid()"
            )
            await admin.execute(f'DROP DATABASE IF EXISTS "{scratch}"')
        finally:
            await admin.close()
