from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from core.node_identity import initialize_node_identity

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


async def test_unknown_node_requires_one_explicit_pairing_decision(db_pool, tmp_path):
    identity = initialize_node_identity(name="Desk Mac", path=tmp_path / "node.json")
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            first = _json(
                await conn.fetchval(
                    "SELECT register_node_handshake($1,$2,$3,$4::jsonb,$5::jsonb)",
                    identity.node_id,
                    identity.public_key,
                    identity.name,
                    json.dumps(["system.run", "screen.capture"]),
                    json.dumps({"platform": "Darwin"}),
                )
            )
            repeated = _json(
                await conn.fetchval(
                    "SELECT register_node_handshake($1,$2,$3,$4::jsonb,$5::jsonb)",
                    identity.node_id,
                    identity.public_key,
                    identity.name,
                    json.dumps(["system.run", "screen.capture"]),
                    "{}",
                )
            )
            assert first["status"] == "pairing_required"
            assert repeated["request_id"] == first["request_id"]
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM node_pairing_requests WHERE node_id=$1 AND status='pending'",
                    identity.node_id,
                )
                == 1
            )
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM hexis_nodes WHERE node_id=$1",
                    identity.node_id,
                )
                == 0
            )
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM outbox_messages WHERE source='node_pairing'"
                )
                == 1
            )
        finally:
            await tr.rollback()


async def test_approval_pairs_exact_key_and_signed_invocation_is_durable(
    db_pool, tmp_path
):
    identity = initialize_node_identity(name="Studio", path=tmp_path / "node.json")
    replacement = initialize_node_identity(
        name="Impostor", path=tmp_path / "replacement.json"
    )
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            pending = _json(
                await conn.fetchval(
                    "SELECT register_node_handshake($1,$2,$3,'[\"system.run\"]'::jsonb,'{}'::jsonb)",
                    identity.node_id,
                    identity.public_key,
                    identity.name,
                )
            )
            approved = _json(
                await conn.fetchval(
                    "SELECT decide_node_pairing($1,'approve','test',NULL)",
                    pending["code"],
                )
            )
            assert approved["status"] == "approved"

            connected = _json(
                await conn.fetchval(
                    "SELECT register_node_handshake($1,$2,$3,'[\"system.run\"]'::jsonb,'{}'::jsonb)",
                    identity.node_id,
                    identity.public_key,
                    identity.name,
                )
            )
            assert connected == {
                "approved": True,
                "status": "paired",
                "node_id": identity.node_id,
            }
            mismatch = _json(
                await conn.fetchval(
                    "SELECT register_node_handshake($1,$2,$3,'[]'::jsonb,'{}'::jsonb)",
                    identity.node_id,
                    replacement.public_key,
                    replacement.name,
                )
            )
            assert mismatch["status"] == "identity_mismatch"

            connection_id = str(uuid4())
            acquired = _json(
                await conn.fetchval(
                    "SELECT mark_node_connection($1,$2,$3::uuid,true,'{}'::jsonb)",
                    identity.node_id,
                    identity.public_key,
                    connection_id,
                )
            )
            competing = _json(
                await conn.fetchval(
                    "SELECT mark_node_connection($1,$2,$3::uuid,true,'{}'::jsonb)",
                    identity.node_id,
                    identity.public_key,
                    str(uuid4()),
                )
            )
            assert acquired["updated"] is True
            assert competing["updated"] is False

            # Upgrade compatibility: a row left online by an older gateway had
            # no session token and must not become permanently unrecoverable.
            await conn.execute(
                """
                UPDATE hexis_nodes
                SET status='online', connection_id=NULL,
                    last_seen_at=CURRENT_TIMESTAMP
                WHERE node_id=$1
                """,
                identity.node_id,
            )
            reclaimed = _json(
                await conn.fetchval(
                    "SELECT mark_node_connection($1,$2,$3::uuid,true,'{}'::jsonb)",
                    identity.node_id,
                    identity.public_key,
                    str(uuid4()),
                )
            )
            assert reclaimed["updated"] is True

            created = _json(
                await conn.fetchval(
                    "SELECT create_node_invocation($1,'system.run',$2::jsonb,'test',30,'{}'::jsonb)",
                    identity.node_id,
                    json.dumps({"command": "notes", "args": []}),
                )
            )
            assert created["queued"] is True
            claimed = _json(
                await conn.fetchval(
                    "SELECT claim_node_invocation($1)", identity.node_id
                )
            )
            assert claimed["invocation_id"] == created["invocation_id"]
            signature = identity.sign(
                {
                    "invocation_id": created["invocation_id"],
                    "success": True,
                    "result": {"stdout": "ok"},
                    "error": None,
                }
            )
            completed = _json(
                await conn.fetchval(
                    "SELECT complete_node_invocation($1::uuid,$2,true,$3::jsonb,NULL,$4)",
                    created["invocation_id"],
                    identity.node_id,
                    json.dumps({"stdout": "ok"}),
                    signature,
                )
            )
            assert completed["updated"] is True
            terminal = _json(
                await conn.fetchval(
                    "SELECT get_node_invocation($1::uuid)", created["invocation_id"]
                )
            )
            assert terminal["status"] == "succeeded"
            assert terminal["result"] == {"stdout": "ok"}
            assert (
                await conn.fetchval(
                    "SELECT result_signature FROM node_invocations WHERE id=$1::uuid",
                    created["invocation_id"],
                )
                == signature
            )
        finally:
            await tr.rollback()


async def test_denial_and_revocation_fail_closed(db_pool, tmp_path):
    identity = initialize_node_identity(name="Denied", path=tmp_path / "node.json")
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            pending = _json(
                await conn.fetchval(
                    "SELECT register_node_handshake($1,$2,$3,'[]'::jsonb,'{}'::jsonb)",
                    identity.node_id,
                    identity.public_key,
                    identity.name,
                )
            )
            denied = _json(
                await conn.fetchval(
                    "SELECT decide_node_pairing($1,'deny','test','not this device')",
                    pending["id"] if "id" in pending else pending["request_id"],
                )
            )
            assert denied["status"] == "denied"
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM hexis_nodes WHERE node_id=$1",
                    identity.node_id,
                )
                == 0
            )
        finally:
            await tr.rollback()


async def test_new_node_capability_requires_a_fresh_explicit_approval(
    db_pool, tmp_path
):
    identity = initialize_node_identity(
        name="Capability change", path=tmp_path / "capability-node.json"
    )
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            original = _json(
                await conn.fetchval(
                    "SELECT register_node_handshake($1,$2,$3,'[\"system.run\"]'::jsonb,'{}'::jsonb)",
                    identity.node_id,
                    identity.public_key,
                    identity.name,
                )
            )
            await conn.fetchval(
                "SELECT decide_node_pairing($1,'approve','test',NULL)",
                original["request_id"],
            )

            escalation = _json(
                await conn.fetchval(
                    "SELECT register_node_handshake($1,$2,$3,'[\"system.run\",\"audio.wake\"]'::jsonb,'{}'::jsonb)",
                    identity.node_id,
                    identity.public_key,
                    identity.name,
                )
            )
            assert escalation["status"] == "pairing_required"
            assert "added node capabilities" in escalation["next_step"]
            assert _json(
                await conn.fetchval(
                    "SELECT capabilities FROM hexis_nodes WHERE node_id=$1",
                    identity.node_id,
                )
            ) == ["system.run"]

            await conn.fetchval(
                "SELECT decide_node_pairing($1,'approve','test',NULL)",
                escalation["request_id"],
            )
            accepted = _json(
                await conn.fetchval(
                    "SELECT register_node_handshake($1,$2,$3,'[\"system.run\",\"audio.wake\"]'::jsonb,'{}'::jsonb)",
                    identity.node_id,
                    identity.public_key,
                    identity.name,
                )
            )
            assert accepted["status"] == "paired"
            assert set(
                _json(
                    await conn.fetchval(
                        "SELECT capabilities FROM hexis_nodes WHERE node_id=$1",
                        identity.node_id,
                    )
                )
            ) == {"system.run", "audio.wake"}
        finally:
            await tr.rollback()


async def test_z_0224_migration_is_self_contained(db_pool):
    migration = (
        Path(__file__).resolve().parents[2]
        / "db"
        / "migrations"
        / "0224_companion_nodes.sql"
    ).read_text(encoding="utf-8")
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute(
                """
                DROP FUNCTION IF EXISTS append_agent_visual_message(UUID,TEXT,TEXT);
                DROP FUNCTION IF EXISTS get_node_invocation(UUID);
                DROP FUNCTION IF EXISTS complete_node_invocation(UUID,TEXT,BOOLEAN,JSONB,TEXT,TEXT);
                DROP FUNCTION IF EXISTS claim_node_invocation(TEXT);
                DROP FUNCTION IF EXISTS create_node_invocation(TEXT,TEXT,JSONB,TEXT,INTEGER,JSONB);
                DROP FUNCTION IF EXISTS revoke_hexis_node(TEXT,TEXT,TEXT);
                DROP FUNCTION IF EXISTS mark_node_connection(TEXT,TEXT,UUID,BOOLEAN,JSONB);
                DROP FUNCTION IF EXISTS list_hexis_nodes();
                DROP FUNCTION IF EXISTS list_node_pairing_requests(TEXT,INTEGER);
                DROP FUNCTION IF EXISTS decide_node_pairing(TEXT,TEXT,TEXT,TEXT);
                DROP FUNCTION IF EXISTS register_node_handshake(TEXT,TEXT,TEXT,JSONB,JSONB);
                DROP TABLE node_invocations, node_pairing_requests, hexis_nodes CASCADE;
                """
            )
            await conn.execute(migration)
            assert await conn.fetchval("SELECT to_regclass('public.hexis_nodes')") == (
                "hexis_nodes"
            )
            assert (
                await conn.fetchval(
                    "SELECT to_regprocedure('append_agent_visual_message(uuid,text,text)') IS NOT NULL"
                )
                is True
            )
            assert _json(await conn.fetchval("SELECT list_hexis_nodes()")) == []
        finally:
            await tr.rollback()
