"""Tests for worker_code_schema_skew_report() (db/88, migration 0242) -- the
detection behind issue #113: a long-running worker's own bundled
db/migrations/ predates what's actually applied to the DB it shares, and
nothing previously checked or surfaced that in `hexis status`/`hexis doctor`.
"""
from __future__ import annotations

import json

import pytest

from tests.utils import get_test_identifier

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


def _j(v):
    return json.loads(v) if isinstance(v, str) else v


async def _latest_migration_version(conn) -> str | None:
    return await conn.fetchval("SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1")


async def _register(conn, *, instance: str, bundled: str | None, status: str = "running"):
    metadata = {"process_id": 1, "host_name": "test-host"}
    if bundled is not None:
        metadata["bundled_latest_migration"] = bundled
    worker_id = await conn.fetchval(
        "SELECT register_worker_instance('heartbeat', $1, $2::jsonb)",
        instance,
        json.dumps(metadata),
    )
    if status != "running":
        await conn.execute(
            "UPDATE worker_instances SET status = $1 WHERE id = $2::uuid", status, worker_id
        )
    return str(worker_id)  # report JSON encodes UUIDs as text


async def test_no_skew_when_bundled_matches_latest(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            latest = await _latest_migration_version(conn)
            instance = get_test_identifier("skew_match")
            worker_id = await _register(conn, instance=instance, bundled=latest)

            report = _j(await conn.fetchval("SELECT worker_code_schema_skew_report()"))
            assert report["db_latest_migration"] == latest
            assert worker_id not in [w["worker_id"] for w in report["skewed_workers"]]
        finally:
            await tr.rollback()


async def test_skew_detected_for_worker_with_older_bundle(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            instance = get_test_identifier("skew_old")
            worker_id = await _register(conn, instance=instance, bundled="0001_fake_old")

            report = _j(await conn.fetchval("SELECT worker_code_schema_skew_report()"))
            skewed = {w["worker_id"]: w for w in report["skewed_workers"]}
            assert worker_id in skewed
            entry = skewed[worker_id]
            assert entry["mode"] == "heartbeat"
            assert entry["instance_name"] == instance
            assert entry["bundled_latest_migration"] == "0001_fake_old"
            assert entry["db_latest_migration"] == report["db_latest_migration"]
        finally:
            await tr.rollback()


async def test_worker_without_bundled_version_is_not_flagged(db_pool):
    # A worker that has never reported its bundled migration (older Hexis
    # build, before #113 landed) is a known-unknown, not a confirmed skew --
    # asserting skew we can't actually verify would be worse than silence.
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            instance = get_test_identifier("skew_unknown")
            worker_id = await _register(conn, instance=instance, bundled=None)

            report = _j(await conn.fetchval("SELECT worker_code_schema_skew_report()"))
            assert worker_id not in [w["worker_id"] for w in report["skewed_workers"]]
        finally:
            await tr.rollback()


async def test_stale_worker_is_not_flagged(db_pool):
    # A worker that stopped heartbeating is already covered by the separate
    # staleness check; re-flagging it here would just be noise.
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            instance = get_test_identifier("skew_stale")
            worker_id = await _register(conn, instance=instance, bundled="0001_fake_old")
            await conn.execute(
                "UPDATE worker_instances SET last_seen_at = CURRENT_TIMESTAMP - INTERVAL '10 minutes' "
                "WHERE id = $1::uuid",
                worker_id,
            )

            report = _j(await conn.fetchval("SELECT worker_code_schema_skew_report()"))
            assert worker_id not in [w["worker_id"] for w in report["skewed_workers"]]
        finally:
            await tr.rollback()


async def test_stopped_worker_is_not_flagged(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            instance = get_test_identifier("skew_stopped")
            worker_id = await _register(
                conn, instance=instance, bundled="0001_fake_old", status="stopped"
            )

            report = _j(await conn.fetchval("SELECT worker_code_schema_skew_report()"))
            assert worker_id not in [w["worker_id"] for w in report["skewed_workers"]]
        finally:
            await tr.rollback()
