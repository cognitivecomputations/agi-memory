import os
import json
import uuid
from pathlib import Path
from typing import Any

import asyncpg
import pytest
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    retry_if_result,
    stop_after_delay,
    wait_fixed,
)

from tests.utils import _db_dsn


@pytest.fixture(autouse=True)
def disable_bundled_gmail_oauth_client(monkeypatch):
    """Keep tests from consuming the real bundled Google Desktop client."""
    monkeypatch.setenv("HEXIS_GMAIL_OAUTH_DISABLE_BUNDLED", "1")


async def _connect_with_retry(dsn: str, *, wait_seconds: int, timeout_seconds: float) -> asyncpg.Connection:
    retrying = AsyncRetrying(
        stop=stop_after_delay(wait_seconds),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async for attempt in retrying:
        with attempt:
            return await asyncpg.connect(dsn, timeout=timeout_seconds)
    raise RuntimeError("connect retry loop exhausted")


@pytest.fixture(scope="module", autouse=True)
async def temp_test_db():
    """Create a dedicated test database per module and drop it after tests."""
    original_db = os.getenv("POSTGRES_DB", "hexis_memory")
    temp_db = f"tmp_test_{uuid.uuid4().hex}"
    os.environ["POSTGRES_DB"] = temp_db

    admin_db = os.getenv("POSTGRES_ADMIN_DB", "postgres")
    admin_dsn = _db_dsn(admin_db)

    wait_seconds = int(os.getenv("POSTGRES_WAIT_SECONDS", "60"))
    connect_timeout = float(os.getenv("POSTGRES_CONNECT_TIMEOUT", "5"))
    try:
        admin_conn = await _connect_with_retry(
            admin_dsn,
            wait_seconds=wait_seconds,
            timeout_seconds=connect_timeout,
        )
    except Exception as exc:
        pytest.fail(
            f"Postgres not available for test DB creation after {wait_seconds}s "
            f"(connect_timeout={connect_timeout}s): {exc!r}"
        )
    try:
        await admin_conn.execute(f'CREATE DATABASE "{temp_db}"')
    finally:
        await admin_conn.close()

    schema_dir = Path(__file__).resolve().parents[1] / "db"
    schema_files = sorted(schema_dir.glob("*.sql"), key=lambda path: path.name)
    try:
        test_conn = await _connect_with_retry(
            _db_dsn(temp_db),
            wait_seconds=wait_seconds,
            timeout_seconds=connect_timeout,
        )
    except Exception as exc:
        pytest.fail(
            f"Postgres not available for schema init after {wait_seconds}s "
            f"(connect_timeout={connect_timeout}s): {exc!r}"
        )
    try:
        for schema_path in schema_files:
            await test_conn.execute(schema_path.read_text(encoding="utf-8"))
    finally:
        await test_conn.close()

    yield

    try:
        admin_conn = await _connect_with_retry(
            admin_dsn,
            wait_seconds=wait_seconds,
            timeout_seconds=connect_timeout,
        )
    except Exception as exc:
        pytest.fail(
            f"Postgres not available for test DB cleanup after {wait_seconds}s "
            f"(connect_timeout={connect_timeout}s): {exc!r}"
        )
    try:
        await admin_conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $1 AND pid <> pg_backend_pid()",
            temp_db,
        )
        await admin_conn.execute(f'DROP DATABASE IF EXISTS "{temp_db}"')
    finally:
        await admin_conn.close()

    if original_db:
        os.environ["POSTGRES_DB"] = original_db
    else:
        os.environ.pop("POSTGRES_DB", None)


@pytest.fixture(scope="module")
async def db_pool(temp_test_db):
    """Create a connection pool for testing."""
    db_url = _db_dsn()
    # Postgres restarts once during initdb, and this repo's schema init can take >60s on cold starts.
    wait_seconds = int(os.getenv("POSTGRES_WAIT_SECONDS", "60"))
    connect_timeout = float(os.getenv("POSTGRES_CONNECT_TIMEOUT", "5"))
    pool = None
    retrying = AsyncRetrying(
        stop=stop_after_delay(wait_seconds),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async for attempt in retrying:
        with attempt:
            pool = await asyncpg.create_pool(
                db_url,
                ssl=False,
                min_size=2,
                max_size=20,
                command_timeout=60.0,
                timeout=connect_timeout,
            )
    assert pool is not None
    yield pool
    await pool.close()


@pytest.fixture(scope="module", autouse=True)
async def sync_test_embedding_dimension_from_db(db_pool, request):
    """
    Sync module-level EMBEDDING_DIMENSION with the database's configured dimension.
    """
    async with db_pool.acquire() as conn:
        dim = await conn.fetchval("SELECT embedding_dimension()")
    if hasattr(request.module, "EMBEDDING_DIMENSION"):
        request.module.EMBEDDING_DIMENSION = int(dim)


@pytest.fixture(scope="module", autouse=True)
async def configure_agent_for_tests(db_pool):
    """
    Heartbeats are gated behind `agent.is_configured`.

    Most tests exercise heartbeat/worker behavior, so default to configured.
    Individual tests can override by deleting the config row in a transaction.
    """
    tracked_keys = [
        "agent.is_configured",
        "agent.objectives",
        "llm.heartbeat",
        "llm.chat",
        "llm.subconscious",
        "agent.consent_status",
        "agent.consent_signature",
        "agent.consent_log_id",
        "agent.consent_recorded_at",
        "agent.consent_memory_ids",
    ]
    original_config: dict[str, Any] = {}
    original_hb_interval: float | None = None
    original_hb_paused: bool | None = None
    original_init_stage: str | None = None
    original_init_data: Any | None = None
    original_init_started_at: Any | None = None
    original_init_completed_at: Any | None = None
    inserted_consent_log_id: str | None = None
    inserted_consent_memory_ids: list[str] = []

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT key, value FROM config WHERE key = ANY($1::text[])",
            tracked_keys,
        )
        original_config = {row["key"]: row["value"] for row in rows}
        # Phase 7 (ReduceScopeCreep): heartbeat_config removed - use unified config
        original_hb_interval = await conn.fetchval(
            "SELECT value FROM config WHERE key = 'heartbeat.heartbeat_interval_minutes'"
        )
        original_hb_paused = await conn.fetchval(
            "SELECT is_paused FROM heartbeat_state WHERE id = 1"
        )
        init_state = await conn.fetchrow(
            "SELECT init_stage, init_data, init_started_at, init_completed_at FROM heartbeat_state WHERE id = 1"
        )
        if init_state:
            original_init_stage = str(init_state["init_stage"])
            original_init_data = init_state["init_data"]
            original_init_started_at = init_state["init_started_at"]
            original_init_completed_at = init_state["init_completed_at"]

        consent_payload = {
            "decision": "consent",
            "signature": "test-consent",
            "memories": [],
        }
        recorded = await conn.fetchval(
            "SELECT record_consent_response($1::jsonb)",
            json.dumps(consent_payload),
        )
        if isinstance(recorded, str):
            try:
                recorded = json.loads(recorded)
            except Exception:
                recorded = {}
        if isinstance(recorded, dict):
            inserted_consent_log_id = str(recorded.get("log_id") or "") or None
            raw_memory_ids = recorded.get("memory_ids") or []
            if isinstance(raw_memory_ids, list):
                inserted_consent_memory_ids = [str(item) for item in raw_memory_ids if item]

        # Minimal config so CLI `hexis config validate` passes in subprocess smoke tests.
        await conn.execute("SELECT set_config('agent.is_configured', 'true'::jsonb)")
        await conn.execute("SELECT set_config('agent.objectives', $1::jsonb)", json.dumps(["test objective"]))
        await conn.execute(
            "SELECT set_config('llm.heartbeat', $1::jsonb)",
            json.dumps({"provider": "openai", "model": "gpt-4o", "endpoint": "", "api_key_env": ""}),
        )
        await conn.execute(
            "SELECT set_config('llm.chat', $1::jsonb)",
            json.dumps({"provider": "openai", "model": "gpt-4o", "endpoint": "", "api_key_env": ""}),
        )
        await conn.execute(
            "SELECT set_config('llm.subconscious', $1::jsonb)",
            json.dumps({"provider": "openai", "model": "gpt-4o", "endpoint": "", "api_key_env": ""}),
        )
        await conn.execute("UPDATE heartbeat_state SET is_paused = FALSE WHERE id = 1")
        await conn.execute(
            """
            UPDATE heartbeat_state
            SET init_stage = 'complete',
                init_started_at = COALESCE(init_started_at, CURRENT_TIMESTAMP),
                init_completed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """
        )
        # Phase 7 (ReduceScopeCreep): use unified config
        await conn.execute("SELECT set_config('heartbeat.heartbeat_interval_minutes', '60'::jsonb)")

    yield

    async with db_pool.acquire() as conn:
        for key in tracked_keys:
            if key in original_config:
                await conn.execute(
                    "SELECT set_config($1, $2::jsonb)",
                    key,
                    json.dumps(original_config[key]),
                )
            else:
                await conn.execute("DELETE FROM config WHERE key = $1", key)

        if inserted_consent_log_id:
            await conn.execute("DELETE FROM consent_log WHERE id = $1::uuid", inserted_consent_log_id)
        if inserted_consent_memory_ids:
            await conn.execute(
                "DELETE FROM memories WHERE id = ANY($1::uuid[])",
                inserted_consent_memory_ids,
            )

        if original_hb_interval is not None:
            # Phase 7 (ReduceScopeCreep): use unified config
            await conn.execute(
                "SELECT set_config('heartbeat.heartbeat_interval_minutes', $1::jsonb)",
                original_hb_interval,
            )
        if original_hb_paused is not None:
            await conn.execute(
                "UPDATE heartbeat_state SET is_paused = $1 WHERE id = 1",
                bool(original_hb_paused),
            )
        if original_init_stage is not None:
            await conn.execute(
                """
                UPDATE heartbeat_state
                SET init_stage = $1::init_stage,
                    init_data = $2::jsonb,
                    init_started_at = $3,
                    init_completed_at = $4,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = 1
                """,
                original_init_stage,
                json.dumps(original_init_data or {}),
                original_init_started_at,
                original_init_completed_at,
            )


@pytest.fixture(scope="module", autouse=True)
async def apply_repo_migrations(db_pool):
    """Apply the forward-delta migrations (db/migrations/) after the baseline replay,
    so each temp test DB equals a real deployment: current = baseline + migrations.
    Uses the same runner that ships in production; additive + idempotent."""
    from core.migrations import apply_pending_migrations

    async with db_pool.acquire() as conn:
        await apply_pending_migrations(conn)


@pytest.fixture(autouse=True)
async def setup_db(db_pool):
    """Setup the database before each test."""
    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute("SET search_path = public, ag_catalog;")
    yield


@pytest.fixture(scope="module")
async def ensure_embedding_service(db_pool):
    """
    Ensure embedding service is available.

    This fixture retries for a short window so tests don't flake while Docker
    is still initializing. If the service never becomes healthy, fail fast
    with a clear timeout error (no skipping).
    """
    async with db_pool.acquire() as conn:
        wait_seconds = int(os.getenv("EMBEDDINGS_WAIT_SECONDS", "30"))
        retrying = AsyncRetrying(
            stop=stop_after_delay(wait_seconds),
            wait=wait_fixed(1),
            retry=(retry_if_result(lambda ok: not ok) | retry_if_exception_type(Exception)),
            reraise=False,
        )

        try:
            ok = await retrying(conn.fetchval, "SELECT check_embedding_service_health()")
            assert ok is True
            return True
        except RetryError as exc:
            last_exc = exc.last_attempt.exception()
            last_res = None
            try:
                last_res = exc.last_attempt.result()
            except Exception:
                pass
            pytest.fail(
                f"Embedding service not available after {wait_seconds}s."
                f"last_exception={last_exc!r} last_result={last_res!r}"
            )
