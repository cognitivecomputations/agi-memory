import json
import os
import uuid
from time import perf_counter

TEST_SESSION_ID = str(uuid.uuid4())[:8]


def get_test_identifier(test_name: str) -> str:
    """Generate a unique identifier for test data."""
    return f"{test_name}_{TEST_SESSION_ID}_{uuid.uuid4().hex[:8]}"


def _db_dsn(db_name: str | None = None) -> str:
    db_host = os.getenv("POSTGRES_HOST", "localhost")
    db_port = os.getenv("POSTGRES_PORT", "43815")
    db_name = db_name or os.getenv("POSTGRES_DB", "hexis_memory")
    db_user = os.getenv("POSTGRES_USER", "hexis_user")
    db_password = os.getenv("POSTGRES_PASSWORD", "hexis_password")
    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


def _coerce_json(val):
    if isinstance(val, str):
        return json.loads(val)
    return val


async def timed_db_call(label: str, coro, conn=None, track_embeddings: bool = False):
    """Time an awaited DB call and optionally report embedding_cache delta."""
    start_count = None
    if track_embeddings and conn is not None:
        start_count = await conn.fetchval("SELECT COUNT(*) FROM embedding_cache")
    start = perf_counter()
    result = await coro
    duration = perf_counter() - start
    if track_embeddings and conn is not None:
        end_count = await conn.fetchval("SELECT COUNT(*) FROM embedding_cache")
        delta = end_count - (start_count or 0)
        print(f"[timing] {label}: {duration:.3f}s (embeddings +{delta})")
    else:
        print(f"[timing] {label}: {duration:.3f}s")
    return result


async def _set_embedding_retry_config(
    conn,
    retry_seconds: int,
    retry_interval_seconds: float,
):
    # Save original values from unified config table
    original_retry_seconds = await conn.fetchval(
        "SELECT value #>> '{}' FROM config WHERE key = 'embedding.retry_seconds'"
    )
    original_retry_interval_seconds = await conn.fetchval(
        "SELECT value #>> '{}' FROM config WHERE key = 'embedding.retry_interval_seconds'"
    )
    # Update unified config table
    await conn.execute(
        """
        INSERT INTO config (key, value, description, updated_at)
        VALUES ('embedding.retry_seconds', $1::jsonb, 'Total seconds to retry embedding requests', CURRENT_TIMESTAMP)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP
        """,
        str(retry_seconds),
    )
    await conn.execute(
        """
        INSERT INTO config (key, value, description, updated_at)
        VALUES ('embedding.retry_interval_seconds', $1::jsonb, 'Seconds between retry attempts', CURRENT_TIMESTAMP)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP
        """,
        str(retry_interval_seconds),
    )
    # Phase 7 (ReduceScopeCreep): embedding_config removed - using unified config only
    return original_retry_seconds, original_retry_interval_seconds


async def _restore_embedding_retry_config(
    conn,
    original_retry_seconds,
    original_retry_interval_seconds,
):
    # Restore unified config table
    if original_retry_seconds is None:
        await conn.execute("DELETE FROM config WHERE key = 'embedding.retry_seconds'")
    else:
        await conn.execute(
            "UPDATE config SET value = $1::jsonb, updated_at = CURRENT_TIMESTAMP WHERE key = 'embedding.retry_seconds'",
            original_retry_seconds,
        )
    if original_retry_interval_seconds is None:
        await conn.execute("DELETE FROM config WHERE key = 'embedding.retry_interval_seconds'")
    else:
        await conn.execute(
            "UPDATE config SET value = $1::jsonb, updated_at = CURRENT_TIMESTAMP WHERE key = 'embedding.retry_interval_seconds'",
            original_retry_interval_seconds,
        )
    # Phase 7 (ReduceScopeCreep): embedding_config removed - using unified config only


async def embed_pending_memories(conn, max_batches: int = 20) -> int:
    """Run the durable-memory embedding pass until the queue is drained.

    Memory writes leave embedding_status='pending' (async embedding, 0129);
    the maintenance worker embeds them later. Tests that assert on vectors or
    vector recall run the same pass the worker runs, on their own connection —
    so an in-transaction get_embedding stub applies to it too.
    """
    from services.memory_embeddings import run_memory_embed_step

    total = 0
    for _ in range(max_batches):
        result = await run_memory_embed_step(conn)
        if result.get("skipped"):
            break
        total += int(result.get("embedded", 0))
    return total
