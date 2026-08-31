from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)


def db_dsn_from_env(instance: str | None = None) -> str:
    """Build DSN, optionally for a specific named instance.

    Args:
        instance: Optional instance name. If provided, looks up DSN from registry.

    Returns:
        PostgreSQL DSN string.
    """
    if instance:
        from core.instance import InstanceRegistry
        return InstanceRegistry().dsn_for(instance)

    # Check for HEXIS_INSTANCE env var
    from_env = os.getenv("HEXIS_INSTANCE")
    if from_env:
        try:
            from core.instance import InstanceRegistry
            registry = InstanceRegistry()
            if registry.exists(from_env):
                return registry.dsn_for(from_env)
        except Exception:
            pass

    # Check for current instance in registry
    try:
        from core.instance import InstanceRegistry
        registry = InstanceRegistry()
        current = registry.get_current()
        if current and registry.exists(current):
            return registry.dsn_for(current)
    except Exception:
        pass

    # Fall back to env vars
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "43815"))
    database = os.getenv("POSTGRES_DB", "hexis_memory")
    user = os.getenv("POSTGRES_USER", "hexis_user")
    password = os.getenv("POSTGRES_PASSWORD", "hexis_password")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def resolve_instance() -> str | None:
    """Get current instance from HEXIS_INSTANCE env or registry."""
    from_env = os.getenv("HEXIS_INSTANCE")
    if from_env:
        return from_env
    try:
        from core.instance import InstanceRegistry
        return InstanceRegistry().get_current()
    except Exception:
        return None


def _resolve_wait_seconds(wait_seconds: int | None) -> int:
    if wait_seconds is None:
        return int(os.getenv("POSTGRES_WAIT_SECONDS", "30"))
    return int(wait_seconds)


async def _connect_with_retry(dsn: str, *, wait_seconds: int = 30) -> asyncpg.Connection:
    deadline = time.monotonic() + wait_seconds
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return await asyncpg.connect(dsn, ssl=False, command_timeout=60.0)
        except Exception as exc:
            last_err = exc
            await asyncio.sleep(1)
    raise TimeoutError(f"Failed to connect to Postgres after {wait_seconds}s: {last_err!r}")


def pool_sizes_from_env(default_min: int = 1, default_max: int = 5) -> tuple[int, int]:
    """Read pool size overrides from environment, falling back to provided defaults."""
    min_size = int(os.getenv("HEXIS_POOL_MIN_SIZE", str(default_min)))
    max_size = int(os.getenv("HEXIS_POOL_MAX_SIZE", str(default_max)))
    return min_size, max_size


async def get_agent_status(dsn: str | None = None) -> dict[str, Any]:
    """Thin wrapper: the DB owns the composite status and the AND-policy for
    'configured' (get_agent_status(), db/65)."""
    dsn = dsn or db_dsn_from_env()
    conn = await _connect_with_retry(dsn, wait_seconds=int(os.getenv("POSTGRES_WAIT_SECONDS", "30")))
    try:
        raw = await conn.fetchval("SELECT get_agent_status()")
        status = json.loads(raw) if isinstance(raw, str) else (raw or {})
        return dict(status)
    finally:
        await conn.close()


async def get_init_defaults(dsn: str | None = None, wait_seconds: int | None = None) -> dict[str, Any]:
    """Get default configuration values from unified config table.

    Phase 7 (ReduceScopeCreep): Uses unified config table instead of legacy heartbeat_config/maintenance_config.
    """
    dsn = dsn or db_dsn_from_env()
    conn = await _connect_with_retry(dsn, wait_seconds=_resolve_wait_seconds(wait_seconds))
    try:
        # Phase 7: Use unified config table with namespaced keys
        rows = await conn.fetch(
            """
            SELECT key, value
            FROM get_config_by_prefixes($1::text[])
            """,
            ["heartbeat.", "maintenance."],
        )
        cfg = {r["key"]: r["value"] for r in rows}

        def get_float(key: str) -> float:
            val = cfg.get(key)
            if val is None:
                raise RuntimeError(f"Missing config default for {key}; run `hexis migrate`.")
            # JSONB values may be strings or numbers
            if isinstance(val, (int, float)):
                return float(val)
            try:
                import json
                return float(json.loads(val) if isinstance(val, str) else val)
            except Exception as exc:
                raise RuntimeError(f"Invalid config default for {key}; run `hexis migrate`.") from exc

        return {
            "heartbeat_interval_minutes": int(get_float("heartbeat.heartbeat_interval_minutes")),
            "max_energy": get_float("heartbeat.max_energy"),
            "base_regeneration": get_float("heartbeat.base_regeneration"),
            "max_active_goals": int(get_float("heartbeat.max_active_goals")),
            "maintenance_interval_seconds": int(get_float("maintenance.maintenance_interval_seconds")),
            "subconscious_interval_seconds": int(get_float("maintenance.subconscious_interval_seconds")),
        }
    finally:
        await conn.close()


async def apply_migrations(dsn: str | None = None, wait_seconds: int | None = None) -> list[str]:
    """Apply any pending schema migrations to the active database (idempotent,
    advisory-locked, non-destructive). Returns the versions applied this call."""
    from core.migrations import apply_pending_migrations

    dsn = dsn or db_dsn_from_env()
    conn = await _connect_with_retry(dsn, wait_seconds=_resolve_wait_seconds(wait_seconds))
    try:
        return await apply_pending_migrations(conn)
    finally:
        await conn.close()


async def ensure_schema_has_config(dsn: str | None = None, wait_seconds: int | None = None) -> None:
    dsn = dsn or db_dsn_from_env()
    conn = await _connect_with_retry(dsn, wait_seconds=_resolve_wait_seconds(wait_seconds))
    try:
        ok = await conn.fetchval("SELECT to_regclass('public.config') IS NOT NULL")
        if ok:
            return
        # Try to bring the schema current before giving up — a change may just
        # need migrating in, not a wipe.
        try:
            from core.migrations import apply_pending_migrations
            await apply_pending_migrations(conn)
            ok = await conn.fetchval("SELECT to_regclass('public.config') IS NOT NULL")
        except Exception:
            ok = False
        if not ok:
            raise RuntimeError(
                "Database schema is missing the `config` table. "
                "Bring it up to date without losing data: `hexis migrate` "
                "(or `hexis upgrade`). Use `hexis reset` only to deliberately wipe."
            )
    finally:
        await conn.close()


async def bootstrap_identity(dsn: str | None = None, wait_seconds: int | None = None) -> str | None:
    dsn = dsn or db_dsn_from_env()
    conn = await _connect_with_retry(dsn, wait_seconds=_resolve_wait_seconds(wait_seconds))
    try:
        try:
            await conn.fetchval("SELECT initialize_personality(NULL)")
            await conn.fetchval("SELECT initialize_core_values(NULL)")
            await conn.fetchval("SELECT initialize_worldview(NULL)")
        except Exception as exc:
            return str(exc)
        return None
    finally:
        await conn.close()


async def get_config(dsn: str | None, key: str) -> Any:
    dsn = dsn or db_dsn_from_env()
    conn = await _connect_with_retry(dsn, wait_seconds=int(os.getenv("POSTGRES_WAIT_SECONDS", "30")))
    try:
        value = await conn.fetchval("SELECT get_config($1)", key)
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value
    finally:
        await conn.close()


async def get_llm_config(dsn: str | None, key: str) -> dict[str, Any]:
    value = await get_config(dsn, key)
    if isinstance(value, dict):
        return value
    return {}


async def get_agent_profile_context(dsn: str | None = None, *, pool: Any = None) -> dict[str, Any]:
    if pool is not None:
        async with pool.acquire() as conn:
            value = await conn.fetchval("SELECT get_agent_profile_context()")
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except Exception:
                    return {}
            return value or {}
    dsn = dsn or db_dsn_from_env()
    conn = await _connect_with_retry(dsn, wait_seconds=int(os.getenv("POSTGRES_WAIT_SECONDS", "30")))
    try:
        value = await conn.fetchval("SELECT get_agent_profile_context()")
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return {}
        return value or {}
    finally:
        await conn.close()


async def apply_agent_config(
    *,
    dsn: str | None = None,
    wait_seconds: int | None = None,
    heartbeat_interval_minutes: int,
    maintenance_interval_seconds: int,
    subconscious_interval_seconds: int | None = None,
    max_energy: float,
    base_regeneration: float,
    max_active_goals: int,
    objectives: list[str],
    guardrails: list[str],
    initial_message: str,
    tools: list[str],
    llm_heartbeat: dict[str, Any],
    llm_chat: dict[str, Any],
    llm_subconscious: dict[str, Any] | None = None,
    contact_channels: list[str],
    contact_destinations: dict[str, str],
    enable_autonomy: bool,
    enable_maintenance: bool,
    enable_subconscious: bool | None = None,
    mark_configured: bool,
) -> None:
    dsn = dsn or db_dsn_from_env()
    conn = await _connect_with_retry(dsn, wait_seconds=_resolve_wait_seconds(wait_seconds))
    try:
        # Thin wrapper: the DB applies the whole configuration atomically
        # (apply_agent_config, db/65). This was ~18 sequential set_config
        # calls threaded through a client transaction.
        config = {
            "heartbeat_interval_minutes": heartbeat_interval_minutes,
            "maintenance_interval_seconds": maintenance_interval_seconds,
            "subconscious_interval_seconds": subconscious_interval_seconds,
            "max_energy": max_energy,
            "base_regeneration": base_regeneration,
            "max_active_goals": max_active_goals,
            "objectives": objectives,
            "guardrails": guardrails,
            "initial_message": initial_message,
            "tools": tools,
            "llm_heartbeat": llm_heartbeat,
            "llm_chat": llm_chat,
            "llm_subconscious": llm_subconscious,
            "contact_channels": contact_channels,
            "contact_destinations": contact_destinations,
            "enable_autonomy": enable_autonomy,
            "enable_maintenance": enable_maintenance,
            "enable_subconscious": enable_subconscious,
            "mark_configured": mark_configured,
        }
        await conn.execute(
            "SELECT apply_agent_config($1::jsonb)", json.dumps(config)
        )
    finally:
        await conn.close()


async def save_init_profile(
    *,
    dsn: str | None = None,
    mode: str,
    agent_name: str,
    agent_pronouns: str,
    agent_voice: str,
    personality_description: str,
    user_name: str,
    relationship_type: str,
    purpose: str,
    values: list[str],
    boundaries: list[str],
    autonomy_level: str,
) -> None:
    dsn = dsn or db_dsn_from_env()
    profile = {
        "mode": mode,
        "agent": {
            "name": agent_name,
            "pronouns": agent_pronouns,
            "voice": agent_voice,
            "personality": personality_description,
        },
        "user": {"name": user_name},
        "relationship": {
            "type": relationship_type,
            "purpose": purpose,
        },
        "values": values,
        "boundaries": boundaries,
        "autonomy_level": autonomy_level,
    }
    conn = await _connect_with_retry(dsn, wait_seconds=int(os.getenv("POSTGRES_WAIT_SECONDS", "30")))
    try:
        await conn.execute("SELECT set_config('agent.mode', $1::jsonb)", json.dumps(mode))
        await conn.execute("SELECT set_config('agent.init_profile', $1::jsonb)", json.dumps(profile))
    finally:
        await conn.close()


async def set_agent_configured(dsn: str | None, *, configured: bool) -> None:
    dsn = dsn or db_dsn_from_env()
    conn = await _connect_with_retry(dsn, wait_seconds=int(os.getenv("POSTGRES_WAIT_SECONDS", "30")))
    try:
        if configured:
            await conn.execute("SELECT set_config('agent.is_configured', 'true'::jsonb)")
        else:
            await conn.execute("SELECT delete_config_key('agent.is_configured')")
    finally:
        await conn.close()


def get_agent_status_sync(dsn: str | None = None) -> dict[str, Any]:
    from core.sync_utils import run_sync

    return run_sync(get_agent_status(dsn))


def get_init_defaults_sync(dsn: str | None = None) -> dict[str, Any]:
    from core.sync_utils import run_sync

    return run_sync(get_init_defaults(dsn))


def get_config_sync(dsn: str | None, key: str) -> Any:
    from core.sync_utils import run_sync

    return run_sync(get_config(dsn, key))


def get_llm_config_sync(dsn: str | None, key: str) -> dict[str, Any]:
    from core.sync_utils import run_sync

    return run_sync(get_llm_config(dsn, key))


def get_agent_profile_context_sync(dsn: str | None = None) -> dict[str, Any]:
    from core.sync_utils import run_sync

    return run_sync(get_agent_profile_context(dsn))


def apply_agent_config_sync(**kwargs: Any) -> None:
    from core.sync_utils import run_sync

    return run_sync(apply_agent_config(**kwargs))


def save_init_profile_sync(**kwargs: Any) -> None:
    from core.sync_utils import run_sync

    return run_sync(save_init_profile(**kwargs))


def set_agent_configured_sync(dsn: str | None, *, configured: bool) -> None:
    from core.sync_utils import run_sync

    return run_sync(set_agent_configured(dsn, configured=configured))


def read_build_id() -> str | None:
    """The image build stamp (#93): env override first, then the stamp file
    the Dockerfile writes after its code layers."""
    stamp = (os.getenv("HEXIS_BUILD_ID") or "").strip()
    if stamp:
        return stamp
    try:
        return Path("/app/.build_id").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def worker_code_stamp_metadata() -> dict[str, str]:
    """Build/migration stamp for a worker's registration metadata (#113):
    what code this process actually is, so `hexis status`/`hexis doctor` can
    flag a running worker whose bundled db/migrations/ predates migrations
    someone else already applied to the database it shares. Shared by every
    worker kind (heartbeat/maintenance, channel) so the stamp means the same
    thing everywhere. Never raises -- a worker must never fail to start
    over a diagnostic stamp."""
    metadata: dict[str, str] = {}
    try:
        build_id = read_build_id()
        if build_id:
            metadata["build_id"] = build_id
    except Exception:
        pass
    try:
        from core.migrations import latest_bundled_migration_version

        bundled = latest_bundled_migration_version()
        if bundled:
            metadata["bundled_latest_migration"] = bundled
    except Exception:
        pass
    return metadata


async def record_build_change(conn, service: str) -> None:
    """Journal a 'my running code changed' entry (#93) when this service's
    build stamp differs from the last one it ran under. Advisory: never
    blocks startup."""
    build_id = read_build_id()
    if not build_id:
        return
    try:
        previous = await conn.fetchval(
            "SELECT get_state($1)->>'build_id'", f"runtime_build.{service}"
        )
        if previous == build_id:
            return
        await conn.execute(
            "SELECT set_state($1, $2::jsonb)",
            f"runtime_build.{service}",
            json.dumps({"build_id": build_id}),
        )
        summary = (
            f"My running code changed: {service} now runs build {build_id}"
            + (f" (was {previous})" if previous else " (first recorded build)")
        )
        await conn.execute(
            "SELECT record_change('code', $1, $2::jsonb)",
            summary,
            json.dumps({"service": service, "build_id": build_id, "previous": previous}),
        )
    except Exception:
        logger.debug("build-change journaling unavailable", exc_info=True)
