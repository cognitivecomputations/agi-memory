from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from typing import Any

from core.agent_api import _connect_with_retry, db_dsn_from_env

logger = logging.getLogger(__name__)


def dashboard_https_check(*, timeout_seconds: float = 3.0) -> dict[str, Any]:
    """Report whether the dashboard has a browser-trusted HTTPS route.

    An explicit public URL is authoritative. Otherwise, discover a running
    Tailscale node and its MagicDNS name without changing Serve state.
    """

    try:
        ui_port = int(os.getenv("HEXIS_UI_PORT", "3477"))
    except ValueError:
        ui_port = 3477
    local_target = f"http://127.0.0.1:{ui_port}"
    explicit = str(
        os.getenv("HEXIS_UI_PUBLIC_URL") or os.getenv("HEXIS_PUBLIC_URL") or ""
    ).strip()
    if explicit:
        if not explicit.lower().startswith("https://"):
            return {
                "label": "Dashboard HTTPS",
                "status": "WARN",
                "detail": (
                    f"{explicit} is not HTTPS — phone app install, push, and microphone "
                    "will be unavailable. Set HEXIS_UI_PUBLIC_URL to the private HTTPS "
                    "address after following docs/operations/secure-remote-access.md."
                ),
            }
        reachable, detail = _probe_dashboard_https(explicit, timeout_seconds)
        return {
            "label": "Dashboard HTTPS",
            "status": "OK" if reachable else "WARN",
            "detail": (
                f"reachable at {explicit} ({detail})"
                if reachable
                else f"configured as {explicit}, but not reachable ({detail}); check the reverse proxy or tailnet, then retry"
            ),
        }

    tailscale = shutil.which("tailscale")
    if not tailscale:
        return {
            "label": "Dashboard HTTPS",
            "status": "WARN",
            "detail": (
                "local dashboard only — localhost works here, but a phone needs private "
                "HTTPS for app install, push, and microphone. Follow "
                "docs/operations/secure-remote-access.md."
            ),
        }
    try:
        status_proc = subprocess.run(
            [tailscale, "status", "--json"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        status = json.loads(status_proc.stdout or "{}")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return {
            "label": "Dashboard HTTPS",
            "status": "WARN",
            "detail": f"Tailscale status is unavailable ({exc}); start Tailscale, then retry",
        }
    self_status = status.get("Self") if isinstance(status, dict) else None
    dns_name = (
        str(self_status.get("DNSName") or "").strip().rstrip(".")
        if isinstance(self_status, dict)
        else ""
    )
    if not dns_name:
        return {
            "label": "Dashboard HTTPS",
            "status": "WARN",
            "detail": "Tailscale is installed but not connected; sign in on this host and phone, then retry",
        }
    url = f"https://{dns_name}"
    reachable, detail = _probe_dashboard_https(url, timeout_seconds)
    if reachable:
        return {
            "label": "Dashboard HTTPS",
            "status": "OK",
            "detail": f"reachable over the tailnet at {url} ({detail})",
        }
    try:
        from core.tunnel import tunnel_status

        route_status = tunnel_status(
            ui_port=ui_port,
            bind_address=os.getenv("HEXIS_BIND_ADDRESS", "127.0.0.1"),
            timeout_seconds=timeout_seconds,
            probe_local=False,
        )
        serve_configured = bool(route_status.get("target_matches"))
    except RuntimeError:
        serve_configured = False
    if serve_configured:
        detail_text = (
            f"Tailscale Serve is configured for {url}, but the HTTPS probe failed ({detail}). "
            f"Run `tailscale serve status`, verify the target is {local_target}, "
            "and confirm the UI is running."
        )
    else:
        detail_text = (
            f"tailnet connected as {dns_name}, but HTTPS is not serving the dashboard. "
            "After enabling MagicDNS and HTTPS certificates in Tailscale, run "
            f"`hexis tunnel start` (local target {local_target}), then open "
            f"{url} from the phone."
        )
    return {"label": "Dashboard HTTPS", "status": "WARN", "detail": detail_text}


def _probe_dashboard_https(url: str, timeout_seconds: float) -> tuple[bool, str]:
    target = f"{url.rstrip('/')}/api/status"
    request = urllib.request.Request(target, headers={"User-Agent": "Hexis doctor"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return True, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        # A verified TLS handshake plus an HTTP response proves reachability;
        # the status itself still explains any application-layer issue.
        return True, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def embedding_service_diagnosis(
    url: str | None, model: str | None = None
) -> tuple[str, list[str]]:
    """Identify the embedding backend from its URL and return (name, fix_steps)."""
    url = (url or "").lower()
    if ":42666" in url:
        return "embeddinggemma local sidecar", [
            "Start it: embeddinggemma",
            "Or run: hexis up",
            "If Hexis started it, check: ~/.hexis/embeddinggemma.log",
        ]
    if ":11434" in url:
        return "legacy embeddinggemma sidecar configuration", [
            "Hexis now defaults to the published embeddinggemma binary on port 42666.",
            "Remove the old EMBEDDING_SERVICE_URL override or set it to http://host.docker.internal:42666/api/embed",
            "Install the published binary: curl -fsSL https://raw.githubusercontent.com/QuixiAI/embeddinggemma.c/main/install.sh | sh",
        ]
    if "embeddings:" in url or "text-embeddings" in url:
        return "TEI (Text Embeddings Inference)", [
            "Uncomment the 'embeddings' service in docker-compose.yml",
            "Then run: docker compose up -d",
        ]
    if "api.openai.com" in url:
        return "OpenAI API", [
            "Check that OPENAI_API_KEY is set in your .env",
            "Verify your API key is valid and has embeddings access",
        ]
    if "localhost" in url or "127.0.0.1" in url or "host.docker.internal" in url:
        return "local embedding service", [
            f"Ensure the service at {url} is running",
            "Or set EMBEDDING_SERVICE_URL in .env to a different endpoint",
        ]
    return "embedding service", [
        f"Ensure the service at {url} is reachable from the DB container",
        "Or set EMBEDDING_SERVICE_URL in .env to a different endpoint",
    ]


def _coerce_json_value(val: Any) -> Any:
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return val
        try:
            return json.loads(s)
        except Exception:
            return val
    return val


def _json_ready(val: Any) -> Any:
    if val is None or isinstance(val, (int, float, bool)):
        return val
    if isinstance(val, str):
        s = val.strip()
        if s[:1] in ("{", "["):
            try:
                return _json_ready(json.loads(s))
            except Exception:
                return val
        return val
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, dict):
        return {str(k): _json_ready(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_json_ready(v) for v in val]
    try:
        json.dumps(val)
        return val
    except TypeError:
        return str(val)


def _record_to_dict(row: Any) -> dict[str, Any]:
    return {str(k): _json_ready(row[k]) for k in row.keys()}


_CURRENT_WORKERS_SQL = """
SELECT *
FROM (
    SELECT
        worker_runtime_status.*,
        row_number() OVER (PARTITION BY mode ORDER BY last_seen_at DESC) AS mode_rank
    FROM worker_runtime_status
    WHERE status NOT IN ('stopped', 'terminated')
) ranked
WHERE mode_rank = 1
ORDER BY mode
"""


async def status_payload(
    dsn: str | None = None,
    *,
    wait_seconds: int = 30,
    include_embedding_health: bool = True,
) -> dict[str, Any]:
    dsn = dsn or db_dsn_from_env()
    conn = await _connect_with_retry(dsn, wait_seconds=wait_seconds)
    try:
        payload: dict[str, Any] = {"dsn": dsn}
        payload["db_time"] = str(await conn.fetchval("SELECT now()"))

        payload["agent_configured"] = bool(
            await conn.fetchval("SELECT is_agent_configured()")
        )
        payload["heartbeat_paused"] = bool(
            await conn.fetchval("SELECT is_paused FROM heartbeat_state WHERE id = 1")
        )
        payload["should_run_heartbeat"] = bool(
            await conn.fetchval("SELECT should_run_heartbeat()")
        )
        try:
            payload["maintenance_paused"] = bool(
                await conn.fetchval(
                    "SELECT is_paused FROM maintenance_state WHERE id = 1"
                )
            )
            payload["should_run_maintenance"] = bool(
                await conn.fetchval("SELECT should_run_maintenance()")
            )
        except Exception:
            payload["maintenance_paused"] = None
            payload["should_run_maintenance"] = None

        payload["pending_external_calls"] = 0
        try:
            payload["pending_outbox_messages"] = int(
                await conn.fetchval(
                    "SELECT count(*) FROM outbox_messages WHERE status IN ('pending', 'publishing')"
                )
            )
        except Exception:
            payload["pending_outbox_messages"] = 0

        payload["embedding_service_url"] = await conn.fetchval(
            "SELECT get_config_text('embedding.service_url')"
        )
        payload["embedding_dimension"] = int(
            await conn.fetchval("SELECT embedding_dimension()")
        )

        try:
            worker_rows = await conn.fetch(_CURRENT_WORKERS_SQL)
            workers = [_record_to_dict(r) for r in worker_rows]
            for worker in workers:
                worker.pop("mode_rank", None)
            payload["workers"] = workers
            task_rows = await conn.fetch("SELECT * FROM worker_task_status")
            payload["worker_tasks"] = [_record_to_dict(r) for r in task_rows]
        except Exception as exc:
            payload["workers"] = []
            payload["worker_tasks"] = []
            payload["worker_status_error"] = str(exc)

        if include_embedding_health:
            try:
                payload["embedding_service_healthy"] = bool(
                    await conn.fetchval("SELECT check_embedding_service_health()")
                )
            except Exception as exc:
                payload["embedding_service_healthy"] = False
                payload["embedding_service_error"] = repr(exc)

        return payload
    finally:
        await conn.close()


async def config_rows(
    dsn: str | None = None, *, wait_seconds: int = 30
) -> dict[str, Any]:
    dsn = dsn or db_dsn_from_env()
    conn = await _connect_with_retry(dsn, wait_seconds=wait_seconds)
    try:
        rows = await conn.fetch("SELECT key, value FROM config ORDER BY key")
        out: dict[str, Any] = {}
        for r in rows:
            out[str(r["key"])] = _coerce_json_value(r["value"])
        return out
    finally:
        await conn.close()


async def config_validate(
    dsn: str | None = None, *, wait_seconds: int = 30
) -> tuple[list[str], list[str]]:
    dsn = dsn or db_dsn_from_env()
    conn = await _connect_with_retry(dsn, wait_seconds=wait_seconds)
    try:
        errors: list[str] = []
        warnings: list[str] = []

        rows = await conn.fetch("SELECT key, value FROM config ORDER BY key")
        cfg: dict[str, Any] = {
            str(r["key"]): _coerce_json_value(r["value"]) for r in rows
        }
        required_keys = [
            "agent.is_configured",
            "agent.objectives",
            "llm.heartbeat",
            "llm.chat",
        ]
        for key in required_keys:
            if key not in cfg:
                errors.append(f"Missing config key: {key}")

        is_conf = cfg.get("agent.is_configured")
        if is_conf is not True:
            if is_conf == "true":
                is_conf = True
        if is_conf is not True:
            errors.append("agent.is_configured is not true (run `hexis init`).")

        objectives = cfg.get("agent.objectives")
        if not isinstance(objectives, list) or not objectives:
            errors.append(
                "agent.objectives must be a non-empty array (run `hexis init`)."
            )

        def _validate_llm(name: str) -> None:
            val = cfg.get(name)
            if not isinstance(val, dict):
                errors.append(f"{name} must be an object (run `hexis init`).")
                return
            provider = str(val.get("provider") or "").strip().lower()
            model = str(val.get("model") or "").strip()
            endpoint = str(val.get("endpoint") or "").strip()
            api_key_env = str(val.get("api_key_env") or "").strip()

            if not provider:
                errors.append(f"{name}.provider is required")
            if not model:
                warnings.append(f"{name}.model is empty (will rely on worker defaults)")

            if provider == "openai-codex":
                oauth = cfg.get("oauth.openai_codex")
                if (
                    not isinstance(oauth, dict)
                    or not oauth.get("refresh")
                    or not oauth.get("access")
                ):
                    errors.append(
                        "OpenAI Codex OAuth is not configured (missing oauth.openai_codex). "
                        "Run: `hexis auth openai-codex login`"
                    )
                return

            # OAuth providers with stored credentials
            _oauth_providers: dict[str, str] = {
                "chutes": "oauth.chutes",
                "github-copilot": "oauth.github_copilot",
                "qwen-portal": "oauth.qwen_portal",
                "minimax-portal": "oauth.minimax_portal",
                "google-gemini-cli": "oauth.google_gemini_cli",
                "google-antigravity": "oauth.google_antigravity",
            }
            oauth_key = _oauth_providers.get(provider)
            if oauth_key:
                oauth = cfg.get(oauth_key)
                if not isinstance(oauth, dict) or not oauth.get("access"):
                    errors.append(
                        f"{provider} is not configured (missing {oauth_key}). "
                        f"Run: `hexis auth {provider} login`"
                    )
                return

            # Anthropic with setup-token fallback
            if (
                provider == "anthropic"
                and not api_key_env
                and not os.getenv("ANTHROPIC_API_KEY")
            ):
                token_cfg = cfg.get("token.anthropic_setup_token")
                if isinstance(token_cfg, dict) and token_cfg.get("token"):
                    return  # setup-token is configured
                warnings.append(
                    f"{name}: no ANTHROPIC_API_KEY env var and no setup-token configured. "
                    "Run: `hexis auth anthropic setup-token` or set api_key_env"
                )

            if provider in {
                "openai",
                "anthropic",
                "openai_compatible",
                "grok",
                "gemini",
            }:
                if api_key_env:
                    if os.getenv(api_key_env) is None:
                        errors.append(
                            f"{name}.api_key_env={api_key_env} is not set in environment"
                        )
                else:
                    if not endpoint or (
                        "localhost" not in endpoint and "127.0.0.1" not in endpoint
                    ):
                        warnings.append(
                            f"{name}.api_key_env not set (LLM calls may fail)"
                        )

        _validate_llm("llm.heartbeat")
        _validate_llm("llm.chat")
        if "llm.subconscious" in cfg:
            _validate_llm("llm.subconscious")

        interval = await conn.fetchval(
            "SELECT get_config_float('heartbeat.heartbeat_interval_minutes')"
        )
        if interval is None or float(interval) <= 0:
            errors.append("heartbeat.heartbeat_interval_minutes must be > 0")

        return errors, warnings
    finally:
        await conn.close()


async def demo(dsn: str | None = None, *, wait_seconds: int = 30) -> dict[str, Any]:
    """Run the rollback-only end-to-end capability proof."""
    from core.capability_maturity import run_alive_demo

    dsn = dsn or db_dsn_from_env()
    conn = await _connect_with_retry(dsn, wait_seconds=wait_seconds)
    try:
        return await run_alive_demo(conn)
    finally:
        await conn.close()


async def maturity_scorecard(
    dsn: str | None = None, *, wait_seconds: int = 30
) -> dict[str, Any]:
    from core.capability_maturity import capability_maturity_scorecard

    dsn = dsn or db_dsn_from_env()
    conn = await _connect_with_retry(dsn, wait_seconds=wait_seconds)
    try:
        return await capability_maturity_scorecard(conn)
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# doctor -- comprehensive health check
# ---------------------------------------------------------------------------


async def _check(
    label: str,
    coro,
    *,
    warn_on_false: str | None = None,
) -> dict[str, Any]:
    """Run a single health check and return a result dict."""
    try:
        result = await coro
        if warn_on_false is not None and result is False:
            return {"label": label, "status": "WARN", "detail": warn_on_false}
        return {"label": label, "status": "OK", "detail": result}
    except Exception as exc:
        return {"label": label, "status": "FAIL", "detail": str(exc)}


async def doctor_payload(
    dsn: str | None = None,
    *,
    wait_seconds: int = 10,
    check_llm: bool = False,
) -> list[dict[str, Any]]:
    """
    Run comprehensive health checks and return a list of check results.

    Each result: {"label": str, "status": "OK"|"WARN"|"FAIL", "detail": Any}

    ``check_llm`` opt-in makes one real LLM call to verify provider/model/key
    (skipped by default so a health check never silently spends a token).
    """
    dsn = dsn or db_dsn_from_env()
    checks: list[dict[str, Any]] = []
    https_check = dashboard_https_check()
    try:
        from core.tunnel import remote_exposure_check

        ui_port = int(os.getenv("HEXIS_UI_PORT", "3477"))
        exposure_check = remote_exposure_check(
            ui_port=ui_port,
            bind_address=os.getenv("HEXIS_BIND_ADDRESS", "127.0.0.1"),
        )
    except (ValueError, RuntimeError) as exc:
        exposure_check = {
            "label": "Remote exposure",
            "status": "WARN",
            "detail": f"could not inspect remote exposure posture ({exc})",
        }

    # 1. PostgreSQL connectivity
    try:
        conn = await _connect_with_retry(dsn, wait_seconds=wait_seconds)
    except Exception as exc:
        low = str(exc).lower()
        if any(s in low for s in ("connect", "refused", "timed out", "timeout")):
            detail = "database not reachable — is the stack running? Run `hexis up`, then retry."
        else:
            detail = str(exc)
        checks.append({"label": "PostgreSQL", "status": "FAIL", "detail": detail})
        checks.append(exposure_check)
        checks.append(https_check)
        return checks  # Can't do anything else without DB

    try:
        db_time = await conn.fetchval("SELECT now()")
        checks.append(
            {"label": "PostgreSQL", "status": "OK", "detail": f"connected ({db_time})"}
        )
        checks.append(exposure_check)
        checks.append(https_check)

        # 2. Embeddings service
        try:
            emb_url = await conn.fetchval(
                "SELECT current_setting('app.embedding_service_url', true)"
            )
            emb_model = await conn.fetchval(
                "SELECT current_setting('app.embedding_model_id', true)"
            )
            healthy = await conn.fetchval("SELECT check_embedding_service_health()")
            if healthy:
                checks.append(
                    {
                        "label": "Embeddings",
                        "status": "OK",
                        "detail": f"healthy — {emb_model} via {emb_url}",
                    }
                )
            else:
                svc_name, steps = embedding_service_diagnosis(emb_url, emb_model)
                fix = "; ".join(steps)
                checks.append(
                    {
                        "label": "Embeddings",
                        "status": "FAIL",
                        "detail": (
                            f"Your config points to {svc_name} ({emb_url}) "
                            f"but it is not responding. {fix}"
                        ),
                    }
                )
        except Exception as exc:
            checks.append(
                {
                    "label": "Embeddings",
                    "status": "FAIL",
                    "detail": f"{exc}",
                }
            )

        # 2b. Canonical schema (#77): a stale ag_catalog twin silently outranks
        # its migrated public version for every runtime connection.
        try:
            shadow_count = await conn.fetchval("""
                SELECT count(*) FROM pg_proc p
                JOIN pg_namespace n ON n.oid = p.pronamespace
                WHERE n.nspname = 'ag_catalog'
                  AND EXISTS (
                      SELECT 1 FROM pg_proc p2
                      JOIN pg_namespace n2 ON n2.oid = p2.pronamespace
                      WHERE n2.nspname = 'public' AND p2.proname = p.proname
                  )
                """)
            if shadow_count:
                checks.append(
                    {
                        "label": "Schema canonical",
                        "status": "FAIL",
                        "detail": (
                            f"{shadow_count} stale ag_catalog function(s) shadow their public "
                            "versions — workers run old code. Fix: `hexis migrate` (the runner "
                            "evicts strays on every apply)."
                        ),
                    }
                )
            else:
                checks.append(
                    {
                        "label": "Schema canonical",
                        "status": "OK",
                        "detail": "no shadowed functions; public resolves first",
                    }
                )
        except Exception as exc:
            checks.append(
                {"label": "Schema canonical", "status": "WARN", "detail": str(exc)}
            )

        # 3. RabbitMQ (check config, not connectivity -- avoids dependency)
        try:
            rmq_url = await conn.fetchval(
                "SELECT value FROM config WHERE key = 'rabbitmq.url'"
            )
            if rmq_url:
                checks.append(
                    {"label": "RabbitMQ", "status": "OK", "detail": "configured"}
                )
            else:
                checks.append(
                    {
                        "label": "RabbitMQ",
                        "status": "WARN",
                        "detail": "not configured (outbox delivery disabled)",
                    }
                )
        except Exception:
            checks.append(
                {"label": "RabbitMQ", "status": "WARN", "detail": "not configured"}
            )

        # 4. Agent configured
        try:
            configured = bool(await conn.fetchval("SELECT is_agent_configured()"))
            if configured:
                profile = await conn.fetchval("SELECT get_agent_profile_context()")
                name = "unknown"
                if profile:
                    p = json.loads(profile) if isinstance(profile, str) else profile
                    name = (
                        p.get("name") or p.get("persona", {}).get("name") or "unnamed"
                    )
                checks.append(
                    {
                        "label": "Agent configured",
                        "status": "OK",
                        "detail": f'yes (identity: "{name}")',
                    }
                )
            else:
                checks.append(
                    {
                        "label": "Agent configured",
                        "status": "WARN",
                        "detail": "no (run 'hexis init')",
                    }
                )
        except Exception as exc:
            checks.append(
                {"label": "Agent configured", "status": "FAIL", "detail": str(exc)}
            )

        # 5. Consent — read the DB (the same store `hexis init` writes: consent_log +
        # config('agent.consent_status')). This is what the runtime actually consults.
        try:
            from core.consent import get_consent_status

            status = await get_consent_status(conn)
            if not status:
                status = await conn.fetchval(
                    "SELECT value #>> '{}' FROM config WHERE key = 'agent.consent_status'"
                )
            effective = (status or "").strip().lower()
            if effective == "consent":
                rows = await conn.fetch(
                    "SELECT DISTINCT provider, model FROM consent_log WHERE decision = 'consent'"
                )
                models = ", ".join(
                    f"{r['provider']}/{r['model']}" for r in rows if r["provider"]
                )
                checks.append(
                    {
                        "label": "Consent",
                        "status": "OK",
                        "detail": "granted" + (f" ({models})" if models else ""),
                    }
                )
            elif effective in ("decline", "abstain"):
                checks.append(
                    {
                        "label": "Consent",
                        "status": "WARN",
                        "detail": f"recorded as '{effective}' — run `hexis init` to (re)establish consent",
                    }
                )
            else:
                checks.append(
                    {
                        "label": "Consent",
                        "status": "WARN",
                        "detail": "not yet recorded — run `hexis init`",
                    }
                )
        except Exception:
            checks.append(
                {
                    "label": "Consent",
                    "status": "WARN",
                    "detail": "consent status unavailable",
                }
            )

        # 6. Heartbeat status
        try:
            hb_row = await conn.fetchrow(
                "SELECT current_energy, last_heartbeat_at, is_paused FROM heartbeat_state WHERE id = 1"
            )
            if hb_row:
                energy = hb_row["current_energy"]
                last_hb = hb_row["last_heartbeat_at"]
                paused = hb_row["is_paused"]
                max_energy = await conn.fetchval(
                    "SELECT get_config_int('heartbeat.max_energy')"
                )
                bank_capacity = await conn.fetchval("SELECT heartbeat_bank_capacity()")
                if max_energy is None:
                    raise RuntimeError(
                        "Missing heartbeat.max_energy default; run `hexis migrate`."
                    )
                if paused:
                    checks.append(
                        {
                            "label": "Heartbeat",
                            "status": "WARN",
                            "detail": f"paused (energy: {energy}/{bank_capacity}; reserve {max_energy})",
                        }
                    )
                elif last_hb:
                    from datetime import timezone as tz

                    now = datetime.now(tz.utc)
                    if hasattr(last_hb, "tzinfo") and last_hb.tzinfo is None:
                        last_hb = last_hb.replace(tzinfo=tz.utc)
                    ago = now - last_hb
                    ago_str = _format_timedelta(ago)
                    checks.append(
                        {
                            "label": "Heartbeat",
                            "status": "OK",
                            "detail": f"running (last: {ago_str} ago, energy: {energy}/{bank_capacity}; reserve {max_energy})",
                        }
                    )
                else:
                    checks.append(
                        {
                            "label": "Heartbeat",
                            "status": "WARN",
                            "detail": f"never run (energy: {energy}/{bank_capacity}; reserve {max_energy})",
                        }
                    )
            else:
                checks.append(
                    {
                        "label": "Heartbeat",
                        "status": "WARN",
                        "detail": "state not initialized",
                    }
                )
        except Exception as exc:
            checks.append({"label": "Heartbeat", "status": "FAIL", "detail": str(exc)})

        # 7. Worker processes
        try:
            worker_rows = await conn.fetch("""
                SELECT mode, status, last_seen_age_s, is_stale, current_task_type
                FROM worker_runtime_status
                WHERE status NOT IN ('stopped', 'terminated')
                ORDER BY last_seen_at DESC
                LIMIT 20
                """)
            latest_by_mode: dict[str, Any] = {}
            for row in worker_rows:
                mode = str(row["mode"])
                if mode not in latest_by_mode:
                    latest_by_mode[mode] = row

            expected_modes = ("heartbeat", "maintenance")
            missing = [m for m in expected_modes if m not in latest_by_mode]
            stale = [
                m
                for m, row in latest_by_mode.items()
                if row["status"] == "stale" or bool(row["is_stale"])
            ]
            if not worker_rows:
                checks.append(
                    {
                        "label": "Workers",
                        "status": "WARN",
                        "detail": (
                            "not running — the heartbeat and maintenance loops are "
                            "down; start them with `hexis up`"
                        ),
                    }
                )
            elif missing or stale:
                details = []
                if missing:
                    details.append("missing " + ", ".join(missing))
                if stale:
                    details.append("stale " + ", ".join(sorted(stale)))
                checks.append(
                    {
                        "label": "Workers",
                        "status": "WARN",
                        "detail": "; ".join(details),
                    }
                )
            else:
                parts = [
                    f"{mode}: {row['status']} ({row['last_seen_age_s']}s ago)"
                    for mode, row in sorted(latest_by_mode.items())
                ]
                checks.append(
                    {
                        "label": "Workers",
                        "status": "OK",
                        "detail": ", ".join(parts),
                    }
                )
        except Exception as exc:
            checks.append({"label": "Workers", "status": "WARN", "detail": str(exc)})

        # 7b. Worker task failures
        try:
            failed_rows = await conn.fetch("""
                SELECT task_type, failures_since_success, latest_error
                FROM worker_task_status
                WHERE failures_since_success > 0
                ORDER BY failures_since_success DESC, task_type
                LIMIT 5
                """)
            if failed_rows:
                detail = "; ".join(
                    f"{r['task_type']}: {r['failures_since_success']} "
                    f"({r['latest_error'] or 'no error detail'})"
                    for r in failed_rows
                )
                checks.append(
                    {
                        "label": "Worker task failures",
                        "status": "WARN",
                        "detail": detail,
                    }
                )
            else:
                checks.append(
                    {
                        "label": "Worker task failures",
                        "status": "OK",
                        "detail": "no task failure streaks",
                    }
                )
        except Exception as exc:
            checks.append(
                {"label": "Worker task failures", "status": "WARN", "detail": str(exc)}
            )

        # 8. Channels
        try:
            ch_rows = await conn.fetch("""
                SELECT channel_type, COUNT(*) AS sessions
                FROM channel_sessions
                GROUP BY channel_type
                ORDER BY channel_type
            """)
            if ch_rows:
                ch_list = [f"{r['channel_type']}" for r in ch_rows]
                checks.append(
                    {
                        "label": "Channels",
                        "status": "OK",
                        "detail": f"{len(ch_rows)} active ({', '.join(ch_list)})",
                    }
                )
            else:
                checks.append(
                    {
                        "label": "Channels",
                        "status": "WARN",
                        "detail": "0 sessions (no channel activity yet)",
                    }
                )
        except Exception:
            checks.append(
                {
                    "label": "Channels",
                    "status": "WARN",
                    "detail": "channel tables not available",
                }
            )

        # 9. Tools
        try:
            import asyncpg
            from core.agent_api import pool_sizes_from_env

            _min, _max = pool_sizes_from_env(1, 2)
            pool = await asyncpg.create_pool(dsn, min_size=_min, max_size=_max)
            try:
                from core.tools import create_full_registry

                registry = await create_full_registry(pool)
                all_handlers = registry.list_all()
                approval_count = sum(
                    1 for h in all_handlers if h.spec.requires_approval
                )
                checks.append(
                    {
                        "label": "Tools",
                        "status": "OK",
                        "detail": f"{len(all_handlers)} registered, {approval_count} requiring approval",
                    }
                )
            finally:
                await pool.close()
        except Exception as exc:
            checks.append({"label": "Tools", "status": "WARN", "detail": str(exc)})

        # 9b. Continuous registry/config/skill reachability snapshots. This is
        # advisory: absence or staleness explains what to start, never blocks.
        try:
            raw = await conn.fetchval("SELECT capability_reachability_health()")
            health = json.loads(raw) if isinstance(raw, str) else (raw or {})
            measured = int(health.get("measured_pairs") or 0)
            gaps = int(health.get("unexpected_gaps") or 0)
            stale = int(health.get("stale_pairs") or 0)
            if measured == 0:
                checks.append(
                    {
                        "label": "Tool reachability",
                        "status": "WARN",
                        "detail": "not measured yet — start workers with `hexis up`, then rerun doctor",
                    }
                )
            elif gaps or stale:
                examples = ", ".join(
                    f"{item.get('worker')}/{item.get('context')}:{item.get('tool')}"
                    for item in (health.get("gap_examples") or [])
                    if isinstance(item, dict)
                )
                detail_parts = []
                if gaps:
                    detail_parts.append(
                        f"{gaps} unexpected gap(s)"
                        + (f" ({examples})" if examples else "")
                    )
                if stale:
                    detail_parts.append(
                        f"{stale} stale measurement(s); restart workers with `hexis up`"
                    )
                checks.append(
                    {
                        "label": "Tool reachability",
                        "status": "WARN",
                        "detail": "; ".join(detail_parts),
                    }
                )
            else:
                checks.append(
                    {
                        "label": "Tool reachability",
                        "status": "OK",
                        "detail": (
                            f"{health.get('available_pairs', 0)}/{measured} worker/context/tool "
                            f"pairs reachable; 0 unexpected gaps"
                        ),
                    }
                )
        except Exception as exc:
            checks.append(
                {
                    "label": "Tool reachability",
                    "status": "WARN",
                    "detail": f"measurement unavailable ({exc}); run `hexis migrate`",
                }
            )

        # 9c. Immutable decision ledger: proves what each turn requested versus
        # what the runtime could actually expose.
        try:
            raw = await conn.fetchval("SELECT tool_surface_audit_health()")
            health = json.loads(raw) if isinstance(raw, str) else (raw or {})
            decisions = int(health.get("decisions_7d") or 0)
            decisions_with_gaps = int(health.get("decisions_with_gaps") or 0)
            if decisions == 0:
                checks.append(
                    {
                        "label": "Tool surface audit",
                        "status": "WARN",
                        "detail": "no audited turns in the last 7 days — use `hexis chat`, then rerun doctor",
                    }
                )
            elif decisions_with_gaps:
                checks.append(
                    {
                        "label": "Tool surface audit",
                        "status": "WARN",
                        "detail": (
                            f"{decisions_with_gaps}/{decisions} recent decision(s) requested tools "
                            "the runtime could not expose; check Tool reachability above"
                        ),
                    }
                )
            else:
                checks.append(
                    {
                        "label": "Tool surface audit",
                        "status": "OK",
                        "detail": f"{decisions} decision(s) in 7d; every selected tool was callable",
                    }
                )
        except Exception as exc:
            checks.append(
                {
                    "label": "Tool surface audit",
                    "status": "WARN",
                    "detail": f"audit unavailable ({exc}); run `hexis migrate`",
                }
            )

        # 10. Skills
        try:
            from skills.loader import load_skills_from_dir, discover_skill_dirs

            total = 0
            names = []
            for d in discover_skill_dirs():
                for spec in load_skills_from_dir(d):
                    total += 1
                    names.append(spec.name)
            if total > 0:
                checks.append(
                    {
                        "label": "Skills",
                        "status": "OK",
                        "detail": f"{total} loaded ({', '.join(names)})",
                    }
                )
            else:
                checks.append(
                    {"label": "Skills", "status": "WARN", "detail": "0 installed"}
                )
        except Exception as exc:
            checks.append({"label": "Skills", "status": "WARN", "detail": str(exc)})

        # 11. Schema (count applied SQL files by checking key tables/functions)
        try:
            table_count = await conn.fetchval("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """)
            func_count = await conn.fetchval("""
                SELECT COUNT(DISTINCT routine_name) FROM information_schema.routines
                WHERE routine_schema = 'public'
            """)
            checks.append(
                {
                    "label": "Schema",
                    "status": "OK",
                    "detail": f"{table_count} tables, {func_count} functions",
                }
            )
        except Exception as exc:
            checks.append({"label": "Schema", "status": "FAIL", "detail": str(exc)})

        # 12. Memory stats
        try:
            mem_stats = await conn.fetch("""
                SELECT type, COUNT(*) AS cnt
                FROM memories
                WHERE status = 'active'
                GROUP BY type
                ORDER BY type
            """)
            total = sum(r["cnt"] for r in mem_stats)
            if total == 0:
                checks.append(
                    {
                        "label": "Memory",
                        "status": "WARN",
                        "detail": "0 memories (run 'hexis init' or 'hexis chat')",
                    }
                )
            else:
                parts = [f"{r['cnt']} {r['type']}" for r in mem_stats]
                checks.append(
                    {
                        "label": "Memory",
                        "status": "OK",
                        "detail": ", ".join(parts),
                    }
                )
        except Exception as exc:
            checks.append({"label": "Memory", "status": "FAIL", "detail": str(exc)})

        # 13. LLM connectivity (opt-in: makes one real call)
        if check_llm:
            try:
                from core.init_api import test_llm_connection
                from core.llm_config import load_llm_config

                llm_config = await load_llm_config(conn, "llm.chat", fallback_key="llm")
                result = await test_llm_connection(llm_config)
                model = (
                    f"{llm_config.get('provider', '?')}/{llm_config.get('model', '?')}"
                )
                checks.append(
                    {
                        "label": "LLM",
                        "status": "OK" if result["ok"] else "FAIL",
                        "detail": f"{model} — {result['message']}",
                    }
                )
            except Exception as exc:
                checks.append({"label": "LLM", "status": "FAIL", "detail": str(exc)})
        else:
            checks.append(
                {
                    "label": "LLM",
                    "status": "WARN",
                    "detail": "not tested (run 'hexis doctor --llm' for a live connectivity check)",
                }
            )

    finally:
        await conn.close()

    return checks


def _format_timedelta(td) -> str:
    """Format a timedelta to a human-readable string like '3d 14h' or '2m'."""
    total_seconds = int(td.total_seconds())
    if total_seconds < 0:
        return "just now"
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m"
    return f"{total_seconds}s"


# ---------------------------------------------------------------------------
# enhanced status -- rich agent overview
# ---------------------------------------------------------------------------


async def status_payload_rich(
    dsn: str | None = None,
    *,
    wait_seconds: int = 30,
) -> dict[str, Any]:
    """
    Return a rich status payload with identity, energy, memory counts,
    channels, goals, and mood.
    """
    dsn = dsn or db_dsn_from_env()
    conn = await _connect_with_retry(dsn, wait_seconds=wait_seconds)
    try:
        payload: dict[str, Any] = {}

        # Instance info
        try:
            from core.instance import InstanceRegistry

            reg = InstanceRegistry()
            current = reg.get_current()
            inst = reg.get(current) if current else None
            payload["instance"] = current or "default"
            payload["database"] = inst.database if inst else "hexis_memory"
        except Exception:
            payload["instance"] = "default"
            payload["database"] = "hexis_memory"

        # Identity (agent name)
        try:
            profile = await conn.fetchval("SELECT get_agent_profile_context()")
            if profile:
                p = json.loads(profile) if isinstance(profile, str) else profile
                payload["identity"] = (
                    p.get("name") or p.get("persona", {}).get("name") or "unnamed"
                )
            else:
                payload["identity"] = None
        except Exception:
            payload["identity"] = None

        # Energy and heartbeat
        try:
            hb_row = await conn.fetchrow(
                "SELECT current_energy, last_heartbeat_at, next_heartbeat_at, "
                "is_paused, heartbeat_count FROM heartbeat_state WHERE id = 1"
            )
            max_energy = await conn.fetchval(
                "SELECT get_config_int('heartbeat.max_energy')"
            )
            interval_min = await conn.fetchval(
                "SELECT get_config_float('heartbeat.heartbeat_interval_minutes')"
            )
            energy_capacity = await conn.fetchval("SELECT heartbeat_bank_capacity()")
            if max_energy is None or interval_min is None:
                raise RuntimeError(
                    "Missing heartbeat config defaults; run `hexis migrate`."
                )

            if hb_row:
                payload["energy"] = hb_row["current_energy"]
                payload["max_energy"] = max_energy
                payload["energy_reserve"] = max_energy
                payload["energy_capacity"] = energy_capacity
                payload["heartbeat_paused"] = hb_row["is_paused"]
                payload["heartbeat_count"] = hb_row["heartbeat_count"]
                payload["heartbeat_interval_minutes"] = float(interval_min)

                last_hb = hb_row["last_heartbeat_at"]
                if last_hb:
                    from datetime import timezone as tz

                    now = datetime.now(tz.utc)
                    if hasattr(last_hb, "tzinfo") and last_hb.tzinfo is None:
                        last_hb = last_hb.replace(tzinfo=tz.utc)
                    payload["last_heartbeat_ago"] = _format_timedelta(now - last_hb)
                else:
                    payload["last_heartbeat_ago"] = None

                next_hb = hb_row["next_heartbeat_at"]
                payload["next_heartbeat_at"] = (
                    next_hb.isoformat() if next_hb is not None else None
                )

                # Regeneration is applied at the next DB-scheduled heartbeat.
                if hb_row["current_energy"] < energy_capacity:
                    from datetime import timezone as tz

                    now = datetime.now(tz.utc)
                    if next_hb is None and last_hb is not None:
                        next_hb = last_hb + timedelta(minutes=float(interval_min))
                    if next_hb is not None:
                        if hasattr(next_hb, "tzinfo") and next_hb.tzinfo is None:
                            next_hb = next_hb.replace(tzinfo=tz.utc)
                        remaining = max(0, (next_hb - now).total_seconds())
                        payload["next_regen_minutes"] = round(remaining / 60, 1)
                    else:
                        payload["next_regen_minutes"] = None
                else:
                    payload["next_regen_minutes"] = None
        except Exception:
            payload["energy"] = None

        # Heartbeat active status
        try:
            payload["heartbeat_active"] = bool(
                await conn.fetchval("SELECT should_run_heartbeat()")
            )
        except Exception:
            payload["heartbeat_active"] = None

        # Memory counts by type
        try:
            mem_rows = await conn.fetch("""
                SELECT type, COUNT(*) AS cnt
                FROM memories WHERE status = 'active'
                GROUP BY type ORDER BY type
            """)
            payload["memories"] = {r["type"]: r["cnt"] for r in mem_rows}
        except Exception:
            payload["memories"] = {}

        # Active channels
        try:
            ch_rows = await conn.fetch("""
                SELECT channel_type, COUNT(*) AS sessions,
                       COUNT(*) FILTER (WHERE last_active > CURRENT_TIMESTAMP - INTERVAL '1 hour') AS active_1h
                FROM channel_sessions
                GROUP BY channel_type ORDER BY channel_type
            """)
            payload["channels"] = [
                {
                    "type": r["channel_type"],
                    "sessions": r["sessions"],
                    "active_1h": r["active_1h"],
                }
                for r in ch_rows
            ]
        except Exception:
            payload["channels"] = []

        # Active goals
        try:
            goal_rows = await conn.fetch("""
                SELECT content, metadata->>'priority' AS priority
                FROM memories
                WHERE type = 'goal' AND status = 'active'
                ORDER BY importance DESC
                LIMIT 5
            """)
            payload["goals"] = [
                {"content": r["content"][:100], "priority": r["priority"]}
                for r in goal_rows
            ]
        except Exception:
            payload["goals"] = []

        # Mood / emotional state
        try:
            emo_raw = await conn.fetchval(
                "SELECT value FROM state WHERE key = 'heartbeat_state'"
            )
            if emo_raw:
                emo = json.loads(emo_raw) if isinstance(emo_raw, str) else emo_raw
                aff = emo.get("affective_state", {})
                if aff:
                    valence = aff.get("valence", 0.0)
                    arousal = aff.get("arousal", 0.0)
                    # The DB owns the valence/arousal -> mood ladder (db/65).
                    payload["mood"] = await conn.fetchval(
                        "SELECT mood_label($1, $2)", float(valence), float(arousal)
                    )
                    payload["valence"] = round(valence, 2)
                else:
                    payload["mood"] = "neutral"
                    payload["valence"] = 0.0
            else:
                payload["mood"] = None
        except Exception:
            payload["mood"] = None

        # Scheduled tasks
        try:
            task_count = await conn.fetchval(
                "SELECT COUNT(*) FROM scheduled_tasks WHERE status = 'active'"
            )
            payload["scheduled_tasks"] = task_count or 0
        except Exception:
            payload["scheduled_tasks"] = 0

        # Worker runtime status
        try:
            worker_rows = await conn.fetch(_CURRENT_WORKERS_SQL)
            workers = [_record_to_dict(r) for r in worker_rows]
            for worker in workers:
                worker.pop("mode_rank", None)
            payload["workers"] = workers
            task_rows = await conn.fetch("SELECT * FROM worker_task_status")
            payload["worker_tasks"] = [_record_to_dict(r) for r in task_rows]
        except Exception:
            payload["workers"] = []
            payload["worker_tasks"] = []

        return payload
    finally:
        await conn.close()
