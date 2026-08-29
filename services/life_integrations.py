"""Provider clients and verified setup for everyday-life integrations.

Postgres owns connector identity, selected capabilities, and connection state.
This module resolves only the exact environment references stored on a
connection and performs bounded provider I/O. Secret values are never written
to Postgres or returned in tool payloads.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import quote, urlparse

from core.integration_reliability import (
    IntegrationHttpError,
    request_json,
)


WAVE_B_CONNECTORS = frozenset({"notion", "home_assistant", "weather", "trello"})
ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
MAX_PROVIDER_BODY_BYTES = 200_000


class LifeIntegrationError(RuntimeError):
    """Expected, user-actionable setup or provider failure."""


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _bounded_json(value: Any, *, label: str) -> Any:
    try:
        encoded = json.dumps(value, default=str).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LifeIntegrationError(f"{label} must be valid JSON.") from exc
    if len(encoded) > MAX_PROVIDER_BODY_BYTES:
        raise LifeIntegrationError(
            f"{label} is too large ({len(encoded)} bytes; maximum {MAX_PROVIDER_BODY_BYTES})."
        )
    return value


def _env_name(value: Any, *, label: str) -> str:
    name = str(value or "").strip()
    if not name:
        raise LifeIntegrationError(f"{label} is required.")
    if not ENV_NAME_RE.fullmatch(name):
        raise LifeIntegrationError(
            f"{label} must be an environment variable name such as {label.upper().replace(' ', '_')}."
        )
    return name


def _selected_env_value(value: Any, *, label: str) -> tuple[str, str]:
    name = _env_name(value, label=label)
    secret = os.getenv(name)
    if not secret:
        raise LifeIntegrationError(
            f"{name} is not set in the Hexis runtime. Set it, restart the relevant Hexis process, "
            "then verify the connection again."
        )
    return name, secret


def _base_url(value: Any, *, label: str) -> str:
    raw = str(value or "").strip().rstrip("/")
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LifeIntegrationError(f"{label} must be a complete http:// or https:// URL.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise LifeIntegrationError(
            f"{label} cannot contain credentials, a query string, or a fragment."
        )
    return raw


def _identifier(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not text or not IDENTIFIER_RE.fullmatch(text):
        raise LifeIntegrationError(f"{label} is missing or contains unsupported characters.")
    return text


async def connector_manifest(pool: Any, connector_id: str) -> dict[str, Any]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT display_name, capability_manifest, setup_manifest, docs_url
            FROM integration_connectors
            WHERE id = $1 AND status = 'available'
            """,
            connector_id,
        )
    if row is None:
        raise LifeIntegrationError(f"{connector_id} is not an available connector.")
    return {
        "display_name": row["display_name"],
        "capability_manifest": _json(row["capability_manifest"]) or {},
        "setup_manifest": _json(row["setup_manifest"]) or {},
        "docs_url": row["docs_url"],
    }


async def connected_account(
    pool: Any,
    connector_id: str,
    account_key: str | None = None,
) -> dict[str, Any]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id::text, connector_id, account_key, display_name,
                   credential_ref, granted_scopes, capabilities, metadata
            FROM integration_connections
            WHERE connector_id = $1
              AND status = 'connected'
              AND ($2::text IS NULL OR account_key = $2)
            ORDER BY last_verified_at DESC NULLS LAST, updated_at DESC
            LIMIT 1
            """,
            connector_id,
            str(account_key).strip() if account_key else None,
        )
    if row is None:
        suffix = f" for {account_key}" if account_key else ""
        raise LifeIntegrationError(
            f"{connector_id} is not connected{suffix}. Use the connector setup flow first."
        )
    return {
        "id": row["id"],
        "connector_id": row["connector_id"],
        "account_key": row["account_key"],
        "display_name": row["display_name"],
        "credential_ref": row["credential_ref"],
        "granted_scopes": list(row["granted_scopes"] or []),
        "capabilities": list(_json(row["capabilities"]) or []),
        "metadata": _json(row["metadata"]) or {},
    }


def require_capability(connection: dict[str, Any], capability: str) -> None:
    if capability not in set(connection.get("capabilities") or []):
        raise LifeIntegrationError(
            f"The connected {connection['connector_id']} account did not grant {capability}. "
            "Reconnect it with that capability first."
        )


def _setup_ui(
    connector_id: str,
    manifest: dict[str, Any],
    *,
    status: str,
    attempt_id: str | None = None,
    capabilities: list[str] | None = None,
    next_step: str | None = None,
    authorization_url: str | None = None,
    connected_accounts: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    setup = manifest.get("setup_manifest") or {}
    ui = {
        "kind": "connector_setup",
        "version": 2,
        "id": f"connector_setup:{connector_id}:{attempt_id or status}",
        "connector_id": connector_id,
        "display_name": manifest.get("display_name") or connector_id,
        "title": (
            f"{manifest.get('display_name') or connector_id} connected"
            if status == "connected"
            else f"Connect {manifest.get('display_name') or connector_id}"
        ),
        "status": status,
        "capabilities": list(capabilities or []),
        "credential_fields": list(setup.get("credential_fields") or []),
        "docs_url": manifest.get("docs_url"),
        "attempt_id": attempt_id,
        "authorization_url": authorization_url,
        "next_step": next_step or setup.get("user_next_step"),
        "connected_accounts": list(connected_accounts or []),
        "safety_note": (
            "Connection access and permission to perform external writes are separate. "
            "Provider writes still require explicit tool approval."
        ),
    }
    if extra:
        ui.update(extra)
    return ui


def enrich_life_setup_status(payload: dict[str, Any], connector_id: str | None) -> dict[str, Any]:
    """Attach one generic setup card to a filtered integration-status payload."""
    normalized = str(connector_id or "").strip().lower().replace("-", "_")
    if normalized not in {*WAVE_B_CONNECTORS, "spotify"}:
        return payload
    connectors = payload.get("connectors")
    if not isinstance(connectors, list) or not connectors:
        return payload
    connector = connectors[0]
    if not isinstance(connector, dict):
        return payload
    manifest = {
        "display_name": connector.get("display_name") or normalized,
        "setup_manifest": _json(connector.get("setup_manifest")) or {},
        "docs_url": connector.get("docs_url"),
    }
    connected = [
        item
        for item in payload.get("connections", [])
        if isinstance(item, dict) and item.get("status") == "connected"
    ]
    pending = next(
        (
            item
            for item in payload.get("recent_attempts", [])
            if isinstance(item, dict)
            and item.get("status") in {"pending_user", "awaiting_input", "error"}
        ),
        None,
    )
    if connected:
        status = "connected"
        capabilities = list(connected[0].get("capabilities") or [])
    elif pending and pending.get("authorization_url"):
        status = "pending_authorization"
        capabilities = list(pending.get("requested_capabilities") or [])
    elif pending:
        status = "needs_configuration"
        capabilities = list(pending.get("requested_capabilities") or [])
    else:
        status = "needs_client" if normalized == "spotify" else "needs_configuration"
        capabilities = list(manifest["setup_manifest"].get("default_capabilities") or [])
    ui = _setup_ui(
        normalized,
        manifest,
        status=status,
        attempt_id=str(pending.get("attempt_id")) if pending else None,
        capabilities=capabilities,
        next_step=(pending or {}).get("user_next_step") if pending else None,
        authorization_url=(pending or {}).get("authorization_url") if pending else None,
        connected_accounts=connected,
        extra={
            "completion_mode": "automatic_callback_or_paste" if normalized == "spotify" else None,
            "manual_completion_available": normalized == "spotify",
        },
    )
    payload["ui"] = ui
    return payload


async def _prepare_attempt(
    pool: Any,
    connector_id: str,
    capabilities: Any,
    *,
    source_channel: str | None,
    source_session_id: str | None,
    flow_state: dict[str, Any],
    next_step: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    async with pool.acquire() as conn:
        raw_plan = await conn.fetchval(
            "SELECT prepare_connection_attempt($1, $2::jsonb)",
            connector_id,
            json.dumps(capabilities) if capabilities is not None else None,
        )
        plan = _json(raw_plan) or {}
        raw_attempt = await conn.fetchval(
            """
            SELECT start_connection_attempt(
                $1,
                $2::jsonb,
                ARRAY[]::text[],
                $3::jsonb,
                NULL,
                $4,
                $5,
                $6,
                CURRENT_TIMESTAMP + INTERVAL '30 minutes'
            )
            """,
            connector_id,
            json.dumps(plan.get("capabilities") or []),
            json.dumps(flow_state),
            next_step,
            source_channel,
            source_session_id,
        )
    return plan, _json(raw_attempt) or {}


async def _mark_attempt_error(pool: Any, attempt_id: str, exc: Exception) -> None:
    async with pool.acquire() as conn:
        await conn.fetchval(
            "SELECT mark_connection_attempt_error($1::uuid, $2)",
            attempt_id,
            str(exc)[:2000],
        )


async def geocode_location(
    pool: Any,
    location: str,
    *,
    count: int = 5,
) -> list[dict[str, Any]]:
    query = str(location or "").strip()
    if len(query) < 2:
        raise LifeIntegrationError("A city or place name with at least two characters is required.")
    manifest = await connector_manifest(pool, "weather")
    setup = manifest["setup_manifest"]
    base = _base_url(setup.get("geocoding_base_url"), label="Weather geocoding URL")
    payload = await request_json(
        "open_meteo_geocoding",
        "GET",
        f"{base}/search",
        params={"name": query, "count": max(1, min(int(count), 10)), "language": "en", "format": "json"},
        timeout=15.0,
        attempts=3,
        max_delay=4.0,
    )
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or not results:
        raise LifeIntegrationError(
            f"Weather could not match {query!r}. Try a city plus state/region or country."
        )
    clean: list[dict[str, Any]] = []
    for item in results[:10]:
        if not isinstance(item, dict) or item.get("latitude") is None or item.get("longitude") is None:
            continue
        clean.append(
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "admin1": item.get("admin1"),
                "country": item.get("country"),
                "country_code": item.get("country_code"),
                "latitude": float(item["latitude"]),
                "longitude": float(item["longitude"]),
                "timezone": item.get("timezone") or "auto",
            }
        )
    if not clean:
        raise LifeIntegrationError("Weather geocoding returned no usable coordinates.")
    return clean


async def forecast_coordinates(
    pool: Any,
    *,
    latitude: float,
    longitude: float,
    days: int = 7,
    timezone: str = "auto",
) -> dict[str, Any]:
    if not -90 <= float(latitude) <= 90 or not -180 <= float(longitude) <= 180:
        raise LifeIntegrationError("Weather coordinates are outside the valid latitude/longitude range.")
    manifest = await connector_manifest(pool, "weather")
    base = _base_url(manifest["setup_manifest"].get("forecast_base_url"), label="Weather forecast URL")
    return await request_json(
        "open_meteo_forecast",
        "GET",
        f"{base}/forecast",
        params={
            "latitude": float(latitude),
            "longitude": float(longitude),
            "timezone": str(timezone or "auto"),
            "forecast_days": max(1, min(int(days), 16)),
            "current": (
                "temperature_2m,apparent_temperature,precipitation,rain,showers,"
                "snowfall,weather_code,cloud_cover,wind_speed_10m,wind_gusts_10m"
            ),
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,"
                "apparent_temperature_min,sunrise,sunset,precipitation_sum,"
                "precipitation_probability_max,wind_speed_10m_max,wind_gusts_10m_max"
            ),
        },
        timeout=20.0,
        attempts=3,
        max_delay=4.0,
    )


async def setup_life_integration(
    pool: Any,
    connector_id: str,
    arguments: dict[str, Any],
    *,
    source_channel: str | None = None,
    source_session_id: str | None = None,
) -> dict[str, Any]:
    connector_id = str(connector_id or "").strip().lower().replace("-", "_")
    if connector_id not in WAVE_B_CONNECTORS:
        raise LifeIntegrationError(
            f"connect_life_integration supports {', '.join(sorted(WAVE_B_CONNECTORS))}; Spotify uses connect_spotify."
        )
    manifest = await connector_manifest(pool, connector_id)
    setup = manifest["setup_manifest"]
    capabilities = arguments.get("capabilities")
    missing: list[str] = []
    if connector_id == "notion" and not arguments.get("token_env"):
        missing = ["token_env"]
    elif connector_id == "home_assistant":
        missing = [name for name in ("base_url", "token_env") if not arguments.get(name)]
    elif connector_id == "weather" and not arguments.get("location"):
        missing = ["location"]
    elif connector_id == "trello":
        missing = [name for name in ("api_key_env", "token_env") if not arguments.get(name)]

    next_step = str(setup.get("user_next_step") or "Configure this connector, then verify it.")
    plan, attempt = await _prepare_attempt(
        pool,
        connector_id,
        capabilities,
        source_channel=source_channel,
        source_session_id=source_session_id,
        flow_state={"setup_kind": setup.get("flow"), "missing_fields": missing, "secret_values_stored": False},
        next_step=next_step,
    )
    attempt_id = str(attempt["attempt_id"])
    if missing:
        payload = {
            **attempt,
            "status": "awaiting_input",
            "missing_fields": missing,
            "next_step": next_step,
        }
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE connection_attempts SET status = 'awaiting_input', updated_at = CURRENT_TIMESTAMP WHERE id = $1::uuid",
                attempt_id,
            )
        payload["ui"] = _setup_ui(
            connector_id,
            manifest,
            status="needs_configuration",
            attempt_id=attempt_id,
            capabilities=list(plan.get("capabilities") or []),
            next_step=next_step,
            extra={"missing_fields": missing},
        )
        return payload

    try:
        metadata: dict[str, Any]
        credential_ref: str | None
        account_key: str
        display_name: str
        if connector_id == "notion":
            token_env, token = _selected_env_value(arguments.get("token_env"), label="token_env")
            api_base = _base_url(setup.get("api_base_url"), label="Notion API URL")
            api_version = str(setup.get("api_version") or "").strip()
            if not api_version:
                raise LifeIntegrationError("The Notion connector manifest is missing api_version.")
            me = await request_json(
                "notion",
                "GET",
                f"{api_base}/v1/users/me",
                headers={"Authorization": f"Bearer {token}", "Notion-Version": api_version},
                timeout=15.0,
                attempts=2,
                max_delay=3.0,
            )
            if not isinstance(me, dict) or not me.get("id"):
                raise LifeIntegrationError("Notion verification did not return an integration user ID.")
            account_key = str(arguments.get("account_key") or f"notion:{me['id']}")
            display_name = str(me.get("name") or "Notion integration")
            credential_ref = f"env:{token_env}"
            metadata = {"token_env": token_env, "api_base_url": api_base, "api_version": api_version, "bot_id": me["id"], "secret_values_stored": False}
        elif connector_id == "home_assistant":
            token_env, token = _selected_env_value(arguments.get("token_env"), label="token_env")
            base = _base_url(arguments.get("base_url"), label="base_url")
            api_base = base if base.endswith("/api") else f"{base}/api"
            result = await request_json(
                "home_assistant",
                "GET",
                api_base,
                headers={"Authorization": f"Bearer {token}"},
                timeout=15.0,
                attempts=2,
                max_delay=3.0,
            )
            if not isinstance(result, dict):
                raise LifeIntegrationError("Home Assistant verification returned an invalid response.")
            parsed = urlparse(base)
            account_key = str(arguments.get("account_key") or f"home_assistant:{parsed.netloc.lower()}")
            display_name = str(arguments.get("display_name") or parsed.netloc)
            credential_ref = f"env:{token_env}"
            metadata = {"token_env": token_env, "base_url": base, "api_base_url": api_base, "secret_values_stored": False}
        elif connector_id == "trello":
            api_key_env, api_key = _selected_env_value(arguments.get("api_key_env"), label="api_key_env")
            token_env, token = _selected_env_value(arguments.get("token_env"), label="token_env")
            api_base = _base_url(setup.get("api_base_url"), label="Trello API URL")
            me = await request_json(
                "trello",
                "GET",
                f"{api_base}/members/me",
                params={"key": api_key, "token": token, "fields": "id,username,fullName,url"},
                timeout=15.0,
                attempts=2,
                max_delay=3.0,
            )
            if not isinstance(me, dict) or not me.get("id"):
                raise LifeIntegrationError("Trello verification did not return a member ID.")
            account_key = str(arguments.get("account_key") or f"trello:{me['id']}")
            display_name = str(me.get("fullName") or me.get("username") or "Trello member")
            credential_ref = f"env:{api_key_env}+env:{token_env}"
            metadata = {"api_key_env": api_key_env, "token_env": token_env, "api_base_url": api_base, "member_id": me["id"], "username": me.get("username"), "secret_values_stored": False}
        else:
            matches = await geocode_location(pool, str(arguments.get("location") or ""), count=5)
            selected = matches[0]
            forecast = await forecast_coordinates(
                pool,
                latitude=selected["latitude"],
                longitude=selected["longitude"],
                days=1,
                timezone=str(selected.get("timezone") or "auto"),
            )
            account_key = str(
                arguments.get("account_key")
                or f"weather:{selected['latitude']:.5f},{selected['longitude']:.5f}"
            )
            pieces = [selected.get("name"), selected.get("admin1"), selected.get("country")]
            display_name = ", ".join(str(piece) for piece in pieces if piece)
            credential_ref = None
            metadata = {"location": display_name, **selected, "provider": "open_meteo", "secret_values_stored": False}
            metadata["verified_timezone"] = forecast.get("timezone") if isinstance(forecast, dict) else None

        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                """
                SELECT complete_connection_attempt(
                    $1::uuid, $2, $3, $4, $5::text[], $6::jsonb, $7::jsonb
                )
                """,
                attempt_id,
                account_key,
                display_name,
                credential_ref,
                list(plan.get("requested_scopes") or []),
                json.dumps(plan.get("capabilities") or []),
                json.dumps(metadata),
            )
        result = _json(raw) or {}
        public_connection = {
            key: result.get(key)
            for key in ("connector_id", "connection_id", "account_key", "display_name", "status", "granted_scopes", "capabilities", "connected_at")
        }
        result["ui"] = _setup_ui(
            connector_id,
            manifest,
            status="connected",
            attempt_id=attempt_id,
            capabilities=list(result.get("capabilities") or []),
            next_step=f"{display_name} is verified and ready to use.",
            connected_accounts=[public_connection],
            extra={"matched_locations": matches if connector_id == "weather" else []},
        )
        result["next_step"] = f"{display_name} is verified and ready to use."
        return result
    except Exception as exc:
        await _mark_attempt_error(pool, attempt_id, exc)
        if isinstance(exc, (LifeIntegrationError, IntegrationHttpError)):
            raise
        raise LifeIntegrationError(str(exc)) from exc


def _credential(connection: dict[str, Any], name: str) -> str:
    metadata = connection.get("metadata") or {}
    _env, secret = _selected_env_value(metadata.get(name), label=name)
    return secret


async def notion_request(
    pool: Any,
    method: str,
    path: str,
    *,
    account_key: str | None = None,
    capability: str,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> Any:
    connection = await connected_account(pool, "notion", account_key)
    require_capability(connection, capability)
    metadata = connection["metadata"]
    token = _credential(connection, "token_env")
    api_base = _base_url(metadata.get("api_base_url"), label="Notion API URL")
    api_version = str(metadata.get("api_version") or "").strip()
    if not api_version:
        raise LifeIntegrationError("The Notion connection is missing its API version. Reconnect Notion.")
    return await request_json(
        "notion",
        method,
        f"{api_base}{path}",
        headers={"Authorization": f"Bearer {token}", "Notion-Version": api_version},
        params=params,
        json_body=_bounded_json(body, label="Notion request") if body is not None else None,
        timeout=30.0,
        attempts=3 if method.upper() == "GET" else 1,
    )


async def home_assistant_request(
    pool: Any,
    method: str,
    path: str,
    *,
    account_key: str | None = None,
    capability: str,
    body: dict[str, Any] | None = None,
) -> Any:
    connection = await connected_account(pool, "home_assistant", account_key)
    require_capability(connection, capability)
    token = _credential(connection, "token_env")
    api_base = _base_url(connection["metadata"].get("api_base_url"), label="Home Assistant API URL")
    return await request_json(
        "home_assistant",
        method,
        f"{api_base}{path}",
        headers={"Authorization": f"Bearer {token}"},
        json_body=_bounded_json(body, label="Home Assistant service data") if body is not None else None,
        timeout=30.0,
        attempts=3 if method.upper() == "GET" else 1,
    )


async def trello_request(
    pool: Any,
    method: str,
    path: str,
    *,
    account_key: str | None = None,
    capability: str,
    params: dict[str, Any] | None = None,
) -> Any:
    connection = await connected_account(pool, "trello", account_key)
    require_capability(connection, capability)
    api_key = _credential(connection, "api_key_env")
    token = _credential(connection, "token_env")
    api_base = _base_url(connection["metadata"].get("api_base_url"), label="Trello API URL")
    query = {"key": api_key, "token": token, **(params or {})}
    return await request_json(
        "trello",
        method,
        f"{api_base}{path}",
        params=query,
        timeout=30.0,
        attempts=3 if method.upper() == "GET" else 1,
    )


async def notion_search(
    pool: Any,
    *,
    query: str,
    object_type: str | None = None,
    page_size: int = 20,
    start_cursor: str | None = None,
    account_key: str | None = None,
) -> Any:
    body: dict[str, Any] = {"query": str(query or ""), "page_size": max(1, min(int(page_size), 100))}
    if object_type:
        normalized = str(object_type).strip().lower()
        if normalized not in {"page", "data_source"}:
            raise LifeIntegrationError("Notion object_type must be page or data_source.")
        body["filter"] = {"property": "object", "value": normalized}
    if start_cursor:
        body["start_cursor"] = str(start_cursor)
    return await notion_request(pool, "POST", "/v1/search", account_key=account_key, capability="search", body=body)


async def notion_get_page(
    pool: Any,
    *,
    page_id: str,
    include_blocks: bool = True,
    page_size: int = 100,
    account_key: str | None = None,
) -> dict[str, Any]:
    identifier = _identifier(page_id, label="page_id")
    page = await notion_request(pool, "GET", f"/v1/pages/{quote(identifier, safe='')}", account_key=account_key, capability="read")
    result = {"page": page}
    if include_blocks:
        result["blocks"] = await notion_request(
            pool,
            "GET",
            f"/v1/blocks/{quote(identifier, safe='')}/children",
            account_key=account_key,
            capability="read",
            params={"page_size": max(1, min(int(page_size), 100))},
        )
    return result


async def notion_query_data_source(
    pool: Any,
    *,
    data_source_id: str,
    filter_value: dict[str, Any] | None = None,
    sorts: list[dict[str, Any]] | None = None,
    page_size: int = 100,
    start_cursor: str | None = None,
    account_key: str | None = None,
) -> Any:
    identifier = _identifier(data_source_id, label="data_source_id")
    body: dict[str, Any] = {"page_size": max(1, min(int(page_size), 100))}
    if filter_value is not None:
        body["filter"] = _bounded_json(filter_value, label="Notion filter")
    if sorts is not None:
        body["sorts"] = _bounded_json(sorts, label="Notion sorts")
    if start_cursor:
        body["start_cursor"] = str(start_cursor)
    return await notion_request(
        pool,
        "POST",
        f"/v1/data_sources/{quote(identifier, safe='')}/query",
        account_key=account_key,
        capability="query",
        body=body,
    )


async def notion_create_page(
    pool: Any,
    *,
    parent_id: str,
    parent_type: str,
    properties: dict[str, Any],
    children: list[dict[str, Any]] | None = None,
    icon: dict[str, Any] | None = None,
    cover: dict[str, Any] | None = None,
    account_key: str | None = None,
) -> Any:
    parent = _identifier(parent_id, label="parent_id")
    normalized_type = str(parent_type or "page_id").strip().lower()
    if normalized_type not in {"page_id", "data_source_id"}:
        raise LifeIntegrationError("parent_type must be page_id or data_source_id.")
    body: dict[str, Any] = {"parent": {normalized_type: parent}, "properties": properties}
    if children is not None:
        body["children"] = children
    if icon is not None:
        body["icon"] = icon
    if cover is not None:
        body["cover"] = cover
    return await notion_request(pool, "POST", "/v1/pages", account_key=account_key, capability="create", body=body)


async def home_assistant_states(
    pool: Any,
    *,
    entity_id: str | None = None,
    account_key: str | None = None,
) -> Any:
    path = "/states"
    if entity_id:
        entity = _identifier(entity_id, label="entity_id")
        path += f"/{quote(entity, safe='')}"
    return await home_assistant_request(pool, "GET", path, account_key=account_key, capability="states")


async def home_assistant_call_service(
    pool: Any,
    *,
    domain: str,
    service: str,
    service_data: dict[str, Any] | None = None,
    entity_id: str | None = None,
    account_key: str | None = None,
) -> Any:
    clean_domain = _identifier(domain, label="domain")
    clean_service = _identifier(service, label="service")
    body = dict(service_data or {})
    if entity_id:
        body.setdefault("entity_id", _identifier(entity_id, label="entity_id"))
    return await home_assistant_request(
        pool,
        "POST",
        f"/services/{quote(clean_domain, safe='')}/{quote(clean_service, safe='')}",
        account_key=account_key,
        capability="service_control",
        body=body,
    )


async def weather_forecast(
    pool: Any,
    *,
    location: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    days: int = 7,
    account_key: str | None = None,
) -> dict[str, Any]:
    place: dict[str, Any]
    if location:
        place = (await geocode_location(pool, location, count=1))[0]
    elif latitude is not None or longitude is not None:
        if latitude is None or longitude is None:
            raise LifeIntegrationError("Provide both latitude and longitude, or neither.")
        place = {"name": None, "latitude": float(latitude), "longitude": float(longitude), "timezone": "auto"}
    else:
        connection = await connected_account(pool, "weather", account_key)
        require_capability(connection, "forecast")
        metadata = connection["metadata"]
        place = {
            "name": metadata.get("location") or connection.get("display_name"),
            "admin1": metadata.get("admin1"),
            "country": metadata.get("country"),
            "latitude": float(metadata["latitude"]),
            "longitude": float(metadata["longitude"]),
            "timezone": metadata.get("timezone") or "auto",
        }
    forecast = await forecast_coordinates(
        pool,
        latitude=float(place["latitude"]),
        longitude=float(place["longitude"]),
        days=days,
        timezone=str(place.get("timezone") or "auto"),
    )
    return {"location": place, "forecast": forecast}


async def trello_list_boards(pool: Any, *, account_key: str | None = None) -> Any:
    return await trello_request(
        pool,
        "GET",
        "/members/me/boards",
        account_key=account_key,
        capability="boards",
        params={"filter": "open", "fields": "id,name,desc,url,closed", "lists": "open", "list_fields": "id,name,closed,pos"},
    )


async def trello_list_cards(
    pool: Any,
    *,
    board_id: str | None = None,
    list_id: str | None = None,
    card_filter: str = "open",
    account_key: str | None = None,
) -> Any:
    if bool(board_id) == bool(list_id):
        raise LifeIntegrationError("Provide exactly one of board_id or list_id.")
    if board_id:
        identifier = _identifier(board_id, label="board_id")
        path = f"/boards/{quote(identifier, safe='')}/cards"
    else:
        identifier = _identifier(list_id, label="list_id")
        path = f"/lists/{quote(identifier, safe='')}/cards"
    return await trello_request(
        pool,
        "GET",
        path,
        account_key=account_key,
        capability="cards",
        params={"filter": str(card_filter or "open"), "fields": "id,name,desc,due,dueComplete,closed,idBoard,idList,url,labels"},
    )


async def trello_create_card(
    pool: Any,
    *,
    list_id: str,
    name: str,
    description: str | None = None,
    due: str | None = None,
    position: str | None = None,
    account_key: str | None = None,
) -> Any:
    clean_name = str(name or "").strip()
    if not clean_name:
        raise LifeIntegrationError("Card name is required.")
    params: dict[str, Any] = {"idList": _identifier(list_id, label="list_id"), "name": clean_name}
    if description is not None:
        params["desc"] = str(description)
    if due is not None:
        params["due"] = str(due)
    if position is not None:
        params["pos"] = str(position)
    _bounded_json(params, label="Trello card")
    return await trello_request(pool, "POST", "/cards", account_key=account_key, capability="create_card", params=params)


async def trello_update_card(
    pool: Any,
    *,
    card_id: str,
    changes: dict[str, Any],
    account_key: str | None = None,
) -> Any:
    allowed = {"name", "desc", "due", "dueComplete", "closed", "idList", "pos"}
    unknown = sorted(set(changes) - allowed)
    if unknown:
        raise LifeIntegrationError(f"Unsupported Trello card fields: {', '.join(unknown)}.")
    if not changes:
        raise LifeIntegrationError("At least one Trello card change is required.")
    _bounded_json(changes, label="Trello card changes")
    identifier = _identifier(card_id, label="card_id")
    return await trello_request(
        pool,
        "PUT",
        f"/cards/{quote(identifier, safe='')}",
        account_key=account_key,
        capability="update_card",
        params=changes,
    )
