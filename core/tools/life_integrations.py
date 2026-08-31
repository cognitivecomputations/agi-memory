"""ToolHandlers for PLAN.md Wave B everyday-life integrations."""

from __future__ import annotations

import json
from typing import Any

from core.integration_reliability import IntegrationHttpError
from services.life_integrations import LifeIntegrationError

from .base import (
    ToolCategory,
    ToolContext,
    ToolErrorType,
    ToolExecutionContext,
    ToolHandler,
    ToolResult,
    ToolSpec,
)
from .integration_http import integration_error_result


ALL_CONTEXTS = {ToolContext.CHAT, ToolContext.HEARTBEAT, ToolContext.MCP}
SETUP_CONTEXTS = {ToolContext.CHAT, ToolContext.MCP}
LIFE_CONNECTOR_IDS = frozenset({"notion", "spotify", "home_assistant", "weather", "trello"})
SPOTIFY_PLAYBACK_ACTIONS = frozenset(
    {"play", "pause", "next", "previous", "queue", "seek", "volume", "transfer", "shuffle", "repeat"}
)


def _pool(context: ToolExecutionContext, tool_name: str) -> Any:
    if not context.registry:
        raise LifeIntegrationError(f"{tool_name} requires an active tool registry.")
    return context.registry.pool


def _error(exc: Exception, provider: str) -> ToolResult:
    if isinstance(exc, IntegrationHttpError):
        return integration_error_result(provider, exc)
    if isinstance(exc, LifeIntegrationError):
        message = str(exc)
        error_type = (
            ToolErrorType.MISSING_CONFIG
            if any(marker in message.lower() for marker in ("not set", "not connected", "reconnect", "missing"))
            else ToolErrorType.INVALID_PARAMS
        )
        return ToolResult.error_result(message, error_type)
    return ToolResult.error_result(str(exc), ToolErrorType.EXECUTION_FAILED)


class ConnectLifeIntegrationHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="connect_life_integration",
            description=(
                "Start or verify Notion, Home Assistant, Weather, or Trello setup. Secret inputs are "
                "environment variable names, never secret values. Call without required fields to get an exact setup card."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "connector_id": {"type": "string", "enum": ["notion", "home_assistant", "weather", "trello"]},
                    "capabilities": {"type": "array", "items": {"type": "string"}},
                    "token_env": {"type": "string", "description": "Explicit token environment variable name for Notion, Home Assistant, or Trello."},
                    "api_key_env": {"type": "string", "description": "Explicit Trello API-key environment variable name."},
                    "base_url": {"type": "string", "description": "Home Assistant base URL, without credentials."},
                    "location": {"type": "string", "description": "Weather default city or place."},
                    "account_key": {"type": "string"},
                    "display_name": {"type": "string"},
                    "source_channel": {"type": "string"},
                    "source_session_id": {"type": "string"},
                },
                "required": ["connector_id"],
            },
            category=ToolCategory.EXTERNAL,
            energy_cost=1,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts=SETUP_CONTEXTS,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from services.life_integrations import setup_life_integration

        connector_id = str(arguments.get("connector_id") or "").strip().lower().replace("-", "_")
        try:
            result = await setup_life_integration(
                _pool(context, self.spec.name),
                connector_id,
                arguments,
                source_channel=arguments.get("source_channel") or context.surface,
                source_session_id=arguments.get("source_session_id") or context.session_id,
            )
        except Exception as exc:
            return _error(exc, connector_id or "integration")
        return ToolResult.success_result(result, display_output=str(result.get("next_step") or result.get("user_next_step") or "Setup updated."))


class ConnectSpotifyHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="connect_spotify",
            description=(
                "Start Spotify Authorization Code + PKCE setup. Ask for the least capabilities needed. "
                "The app client ID is non-secret; use client_id_env only after the user explicitly selects that environment variable."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "capabilities": {"type": "array", "items": {"type": "string"}},
                    "client_id": {"type": "string", "description": "Spotify app client ID (not a secret)."},
                    "client_id_env": {"type": "string", "description": "Explicitly selected environment variable containing the Spotify app client ID."},
                    "source_channel": {"type": "string"},
                    "source_session_id": {"type": "string"},
                },
            },
            category=ToolCategory.EXTERNAL,
            energy_cost=1,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts=SETUP_CONTEXTS,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from core.auth.spotify import SpotifyOAuthError, SpotifyOAuthStart, start_spotify_oauth
        from services.life_integrations import connector_manifest

        pool = _pool(context, self.spec.name)
        try:
            started = await start_spotify_oauth(
                pool,
                capabilities=arguments.get("capabilities"),
                client_id=arguments.get("client_id"),
                client_id_env=arguments.get("client_id_env"),
                source_channel=arguments.get("source_channel") or context.surface,
                source_session_id=arguments.get("source_session_id") or context.session_id,
            )
            manifest = await connector_manifest(pool, "spotify")
        except SpotifyOAuthError as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.MISSING_CONFIG)
        except Exception as exc:
            return _error(exc, "Spotify")

        if isinstance(started, dict):
            payload = started
            payload["ui"] = {
                "kind": "connector_setup",
                "version": 2,
                "id": "connector_setup:spotify:needs_client",
                "connector_id": "spotify",
                "display_name": "Spotify",
                "title": "Connect Spotify",
                "status": "needs_client",
                "capabilities": list(arguments.get("capabilities") or manifest["setup_manifest"].get("default_capabilities") or []),
                "credential_fields": list(manifest["setup_manifest"].get("credential_fields") or []),
                "redirect_uri": payload.get("redirect_uri"),
                "docs_url": manifest.get("docs_url"),
                "next_step": payload.get("next_step"),
                "safety_note": "Playback control remains separately approval-gated after connection.",
            }
            return ToolResult.success_result(payload, display_output=str(payload["next_step"]))

        assert isinstance(started, SpotifyOAuthStart)
        payload = started.attempt_payload
        payload["ui"] = {
            "kind": "connector_setup",
            "version": 2,
            "id": f"connector_setup:spotify:{payload['attempt_id']}",
            "connector_id": "spotify",
            "display_name": "Spotify",
            "title": "Connect Spotify",
            "status": "pending_authorization",
            "capabilities": list(payload.get("requested_capabilities") or []),
            "authorization_url": payload.get("authorization_url"),
            "redirect_uri": payload.get("redirect_uri"),
            "attempt_id": payload.get("attempt_id"),
            "completion_mode": "automatic_callback_or_paste",
            "manual_completion_available": True,
            "docs_url": manifest.get("docs_url"),
            "next_step": payload.get("user_next_step"),
            "safety_note": "Spotify shows the exact scopes before approval. Playback changes remain separately approval-gated.",
        }
        return ToolResult.success_result(
            payload,
            display_output=(
                "Spotify sign-in started. Open the authorization URL and approve the selected powers; "
                "Hexis completes setup when Spotify returns to the local callback."
            ),
        )


class CompleteSpotifyConnectionHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="complete_spotify_connection",
            description="Complete a pending Spotify PKCE setup from the full callback URL or raw code. Normally the local callback does this automatically.",
            parameters={
                "type": "object",
                "properties": {
                    "authorization_response": {"type": "string"},
                    "attempt_id": {"type": "string"},
                },
                "required": ["authorization_response"],
            },
            category=ToolCategory.EXTERNAL,
            energy_cost=1,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts=SETUP_CONTEXTS,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from core.auth.spotify import SpotifyOAuthError, complete_spotify_oauth

        try:
            completed = await complete_spotify_oauth(
                _pool(context, self.spec.name),
                authorization_response=str(arguments.get("authorization_response") or ""),
                attempt_id=arguments.get("attempt_id"),
            )
        except SpotifyOAuthError as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.AUTH_FAILED)
        payload = {
            "connector_id": "spotify",
            "account_key": completed.account_key,
            "display_name": completed.display_name,
            "credential_ref": completed.credential_ref,
            "granted_scopes": completed.granted_scopes,
            "capabilities": completed.capabilities,
            "status": "connected",
            "ui": {
                "kind": "connector_setup",
                "version": 2,
                "id": f"connector_setup:spotify:connected:{completed.account_key}",
                "connector_id": "spotify",
                "display_name": "Spotify",
                "title": "Spotify connected",
                "status": "connected",
                "capabilities": completed.capabilities,
                "connected_accounts": [{"account_key": completed.account_key, "display_name": completed.display_name, "status": "connected"}],
                "next_step": "Spotify is ready. Catalog search and playback-state reads are available; playback changes remain approval-gated.",
            },
        }
        return ToolResult.success_result(payload, display_output=f"Spotify connected as {completed.display_name}.")


class RevokeLifeIntegrationHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="revoke_life_integration",
            description="Revoke a Notion, Spotify, Home Assistant, Weather, or Trello connection. Spotify access tokens are also removed from the private auth store.",
            parameters={
                "type": "object",
                "properties": {
                    "connector_id": {"type": "string", "enum": ["notion", "spotify", "home_assistant", "weather", "trello"]},
                    "account_key": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["connector_id"],
            },
            category=ToolCategory.EXTERNAL,
            energy_cost=1,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts=SETUP_CONTEXTS,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        connector_id = str(arguments.get("connector_id") or "").strip().lower().replace("-", "_")
        if connector_id not in LIFE_CONNECTOR_IDS:
            return ToolResult.error_result(
                f"connector_id must be one of: {', '.join(sorted(LIFE_CONNECTOR_IDS))}.",
                ToolErrorType.INVALID_PARAMS,
            )
        try:
            async with _pool(context, self.spec.name).acquire() as conn:
                raw = await conn.fetchval(
                    "SELECT revoke_integration_connection($1, $2, $3)",
                    connector_id,
                    arguments.get("account_key"),
                    arguments.get("reason") or "revoked by user",
                )
            payload = json.loads(raw) if isinstance(raw, str) else raw
            if connector_id == "spotify" and int((payload or {}).get("revoked") or 0) > 0:
                from core.auth.spotify import delete_spotify_credentials

                delete_spotify_credentials()
        except Exception as exc:
            return _error(exc, connector_id)
        return ToolResult.success_result(payload, display_output=f"{connector_id} connections revoked: {(payload or {}).get('revoked', 0)}.")


class NotionSearchHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="notion_search",
            description="Search pages and data sources shared with the connected Notion integration.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}, "object_type": {"type": "string", "enum": ["page", "data_source"]}, "page_size": {"type": "integer", "default": 20}, "start_cursor": {"type": "string"}, "account_key": {"type": "string"}}, "required": ["query"]},
            category=ToolCategory.EXTERNAL,
            energy_cost=2,
            is_read_only=True,
            supports_parallel=True,
            allowed_contexts=ALL_CONTEXTS,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from services.life_integrations import notion_search
        try:
            result = await notion_search(_pool(context, self.spec.name), query=str(arguments.get("query") or ""), object_type=arguments.get("object_type"), page_size=int(arguments.get("page_size") or 20), start_cursor=arguments.get("start_cursor"), account_key=arguments.get("account_key"))
        except Exception as exc:
            return _error(exc, "Notion")
        count = len(result.get("results") or []) if isinstance(result, dict) else 0
        return ToolResult.success_result(result, display_output=f"Notion returned {count} result(s).")


class NotionGetPageHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="notion_get_page",
            description="Retrieve a Notion page and, by default, its first page of content blocks.",
            parameters={"type": "object", "properties": {"page_id": {"type": "string"}, "include_blocks": {"type": "boolean", "default": True}, "page_size": {"type": "integer", "default": 100}, "account_key": {"type": "string"}}, "required": ["page_id"]},
            category=ToolCategory.EXTERNAL,
            energy_cost=2,
            is_read_only=True,
            supports_parallel=True,
            allowed_contexts=ALL_CONTEXTS,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from services.life_integrations import notion_get_page
        try:
            result = await notion_get_page(_pool(context, self.spec.name), page_id=str(arguments.get("page_id") or ""), include_blocks=bool(arguments.get("include_blocks", True)), page_size=int(arguments.get("page_size") or 100), account_key=arguments.get("account_key"))
        except Exception as exc:
            return _error(exc, "Notion")
        return ToolResult.success_result(result, display_output=f"Notion page {arguments.get('page_id')} retrieved.")


class NotionQueryDataSourceHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="notion_query_data_source",
            description="Query a Notion data source using optional native Notion filter and sort JSON.",
            parameters={"type": "object", "properties": {"data_source_id": {"type": "string"}, "filter": {"type": "object"}, "sorts": {"type": "array", "items": {"type": "object"}}, "page_size": {"type": "integer", "default": 100}, "start_cursor": {"type": "string"}, "account_key": {"type": "string"}}, "required": ["data_source_id"]},
            category=ToolCategory.EXTERNAL,
            energy_cost=2,
            is_read_only=True,
            supports_parallel=True,
            allowed_contexts=ALL_CONTEXTS,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from services.life_integrations import notion_query_data_source
        try:
            result = await notion_query_data_source(_pool(context, self.spec.name), data_source_id=str(arguments.get("data_source_id") or ""), filter_value=arguments.get("filter"), sorts=arguments.get("sorts"), page_size=int(arguments.get("page_size") or 100), start_cursor=arguments.get("start_cursor"), account_key=arguments.get("account_key"))
        except Exception as exc:
            return _error(exc, "Notion")
        count = len(result.get("results") or []) if isinstance(result, dict) else 0
        return ToolResult.success_result(result, display_output=f"Notion data source returned {count} row(s).")


class NotionCreatePageHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="notion_create_page",
            description="Create a Notion page beneath a shared page or data source. This is an external write and always requires approval.",
            parameters={"type": "object", "properties": {"parent_id": {"type": "string"}, "parent_type": {"type": "string", "enum": ["page_id", "data_source_id"], "default": "page_id"}, "properties": {"type": "object"}, "children": {"type": "array", "items": {"type": "object"}}, "icon": {"type": "object"}, "cover": {"type": "object"}, "account_key": {"type": "string"}}, "required": ["parent_id", "properties"]},
            category=ToolCategory.EXTERNAL,
            energy_cost=3,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts=ALL_CONTEXTS,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from services.life_integrations import notion_create_page
        try:
            result = await notion_create_page(_pool(context, self.spec.name), parent_id=str(arguments.get("parent_id") or ""), parent_type=str(arguments.get("parent_type") or "page_id"), properties=dict(arguments.get("properties") or {}), children=arguments.get("children"), icon=arguments.get("icon"), cover=arguments.get("cover"), account_key=arguments.get("account_key"))
        except Exception as exc:
            return _error(exc, "Notion")
        return ToolResult.success_result(result, display_output=f"Notion page created: {result.get('url') or result.get('id') if isinstance(result, dict) else 'complete'}")


class SpotifySearchHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="spotify_search",
            description="Search the Spotify catalog. Spotify content must not be ingested into training or long-term model datasets.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}, "types": {"type": "array", "items": {"type": "string", "enum": ["album", "artist", "playlist", "track", "show", "episode", "audiobook"]}}, "limit": {"type": "integer", "default": 5}, "offset": {"type": "integer", "default": 0}, "market": {"type": "string"}, "account_key": {"type": "string"}}, "required": ["query"]},
            category=ToolCategory.EXTERNAL,
            energy_cost=2,
            is_read_only=True,
            supports_parallel=True,
            allowed_contexts=ALL_CONTEXTS,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from core.auth.spotify import SpotifyOAuthError, spotify_api_request
        types = arguments.get("types") or ["track", "artist", "album", "playlist"]
        allowed = {"album", "artist", "playlist", "track", "show", "episode", "audiobook"}
        if not isinstance(types, list) or not types or any(str(item) not in allowed for item in types):
            return ToolResult.error_result("Spotify types contains an unsupported item type.", ToolErrorType.INVALID_PARAMS)
        params: dict[str, Any] = {"q": str(arguments.get("query") or ""), "type": ",".join(dict.fromkeys(str(item) for item in types)), "limit": max(1, min(int(arguments.get("limit") or 5), 10)), "offset": max(0, min(int(arguments.get("offset") or 0), 1000))}
        if arguments.get("market"):
            params["market"] = str(arguments["market"]).upper()
        try:
            result = await spotify_api_request(_pool(context, self.spec.name), "GET", "/search", capability="search", account_key=arguments.get("account_key"), params=params)
        except SpotifyOAuthError as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.AUTH_FAILED)
        except Exception as exc:
            return _error(exc, "Spotify")
        return ToolResult.success_result(result, display_output="Spotify catalog search complete.")


class SpotifyPlaybackStateHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="spotify_playback_state",
            description="Read current Spotify playback state and optionally available devices.",
            parameters={"type": "object", "properties": {"include_devices": {"type": "boolean", "default": True}, "market": {"type": "string"}, "account_key": {"type": "string"}}},
            category=ToolCategory.EXTERNAL,
            energy_cost=1,
            is_read_only=True,
            supports_parallel=True,
            allowed_contexts=ALL_CONTEXTS,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from core.auth.spotify import SpotifyOAuthError, spotify_api_request
        pool = _pool(context, self.spec.name)
        params = {"market": str(arguments["market"]).upper()} if arguments.get("market") else None
        try:
            state = await spotify_api_request(pool, "GET", "/me/player", capability="playback_state", account_key=arguments.get("account_key"), params=params)
            devices = await spotify_api_request(pool, "GET", "/me/player/devices", capability="playback_state", account_key=arguments.get("account_key")) if arguments.get("include_devices", True) else None
        except SpotifyOAuthError as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.AUTH_FAILED)
        except Exception as exc:
            return _error(exc, "Spotify")
        result = {"playback": state, "devices": devices}
        return ToolResult.success_result(result, display_output="Spotify playback state retrieved." if state else "Spotify has no active playback state.")


class SpotifyControlPlaybackHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="spotify_control_playback",
            description="Control Spotify playback. Most playback controls require Spotify Premium and every call is approval-gated.",
            parameters={"type": "object", "properties": {"action": {"type": "string", "enum": ["play", "pause", "next", "previous", "queue", "seek", "volume", "transfer", "shuffle", "repeat"]}, "device_id": {"type": "string"}, "uri": {"type": "string"}, "uris": {"type": "array", "items": {"type": "string"}}, "context_uri": {"type": "string"}, "position_ms": {"type": "integer"}, "volume_percent": {"type": "integer"}, "state": {"description": "Boolean for shuffle or off/context/track for repeat."}, "play": {"type": "boolean"}, "account_key": {"type": "string"}}, "required": ["action"]},
            category=ToolCategory.EXTERNAL,
            energy_cost=3,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts=ALL_CONTEXTS,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from core.auth.spotify import SpotifyOAuthError, spotify_api_request
        action = str(arguments.get("action") or "")
        if action not in SPOTIFY_PLAYBACK_ACTIONS:
            return ToolResult.error_result(
                f"action must be one of: {', '.join(sorted(SPOTIFY_PLAYBACK_ACTIONS))}.",
                ToolErrorType.INVALID_PARAMS,
            )
        device_id = arguments.get("device_id")
        params: dict[str, Any] = {"device_id": device_id} if device_id else {}
        body: dict[str, Any] | None = None
        method = "PUT"
        path = f"/me/player/{action}"
        if action == "play":
            path = "/me/player/play"
            body = {}
            for key in ("context_uri", "uris", "position_ms"):
                if arguments.get(key) is not None:
                    body[key] = arguments[key]
        elif action == "pause":
            path = "/me/player/pause"
        elif action in {"next", "previous"}:
            method = "POST"
        elif action == "queue":
            method = "POST"
            if not arguments.get("uri"):
                return ToolResult.error_result("uri is required for queue.", ToolErrorType.INVALID_PARAMS)
            params["uri"] = arguments["uri"]
        elif action == "seek":
            if arguments.get("position_ms") is None:
                return ToolResult.error_result("position_ms is required for seek.", ToolErrorType.INVALID_PARAMS)
            params["position_ms"] = max(0, int(arguments["position_ms"]))
        elif action == "volume":
            if arguments.get("volume_percent") is None:
                return ToolResult.error_result("volume_percent is required for volume.", ToolErrorType.INVALID_PARAMS)
            params["volume_percent"] = max(0, min(int(arguments["volume_percent"]), 100))
        elif action == "transfer":
            if not device_id:
                return ToolResult.error_result("device_id is required for transfer.", ToolErrorType.INVALID_PARAMS)
            path = "/me/player"
            params = {}
            body = {"device_ids": [device_id], "play": bool(arguments.get("play", False))}
        elif action == "shuffle":
            if not isinstance(arguments.get("state"), bool):
                return ToolResult.error_result("state must be true or false for shuffle.", ToolErrorType.INVALID_PARAMS)
            params["state"] = str(arguments["state"]).lower()
        elif action == "repeat":
            state = str(arguments.get("state") or "")
            if state not in {"off", "context", "track"}:
                return ToolResult.error_result("state must be off, context, or track for repeat.", ToolErrorType.INVALID_PARAMS)
            params["state"] = state
        try:
            result = await spotify_api_request(_pool(context, self.spec.name), method, path, capability="playback_control", account_key=arguments.get("account_key"), params=params or None, body=body)
        except SpotifyOAuthError as exc:
            return ToolResult.error_result(str(exc), ToolErrorType.AUTH_FAILED)
        except Exception as exc:
            return _error(exc, "Spotify")
        return ToolResult.success_result({"action": action, "provider_result": result}, display_output=f"Spotify playback action completed: {action}.")


class HomeAssistantStatesHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="home_assistant_states",
            description="Read one Home Assistant entity state or all entity states.",
            parameters={"type": "object", "properties": {"entity_id": {"type": "string"}, "account_key": {"type": "string"}}},
            category=ToolCategory.EXTERNAL,
            energy_cost=1,
            is_read_only=True,
            supports_parallel=True,
            allowed_contexts=ALL_CONTEXTS,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from services.life_integrations import home_assistant_states
        try:
            result = await home_assistant_states(_pool(context, self.spec.name), entity_id=arguments.get("entity_id"), account_key=arguments.get("account_key"))
        except Exception as exc:
            return _error(exc, "Home Assistant")
        count = len(result) if isinstance(result, list) else 1
        return ToolResult.success_result(result, display_output=f"Home Assistant returned {count} state(s).")


class HomeAssistantCallServiceHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="home_assistant_call_service",
            description="Call a Home Assistant service for an explicit domain/service and optional entity. This changes the home and always requires approval.",
            parameters={"type": "object", "properties": {"domain": {"type": "string"}, "service": {"type": "string"}, "entity_id": {"type": "string"}, "service_data": {"type": "object"}, "account_key": {"type": "string"}}, "required": ["domain", "service"]},
            category=ToolCategory.EXTERNAL,
            energy_cost=3,
            is_read_only=False,
            requires_approval=True,
            supports_parallel=False,
            allowed_contexts=ALL_CONTEXTS,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from services.life_integrations import home_assistant_call_service
        try:
            result = await home_assistant_call_service(_pool(context, self.spec.name), domain=str(arguments.get("domain") or ""), service=str(arguments.get("service") or ""), entity_id=arguments.get("entity_id"), service_data=arguments.get("service_data"), account_key=arguments.get("account_key"))
        except Exception as exc:
            return _error(exc, "Home Assistant")
        return ToolResult.success_result(result, display_output=f"Home Assistant service called: {arguments.get('domain')}.{arguments.get('service')}.")


class WeatherForecastHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="weather_forecast",
            description="Get current conditions and a 1–16 day Open-Meteo forecast for a named place, coordinates, or the connected default location.",
            parameters={"type": "object", "properties": {"location": {"type": "string"}, "latitude": {"type": "number"}, "longitude": {"type": "number"}, "days": {"type": "integer", "default": 7}, "account_key": {"type": "string"}}},
            category=ToolCategory.EXTERNAL,
            energy_cost=1,
            is_read_only=True,
            supports_parallel=True,
            allowed_contexts=ALL_CONTEXTS,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from services.life_integrations import weather_forecast
        try:
            result = await weather_forecast(_pool(context, self.spec.name), location=arguments.get("location"), latitude=arguments.get("latitude"), longitude=arguments.get("longitude"), days=int(arguments.get("days") or 7), account_key=arguments.get("account_key"))
        except Exception as exc:
            return _error(exc, "Weather")
        label = result.get("location", {}).get("name") or "requested coordinates"
        return ToolResult.success_result(result, display_output=f"Weather forecast retrieved for {label}.")


class TrelloListBoardsHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name="trello_list_boards", description="List open Trello boards and their open lists.", parameters={"type": "object", "properties": {"account_key": {"type": "string"}}}, category=ToolCategory.EXTERNAL, energy_cost=1, is_read_only=True, supports_parallel=True, allowed_contexts=ALL_CONTEXTS)

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from services.life_integrations import trello_list_boards
        try:
            result = await trello_list_boards(_pool(context, self.spec.name), account_key=arguments.get("account_key"))
        except Exception as exc:
            return _error(exc, "Trello")
        return ToolResult.success_result(result, display_output=f"Trello returned {len(result) if isinstance(result, list) else 0} open board(s).")


class TrelloListCardsHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name="trello_list_cards", description="List Trello cards on exactly one board or list.", parameters={"type": "object", "properties": {"board_id": {"type": "string"}, "list_id": {"type": "string"}, "filter": {"type": "string", "default": "open"}, "account_key": {"type": "string"}}}, category=ToolCategory.EXTERNAL, energy_cost=1, is_read_only=True, supports_parallel=True, allowed_contexts=ALL_CONTEXTS)

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from services.life_integrations import trello_list_cards
        try:
            result = await trello_list_cards(_pool(context, self.spec.name), board_id=arguments.get("board_id"), list_id=arguments.get("list_id"), card_filter=str(arguments.get("filter") or "open"), account_key=arguments.get("account_key"))
        except Exception as exc:
            return _error(exc, "Trello")
        return ToolResult.success_result(result, display_output=f"Trello returned {len(result) if isinstance(result, list) else 0} card(s).")


class TrelloCreateCardHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name="trello_create_card", description="Create a Trello card on a specific list. This external write always requires approval.", parameters={"type": "object", "properties": {"list_id": {"type": "string"}, "name": {"type": "string"}, "description": {"type": "string"}, "due": {"type": "string"}, "position": {"type": "string"}, "account_key": {"type": "string"}}, "required": ["list_id", "name"]}, category=ToolCategory.EXTERNAL, energy_cost=3, is_read_only=False, requires_approval=True, supports_parallel=False, allowed_contexts=ALL_CONTEXTS)

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from services.life_integrations import trello_create_card
        try:
            result = await trello_create_card(_pool(context, self.spec.name), list_id=str(arguments.get("list_id") or ""), name=str(arguments.get("name") or ""), description=arguments.get("description"), due=arguments.get("due"), position=arguments.get("position"), account_key=arguments.get("account_key"))
        except Exception as exc:
            return _error(exc, "Trello")
        return ToolResult.success_result(result, display_output=f"Trello card created: {result.get('name') if isinstance(result, dict) else arguments.get('name')}.")


class TrelloUpdateCardHandler(ToolHandler):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name="trello_update_card", description="Update selected fields on a Trello card. Closing or moving a card is explicit in changes and always approval-gated.", parameters={"type": "object", "properties": {"card_id": {"type": "string"}, "changes": {"type": "object", "description": "Any of name, desc, due, dueComplete, closed, idList, or pos."}, "account_key": {"type": "string"}}, "required": ["card_id", "changes"]}, category=ToolCategory.EXTERNAL, energy_cost=3, is_read_only=False, requires_approval=True, supports_parallel=False, allowed_contexts=ALL_CONTEXTS)

    async def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from services.life_integrations import trello_update_card
        try:
            result = await trello_update_card(_pool(context, self.spec.name), card_id=str(arguments.get("card_id") or ""), changes=dict(arguments.get("changes") or {}), account_key=arguments.get("account_key"))
        except Exception as exc:
            return _error(exc, "Trello")
        return ToolResult.success_result(result, display_output=f"Trello card updated: {arguments.get('card_id')}.")


def create_life_integration_tools() -> list[ToolHandler]:
    return [
        ConnectLifeIntegrationHandler(),
        ConnectSpotifyHandler(),
        CompleteSpotifyConnectionHandler(),
        RevokeLifeIntegrationHandler(),
        NotionSearchHandler(),
        NotionGetPageHandler(),
        NotionQueryDataSourceHandler(),
        NotionCreatePageHandler(),
        SpotifySearchHandler(),
        SpotifyPlaybackStateHandler(),
        SpotifyControlPlaybackHandler(),
        HomeAssistantStatesHandler(),
        HomeAssistantCallServiceHandler(),
        WeatherForecastHandler(),
        TrelloListBoardsHandler(),
        TrelloListCardsHandler(),
        TrelloCreateCardHandler(),
        TrelloUpdateCardHandler(),
    ]
