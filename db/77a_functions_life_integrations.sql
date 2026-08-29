-- Everyday-life API connector manifests (PLAN.md Wave B).
--
-- Provider secrets remain outside Postgres. Connections record only the exact
-- environment variable names the user selected, or a private auth-store key.
SET search_path = public, ag_catalog, "$user";

INSERT INTO integration_connectors (
    id,
    display_name,
    category,
    auth_type,
    status,
    capability_manifest,
    setup_manifest,
    docs_url,
    metadata
) VALUES
(
    'notion',
    'Notion',
    'productivity',
    'api_key',
    'available',
    '{
      "search": {"label": "Search shared pages and data sources", "scope_kind": "read", "status": "available", "scopes": ["read_content"]},
      "read": {"label": "Read shared pages and blocks", "scope_kind": "read", "status": "available", "scopes": ["read_content"]},
      "query": {"label": "Query shared data sources", "scope_kind": "read", "status": "available", "scopes": ["read_content"]},
      "create": {"label": "Create pages in shared parents", "scope_kind": "write", "status": "available", "scopes": ["insert_content"]}
    }'::jsonb,
    '{
      "flow": "explicit_env_reference",
      "secret_storage": "environment",
      "supported_surfaces": ["chat", "cli", "web", "mcp"],
      "default_capabilities": ["search", "read", "query"],
      "capability_order": ["search", "read", "query", "create"],
      "required_scopes": [],
      "scope_order": ["read_content", "insert_content"],
      "api_base_url": "https://api.notion.com",
      "api_version": "2026-03-11",
      "credential_fields": [{"name": "token_env", "label": "Notion token environment variable", "secret": true, "example": "NOTION_TOKEN"}],
      "capability_aliases": {"pages": "read", "blocks": "read", "databases": "query", "data_sources": "query", "write": "create"},
      "user_next_step": "Create a Notion internal integration, share the pages or data sources it may use with that integration, put the token in an environment variable, then connect Notion with that environment variable name."
    }'::jsonb,
    'https://developers.notion.com/docs/create-a-notion-integration',
    '{"provider": "notion", "seeded_by": "db/77a_functions_life_integrations.sql"}'::jsonb
),
(
    'spotify',
    'Spotify',
    'media',
    'oauth2',
    'available',
    '{
      "search": {"label": "Search the Spotify catalog", "scope_kind": "read", "status": "available", "scopes": ["user-read-private"]},
      "playback_state": {"label": "Read playback state and devices", "scope_kind": "read", "status": "available", "scopes": ["user-read-playback-state"]},
      "playback_control": {"label": "Control playback", "scope_kind": "modify", "status": "available", "scopes": ["user-modify-playback-state"]}
    }'::jsonb,
    '{
      "flow": "oauth2_authorization_code_pkce",
      "secret_storage": "~/.hexis/auth",
      "supported_surfaces": ["chat", "cli", "web", "mcp"],
      "default_capabilities": ["search", "playback_state"],
      "capability_order": ["search", "playback_state", "playback_control"],
      "required_scopes": [],
      "scope_order": ["user-read-private", "user-read-playback-state", "user-modify-playback-state"],
      "authorize_url": "https://accounts.spotify.com/authorize",
      "token_url": "https://accounts.spotify.com/api/token",
      "api_base_url": "https://api.spotify.com/v1",
      "callback_path": "/api/integrations/spotify/callback",
      "credential_fields": [
        {"name": "client_id", "label": "Spotify app client ID", "secret": false},
        {"name": "client_id_env", "label": "Spotify client-ID environment variable", "secret": false, "example": "SPOTIFY_CLIENT_ID"}
      ],
      "capability_aliases": {"catalog": "search", "player": "playback_state", "playback": "playback_state", "control": "playback_control", "play": "playback_control"},
      "user_next_step": "Create a Spotify app, add the exact loopback callback URI shown by Hexis, then connect with the app client ID or an explicitly selected client-ID environment variable."
    }'::jsonb,
    'https://developer.spotify.com/documentation/web-api/concepts/apps',
    '{"provider": "spotify", "seeded_by": "db/77a_functions_life_integrations.sql"}'::jsonb
),
(
    'home_assistant',
    'Home Assistant',
    'home',
    'api_key',
    'available',
    '{
      "states": {"label": "Read entity states", "scope_kind": "read", "status": "available", "scopes": []},
      "service_control": {"label": "Call Home Assistant services", "scope_kind": "modify", "status": "available", "scopes": []}
    }'::jsonb,
    '{
      "flow": "explicit_env_reference",
      "secret_storage": "environment",
      "supported_surfaces": ["chat", "cli", "web", "mcp"],
      "default_capabilities": ["states"],
      "capability_order": ["states", "service_control"],
      "required_scopes": [],
      "scope_order": [],
      "credential_fields": [
        {"name": "base_url", "label": "Home Assistant URL", "secret": false, "example": "http://homeassistant.local:8123"},
        {"name": "token_env", "label": "Long-lived access-token environment variable", "secret": true, "example": "HOME_ASSISTANT_TOKEN"}
      ],
      "capability_aliases": {"read": "states", "entities": "states", "services": "service_control", "control": "service_control", "write": "service_control"},
      "user_next_step": "Create a long-lived access token in the Home Assistant profile, store it in an environment variable, then connect using the Home Assistant base URL and that environment variable name."
    }'::jsonb,
    'https://developers.home-assistant.io/docs/api/rest/',
    '{"provider": "home_assistant", "seeded_by": "db/77a_functions_life_integrations.sql"}'::jsonb
),
(
    'weather',
    'Weather',
    'information',
    'manual',
    'available',
    '{
      "forecast": {"label": "Current conditions and forecast", "scope_kind": "read", "status": "available", "scopes": []}
    }'::jsonb,
    '{
      "flow": "location_verification",
      "secret_storage": "none",
      "supported_surfaces": ["chat", "cli", "web", "heartbeat", "mcp"],
      "default_capabilities": ["forecast"],
      "capability_order": ["forecast"],
      "required_scopes": [],
      "scope_order": [],
      "geocoding_base_url": "https://geocoding-api.open-meteo.com/v1",
      "forecast_base_url": "https://api.open-meteo.com/v1",
      "credential_fields": [{"name": "location", "label": "Default location", "secret": false, "example": "Boston, MA"}],
      "capability_aliases": {"current": "forecast", "conditions": "forecast"},
      "user_next_step": "Choose a default city or place. Hexis will verify a forecast for the best provider match, save that reversible default, and show the matched place; individual forecast calls can still use another location."
    }'::jsonb,
    'https://open-meteo.com/en/docs',
    '{"provider": "open_meteo", "seeded_by": "db/77a_functions_life_integrations.sql"}'::jsonb
),
(
    'trello',
    'Trello',
    'productivity',
    'api_key',
    'available',
    '{
      "boards": {"label": "List boards and lists", "scope_kind": "read", "status": "available", "scopes": ["read"]},
      "cards": {"label": "Read cards", "scope_kind": "read", "status": "available", "scopes": ["read"]},
      "create_card": {"label": "Create cards", "scope_kind": "write", "status": "available", "scopes": ["write"]},
      "update_card": {"label": "Update cards", "scope_kind": "write", "status": "available", "scopes": ["write"]}
    }'::jsonb,
    '{
      "flow": "explicit_env_reference",
      "secret_storage": "environment",
      "supported_surfaces": ["chat", "cli", "web", "mcp"],
      "default_capabilities": ["boards", "cards"],
      "capability_order": ["boards", "cards", "create_card", "update_card"],
      "required_scopes": [],
      "scope_order": ["read", "write"],
      "api_base_url": "https://api.trello.com/1",
      "credential_fields": [
        {"name": "api_key_env", "label": "Trello API-key environment variable", "secret": true, "example": "TRELLO_API_KEY"},
        {"name": "token_env", "label": "Trello token environment variable", "secret": true, "example": "TRELLO_TOKEN"}
      ],
      "capability_aliases": {"lists": "boards", "read": "cards", "create": "create_card", "update": "update_card", "write": "create_card"},
      "user_next_step": "Get the API key for your Trello Power-Up, authorize a token with only the requested read/write powers, store both in environment variables, then connect using those environment variable names."
    }'::jsonb,
    'https://developer.atlassian.com/cloud/trello/guides/rest-api/authorization/',
    '{"provider": "trello", "seeded_by": "db/77a_functions_life_integrations.sql"}'::jsonb
)
ON CONFLICT (id) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    category = EXCLUDED.category,
    auth_type = EXCLUDED.auth_type,
    status = EXCLUDED.status,
    capability_manifest = EXCLUDED.capability_manifest,
    setup_manifest = EXCLUDED.setup_manifest,
    docs_url = EXCLUDED.docs_url,
    metadata = integration_connectors.metadata || EXCLUDED.metadata,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO connector_action_tool_map (
    tool_name,
    connector_id,
    action_kind,
    target_argument,
    account_argument,
    sensitivity,
    metadata
) VALUES
    ('notion_create_page', 'notion', 'create_page', 'parent_id', 'account_key', 'provider_state_change',
     '{"tool_module": "core.tools.life_integrations", "provider_endpoint": "POST /v1/pages"}'::jsonb),
    ('spotify_control_playback', 'spotify', 'control_playback', 'action', 'account_key', 'provider_state_change',
     '{"tool_module": "core.tools.life_integrations", "provider_endpoint": "player action selected at runtime"}'::jsonb),
    ('home_assistant_call_service', 'home_assistant', 'call_service', 'entity_id', 'account_key', 'provider_state_change',
     '{"tool_module": "core.tools.life_integrations", "fallback_target_argument": "domain"}'::jsonb),
    ('trello_create_card', 'trello', 'create_card', 'list_id', 'account_key', 'provider_state_change',
     '{"tool_module": "core.tools.life_integrations", "provider_endpoint": "POST /1/cards"}'::jsonb),
    ('trello_update_card', 'trello', 'update_card', 'card_id', 'account_key', 'provider_state_change',
     '{"tool_module": "core.tools.life_integrations", "provider_endpoint": "PUT /1/cards/{id}"}'::jsonb)
ON CONFLICT (tool_name) DO UPDATE SET
    connector_id = EXCLUDED.connector_id,
    action_kind = EXCLUDED.action_kind,
    target_argument = EXCLUDED.target_argument,
    account_argument = EXCLUDED.account_argument,
    sensitivity = EXCLUDED.sensitivity,
    enabled = TRUE,
    metadata = connector_action_tool_map.metadata || EXCLUDED.metadata,
    updated_at = CURRENT_TIMESTAMP;
