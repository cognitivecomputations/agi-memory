-- First-class integration connector setup and status.
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
) VALUES (
    'gmail',
    'Gmail',
    'personal_data',
    'oauth2',
    'available',
    '{
      "read": {
        "label": "Read messages",
        "scope_kind": "read",
        "status": "available",
        "scopes": ["https://www.googleapis.com/auth/gmail.readonly"]
      },
      "search": {
        "label": "Search messages",
        "scope_kind": "read",
        "status": "available",
        "scopes": ["https://www.googleapis.com/auth/gmail.readonly"]
      },
      "ingest": {
        "label": "Remember what is read",
        "scope_kind": "local",
        "status": "available",
        "scopes": []
      },
      "label": {
        "label": "Apply/remove labels",
        "scope_kind": "modify",
        "status": "available",
        "scopes": ["https://www.googleapis.com/auth/gmail.modify"]
      },
      "spam_triage": {
        "label": "Triage spam and inbox labels",
        "scope_kind": "modify",
        "status": "available",
        "scopes": ["https://www.googleapis.com/auth/gmail.modify"]
      },
      "send": {
        "label": "Send new messages",
        "scope_kind": "send",
        "status": "available",
        "scopes": ["https://www.googleapis.com/auth/gmail.send"]
      },
      "reply": {
        "label": "Reply in existing threads",
        "scope_kind": "send",
        "status": "available",
        "scopes": ["https://www.googleapis.com/auth/gmail.send"]
      },
      "delete": {
        "label": "Delete messages",
        "scope_kind": "destructive",
        "status": "available",
        "scopes": ["https://www.googleapis.com/auth/gmail.modify"]
      }
    }'::jsonb,
    '{
      "flow": "oauth2_authorization_code_pkce",
      "redirect_uri": "http://localhost",
      "requires_user_client_secret": false,
      "preferred_credential_mode": "hosted_oauth",
	      "credential_modes": ["hosted_oauth", "advanced_self_hosted", "upload_json", "path", "configured_env"],
	      "hosted_oauth_configured": false,
	      "setup_label": "Gmail sign-in",
	      "secret_storage": "~/.hexis/auth",
	      "supported_surfaces": ["chat", "cli", "web", "channels"],
	      "default_capabilities": ["read", "search"],
      "capability_order": ["read", "search", "ingest", "label", "spam_triage", "send", "reply", "delete"],
      "required_scopes": ["https://www.googleapis.com/auth/userinfo.email"],
      "scope_order": [
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.modify",
        "https://www.googleapis.com/auth/gmail.send"
      ],
      "capability_aliases": {
        "email": "read",
        "emails": "read",
        "mail": "read",
        "gmail": "read",
        "learn": "ingest",
        "memory": "ingest",
        "remember": "ingest",
        "retain": "ingest",
        "store": "ingest",
        "filter": "spam_triage",
        "filter_spam": "spam_triage",
        "spam": "spam_triage",
        "modify": "label",
        "labels": "label",
	        "write": "send",
	        "respond": "reply"
	      },
	      "setup_steps": [
	        "Open the Google setup page.",
	        "Create or choose a project named Hexis.",
	        "Enable the Gmail API for that project.",
	        "Set up the app consent screen. For a personal Gmail account, choose External and add your Gmail address as a test user if Google asks.",
	        "On the Credentials page, click Create credentials, choose Google''s sign-in client option, set Application type to Desktop app, and name it Hexis.",
	        "Download the setup file Google gives you.",
	        "Upload that setup file here, then start Google sign-in."
	      ],
	      "technical_next_step": "For developers and hosted builds: configure HEXIS_GMAIL_OAUTH_CLIENT_ID and HEXIS_GMAIL_OAUTH_CLIENT_SECRET in the Hexis API environment to make built-in Google sign-in available.",
	      "notes": [
	        "Use the structured Gmail setup UI to choose provider powers and email memory policy before Google sign-in.",
	        "The default user path is built-in Google sign-in when configured for the build.",
	        "If this local build needs setup, the UI should walk the user through Google Cloud step by step and accept the downloaded setup file.",
	        "CLI surfaces can accept a local JSON path directly after the guide explains what file to download.",
	        "Paste the full localhost redirect URL back into the conversation.",
	        "Long-lived tokens are stored outside Postgres in the private auth store."
	      ]
    }'::jsonb,
    'https://console.cloud.google.com/apis/credentials',
    '{"provider": "google", "seeded_by": "db/77_functions_integrations.sql"}'::jsonb
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

INSERT INTO config_defaults (key, value, description) VALUES
    (
        'integrations.gmail.memory_policy',
        '"ask"'::jsonb,
        'Controls whether Gmail reads may feed Hexis ingestion and memory by default; ask until the user chooses remember or forget.'
    ),
    (
        'integrations.gmail.heartbeat_digest_enabled',
        'false'::jsonb,
        'Controls whether heartbeat may proactively check connected Gmail for digests or important messages without a live user turn.'
    )
ON CONFLICT (key) DO UPDATE SET
    value = EXCLUDED.value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

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
    'slack',
    'Slack',
    'communication',
    'api_key',
    'available',
    '{
      "live_chat": {"label": "Receive live Slack messages", "status": "available", "scopes": ["app_mentions:read", "channels:history"]},
      "send": {"label": "Send Slack messages and private approval DMs", "status": "available", "scopes": ["chat:write", "im:write"]},
      "ingest_live": {"label": "Preserve and ingest live Slack messages", "status": "available", "scopes": ["app_mentions:read", "channels:history"]},
      "media": {"label": "Read Slack file metadata from live events", "status": "available", "scopes": ["files:read"]},
      "backfill": {"label": "Import historical Slack messages", "status": "available", "scopes": ["channels:history", "groups:history", "im:history", "mpim:history"]},
      "admin": {"label": "Administer Slack workspace settings", "status": "planned", "scopes": []}
    }'::jsonb,
    '{
      "flow": "manual_channel_token_config",
      "secret_storage": "environment",
      "supported_surfaces": ["chat", "cli", "web", "channels"],
      "default_capabilities": ["live_chat", "send", "ingest_live"],
      "capability_order": ["live_chat", "send", "ingest_live", "media", "backfill", "admin"],
      "required_scopes": [],
      "scope_order": ["app_mentions:read", "channels:history", "chat:write", "im:write", "files:read", "groups:history", "im:history", "mpim:history"],
      "config_keys": ["channel.slack.bot_token", "channel.slack.app_token", "channel.slack.signing_secret", "channel.slack.operator_user_id", "channel.slack.allowed_channels"],
      "env_vars": ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_SIGNING_SECRET"],
      "capability_aliases": {
        "read": "live_chat",
        "receive": "live_chat",
        "listen": "live_chat",
        "message": "send",
        "messages": "live_chat",
        "write": "send",
        "history": "backfill",
        "ingest": "ingest_live"
      },
      "user_next_step": "Create or choose a Slack app, enable Socket Mode, set SLACK_BOT_TOKEN to a bot token env var and SLACK_APP_TOKEN to an app-level token env var, then verify this connector. The channel worker starts Slack when those values resolve.",
      "notes": [
        "Store token values in environment variables, not in chat or Postgres.",
        "Use channel.slack.allowed_channels to restrict where Hexis listens.",
        "Set channel.slack.operator_user_id to your U... user ID for private protected-tool approvals and standing instructions."
      ]
    }'::jsonb,
    'https://api.slack.com/start/quickstart',
    '{"provider": "slack", "seeded_by": "db/77_functions_integrations.sql"}'::jsonb
),
(
    'telegram',
    'Telegram',
    'communication',
    'api_key',
    'available',
    '{
      "live_chat": {"label": "Receive live Telegram messages", "status": "available", "scopes": []},
      "send": {"label": "Send Telegram messages", "status": "available", "scopes": []},
      "ingest_live": {"label": "Preserve and ingest live Telegram messages", "status": "available", "scopes": []},
      "media": {"label": "Read Telegram attachment metadata from live events", "status": "available", "scopes": []},
      "backfill": {"label": "Import historical Telegram messages", "status": "planned", "scopes": []}
    }'::jsonb,
    '{
      "flow": "manual_channel_token_config",
      "secret_storage": "environment",
      "supported_surfaces": ["chat", "cli", "web", "channels"],
      "default_capabilities": ["live_chat", "send", "ingest_live"],
      "capability_order": ["live_chat", "send", "ingest_live", "media", "backfill"],
      "required_scopes": [],
      "scope_order": [],
      "config_keys": ["channel.telegram.bot_token", "channel.telegram.operator_user_id", "channel.telegram.allowed_chat_ids"],
      "env_vars": ["TELEGRAM_BOT_TOKEN"],
      "capability_aliases": {
        "read": "live_chat",
        "receive": "live_chat",
        "listen": "live_chat",
        "message": "send",
        "messages": "live_chat",
        "write": "send",
        "history": "backfill",
        "ingest": "ingest_live"
      },
      "user_next_step": "Create a Telegram bot with BotFather, store the bot token in TELEGRAM_BOT_TOKEN or another env var, set channel.telegram.bot_token to that env var name if needed, then verify this connector. The channel worker starts Telegram when the token resolves.",
      "notes": [
        "Store bot token values in environment variables, not in chat or Postgres.",
        "Use channel.telegram.allowed_chat_ids to restrict where Hexis listens.",
        "Set channel.telegram.operator_user_id to your numeric user ID if private messages may create standing instructions."
      ]
    }'::jsonb,
    'https://core.telegram.org/bots/tutorial',
    '{"provider": "telegram", "seeded_by": "db/77_functions_integrations.sql"}'::jsonb
),
(
    'signal',
    'Signal',
    'communication',
    'pairing',
    'available',
    '{
      "live_chat": {"label": "Receive live Signal messages", "status": "available", "scopes": []},
      "send": {"label": "Send Signal messages", "status": "available", "scopes": []},
      "ingest_live": {"label": "Preserve and ingest live Signal messages", "status": "available", "scopes": []},
      "media": {"label": "Read Signal attachment metadata from live events", "status": "available", "scopes": []},
      "backfill": {"label": "Import historical Signal messages", "status": "planned", "scopes": []}
    }'::jsonb,
    '{
      "flow": "signal_cli_rest_sidecar",
      "secret_storage": "environment",
      "supported_surfaces": ["chat", "cli", "web", "channels"],
      "default_capabilities": ["live_chat", "send", "ingest_live"],
      "capability_order": ["live_chat", "send", "ingest_live", "media", "backfill"],
      "required_scopes": [],
      "scope_order": [],
      "config_keys": ["channel.signal.phone_number", "channel.signal.api_url", "channel.signal.operator_user_id", "channel.signal.allowed_numbers"],
      "env_vars": ["SIGNAL_PHONE_NUMBER", "SIGNAL_API_URL"],
      "capability_aliases": {
        "read": "live_chat",
        "receive": "live_chat",
        "listen": "live_chat",
        "message": "send",
        "messages": "live_chat",
        "write": "send",
        "history": "backfill",
        "ingest": "ingest_live"
      },
      "user_next_step": "Run or connect a signal-cli-rest-api sidecar, register/link the Signal phone number, set SIGNAL_PHONE_NUMBER and optionally SIGNAL_API_URL, then verify this connector. The channel worker starts Signal when the phone number resolves.",
      "notes": [
        "Signal protocol state lives in the sidecar, not in Postgres.",
        "Use channel.signal.allowed_numbers to restrict who Hexis listens to.",
        "Set channel.signal.operator_user_id to your number if private messages may create standing instructions."
      ]
    }'::jsonb,
    'https://github.com/bbernhard/signal-cli-rest-api',
    '{"provider": "signal", "seeded_by": "db/77_functions_integrations.sql"}'::jsonb
),
(
    'twitter_x',
    'Twitter/X',
    'communication',
    'oauth2',
    'available',
    '{
      "read": {"label": "Read posts, mentions, and basic account identity", "scope_kind": "read", "status": "available", "scopes": ["tweet.read", "users.read", "offline.access"]},
      "search": {"label": "Search recent posts", "scope_kind": "read", "status": "available", "scopes": ["tweet.read", "users.read", "offline.access"]},
      "ingest": {"label": "Ingest live posts and mentions", "scope_kind": "read", "status": "available", "scopes": ["tweet.read", "users.read", "offline.access"]},
      "dm_read": {"label": "Read Direct Messages", "scope_kind": "read_private", "status": "available", "scopes": ["dm.read", "tweet.read", "users.read", "offline.access"]},
      "send": {"label": "Create posts and replies", "scope_kind": "send", "status": "available", "scopes": ["tweet.read", "tweet.write", "users.read", "offline.access"]},
      "dm_send": {"label": "Send Direct Messages", "scope_kind": "send_private", "status": "available", "scopes": ["dm.write", "tweet.read", "users.read", "offline.access"]},
      "archive_import": {"label": "Import historical posts and DMs from a Twitter/X archive", "scope_kind": "local_history", "status": "available", "scopes": []}
    }'::jsonb,
    '{
      "flow": "oauth2_authorization_code_pkce",
      "redirect_uri": "http://localhost:1",
      "requires_user_client": true,
      "secret_storage": "~/.hexis/auth",
      "supported_surfaces": ["chat", "cli", "web", "channels"],
      "default_capabilities": ["read", "search", "ingest"],
      "capability_order": ["read", "search", "ingest", "dm_read", "send", "dm_send", "archive_import"],
      "required_scopes": [],
      "scope_order": ["tweet.read", "users.read", "offline.access", "dm.read", "tweet.write", "dm.write"],
      "history_import": {
        "flow": "local_archive_import",
        "capability": "archive_import",
        "accepted_inputs": ["export_path", "import_path"],
        "notes": ["Download your Twitter/X archive and point Hexis at the extracted archive directory or tweet/direct-message JS file."]
      },
      "capability_aliases": {
        "x": "read",
        "twitter": "read",
        "tweets": "read",
        "posts": "read",
        "timeline": "read",
        "mentions": "read",
        "search_posts": "search",
        "history": "archive_import",
        "archive": "archive_import",
        "import": "archive_import",
        "ingest_history": "archive_import",
        "dm": "dm_send",
        "dms": "dm_read",
        "direct_messages": "dm_read",
        "message": "dm_send",
        "write": "send",
        "post": "send",
        "reply": "send",
        "respond": "send"
      },
      "user_next_step": "Create or choose an X Developer app, enable user sign-in, register http://localhost:1 as the callback URI, then start Twitter/X connection setup. Request only the capabilities you want; archive import is still available through a local export path."
    }'::jsonb,
    'https://docs.x.com',
    '{"provider": "twitter_x", "twitter_x_live_oauth": true, "twitter_x_archive_import": true, "seeded_by": "db/77_functions_integrations.sql"}'::jsonb
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

CREATE OR REPLACE FUNCTION list_integration_connectors(
    p_include_disabled BOOLEAN DEFAULT FALSE
) RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    SELECT COALESCE(jsonb_agg(
        jsonb_build_object(
            'id', id,
            'display_name', display_name,
            'category', category,
            'auth_type', auth_type,
            'status', status,
            'capability_manifest', capability_manifest,
            'setup_manifest', setup_manifest,
            'docs_url', docs_url,
            'metadata', metadata,
            'updated_at', updated_at
        )
        ORDER BY category, display_name, id
    ), '[]'::jsonb)
    FROM integration_connectors
    WHERE p_include_disabled OR status <> 'disabled';
$$;

CREATE OR REPLACE FUNCTION prepare_connection_attempt(
    p_connector_id TEXT,
    p_requested_capabilities JSONB DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    row_connector integration_connectors%ROWTYPE;
    raw_requested JSONB := COALESCE(p_requested_capabilities, 'null'::jsonb);
    requested JSONB := '[]'::jsonb;
    capabilities JSONB := '[]'::jsonb;
    requested_scopes JSONB := '[]'::jsonb;
    unsupported TEXT;
    unavailable TEXT;
BEGIN
    SELECT *
    INTO row_connector
    FROM integration_connectors
    WHERE id = p_connector_id;

    IF NOT FOUND OR row_connector.status <> 'available' THEN
        RAISE EXCEPTION 'integration connector % is not available', p_connector_id;
    END IF;

    IF raw_requested = 'null'::jsonb THEN
        requested := COALESCE(row_connector.setup_manifest->'default_capabilities', '[]'::jsonb);
    ELSIF jsonb_typeof(raw_requested) = 'array' THEN
        requested := raw_requested;
    ELSIF jsonb_typeof(raw_requested) = 'string' THEN
        SELECT COALESCE(jsonb_agg(part), '[]'::jsonb)
        INTO requested
        FROM regexp_split_to_table(raw_requested #>> '{}', '[,[:space:]]+') AS part
        WHERE btrim(part) <> '';
    ELSE
        requested := jsonb_build_array(raw_requested #>> '{}');
    END IF;

    IF jsonb_array_length(requested) = 0 THEN
        requested := COALESCE(row_connector.setup_manifest->'default_capabilities', '[]'::jsonb);
    END IF;

    WITH raw AS (
        SELECT value AS original, ord
        FROM jsonb_array_elements_text(requested) WITH ORDINALITY AS item(value, ord)
        WHERE btrim(value) <> ''
    ),
    normalized AS (
        SELECT
            original,
            COALESCE(
                row_connector.setup_manifest #>> ARRAY[
                    'capability_aliases',
                    lower(replace(btrim(original), '-', '_'))
                ],
                lower(replace(btrim(original), '-', '_'))
            ) AS capability,
            min(ord) AS first_seen
        FROM raw
        GROUP BY original, capability
    )
    SELECT string_agg(DISTINCT original, ', ' ORDER BY original)
    INTO unsupported
    FROM normalized
    WHERE NOT row_connector.capability_manifest ? capability;

    IF unsupported IS NOT NULL THEN
        RAISE EXCEPTION 'unsupported % capability: %', p_connector_id, unsupported;
    END IF;

    WITH raw AS (
        SELECT value AS original, ord
        FROM jsonb_array_elements_text(requested) WITH ORDINALITY AS item(value, ord)
        WHERE btrim(value) <> ''
    ),
    normalized AS (
        SELECT
            COALESCE(
                row_connector.setup_manifest #>> ARRAY[
                    'capability_aliases',
                    lower(replace(btrim(original), '-', '_'))
                ],
                lower(replace(btrim(original), '-', '_'))
            ) AS capability,
            min(ord) AS first_seen
        FROM raw
        GROUP BY capability
    )
    SELECT string_agg(
        capability || ' (' || COALESCE(row_connector.capability_manifest->capability->>'status', 'available') || ')',
        ', '
        ORDER BY capability
    )
    INTO unavailable
    FROM normalized
    WHERE COALESCE(row_connector.capability_manifest->capability->>'status', 'available') <> 'available';

    IF unavailable IS NOT NULL THEN
        RAISE EXCEPTION '% capability is not available: %', p_connector_id, unavailable;
    END IF;

    WITH raw AS (
        SELECT value AS original, ord
        FROM jsonb_array_elements_text(requested) WITH ORDINALITY AS item(value, ord)
        WHERE btrim(value) <> ''
    ),
    normalized AS (
        SELECT
            COALESCE(
                row_connector.setup_manifest #>> ARRAY[
                    'capability_aliases',
                    lower(replace(btrim(original), '-', '_'))
                ],
                lower(replace(btrim(original), '-', '_'))
            ) AS capability,
            min(ord) AS first_seen
        FROM raw
        GROUP BY capability
    ),
    ordered AS (
        SELECT
            n.capability,
            COALESCE(o.ord, 100000 + n.first_seen) AS sort_order
        FROM normalized n
        LEFT JOIN LATERAL (
            SELECT ord
            FROM jsonb_array_elements_text(
                COALESCE(row_connector.setup_manifest->'capability_order', '[]'::jsonb)
            ) WITH ORDINALITY AS item(value, ord)
            WHERE value = n.capability
            LIMIT 1
        ) o ON TRUE
    )
    SELECT COALESCE(jsonb_agg(capability ORDER BY sort_order, capability), '[]'::jsonb)
    INTO capabilities
    FROM ordered;

    WITH selected_caps AS (
        SELECT value AS capability
        FROM jsonb_array_elements_text(capabilities) AS item(value)
    ),
    all_scopes AS (
        SELECT value AS scope
        FROM jsonb_array_elements_text(
            COALESCE(row_connector.setup_manifest->'required_scopes', '[]'::jsonb)
        ) AS item(value)
        UNION
        SELECT scope.value AS scope
        FROM selected_caps cap
        CROSS JOIN LATERAL jsonb_array_elements_text(
            COALESCE(row_connector.capability_manifest->cap.capability->'scopes', '[]'::jsonb)
        ) AS scope(value)
    ),
    ordered_scopes AS (
        SELECT
            scope,
            COALESCE(o.ord, 100000) AS sort_order
        FROM all_scopes s
        LEFT JOIN LATERAL (
            SELECT ord
            FROM jsonb_array_elements_text(
                COALESCE(row_connector.setup_manifest->'scope_order', '[]'::jsonb)
            ) WITH ORDINALITY AS item(value, ord)
            WHERE value = s.scope
            LIMIT 1
        ) o ON TRUE
    )
    SELECT COALESCE(jsonb_agg(scope ORDER BY sort_order, scope), '[]'::jsonb)
    INTO requested_scopes
    FROM ordered_scopes;

    RETURN jsonb_build_object(
        'connector_id', row_connector.id,
        'display_name', row_connector.display_name,
        'auth_type', row_connector.auth_type,
        'capabilities', capabilities,
        'requested_capabilities', capabilities,
        'requested_scopes', requested_scopes,
        'scope_count', jsonb_array_length(requested_scopes),
        'setup_manifest', row_connector.setup_manifest,
        'docs_url', row_connector.docs_url
    );
END;
$$;

CREATE OR REPLACE FUNCTION start_connection_attempt(
    p_connector_id TEXT,
    p_requested_capabilities JSONB DEFAULT '[]'::jsonb,
    p_requested_scopes TEXT[] DEFAULT ARRAY[]::TEXT[],
    p_flow_state JSONB DEFAULT '{}'::jsonb,
    p_authorization_url TEXT DEFAULT NULL,
    p_user_next_step TEXT DEFAULT NULL,
    p_source_channel TEXT DEFAULT NULL,
    p_source_session_id TEXT DEFAULT NULL,
    p_expires_at TIMESTAMPTZ DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    row_attempt connection_attempts%ROWTYPE;
    prepared JSONB;
    derived_scopes TEXT[];
BEGIN
    prepared := prepare_connection_attempt(p_connector_id, p_requested_capabilities);
    SELECT COALESCE(array_agg(value ORDER BY ord), ARRAY[]::TEXT[])
    INTO derived_scopes
    FROM jsonb_array_elements_text(prepared->'requested_scopes') WITH ORDINALITY AS item(value, ord);

    INSERT INTO connection_attempts (
        connector_id,
        requested_capabilities,
        requested_scopes,
        flow_state,
        authorization_url,
        user_next_step,
        source_channel,
        source_session_id,
        expires_at
    )
    VALUES (
        p_connector_id,
        prepared->'requested_capabilities',
        derived_scopes,
        COALESCE(p_flow_state, '{}'::jsonb),
        NULLIF(p_authorization_url, ''),
        NULLIF(p_user_next_step, ''),
        NULLIF(p_source_channel, ''),
        NULLIF(p_source_session_id, ''),
        p_expires_at
    )
    RETURNING * INTO row_attempt;

    RETURN jsonb_build_object(
        'attempt_id', row_attempt.id::text,
        'connector_id', row_attempt.connector_id,
        'status', row_attempt.status,
        'requested_capabilities', row_attempt.requested_capabilities,
        'requested_scopes', to_jsonb(row_attempt.requested_scopes),
        'authorization_url', row_attempt.authorization_url,
        'user_next_step', row_attempt.user_next_step,
        'source_channel', row_attempt.source_channel,
        'source_session_id', row_attempt.source_session_id,
        'expires_at', row_attempt.expires_at,
        'created_at', row_attempt.created_at
    );
END;
$$;

CREATE OR REPLACE FUNCTION mark_connection_attempt_exchanging(
    p_attempt_id UUID
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    row_attempt connection_attempts%ROWTYPE;
BEGIN
    UPDATE connection_attempts
    SET status = 'exchanging',
        error = NULL,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_attempt_id
      AND status IN ('pending_user', 'awaiting_input', 'error')
    RETURNING * INTO row_attempt;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'connection attempt % is not active', p_attempt_id;
    END IF;

    RETURN jsonb_build_object(
        'attempt_id', row_attempt.id::text,
        'connector_id', row_attempt.connector_id,
        'status', row_attempt.status
    );
END;
$$;

CREATE OR REPLACE FUNCTION mark_connection_attempt_error(
    p_attempt_id UUID,
    p_error TEXT
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    row_attempt connection_attempts%ROWTYPE;
BEGIN
    UPDATE connection_attempts
    SET status = 'error',
        error = COALESCE(NULLIF(p_error, ''), 'connection setup failed'),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_attempt_id
    RETURNING * INTO row_attempt;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'connection attempt % not found', p_attempt_id;
    END IF;

    RETURN jsonb_build_object(
        'attempt_id', row_attempt.id::text,
        'connector_id', row_attempt.connector_id,
        'status', row_attempt.status,
        'error', row_attempt.error
    );
END;
$$;

CREATE OR REPLACE FUNCTION complete_connection_attempt(
    p_attempt_id UUID,
    p_account_key TEXT,
    p_display_name TEXT DEFAULT NULL,
    p_credential_ref TEXT DEFAULT NULL,
    p_granted_scopes TEXT[] DEFAULT ARRAY[]::TEXT[],
    p_capabilities JSONB DEFAULT '[]'::jsonb,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    row_attempt connection_attempts%ROWTYPE;
    row_connection integration_connections%ROWTYPE;
    normalized_account TEXT := NULLIF(btrim(COALESCE(p_account_key, '')), '');
    capability_input JSONB;
    prepared JSONB;
BEGIN
    IF normalized_account IS NULL THEN
        RAISE EXCEPTION 'account_key is required';
    END IF;

    SELECT * INTO row_attempt
    FROM connection_attempts
    WHERE id = p_attempt_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'connection attempt % not found', p_attempt_id;
    END IF;

    IF p_capabilities IS NULL
       OR p_capabilities = 'null'::jsonb
       OR (jsonb_typeof(p_capabilities) = 'array' AND jsonb_array_length(p_capabilities) = 0) THEN
        capability_input := row_attempt.requested_capabilities;
    ELSE
        capability_input := p_capabilities;
    END IF;
    prepared := prepare_connection_attempt(row_attempt.connector_id, capability_input);

    INSERT INTO integration_connections (
        connector_id,
        account_key,
        display_name,
        status,
        credential_ref,
        granted_scopes,
        capabilities,
        source_channel,
        source_session_id,
        metadata,
        connected_at,
        last_verified_at,
        revoked_at,
        last_error
    )
    VALUES (
        row_attempt.connector_id,
        normalized_account,
        NULLIF(btrim(COALESCE(p_display_name, '')), ''),
        'connected',
        NULLIF(btrim(COALESCE(p_credential_ref, '')), ''),
        COALESCE(p_granted_scopes, ARRAY[]::TEXT[]),
        prepared->'capabilities',
        row_attempt.source_channel,
        row_attempt.source_session_id,
        COALESCE(p_metadata, '{}'::jsonb),
        CURRENT_TIMESTAMP,
        CURRENT_TIMESTAMP,
        NULL,
        NULL
    )
    ON CONFLICT (connector_id, account_key) DO UPDATE SET
        display_name = COALESCE(EXCLUDED.display_name, integration_connections.display_name),
        status = 'connected',
        credential_ref = COALESCE(EXCLUDED.credential_ref, integration_connections.credential_ref),
        granted_scopes = EXCLUDED.granted_scopes,
        capabilities = EXCLUDED.capabilities,
        source_channel = COALESCE(EXCLUDED.source_channel, integration_connections.source_channel),
        source_session_id = COALESCE(EXCLUDED.source_session_id, integration_connections.source_session_id),
        metadata = integration_connections.metadata || EXCLUDED.metadata,
        connected_at = COALESCE(integration_connections.connected_at, CURRENT_TIMESTAMP),
        last_verified_at = CURRENT_TIMESTAMP,
        revoked_at = NULL,
        last_error = NULL,
        updated_at = CURRENT_TIMESTAMP
    RETURNING * INTO row_connection;

    UPDATE connection_attempts
    SET account_key = normalized_account,
        status = 'complete',
        credential_ref = p_credential_ref,
        error = NULL,
        completed_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_attempt_id;

    RETURN jsonb_build_object(
        'attempt_id', row_attempt.id::text,
        'connector_id', row_connection.connector_id,
        'connection_id', row_connection.id::text,
        'account_key', row_connection.account_key,
        'display_name', row_connection.display_name,
        'status', row_connection.status,
        'credential_ref', row_connection.credential_ref,
        'granted_scopes', to_jsonb(row_connection.granted_scopes),
        'capabilities', row_connection.capabilities,
        'connected_at', row_connection.connected_at
    );
END;
$$;

CREATE OR REPLACE FUNCTION revoke_integration_connection(
    p_connector_id TEXT,
    p_account_key TEXT DEFAULT NULL,
    p_reason TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    affected INT;
BEGIN
    UPDATE integration_connections
    SET status = 'revoked',
        last_error = NULLIF(p_reason, ''),
        revoked_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE connector_id = p_connector_id
      AND (NULLIF(btrim(COALESCE(p_account_key, '')), '') IS NULL OR account_key = p_account_key)
      AND status <> 'revoked';

    GET DIAGNOSTICS affected = ROW_COUNT;

    RETURN jsonb_build_object(
        'connector_id', p_connector_id,
        'account_key', NULLIF(btrim(COALESCE(p_account_key, '')), ''),
        'revoked', affected
    );
END;
$$;

CREATE OR REPLACE FUNCTION integration_status(
    p_connector_id TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    connectors JSONB;
    connections JSONB;
    attempts JSONB;
BEGIN
    SELECT COALESCE(jsonb_agg(
        jsonb_build_object(
            'id', id,
            'display_name', display_name,
            'category', category,
            'auth_type', auth_type,
            'status', status,
            'capability_manifest', capability_manifest,
            'setup_manifest', setup_manifest,
            'docs_url', docs_url
        )
        ORDER BY display_name, id
    ), '[]'::jsonb)
    INTO connectors
    FROM integration_connectors
    WHERE p_connector_id IS NULL OR id = p_connector_id;

    SELECT COALESCE(jsonb_agg(
        jsonb_build_object(
            'id', id::text,
            'connector_id', connector_id,
            'account_key', account_key,
            'display_name', display_name,
            'status', status,
            'credential_ref', credential_ref,
            'granted_scopes', to_jsonb(granted_scopes),
            'capabilities', capabilities,
            'source_channel', source_channel,
            'source_session_id', source_session_id,
            'last_error', last_error,
            'connected_at', connected_at,
            'last_verified_at', last_verified_at,
            'revoked_at', revoked_at,
            'updated_at', updated_at
        )
        ORDER BY updated_at DESC, connector_id, account_key
    ), '[]'::jsonb)
    INTO connections
    FROM integration_connections
    WHERE p_connector_id IS NULL OR connector_id = p_connector_id;

    SELECT COALESCE(jsonb_agg(
        jsonb_build_object(
            'attempt_id', id::text,
            'connector_id', connector_id,
            'account_key', account_key,
            'status', status,
            'requested_capabilities', requested_capabilities,
            'requested_scopes', to_jsonb(requested_scopes),
            'authorization_url', authorization_url,
            'user_next_step', user_next_step,
            'source_channel', source_channel,
            'source_session_id', source_session_id,
            'credential_ref', credential_ref,
            'error', error,
            'expires_at', expires_at,
            'completed_at', completed_at,
            'created_at', created_at,
            'updated_at', updated_at
        )
        ORDER BY created_at DESC
    ), '[]'::jsonb)
    INTO attempts
    FROM connection_attempts
    WHERE (p_connector_id IS NULL OR connector_id = p_connector_id)
      AND created_at >= CURRENT_TIMESTAMP - INTERVAL '1 day';

    RETURN jsonb_build_object(
        'connectors', connectors,
        'connections', connections,
        'recent_attempts', attempts
    );
END;
$$;
