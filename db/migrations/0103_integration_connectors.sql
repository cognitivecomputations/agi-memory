-- 0103: first-class integration connector setup substrate.
SET search_path = public, ag_catalog, "$user";

CREATE TABLE IF NOT EXISTS integration_connectors (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    category TEXT NOT NULL,
    auth_type TEXT NOT NULL
        CHECK (auth_type IN ('oauth2', 'api_key', 'device_code', 'pairing', 'manual')),
    status TEXT NOT NULL DEFAULT 'available'
        CHECK (status IN ('available', 'planned', 'disabled')),
    capability_manifest JSONB NOT NULL DEFAULT '{}'::jsonb,
    setup_manifest JSONB NOT NULL DEFAULT '{}'::jsonb,
    docs_url TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_integration_connectors_status
    ON integration_connectors (status, category, id);

CREATE TABLE IF NOT EXISTS integration_connections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    connector_id TEXT NOT NULL REFERENCES integration_connectors(id) ON DELETE CASCADE,
    account_key TEXT NOT NULL,
    display_name TEXT,
    status TEXT NOT NULL DEFAULT 'connected'
        CHECK (status IN ('pending', 'connected', 'needs_reauth', 'revoked', 'error')),
    credential_ref TEXT,
    granted_scopes TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    capabilities JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_channel TEXT,
    source_session_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    last_error TEXT,
    connected_at TIMESTAMPTZ,
    last_verified_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (connector_id, account_key)
);

CREATE INDEX IF NOT EXISTS idx_integration_connections_status
    ON integration_connections (connector_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_integration_connections_metadata
    ON integration_connections USING GIN (metadata);

CREATE TABLE IF NOT EXISTS connection_attempts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    connector_id TEXT NOT NULL REFERENCES integration_connectors(id) ON DELETE CASCADE,
    account_key TEXT,
    status TEXT NOT NULL DEFAULT 'pending_user'
        CHECK (status IN ('pending_user', 'awaiting_input', 'exchanging', 'complete', 'error', 'expired', 'cancelled')),
    requested_capabilities JSONB NOT NULL DEFAULT '[]'::jsonb,
    requested_scopes TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    flow_state JSONB NOT NULL DEFAULT '{}'::jsonb,
    authorization_url TEXT,
    user_next_step TEXT,
    source_channel TEXT,
    source_session_id TEXT,
    credential_ref TEXT,
    error TEXT,
    expires_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_connection_attempts_status
    ON connection_attempts (connector_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_connection_attempts_session
    ON connection_attempts (source_channel, source_session_id, created_at DESC);

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
      "read": {"label": "Read messages", "scope_kind": "read"},
      "search": {"label": "Search messages", "scope_kind": "read"},
      "ingest": {"label": "Ingest message history", "scope_kind": "read"},
      "label": {"label": "Apply/remove labels", "scope_kind": "modify"},
      "spam_triage": {"label": "Triage spam and inbox labels", "scope_kind": "modify"},
      "send": {"label": "Send new messages", "scope_kind": "send"},
      "reply": {"label": "Reply in existing threads", "scope_kind": "send"},
      "delete": {"label": "Delete messages", "scope_kind": "planned"}
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
    '{"provider": "google", "seeded_by": "db/migrations/0103_integration_connectors.sql"}'::jsonb
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
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM integration_connectors
        WHERE id = p_connector_id
          AND status = 'available'
    ) THEN
        RAISE EXCEPTION 'integration connector % is not available', p_connector_id;
    END IF;

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
        CASE
            WHEN jsonb_typeof(COALESCE(p_requested_capabilities, '[]'::jsonb)) = 'array'
            THEN COALESCE(p_requested_capabilities, '[]'::jsonb)
            ELSE jsonb_build_array(p_requested_capabilities)
        END,
        COALESCE(p_requested_scopes, ARRAY[]::TEXT[]),
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
        CASE
            WHEN jsonb_typeof(COALESCE(p_capabilities, '[]'::jsonb)) = 'array'
            THEN COALESCE(p_capabilities, '[]'::jsonb)
            ELSE jsonb_build_array(p_capabilities)
        END,
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
