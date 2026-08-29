-- Phase 4: installable PWA delivery and presence configuration.
SET search_path = public, ag_catalog, "$user";

CREATE TABLE IF NOT EXISTS web_push_subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    endpoint TEXT NOT NULL UNIQUE,
    p256dh TEXT NOT NULL,
    auth TEXT NOT NULL,
    expiration_time BIGINT,
    user_agent TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    failure_count INTEGER NOT NULL DEFAULT 0 CHECK (failure_count >= 0),
    last_error TEXT,
    last_delivered_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_web_push_subscriptions_active
    ON web_push_subscriptions (updated_at DESC)
    WHERE revoked_at IS NULL;

INSERT INTO config_defaults (key, value, description) VALUES
    ('pwa.push.enabled', 'true'::jsonb,
     'Allow explicitly subscribed PWA clients to receive web-inbox notifications.'),
    ('pwa.push.vapid_subject', '"https://github.com/QuixiAI/Hexis"'::jsonb,
     'VAPID contact URI used when delivering Web Push notifications.'),
    ('pwa.push.show_message_previews', 'false'::jsonb,
     'Include message text in lock-screen push notifications; off by default for privacy.'),
    ('pwa.presence.enabled', 'true'::jsonb,
     'Record short-lived foreground PWA presence in the channel presence ledger.')
ON CONFLICT (key) DO UPDATE SET
    value = EXCLUDED.value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

CREATE OR REPLACE FUNCTION upsert_web_push_subscription(
    p_endpoint TEXT,
    p_p256dh TEXT,
    p_auth TEXT,
    p_expiration_time BIGINT DEFAULT NULL,
    p_user_agent TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_endpoint TEXT := btrim(COALESCE(p_endpoint, ''));
    v_row web_push_subscriptions%ROWTYPE;
BEGIN
    IF v_endpoint = '' OR length(v_endpoint) > 4096
       OR v_endpoint !~ '^https://' THEN
        RAISE EXCEPTION 'web push endpoint must be an HTTPS URL no longer than 4096 characters';
    END IF;
    IF NULLIF(btrim(COALESCE(p_p256dh, '')), '') IS NULL
       OR length(p_p256dh) > 1024
       OR NULLIF(btrim(COALESCE(p_auth, '')), '') IS NULL
       OR length(p_auth) > 1024 THEN
        RAISE EXCEPTION 'web push subscription keys are required and must be at most 1024 characters';
    END IF;

    INSERT INTO web_push_subscriptions (
        endpoint, p256dh, auth, expiration_time, user_agent, metadata
    ) VALUES (
        v_endpoint, btrim(p_p256dh), btrim(p_auth), p_expiration_time,
        NULLIF(left(btrim(COALESCE(p_user_agent, '')), 1000), ''),
        COALESCE(p_metadata, '{}'::jsonb)
    )
    ON CONFLICT (endpoint) DO UPDATE SET
        p256dh = EXCLUDED.p256dh,
        auth = EXCLUDED.auth,
        expiration_time = EXCLUDED.expiration_time,
        user_agent = EXCLUDED.user_agent,
        metadata = web_push_subscriptions.metadata || EXCLUDED.metadata,
        failure_count = 0,
        last_error = NULL,
        revoked_at = NULL,
        updated_at = CURRENT_TIMESTAMP
    RETURNING * INTO v_row;

    RETURN jsonb_build_object(
        'id', v_row.id,
        'active', v_row.revoked_at IS NULL,
        'created_at', v_row.created_at,
        'updated_at', v_row.updated_at
    );
END;
$$;

CREATE OR REPLACE FUNCTION revoke_web_push_subscription(p_endpoint TEXT)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    v_updated INTEGER;
BEGIN
    UPDATE web_push_subscriptions
    SET revoked_at = COALESCE(revoked_at, CURRENT_TIMESTAMP),
        updated_at = CURRENT_TIMESTAMP
    WHERE endpoint = btrim(COALESCE(p_endpoint, ''))
      AND revoked_at IS NULL;
    GET DIAGNOSTICS v_updated = ROW_COUNT;
    RETURN v_updated > 0;
END;
$$;

COMMENT ON TABLE web_push_subscriptions IS
    'Explicit browser push grants. Endpoint/key material is transport state; message content is not stored here.';
