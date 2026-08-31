-- Installable web client: explicit Web Push subscriptions and presence gates.
SET search_path = public, ag_catalog, "$user";

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

CREATE OR REPLACE FUNCTION record_pwa_presence(
    p_device_id TEXT,
    p_presence_kind TEXT,
    p_display_mode TEXT DEFAULT NULL,
    p_visibility TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_device_id TEXT := NULLIF(left(btrim(COALESCE(p_device_id, '')), 200), '');
    v_event JSONB;
BEGIN
    IF v_device_id IS NULL THEN
        RAISE EXCEPTION 'PWA device_id is required';
    END IF;

    -- Presence is ephemeral state, not an activity log. Serialize updates for
    -- this device and retain only its latest row so an open dashboard does not
    -- grow channel_presence_events every 30 seconds forever.
    PERFORM pg_advisory_xact_lock(hashtextextended('pwa-presence:' || v_device_id, 0));
    DELETE FROM channel_presence_events
    WHERE channel_type = 'web'
      AND channel_id = v_device_id;

    SELECT record_channel_presence(
        'web',
        v_device_id,
        p_presence_kind,
        'inbound',
        v_device_id,
        NULL,
        jsonb_strip_nulls(jsonb_build_object(
            'display_mode', NULLIF(btrim(COALESCE(p_display_mode, '')), ''),
            'visibility', NULLIF(btrim(COALESCE(p_visibility, '')), '')
        )),
        60
    )
    INTO v_event;

    RETURN v_event;
END;
$$;

COMMENT ON FUNCTION record_pwa_presence(TEXT, TEXT, TEXT, TEXT) IS
    'Coalesce one PWA device to its latest short-lived channel presence record.';
