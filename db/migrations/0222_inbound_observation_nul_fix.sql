-- Correct passive observation storage: PostgreSQL cannot construct chr(0).
SET search_path = public, ag_catalog, "$user";

CREATE OR REPLACE FUNCTION record_inbound_disposition_observation(
    p_audit_id BIGINT,
    p_channel_type TEXT,
    p_channel_id TEXT,
    p_sender_id TEXT,
    p_sender_name TEXT,
    p_content TEXT,
    p_platform_message_id TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
AS $$
DECLARE
    v_session_id UUID;
    message_id UUID;
    existing_message_id UUID;
    safe_content TEXT := left(COALESCE(p_content, ''), 20000);
    metadata JSONB := CASE
        WHEN jsonb_typeof(p_metadata) = 'object' THEN p_metadata
        ELSE '{}'::jsonb
    END;
BEGIN
    INSERT INTO channel_sessions (
        channel_type, channel_id, sender_id, sender_name, last_active
    ) VALUES (
        lower(p_channel_type), p_channel_id, p_sender_id, p_sender_name,
        CURRENT_TIMESTAMP
    )
    ON CONFLICT (channel_type, channel_id, sender_id) DO UPDATE
    SET sender_name = COALESCE(EXCLUDED.sender_name, channel_sessions.sender_name),
        last_active = CURRENT_TIMESTAMP
    RETURNING id INTO v_session_id;

    IF NULLIF(p_platform_message_id, '') IS NOT NULL THEN
        SELECT id INTO existing_message_id
        FROM channel_messages
        WHERE channel_messages.session_id = v_session_id
          AND direction = 'inbound'
          AND platform_message_id = p_platform_message_id
        ORDER BY created_at DESC
        LIMIT 1;
    END IF;

    IF existing_message_id IS NULL THEN
        INSERT INTO channel_messages (
            session_id, direction, content, platform_message_id, metadata
        ) VALUES (
            v_session_id,
            'inbound',
            safe_content,
            NULLIF(p_platform_message_id, ''),
            metadata || jsonb_build_object(
                'passive', TRUE,
                'inbound_disposition_event_id', p_audit_id
            )
        ) RETURNING id INTO message_id;
    ELSE
        message_id := existing_message_id;
    END IF;

    UPDATE inbound_disposition_events
    SET session_id = v_session_id
    WHERE id = p_audit_id;

    RETURN jsonb_build_object(
        'session_id', v_session_id,
        'message_id', message_id,
        'duplicate', existing_message_id IS NOT NULL
    );
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'record_inbound_disposition_observation failed: % (SQLSTATE %)', SQLERRM, SQLSTATE;
    RETURN jsonb_build_object(
        'session_id', NULL,
        'message_id', NULL,
        'duplicate', FALSE,
        'error', SQLERRM
    );
END;
$$;

