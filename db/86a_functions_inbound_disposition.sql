-- DB-owned inbound engagement policy and passive-ingestion bridge.
SET search_path = public, ag_catalog, "$user";

INSERT INTO config_defaults (key, value, description) VALUES
    ('channel.disposition.enabled', 'false'::jsonb,
     'Master switch for SQL-owned inbound engage/observe/wake/drop routing; disabled by default'),
    ('channel.imessage.disposition.trigger_word', '""'::jsonb,
     'Optional leading word that addresses Hexis on iMessage; empty preserves direct allowed conversations'),
    ('channel.signal.disposition.trigger_word', '""'::jsonb,
     'Optional leading word that addresses Hexis on Signal; empty keeps allowed conversations direct'),
    ('channel.slack.disposition.trigger_word', '""'::jsonb,
     'Optional leading word that addresses Hexis on Slack; empty keeps allowed conversations direct'),
    ('channel.discord.disposition.trigger_word', '""'::jsonb,
     'Optional leading word that addresses Hexis on Discord; empty keeps allowed conversations direct'),
    ('channel.telegram.disposition.trigger_word', '""'::jsonb,
     'Optional leading word that addresses Hexis on Telegram; empty keeps allowed conversations direct'),
    ('channel.whatsapp.disposition.trigger_word', '""'::jsonb,
     'Optional leading word that addresses Hexis on WhatsApp; empty keeps allowed conversations direct'),
    ('channel.matrix.disposition.trigger_word', '""'::jsonb,
     'Optional leading word that addresses Hexis on Matrix; empty keeps allowed conversations direct'),
    ('channel.imessage.disposition.continuation_window_seconds', '0'::jsonb,
     'Seconds after an outbound iMessage in the same session that an unaddressed inbound remains engaged; 0 disables'),
    ('channel.signal.disposition.continuation_window_seconds', '0'::jsonb,
     'Seconds after an outbound Signal message that an unaddressed inbound remains engaged; 0 disables'),
    ('channel.slack.disposition.continuation_window_seconds', '0'::jsonb,
     'Seconds after an outbound Slack message that an unaddressed inbound remains engaged; 0 disables'),
    ('channel.discord.disposition.continuation_window_seconds', '0'::jsonb,
     'Seconds after an outbound Discord message that an unaddressed inbound remains engaged; 0 disables'),
    ('channel.telegram.disposition.continuation_window_seconds', '0'::jsonb,
     'Seconds after an outbound Telegram message that an unaddressed inbound remains engaged; 0 disables'),
    ('channel.whatsapp.disposition.continuation_window_seconds', '0'::jsonb,
     'Seconds after an outbound WhatsApp message that an unaddressed inbound remains engaged; 0 disables'),
    ('channel.matrix.disposition.continuation_window_seconds', '0'::jsonb,
     'Seconds after an outbound Matrix message that an unaddressed inbound remains engaged; 0 disables'),
    ('channel.disposition.mention_anywhere_engages', 'true'::jsonb,
     'Treat a configured trigger word or native platform mention anywhere in the message as addressed'),
    ('channel.disposition.wake_on_correction', 'true'::jsonb,
     'Allow a fresh identity-verified operator correction to request a heartbeat wake'),
    ('channel.disposition.wake_max_age_seconds', '600'::jsonb,
     'Maximum age of an unprocessed inbound correction wake; prevents stale sync from waking the agent'),
    ('channel.disposition.classifier_enabled', 'true'::jsonb,
     'Classify otherwise-ambiguous operator messages with the configured Hexis LLM and retain deterministic output on failure'),
    ('channel.disposition.classifier_timeout_seconds', '10'::jsonb,
     'Timeout for the optional ambiguous-message classifier'),
    ('llm.inbound_disposition', 'null'::jsonb,
     'Optional LLM override for inbound ambiguity classification; falls back to llm.subconscious')
ON CONFLICT (key) DO NOTHING;

CREATE OR REPLACE FUNCTION _inbound_disposition_allowlist_contains(
    p_key TEXT,
    p_candidate TEXT
) RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    allowlist JSONB := get_config(p_key);
BEGIN
    IF allowlist IS NULL THEN
        RETURN TRUE;
    END IF;
    IF jsonb_typeof(allowlist) = 'string' AND allowlist #>> '{}' = '*' THEN
        RETURN TRUE;
    END IF;
    IF jsonb_typeof(allowlist) <> 'array' OR p_candidate IS NULL THEN
        RETURN FALSE;
    END IF;
    RETURN EXISTS (
        SELECT 1
        FROM jsonb_array_elements_text(allowlist) AS item(value)
        WHERE lower(item.value) = lower(p_candidate)
    );
END;
$$;

CREATE OR REPLACE FUNCTION inbound_disposition_reply_allowed(
    p_channel_type TEXT,
    p_sender_id TEXT,
    p_channel_id TEXT,
    p_metadata JSONB DEFAULT '{}'::jsonb,
    p_is_operator BOOLEAN DEFAULT FALSE
) RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    channel_name TEXT := lower(COALESCE(NULLIF(btrim(p_channel_type), ''), ''));
    metadata JSONB := CASE
        WHEN jsonb_typeof(p_metadata) = 'object' THEN p_metadata
        ELSE '{}'::jsonb
    END;
    is_group BOOLEAN;
    is_mention BOOLEAN;
    guild_id TEXT;
BEGIN
    IF channel_name = '' OR NULLIF(btrim(COALESCE(p_sender_id, '')), '') IS NULL THEN
        RETURN FALSE;
    END IF;

    is_group := COALESCE((metadata->>'is_group')::boolean, FALSE);
    is_mention := COALESCE((metadata->>'is_mention')::boolean, FALSE);
    guild_id := NULLIF(metadata->>'guild_id', '');

    -- A generic sender allowlist is always a ceiling for non-operators.
    IF NOT p_is_operator
       AND NOT _inbound_disposition_allowlist_contains(
           'channel.' || channel_name || '.allowed_users', p_sender_id
       ) THEN
        RETURN FALSE;
    END IF;

    IF channel_name = 'imessage' THEN
        RETURN p_is_operator OR _inbound_disposition_allowlist_contains(
            'channel.imessage.allowed_handles', p_sender_id
        );
    ELSIF channel_name = 'signal' THEN
        RETURN p_is_operator OR _inbound_disposition_allowlist_contains(
            'channel.signal.allowed_numbers', p_sender_id
        );
    ELSIF channel_name = 'whatsapp' THEN
        RETURN p_is_operator OR _inbound_disposition_allowlist_contains(
            'channel.whatsapp.allowed_numbers', p_sender_id
        );
    ELSIF channel_name = 'slack' THEN
        RETURN _inbound_disposition_allowlist_contains(
                   'channel.slack.allowed_channels', p_channel_id
               )
               OR is_mention
               OR (p_is_operator AND NOT is_group);
    ELSIF channel_name = 'discord' THEN
        IF NOT is_group THEN
            RETURN TRUE;
        END IF;
        IF NOT _inbound_disposition_allowlist_contains(
            'channel.discord.allowed_guilds', guild_id
        ) THEN
            RETURN FALSE;
        END IF;
        RETURN _inbound_disposition_allowlist_contains(
                   'channel.discord.allowed_channels', p_channel_id
               ) OR is_mention;
    ELSIF channel_name = 'telegram' THEN
        RETURN NOT is_group
               OR _inbound_disposition_allowlist_contains(
                   'channel.telegram.allowed_chat_ids', p_channel_id
               )
               OR is_mention;
    ELSIF channel_name = 'matrix' THEN
        RETURN _inbound_disposition_allowlist_contains(
            'channel.matrix.allowed_rooms', p_channel_id
        );
    END IF;

    RETURN TRUE;
EXCEPTION WHEN invalid_text_representation THEN
    RETURN FALSE;
END;
$$;

DROP FUNCTION IF EXISTS resolve_inbound_disposition(TEXT, TEXT, UUID, TEXT, JSONB, BOOLEAN);

CREATE OR REPLACE FUNCTION resolve_inbound_disposition(
    p_channel_type TEXT,
    p_sender_id TEXT,
    p_session_key TEXT DEFAULT NULL,
    p_text TEXT DEFAULT '',
    p_metadata JSONB DEFAULT '{}'::jsonb,
    p_dry_run BOOLEAN DEFAULT FALSE
) RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
AS $$
DECLARE
    channel_name TEXT := lower(COALESCE(NULLIF(btrim(p_channel_type), ''), 'unknown'));
    metadata JSONB := CASE
        WHEN jsonb_typeof(p_metadata) = 'object' THEN p_metadata
        ELSE '{}'::jsonb
    END;
    v_session_id UUID;
    candidate_session_id UUID;
    normalized_text TEXT;
    lower_text TEXT;
    disposition TEXT := 'observe';
    reason TEXT := 'default_observe';
    ambiguous BOOLEAN := FALSE;
    is_operator BOOLEAN := FALSE;
    reply_allowed BOOLEAN := FALSE;
    stripped_text TEXT;
    audit_id BIGINT;
    decided BOOLEAN := FALSE;
    trigger_word TEXT;
    trigger_token TEXT;
    continuation_window INT;
    correction_window INT;
    mention_anywhere BOOLEAN;
    wake_on_correction BOOLEAN;
    classifier_enabled BOOLEAN;
    is_native_mention BOOLEAN;
    has_attachments BOOLEAN;
BEGIN
    normalized_text := regexp_replace(COALESCE(p_text, ''), '^\s+|\s+$', '', 'g');
    lower_text := lower(normalized_text);
    is_native_mention := COALESCE((metadata->>'is_mention')::boolean, FALSE);
    has_attachments := COALESCE((metadata->>'has_attachments')::boolean, FALSE);

    candidate_session_id := _db_brain_try_uuid(p_session_key);
    IF candidate_session_id IS NOT NULL THEN
        SELECT id INTO v_session_id
        FROM channel_sessions
        WHERE id = candidate_session_id;
    END IF;
    IF v_session_id IS NULL AND p_session_key IS NOT NULL THEN
        SELECT id INTO v_session_id
        FROM channel_sessions
        WHERE channel_type = channel_name
          AND channel_id = p_session_key
          AND sender_id = p_sender_id
        ORDER BY last_active DESC NULLS LAST, created_at DESC
        LIMIT 1;
    END IF;

    is_operator := channel_sender_is_operator(channel_name, p_sender_id);
    reply_allowed := inbound_disposition_reply_allowed(
        channel_name,
        p_sender_id,
        COALESCE(metadata->>'channel_id', p_session_key),
        metadata,
        is_operator
    );

    IF normalized_text = '' AND NOT has_attachments THEN
        disposition := 'drop';
        reason := 'empty';
        decided := TRUE;
    ELSIF NOT reply_allowed THEN
        disposition := 'observe';
        reason := 'allowlist_ceiling';
        decided := TRUE;
    ELSIF normalized_text = '' AND has_attachments THEN
        disposition := 'engage';
        reason := 'allowed_attachment';
        decided := TRUE;
    END IF;

    IF NOT decided THEN
        trigger_word := COALESCE(
            get_config_text('channel.' || channel_name || '.disposition.trigger_word'),
            ''
        );
        continuation_window := GREATEST(COALESCE(
            get_config_int(
                'channel.' || channel_name || '.disposition.continuation_window_seconds'
            ),
            0
        ), 0);
        mention_anywhere := COALESCE(
            get_config_bool('channel.disposition.mention_anywhere_engages'), TRUE
        );
        wake_on_correction := COALESCE(
            get_config_bool('channel.disposition.wake_on_correction'), TRUE
        );
        classifier_enabled := COALESCE(
            get_config_bool('channel.disposition.classifier_enabled'), TRUE
        );
        trigger_token := lower(regexp_replace(trigger_word, '^@', ''));

        IF trigger_word <> ''
           AND left(lower_text, char_length(trigger_word)) = lower(trigger_word)
           AND (
               char_length(lower_text) = char_length(trigger_word)
               OR substr(lower_text, char_length(trigger_word) + 1, 1) !~ '[a-z0-9_]'
           ) THEN
            disposition := 'engage';
            reason := 'trigger_match';
            stripped_text := regexp_replace(
                ltrim(
                    regexp_replace(
                        substr(normalized_text, char_length(trigger_word) + 1),
                        '^\s+|\s+$', '', 'g'
                    ),
                    ':,'
                ),
                '^\s+|\s+$', '', 'g'
            );
            IF stripped_text = '' THEN
                stripped_text := NULL;
            END IF;
            decided := TRUE;
        END IF;

        IF NOT decided
           AND mention_anywhere
           AND trigger_word <> ''
           AND (
               is_native_mention
               OR trigger_token = ANY(
                   regexp_split_to_array(
                       regexp_replace(lower_text, '[^a-z0-9_]+', ' ', 'g'),
                       '\s+'
                   )
               )
           ) THEN
            disposition := 'engage';
            reason := 'mention_match';
            stripped_text := normalized_text;
            decided := TRUE;
        END IF;

        IF NOT decided
           AND continuation_window > 0
           AND v_session_id IS NOT NULL
           AND EXISTS (
               SELECT 1
               FROM channel_messages
               WHERE channel_messages.session_id = v_session_id
                 AND direction = 'outbound'
                 AND created_at > CURRENT_TIMESTAMP
                     - make_interval(secs => continuation_window)
           ) THEN
            disposition := 'engage';
            reason := 'continuation_window';
            stripped_text := normalized_text;
            decided := TRUE;
        END IF;

        IF NOT decided
           AND is_operator
           AND wake_on_correction
           AND lower_text ~ '^(no\y|no no|nope\y|wrong\y|not that\y|that''s not|stop\y|incorrect\y)' THEN
            correction_window := greatest(continuation_window * 4, 3600);
            IF v_session_id IS NOT NULL AND EXISTS (
                SELECT 1
                FROM channel_messages
                WHERE channel_messages.session_id = v_session_id
                  AND direction = 'outbound'
                  AND created_at > CURRENT_TIMESTAMP
                      - make_interval(secs => correction_window)
            ) THEN
                disposition := 'wake';
                reason := 'correction_shape';
                stripped_text := normalized_text;
                decided := TRUE;
            END IF;
        END IF;

        -- Channels without trigger gating retain their established direct
        -- conversation behavior once the live allowlist permits a reply.
        IF NOT decided AND trigger_word = '' THEN
            disposition := 'engage';
            reason := 'allowed_conversation';
            stripped_text := normalized_text;
            decided := TRUE;
        END IF;

        IF NOT decided AND is_operator AND classifier_enabled THEN
            disposition := 'observe';
            reason := 'ambiguous_operator';
            ambiguous := TRUE;
            decided := TRUE;
        END IF;
    END IF;

    IF NOT p_dry_run THEN
        INSERT INTO inbound_disposition_events (
            channel_type,
            channel_id,
            sender_id,
            session_id,
            platform_message_id,
            disposition,
            reason,
            ambiguous,
            is_operator,
            reply_allowed,
            preview,
            metadata
        ) VALUES (
            channel_name,
            COALESCE(metadata->>'channel_id', p_session_key),
            p_sender_id,
            v_session_id,
            NULLIF(metadata->>'platform_message_id', ''),
            disposition,
            reason,
            ambiguous,
            is_operator,
            reply_allowed,
            left(normalized_text, 200),
            metadata
        ) RETURNING id INTO audit_id;
    END IF;

    RETURN jsonb_build_object(
        'disposition', disposition,
        'reason', reason,
        'ambiguous', ambiguous,
        'is_operator', is_operator,
        'reply_allowed', reply_allowed,
        'trigger_stripped_text', stripped_text,
        'session_id', v_session_id,
        'audit_id', audit_id
    );
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'resolve_inbound_disposition failed: % (SQLSTATE %)', SQLERRM, SQLSTATE;
    RETURN jsonb_build_object(
        'disposition', 'observe',
        'reason', 'error_fallback',
        'ambiguous', FALSE,
        'is_operator', FALSE,
        'reply_allowed', FALSE,
        'trigger_stripped_text', NULL,
        'session_id', NULL,
        'audit_id', NULL,
        'error_sqlstate', SQLSTATE,
        'error_detail', SQLERRM
    );
END;
$$;

CREATE OR REPLACE FUNCTION finalize_inbound_disposition(
    p_audit_id BIGINT,
    p_disposition TEXT,
    p_classifier_label TEXT
) RETURNS BOOLEAN
LANGUAGE plpgsql
VOLATILE
AS $$
DECLARE
    event_row inbound_disposition_events%ROWTYPE;
BEGIN
    IF p_disposition IS NULL
       OR p_disposition NOT IN ('engage', 'observe', 'wake', 'drop') THEN
        RAISE EXCEPTION 'invalid inbound disposition: %', p_disposition;
    END IF;

    SELECT * INTO event_row
    FROM inbound_disposition_events
    WHERE id = p_audit_id
    FOR UPDATE;
    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;
    IF p_disposition IN ('engage', 'wake') AND NOT event_row.reply_allowed THEN
        RAISE EXCEPTION 'classifier cannot exceed the reply allowlist ceiling';
    END IF;
    IF p_disposition = 'wake' AND NOT event_row.is_operator THEN
        RAISE EXCEPTION 'only the identity-verified operator can wake a heartbeat';
    END IF;

    UPDATE inbound_disposition_events
    SET disposition = p_disposition,
        classifier_used = TRUE,
        classifier_label = left(NULLIF(btrim(p_classifier_label), ''), 120)
    WHERE id = p_audit_id;
    RETURN TRUE;
END;
$$;

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

CREATE OR REPLACE FUNCTION has_pending_inbound_disposition_wake()
RETURNS BOOLEAN
LANGUAGE sql
STABLE
AS $$
    SELECT COALESCE(get_config_bool('channel.disposition.enabled'), FALSE)
       AND EXISTS (
           SELECT 1
           FROM inbound_disposition_events
           WHERE disposition = 'wake'
             AND is_operator
             AND wake_processed_at IS NULL
             AND ts > CURRENT_TIMESTAMP - make_interval(
                 secs => LEAST(GREATEST(COALESCE(
                     get_config_int('channel.disposition.wake_max_age_seconds'),
                     600
                 ), 30), 86400)
             )
       )
$$;

-- The ordinary heartbeat worker calls this function once per poll. A fresh
-- operator correction may start a heartbeat even when the cadence timer is
-- not due, but never overrides pause, termination, initialization, or an
-- already-active heartbeat. The advisory lock also closes the scheduled-vs-
-- correction double-start race.
CREATE OR REPLACE FUNCTION run_heartbeat()
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
AS $$
DECLARE
    heartbeat_payload JSONB;
    wake_event inbound_disposition_events%ROWTYPE;
    wake_context JSONB;
    state_record RECORD;
    wake_max_age INT := LEAST(GREATEST(COALESCE(
        get_config_int('channel.disposition.wake_max_age_seconds'), 600
    ), 30), 86400);
BEGIN
    PERFORM pg_advisory_xact_lock(hashtextextended('hexis:heartbeat-start', 0));

    IF COALESCE(get_config_bool('channel.disposition.enabled'), FALSE) THEN
        UPDATE inbound_disposition_events
        SET wake_processed_at = CURRENT_TIMESTAMP,
            wake_outcome = 'stale'
        WHERE disposition = 'wake'
          AND wake_processed_at IS NULL
          AND ts <= CURRENT_TIMESTAMP - make_interval(secs => wake_max_age);

        SELECT * INTO wake_event
        FROM inbound_disposition_events
        WHERE disposition = 'wake'
          AND is_operator
          AND wake_processed_at IS NULL
          AND ts > CURRENT_TIMESTAMP - make_interval(secs => wake_max_age)
        ORDER BY ts, id
        FOR UPDATE SKIP LOCKED
        LIMIT 1;
    END IF;

    IF wake_event.id IS NOT NULL
       AND NOT is_agent_terminated()
       AND is_agent_configured()
       AND is_init_complete() THEN
        SELECT * INTO state_record FROM heartbeat_state WHERE id = 1;
        IF NOT state_record.is_paused
           AND state_record.active_heartbeat_id IS NULL THEN
            heartbeat_payload := start_heartbeat();
            IF heartbeat_payload IS NOT NULL THEN
                wake_context := jsonb_build_object(
                    'event_id', wake_event.id,
                    'channel_type', wake_event.channel_type,
                    'channel_id', wake_event.channel_id,
                    'sender_id', wake_event.sender_id,
                    'reason', wake_event.reason,
                    'preview', wake_event.preview,
                    'received_at', wake_event.ts
                );
                heartbeat_payload := heartbeat_payload || jsonb_build_object(
                    'inbound_disposition', wake_context
                );
                heartbeat_payload := jsonb_set(
                    heartbeat_payload,
                    '{external_calls,0,input,context,inbound_disposition}',
                    wake_context,
                    TRUE
                );
                UPDATE inbound_disposition_events
                SET wake_processed_at = CURRENT_TIMESTAMP,
                    wake_heartbeat_id = (heartbeat_payload->>'heartbeat_id')::uuid,
                    wake_outcome = 'started'
                WHERE id = wake_event.id;
                RETURN heartbeat_payload;
            END IF;
        END IF;
    END IF;

    IF NOT should_run_heartbeat() THEN
        RETURN NULL;
    END IF;
    RETURN start_heartbeat();
END;
$$;
