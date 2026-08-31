-- Agent administration pushdown (plans/db_pushdown.md Tranche 2).
-- Multi-statement Python sagas become single atomic functions: applying the
-- agent configuration, reading composite status, energy debits, contact
-- marking, mood labeling, contact upserts (the codebase's worst N+1), and
-- the tool-execution audit insert.
SET search_path = public, ag_catalog, "$user";

-- 2.1: the whole agent configuration applies atomically. Python's version
-- was ~18 sequential set_config calls inside a client transaction.
CREATE OR REPLACE FUNCTION apply_agent_config(
    p_config JSONB
) RETURNS JSONB AS $$
DECLARE
    c JSONB := COALESCE(p_config, '{}'::jsonb);
    hb_interval FLOAT;
    max_energy FLOAT;
    base_regeneration FLOAT;
    max_active_goals FLOAT;
    maintenance_interval FLOAT;
BEGIN
    hb_interval := COALESCE(
        NULLIF(c->>'heartbeat_interval_minutes', '')::float,
        get_config_float('heartbeat.heartbeat_interval_minutes')
    );
    max_energy := COALESCE(
        NULLIF(c->>'max_energy', '')::float,
        get_config_float('heartbeat.max_energy')
    );
    base_regeneration := COALESCE(
        NULLIF(c->>'base_regeneration', '')::float,
        get_config_float('heartbeat.base_regeneration')
    );
    max_active_goals := COALESCE(
        NULLIF(c->>'max_active_goals', '')::float,
        get_config_float('heartbeat.max_active_goals')
    );
    maintenance_interval := COALESCE(
        NULLIF(c->>'maintenance_interval_seconds', '')::float,
        get_config_float('maintenance.maintenance_interval_seconds')
    );

    PERFORM set_config('heartbeat.heartbeat_interval_minutes',
        to_jsonb(hb_interval));
    PERFORM set_config('heartbeat.max_energy',
        to_jsonb(max_energy));
    PERFORM set_config('heartbeat.base_regeneration',
        to_jsonb(base_regeneration));
    PERFORM set_config('heartbeat.max_active_goals',
        to_jsonb(max_active_goals));
    PERFORM set_config('maintenance.maintenance_interval_seconds',
        to_jsonb(maintenance_interval));
    IF c ? 'subconscious_interval_seconds' AND c->>'subconscious_interval_seconds' IS NOT NULL THEN
        PERFORM set_config('maintenance.subconscious_interval_seconds',
            to_jsonb((c->>'subconscious_interval_seconds')::float));
    END IF;
    IF c ? 'enable_subconscious' AND c->>'enable_subconscious' IS NOT NULL THEN
        PERFORM set_config('maintenance.subconscious_enabled',
            to_jsonb((c->>'enable_subconscious')::boolean));
    END IF;

    PERFORM set_config('agent.objectives', COALESCE(c->'objectives', '[]'::jsonb));
    PERFORM set_config('agent.budget', jsonb_build_object(
        'max_energy', max_energy,
        'base_regeneration', base_regeneration,
        'heartbeat_interval_minutes', hb_interval::int,
        'max_active_goals', max_active_goals::int
    ));
    PERFORM set_config('agent.guardrails', COALESCE(c->'guardrails', '[]'::jsonb));
    PERFORM set_config('agent.initial_message', COALESCE(c->'initial_message', '""'::jsonb));
    PERFORM set_config('agent.tools', COALESCE(
        (SELECT jsonb_agg(jsonb_build_object('name', t.value, 'enabled', true))
         FROM jsonb_array_elements_text(c->'tools') t),
        '[]'::jsonb));

    PERFORM set_config('llm.heartbeat', COALESCE(c->'llm_heartbeat', 'null'::jsonb));
    PERFORM set_config('llm.chat', COALESCE(c->'llm_chat', 'null'::jsonb));
    PERFORM set_config('llm.subconscious',
        COALESCE(NULLIF(c->'llm_subconscious', 'null'::jsonb), c->'llm_heartbeat', 'null'::jsonb));
    PERFORM set_config('user.contact', jsonb_build_object(
        'channels', COALESCE(c->'contact_channels', '[]'::jsonb),
        'destinations', COALESCE(c->'contact_destinations', '{}'::jsonb)
    ));

    IF COALESCE((c->>'mark_configured')::boolean, false) THEN
        PERFORM set_config('agent.is_configured', 'true'::jsonb);
    END IF;

    UPDATE heartbeat_state
    SET is_paused = NOT COALESCE((c->>'enable_autonomy')::boolean, false)
    WHERE id = 1;
    BEGIN
        UPDATE maintenance_state
        SET is_paused = NOT COALESCE((c->>'enable_maintenance')::boolean, true)
        WHERE id = 1;
    EXCEPTION WHEN OTHERS THEN
        NULL;  -- maintenance_state may not exist on older schemas
    END;

    RETURN jsonb_build_object('applied', true);
END;
$$ LANGUAGE plpgsql;

-- 2.2: composite status in one round-trip; the DB owns the AND-policy for
-- "configured" (flag + consent contract + consent decision).
CREATE OR REPLACE FUNCTION get_agent_status()
RETURNS JSONB AS $$
DECLARE
    consent TEXT := get_agent_consent_status();
    consent_log_id JSONB := get_config('agent.consent_log_id');
BEGIN
    RETURN jsonb_build_object(
        'configured', COALESCE(is_agent_configured(), false)
                      AND consent_log_id IS NOT NULL
                      AND consent = 'consent',
        'terminated', COALESCE(is_agent_terminated(), false),
        'consent_status', consent,
        'consent_log_id', consent_log_id
    );
END;
$$ LANGUAGE plpgsql STABLE;

-- 2.5: conditional energy debit — TRUE when the debit happened.
CREATE OR REPLACE FUNCTION spend_energy(
    p_cost FLOAT
) RETURNS BOOLEAN AS $$
DECLARE
    spent BOOLEAN := FALSE;
BEGIN
    UPDATE heartbeat_state
    SET current_energy = current_energy - GREATEST(COALESCE(p_cost, 0.0), 0.0)
    WHERE id = 1
      AND current_energy >= GREATEST(COALESCE(p_cost, 0.0), 0.0)
    RETURNING TRUE INTO spent;
    RETURN COALESCE(spent, FALSE);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION mark_user_contact()
RETURNS VOID AS $$
BEGIN
    UPDATE heartbeat_state
    SET last_user_contact = CURRENT_TIMESTAMP
    WHERE id = 1;
END;
$$ LANGUAGE plpgsql;

-- 2.7: the valence/arousal -> mood word ladder, co-located with the
-- affective substrate instead of a CLI-side threshold pyramid.
CREATE OR REPLACE FUNCTION mood_label(
    p_valence FLOAT,
    p_arousal FLOAT
) RETURNS TEXT AS $$
DECLARE
    v FLOAT := COALESCE(p_valence, 0.0);
    a FLOAT := COALESCE(p_arousal, 0.0);
BEGIN
    RETURN CASE
        WHEN v > 0.5 THEN CASE WHEN a > 0.5 THEN 'enthusiastic' ELSE 'content' END
        WHEN v > 0.2 THEN CASE WHEN a > 0.3 THEN 'curious' ELSE 'calm' END
        WHEN v > -0.2 THEN CASE WHEN a > 0.3 THEN 'focused' ELSE 'neutral' END
        WHEN v > -0.5 THEN CASE WHEN a > 0.3 THEN 'concerned' ELSE 'subdued' END
        ELSE CASE WHEN a > 0.5 THEN 'distressed' ELSE 'withdrawn' END
    END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- 2.3: the contact upsert, set-based. Python looped fetch-then-branch per
-- attendee in five different files.
CREATE OR REPLACE FUNCTION upsert_contact(
    p_name TEXT,
    p_email TEXT,
    p_source TEXT DEFAULT 'manual'
) RETURNS JSONB AS $$
DECLARE
    existing BIGINT;
    new_id BIGINT;
BEGIN
    IF NULLIF(trim(COALESCE(p_email, '')), '') IS NULL OR position('@' IN p_email) = 0 THEN
        RETURN jsonb_build_object('skipped', true, 'reason', 'invalid_email');
    END IF;

    SELECT id INTO existing FROM contacts WHERE email = p_email;
    IF existing IS NOT NULL THEN
        PERFORM touch_contact(existing);
        RETURN jsonb_build_object('id', existing, 'created', false);
    END IF;

    SELECT create_contact(
        COALESCE(NULLIF(trim(COALESCE(p_name, '')), ''), split_part(p_email, '@', 1)),
        p_email, NULL, NULL, NULL, NULL, ARRAY[]::TEXT[], COALESCE(p_source, 'manual')
    ) INTO new_id;
    RETURN jsonb_build_object('id', new_id, 'created', true);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION upsert_contacts_from_attendees(
    p_attendees JSONB,
    p_source TEXT DEFAULT 'calendar'
) RETURNS JSONB AS $$
DECLARE
    att JSONB;
    email TEXT;
    display TEXT;
    result JSONB;
    created INT := 0;
    updated INT := 0;
BEGIN
    FOR att IN SELECT value FROM jsonb_array_elements(COALESCE(p_attendees, '[]'::jsonb))
    LOOP
        IF jsonb_typeof(att) = 'string' THEN
            email := att #>> '{}';
            display := split_part(email, '@', 1);
        ELSE
            email := COALESCE(att->>'email', '');
            display := COALESCE(NULLIF(att->>'displayName', ''), NULLIF(att->>'name', ''),
                                split_part(email, '@', 1));
        END IF;

        result := upsert_contact(display, email, p_source);
        IF COALESCE((result->>'created')::boolean, false) THEN
            created := created + 1;
        ELSIF result ? 'id' THEN
            updated := updated + 1;
        END IF;
    END LOOP;

    RETURN jsonb_build_object('created', created, 'updated', updated);
END;
$$ LANGUAGE plpgsql;

-- 2.10: the tool-execution audit insert (was an inline INSERT in the hook).
CREATE OR REPLACE FUNCTION record_tool_execution(
    p_record JSONB
) RETURNS UUID AS $$
DECLARE
    rec_id UUID;
BEGIN
    INSERT INTO tool_executions (
        tool_name, arguments, tool_context, call_id, session_id,
        success, output, error, error_type, energy_spent, duration_seconds
    )
    VALUES (
        p_record->>'tool_name',
        COALESCE(p_record->'arguments', 'null'::jsonb),
        p_record->>'tool_context',
        p_record->>'call_id',
        p_record->>'session_id',
        COALESCE((p_record->>'success')::boolean, false),
        p_record->'output',
        p_record->>'error',
        p_record->>'error_type',
        COALESCE((p_record->>'energy_spent')::float, 0.0)::int,
        (p_record->>'duration_seconds')::float
    )
    RETURNING id INTO rec_id;
    RETURN rec_id;
END;
$$ LANGUAGE plpgsql;

-- Name-only variant for sources without addresses (fathom attendees):
-- approximate match by name, else create.
CREATE OR REPLACE FUNCTION upsert_contact_by_name(
    p_name TEXT,
    p_source TEXT DEFAULT 'manual'
) RETURNS JSONB AS $$
DECLARE
    existing BIGINT;
    new_id BIGINT;
BEGIN
    IF NULLIF(trim(COALESCE(p_name, '')), '') IS NULL THEN
        RETURN jsonb_build_object('skipped', true, 'reason', 'no_name');
    END IF;

    SELECT id INTO existing FROM contacts WHERE name ILIKE p_name LIMIT 1;
    IF existing IS NOT NULL THEN
        PERFORM touch_contact(existing);
        RETURN jsonb_build_object('id', existing, 'created', false);
    END IF;

    SELECT create_contact(trim(p_name), NULL, NULL, NULL, NULL, NULL,
                          ARRAY[]::TEXT[], COALESCE(p_source, 'manual'))
    INTO new_id;
    RETURN jsonb_build_object('id', new_id, 'created', true);
END;
$$ LANGUAGE plpgsql;

-- Channel-wizard pushdown: the setting catalog and config writes are
-- DB-owned; the CLI wizard only gathers answers.
CREATE OR REPLACE FUNCTION channel_setting_names(
    p_channel TEXT
) RETURNS TEXT[] AS $$
DECLARE
    catalog JSONB := '{
        "discord":  ["bot_token", "operator_user_id", "allowed_guilds"],
        "telegram": ["bot_token", "operator_user_id", "allowed_chat_ids"],
        "slack":    ["bot_token", "app_token", "signing_secret", "operator_user_id", "allowed_channels"],
        "signal":   ["phone_number", "api_url", "operator_user_id", "allowed_numbers"],
        "whatsapp": ["access_token", "phone_number_id", "verify_token", "webhook_port", "operator_user_id", "allowed_numbers"],
        "imessage": ["api_url", "password", "operator_recipient", "allowed_handles"],
        "matrix":   ["homeserver", "user_id", "access_token", "operator_user_id", "allowed_rooms"]
    }'::jsonb;
BEGIN
    IF NOT catalog ? COALESCE(p_channel, '') THEN
        RAISE EXCEPTION 'Unknown channel type: %; expected one of %',
            COALESCE(p_channel, '(null)'),
            (SELECT string_agg(key, ', ' ORDER BY key) FROM jsonb_object_keys(catalog) key);
    END IF;
    RETURN ARRAY(SELECT jsonb_array_elements_text(catalog->p_channel));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION apply_channel_config(
    p_channel TEXT,
    p_settings JSONB
) RETURNS JSONB AS $$
DECLARE
    known TEXT[] := channel_setting_names(p_channel);
    unknown TEXT[];
    applied TEXT[] := ARRAY[]::TEXT[];
    setting RECORD;
BEGIN
    IF jsonb_typeof(p_settings) <> 'object' OR p_settings = '{}'::jsonb THEN
        RAISE EXCEPTION 'settings must be a non-empty object of channel settings';
    END IF;
    unknown := ARRAY(
        SELECT key FROM jsonb_object_keys(p_settings) key
        WHERE key <> ALL(known) ORDER BY key);
    IF cardinality(unknown) > 0 THEN
        RAISE EXCEPTION 'Unknown % setting(s): %; expected among: %',
            p_channel, array_to_string(unknown, ', '), array_to_string(known, ', ');
    END IF;
    FOR setting IN SELECT key, value FROM jsonb_each(p_settings) LOOP
        PERFORM set_config('channel.' || p_channel || '.' || setting.key, setting.value);
        applied := array_append(applied, setting.key);
    END LOOP;
    RETURN jsonb_build_object('channel', p_channel, 'applied', to_jsonb(applied));
END;
$$ LANGUAGE plpgsql;
