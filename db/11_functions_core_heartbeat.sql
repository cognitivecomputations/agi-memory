-- Hexis schema: core heartbeat functions.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

CREATE OR REPLACE FUNCTION start_heartbeat()
RETURNS JSONB AS $$
DECLARE
    heartbeat_id UUID;
    state_record RECORD;
    new_energy FLOAT;
    regeneration JSONB;
    economy JSONB;
    context JSONB;
    decision_max_tokens INT;
    hb_number INT;
    external_calls JSONB := '[]'::jsonb;
BEGIN
    IF NOT is_agent_configured() THEN
        RETURN NULL;
    END IF;
    IF NOT is_init_complete() THEN
        RETURN NULL;
    END IF;

    PERFORM ensure_emotion_bootstrap();
    PERFORM ensure_self_node();
    PERFORM ensure_current_life_chapter();
    SELECT * INTO state_record FROM heartbeat_state WHERE id = 1;
    hb_number := state_record.heartbeat_count + 1;
    heartbeat_id := gen_random_uuid();
    PERFORM update_drives();
    regeneration := regenerate_heartbeat_energy(CURRENT_TIMESTAMP);
    new_energy := COALESCE(
        NULLIF(regeneration->>'after_energy', '')::float,
        state_record.current_energy
    );
    UPDATE heartbeat_state SET
        current_energy = new_energy,
        heartbeat_count = hb_number,
        last_heartbeat_at = CURRENT_TIMESTAMP,
        next_heartbeat_at = NULL,
        active_heartbeat_id = heartbeat_id,
        active_heartbeat_number = hb_number,
        active_actions = '[]'::jsonb,
        active_reasoning = NULL,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;
    economy := begin_heartbeat_outcome(heartbeat_id, hb_number, regeneration);
    IF COALESCE(get_config_bool('heartbeat.use_rlm'), FALSE) THEN
        context := gather_turn_snapshot();
    ELSE
        context := gather_turn_context();
    END IF;
    context := context || jsonb_build_object(
        'belief_updates_recent',
        recent_belief_updates_json(20, state_record.last_heartbeat_at),
        'heartbeat_economy', economy
    );
    decision_max_tokens := get_config_int('heartbeat.max_decision_tokens');
    external_calls := jsonb_build_array(
        build_external_call(
            'think',
            jsonb_build_object(
                'kind', CASE
                    WHEN COALESCE(get_config_bool('heartbeat.use_rlm'), FALSE) THEN 'heartbeat_decision_rlm'
                    ELSE 'heartbeat_decision'
                END,
                'context', context,
                'heartbeat_id', heartbeat_id,
                'max_tokens', decision_max_tokens
            )
        )
    );

    RETURN jsonb_build_object(
        'heartbeat_id', heartbeat_id,
        'heartbeat_number', hb_number,
        'external_calls', external_calls,
        'outbox_messages', '[]'::jsonb
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION run_heartbeat()
RETURNS JSONB AS $$
DECLARE
    hb_payload JSONB;
BEGIN
    IF NOT should_run_heartbeat() THEN
        RETURN NULL;
    END IF;
    hb_payload := start_heartbeat();

    RETURN hb_payload;
END;
$$ LANGUAGE plpgsql;

SET check_function_bodies = on;
