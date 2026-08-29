-- DB-owned agent-loop state and external-call dispatch helpers.
SET search_path = public, ag_catalog, "$user";

CREATE OR REPLACE FUNCTION record_agent_turn_event(
    p_turn_id UUID,
    p_event_type TEXT,
    p_payload JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    event_id UUID;
BEGIN
    INSERT INTO agent_turn_events (turn_id, event_type, payload)
    VALUES (p_turn_id, p_event_type, COALESCE(p_payload, '{}'::jsonb))
    RETURNING id INTO event_id;
    RETURN jsonb_build_object('event_id', event_id::text);
END;
$$;

CREATE OR REPLACE FUNCTION start_agent_turn(
    p_mode TEXT,
    p_user_message TEXT,
    p_session_id UUID DEFAULT NULL,
    p_context JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    turn_id UUID;
    initial_messages JSONB;
    runtime JSONB;
BEGIN
    initial_messages := COALESCE(p_context->'messages', '[]'::jsonb);
    runtime := jsonb_build_object(
        'iterations', 0,
        'energy_spent', 0,
        'energy_budget', p_context->'energy_budget',
        'max_iterations', p_context->'max_iterations',
        'continuations_used', 0,
        'max_continuations', COALESCE(NULLIF(p_context->>'max_continuations', '')::int, 0),
        'last_text', '',
        'tool_calls_made', '[]'::jsonb,
        'phases_completed', '[]'::jsonb
    ) || COALESCE(p_context->'runtime_state', '{}'::jsonb);

    INSERT INTO agent_turns (
        mode, session_id, heartbeat_id, user_message, messages, runtime_state, phase, status
    )
    VALUES (
        COALESCE(NULLIF(p_mode, ''), 'chat'),
        p_session_id,
        _db_brain_try_uuid(p_context->>'heartbeat_id'),
        p_user_message,
        initial_messages,
        runtime,
        COALESCE(NULLIF(p_context->>'phase', ''), 'execute'),
        'running'
    )
    RETURNING id INTO turn_id;

    PERFORM record_agent_turn_event(turn_id, 'loop_start', p_context);
    RETURN jsonb_build_object('turn_id', turn_id::text, 'status', 'running', 'runtime_state', runtime);
END;
$$;

CREATE OR REPLACE FUNCTION next_agent_step(
    p_turn_id UUID
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    turn agent_turns%ROWTYPE;
    iterations INT;
    max_iterations INT;
    energy_spent INT;
    energy_budget INT;
BEGIN
    SELECT * INTO turn FROM agent_turns WHERE id = p_turn_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'agent turn not found: %', p_turn_id;
    END IF;
    IF turn.status <> 'running' THEN
        RETURN jsonb_build_object('action', 'done', 'status', turn.status, 'reason', turn.stopped_reason);
    END IF;

    iterations := COALESCE(NULLIF(turn.runtime_state->>'iterations', '')::int, 0);
    BEGIN max_iterations := NULLIF(turn.runtime_state->>'max_iterations', '')::int;
    EXCEPTION WHEN OTHERS THEN max_iterations := NULL; END;
    energy_spent := COALESCE(NULLIF(turn.runtime_state->>'energy_spent', '')::int, 0);
    BEGIN energy_budget := NULLIF(turn.runtime_state->>'energy_budget', '')::int;
    EXCEPTION WHEN OTHERS THEN energy_budget := NULL; END;

    IF max_iterations IS NOT NULL AND iterations >= max_iterations THEN
        UPDATE agent_turns
        SET stopped_reason = 'max_iterations',
            runtime_state = jsonb_set(runtime_state, '{stop_decision}', '"max_iterations"', true),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = p_turn_id;
        RETURN jsonb_build_object('action', 'stop', 'reason', 'max_iterations', 'iterations', iterations, 'energy_spent', energy_spent);
    END IF;
    IF energy_budget IS NOT NULL AND energy_spent >= energy_budget THEN
        PERFORM record_agent_turn_event(p_turn_id, 'energy_exhausted', jsonb_build_object('budget', energy_budget, 'spent', energy_spent));
        UPDATE agent_turns
        SET stopped_reason = 'energy',
            runtime_state = jsonb_set(runtime_state, '{stop_decision}', '"energy"', true),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = p_turn_id;
        RETURN jsonb_build_object('action', 'stop', 'reason', 'energy', 'iterations', iterations, 'energy_spent', energy_spent);
    END IF;

    -- Hand the caller the DB-authoritative message log to send to the model.
    -- The loop reads this back each step instead of keeping a parallel Python list.
    RETURN jsonb_build_object(
        'action', 'llm',
        'iteration', iterations + 1,
        'energy_spent', energy_spent,
        'messages', COALESCE(turn.messages, '[]'::jsonb)
    );
END;
$$;

CREATE OR REPLACE FUNCTION apply_agent_llm_result(
    p_turn_id UUID,
    p_result JSONB
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    turn agent_turns%ROWTYPE;
    content TEXT := COALESCE(p_result->>'content', '');
    tool_calls JSONB := COALESCE(p_result->'tool_calls', '[]'::jsonb);
    iterations INT;
    message JSONB;
    runtime JSONB;
BEGIN
    SELECT * INTO turn FROM agent_turns WHERE id = p_turn_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'agent turn not found: %', p_turn_id;
    END IF;
    iterations := COALESCE(NULLIF(turn.runtime_state->>'iterations', '')::int, 0) + 1;
    message := jsonb_build_object('role', 'assistant', 'content', content);
    IF jsonb_typeof(tool_calls) = 'array' AND jsonb_array_length(tool_calls) > 0 THEN
        message := message || jsonb_build_object('tool_calls', tool_calls);
    END IF;
    runtime := jsonb_set(turn.runtime_state, '{iterations}', to_jsonb(iterations), true);
    runtime := jsonb_set(runtime, '{last_text}', to_jsonb(content), true);
    runtime := jsonb_set(runtime, '{last_tool_calls}', tool_calls, true);

    UPDATE agent_turns
    SET messages = COALESCE(messages, '[]'::jsonb) || jsonb_build_array(message),
        runtime_state = runtime,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_turn_id;
    PERFORM record_agent_turn_event(p_turn_id, 'llm_result', p_result || jsonb_build_object('iteration', iterations));

    RETURN jsonb_build_object('turn_id', p_turn_id::text, 'iterations', iterations, 'tool_call_count', COALESCE(jsonb_array_length(tool_calls), 0));
END;
$$;

CREATE OR REPLACE FUNCTION apply_agent_tool_result(
    p_turn_id UUID,
    p_tool_call_id TEXT,
    p_result JSONB
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    turn agent_turns%ROWTYPE;
    spent INT := COALESCE(NULLIF(p_result->>'energy_spent', '')::int, 0);
    total_spent INT;
    energy_budget INT;
    call_record JSONB;
    runtime JSONB;
BEGIN
    SELECT * INTO turn FROM agent_turns WHERE id = p_turn_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'agent turn not found: %', p_turn_id;
    END IF;
    total_spent := COALESCE(NULLIF(turn.runtime_state->>'energy_spent', '')::int, 0) + spent;
    energy_budget := NULLIF(turn.runtime_state->>'energy_budget', '')::int;
    call_record := jsonb_build_object(
        'id', p_tool_call_id,
        'name', p_result->>'tool_name',
        'arguments', COALESCE(p_result->'arguments', '{}'::jsonb),
        'success', COALESCE((p_result->>'success')::boolean, false),
        'energy_spent', spent,
        'error', p_result->>'error'
    );
    runtime := jsonb_set(turn.runtime_state, '{energy_spent}', to_jsonb(total_spent), true);
    runtime := jsonb_set(runtime, '{tool_calls_made}', COALESCE(turn.runtime_state->'tool_calls_made', '[]'::jsonb) || jsonb_build_array(call_record), true);

    UPDATE agent_turns
    SET messages = COALESCE(messages, '[]'::jsonb) || jsonb_build_array(jsonb_build_object(
            'role', 'tool',
            'tool_call_id', p_tool_call_id,
            -- Budgeted turns see their remaining energy on every result (#44):
            -- the budget is a constraint the model must reason under, so it
            -- must be visible, not just enforced.
            'content', COALESCE(p_result->>'model_output', p_result->>'display_output', p_result->>'error', '')
                || CASE WHEN energy_budget IS NOT NULL
                        THEN E'\n\n[energy: ' || total_spent || '/' || energy_budget || ' spent]'
                        ELSE '' END
        )),
        runtime_state = runtime,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_turn_id;
    PERFORM record_agent_turn_event(p_turn_id, 'tool_result', p_result || jsonb_build_object('total_energy_spent', total_spent));

    RETURN jsonb_build_object('turn_id', p_turn_id::text, 'energy_spent', total_spent);
END;
$$;

-- Append an arbitrary message (user/system/etc.) to the DB-owned message log.
-- Used for continuation nudges and plan/verify phase prompts so the message
-- list stays authoritative in the DB rather than in a parallel Python list.
CREATE OR REPLACE FUNCTION append_agent_message(
    p_turn_id UUID,
    p_role TEXT,
    p_content TEXT
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    turn agent_turns%ROWTYPE;
BEGIN
    UPDATE agent_turns
    SET messages = COALESCE(messages, '[]'::jsonb)
            || jsonb_build_array(jsonb_build_object(
                'role', COALESCE(NULLIF(p_role, ''), 'user'),
                'content', COALESCE(p_content, '')
            )),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_turn_id
    RETURNING * INTO turn;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'agent turn not found: %', p_turn_id;
    END IF;

    RETURN jsonb_build_object('turn_id', p_turn_id::text, 'messages', turn.messages);
END;
$$;

-- Append trusted multimodal context produced inside the tool runtime. This is
-- separate from append_agent_message so arbitrary text callers cannot smuggle
-- provider-specific content shapes into the authoritative turn log.
CREATE OR REPLACE FUNCTION append_agent_visual_message(
    p_turn_id UUID,
    p_label TEXT,
    p_data_url TEXT
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    turn agent_turns%ROWTYPE;
BEGIN
    IF COALESCE(p_data_url, '') !~ '^data:image/(png|jpeg|webp|gif);base64,' THEN
        RAISE EXCEPTION 'visual message requires a supported image data URL';
    END IF;
    UPDATE agent_turns
    SET messages = COALESCE(messages, '[]'::jsonb)
            || jsonb_build_array(jsonb_build_object(
                'role', 'user',
                'content', jsonb_build_array(
                    jsonb_build_object(
                        'type', 'input_text',
                        'text', format('[Fresh tool-provided visual context: %s]',
                            COALESCE(NULLIF(p_label, ''), 'image'))
                    ),
                    jsonb_build_object(
                        'type', 'input_image',
                        'image_url', p_data_url,
                        'detail', 'auto'
                    )
                )
            )),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_turn_id
    RETURNING * INTO turn;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'agent turn not found: %', p_turn_id;
    END IF;
    RETURN jsonb_build_object('turn_id', p_turn_id::text, 'messages', turn.messages);
END;
$$;

CREATE OR REPLACE FUNCTION finish_agent_turn(
    p_turn_id UUID,
    p_result JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    row_out agent_turns%ROWTYPE;
BEGIN
    UPDATE agent_turns
    SET status = COALESCE(NULLIF(p_result->>'status', ''), 'completed'),
        stopped_reason = COALESCE(NULLIF(p_result->>'stopped_reason', ''), stopped_reason, 'completed'),
        result = COALESCE(p_result, '{}'::jsonb),
        completed_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_turn_id
    RETURNING * INTO row_out;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'agent turn not found: %', p_turn_id;
    END IF;
    PERFORM record_agent_turn_event(p_turn_id, 'loop_end', p_result);

    -- Heartbeat turns join the conscious-episode substrate so the extraction
    -- sweep covers autonomous activity too (#37). Advisory: a mirroring
    -- failure must never break turn finalization.
    IF row_out.mode = 'heartbeat' THEN
        BEGIN
            PERFORM record_heartbeat_episode_unit(row_out);
        EXCEPTION WHEN OTHERS THEN
            RAISE WARNING 'heartbeat episode mirroring failed for turn %: %', p_turn_id, SQLERRM;
        END;
    END IF;

    RETURN to_jsonb(row_out);
END;
$$;

CREATE OR REPLACE FUNCTION resolve_external_call_kind(
    p_call JSONB
) RETURNS JSONB
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    call_type TEXT := COALESCE(NULLIF(p_call->>'call_type', ''), NULLIF(p_call->>'type', ''));
    input JSONB := COALESCE(p_call->'input', p_call);
    think_kind TEXT;
BEGIN
    IF call_type = 'tool_use' OR input ? 'tool_name' OR input ? 'name' THEN
        RETURN jsonb_build_object('call_type', 'tool_use', 'kind', 'tool_use', 'input', input);
    END IF;
    IF call_type = 'embed' THEN
        RETURN jsonb_build_object('call_type', 'embed', 'kind', 'embed', 'input', input, 'supported', false);
    END IF;
    think_kind := COALESCE(NULLIF(input->>'kind', ''), 'heartbeat_decision');
    RETURN jsonb_build_object('call_type', 'think', 'kind', think_kind, 'input', input, 'supported', think_kind IN (
        'heartbeat_decision_rlm',
        'heartbeat_decision',
        'brainstorm_goals',
        'inquire',
        'reflect',
        'termination_confirm',
        'consent_request'
    ));
END;
$$;

CREATE OR REPLACE FUNCTION apply_think_result(
    p_call JSONB,
    p_result JSONB
) RETURNS JSONB
LANGUAGE sql
AS $$
    SELECT apply_external_call_result(p_call, p_result);
$$;

CREATE OR REPLACE FUNCTION apply_tool_use_result(
    p_call JSONB,
    p_result JSONB
) RETURNS JSONB
LANGUAGE sql
AS $$
    SELECT apply_external_call_result(p_call, p_result);
$$;
