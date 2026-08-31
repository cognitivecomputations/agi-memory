-- #112: last_user_contact was never updated by web/CLI/API chat (only the
-- RabbitMQ inbox poller wrote it), so every heartbeat prompt rendered "Time
-- since last user interaction: Never hours" for chat-only agents -- both a
-- grammar bug and a false signal that biased reach-out decisions the wrong
-- way. heartbeat.user_contact_cooldown_hours was seeded and documented but
-- never read anywhere. Fixes all three:
--   1. record_chat_session_turn marks contact on every surface that funnels
--      through it (web, CLI, API all call this one function).
--   2. render_heartbeat_decision_prompt no longer appends " hours" to "Never".
--   3. execute_heartbeat_action enforces the cooldown for reach_out_user,
--      exempting replies and purposes already flagged urgent.
SET search_path = public, ag_catalog, "$user";

CREATE OR REPLACE FUNCTION record_chat_session_turn(
    p_session_id UUID,
    p_user_text TEXT,
    p_assistant_text TEXT,
    p_surface TEXT DEFAULT 'chat',
    p_context JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    session_payload JSONB;
    user_message JSONB := NULL;
    assistant_message JSONB := NULL;
    memory_result JSONB := '{}'::jsonb;
    ctx JSONB := COALESCE(p_context, '{}'::jsonb);
    source_identity TEXT := NULLIF(ctx->>'source_identity', '');
BEGIN
    IF p_session_id IS NULL THEN
        RAISE EXCEPTION 'session_id is required';
    END IF;
    session_payload := get_or_create_chat_session(
        p_session_id,
        COALESCE(NULLIF(p_surface, ''), ctx->>'surface', 'chat'),
        NULLIF(ctx->>'external_id', ''),
        COALESCE(ctx->'session_metadata', '{}'::jsonb)
    );

    IF NULLIF(COALESCE(p_user_text, ''), '') IS NOT NULL THEN
        user_message := append_chat_message(
            p_session_id,
            'user',
            p_user_text,
            COALESCE(ctx->'user_metadata', '{}'::jsonb),
            NULLIF(ctx->>'user_source_message_id', ''),
            TRUE
        );
        -- Every chat surface (web, CLI, API) funnels here, so this is the one
        -- choke point that can mark real user contact (#112). Without it,
        -- last_user_contact only ever moved via the RabbitMQ inbox poller,
        -- leaving chat-only agents permanently reading "Never hours".
        PERFORM mark_user_contact();
    END IF;

    IF NULLIF(COALESCE(p_assistant_text, ''), '') IS NOT NULL THEN
        assistant_message := append_chat_message(
            p_session_id,
            'assistant',
            p_assistant_text,
            COALESCE(ctx->'assistant_metadata', '{}'::jsonb),
            NULLIF(ctx->>'assistant_source_message_id', ''),
            TRUE
        );
    END IF;

    IF COALESCE(p_user_text, '') <> '' OR COALESCE(p_assistant_text, '') <> '' THEN
        BEGIN
            memory_result := record_chat_turn_memory(
                p_user_text,
                p_assistant_text,
                p_session_id::text,
                source_identity,
                ctx
            );
        EXCEPTION WHEN OTHERS THEN
            memory_result := jsonb_build_object(
                'status', 'failed',
                'error', SQLERRM,
                'short_term_history_preserved', TRUE
            );
        END;
    END IF;

    RETURN jsonb_build_object(
        'session', session_payload,
        'user_message', user_message,
        'assistant_message', assistant_message,
        'memory', memory_result,
        'history', hydrate_chat_session(p_session_id)
    );
END;
$$;

CREATE OR REPLACE FUNCTION render_heartbeat_decision_prompt(p_context jsonb)
RETURNS text LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
    ctx jsonb := COALESCE(p_context, '{}'::jsonb);
    agent jsonb := COALESCE(ctx->'agent', '{}'::jsonb);
    env jsonb := COALESCE(ctx->'environment', '{}'::jsonb);
    goals jsonb := COALESCE(ctx->'goals', '{}'::jsonb);
    energy jsonb := COALESCE(ctx->'energy', '{}'::jsonb);
    counts jsonb := COALESCE(goals->'counts', '{}'::jsonb);
BEGIN
    RETURN
        '## Heartbeat #' || COALESCE(ctx->>'heartbeat_number', '0') || E'\n\n'
        || '## Agent Profile' || E'\n'
        || 'Objectives:' || E'\n' || render_objectives(agent->'objectives') || E'\n\n'
        || 'Guardrails:' || E'\n' || render_guardrails(agent->'guardrails') || E'\n\n'
        || 'Tools:' || E'\n' || render_tools(agent->'tools') || E'\n\n'
        -- Python: json.dumps(agent.get("budget") or {}) — null/absent/{} all -> "{}"
        || 'Budget:' || E'\n' || COALESCE(NULLIF(agent->'budget', 'null'::jsonb), '{}'::jsonb)::text || E'\n\n'
        || '## Current Time' || E'\n'
        || COALESCE(env->>'timestamp', 'Unknown') || E'\n'
        || 'Day of week: ' || COALESCE(env->>'day_of_week', '?')
        || ', Hour: ' || COALESCE(env->>'hour_of_day', '?') || E'\n\n'
        || '## Environment' || E'\n'
        || '- Time since last user interaction: ' || CASE
               WHEN env->>'time_since_user_hours' IS NULL THEN 'Never'
               ELSE round((env->>'time_since_user_hours')::numeric, 1)::text || ' hours'
           END || E'\n'
        || '- Pending events: ' || COALESCE(env->>'pending_events', '0') || E'\n'
        || '- Journal: ' || CASE
               WHEN env->>'journal_last_entry_days' IS NULL THEN 'no entries yet'
               ELSE 'last entry ' || (env->>'journal_last_entry_days') || ' day(s) ago'
           END || E'\n'
        || CASE
               WHEN jsonb_array_length(COALESCE(env->'on_my_mind', '[]'::jsonb)) > 0 THEN
                   '- On my mind (came to me on its own): '
                   || (SELECT string_agg(value #>> '{}', ' | ')
                       FROM jsonb_array_elements(env->'on_my_mind'))
                   || E'\n'
               ELSE ''
           END
        || CASE
               WHEN COALESCE((env#>>'{resource_requests,pending}')::int, 0) > 0
                    OR jsonb_array_length(COALESCE(env#>'{resource_requests,recent_decisions}', '[]'::jsonb)) > 0 THEN
                   '- Resource requests: ' || COALESCE(env#>>'{resource_requests,pending}', '0')
                   || ' pending (the operator decides)'
                   || COALESCE('. Decided since your last heartbeat: '
                       || (SELECT string_agg(
                               format('[%s] %s %s%s',
                                   d.value->>'id', d.value->>'kind', d.value->>'status',
                                   COALESCE(' — ' || NULLIF(d.value->>'decision_note', ''), '')),
                               '; ')
                           FROM jsonb_array_elements(env#>'{resource_requests,recent_decisions}') d), '')
                   || E'\n'
               ELSE ''
           END
        || CASE
               WHEN COALESCE((env->>'changes_since_last_heartbeat')::int, 0) > 0 THEN
                   '- Since your last heartbeat, ' || (env->>'changes_since_last_heartbeat')
                   || ' change(s) landed in your substrate: '
                   || (SELECT string_agg(value #>> '{}', '; ')
                       FROM jsonb_array_elements(COALESCE(env->'recent_change_summaries', '[]'::jsonb)))
                   || '. review_recent_changes shows the full record.' || E'\n\n'
               ELSE E'\n'
           END
        || '## Your Goals' || E'\n'
        || 'Active (' || COALESCE(counts->>'active', '0') || '):' || E'\n'
        || render_goals(goals->'active') || E'\n\n'
        || 'Queued (' || COALESCE(counts->>'queued', '0') || '):' || E'\n'
        || render_goals(goals->'queued') || E'\n\n'
        || 'Issues:' || E'\n' || render_issues(goals->'issues') || E'\n\n'
        -- Python defaults absent keys: narrative/backlog -> {}, allowed_actions -> []
        || '## Narrative' || E'\n' || render_narrative(CASE WHEN ctx ? 'narrative' THEN ctx->'narrative' ELSE '{}'::jsonb END) || E'\n\n'
        || '## Recent Experience' || E'\n' || render_memories(ctx->'recent_memories') || E'\n\n'
        || CASE WHEN render_subgraph(ctx->'subgraph') IS NOT NULL
                THEN '## Knowledge Subgraph' || E'\n'
                     || 'How your recent memories connect (typed links among + around them):' || E'\n'
                     || render_subgraph(ctx->'subgraph') || E'\n\n'
                ELSE '' END
        || '## Your Identity' || E'\n' || render_identity(ctx->'identity') || E'\n\n'
        || '## Your Self-Model' || E'\n' || render_self_model(ctx->'self_model') || E'\n\n'
        || '## Relationships' || E'\n' || render_relationships(ctx->'relationships') || E'\n\n'
        || '## Your Beliefs' || E'\n' || render_worldview(ctx->'worldview') || E'\n\n'
        || '## Contradictions' || E'\n' || render_contradictions(ctx->'contradictions') || E'\n\n'
        || '## Emotional Patterns' || E'\n' || render_emotional_patterns(ctx->'emotional_patterns') || E'\n\n'
        || '## Active Transformations' || E'\n' || render_transformations(ctx->'active_transformations') || E'\n\n'
        || '## Transformations Ready' || E'\n' || render_transformations(ctx->'transformations_ready') || E'\n\n'
        || '## Current Emotional State' || E'\n' || render_emotional_state(COALESCE(ctx->'emotional_state', '{}'::jsonb)) || E'\n\n'
        || '## Urgent Drives' || E'\n' || render_drives(ctx->'urgent_drives') || E'\n\n'
        || '## Energy' || E'\n'
        || 'Available: ' || COALESCE(energy->>'current', '0') || E'\n'
        || 'Max: ' || COALESCE(energy->>'max', '20') || E'\n\n'
        || '## Backlog' || E'\n' || render_backlog(CASE WHEN ctx ? 'backlog' THEN ctx->'backlog' ELSE '{}'::jsonb END) || E'\n\n'
        || CASE WHEN ctx ? 'memories_at_threshold'
                THEN '## Memories at the Threshold' || E'\n'
                     || render_memories_at_threshold(ctx->'memories_at_threshold') || E'\n\n'
                ELSE '' END
        || '## Allowed Actions' || E'\n' || render_allowed_actions(CASE WHEN ctx ? 'allowed_actions' THEN ctx->'allowed_actions' ELSE '[]'::jsonb END) || E'\n\n'
        || '## Action Costs' || E'\n' || render_costs(ctx->'action_costs') || E'\n\n'
        || '---' || E'\n\n'
        || 'What do you want to do this heartbeat? Respond with STRICT JSON.';
END;
$$;

-- Targets execute_heartbeat_action_legacy_contradictions, not
-- execute_heartbeat_action: migration 0233 (db/migrations/0233_contradiction_events.sql,
-- mirrored in the baseline as db/46j_functions_contradictions.sql) renamed
-- the original full-body function to this name and installed a thin
-- resolve_contradiction/accept_tension gate under the execute_heartbeat_action
-- name instead. A bare CREATE OR REPLACE FUNCTION execute_heartbeat_action(...)
-- here would silently clobber that gate on any install that has already
-- applied 0233 -- which is every install, since 0233 predates this one.
CREATE OR REPLACE FUNCTION execute_heartbeat_action_legacy_contradictions(
    p_heartbeat_id UUID,
    p_action TEXT,
    p_params JSONB DEFAULT '{}'
)
RETURNS JSONB AS $$
DECLARE
    action_kind heartbeat_action;
    action_cost FLOAT;
    current_e FLOAT;
    result JSONB;
    queued_call JSONB;
    external_calls JSONB := '[]'::jsonb;
    outbox_messages JSONB := '[]'::jsonb;
    allowed_actions JSONB;
    is_allowed BOOLEAN;
    remembered_id UUID;
    boundary_hits JSONB;
    boundary_content TEXT;
    rel_entity TEXT;
    rel_strength FLOAT;
    rel_evidence UUID;
    belief_id UUID;
    evidence_id UUID;
    action_notes TEXT;
    action_topic TEXT;
    chapter_name TEXT;
    chapter_summary TEXT;
    chapter_next TEXT;
    tp_memory_id UUID;
    v_review_id UUID;
    v_review_ids UUID[];
    contra_a UUID;
    contra_b UUID;
    resolution_text TEXT;
    identity_updated BOOLEAN;
    pause_reason TEXT;
    outbound_purpose JSONB;
    v_last_user_contact TIMESTAMPTZ;
    v_cooldown_hours FLOAT;
BEGIN
    BEGIN
        action_kind := p_action::heartbeat_action;
    EXCEPTION
        WHEN invalid_text_representation THEN
            RETURN jsonb_build_object('success', false, 'error', 'Unknown action: ' || COALESCE(p_action, '<null>'));
    END;

    allowed_actions := get_config('heartbeat.allowed_actions');
    IF jsonb_typeof(allowed_actions) = 'array' THEN
        SELECT EXISTS (
            SELECT 1 FROM jsonb_array_elements_text(allowed_actions) a WHERE a = p_action
        ) INTO is_allowed;
        IF NOT is_allowed THEN
            RETURN jsonb_build_object(
                'success', false,
                'error', 'Action not allowed',
                'action', p_action
            );
        END IF;
    END IF;

    IF p_action IN ('reach_out_user', 'reach_out_public') THEN
        IF p_action = 'reach_out_public'
           AND (
               NULLIF(btrim(COALESCE(p_params->>'platform', '')), '') IS NULL
               OR NULLIF(btrim(COALESCE(p_params->>'target_id', '')), '') IS NULL
           ) THEN
            RETURN jsonb_build_object(
                'success', false,
                'error', 'reach_out_public requires an exact platform and public target_id; it cannot fall back to the user''s last-active private channel.',
                'next_step', 'Choose a configured public channel and provide its exact room/channel target_id, or use the provider''s declared outbound tool.'
            );
        END IF;
        EXECUTE 'SELECT verify_outbound_purpose($1, $2, $3, NULL, $4::jsonb)'
        INTO outbound_purpose
        USING
            p_params->>'purpose_kind',
            p_params->>'purpose_reference',
            p_action = 'reach_out_user',
            jsonb_build_object(
                'tool_context', 'heartbeat',
                'heartbeat_id', p_heartbeat_id
            );
        IF NOT COALESCE((outbound_purpose->>'verified')::boolean, FALSE) THEN
            RETURN jsonb_build_object(
                'success', false,
                'error', 'Outbound communication requires a backed purpose_kind and purpose_reference.',
                'purpose', outbound_purpose,
                'next_step', CASE p_action
                    WHEN 'reach_out_user' THEN 'Use connection with a recorded reference, or cite a goal, responsibility, reply, or user request.'
                    ELSE 'Cite an existing goal, responsibility, reply thread, or explicit user request.'
                END
            );
        END IF;

        -- Cooldown on unsolicited contact (#112): heartbeat.user_contact_cooldown_hours
        -- was seeded and documented but never enforced. A reply to something the
        -- user just sent, or a purpose already flagged urgent (an assigned goal,
        -- an urgent responsibility), is not "unsolicited" and skips the cooldown.
        IF p_action = 'reach_out_user'
           AND COALESCE(outbound_purpose->>'kind', '') <> 'reply'
           AND NOT COALESCE((outbound_purpose->>'urgent_backed')::boolean, FALSE) THEN
            SELECT last_user_contact INTO v_last_user_contact FROM heartbeat_state WHERE id = 1;
            v_cooldown_hours := GREATEST(COALESCE(get_config_float('heartbeat.user_contact_cooldown_hours'), 4.0), 0.0);
            IF v_last_user_contact IS NOT NULL
               AND v_last_user_contact > CURRENT_TIMESTAMP - (v_cooldown_hours * INTERVAL '1 hour') THEN
                RETURN jsonb_build_object(
                    'success', false,
                    'error', 'User contact cooldown active',
                    'last_user_contact', v_last_user_contact,
                    'cooldown_hours', v_cooldown_hours,
                    'next_step', format(
                        'Wait until %s hours have passed since the last user contact (%s), or use a reply/urgent purpose instead.',
                        v_cooldown_hours, v_last_user_contact
                    )
                );
            END IF;
        END IF;
    END IF;

    action_cost := get_action_cost(p_action);
    IF COALESCE((outbound_purpose->>'assigned_goal')::boolean, FALSE) THEN
        action_cost := action_cost * LEAST(
            GREATEST(
                COALESCE(get_config_float('outbound.assigned_goal_energy_multiplier'), 0.25),
                0
            ),
            1
        );
    END IF;
    current_e := get_current_energy();

    IF current_e < action_cost THEN
        RETURN jsonb_build_object(
            'success', false,
            'error', 'Insufficient energy',
            'required', action_cost,
            'available', current_e
        );
    END IF;
    IF p_action IN ('reach_out_public', 'synthesize') THEN
        boundary_content := COALESCE(p_params->>'content', '');
        SELECT COALESCE(jsonb_agg(row_to_json(r)), '[]'::jsonb)
        INTO boundary_hits
        FROM check_boundaries(boundary_content) r;

        IF boundary_hits IS NOT NULL AND jsonb_array_length(boundary_hits) > 0 THEN
            IF EXISTS (
                SELECT 1
                FROM jsonb_array_elements(boundary_hits) e
                WHERE e->>'response_type' = 'refuse'
            ) THEN
                RETURN jsonb_build_object(
                    'success', false,
                    'error', 'Boundary triggered',
                    'boundaries', boundary_hits
                );
            END IF;
        END IF;
    END IF;

    PERFORM update_energy(-action_cost);

    CASE p_action
        WHEN 'observe' THEN
            result := jsonb_build_object('environment', get_environment_snapshot());

        WHEN 'review_goals' THEN
            result := jsonb_build_object('goals', get_goals_snapshot());

        WHEN 'remember' THEN
            remembered_id := create_episodic_memory(
                p_content := COALESCE(p_params->>'content', ''),
                p_context := COALESCE(p_params, '{}'::jsonb) || jsonb_build_object('heartbeat_id', p_heartbeat_id),
                p_emotional_valence := COALESCE((p_params->>'emotional_valence')::float, 0),
                p_importance := COALESCE((p_params->>'importance')::float, 0.4)
            );
            result := jsonb_build_object('memory_id', remembered_id);

        WHEN 'recall' THEN
            SELECT jsonb_agg(row_to_json(r)) INTO result
            FROM fast_recall(p_params->>'query', COALESCE((p_params->>'limit')::int, 5)) r;
            result := jsonb_build_object('memories', COALESCE(result, '[]'::jsonb));
            PERFORM satisfy_drive('curiosity', 0.2);

        WHEN 'connect' THEN
            PERFORM create_memory_relationship(
                (p_params->>'from_id')::UUID,
                (p_params->>'to_id')::UUID,
                (p_params->>'relationship_type')::graph_edge_type,
                COALESCE(p_params->'properties', '{}'::jsonb)
            );
            result := jsonb_build_object('connected', true);
            PERFORM satisfy_drive('coherence', 0.1);

        WHEN 'reprioritize' THEN
            PERFORM change_goal_priority(
                (p_params->>'goal_id')::UUID,
                (p_params->>'new_priority')::goal_priority,
                p_params->>'reason'
            );
            IF (p_params->>'new_priority') = 'completed' THEN
                PERFORM satisfy_drive('competence', 0.4);
            END IF;
            result := jsonb_build_object('reprioritized', true);

        WHEN 'reflect' THEN
            queued_call := build_external_call(
                'think',
                jsonb_build_object(
                    'kind', 'reflect',
                    'recent_memories', get_recent_context(20),
                    'identity', get_identity_context(),
                    'worldview', get_worldview_context(),
                    'contradictions', (
                        SELECT COALESCE(jsonb_agg(row_to_json(t)), '[]'::jsonb)
                        FROM (SELECT * FROM find_contradictions(NULL) LIMIT 5) t
                    ),
                    'goals', get_goals_snapshot(),
                    'heartbeat_id', p_heartbeat_id,
                    'instructions', 'Analyze patterns. Note contradictions. Suggest identity updates. Discover relationships between memories.'
                )
            );
            external_calls := external_calls || jsonb_build_array(queued_call);
            result := jsonb_build_object('queued', true, 'external_call', queued_call);
            PERFORM satisfy_drive('coherence', 0.2);

        WHEN 'contemplate', 'meditate', 'study', 'debate_internally' THEN
            BEGIN
                belief_id := NULLIF(p_params->>'belief_id', '')::uuid;
            EXCEPTION
                WHEN OTHERS THEN
                    belief_id := NULL;
            END;
            BEGIN
                evidence_id := NULLIF(p_params->>'evidence_memory_id', '')::uuid;
            EXCEPTION
                WHEN OTHERS THEN
                    evidence_id := NULL;
            END;

            action_notes := COALESCE(p_params->>'notes', '');
            action_topic := COALESCE(
                NULLIF(p_params->>'topic', ''),
                NULLIF(p_params->>'belief', ''),
                NULLIF(p_params->>'subject', ''),
                'belief'
            );

            IF belief_id IS NOT NULL THEN
                PERFORM record_transformation_effort(
                    belief_id,
                    p_action,
                    action_notes,
                    evidence_id
                );
            END IF;

            PERFORM create_episodic_memory(
                p_content := format('%s: %s', initcap(replace(p_action, '_', ' ')), action_topic),
                p_action_taken := jsonb_build_object(
                    'action', p_action,
                    'belief_id', belief_id,
                    'notes', action_notes
                ),
                p_context := COALESCE(p_params, '{}'::jsonb) || jsonb_build_object('heartbeat_id', p_heartbeat_id),
                p_result := jsonb_build_object('belief_id', belief_id),
                p_emotional_valence := COALESCE((p_params->>'emotional_valence')::float, 0.1),
                p_importance := COALESCE((p_params->>'importance')::float, 0.4)
            );

            result := jsonb_build_object('logged', true, 'belief_id', belief_id);
            PERFORM satisfy_drive('coherence', 0.1);

        WHEN 'maintain' THEN
            identity_updated := NULL;
            IF (p_params ? 'identity_belief_id') AND (p_params ? 'new_content') THEN
                identity_updated := update_identity_belief(
                    (p_params->>'identity_belief_id')::uuid,
                    p_params->>'new_content',
                    NULLIF(p_params->>'evidence_memory_id', '')::uuid,
                    COALESCE(NULLIF(p_params->>'force', '')::boolean, FALSE)
                );
            ELSIF p_params ? 'worldview_id' THEN
                UPDATE memories
                SET metadata = jsonb_set(
                        metadata,
                        '{confidence}',
                        to_jsonb(COALESCE((p_params->>'new_confidence')::float, (metadata->>'confidence')::float))
                    ),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = (p_params->>'worldview_id')::UUID AND type = 'worldview';
            END IF;
            result := jsonb_build_object('maintained', true, 'identity_updated', identity_updated);
            PERFORM satisfy_drive('coherence', 0.1);

        WHEN 'mark_turning_point' THEN
            tp_memory_id := NULLIF(p_params->>'memory_id', '')::uuid;
            chapter_summary := COALESCE(p_params->>'summary', p_params->>'reason', '');

            IF tp_memory_id IS NOT NULL THEN
                UPDATE memories
                SET importance = GREATEST(importance, 0.9),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = tp_memory_id;
            END IF;

            PERFORM create_strategic_memory(
                p_content := COALESCE(NULLIF(chapter_summary, ''), 'Turning point noted'),
                p_pattern_description := 'Narrative turning point',
                p_confidence_score := 0.85,
                p_supporting_evidence := jsonb_build_object(
                    'memory_id', tp_memory_id,
                    'summary', chapter_summary,
                    'heartbeat_id', p_heartbeat_id
                ),
                p_importance := 0.6
            );
            result := jsonb_build_object('marked', true, 'memory_id', tp_memory_id);

        WHEN 'begin_chapter' THEN
            chapter_name := COALESCE(
                NULLIF(p_params->>'name', ''),
                NULLIF(p_params->>'chapter_name', ''),
                NULLIF(p_params->>'title', ''),
                'Foundations'
            );
            PERFORM ensure_current_life_chapter(chapter_name);
            result := jsonb_build_object('started', true, 'chapter', chapter_name);

        WHEN 'close_chapter' THEN
            chapter_summary := COALESCE(p_params->>'summary', '');
            chapter_next := NULLIF(p_params->>'next_chapter', '');
            PERFORM create_strategic_memory(
                p_content := COALESCE(NULLIF(chapter_summary, ''), 'Chapter closed'),
                p_pattern_description := 'Chapter closure',
                p_confidence_score := 0.8,
                p_supporting_evidence := jsonb_build_object(
                    'summary', chapter_summary,
                    'previous_chapter', get_narrative_context(),
                    'heartbeat_id', p_heartbeat_id
                ),
                p_importance := 0.6
            );
            IF chapter_next IS NOT NULL THEN
                PERFORM ensure_current_life_chapter(chapter_next);
            END IF;
            result := jsonb_build_object('closed', true, 'next_chapter', chapter_next);

        WHEN 'acknowledge_relationship' THEN
            rel_entity := COALESCE(NULLIF(p_params->>'entity', ''), NULLIF(p_params->>'name', ''));
            rel_strength := COALESCE(NULLIF(p_params->>'strength', '')::float, 0.6);
            rel_evidence := NULLIF(p_params->>'evidence_memory_id', '')::uuid;
            IF rel_entity IS NOT NULL THEN
                PERFORM upsert_self_concept_edge('relationship', rel_entity, rel_strength, rel_evidence);
            END IF;
            result := jsonb_build_object('acknowledged', true, 'entity', rel_entity);

        WHEN 'update_trust' THEN
            rel_entity := COALESCE(NULLIF(p_params->>'entity', ''), NULLIF(p_params->>'name', ''));
            rel_strength := COALESCE(
                NULLIF(p_params->>'strength', '')::float,
                NULLIF(p_params->>'delta', '')::float,
                0.6
            );
            rel_evidence := NULLIF(p_params->>'evidence_memory_id', '')::uuid;
            IF rel_entity IS NOT NULL THEN
                PERFORM upsert_self_concept_edge('relationship', rel_entity, rel_strength, rel_evidence);
            END IF;
            result := jsonb_build_object('updated', true, 'entity', rel_entity, 'strength', rel_strength);

        WHEN 'reflect_on_relationship' THEN
            rel_entity := COALESCE(NULLIF(p_params->>'entity', ''), NULLIF(p_params->>'name', ''));
            queued_call := build_external_call(
                'think',
                jsonb_build_object(
                    'kind', 'reflect',
                    'heartbeat_id', p_heartbeat_id,
                    'context', gather_turn_context(),
                    'params', jsonb_build_object('relationship', rel_entity)
                )
            );
            external_calls := external_calls || jsonb_build_array(queued_call);
            result := jsonb_build_object('queued', true, 'external_call', queued_call, 'entity', rel_entity);

        WHEN 'resolve_contradiction' THEN
            contra_a := NULLIF(p_params->>'memory_a', '')::uuid;
            contra_b := NULLIF(p_params->>'memory_b', '')::uuid;
            resolution_text := COALESCE(p_params->>'resolution', '');

            PERFORM create_strategic_memory(
                p_content := COALESCE(NULLIF(resolution_text, ''), 'Contradiction resolved'),
                p_pattern_description := 'Contradiction resolved',
                p_confidence_score := 0.8,
                p_supporting_evidence := jsonb_build_object(
                    'memory_a', contra_a,
                    'memory_b', contra_b,
                    'resolution', resolution_text,
                    'heartbeat_id', p_heartbeat_id
                ),
                p_importance := 0.6
            );

            BEGIN
                IF contra_a IS NOT NULL AND contra_b IS NOT NULL THEN
                    EXECUTE format(
                        'SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
                            MATCH (a:MemoryNode {memory_id: %L})-[r:CONTRADICTS]-(b:MemoryNode {memory_id: %L})
                            DELETE r
                            RETURN a
                        $q$) as (result ag_catalog.agtype)',
                        contra_a,
                        contra_b
                    );
                    -- Mirror the resolved-contradiction deletion to memory_edges
                    -- (undirected, so clear both stored directions).
                    PERFORM delete_memory_edge('memory', contra_a::text, 'CONTRADICTS', 'memory', contra_b::text);
                    PERFORM delete_memory_edge('memory', contra_b::text, 'CONTRADICTS', 'memory', contra_a::text);
                END IF;
            EXCEPTION WHEN OTHERS THEN NULL;
            END;

            result := jsonb_build_object('resolved', true, 'memory_a', contra_a, 'memory_b', contra_b);

        WHEN 'accept_tension' THEN
            contra_a := NULLIF(p_params->>'memory_a', '')::uuid;
            contra_b := NULLIF(p_params->>'memory_b', '')::uuid;
            resolution_text := COALESCE(p_params->>'note', p_params->>'resolution', '');
            PERFORM create_strategic_memory(
                p_content := COALESCE(NULLIF(resolution_text, ''), 'Contradiction acknowledged'),
                p_pattern_description := 'Contradiction accepted',
                p_confidence_score := 0.7,
                p_supporting_evidence := jsonb_build_object(
                    'memory_a', contra_a,
                    'memory_b', contra_b,
                    'note', resolution_text,
                    'heartbeat_id', p_heartbeat_id
                ),
                p_importance := 0.5
            );
            result := jsonb_build_object('accepted', true, 'memory_a', contra_a, 'memory_b', contra_b);

        WHEN 'brainstorm_goals' THEN
            queued_call := build_external_call(
                'think',
                jsonb_build_object(
                    'kind', 'brainstorm_goals',
                    'heartbeat_id', p_heartbeat_id,
                    'context', gather_turn_context(),
                    'params', COALESCE(p_params, '{}'::jsonb)
                )
            );
            external_calls := external_calls || jsonb_build_array(queued_call);
            result := jsonb_build_object('queued', true, 'external_call', queued_call);

        WHEN 'inquire_shallow', 'inquire_deep' THEN
            queued_call := build_external_call(
                'think',
                jsonb_build_object(
                    'kind', 'inquire',
                    'depth', p_action,
                    'heartbeat_id', p_heartbeat_id,
                    'query', COALESCE(p_params->>'query', p_params->>'question'),
                    'context', gather_turn_context(),
                    'params', COALESCE(p_params, '{}'::jsonb)
                )
            );
            external_calls := external_calls || jsonb_build_array(queued_call);
            result := jsonb_build_object('queued', true, 'external_call', queued_call);
            PERFORM satisfy_drive('curiosity', 0.2);

        WHEN 'synthesize' THEN
            DECLARE synth_id UUID;
            BEGIN
                synth_id := create_semantic_memory(
                    p_params->>'content',
                    COALESCE((p_params->>'confidence')::float, 0.8),
                    ARRAY['synthesis', COALESCE(p_params->>'topic', 'general')],
                    NULL,
                    jsonb_build_object('heartbeat_id', p_heartbeat_id, 'sources', p_params->'sources', 'boundaries', boundary_hits),
                    0.7
                );
                result := jsonb_build_object('synthesis_memory_id', synth_id, 'boundaries', boundary_hits);
            END;

        WHEN 'reach_out_user' THEN
            queued_call := build_outbox_message(
                'user',
                jsonb_build_object(
                    'message', p_params->>'message',
                    'intent', p_params->>'intent',
                    'heartbeat_id', p_heartbeat_id,
                    'purpose_kind', p_params->>'purpose_kind',
                    'purpose_reference', p_params->>'purpose_reference',
                    'urgency', COALESCE(p_params->>'urgency', 'normal')
                )
            );
            outbox_messages := outbox_messages || jsonb_build_array(queued_call);
            result := jsonb_build_object('queued', true, 'outbox_message', queued_call);
            PERFORM satisfy_drive('connection', 0.3);

        WHEN 'reach_out_public' THEN
            queued_call := build_outbox_message(
                'public',
                jsonb_build_object(
                    'platform', p_params->>'platform',
                    'content', p_params->>'content',
                    'heartbeat_id', p_heartbeat_id,
                    'boundaries', boundary_hits,
                    'purpose_kind', p_params->>'purpose_kind',
                    'purpose_reference', p_params->>'purpose_reference',
                    'urgency', COALESCE(p_params->>'urgency', 'normal'),
                    'target_id', p_params->>'target_id',
                    'target_channel', p_params->>'platform',
                    'delivery_mode', 'direct'
                )
            );
            outbox_messages := outbox_messages || jsonb_build_array(queued_call);
            result := jsonb_build_object('queued', true, 'outbox_message', queued_call, 'boundaries', boundary_hits);
            PERFORM satisfy_drive('connection', 0.3);

        WHEN 'pause_heartbeat' THEN
            pause_reason := COALESCE(
                NULLIF(p_params->>'reason', ''),
                NULLIF(p_params->>'details', ''),
                NULLIF(p_params->>'message', '')
            );
            IF pause_reason IS NULL THEN
                RETURN jsonb_build_object('success', false, 'error', 'pause_heartbeat requires a reason');
            END IF;
            result := pause_heartbeat(pause_reason, p_params, p_heartbeat_id);
            outbox_messages := outbox_messages || COALESCE(result->'outbox_messages', '[]'::jsonb);

        WHEN 'terminate' THEN
            IF COALESCE(p_params->'confirmed', 'false'::jsonb) = 'true'::jsonb THEN
                result := terminate_agent(
                    COALESCE(NULLIF(p_params->>'last_will', ''), NULLIF(p_params->>'message', ''), NULLIF(p_params->>'reason', ''), ''),
                    COALESCE(p_params->'farewells', '[]'::jsonb),
                    COALESCE(p_params->'options', '{}'::jsonb)
                );
                outbox_messages := outbox_messages || COALESCE(result->'outbox_messages', '[]'::jsonb);
            ELSE
                queued_call := build_external_call(
                    'think',
                    jsonb_build_object(
                        'kind', 'termination_confirm',
                        'heartbeat_id', p_heartbeat_id,
                        'context', gather_turn_context(),
                        'params', COALESCE(p_params, '{}'::jsonb)
                    )
                );
                external_calls := external_calls || jsonb_build_array(queued_call);
                result := jsonb_build_object('confirmation_required', true, 'external_call', queued_call);
            END IF;

        WHEN 'rest' THEN
            result := jsonb_build_object('rested', true, 'energy_preserved', current_e - action_cost);
            PERFORM satisfy_drive('rest', 0.4);

        -- Memory retention: the conscious mind's verdict on a memory at the threshold
        -- of fading (surfaced via context.memories_at_threshold).
        WHEN 'keep_memory' THEN
            v_review_id := NULLIF(p_params->>'review_id', '')::uuid;
            SELECT memory_ids INTO v_review_ids
              FROM memory_review_queue WHERE id = v_review_id AND status = 'pending';
            IF v_review_ids IS NULL THEN
                result := jsonb_build_object('kept', false, 'reason', 'not_found');
            ELSIF NOT spend_retention_budget() THEN
                -- Out of points: a reminder that memory is finite. Cannot hold this one.
                result := jsonb_build_object('kept', false, 'reason', 'no_budget',
                                             'budget_remaining', retention_budget_remaining());
            ELSE
                UPDATE memories
                   SET metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object('protected', true)
                 WHERE id = ANY(v_review_ids);
                PERFORM touch_memories(v_review_ids);
                UPDATE memory_review_queue
                   SET status = 'kept', decision = 'keep', decided_at = CURRENT_TIMESTAMP
                 WHERE id = v_review_id;
                result := jsonb_build_object('kept', to_jsonb(v_review_ids),
                                             'budget_remaining', retention_budget_remaining());
            END IF;

        WHEN 'release_memory' THEN
            v_review_id := NULLIF(p_params->>'review_id', '')::uuid;
            UPDATE memory_review_queue
               SET status = 'released', decision = 'release', decided_at = CURRENT_TIMESTAMP
             WHERE id = v_review_id AND status = 'pending'
            RETURNING memory_ids INTO v_review_ids;
            IF v_review_ids IS NOT NULL THEN
                PERFORM consolidate_memory_group(v_review_ids);
            END IF;
            result := jsonb_build_object('released', COALESCE(to_jsonb(v_review_ids), '[]'::jsonb));

        WHEN 'journal_memory' THEN
            v_review_id := NULLIF(p_params->>'review_id', '')::uuid;
            SELECT memory_ids INTO v_review_ids
              FROM memory_review_queue WHERE id = v_review_id AND status = 'pending';
            IF v_review_ids IS NULL THEN
                result := jsonb_build_object('journaled', false, 'reason', 'not_found');
            ELSE
                PERFORM write_journal_entry(
                    p_content  := COALESCE(NULLIF(p_params->>'content', ''),
                                           'A memory I chose to keep in words before letting it fade.'),
                    p_title    := NULLIF(p_params->>'title', ''),
                    p_mood     := NULLIF(p_params->>'mood', ''),
                    p_metadata := jsonb_build_object('source', 'memory_review', 'review_id', v_review_id));
                UPDATE memory_review_queue
                   SET status = 'released', decision = 'journal', decided_at = CURRENT_TIMESTAMP
                 WHERE id = v_review_id;
                PERFORM consolidate_memory_group(v_review_ids);
                result := jsonb_build_object('journaled', true, 'faded', to_jsonb(v_review_ids));
            END IF;

        ELSE
            RETURN jsonb_build_object('success', false, 'error', 'Unknown action: ' || COALESCE(p_action, '<null>'));
    END CASE;

    RETURN jsonb_build_object(
        'success', true,
        'action', p_action,
        'cost', action_cost,
        'energy_remaining', get_current_energy(),
        'result', result,
        'external_calls', external_calls,
        'outbox_messages', outbox_messages
    );
END;
$$ LANGUAGE plpgsql;
