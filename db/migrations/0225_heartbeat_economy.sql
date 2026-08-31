-- Banked, outcome-sensitive heartbeat energy and adaptive cadence.
-- Re-establish start_heartbeat after 0214, which intentionally replaced it.
SET search_path = public, ag_catalog, "$user";

-- The heartbeat economy is auditable state, not a timer-side calculation.
-- One singleton anchors time-proportional regeneration; each beat and each
-- useful outcome signal remains inspectable after scheduling decisions.
CREATE TABLE IF NOT EXISTS heartbeat_economy_state (
    id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    last_regenerated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO heartbeat_economy_state (id, last_regenerated_at)
SELECT 1, COALESCE(
    (SELECT NULLIF(value->>'last_heartbeat_at', '')::timestamptz
     FROM state WHERE key = 'heartbeat_state'),
    CURRENT_TIMESTAMP
)
ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS heartbeat_outcomes (
    heartbeat_id UUID PRIMARY KEY,
    heartbeat_number INT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'error', 'cancelled')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    stopped_reason TEXT,
    energy_before FLOAT NOT NULL DEFAULT 0,
    elapsed_regen_hours FLOAT NOT NULL DEFAULT 0,
    surplus_decayed FLOAT NOT NULL DEFAULT 0,
    energy_regenerated FLOAT NOT NULL DEFAULT 0,
    regen_multiplier FLOAT NOT NULL DEFAULT 1,
    energy_after_regen FLOAT NOT NULL DEFAULT 0,
    energy_spent FLOAT NOT NULL DEFAULT 0,
    durable_memories_created INT NOT NULL DEFAULT 0,
    contradictions_resolved INT NOT NULL DEFAULT 0,
    goals_advanced INT NOT NULL DEFAULT 0,
    proactive_contact BOOLEAN NOT NULL DEFAULT FALSE,
    user_feedback_score FLOAT NOT NULL DEFAULT 0,
    outcome_score FLOAT NOT NULL DEFAULT 0,
    outcome_tier TEXT NOT NULL DEFAULT 'none'
        CHECK (outcome_tier IN ('none', 'useful', 'high_value')),
    urgency_ratio FLOAT NOT NULL DEFAULT 0,
    cadence_minutes FLOAT,
    next_heartbeat_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_heartbeat_outcomes_completed
    ON heartbeat_outcomes (completed_at DESC)
    WHERE completed_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS heartbeat_outcome_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    heartbeat_id UUID NOT NULL REFERENCES heartbeat_outcomes(heartbeat_id) ON DELETE CASCADE,
    signal_kind TEXT NOT NULL CHECK (signal_kind IN (
        'durable_memory', 'contradiction_resolved', 'goal_advanced',
        'proactive_contact', 'user_feedback', 'tool_success', 'tool_failure'
    )),
    signal_key TEXT NOT NULL,
    amount FLOAT NOT NULL DEFAULT 1,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (heartbeat_id, signal_key)
);
CREATE INDEX IF NOT EXISTS idx_heartbeat_outcome_signals_kind
    ON heartbeat_outcome_signals (heartbeat_id, signal_kind, created_at);


-- Banked, outcome-sensitive heartbeat energy and adaptive cadence.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

INSERT INTO config_defaults (key, value, description) VALUES
    ('heartbeat.energy_bank_multiplier', '3.0'::jsonb,
     'Hard bank capacity as a multiple of heartbeat.max_energy'),
    ('heartbeat.energy_surplus_half_life_hours', '12.0'::jsonb,
     'Half-life of energy stored above the normal heartbeat reserve'),
    ('heartbeat.outcome_regen_floor_multiplier', '0.75'::jsonb,
     'Regeneration multiplier after a beat with no durable outcome'),
    ('heartbeat.outcome_regen_score_scale', '0.5'::jsonb,
     'Regeneration multiplier added per point of prior outcome score'),
    ('heartbeat.outcome_regen_ceiling_multiplier', '1.5'::jsonb,
     'Maximum outcome-sensitive regeneration multiplier'),
    ('heartbeat.user_feedback_window_hours', '24'::jsonb,
     'Window in which verified operator thanks can credit a proactive heartbeat'),
    ('heartbeat.cadence_min_minutes', '15'::jsonb,
     'Shortest adaptive heartbeat cadence at high drive urgency'),
    ('heartbeat.cadence_max_minutes', '120'::jsonb,
     'Longest adaptive heartbeat cadence while drives are quiet'),
    ('heartbeat.cadence_idle_multiplier', '1.5'::jsonb,
     'Base-interval multiplier when drive urgency is zero'),
    ('heartbeat.cadence_urgency_slope', '0.75'::jsonb,
     'Cadence multiplier reduction per unit of maximum drive urgency')
ON CONFLICT (key) DO NOTHING;

UPDATE config_defaults
SET description = 'Base energy regenerated per elapsed hour'
WHERE key = 'heartbeat.base_regeneration';
UPDATE config_defaults
SET description = 'Normal heartbeat energy reserve'
WHERE key = 'heartbeat.max_energy';

CREATE OR REPLACE FUNCTION heartbeat_bank_capacity()
RETURNS FLOAT
LANGUAGE sql
STABLE
AS $$
    SELECT GREATEST(COALESCE(get_config_float('heartbeat.max_energy'), 20.0), 1.0)
         * GREATEST(COALESCE(get_config_float('heartbeat.energy_bank_multiplier'), 3.0), 1.0)
$$;

CREATE OR REPLACE FUNCTION heartbeat_outcome_regen_multiplier()
RETURNS FLOAT
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    latest_score FLOAT;
    floor_multiplier FLOAT := GREATEST(
        COALESCE(get_config_float('heartbeat.outcome_regen_floor_multiplier'), 0.75),
        0.0
    );
    score_scale FLOAT := GREATEST(
        COALESCE(get_config_float('heartbeat.outcome_regen_score_scale'), 0.5),
        0.0
    );
    ceiling_multiplier FLOAT := GREATEST(
        COALESCE(get_config_float('heartbeat.outcome_regen_ceiling_multiplier'), 1.5),
        floor_multiplier
    );
BEGIN
    SELECT outcome_score INTO latest_score
    FROM heartbeat_outcomes
    WHERE status IN ('completed', 'error')
    ORDER BY completed_at DESC
    LIMIT 1;
    IF latest_score IS NULL THEN
        RETURN 1.0;
    END IF;
    RETURN LEAST(ceiling_multiplier, floor_multiplier + GREATEST(latest_score, 0.0) * score_scale);
END;
$$;

CREATE OR REPLACE FUNCTION regenerate_heartbeat_energy(
    p_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    economy heartbeat_economy_state%ROWTYPE;
    before_energy FLOAT;
    reserve_energy FLOAT := GREATEST(
        COALESCE(get_config_float('heartbeat.max_energy'), 20.0), 1.0
    );
    bank_capacity FLOAT := heartbeat_bank_capacity();
    base_regen FLOAT := GREATEST(
        COALESCE(get_config_float('heartbeat.base_regeneration'), 10.0), 0.0
    );
    half_life FLOAT := GREATEST(
        COALESCE(get_config_float('heartbeat.energy_surplus_half_life_hours'), 12.0),
        0.25
    );
    elapsed_hours FLOAT;
    surplus_before FLOAT;
    surplus_after_decay FLOAT;
    decayed_amount FLOAT;
    multiplier FLOAT := heartbeat_outcome_regen_multiplier();
    regeneration_available FLOAT;
    regenerated FLOAT;
    energy_after_decay FLOAT;
    after_energy FLOAT;
BEGIN
    INSERT INTO heartbeat_economy_state (id, last_regenerated_at)
    VALUES (1, COALESCE(p_at, CURRENT_TIMESTAMP))
    ON CONFLICT (id) DO NOTHING;
    SELECT * INTO economy FROM heartbeat_economy_state WHERE id = 1 FOR UPDATE;
    SELECT COALESCE(current_energy, 0.0) INTO before_energy
    FROM heartbeat_state WHERE id = 1;

    elapsed_hours := GREATEST(
        EXTRACT(EPOCH FROM (COALESCE(p_at, CURRENT_TIMESTAMP) - economy.last_regenerated_at)) / 3600.0,
        0.0
    );
    before_energy := LEAST(GREATEST(before_energy, 0.0), bank_capacity);
    surplus_before := GREATEST(before_energy - reserve_energy, 0.0);
    surplus_after_decay := surplus_before
        * power(0.5::double precision, elapsed_hours / half_life);
    decayed_amount := GREATEST(surplus_before - surplus_after_decay, 0.0);
    energy_after_decay := LEAST(
        bank_capacity,
        GREATEST(0.0, LEAST(before_energy, reserve_energy) + surplus_after_decay)
    );
    regeneration_available := base_regen * elapsed_hours * multiplier;
    after_energy := LEAST(bank_capacity, energy_after_decay + regeneration_available);
    regenerated := GREATEST(after_energy - energy_after_decay, 0.0);

    UPDATE heartbeat_state
    SET current_energy = after_energy, updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;
    UPDATE heartbeat_economy_state
    SET last_regenerated_at = COALESCE(p_at, CURRENT_TIMESTAMP),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;

    RETURN jsonb_build_object(
        'before_energy', before_energy,
        'reserve_energy', reserve_energy,
        'bank_capacity', bank_capacity,
        'elapsed_hours', elapsed_hours,
        'surplus_decayed', decayed_amount,
        'regen_multiplier', multiplier,
        'regeneration_available', regeneration_available,
        'energy_regenerated', regenerated,
        'after_energy', after_energy
    );
END;
$$;

CREATE OR REPLACE FUNCTION heartbeat_urgency_snapshot()
RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    WITH ranked AS (
        SELECT name, current_level, urgency_threshold,
               GREATEST(current_level / NULLIF(urgency_threshold, 0), 0.0) AS ratio
        FROM drives
    )
    SELECT jsonb_build_object(
        'max_urgency_ratio', COALESCE(MAX(ratio), 0.0),
        'urgent_drives', COALESCE(
            jsonb_agg(
                jsonb_build_object(
                    'name', name,
                    'level', current_level,
                    'urgency_threshold', urgency_threshold,
                    'urgency_ratio', ratio
                ) ORDER BY ratio DESC
            ) FILTER (WHERE ratio >= 0.8),
            '[]'::jsonb
        )
    )
    FROM ranked
$$;

CREATE OR REPLACE FUNCTION heartbeat_adaptive_cadence(
    p_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
) RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    urgency JSONB := heartbeat_urgency_snapshot();
    urgency_ratio FLOAT := GREATEST(
        COALESCE(NULLIF(urgency->>'max_urgency_ratio', '')::float, 0.0), 0.0
    );
    base_minutes FLOAT := GREATEST(
        COALESCE(get_config_float('heartbeat.heartbeat_interval_minutes'), 60.0), 0.0
    );
    min_minutes FLOAT;
    max_minutes FLOAT;
    idle_multiplier FLOAT := GREATEST(
        COALESCE(get_config_float('heartbeat.cadence_idle_multiplier'), 1.5), 0.0
    );
    urgency_slope FLOAT := GREATEST(
        COALESCE(get_config_float('heartbeat.cadence_urgency_slope'), 0.75), 0.0
    );
    raw_minutes FLOAT;
    cadence_minutes FLOAT;
BEGIN
    IF base_minutes <= 0 THEN
        RETURN urgency || jsonb_build_object(
            'base_minutes', base_minutes,
            'cadence_minutes', 0.0,
            'next_heartbeat_at', COALESCE(p_at, CURRENT_TIMESTAMP)
        );
    END IF;
    min_minutes := LEAST(
        base_minutes,
        GREATEST(COALESCE(get_config_float('heartbeat.cadence_min_minutes'), 15.0), 1.0)
    );
    max_minutes := GREATEST(
        base_minutes,
        COALESCE(get_config_float('heartbeat.cadence_max_minutes'), 120.0),
        min_minutes
    );
    raw_minutes := base_minutes * GREATEST(
        min_minutes / base_minutes,
        idle_multiplier - LEAST(urgency_ratio, 2.0) * urgency_slope
    );
    cadence_minutes := LEAST(max_minutes, GREATEST(min_minutes, raw_minutes));
    RETURN urgency || jsonb_build_object(
        'base_minutes', base_minutes,
        'cadence_minutes', cadence_minutes,
        'next_heartbeat_at', COALESCE(p_at, CURRENT_TIMESTAMP)
            + make_interval(secs => cadence_minutes * 60.0)
    );
END;
$$;

CREATE OR REPLACE FUNCTION begin_heartbeat_outcome(
    p_heartbeat_id UUID,
    p_heartbeat_number INT,
    p_regeneration JSONB
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    urgency JSONB := heartbeat_urgency_snapshot();
BEGIN
    INSERT INTO heartbeat_outcomes (
        heartbeat_id, heartbeat_number, energy_before, elapsed_regen_hours,
        surplus_decayed, energy_regenerated, regen_multiplier,
        energy_after_regen, urgency_ratio, metadata
    ) VALUES (
        p_heartbeat_id,
        p_heartbeat_number,
        COALESCE(NULLIF(p_regeneration->>'before_energy', '')::float, 0.0),
        COALESCE(NULLIF(p_regeneration->>'elapsed_hours', '')::float, 0.0),
        COALESCE(NULLIF(p_regeneration->>'surplus_decayed', '')::float, 0.0),
        COALESCE(NULLIF(p_regeneration->>'energy_regenerated', '')::float, 0.0),
        COALESCE(NULLIF(p_regeneration->>'regen_multiplier', '')::float, 1.0),
        COALESCE(NULLIF(p_regeneration->>'after_energy', '')::float, 0.0),
        COALESCE(NULLIF(urgency->>'max_urgency_ratio', '')::float, 0.0),
        jsonb_build_object(
            'urgency_at_start', urgency,
            'regeneration', COALESCE(p_regeneration, '{}'::jsonb)
        )
    )
    ON CONFLICT (heartbeat_id) DO NOTHING;
    RETURN jsonb_build_object(
        'heartbeat_id', p_heartbeat_id,
        'regeneration', p_regeneration,
        'urgency', urgency
    );
END;
$$;

CREATE OR REPLACE FUNCTION record_heartbeat_outcome_signal(
    p_heartbeat_id UUID,
    p_signal_kind TEXT,
    p_signal_key TEXT,
    p_amount FLOAT DEFAULT 1.0,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE inserted_count INT;
BEGIN
    IF p_signal_kind NOT IN (
        'durable_memory', 'contradiction_resolved', 'goal_advanced',
        'proactive_contact', 'user_feedback', 'tool_success', 'tool_failure'
    ) THEN
        RAISE EXCEPTION 'unsupported heartbeat outcome signal: %', p_signal_kind;
    END IF;
    INSERT INTO heartbeat_outcome_signals (
        heartbeat_id, signal_kind, signal_key, amount, metadata
    ) VALUES (
        p_heartbeat_id,
        p_signal_kind,
        COALESCE(NULLIF(p_signal_key, ''), p_signal_kind || ':' || gen_random_uuid()::text),
        GREATEST(COALESCE(p_amount, 0.0), 0.0),
        COALESCE(p_metadata, '{}'::jsonb)
    )
    ON CONFLICT (heartbeat_id, signal_key) DO NOTHING;
    GET DIAGNOSTICS inserted_count = ROW_COUNT;
    RETURN inserted_count > 0;
EXCEPTION WHEN foreign_key_violation THEN
    RETURN FALSE;
END;
$$;

CREATE OR REPLACE FUNCTION record_heartbeat_tool_outcome(
    p_heartbeat_id UUID,
    p_turn_id UUID,
    p_tool_call_id TEXT,
    p_result JSONB
) RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    tool_name TEXT := COALESCE(p_result->>'tool_name', 'unknown');
    succeeded BOOLEAN := COALESCE(NULLIF(p_result->>'success', '')::boolean, FALSE);
    signal_prefix TEXT := 'tool:' || p_turn_id::text || ':' || COALESCE(p_tool_call_id, 'unknown');
    output JSONB := COALESCE(p_result->'output', '{}'::jsonb);
    arguments JSONB := COALESCE(p_result->'arguments', '{}'::jsonb);
BEGIN
    PERFORM record_heartbeat_outcome_signal(
        p_heartbeat_id,
        CASE WHEN succeeded THEN 'tool_success' ELSE 'tool_failure' END,
        signal_prefix || ':completion',
        1.0,
        jsonb_build_object('tool_name', tool_name)
    );
    IF NOT succeeded THEN
        RETURN;
    END IF;
    IF NULLIF(output->>'memory_id', '') IS NOT NULL
       AND NOT COALESCE(NULLIF(output->>'reused', '')::boolean, FALSE) THEN
        PERFORM record_heartbeat_outcome_signal(
            p_heartbeat_id, 'durable_memory', signal_prefix || ':memory', 1.0,
            jsonb_build_object('tool_name', tool_name, 'memory_id', output->>'memory_id')
        );
    END IF;
    IF COALESCE(NULLIF(output->>'resolved', '')::boolean, FALSE) THEN
        PERFORM record_heartbeat_outcome_signal(
            p_heartbeat_id, 'contradiction_resolved', signal_prefix || ':contradiction', 1.0,
            jsonb_build_object('tool_name', tool_name)
        );
    END IF;
    IF tool_name = 'manage_goals'
       AND (
           arguments->>'action' = 'add_progress'
           OR (arguments->>'action' = 'update_priority' AND arguments->>'priority' = 'completed')
       ) THEN
        PERFORM record_heartbeat_outcome_signal(
            p_heartbeat_id, 'goal_advanced', signal_prefix || ':goal', 1.0,
            jsonb_build_object('tool_name', tool_name, 'goal_id', output->>'goal_id')
        );
    END IF;
    IF tool_name = 'queue_user_message' THEN
        PERFORM record_heartbeat_outcome_signal(
            p_heartbeat_id, 'proactive_contact', signal_prefix || ':contact', 1.0,
            jsonb_build_object('tool_name', tool_name)
        );
    END IF;
END;
$$;

CREATE OR REPLACE FUNCTION enforce_heartbeat_plan_energy(
    p_plan JSONB
) RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    plan JSONB := COALESCE(p_plan, '{}'::jsonb);
    available FLOAT := GREATEST(
        COALESCE(NULLIF(plan#>>'{context,energy,current}', '')::float, 0.0), 0.0
    );
    reserve FLOAT := GREATEST(
        COALESCE(NULLIF(plan#>>'{context,energy,max}', '')::float,
                 get_config_float('heartbeat.max_energy'), 20.0),
        1.0
    );
    has_tasks BOOLEAN := COALESCE(NULLIF(plan->>'has_backlog_tasks', '')::boolean, FALSE);
    task_multiplier FLOAT := GREATEST(
        COALESCE(get_config_float('heartbeat.task_energy_multiplier'), 2.0), 1.0
    );
    budget FLOAT;
BEGIN
    -- Ordinary beats spend only the reserve. Actionable backlog may draw from
    -- the bank, but never more energy than is actually present.
    budget := LEAST(
        available,
        reserve * CASE WHEN has_tasks THEN task_multiplier ELSE 1.0 END
    );
    RETURN jsonb_set(plan, '{energy_budget}', to_jsonb(GREATEST(budget, 0.0)), TRUE)
        || jsonb_build_object(
            'energy_available', available,
            'energy_reserve', reserve,
            'energy_bank_capacity', heartbeat_bank_capacity()
        );
END;
$$;

CREATE OR REPLACE FUNCTION refresh_heartbeat_outcome(
    p_heartbeat_id UUID
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    durable_count INT;
    contradiction_count INT;
    goal_count INT;
    contact BOOLEAN;
    feedback FLOAT;
    score FLOAT;
    tier TEXT;
    outcome heartbeat_outcomes%ROWTYPE;
BEGIN
    SELECT
        COUNT(*) FILTER (WHERE signal_kind = 'durable_memory'),
        COUNT(*) FILTER (WHERE signal_kind = 'contradiction_resolved'),
        COUNT(*) FILTER (WHERE signal_kind = 'goal_advanced'),
        COUNT(*) FILTER (WHERE signal_kind = 'proactive_contact') > 0,
        COALESCE(SUM(amount) FILTER (WHERE signal_kind = 'user_feedback'), 0.0)
    INTO durable_count, contradiction_count, goal_count, contact, feedback
    FROM heartbeat_outcome_signals
    WHERE heartbeat_id = p_heartbeat_id;

    score := LEAST(durable_count, 2) * 0.35
        + LEAST(contradiction_count, 2) * 0.5
        + LEAST(goal_count, 2) * 0.3
        + LEAST(feedback, 1.0) * 0.5;
    tier := CASE
        WHEN score >= 1.0 THEN 'high_value'
        WHEN score > 0 THEN 'useful'
        ELSE 'none'
    END;
    UPDATE heartbeat_outcomes
    SET durable_memories_created = durable_count,
        contradictions_resolved = contradiction_count,
        goals_advanced = goal_count,
        proactive_contact = contact,
        user_feedback_score = feedback,
        outcome_score = score,
        outcome_tier = tier
    WHERE heartbeat_id = p_heartbeat_id
    RETURNING * INTO outcome;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('updated', FALSE, 'reason', 'unknown_heartbeat');
    END IF;
    RETURN to_jsonb(outcome) || jsonb_build_object('updated', TRUE);
END;
$$;

CREATE OR REPLACE FUNCTION finalize_heartbeat_economy(
    p_heartbeat_id UUID,
    p_energy_spent FLOAT DEFAULT 0.0,
    p_stopped_reason TEXT DEFAULT 'completed',
    p_legacy_actions JSONB DEFAULT '[]'::jsonb,
    p_goal_changes JSONB DEFAULT '[]'::jsonb,
    p_allow_released_claim BOOLEAN DEFAULT FALSE
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    outcome heartbeat_outcomes%ROWTYPE;
    active_id UUID;
    action_item JSONB;
    goal_item JSONB;
    action_name TEXT;
    action_success BOOLEAN;
    v_ordinality BIGINT;
    tool_event RECORD;
    cadence JSONB;
    refreshed JSONB;
    v_completed_at TIMESTAMPTZ := CURRENT_TIMESTAMP;
    spent FLOAT := GREATEST(COALESCE(p_energy_spent, 0.0), 0.0);
BEGIN
    SELECT * INTO outcome
    FROM heartbeat_outcomes
    WHERE heartbeat_id = p_heartbeat_id
    FOR UPDATE;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('applied', FALSE, 'reason', 'unknown_heartbeat');
    END IF;
    IF outcome.status <> 'running' THEN
        RETURN to_jsonb(outcome) || jsonb_build_object('applied', FALSE, 'reason', 'already_finalized');
    END IF;
    SELECT active_heartbeat_id INTO active_id FROM heartbeat_state WHERE id = 1;
    IF active_id IS DISTINCT FROM p_heartbeat_id
       AND NOT COALESCE(p_allow_released_claim, FALSE) THEN
        RETURN jsonb_build_object('applied', FALSE, 'reason', 'heartbeat_claim_mismatch');
    END IF;

    -- Agentic turns already contain exact, durable tool receipts. Derive
    -- outcome signals from those receipts instead of trusting a summary.
    FOR tool_event IN
        SELECT e.id, t.id AS turn_id, e.payload
        FROM agent_turns t
        JOIN agent_turn_events e ON e.turn_id = t.id
        WHERE t.heartbeat_id = p_heartbeat_id
          AND e.event_type = 'tool_result'
        ORDER BY e.created_at, e.id
    LOOP
        PERFORM record_heartbeat_tool_outcome(
            p_heartbeat_id,
            tool_event.turn_id,
            tool_event.id::text,
            tool_event.payload
        );
    END LOOP;

    IF jsonb_typeof(COALESCE(p_legacy_actions, '[]'::jsonb)) = 'array' THEN
        FOR action_item, v_ordinality IN
            SELECT item.value, item.ordinality
            FROM jsonb_array_elements(COALESCE(p_legacy_actions, '[]'::jsonb))
                WITH ORDINALITY AS item(value, ordinality)
        LOOP
            action_name := COALESCE(action_item->>'action', 'unknown');
            action_success := COALESCE(
                NULLIF(action_item#>>'{result,success}', '')::boolean,
                TRUE
            );
            PERFORM record_heartbeat_outcome_signal(
                p_heartbeat_id,
                CASE WHEN action_success THEN 'tool_success' ELSE 'tool_failure' END,
                'legacy:' || v_ordinality || ':completion',
                1.0,
                jsonb_build_object('action', action_name)
            );
            IF action_success AND action_name IN (
                'remember', 'contemplate', 'meditate', 'study',
                'debate_internally', 'mark_turning_point', 'close_chapter',
                'resolve_contradiction', 'accept_tension', 'synthesize',
                'journal_memory'
            ) THEN
                PERFORM record_heartbeat_outcome_signal(
                    p_heartbeat_id, 'durable_memory',
                    'legacy:' || v_ordinality || ':memory', 1.0,
                    jsonb_build_object('action', action_name)
                );
            END IF;
            IF action_success AND action_name = 'resolve_contradiction' THEN
                PERFORM record_heartbeat_outcome_signal(
                    p_heartbeat_id, 'contradiction_resolved',
                    'legacy:' || v_ordinality || ':contradiction', 1.0,
                    jsonb_build_object('action', action_name)
                );
            END IF;
            IF action_success AND action_name = 'reprioritize'
               AND action_item#>>'{params,new_priority}' = 'completed' THEN
                PERFORM record_heartbeat_outcome_signal(
                    p_heartbeat_id, 'goal_advanced',
                    'legacy:' || v_ordinality || ':goal', 1.0,
                    jsonb_build_object('action', action_name)
                );
            END IF;
            IF action_success AND action_name = 'reach_out_user' THEN
                PERFORM record_heartbeat_outcome_signal(
                    p_heartbeat_id, 'proactive_contact',
                    'legacy:' || v_ordinality || ':contact', 1.0,
                    jsonb_build_object('action', action_name)
                );
            END IF;
        END LOOP;
    END IF;
    IF jsonb_typeof(COALESCE(p_goal_changes, '[]'::jsonb)) = 'array' THEN
        FOR goal_item, v_ordinality IN
            SELECT item.value, item.ordinality
            FROM jsonb_array_elements(COALESCE(p_goal_changes, '[]'::jsonb))
                WITH ORDINALITY AS item(value, ordinality)
        LOOP
            IF COALESCE(goal_item->>'new_priority', goal_item->>'change', goal_item->>'priority') = 'completed' THEN
                PERFORM record_heartbeat_outcome_signal(
                    p_heartbeat_id, 'goal_advanced',
                    'goal-change:' || v_ordinality, 1.0,
                    jsonb_build_object('goal_id', goal_item->>'goal_id')
                );
            END IF;
        END LOOP;
    END IF;

    IF spent > 0 THEN
        PERFORM update_energy(-spent);
    END IF;
    refreshed := refresh_heartbeat_outcome(p_heartbeat_id);
    cadence := heartbeat_adaptive_cadence(v_completed_at);
    UPDATE heartbeat_outcomes
    SET status = CASE WHEN COALESCE(p_stopped_reason, 'completed') = 'error'
                      THEN 'error' ELSE 'completed' END,
        completed_at = v_completed_at,
        stopped_reason = COALESCE(p_stopped_reason, 'completed'),
        energy_spent = spent,
        urgency_ratio = COALESCE(NULLIF(cadence->>'max_urgency_ratio', '')::float, 0.0),
        cadence_minutes = COALESCE(NULLIF(cadence->>'cadence_minutes', '')::float, 0.0),
        next_heartbeat_at = (cadence->>'next_heartbeat_at')::timestamptz,
        metadata = metadata || jsonb_build_object('urgency_at_completion', cadence)
    WHERE heartbeat_id = p_heartbeat_id;
    UPDATE heartbeat_state
    SET next_heartbeat_at = (cadence->>'next_heartbeat_at')::timestamptz,
        last_heartbeat_at = v_completed_at,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1
      AND (
          active_heartbeat_id = p_heartbeat_id
          OR COALESCE(p_allow_released_claim, FALSE)
      );

    RETURN COALESCE(
        (SELECT to_jsonb(o) FROM heartbeat_outcomes o WHERE o.heartbeat_id = p_heartbeat_id),
        refreshed
    ) || jsonb_build_object('applied', TRUE);
END;
$$;

-- Compatibility bridge for the legacy JSON heartbeat path and guarded error
-- releases. The canonical heartbeat_state view writes through the state table;
-- when an active claim transitions to NULL, finalize any still-running outcome.
CREATE OR REPLACE FUNCTION heartbeat_economy_state_transition_trigger()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    released_id UUID;
    actions JSONB;
BEGIN
    IF NEW.key <> 'heartbeat_state' THEN
        RETURN NEW;
    END IF;
    released_id := _db_brain_try_uuid(OLD.value->>'active_heartbeat_id');
    IF released_id IS NULL
       OR _db_brain_try_uuid(NEW.value->>'active_heartbeat_id') IS NOT NULL THEN
        RETURN NEW;
    END IF;
    actions := COALESCE(OLD.value->'active_actions', '[]'::jsonb);
    PERFORM finalize_heartbeat_economy(
        released_id,
        0.0,
        'completed',
        actions,
        '[]'::jsonb,
        TRUE
    );
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION credit_heartbeat_user_feedback(
    p_message TEXT,
    p_is_operator BOOLEAN,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    target heartbeat_outcomes%ROWTYPE;
    feedback_window FLOAT := GREATEST(
        COALESCE(get_config_float('heartbeat.user_feedback_window_hours'), 24.0),
        0.0
    );
    signal_key TEXT;
    inserted BOOLEAN;
BEGIN
    IF NOT COALESCE(p_is_operator, FALSE) THEN
        RETURN jsonb_build_object('credited', FALSE, 'reason', 'not_verified_operator');
    END IF;
    IF lower(COALESCE(p_message, '')) !~
       '(^|[^[:alnum:]])(thank you|thanks|thx|appreciate (it|that|this|you))([^[:alnum:]]|$)' THEN
        RETURN jsonb_build_object('credited', FALSE, 'reason', 'no_explicit_appreciation');
    END IF;
    SELECT * INTO target
    FROM heartbeat_outcomes o
    WHERE o.status = 'completed'
      AND o.proactive_contact
      AND o.completed_at >= CURRENT_TIMESTAMP - make_interval(secs => feedback_window * 3600.0)
    ORDER BY o.completed_at DESC
    LIMIT 1;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('credited', FALSE, 'reason', 'no_recent_proactive_heartbeat');
    END IF;
    IF EXISTS (
        SELECT 1 FROM heartbeat_outcome_signals s
        WHERE s.heartbeat_id = target.heartbeat_id
          AND s.signal_kind = 'user_feedback'
    ) THEN
        RETURN jsonb_build_object(
            'credited', FALSE,
            'heartbeat_id', target.heartbeat_id,
            'reason', 'duplicate_feedback'
        );
    END IF;
    signal_key := 'user-feedback';
    inserted := record_heartbeat_outcome_signal(
        target.heartbeat_id, 'user_feedback', signal_key, 1.0,
        jsonb_strip_nulls(jsonb_build_object(
            'surface', p_metadata->>'surface',
            'session_id', p_metadata->>'session_id',
            'agent_turn_id', p_metadata->>'agent_turn_id'
        ))
    );
    IF inserted THEN
        PERFORM refresh_heartbeat_outcome(target.heartbeat_id);
    END IF;
    RETURN jsonb_build_object(
        'credited', inserted,
        'heartbeat_id', target.heartbeat_id,
        'reason', CASE WHEN inserted THEN 'explicit_operator_appreciation' ELSE 'duplicate_feedback' END
    );
END;
$$;

CREATE OR REPLACE FUNCTION heartbeat_economy_status()
RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    SELECT jsonb_build_object(
        'current_energy', h.current_energy,
        'reserve_energy', COALESCE(get_config_float('heartbeat.max_energy'), 20.0),
        'bank_capacity', heartbeat_bank_capacity(),
        'last_regenerated_at', e.last_regenerated_at,
        'next_heartbeat_at', h.next_heartbeat_at,
        'next_regen_multiplier', heartbeat_outcome_regen_multiplier(),
        'latest_outcome', (
            SELECT to_jsonb(o) FROM heartbeat_outcomes o
            WHERE o.status IN ('completed', 'error')
            ORDER BY o.completed_at DESC NULLS LAST LIMIT 1
        )
    )
    FROM heartbeat_state h
    CROSS JOIN heartbeat_economy_state e
    WHERE h.id = 1 AND e.id = 1
$$;

SET check_function_bodies = on;

CREATE OR REPLACE FUNCTION update_energy(p_delta FLOAT)
RETURNS FLOAT AS $$
DECLARE
    max_e FLOAT;
    new_e FLOAT;
BEGIN
    max_e := heartbeat_bank_capacity();

    UPDATE heartbeat_state
    SET current_energy = GREATEST(0, LEAST(current_energy + p_delta, max_e)),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1
    RETURNING current_energy INTO new_e;

    RETURN new_e;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION should_run_heartbeat()
RETURNS BOOLEAN AS $$
DECLARE
    state_record RECORD;
    interval_minutes FLOAT;
BEGIN
    IF is_agent_terminated() THEN
        RETURN FALSE;
    END IF;
    IF NOT is_agent_configured() THEN
        RETURN FALSE;
    END IF;
    IF NOT is_init_complete() THEN
        RETURN FALSE;
    END IF;

    SELECT * INTO state_record FROM heartbeat_state WHERE id = 1;
    IF state_record.is_paused THEN
        RETURN FALSE;
    END IF;
    IF state_record.active_heartbeat_id IS NOT NULL THEN
        RETURN FALSE;
    END IF;
    IF state_record.last_heartbeat_at IS NULL THEN
        RETURN TRUE;
    END IF;
    IF state_record.next_heartbeat_at IS NOT NULL THEN
        RETURN CURRENT_TIMESTAMP >= state_record.next_heartbeat_at;
    END IF;
    interval_minutes := get_config_float('heartbeat.heartbeat_interval_minutes');

    RETURN CURRENT_TIMESTAMP >= state_record.last_heartbeat_at + (interval_minutes || ' minutes')::INTERVAL;
END;
$$ LANGUAGE plpgsql;

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

DROP TRIGGER IF EXISTS trg_heartbeat_economy_state_transition ON state;
CREATE TRIGGER trg_heartbeat_economy_state_transition
AFTER UPDATE ON state
FOR EACH ROW
WHEN (OLD.key = 'heartbeat_state' AND NEW.key = 'heartbeat_state')
EXECUTE FUNCTION heartbeat_economy_state_transition_trigger();
