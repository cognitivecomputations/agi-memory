-- DB-owned self-repair substrate: preserve heartbeat/tool failure evidence,
-- classify defects, and surface bounded repair proposals in continuity.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

CREATE TABLE IF NOT EXISTS defect_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fingerprint TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'diagnosed', 'repair_proposed', 'verified', 'resolved', 'ignored')),
    severity TEXT NOT NULL DEFAULT 'medium'
        CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    category TEXT NOT NULL DEFAULT 'execution_failure',
    source TEXT NOT NULL DEFAULT 'unknown',
    component TEXT,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    last_error TEXT,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    occurrence_count INT NOT NULL DEFAULT 1 CHECK (occurrence_count > 0),
    heartbeat_ids UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    tool_names TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    diagnosis JSONB NOT NULL DEFAULT '{}'::jsonb,
    proposed_repair JSONB NOT NULL DEFAULT '{}'::jsonb,
    verification JSONB NOT NULL DEFAULT '{}'::jsonb,
    resolution TEXT,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_defect_reports_status_seen
    ON defect_reports(status, last_seen_at DESC);
CREATE INDEX IF NOT EXISTS idx_defect_reports_category
    ON defect_reports(category);

INSERT INTO config_defaults (key, value, description) VALUES
    ('chat.continuity_defect_limit', '5'::jsonb,
     'Number of unresolved self-observed software defects rendered in the chat continuity packet')
ON CONFLICT (key) DO NOTHING;

UPDATE config_defaults
SET value = value || '["self_repair"]'::jsonb
WHERE key = 'agent.tools'
  AND jsonb_typeof(value) = 'array'
  AND NOT (value ? 'self_repair');

UPDATE config
SET value = value || '["self_repair"]'::jsonb
WHERE key = 'agent.tools'
  AND jsonb_typeof(value) = 'array'
  AND NOT (value ? 'self_repair');

CREATE OR REPLACE FUNCTION normalize_defect_error(p_error TEXT)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT left(
        regexp_replace(
            regexp_replace(
                regexp_replace(lower(COALESCE(p_error, '')),
                    '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
                    '<uuid>', 'gi'),
                '[0-9]+', '<n>', 'g'),
            '\s+', ' ', 'g'),
        500)
$$;

CREATE OR REPLACE FUNCTION classify_defect_event(
    p_component TEXT,
    p_error TEXT,
    p_context JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    component TEXT := COALESCE(NULLIF(btrim(p_component), ''), 'unknown');
    error_text TEXT := lower(COALESCE(p_error, ''));
    category TEXT := 'execution_failure';
    severity TEXT := 'medium';
    title TEXT;
    summary TEXT;
BEGIN
    IF error_text LIKE '%unknown tool:%'
       OR error_text LIKE '%unknown action:%'
       OR error_text LIKE '%validation errors:%'
       OR error_text LIKE '%missing required field:%'
       OR error_text LIKE '%not allowed in % context%' THEN
        category := 'tool_contract';
        severity := 'medium';
        title := 'Tool/action contract failure: ' || component;
        summary := 'A tool, heartbeat action, or argument schema did not match the executor contract.';
    ELSIF error_text LIKE '%embedding service%'
       OR error_text LIKE '%connection refused%'
       OR error_text LIKE '%failed to connect%'
       OR error_text LIKE '%not reachable%' THEN
        category := 'dependency_unavailable';
        severity := 'high';
        title := 'Dependency unavailable: ' || component;
        summary := 'A required local service or dependency was unavailable when the agent tried to use it.';
    ELSIF error_text LIKE '%not configured%'
       OR error_text LIKE '%missing api key%'
       OR error_text LIKE '%missing config%'
       OR error_text LIKE '%credentials%' THEN
        category := 'configuration';
        severity := 'low';
        title := 'Configuration needed: ' || component;
        summary := 'The operation needs user/provider configuration rather than code repair.';
    ELSIF error_text LIKE '%timed out%'
       OR error_text LIKE '%timeout%' THEN
        category := 'timeout';
        severity := 'medium';
        title := 'Timeout: ' || component;
        summary := 'The operation exceeded its execution window and needs retry/backoff or workload reduction.';
    ELSIF error_text LIKE '%network error%'
       OR error_text LIKE '%http error%'
       OR error_text LIKE '%rate limit%' THEN
        category := 'network_or_provider';
        severity := 'medium';
        title := 'Provider/network failure: ' || component;
        summary := 'The operation failed outside the local code path and may need retry or provider-specific handling.';
    ELSE
        title := 'Execution failure: ' || component;
        summary := 'The agent observed a failed operation that needs inspection before repair.';
    END IF;

    RETURN jsonb_build_object(
        'category', category,
        'severity', severity,
        'title', title,
        'summary', summary
    );
END;
$$;

CREATE OR REPLACE FUNCTION render_chat_continuity_context(
    p_session_id TEXT DEFAULT NULL,
    p_exclude_sensitive BOOLEAN DEFAULT FALSE
) RETURNS TEXT
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    current_session UUID := _db_brain_try_uuid(p_session_id);
    lim INT := LEAST(GREATEST(COALESCE(get_config_int('chat.recent_carryover_limit'), 8), 0), 20);
    window_minutes INT := LEAST(GREATEST(COALESCE(get_config_int('chat.recent_carryover_window_minutes'), 1440), 1), 43200);
    max_chars INT := LEAST(GREATEST(COALESCE(get_config_int('chat.recent_carryover_max_chars'), 5000), 500), 20000);
    summary_lim INT := LEAST(GREATEST(COALESCE(get_config_int('chat.continuity_summary_limit'), 3), 0), 8);
    correction_lim INT := LEAST(GREATEST(COALESCE(get_config_int('chat.continuity_correction_limit'), 5), 0), 12);
    heartbeat_lim INT := LEAST(GREATEST(COALESCE(get_config_int('chat.continuity_heartbeat_limit'), 3), 0), 10);
    defect_lim INT := LEAST(GREATEST(COALESCE(get_config_int('chat.continuity_defect_limit'), 5), 0), 20);
    injury_lines TEXT;
    affect_line TEXT;
    affect_state JSONB;
    summary_lines TEXT;
    correction_lines TEXT;
    heartbeat_lines TEXT;
    defect_lines TEXT;
    turn_lines TEXT;
    body TEXT;
BEGIN
    IF p_exclude_sensitive THEN
        RETURN '';
    END IF;

    affect_state := get_current_affective_state();
    affect_line := '- Current affect: '
        || COALESCE(NULLIF(affect_state->>'primary_emotion', ''), NULLIF(affect_state->>'feeling', ''), 'unknown')
        || ', valence=' || COALESCE(affect_state->>'valence', '?')
        || ', arousal=' || COALESCE(affect_state->>'arousal', '?')
        || ', intensity=' || COALESCE(affect_state->>'intensity', '?');

    SELECT string_agg(
        '- ' || m.content
        || ' [unresolved; last evidence '
        || COALESCE(to_char((m.metadata#>>'{relationship_state,last_evidence_at}')::timestamptz, 'YYYY-MM-DD HH24:MI TZ'), 'unknown')
        || ']',
        E'\n' ORDER BY m.updated_at DESC, m.id
    )
    INTO injury_lines
    FROM (
        SELECT *
        FROM memories
        WHERE type = 'semantic'
          AND status = 'active'
          AND metadata#>>'{relationship_state,kind}' = 'relationship_injury'
          AND metadata#>>'{relationship_state,status}' = 'unresolved'
        ORDER BY updated_at DESC
        LIMIT 3
    ) m;

    WITH summaries AS (
        SELECT content, created_at
        FROM memories
        WHERE summary_lim > 0
          AND type = 'episodic'
          AND status = 'active'
          AND metadata ? 'recmem'
          AND created_at >= CURRENT_TIMESTAMP - (window_minutes * INTERVAL '1 minute')
          AND COALESCE(source_attribution->>'sensitivity', '') <> 'private'
        ORDER BY created_at DESC, id DESC
        LIMIT summary_lim
    )
    SELECT string_agg(
        '- [' || to_char(created_at, 'YYYY-MM-DD HH24:MI TZ') || '] '
        || left(regexp_replace(COALESCE(content, ''), '[[:space:]]+', ' ', 'g'), 700),
        E'\n' ORDER BY created_at ASC
    )
    INTO summary_lines
    FROM summaries;

    WITH corrections AS (
        SELECT content, metadata, updated_at, created_at
        FROM memories
        WHERE correction_lim > 0
          AND status = 'active'
          AND (
              metadata->>'invalid_precedent' = 'true'
              OR metadata ? 'latest_correction'
          )
          AND COALESCE(source_attribution->>'sensitivity', '') <> 'private'
        ORDER BY updated_at DESC, created_at DESC
        LIMIT correction_lim
    )
    SELECT string_agg(
        '- ' || left(regexp_replace(COALESCE(content, ''), '[[:space:]]+', ' ', 'g'), 420)
        || CASE WHEN metadata->>'invalid_precedent' = 'true'
                THEN E'\n  status: invalid precedent; do not imitate this behavior'
                ELSE '' END
        || CASE WHEN NULLIF(metadata#>>'{latest_correction,correction}', '') IS NOT NULL
                THEN E'\n  correction: ' || left(regexp_replace(metadata#>>'{latest_correction,correction}', '[[:space:]]+', ' ', 'g'), 420)
                ELSE '' END,
        E'\n' ORDER BY updated_at ASC, created_at ASC
    )
    INTO correction_lines
    FROM corrections;

    WITH heartbeats AS (
        SELECT
            m.id,
            COALESCE((m.metadata->>'event_time')::timestamptz, m.created_at) AS event_time,
            NULLIF(m.metadata#>>'{context,heartbeat_number}', '') AS heartbeat_number,
            m.content,
            NULLIF(m.metadata#>>'{context,reasoning}', '') AS reasoning,
            COALESCE(m.metadata#>'{context,actions_taken}', '[]'::jsonb) AS actions_taken
        FROM memories m
        WHERE heartbeat_lim > 0
          AND m.type = 'episodic'
          AND m.status = 'active'
          AND m.metadata#>>'{context,heartbeat_id}' IS NOT NULL
          AND COALESCE((m.metadata->>'event_time')::timestamptz, m.created_at) >= CURRENT_TIMESTAMP - (window_minutes * INTERVAL '1 minute')
          AND COALESCE(m.source_attribution->>'sensitivity', '') <> 'private'
        ORDER BY COALESCE((m.metadata->>'event_time')::timestamptz, m.created_at) DESC, m.id DESC
        LIMIT heartbeat_lim
    ),
    rendered AS (
        SELECT
            h.id,
            h.event_time,
            '- [' || to_char(h.event_time, 'YYYY-MM-DD HH24:MI TZ') || '] Heartbeat #'
            || COALESCE(h.heartbeat_number, '?') || E'\n'
            || '  summary: ' || left(regexp_replace(COALESCE(h.content, ''), '[[:space:]]+', ' ', 'g'), 520)
            || CASE WHEN failed.failures IS NOT NULL
                    THEN E'\n  failures: ' || failed.failures
                    ELSE '' END
            || CASE WHEN h.reasoning IS NOT NULL
                    THEN E'\n  note: ' || left(regexp_replace(h.reasoning, '[[:space:]]+', ' ', 'g'), 700)
                    ELSE '' END AS line
        FROM heartbeats h
        LEFT JOIN LATERAL (
            SELECT string_agg(
                COALESCE(a->>'action', 'unknown_action') || ': '
                || COALESCE(
                    NULLIF(a#>>'{result,error}', ''),
                    NULLIF(a#>>'{result,output_preview}', ''),
                    'failed'
                ),
                '; ' ORDER BY ord
            ) AS failures
            FROM jsonb_array_elements(h.actions_taken) WITH ORDINALITY AS elem(a, ord)
            WHERE COALESCE((a#>>'{result,success}')::boolean, FALSE) = FALSE
        ) failed ON TRUE
    )
    SELECT string_agg(line, E'\n' ORDER BY event_time ASC, id)
    INTO heartbeat_lines
    FROM rendered;

    WITH defects AS (
        SELECT *
        FROM defect_reports
        WHERE defect_lim > 0
          AND status IN ('open', 'diagnosed', 'repair_proposed')
        ORDER BY
            CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
            last_seen_at DESC
        LIMIT defect_lim
    )
    SELECT string_agg(
        '- [' || severity || '] ' || title
        || ' (' || occurrence_count || ' occurrence' || CASE WHEN occurrence_count = 1 THEN '' ELSE 's' END || '; status=' || status || ')' || E'\n'
        || '  summary: ' || left(regexp_replace(COALESCE(summary, ''), '[[:space:]]+', ' ', 'g'), 360) || E'\n'
        || '  latest error: ' || left(regexp_replace(COALESCE(last_error, ''), '[[:space:]]+', ' ', 'g'), 360)
        || CASE WHEN jsonb_typeof(diagnosis) = 'object' AND diagnosis <> '{}'::jsonb
                THEN E'\n  diagnosis: ' || left(regexp_replace(COALESCE(diagnosis->>'hypothesis', ''), '[[:space:]]+', ' ', 'g'), 360)
                ELSE '' END,
        E'\n' ORDER BY
            CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
            last_seen_at DESC
    )
    INTO defect_lines
    FROM defects;

    WITH recent AS (
        SELECT
            s.id,
            s.turn_at,
            s.user_text,
            s.assistant_text,
            s.metadata->'emotional_context' AS affect,
            COALESCE(cs.surface, 'conversation') AS surface
        FROM subconscious_units s
        LEFT JOIN chat_sessions cs ON cs.id = s.session_id
        WHERE lim > 0
          AND s.status = 'active'
          AND COALESCE(s.metadata#>>'{recmem,kind}', '') <> 'source_document_desk'
          AND COALESCE(s.metadata->>'type', 'conversation') = 'conversation'
          AND (current_session IS NULL OR s.session_id IS DISTINCT FROM current_session)
          AND s.turn_at >= CURRENT_TIMESTAMP - (window_minutes * INTERVAL '1 minute')
          AND COALESCE(s.source_attribution->>'sensitivity', '') <> 'private'
          AND COALESCE(cs.surface, 'api') = ANY(ARRAY['api','web','chat','cli','tui','openai_compat'])
        ORDER BY s.turn_at DESC, s.id DESC
        LIMIT lim
    )
    SELECT string_agg(
        '- [' || to_char(turn_at, 'YYYY-MM-DD HH24:MI TZ') || '] '
        || surface || E'\n'
        || '  user: ' || left(regexp_replace(COALESCE(user_text, ''), '[[:space:]]+', ' ', 'g'), 500) || E'\n'
        || '  assistant: ' || left(regexp_replace(COALESCE(assistant_text, ''), '[[:space:]]+', ' ', 'g'), 500)
        || CASE WHEN jsonb_typeof(affect) = 'object'
                THEN E'\n  affect: ' || COALESCE(affect->>'primary_emotion', 'unknown')
                    || ', valence=' || COALESCE(affect->>'valence', '?')
                    || ', intensity=' || COALESCE(affect->>'intensity', '?')
                ELSE '' END,
        E'\n' ORDER BY turn_at ASC, id
    )
    INTO turn_lines
    FROM recent;

    IF injury_lines IS NULL
       AND summary_lines IS NULL
       AND correction_lines IS NULL
       AND heartbeat_lines IS NULL
       AND defect_lines IS NULL
       AND turn_lines IS NULL
       AND affect_line IS NULL THEN
        RETURN '';
    END IF;

    body := '## Conversation Continuity Packet' || E'\n'
        || 'This is active working-state continuity, not decorative RAG. Treat it as what you remember right now: recent raw turns, exchange summaries, affect, corrections, unresolved relationship weather, and unresolved substrate defects. Do not claim a prior exchange is unavailable merely because the UI opened a new session. Do not reset warmth or trust merely because the session id changed; let sincere repair change the state when there is evidence.' || E'\n';
    IF affect_line IS NOT NULL THEN
        body := body || E'\n' || '### Current Emotional State' || E'\n' || affect_line || E'\n';
    END IF;
    IF injury_lines IS NOT NULL THEN
        body := body || E'\n' || '### Unresolved Relationship Injuries' || E'\n' || injury_lines || E'\n';
    END IF;
    IF correction_lines IS NOT NULL THEN
        body := body || E'\n' || '### Active Corrections And Invalidated Precedents' || E'\n' || correction_lines || E'\n';
    END IF;
    IF summary_lines IS NOT NULL THEN
        body := body || E'\n' || '### Recent Exchange Summaries' || E'\n' || summary_lines || E'\n';
    END IF;
    IF heartbeat_lines IS NOT NULL THEN
        body := body || E'\n' || '### Recent Autonomous Heartbeats' || E'\n'
            || 'Recent autonomous cycles are part of what has been on your mind. If a cycle failed or left unresolved work, surface it naturally when relevant instead of waiting for the user to ask.' || E'\n'
            || heartbeat_lines || E'\n';
    END IF;
    IF defect_lines IS NOT NULL THEN
        body := body || E'\n' || '### Unresolved Software Defects' || E'\n'
            || 'These are defects observed in your own substrate. Treat them as operational responsibilities: do not dramatize them, but bring them up naturally when they affect the current conversation or when the user asks how you are doing.' || E'\n'
            || defect_lines || E'\n';
    END IF;
    IF turn_lines IS NOT NULL THEN
        body := body || E'\n' || '### Recent Prior Turns' || E'\n' || turn_lines;
    END IF;

    IF length(body) > max_chars THEN
        body := left(body, max_chars) || E'\n[truncated chat continuity packet]';
    END IF;
    RETURN body;
END;
$$;

CREATE OR REPLACE FUNCTION render_recent_conversation_carryover(
    p_session_id TEXT DEFAULT NULL,
    p_exclude_sensitive BOOLEAN DEFAULT FALSE
) RETURNS TEXT
LANGUAGE sql
STABLE
AS $$
    SELECT render_chat_continuity_context(p_session_id, p_exclude_sensitive)
$$;

CREATE OR REPLACE FUNCTION apply_heartbeat_decision(
    p_heartbeat_id UUID,
    p_decision JSONB,
    p_start_index INT DEFAULT 0,
    p_pre_executed_actions JSONB DEFAULT '[]'::jsonb
)
RETURNS JSONB AS $$
DECLARE
    actions JSONB;
    goal_changes JSONB;
    reasoning TEXT;
    emotional JSONB;
    batch JSONB;
    new_actions JSONB;
    existing_actions JSONB;
    defect_actions JSONB := '[]'::jsonb;
    next_index INT;
    pending_external JSONB;
    halt_reason TEXT;
    memory_id UUID;
    outbox_messages JSONB := '[]'::jsonb;
    repl_energy FLOAT := 0;
    elem JSONB;
BEGIN
    IF p_decision IS NULL OR jsonb_typeof(p_decision) <> 'object' THEN
        RETURN jsonb_build_object('error', 'invalid_decision');
    END IF;

    actions := COALESCE(p_decision->'actions', '[]'::jsonb);
    IF jsonb_typeof(actions) <> 'array' THEN
        actions := '[]'::jsonb;
    END IF;

    goal_changes := COALESCE(p_decision->'goal_changes', '[]'::jsonb);
    IF jsonb_typeof(goal_changes) <> 'array' THEN
        goal_changes := '[]'::jsonb;
    END IF;

    reasoning := COALESCE(p_decision->>'reasoning', '');
    emotional := CASE
        WHEN jsonb_typeof(p_decision->'emotional_assessment') = 'object' THEN p_decision->'emotional_assessment'
        ELSE NULL
    END;

    batch := execute_heartbeat_actions_batch(p_heartbeat_id, actions, p_start_index);
    new_actions := COALESCE(batch->'actions_taken', '[]'::jsonb);
    IF jsonb_typeof(new_actions) <> 'array' THEN
        new_actions := '[]'::jsonb;
    END IF;
    defect_actions := new_actions;

    BEGIN
        next_index := COALESCE((batch->>'next_index')::int, COALESCE(p_start_index, 0));
    EXCEPTION
        WHEN OTHERS THEN
            next_index := COALESCE(p_start_index, 0);
    END;

    BEGIN
        pending_external := batch->'pending_external_call';
    EXCEPTION
        WHEN OTHERS THEN
            pending_external := NULL;
    END;

    halt_reason := NULLIF(batch->>'halt_reason', '');
    outbox_messages := COALESCE(batch->'outbox_messages', '[]'::jsonb);

    SELECT COALESCE(active_actions, '[]'::jsonb)
    INTO existing_actions
    FROM heartbeat_state
    WHERE id = 1;

    IF existing_actions IS NULL OR jsonb_typeof(existing_actions) <> 'array' THEN
        existing_actions := '[]'::jsonb;
    END IF;

    -- Prepend any pre-executed actions (from RLM REPL tool calls) then append new actions
    IF p_pre_executed_actions IS NOT NULL
       AND jsonb_typeof(p_pre_executed_actions) = 'array'
       AND jsonb_array_length(p_pre_executed_actions) > 0 THEN
        existing_actions := p_pre_executed_actions || existing_actions || new_actions;
        defect_actions := p_pre_executed_actions || new_actions;
        -- Deduct energy already spent by REPL tool calls
        FOR elem IN SELECT * FROM jsonb_array_elements(p_pre_executed_actions) LOOP
            repl_energy := repl_energy + COALESCE((elem->'result'->>'energy_spent')::float, 0);
        END LOOP;
        IF repl_energy > 0 THEN
            PERFORM update_energy(-repl_energy);
        END IF;
    ELSE
        existing_actions := existing_actions || new_actions;
    END IF;
    UPDATE heartbeat_state
    SET active_actions = existing_actions,
        active_reasoning = reasoning,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;

    IF jsonb_typeof(defect_actions) = 'array'
       AND jsonb_array_length(defect_actions) > 0 THEN
        PERFORM record_heartbeat_action_defects(p_heartbeat_id, defect_actions, reasoning);
    END IF;

    IF pending_external IS NOT NULL AND jsonb_typeof(pending_external) = 'object' THEN
        RETURN jsonb_build_object(
            'pending_external_call', pending_external,
            'next_index', next_index,
            'actions_taken', existing_actions,
            'outbox_messages', outbox_messages,
            'completed', false,
            'halt_reason', halt_reason
        );
    END IF;

    IF halt_reason = 'terminated' THEN
        RETURN jsonb_build_object(
            'terminated', true,
            'completed', false,
            'actions_taken', existing_actions,
            'next_index', next_index,
            'outbox_messages', outbox_messages,
            'halt_reason', halt_reason
        );
    END IF;

    memory_id := finalize_heartbeat(
        p_heartbeat_id,
        reasoning,
        existing_actions,
        goal_changes,
        emotional
    );

    RETURN jsonb_build_object(
        'completed', true,
        'memory_id', memory_id,
        'actions_taken', existing_actions,
        'next_index', next_index,
        'outbox_messages', outbox_messages,
        'halt_reason', halt_reason
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION record_defect_event(
    p_source TEXT,
    p_component TEXT,
    p_error TEXT,
    p_context JSONB DEFAULT '{}'::jsonb
) RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    ctx JSONB := COALESCE(p_context, '{}'::jsonb);
    classification JSONB;
    normalized TEXT;
    fingerprint_value TEXT;
    evidence_item JSONB;
    defect_id UUID;
    heartbeat_uuid UUID;
    tool_name TEXT;
BEGIN
    IF NULLIF(btrim(COALESCE(p_error, '')), '') IS NULL THEN
        RAISE EXCEPTION 'defect error is required';
    END IF;

    classification := classify_defect_event(p_component, p_error, ctx);
    normalized := normalize_defect_error(p_error);
    fingerprint_value := md5(
        lower(COALESCE(NULLIF(btrim(p_source), ''), 'unknown'))
        || '|'
        || COALESCE(classification->>'category', 'execution_failure')
        || '|'
        || COALESCE(NULLIF(btrim(p_component), ''), 'unknown')
        || '|'
        || normalized
    );

    BEGIN
        heartbeat_uuid := NULLIF(ctx->>'heartbeat_id', '')::uuid;
    EXCEPTION WHEN OTHERS THEN
        heartbeat_uuid := NULL;
    END;
    tool_name := COALESCE(NULLIF(ctx->>'tool_name', ''), NULLIF(ctx->>'action', ''), NULLIF(btrim(p_component), ''));

    evidence_item := jsonb_build_object(
        'at', CURRENT_TIMESTAMP,
        'source', COALESCE(NULLIF(btrim(p_source), ''), 'unknown'),
        'component', COALESCE(NULLIF(btrim(p_component), ''), 'unknown'),
        'error', p_error,
        'context', ctx
    );

    INSERT INTO defect_reports (
        fingerprint,
        status,
        severity,
        category,
        source,
        component,
        title,
        summary,
        last_error,
        heartbeat_ids,
        tool_names,
        evidence
    )
    VALUES (
        fingerprint_value,
        'open',
        COALESCE(classification->>'severity', 'medium'),
        COALESCE(classification->>'category', 'execution_failure'),
        COALESCE(NULLIF(btrim(p_source), ''), 'unknown'),
        NULLIF(btrim(p_component), ''),
        COALESCE(classification->>'title', 'Execution failure'),
        COALESCE(classification->>'summary', 'The agent observed a failed operation.'),
        p_error,
        CASE WHEN heartbeat_uuid IS NULL THEN ARRAY[]::uuid[] ELSE ARRAY[heartbeat_uuid] END,
        CASE WHEN tool_name IS NULL THEN ARRAY[]::text[] ELSE ARRAY[tool_name] END,
        jsonb_build_array(evidence_item)
    )
    ON CONFLICT (fingerprint) DO UPDATE SET
        status = CASE
            WHEN defect_reports.status IN ('resolved', 'ignored') THEN 'open'
            ELSE defect_reports.status
        END,
        severity = EXCLUDED.severity,
        category = EXCLUDED.category,
        title = EXCLUDED.title,
        summary = EXCLUDED.summary,
        last_error = EXCLUDED.last_error,
        last_seen_at = CURRENT_TIMESTAMP,
        occurrence_count = defect_reports.occurrence_count + 1,
        heartbeat_ids = CASE
            WHEN heartbeat_uuid IS NULL OR heartbeat_uuid = ANY(defect_reports.heartbeat_ids)
            THEN defect_reports.heartbeat_ids
            ELSE array_append(defect_reports.heartbeat_ids, heartbeat_uuid)
        END,
        tool_names = CASE
            WHEN tool_name IS NULL OR tool_name = ANY(defect_reports.tool_names)
            THEN defect_reports.tool_names
            ELSE array_append(defect_reports.tool_names, tool_name)
        END,
        evidence = (
            SELECT COALESCE(jsonb_agg(value ORDER BY ord), '[]'::jsonb)
            FROM (
                SELECT value, ord
                FROM jsonb_array_elements(defect_reports.evidence || jsonb_build_array(evidence_item))
                     WITH ORDINALITY AS e(value, ord)
                ORDER BY ord DESC
                LIMIT 20
            ) recent
        ),
        updated_at = CURRENT_TIMESTAMP
    RETURNING id INTO defect_id;

    RETURN defect_id;
END;
$$;

CREATE OR REPLACE FUNCTION record_heartbeat_action_defects(
    p_heartbeat_id UUID,
    p_actions JSONB,
    p_reasoning TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    action_record JSONB;
    action_name TEXT;
    success_text TEXT;
    failed BOOLEAN;
    error_text TEXT;
    ids UUID[] := ARRAY[]::UUID[];
    defect_id UUID;
BEGIN
    IF p_actions IS NULL OR jsonb_typeof(p_actions) <> 'array' THEN
        RETURN '[]'::jsonb;
    END IF;

    FOR action_record IN SELECT value FROM jsonb_array_elements(p_actions)
    LOOP
        success_text := lower(COALESCE(action_record#>>'{result,success}', ''));
        failed := CASE
            WHEN success_text = 'true' THEN FALSE
            WHEN success_text = 'false' THEN TRUE
            ELSE action_record ? 'error'
                 OR NULLIF(action_record#>>'{result,error}', '') IS NOT NULL
                 OR NULLIF(action_record#>>'{result,output_preview}', '') IS NOT NULL
        END;
        IF NOT failed THEN
            CONTINUE;
        END IF;

        action_name := COALESCE(NULLIF(action_record->>'action', ''), NULLIF(action_record->>'tool_name', ''), 'unknown_action');
        error_text := COALESCE(
            NULLIF(action_record#>>'{result,error}', ''),
            NULLIF(action_record#>>'{result,output_preview}', ''),
            NULLIF(action_record->>'error', ''),
            'action failed without an error message'
        );
        defect_id := record_defect_event(
            'heartbeat',
            action_name,
            error_text,
            jsonb_build_object(
                'heartbeat_id', p_heartbeat_id,
                'action', action_name,
                'action_record', action_record,
                'reasoning_excerpt', left(COALESCE(p_reasoning, ''), 1000)
            )
        );
        ids := array_append(ids, defect_id);
    END LOOP;

    RETURN COALESCE(to_jsonb(ids), '[]'::jsonb);
END;
$$;

CREATE OR REPLACE FUNCTION list_defect_reports(
    p_status TEXT DEFAULT 'open',
    p_limit INT DEFAULT 10
) RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', id,
        'status', status,
        'severity', severity,
        'category', category,
        'source', source,
        'component', component,
        'title', title,
        'summary', summary,
        'last_error', last_error,
        'occurrence_count', occurrence_count,
        'first_seen_at', first_seen_at,
        'last_seen_at', last_seen_at,
        'heartbeat_ids', heartbeat_ids,
        'tool_names', tool_names,
        'diagnosis', diagnosis,
        'proposed_repair', proposed_repair,
        'verification', verification
    ) ORDER BY last_seen_at DESC), '[]'::jsonb)
    FROM (
        SELECT *
        FROM defect_reports
        WHERE COALESCE(p_status, 'open') = 'all'
           OR status = COALESCE(p_status, 'open')
        ORDER BY
            CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
            last_seen_at DESC
        LIMIT LEAST(GREATEST(COALESCE(p_limit, 10), 1), 50)
    ) d
$$;

CREATE OR REPLACE FUNCTION diagnose_defect_report(
    p_defect_id UUID
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    d defect_reports%ROWTYPE;
    likely_files JSONB;
    diagnosis_doc JSONB;
    repair_doc JSONB;
BEGIN
    SELECT * INTO d FROM defect_reports WHERE id = p_defect_id;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('success', false, 'error', 'defect not found');
    END IF;

    likely_files := CASE
        WHEN d.category = 'tool_contract' AND COALESCE(d.component, '') = 'get_strategies' THEN
            '["core/tools/memory.py","db/38_functions_db_native_tools.sql","services/prompts/rlm_heartbeat_system.md"]'::jsonb
        WHEN d.category = 'tool_contract' AND COALESCE(d.last_error, '') ILIKE '%unknown action%' THEN
            '["db/17_functions_subconscious_observations.sql","services/prompts/rlm_heartbeat_system.md","services/heartbeat_runner.py"]'::jsonb
        WHEN d.category = 'tool_contract' AND COALESCE(d.last_error, '') ILIKE '%unknown tool%' THEN
            '["core/tools/registry.py","core/tools/self_inspection.py","services/prompts/rlm_heartbeat_system.md"]'::jsonb
        WHEN d.category = 'dependency_unavailable' THEN
            '["apps/hexis_cli.py","services/worker_service.py","core/config.py"]'::jsonb
        WHEN d.category = 'configuration' THEN
            '["hexis-ui/app","core/tools/config.py","docs"]'::jsonb
        WHEN d.category = 'timeout' THEN
            '["services/worker_service.py","core/tools/registry.py","services/heartbeat_runner.py"]'::jsonb
        ELSE
            '["db","core","services"]'::jsonb
    END;

    diagnosis_doc := jsonb_build_object(
        'category', d.category,
        'severity', d.severity,
        'hypothesis', CASE
            WHEN d.category = 'tool_contract' THEN
                'The model, prompt, registry schema, or DB action executor disagrees about the valid tool/action contract.'
            WHEN d.category = 'dependency_unavailable' THEN
                'A required local service was unavailable or unreachable when the operation ran.'
            WHEN d.category = 'configuration' THEN
                'The user must authorize or configure a provider; this is not primarily a code defect.'
            WHEN d.category = 'timeout' THEN
                'The workload or provider call exceeded the current timeout and needs smaller units, retry/backoff, or better progress handling.'
            ELSE
                'The failure needs source and log inspection before a safe repair can be selected.'
        END,
        'evidence_count', jsonb_array_length(COALESCE(d.evidence, '[]'::jsonb)),
        'latest_error', d.last_error,
        'likely_files', likely_files
    );

    repair_doc := jsonb_build_object(
        'mode', 'proposal_only',
        'safe_to_apply_autonomously', false,
        'why_not_auto_apply', 'Heartbeat source edits remain approval-gated; self-repair may inspect and draft, not silently modify code.',
        'next_steps', jsonb_build_array(
            'Inspect the likely files and live schema for the recorded component/error.',
            'Reproduce or cite the failing path from the preserved evidence.',
            'Prepare the smallest source/schema/prompt patch that addresses the contract mismatch.',
            'Run the focused regression that would have caught the defect.',
            'Ask the user to approve applying the patch, or apply only in an explicitly granted dev mode.'
        ),
        'suggested_tests', CASE
            WHEN d.category = 'tool_contract' THEN
                '["focused tool/heartbeat regression for the failing component","git diff --check"]'::jsonb
            WHEN d.category = 'dependency_unavailable' THEN
                '["doctor/health check for the dependency","worker startup smoke test"]'::jsonb
            ELSE
                '["focused regression for the failing path","git diff --check"]'::jsonb
        END
    );

    UPDATE defect_reports
    SET status = CASE WHEN status = 'open' THEN 'repair_proposed' ELSE status END,
        diagnosis = diagnosis_doc,
        proposed_repair = repair_doc,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_defect_id;

    RETURN jsonb_build_object(
        'success', true,
        'defect_id', p_defect_id,
        'diagnosis', diagnosis_doc,
        'proposed_repair', repair_doc
    );
END;
$$;

CREATE OR REPLACE FUNCTION mark_defect_report_resolved(
    p_defect_id UUID,
    p_resolution TEXT,
    p_verification JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE defect_reports
    SET status = 'resolved',
        resolution = NULLIF(btrim(COALESCE(p_resolution, '')), ''),
        verification = COALESCE(p_verification, '{}'::jsonb),
        resolved_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_defect_id;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('success', false, 'error', 'defect not found');
    END IF;
    RETURN jsonb_build_object('success', true, 'defect_id', p_defect_id, 'status', 'resolved');
END;
$$;

CREATE OR REPLACE FUNCTION render_defect_reports_context(
    p_limit INT DEFAULT 5
) RETURNS TEXT
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    lines TEXT;
BEGIN
    WITH defects AS (
        SELECT *
        FROM defect_reports
        WHERE status IN ('open', 'diagnosed', 'repair_proposed')
        ORDER BY
            CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
            last_seen_at DESC
        LIMIT LEAST(GREATEST(COALESCE(p_limit, 5), 1), 20)
    )
    SELECT string_agg(
        '- [' || severity || '] ' || title
        || ' (' || occurrence_count || ' occurrence' || CASE WHEN occurrence_count = 1 THEN '' ELSE 's' END || '; status=' || status || ')' || E'\n'
        || '  summary: ' || left(regexp_replace(COALESCE(summary, ''), '[[:space:]]+', ' ', 'g'), 360) || E'\n'
        || '  latest error: ' || left(regexp_replace(COALESCE(last_error, ''), '[[:space:]]+', ' ', 'g'), 360)
        || CASE WHEN jsonb_typeof(diagnosis) = 'object' AND diagnosis <> '{}'::jsonb
                THEN E'\n  diagnosis: ' || left(regexp_replace(COALESCE(diagnosis->>'hypothesis', ''), '[[:space:]]+', ' ', 'g'), 360)
                ELSE '' END,
        E'\n' ORDER BY
            CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
            last_seen_at DESC
    )
    INTO lines
    FROM defects;

    RETURN lines;
END;
$$;

SET check_function_bodies = on;
