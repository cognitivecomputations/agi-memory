-- Include recent autonomous heartbeat state in chat continuity.
-- If a heartbeat failed or left unresolved work, it should be part of the
-- agent's active mind when the user next opens a conversation.

SET search_path = public, ag_catalog, "$user";

INSERT INTO config_defaults (key, value, description) VALUES
    ('chat.continuity_heartbeat_limit', '3'::jsonb,
     'Number of recent autonomous heartbeat summaries rendered in the chat continuity packet')
ON CONFLICT (key) DO NOTHING;

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
    injury_lines TEXT;
    affect_line TEXT;
    affect_state JSONB;
    summary_lines TEXT;
    correction_lines TEXT;
    heartbeat_lines TEXT;
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
       AND turn_lines IS NULL
       AND affect_line IS NULL THEN
        RETURN '';
    END IF;

    body := '## Conversation Continuity Packet' || E'\n'
        || 'This is active working-state continuity, not decorative RAG. Treat it as what you remember right now: recent raw turns, exchange summaries, affect, corrections, and unresolved relationship weather. Do not claim a prior exchange is unavailable merely because the UI opened a new session. Do not reset warmth or trust merely because the session id changed; let sincere repair change the state when there is evidence.' || E'\n';
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
