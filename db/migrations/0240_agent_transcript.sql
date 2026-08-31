-- `hexis transcript` (#54): a CLI command to render a conversation -- turns,
-- tool calls, subconscious signals, energy, and any [Correction] events --
-- from the DB, without copy-pasting out of the UI. Everything it needs is
-- already in Postgres (agent_turns for the exchange, agent_turn_events for
-- the per-turn signal trace); this just adds one read-only assembly function
-- plus lookup indices for --session.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

-- `hexis transcript --session <id>` looks up by either grouping.
CREATE INDEX IF NOT EXISTS idx_agent_turns_session
    ON agent_turns (session_id, created_at) WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_agent_turns_heartbeat
    ON agent_turns (heartbeat_id, created_at) WHERE heartbeat_id IS NOT NULL;

-- Renders conversations for `hexis transcript`: everything is already in
-- Postgres (agent_turns for the exchange, agent_turn_events for the tool/
-- subconscious signal trace -- keyed by turn_id, a tighter join than the
-- loosely-typed tool_executions audit table); this just assembles it. Exactly
-- one of p_last / p_session_id is meaningful at a time: p_session_id (a
-- chat_sessions.id OR a heartbeat_id) returns every turn for that session in
-- order; otherwise the most recent p_last turns across all sessions/modes.
CREATE OR REPLACE FUNCTION get_agent_transcript(
    p_last INT DEFAULT 20,
    p_session_id UUID DEFAULT NULL
) RETURNS JSONB
LANGUAGE sql STABLE
AS $$
    WITH selected_turns AS (
        SELECT *
        FROM agent_turns
        WHERE p_session_id IS NULL
           OR session_id = p_session_id
           OR heartbeat_id = p_session_id
        ORDER BY created_at DESC
        LIMIT CASE WHEN p_session_id IS NULL THEN GREATEST(COALESCE(p_last, 20), 1) END
    )
    SELECT COALESCE(jsonb_agg(
        jsonb_build_object(
            'id', t.id,
            'mode', t.mode,
            'session_id', t.session_id,
            'heartbeat_id', t.heartbeat_id,
            'status', t.status,
            'stopped_reason', t.stopped_reason,
            'user_message', t.user_message,
            'reply_text', COALESCE(t.result->>'visible_text', t.result->>'text'),
            'iterations', COALESCE((t.result->>'iterations')::int, (t.runtime_state->>'iterations')::int),
            'energy_spent', COALESCE((t.result->>'energy_spent')::numeric, (t.runtime_state->>'energy_spent')::numeric),
            'created_at', t.created_at,
            'completed_at', t.completed_at,
            'events', COALESCE(ev.events, '[]'::jsonb)
        ) ORDER BY t.created_at
    ), '[]'::jsonb)
    FROM selected_turns t
    LEFT JOIN LATERAL (
        SELECT jsonb_agg(
            jsonb_build_object(
                'event_type', e.event_type,
                'payload', e.payload,
                'created_at', e.created_at
            ) ORDER BY e.created_at
        ) AS events
        FROM agent_turn_events e
        WHERE e.turn_id = t.id
          -- The signals worth a transcript line; llm_request/response and raw
          -- text_delta streaming chunks would just be noise here.
          AND e.event_type IN (
              'tool_start', 'tool_result', 'energy_exhausted',
              'claim_flagged', 'question', 'approval_request',
              'error', 'phase_change'
          )
    ) ev ON true;
$$;
