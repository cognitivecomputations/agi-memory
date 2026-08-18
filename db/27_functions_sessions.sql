-- =============================================================
-- Sub-Agent Session Functions
-- =============================================================

-- Create a new sub-agent session
-- Each baseline file runs as its own psql session at initdb, so it must pin
-- its own search_path: under an ag_catalog-first cluster default, unqualified
-- CREATEs would land Hexis objects in ag_catalog (the #77 fossil bug).
SET search_path = public, ag_catalog, "$user";

CREATE OR REPLACE FUNCTION create_sub_agent_session(
    p_task TEXT,
    p_energy_budget INT DEFAULT 5,
    p_parent_heartbeat_id UUID DEFAULT NULL,
    p_parent_session_id TEXT DEFAULT NULL,
    p_tool_context TEXT DEFAULT 'heartbeat',
    p_notify_on_complete BOOLEAN DEFAULT true
) RETURNS UUID
LANGUAGE plpgsql AS $$
DECLARE
    v_id UUID;
BEGIN
    INSERT INTO sub_agent_sessions (
        task, energy_budget, parent_heartbeat_id, parent_session_id,
        tool_context, notify_on_complete
    ) VALUES (
        p_task, p_energy_budget, p_parent_heartbeat_id, p_parent_session_id,
        p_tool_context, p_notify_on_complete
    ) RETURNING id INTO v_id;

    RETURN v_id;
END;
$$;

-- List sub-agent sessions with optional status filter
CREATE OR REPLACE FUNCTION list_sub_agent_sessions(
    p_status TEXT DEFAULT NULL,
    p_limit INT DEFAULT 20
) RETURNS JSONB
LANGUAGE plpgsql AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT COALESCE(jsonb_agg(row_to_json(s)::jsonb ORDER BY s.created_at DESC), '[]'::jsonb)
    INTO v_result
    FROM (
        SELECT
            id, task, status, energy_budget, energy_spent,
            result, error, tool_context, notify_on_complete,
            created_at, started_at, completed_at
        FROM sub_agent_sessions
        WHERE (p_status IS NULL OR status = p_status)
        ORDER BY created_at DESC
        LIMIT p_limit
    ) s;

    RETURN v_result;
END;
$$;

-- Get a single sub-agent session with full details including transcript
CREATE OR REPLACE FUNCTION get_sub_agent_session(
    p_id UUID
) RETURNS JSONB
LANGUAGE plpgsql AS $$
DECLARE
    v_result JSONB;
BEGIN
    SELECT row_to_json(s)::jsonb INTO v_result
    FROM (
        SELECT
            id, task, status, energy_budget, energy_spent,
            result, error, tool_context, transcript,
            notify_on_complete, memory_id,
            created_at, started_at, completed_at
        FROM sub_agent_sessions
        WHERE id = p_id
    ) s;

    RETURN COALESCE(v_result, '{}'::jsonb);
END;
$$;

-- Update a sub-agent session status and results
CREATE OR REPLACE FUNCTION update_sub_agent_session(
    p_id UUID,
    p_status TEXT DEFAULT NULL,
    p_result TEXT DEFAULT NULL,
    p_error TEXT DEFAULT NULL,
    p_energy_spent INT DEFAULT NULL,
    p_transcript JSONB DEFAULT NULL,
    p_memory_id UUID DEFAULT NULL
) RETURNS BOOLEAN
LANGUAGE plpgsql AS $$
BEGIN
    UPDATE sub_agent_sessions SET
        status = COALESCE(p_status, status),
        result = COALESCE(p_result, result),
        error = COALESCE(p_error, error),
        energy_spent = COALESCE(p_energy_spent, energy_spent),
        transcript = COALESCE(p_transcript, transcript),
        memory_id = COALESCE(p_memory_id, memory_id),
        started_at = CASE
            WHEN p_status = 'running' AND started_at IS NULL THEN CURRENT_TIMESTAMP
            ELSE started_at
        END,
        completed_at = CASE
            WHEN p_status IN ('completed', 'failed', 'cancelled') THEN CURRENT_TIMESTAMP
            ELSE completed_at
        END
    WHERE id = p_id;

    RETURN FOUND;
END;
$$;

-- Cancel a sub-agent session (only if pending or running)
CREATE OR REPLACE FUNCTION cancel_sub_agent_session(
    p_id UUID
) RETURNS BOOLEAN
LANGUAGE plpgsql AS $$
BEGIN
    UPDATE sub_agent_sessions
    SET status = 'cancelled', completed_at = CURRENT_TIMESTAMP
    WHERE id = p_id AND status IN ('pending', 'running');

    RETURN FOUND;
END;
$$;
