-- Skill selection and tool-gate telemetry.
--
-- Which skills a turn activates decides which tools exist for it, and a tool
-- outside the active set is hard-refused. Nothing recorded either decision, so
-- a selector that silently failed to activate the right skill was invisible —
-- it took a hand audit to notice that seven of ten ordinary requests reached
-- only `core-memory`. These two tables make that measurable instead.
SET search_path = public, ag_catalog, "$user";

CREATE TABLE IF NOT EXISTS skill_selection_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID,
    surface TEXT NOT NULL DEFAULT 'chat',
    tool_context TEXT NOT NULL DEFAULT 'chat',
    query_preview TEXT,
    selected TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    -- Every candidate with its score and why it lost: [{name, score, gated}]
    considered JSONB NOT NULL DEFAULT '[]'::jsonb,
    exposed_tool_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_skill_selection_events_created
    ON skill_selection_events (created_at DESC);

CREATE TABLE IF NOT EXISTS tool_gate_refusals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID,
    tool_name TEXT NOT NULL,
    reason TEXT NOT NULL
        CHECK (reason IN ('not_available_in_active_skills', 'no_approver_available', 'denied')),
    active_skills TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tool_gate_refusals_created
    ON tool_gate_refusals (tool_name, created_at DESC);

-- Fail-soft by construction: telemetry must never break a turn.
CREATE OR REPLACE FUNCTION record_skill_selection(
    p_session_id UUID,
    p_surface TEXT,
    p_tool_context TEXT,
    p_query_preview TEXT,
    p_selected TEXT[],
    p_considered JSONB,
    p_exposed_tool_count INT
) RETURNS UUID AS $$
DECLARE
    new_id UUID;
BEGIN
    INSERT INTO skill_selection_events (
        session_id, surface, tool_context, query_preview,
        selected, considered, exposed_tool_count
    ) VALUES (
        p_session_id,
        COALESCE(NULLIF(btrim(p_surface), ''), 'chat'),
        COALESCE(NULLIF(btrim(p_tool_context), ''), 'chat'),
        left(COALESCE(p_query_preview, ''), 400),
        COALESCE(p_selected, ARRAY[]::TEXT[]),
        COALESCE(p_considered, '[]'::jsonb),
        GREATEST(COALESCE(p_exposed_tool_count, 0), 0)
    )
    RETURNING id INTO new_id;
    RETURN new_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION record_tool_gate_refusal(
    p_session_id UUID,
    p_tool_name TEXT,
    p_reason TEXT,
    p_active_skills TEXT[]
) RETURNS UUID AS $$
DECLARE
    new_id UUID;
BEGIN
    IF COALESCE(btrim(p_tool_name), '') = '' THEN
        RETURN NULL;
    END IF;
    INSERT INTO tool_gate_refusals (session_id, tool_name, reason, active_skills)
    VALUES (
        p_session_id, btrim(p_tool_name), p_reason,
        COALESCE(p_active_skills, ARRAY[]::TEXT[])
    )
    RETURNING id INTO new_id;
    RETURN new_id;
END;
$$ LANGUAGE plpgsql;

-- What the selector is doing lately: the reachability question, answerable.
CREATE OR REPLACE VIEW skill_selection_health AS
SELECT
    count(*) AS turns,
    count(*) FILTER (WHERE selected <@ ARRAY['core-memory']::TEXT[]) AS defaults_only,
    round(
        100.0 * count(*) FILTER (WHERE selected <@ ARRAY['core-memory']::TEXT[])
        / NULLIF(count(*), 0), 1
    ) AS defaults_only_pct,
    round(avg(exposed_tool_count), 1) AS avg_tools_exposed,
    max(created_at) AS last_turn_at
FROM skill_selection_events
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '7 days';
