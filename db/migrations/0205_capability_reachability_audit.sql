-- Continuous worker/tool reachability and immutable per-turn tool-surface audit.
SET search_path = public, ag_catalog, "$user";

CREATE TABLE IF NOT EXISTS worker_capabilities (
    worker_name TEXT NOT NULL,
    worker_id UUID,
    tool_name TEXT NOT NULL,
    tool_context TEXT NOT NULL CHECK (tool_context IN ('heartbeat', 'chat', 'mcp')),
    available BOOLEAN NOT NULL,
    reason_code TEXT,
    reason_if_missing TEXT,
    registry_kind TEXT NOT NULL DEFAULT 'default',
    last_checked_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (worker_name, tool_context, tool_name)
);

CREATE INDEX IF NOT EXISTS idx_worker_capabilities_checked
    ON worker_capabilities (last_checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_worker_capabilities_gaps
    ON worker_capabilities (worker_name, reason_code, last_checked_at DESC)
    WHERE available = FALSE;

CREATE TABLE IF NOT EXISTS tool_surface_decision_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID,
    surface TEXT NOT NULL DEFAULT 'chat',
    tool_context TEXT NOT NULL,
    decision_kind TEXT NOT NULL DEFAULT 'selection'
        CHECK (decision_kind IN ('selection', 'skill_activation')),
    input_text_hash TEXT NOT NULL,
    selected_skills TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    considered JSONB NOT NULL DEFAULT '[]'::jsonb,
    allowed_tools TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    reachable_tools TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    unreachable_tools TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    available_skill_count INT NOT NULL DEFAULT 0,
    registry_kind TEXT NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tool_surface_decisions_created
    ON tool_surface_decision_events (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tool_surface_decisions_session
    ON tool_surface_decision_events (session_id, created_at DESC)
    WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tool_surface_decisions_gaps
    ON tool_surface_decision_events (created_at DESC)
    WHERE cardinality(unreachable_tools) > 0;

INSERT INTO config_defaults (key, value, description, source_path) VALUES
    ('capability_probe.interval_minutes', '15'::jsonb,
     'Minutes between worker/tool reachability probes.',
     'db/migrations/0205_capability_reachability_audit.sql'),
    ('capability_probe.stale_multiplier', '2'::jsonb,
     'A capability snapshot is stale after interval_minutes times this multiplier.',
     'db/migrations/0205_capability_reachability_audit.sql'),
    ('tool_surface.audit_enabled', 'true'::jsonb,
     'Persist an immutable record of each selected and actually reachable tool surface.',
     'db/migrations/0205_capability_reachability_audit.sql')
ON CONFLICT (key) DO NOTHING;

CREATE OR REPLACE FUNCTION record_worker_capabilities(
    p_worker_name TEXT,
    p_worker_id UUID,
    p_registry_kind TEXT,
    p_results JSONB
) RETURNS INT AS $$
DECLARE
    item JSONB;
    recorded INT := 0;
BEGIN
    IF NULLIF(btrim(p_worker_name), '') IS NULL THEN
        RAISE EXCEPTION 'worker name is required';
    END IF;
    IF jsonb_typeof(COALESCE(p_results, '[]'::jsonb)) <> 'array' THEN
        RAISE EXCEPTION 'capability results must be an array';
    END IF;

    FOR item IN SELECT * FROM jsonb_array_elements(COALESCE(p_results, '[]'::jsonb))
    LOOP
        IF NULLIF(btrim(item->>'tool_name'), '') IS NULL
           OR (item->>'tool_context') NOT IN ('heartbeat', 'chat', 'mcp') THEN
            CONTINUE;
        END IF;
        INSERT INTO worker_capabilities (
            worker_name, worker_id, tool_name, tool_context, available,
            reason_code, reason_if_missing, registry_kind, last_checked_at
        ) VALUES (
            btrim(p_worker_name), p_worker_id, btrim(item->>'tool_name'),
            item->>'tool_context', COALESCE((item->>'available')::boolean, FALSE),
            NULLIF(item->>'reason_code', ''), NULLIF(item->>'reason_if_missing', ''),
            COALESCE(NULLIF(btrim(p_registry_kind), ''), 'default'), CURRENT_TIMESTAMP
        )
        ON CONFLICT (worker_name, tool_context, tool_name) DO UPDATE SET
            worker_id = EXCLUDED.worker_id,
            available = EXCLUDED.available,
            reason_code = EXCLUDED.reason_code,
            reason_if_missing = EXCLUDED.reason_if_missing,
            registry_kind = EXCLUDED.registry_kind,
            last_checked_at = EXCLUDED.last_checked_at;
        recorded := recorded + 1;
    END LOOP;
    RETURN recorded;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION record_tool_surface_decision(
    p_session_id UUID,
    p_surface TEXT,
    p_tool_context TEXT,
    p_decision_kind TEXT,
    p_input_text_hash TEXT,
    p_selected_skills TEXT[],
    p_considered JSONB,
    p_allowed_tools TEXT[],
    p_reachable_tools TEXT[],
    p_unreachable_tools TEXT[],
    p_available_skill_count INT,
    p_registry_kind TEXT
) RETURNS UUID AS $$
DECLARE
    new_id UUID;
BEGIN
    INSERT INTO tool_surface_decision_events (
        session_id, surface, tool_context, decision_kind, input_text_hash, selected_skills,
        considered, allowed_tools, reachable_tools, unreachable_tools,
        available_skill_count, registry_kind
    ) VALUES (
        p_session_id,
        COALESCE(NULLIF(btrim(p_surface), ''), 'chat'),
        COALESCE(NULLIF(btrim(p_tool_context), ''), 'chat'),
        CASE WHEN p_decision_kind = 'skill_activation' THEN p_decision_kind ELSE 'selection' END,
        COALESCE(NULLIF(btrim(p_input_text_hash), ''), repeat('0', 64)),
        COALESCE(p_selected_skills, ARRAY[]::TEXT[]),
        COALESCE(p_considered, '[]'::jsonb),
        COALESCE(p_allowed_tools, ARRAY[]::TEXT[]),
        COALESCE(p_reachable_tools, ARRAY[]::TEXT[]),
        COALESCE(p_unreachable_tools, ARRAY[]::TEXT[]),
        GREATEST(COALESCE(p_available_skill_count, 0), 0),
        COALESCE(NULLIF(btrim(p_registry_kind), ''), 'default')
    ) RETURNING id INTO new_id;
    RETURN new_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION reject_tool_surface_audit_mutation()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION '% is append-only; % is not permitted', TG_TABLE_NAME, TG_OP;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_tool_surface_decisions_immutable ON tool_surface_decision_events;
CREATE TRIGGER trg_tool_surface_decisions_immutable
    BEFORE UPDATE OR DELETE ON tool_surface_decision_events
    FOR EACH ROW EXECUTE FUNCTION reject_tool_surface_audit_mutation();

CREATE OR REPLACE FUNCTION capability_reachability_health()
RETURNS JSONB AS $$
DECLARE
    interval_minutes INT := GREATEST(
        COALESCE(get_config_int('capability_probe.interval_minutes'), 15), 1
    );
    stale_multiplier FLOAT := GREATEST(
        COALESCE(get_config_float('capability_probe.stale_multiplier'), 2.0), 1.0
    );
    stale_before TIMESTAMPTZ;
    result JSONB;
BEGIN
    stale_before := CURRENT_TIMESTAMP
        - make_interval(secs => (interval_minutes * stale_multiplier * 60)::INT);
    SELECT jsonb_build_object(
        'workers', count(DISTINCT worker_name),
        'measured_pairs', count(*),
        'available_pairs', count(*) FILTER (WHERE available),
        'unavailable_pairs', count(*) FILTER (WHERE NOT available),
        'unexpected_gaps', count(*) FILTER (
            WHERE NOT available
              AND (
                  reason_code = 'handler_not_registered'
                  OR (
                      reason_code = 'skill_unbound'
                      AND NOT EXISTS (
                          SELECT 1 FROM worker_capabilities peer
                          WHERE peer.worker_name = worker_capabilities.worker_name
                            AND peer.tool_name = worker_capabilities.tool_name
                            AND peer.tool_context IN ('chat', 'heartbeat')
                            AND peer.available
                      )
                  )
              )
        ),
        'stale_pairs', count(*) FILTER (WHERE last_checked_at < stale_before),
        'last_checked_at', max(last_checked_at),
        'gap_examples', COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
                'worker', gap.worker_name,
                'context', gap.tool_context,
                'tool', gap.tool_name,
                'reason', gap.reason_code
            ))
            FROM (
                SELECT worker_name, tool_context, tool_name, reason_code
                FROM worker_capabilities
                WHERE NOT available
                  AND (
                      reason_code = 'handler_not_registered'
                      OR (
                          reason_code = 'skill_unbound'
                          AND NOT EXISTS (
                              SELECT 1 FROM worker_capabilities peer
                              WHERE peer.worker_name = worker_capabilities.worker_name
                                AND peer.tool_name = worker_capabilities.tool_name
                                AND peer.tool_context IN ('chat', 'heartbeat')
                                AND peer.available
                          )
                      )
                  )
                ORDER BY last_checked_at DESC, worker_name, tool_context, tool_name
                LIMIT 5
            ) gap
        ), '[]'::jsonb)
    ) INTO result
    FROM worker_capabilities;
    RETURN result;
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION tool_surface_audit_health()
RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'decisions_7d', count(*),
        'decisions_with_gaps', count(*) FILTER (WHERE cardinality(unreachable_tools) > 0),
        'avg_reachable_tools', round(avg(cardinality(reachable_tools)), 1),
        'last_decision_at', max(created_at)
    ) INTO result
    FROM tool_surface_decision_events
    WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '7 days';
    RETURN result;
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON TABLE worker_capabilities IS
    'Current per-worker/context/tool reachability, derived from registry + config + skills.';
COMMENT ON TABLE tool_surface_decision_events IS
    'Append-only record of each selected tool surface and the tools actually callable.';
