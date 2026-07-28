-- Resource requests (#84): a structured channel for the agent to ask the
-- operator for resources — more energy, a config change, a backup, or
-- anything else — with a rationale. The operator decides (hexis requests
-- grant/deny); granted config changes apply immediately and land in the
-- change journal; every decision surfaces in the agent's context so she
-- learns what asks succeed. The agent asks; the operator decides — filing a
-- request never changes state by itself.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

CREATE TABLE IF NOT EXISTS resource_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- 'backup' is filed by the continuity drive when backups go stale.
    kind TEXT NOT NULL CHECK (kind IN ('energy_boost', 'config_change', 'backup', 'other')),
    target_key TEXT,
    requested_value JSONB,
    rationale TEXT NOT NULL,
    duration TEXT,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'granted', 'denied', 'modified')),
    decision_note TEXT,
    applied_value JSONB,
    requested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    decided_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_resource_requests_status
    ON resource_requests (status, requested_at DESC);

-- File a request. Requires a rationale; config changes require the target
-- key. The operator hears about it through the outbox (the same channel as
-- any other agent-initiated message).
CREATE OR REPLACE FUNCTION file_resource_request(
    p_kind TEXT,
    p_rationale TEXT,
    p_target_key TEXT DEFAULT NULL,
    p_requested_value JSONB DEFAULT NULL,
    p_duration TEXT DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    new_id UUID;
    summary TEXT;
BEGIN
    IF p_kind IS NULL OR p_kind NOT IN ('energy_boost', 'config_change', 'backup', 'other') THEN
        RAISE EXCEPTION 'kind must be energy_boost, config_change, backup, or other (got %)', p_kind;
    END IF;
    IF NULLIF(btrim(COALESCE(p_rationale, '')), '') IS NULL THEN
        RAISE EXCEPTION 'a rationale is required: say what you need and why';
    END IF;
    IF p_kind = 'config_change' AND NULLIF(btrim(COALESCE(p_target_key, '')), '') IS NULL THEN
        RAISE EXCEPTION 'config_change requests require target_key';
    END IF;

    INSERT INTO resource_requests (kind, target_key, requested_value, rationale, duration)
    VALUES (p_kind, NULLIF(btrim(COALESCE(p_target_key, '')), ''), p_requested_value,
            btrim(p_rationale), NULLIF(btrim(COALESCE(p_duration, '')), ''))
    RETURNING id INTO new_id;

    summary := format('Resource request [%s] %s%s: %s',
        left(new_id::text, 8), p_kind,
        CASE WHEN p_target_key IS NOT NULL THEN ' (' || p_target_key || ')' ELSE '' END,
        btrim(p_rationale));
    BEGIN
        -- delivery doc carries the request id in machine-readable form so the
        -- dashboard inbox can offer grant/deny without parsing the prose.
        PERFORM queue_outbox_message(
            summary || E'\nDecide with: hexis requests grant/deny ' || left(new_id::text, 8),
            'resource_request', 'resource_request',
            jsonb_build_object('mode', 'web_inbox', 'request_id', new_id));
    EXCEPTION WHEN OTHERS THEN
        RAISE WARNING 'resource request % filed but outbox notification failed: %', new_id, SQLERRM;
    END;

    RETURN jsonb_build_object(
        'request_id', new_id,
        'status', 'pending',
        'note', 'The operator decides; the decision will appear in your context.'
    );
END;
$$ LANGUAGE plpgsql;

-- Operator decision. 'granted' and 'modified' apply the effect immediately
-- where one is applicable: config changes go through set_config and land in
-- the change journal as a config_flip; energy boosts go through
-- update_energy (clamped by heartbeat.max_energy). 'modified' means granted
-- with a different value — pass it as p_applied_value.
CREATE OR REPLACE FUNCTION decide_resource_request(
    p_request_id UUID,
    p_decision TEXT,
    p_note TEXT DEFAULT NULL,
    p_applied_value JSONB DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    req resource_requests%ROWTYPE;
    effective JSONB;
    applied TEXT := 'none';
    new_energy FLOAT;
BEGIN
    IF p_decision IS NULL OR p_decision NOT IN ('granted', 'denied', 'modified') THEN
        RAISE EXCEPTION 'decision must be granted, denied, or modified (got %)', p_decision;
    END IF;
    IF p_decision = 'modified' AND p_applied_value IS NULL THEN
        RAISE EXCEPTION 'modified decisions carry the value actually granted (p_applied_value)';
    END IF;

    SELECT * INTO req FROM resource_requests WHERE id = p_request_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'resource request % does not exist', p_request_id;
    END IF;
    IF req.status <> 'pending' THEN
        RAISE EXCEPTION 'resource request % was already decided (%)', p_request_id, req.status;
    END IF;

    effective := COALESCE(p_applied_value, req.requested_value);

    IF p_decision IN ('granted', 'modified') THEN
        IF req.kind = 'config_change' THEN
            PERFORM set_config(req.target_key, effective);
            BEGIN
                PERFORM record_change('config_flip',
                    format('%s set to %s (resource request %s %s)',
                           req.target_key, effective::text, left(p_request_id::text, 8), p_decision),
                    jsonb_build_object('request_id', p_request_id, 'target_key', req.target_key,
                                       'value', effective, 'decision', p_decision));
            EXCEPTION WHEN undefined_function THEN NULL;
            END;
            applied := 'config';
        ELSIF req.kind = 'energy_boost' THEN
            new_energy := update_energy(COALESCE((effective #>> '{}')::float, 5.0));
            applied := 'energy';
        END IF;
    END IF;

    UPDATE resource_requests
    SET status = p_decision,
        decision_note = NULLIF(btrim(COALESCE(p_note, '')), ''),
        applied_value = CASE WHEN p_decision IN ('granted', 'modified') THEN effective END,
        decided_at = CURRENT_TIMESTAMP
    WHERE id = p_request_id;

    IF p_decision IN ('granted', 'modified') THEN
        BEGIN
            PERFORM record_reward_event(
                'resource_request_' || p_decision || ':' || req.kind,
                CASE WHEN req.kind = 'energy_boost' THEN 0.65 ELSE 0.45 END,
                CASE WHEN req.kind = 'energy_boost' THEN 0.7 ELSE 0.55 END,
                'resource_request',
                jsonb_build_object(
                    'request_id', p_request_id,
                    'kind', req.kind,
                    'target_key', req.target_key,
                    'applied', applied,
                    'requested_value', req.requested_value,
                    'applied_value', effective,
                    'decision_note', NULLIF(btrim(COALESCE(p_note, '')), '')
                )
            );
        EXCEPTION WHEN OTHERS THEN
            RAISE LOG 'record_reward_event failed for resource request %: %', p_request_id, SQLERRM;
        END;
    END IF;

    RETURN jsonb_build_object(
        'request_id', p_request_id,
        'status', p_decision,
        'applied', applied,
        'new_energy', new_energy
    );
END;
$$ LANGUAGE plpgsql;

-- Context summary: pending asks plus decisions made since the last
-- heartbeat, so each outcome surfaces exactly once where the agent thinks
-- (the #93 window). Used by the environment snapshot and the heartbeat plan
-- gap-fill.
CREATE OR REPLACE FUNCTION resource_requests_summary()
RETURNS JSONB AS $$
    SELECT jsonb_build_object(
        'pending', (SELECT COUNT(*) FROM resource_requests WHERE status = 'pending'),
        'recent_decisions', COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
                'id', left(d.id::text, 8),
                'kind', d.kind,
                'target_key', d.target_key,
                'status', d.status,
                'decision_note', d.decision_note,
                'decided_at', d.decided_at
            ) ORDER BY d.decided_at DESC)
            FROM (
                SELECT * FROM resource_requests
                WHERE decided_at > COALESCE(
                    (SELECT last_heartbeat_at FROM heartbeat_state WHERE id = 1),
                    CURRENT_TIMESTAMP - INTERVAL '7 days')
                ORDER BY decided_at DESC
                LIMIT 3
            ) d
        ), '[]'::jsonb)
    );
$$ LANGUAGE sql STABLE;

-- Operator list view (hexis requests).
CREATE OR REPLACE FUNCTION list_resource_requests(
    p_status TEXT DEFAULT NULL,
    p_limit INT DEFAULT 20
) RETURNS JSONB AS $$
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', r.id,
        'kind', r.kind,
        'target_key', r.target_key,
        'requested_value', r.requested_value,
        'rationale', r.rationale,
        'duration', r.duration,
        'status', r.status,
        'decision_note', r.decision_note,
        'applied_value', r.applied_value,
        'requested_at', r.requested_at,
        'decided_at', r.decided_at
    ) ORDER BY r.requested_at DESC), '[]'::jsonb)
    FROM (
        SELECT * FROM resource_requests
        WHERE (p_status IS NULL AND status = 'pending') OR status = p_status OR p_status = 'all'
        ORDER BY requested_at DESC
        LIMIT GREATEST(COALESCE(p_limit, 20), 1)
    ) r;
$$ LANGUAGE sql STABLE;
