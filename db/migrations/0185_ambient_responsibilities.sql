-- Ambient responsibilities: durable observe/evaluate/notify commitments.
--
-- scheduled_tasks is a timed action runner. Ambient responsibilities add the
-- missing condition-monitoring layer: source observations, evaluator state,
-- check-ins, run audit, and DB-owned due claiming.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

INSERT INTO config_defaults (key, value, description) VALUES
    ('ambient.enabled', 'true'::jsonb, 'Run the ambient responsibility worker'),
    ('ambient.batch_size', '20'::jsonb, 'Ambient responsibilities claimed per worker tick'),
    ('ambient.default_poll_interval_seconds', '60'::jsonb, 'Default condition-check cadence for ambient responsibilities'),
    ('ambient.claim_timeout_seconds', '300'::jsonb, 'Seconds before an in-progress ambient check can be reclaimed'),
    ('ambient.failure_retry_base_seconds', '60'::jsonb, 'Base seconds for ambient responsibility failure backoff'),
    ('ambient.gmail_default_page_size', '10'::jsonb, 'Messages fetched per Gmail ambient monitor check')
ON CONFLICT (key) DO NOTHING;

CREATE TABLE IF NOT EXISTS ambient_responsibilities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    description TEXT,
    kind TEXT NOT NULL DEFAULT 'monitor'
        CHECK (kind IN ('reminder', 'monitor', 'checkin', 'threshold', 'digest', 'custom')),
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('proposed', 'active', 'paused', 'blocked', 'expired', 'revoked', 'disabled')),
    priority TEXT NOT NULL DEFAULT 'normal'
        CHECK (priority IN ('low', 'normal', 'high', 'urgent')),
    user_intent TEXT NOT NULL,
    trigger JSONB NOT NULL DEFAULT '{}'::jsonb,
    evaluator JSONB NOT NULL DEFAULT '{}'::jsonb,
    sources JSONB NOT NULL DEFAULT '[]'::jsonb,
    actions JSONB NOT NULL DEFAULT '[]'::jsonb,
    delivery JSONB NOT NULL DEFAULT '{"mode":"outbox"}'::jsonb,
    memory_policy TEXT NOT NULL DEFAULT 'task_scoped'
        CHECK (memory_policy IN ('remember', 'task_scoped', 'forget')),
    timezone TEXT NOT NULL DEFAULT 'UTC',
    next_check_at TIMESTAMPTZ,
    last_checked_at TIMESTAMPTZ,
    last_fired_at TIMESTAMPTZ,
    last_observation_id UUID,
    last_run_id UUID,
    consecutive_errors INT NOT NULL DEFAULT 0,
    consecutive_silent INT NOT NULL DEFAULT 0,
    cooldown_until TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    created_by TEXT NOT NULL DEFAULT 'agent',
    source_session_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ambient_responsibilities_due
    ON ambient_responsibilities (next_check_at, priority, created_at)
    WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_ambient_responsibilities_status
    ON ambient_responsibilities (status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_ambient_responsibilities_sources
    ON ambient_responsibilities USING GIN (sources);
CREATE INDEX IF NOT EXISTS idx_ambient_responsibilities_evaluator
    ON ambient_responsibilities USING GIN (evaluator);

CREATE TABLE IF NOT EXISTS ambient_observations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    responsibility_id UUID REFERENCES ambient_responsibilities(id) ON DELETE CASCADE,
    connector_id TEXT,
    account_key TEXT,
    item_kind TEXT NOT NULL DEFAULT 'event',
    provider_item_id TEXT NOT NULL DEFAULT gen_random_uuid()::text,
    provider_thread_id TEXT,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    title TEXT,
    content TEXT,
    participants JSONB NOT NULL DEFAULT '[]'::jsonb,
    labels TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    source_item_id UUID REFERENCES connector_source_items(id) ON DELETE SET NULL,
    source_document_id UUID REFERENCES source_documents(id) ON DELETE SET NULL,
    raw JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (responsibility_id, connector_id, account_key, item_kind, provider_item_id)
);

CREATE INDEX IF NOT EXISTS idx_ambient_observations_responsibility
    ON ambient_observations (responsibility_id, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_ambient_observations_connector
    ON ambient_observations (connector_id, account_key, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_ambient_observations_raw
    ON ambient_observations USING GIN (raw);

CREATE TABLE IF NOT EXISTS ambient_responsibility_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    responsibility_id UUID NOT NULL REFERENCES ambient_responsibilities(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'checking'
        CHECK (status IN ('checking', 'silent', 'fired', 'blocked', 'failed', 'skipped')),
    due_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMPTZ,
    trigger_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    observations JSONB NOT NULL DEFAULT '[]'::jsonb,
    decision JSONB NOT NULL DEFAULT '{}'::jsonb,
    outbox_messages JSONB NOT NULL DEFAULT '[]'::jsonb,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ambient_responsibility_runs_responsibility
    ON ambient_responsibility_runs (responsibility_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_ambient_responsibility_runs_status
    ON ambient_responsibility_runs (status, started_at DESC);

CREATE TABLE IF NOT EXISTS ambient_checkins (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    responsibility_id UUID NOT NULL REFERENCES ambient_responsibilities(id) ON DELETE CASCADE,
    label TEXT,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    note TEXT,
    source TEXT NOT NULL DEFAULT 'user',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ambient_checkins_responsibility
    ON ambient_checkins (responsibility_id, occurred_at DESC);

CREATE OR REPLACE FUNCTION ambient_compute_next_check(
    p_trigger JSONB DEFAULT '{}'::jsonb,
    p_timezone TEXT DEFAULT 'UTC',
    p_after TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
) RETURNS TIMESTAMPTZ
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    trigger_doc JSONB := COALESCE(p_trigger, '{}'::jsonb);
    trigger_kind TEXT := lower(COALESCE(NULLIF(trigger_doc->>'kind', ''), 'interval'));
    tz TEXT;
    secs INT;
    times_doc JSONB;
    time_text TEXT;
    target_time TIME;
    local_after TIMESTAMP;
    local_day DATE;
    candidate TIMESTAMPTZ;
    best TIMESTAMPTZ;
    d INT;
BEGIN
    BEGIN
        tz := normalize_timezone(p_timezone);
    EXCEPTION WHEN OTHERS THEN
        tz := 'UTC';
    END;

    IF trigger_kind IN ('manual', 'event') THEN
        RETURN NULL;
    END IF;

    IF trigger_kind = 'cron' AND NULLIF(btrim(COALESCE(trigger_doc->>'cron', '')), '') IS NOT NULL THEN
        RETURN cron_next_fire(trigger_doc->>'cron', tz, p_after);
    END IF;

    IF trigger_kind IN ('daily', 'daily_window', 'time_of_day') THEN
        local_after := p_after AT TIME ZONE tz;
        times_doc := trigger_doc->'times';
        FOR d IN 0..7 LOOP
            local_day := (local_after::date + d);
            best := NULL;

            IF jsonb_typeof(times_doc) = 'array' THEN
                FOR time_text IN SELECT value FROM jsonb_array_elements_text(times_doc)
                LOOP
                    BEGIN
                        target_time := parse_time_of_day(time_text);
                    EXCEPTION WHEN OTHERS THEN
                        CONTINUE;
                    END;
                    candidate := (local_day + target_time) AT TIME ZONE tz;
                    IF candidate > p_after AND (best IS NULL OR candidate < best) THEN
                        best := candidate;
                    END IF;
                END LOOP;
            ELSIF NULLIF(btrim(COALESCE(trigger_doc->>'time', '')), '') IS NOT NULL THEN
                target_time := parse_time_of_day(trigger_doc->>'time');
                candidate := (local_day + target_time) AT TIME ZONE tz;
                IF candidate > p_after THEN
                    best := candidate;
                END IF;
            END IF;

            IF best IS NOT NULL THEN
                RETURN best;
            END IF;
        END LOOP;
    END IF;

    BEGIN
        secs := COALESCE(
            NULLIF(trigger_doc->>'every_seconds', '')::int,
            NULLIF(trigger_doc->>'interval_seconds', '')::int,
            get_config_int('ambient.default_poll_interval_seconds'),
            60
        );
    EXCEPTION WHEN OTHERS THEN
        secs := COALESCE(get_config_int('ambient.default_poll_interval_seconds'), 60);
    END;
    secs := LEAST(GREATEST(COALESCE(secs, 60), 5), 86400);
    RETURN p_after + make_interval(secs => secs);
END;
$$;

CREATE OR REPLACE FUNCTION ambient_missing_connectors(
    p_sources JSONB DEFAULT '[]'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    sources_doc JSONB := CASE WHEN jsonb_typeof(COALESCE(p_sources, '[]'::jsonb)) = 'array'
                              THEN COALESCE(p_sources, '[]'::jsonb)
                              ELSE '[]'::jsonb END;
    source_doc JSONB;
    connector TEXT;
    required BOOLEAN;
    connected_count INT;
    missing JSONB := '[]'::jsonb;
BEGIN
    FOR source_doc IN SELECT value FROM jsonb_array_elements(sources_doc)
    LOOP
        connector := lower(NULLIF(btrim(COALESCE(source_doc->>'connector_id', source_doc->>'connector', '')), ''));
        IF connector IS NULL THEN
            CONTINUE;
        END IF;
        BEGIN
            required := COALESCE((source_doc->>'require_connection')::boolean, TRUE);
        EXCEPTION WHEN OTHERS THEN
            required := TRUE;
        END;
        IF NOT required THEN
            CONTINUE;
        END IF;

        SELECT COUNT(*)
        INTO connected_count
        FROM integration_connections
        WHERE connector_id = connector
          AND status = 'connected'
          AND (NULLIF(source_doc->>'account_key', '') IS NULL
               OR account_key = source_doc->>'account_key');

        IF connected_count = 0 THEN
            missing := missing || jsonb_build_array(jsonb_strip_nulls(jsonb_build_object(
                'connector_id', connector,
                'account_key', NULLIF(source_doc->>'account_key', ''),
                'status', 'not_connected'
            )));
        END IF;
    END LOOP;
    RETURN missing;
END;
$$;

CREATE OR REPLACE FUNCTION build_ambient_delivery(p_args JSONB)
RETURNS JSONB
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT CASE
        WHEN jsonb_typeof(COALESCE(p_args->'delivery', 'null'::jsonb)) = 'object'
            THEN p_args->'delivery'
        WHEN COALESCE(NULLIF(p_args->>'delivery_mode', ''), 'outbox') = 'channel'
            THEN jsonb_strip_nulls(jsonb_build_object(
                'mode', 'channel',
                'channel', NULLIF(p_args->>'delivery_channel', ''),
                'target_id', NULLIF(p_args->>'delivery_target_id', ''),
                'topic', NULLIF(p_args->>'delivery_topic', '')
            ))
        WHEN COALESCE(NULLIF(p_args->>'delivery_mode', ''), 'outbox') = 'webhook'
            THEN jsonb_strip_nulls(jsonb_build_object(
                'mode', 'webhook',
                'url', NULLIF(p_args->>'delivery_webhook_url', '')
            ))
        WHEN COALESCE(NULLIF(p_args->>'delivery_mode', ''), 'outbox') = 'silent'
            THEN '{"mode":"silent"}'::jsonb
        ELSE '{"mode":"outbox"}'::jsonb
    END;
$$;

CREATE OR REPLACE FUNCTION create_ambient_responsibility(
    p_args JSONB
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    title_value TEXT;
    intent_value TEXT;
    kind_value TEXT;
    status_value TEXT;
    priority_value TEXT;
    trigger_doc JSONB;
    evaluator_doc JSONB;
    sources_doc JSONB;
    actions_doc JSONB;
    delivery_doc JSONB;
    memory_policy_value TEXT;
    tz TEXT;
    missing JSONB;
    next_check TIMESTAMPTZ;
    row_resp ambient_responsibilities%ROWTYPE;
BEGIN
    intent_value := NULLIF(btrim(COALESCE(p_args->>'user_intent', p_args->>'description', p_args->>'message', '')), '');
    title_value := NULLIF(btrim(COALESCE(p_args->>'title', '')), '');
    IF intent_value IS NULL THEN
        RETURN jsonb_build_object('success', false, 'error', 'user_intent is required', 'error_type', 'invalid_params');
    END IF;
    IF title_value IS NULL THEN
        title_value := left(intent_value, 90);
    END IF;

    kind_value := lower(COALESCE(NULLIF(p_args->>'kind', ''), 'monitor'));
    IF kind_value NOT IN ('reminder', 'monitor', 'checkin', 'threshold', 'digest', 'custom') THEN
        RETURN jsonb_build_object('success', false, 'error', format('Invalid responsibility kind %L', kind_value), 'error_type', 'invalid_params');
    END IF;

    priority_value := lower(COALESCE(NULLIF(p_args->>'priority', ''), 'normal'));
    IF priority_value NOT IN ('low', 'normal', 'high', 'urgent') THEN
        priority_value := 'normal';
    END IF;

    trigger_doc := CASE WHEN jsonb_typeof(COALESCE(p_args->'trigger', 'null'::jsonb)) = 'object'
                        THEN p_args->'trigger'
                        WHEN NULLIF(p_args->>'schedule', '') IS NOT NULL
                        THEN jsonb_build_object('kind', 'cron', 'cron', p_args->>'schedule')
                        ELSE jsonb_build_object('kind', 'interval', 'every_seconds', COALESCE(get_config_int('ambient.default_poll_interval_seconds'), 60))
                   END;
    evaluator_doc := CASE WHEN jsonb_typeof(COALESCE(p_args->'evaluator', 'null'::jsonb)) = 'object'
                          THEN p_args->'evaluator'
                          ELSE '{}'::jsonb END;
    sources_doc := CASE WHEN jsonb_typeof(COALESCE(p_args->'sources', 'null'::jsonb)) = 'array'
                        THEN p_args->'sources'
                        ELSE '[]'::jsonb END;
    actions_doc := CASE WHEN jsonb_typeof(COALESCE(p_args->'actions', 'null'::jsonb)) = 'array'
                        THEN p_args->'actions'
                        WHEN NULLIF(p_args->>'message', '') IS NOT NULL
                        THEN jsonb_build_array(jsonb_build_object('type', 'notify_user', 'message', p_args->>'message'))
                        ELSE jsonb_build_array(jsonb_build_object('type', 'notify_user'))
                   END;
    delivery_doc := build_ambient_delivery(p_args);

    IF delivery_doc->>'mode' = 'channel' AND NULLIF(delivery_doc->>'target_id', '') IS NULL THEN
        RETURN jsonb_build_object('success', false, 'error', 'delivery_target_id is required when delivery_mode is channel', 'error_type', 'invalid_params');
    END IF;
    IF delivery_doc->>'mode' = 'webhook' AND NULLIF(delivery_doc->>'url', '') IS NULL THEN
        RETURN jsonb_build_object('success', false, 'error', 'delivery_webhook_url is required when delivery_mode is webhook', 'error_type', 'invalid_params');
    END IF;

    memory_policy_value := lower(COALESCE(NULLIF(p_args->>'memory_policy', ''), 'task_scoped'));
    IF memory_policy_value NOT IN ('remember', 'task_scoped', 'forget') THEN
        memory_policy_value := 'task_scoped';
    END IF;

    BEGIN
        tz := normalize_timezone(COALESCE(NULLIF(p_args->>'timezone', ''), get_config_text('agent.timezone'), 'UTC'));
    EXCEPTION WHEN OTHERS THEN
        tz := 'UTC';
    END;

    missing := ambient_missing_connectors(sources_doc);
    status_value := lower(COALESCE(NULLIF(p_args->>'status', ''), 'active'));
    IF jsonb_array_length(missing) > 0 AND status_value = 'active' THEN
        status_value := 'blocked';
    END IF;
    IF status_value NOT IN ('proposed', 'active', 'paused', 'blocked', 'expired', 'revoked', 'disabled') THEN
        status_value := 'active';
    END IF;

    IF status_value = 'active' THEN
        IF NULLIF(p_args->>'next_check_at', '') IS NOT NULL THEN
            next_check := (p_args->>'next_check_at')::timestamptz;
        ELSE
            next_check := ambient_compute_next_check(trigger_doc, tz, CURRENT_TIMESTAMP - INTERVAL '1 second');
        END IF;
    ELSE
        next_check := NULL;
    END IF;

    INSERT INTO ambient_responsibilities (
        title, description, kind, status, priority, user_intent, trigger,
        evaluator, sources, actions, delivery, memory_policy, timezone,
        next_check_at, expires_at, created_by, source_session_id, metadata
    )
    VALUES (
        title_value,
        NULLIF(p_args->>'description', ''),
        kind_value,
        status_value,
        priority_value,
        intent_value,
        trigger_doc,
        evaluator_doc,
        sources_doc,
        actions_doc,
        delivery_doc,
        memory_policy_value,
        tz,
        next_check,
        NULLIF(p_args->>'expires_at', '')::timestamptz,
        COALESCE(NULLIF(p_args->>'created_by', ''), 'agent'),
        NULLIF(p_args->>'source_session_id', ''),
        COALESCE(p_args->'metadata', '{}'::jsonb) || jsonb_build_object('missing_connectors', missing)
    )
    RETURNING * INTO row_resp;

    RETURN jsonb_build_object(
        'success', true,
        'output', jsonb_build_object(
            'responsibility_id', row_resp.id::text,
            'title', row_resp.title,
            'kind', row_resp.kind,
            'status', row_resp.status,
            'priority', row_resp.priority,
            'trigger', row_resp.trigger,
            'evaluator', row_resp.evaluator,
            'sources', row_resp.sources,
            'actions', row_resp.actions,
            'delivery', row_resp.delivery,
            'memory_policy', row_resp.memory_policy,
            'timezone', row_resp.timezone,
            'next_check_at', row_resp.next_check_at,
            'missing_connectors', missing
        ),
        'display_output', CASE WHEN row_resp.status = 'blocked'
            THEN format('Created ambient responsibility but blocked on setup: %s', row_resp.title)
            ELSE format('Created ambient responsibility: %s', row_resp.title)
        END
    );
EXCEPTION WHEN OTHERS THEN
    RETURN jsonb_build_object('success', false, 'error', SQLERRM, 'error_type', 'execution_failed');
END;
$$;

CREATE OR REPLACE FUNCTION ambient_responsibility_id_from_args(p_args JSONB)
RETURNS UUID
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    rid UUID;
BEGIN
    IF NULLIF(p_args->>'responsibility_id', '') IS NOT NULL THEN
        RETURN (p_args->>'responsibility_id')::uuid;
    END IF;
    IF NULLIF(p_args->>'title', '') IS NOT NULL THEN
        SELECT id INTO rid
        FROM ambient_responsibilities
        WHERE lower(title) = lower(p_args->>'title')
          AND status IN ('active', 'paused', 'blocked', 'proposed')
        ORDER BY updated_at DESC
        LIMIT 1;
        IF rid IS NOT NULL THEN
            RETURN rid;
        END IF;
    END IF;
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION list_ambient_responsibilities(
    p_status TEXT DEFAULT NULL,
    p_limit INT DEFAULT 50
) RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    lim INT := LEAST(GREATEST(COALESCE(p_limit, 50), 1), 200);
    result JSONB;
BEGIN
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', r.id::text,
        'title', r.title,
        'kind', r.kind,
        'status', r.status,
        'priority', r.priority,
        'user_intent', r.user_intent,
        'trigger', r.trigger,
        'evaluator', r.evaluator,
        'sources', r.sources,
        'actions', r.actions,
        'delivery', r.delivery,
        'memory_policy', r.memory_policy,
        'timezone', r.timezone,
        'next_check_at', r.next_check_at,
        'last_checked_at', r.last_checked_at,
        'last_fired_at', r.last_fired_at,
        'consecutive_errors', r.consecutive_errors,
        'consecutive_silent', r.consecutive_silent,
        'last_error', r.last_error,
        'created_at', r.created_at
    ) ORDER BY r.updated_at DESC), '[]'::jsonb)
    INTO result
    FROM (
        SELECT *
        FROM ambient_responsibilities
        WHERE NULLIF(p_status, '') IS NULL OR status = lower(p_status)
        ORDER BY updated_at DESC
        LIMIT lim
    ) r;
    RETURN result;
END;
$$;

CREATE OR REPLACE FUNCTION manage_ambient_responsibility_tool(
    p_args JSONB
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    action TEXT := lower(COALESCE(p_args->>'action', ''));
    created JSONB;
    rid UUID;
    missing JSONB;
    row_resp ambient_responsibilities%ROWTYPE;
    rows_doc JSONB;
    next_check TIMESTAMPTZ;
BEGIN
    IF action NOT IN ('create', 'list', 'pause', 'resume', 'cancel', 'status', 'checkin', 'evaluate_now') THEN
        RETURN jsonb_build_object('success', false, 'error', format('Invalid action %L', action), 'error_type', 'invalid_params');
    END IF;

    IF action = 'create' THEN
        RETURN create_ambient_responsibility(p_args);
    END IF;

    IF action = 'list' THEN
        rows_doc := list_ambient_responsibilities(NULLIF(p_args->>'status', ''), COALESCE(NULLIF(p_args->>'limit', '')::int, 50));
        RETURN jsonb_build_object('success', true, 'output', jsonb_build_object(
            'responsibilities', rows_doc,
            'count', jsonb_array_length(rows_doc)
        ), 'display_output', format('Found %s ambient responsibility(s)', jsonb_array_length(rows_doc)));
    END IF;

    rid := ambient_responsibility_id_from_args(p_args);
    IF rid IS NULL THEN
        RETURN jsonb_build_object('success', false, 'error', 'responsibility_id or title is required', 'error_type', 'invalid_params');
    END IF;

    IF action = 'status' THEN
        SELECT * INTO row_resp FROM ambient_responsibilities WHERE id = rid;
        IF NOT FOUND THEN
            RETURN jsonb_build_object('success', false, 'error', format('Ambient responsibility %s not found', rid), 'error_type', 'invalid_params');
        END IF;
        RETURN jsonb_build_object('success', true, 'output', (list_ambient_responsibilities(row_resp.status, 200) -> 0));
    ELSIF action = 'pause' THEN
        UPDATE ambient_responsibilities
        SET status = 'paused',
            next_check_at = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = rid
        RETURNING * INTO row_resp;
        RETURN jsonb_build_object('success', true, 'output', jsonb_build_object('responsibility_id', row_resp.id::text, 'status', row_resp.status), 'display_output', format('Paused ambient responsibility: %s', row_resp.title));
    ELSIF action = 'resume' THEN
        SELECT * INTO row_resp FROM ambient_responsibilities WHERE id = rid FOR UPDATE;
        IF NOT FOUND THEN
            RETURN jsonb_build_object('success', false, 'error', format('Ambient responsibility %s not found', rid), 'error_type', 'invalid_params');
        END IF;
        missing := ambient_missing_connectors(row_resp.sources);
        IF jsonb_array_length(missing) = 0 THEN
            next_check := ambient_compute_next_check(row_resp.trigger, row_resp.timezone, CURRENT_TIMESTAMP - INTERVAL '1 second');
            UPDATE ambient_responsibilities
            SET status = 'active',
                next_check_at = next_check,
                cooldown_until = NULL,
                last_error = NULL,
                metadata = metadata || jsonb_build_object('missing_connectors', '[]'::jsonb),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = rid
            RETURNING * INTO row_resp;
        ELSE
            UPDATE ambient_responsibilities
            SET status = 'blocked',
                next_check_at = NULL,
                metadata = metadata || jsonb_build_object('missing_connectors', missing),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = rid
            RETURNING * INTO row_resp;
        END IF;
        RETURN jsonb_build_object('success', true, 'output', jsonb_build_object(
            'responsibility_id', row_resp.id::text,
            'status', row_resp.status,
            'next_check_at', row_resp.next_check_at,
            'missing_connectors', missing
        ), 'display_output', format('Resumed ambient responsibility: %s (%s)', row_resp.title, row_resp.status));
    ELSIF action = 'cancel' THEN
        UPDATE ambient_responsibilities
        SET status = 'disabled',
            next_check_at = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = rid
        RETURNING * INTO row_resp;
        RETURN jsonb_build_object('success', true, 'output', jsonb_build_object('responsibility_id', row_resp.id::text, 'status', row_resp.status), 'display_output', format('Cancelled ambient responsibility: %s', row_resp.title));
    ELSIF action = 'evaluate_now' THEN
        UPDATE ambient_responsibilities
        SET status = CASE WHEN status = 'blocked' THEN 'blocked' ELSE 'active' END,
            next_check_at = CASE WHEN status = 'blocked' THEN NULL ELSE CURRENT_TIMESTAMP END,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = rid
        RETURNING * INTO row_resp;
        RETURN jsonb_build_object('success', true, 'output', jsonb_build_object('responsibility_id', row_resp.id::text, 'status', row_resp.status, 'next_check_at', row_resp.next_check_at), 'display_output', format('Queued ambient check: %s', row_resp.title));
    ELSE
        INSERT INTO ambient_checkins (responsibility_id, label, occurred_at, note, source, metadata)
        VALUES (
            rid,
            NULLIF(p_args->>'label', ''),
            COALESCE(NULLIF(p_args->>'occurred_at', '')::timestamptz, CURRENT_TIMESTAMP),
            NULLIF(p_args->>'note', ''),
            COALESCE(NULLIF(p_args->>'source', ''), 'user'),
            COALESCE(p_args->'metadata', '{}'::jsonb)
        );
        UPDATE ambient_responsibilities
        SET updated_at = CURRENT_TIMESTAMP
        WHERE id = rid
        RETURNING * INTO row_resp;
        RETURN jsonb_build_object('success', true, 'output', jsonb_build_object('responsibility_id', rid::text, 'checked_in', true), 'display_output', format('Recorded check-in: %s', row_resp.title));
    END IF;
EXCEPTION WHEN OTHERS THEN
    RETURN jsonb_build_object('success', false, 'error', SQLERRM, 'error_type', 'execution_failed');
END;
$$;

CREATE OR REPLACE FUNCTION record_ambient_observation(
    p_payload JSONB
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    rid UUID := NULL;
    source_item UUID := NULL;
    source_doc UUID := NULL;
    connector TEXT := lower(NULLIF(btrim(COALESCE(p_payload->>'connector_id', '')), ''));
    account TEXT := NULLIF(btrim(COALESCE(p_payload->>'account_key', '')), '');
    kind TEXT := COALESCE(NULLIF(btrim(COALESCE(p_payload->>'item_kind', '')), ''), 'event');
    provider_id TEXT := COALESCE(NULLIF(btrim(COALESCE(p_payload->>'provider_item_id', '')), ''), gen_random_uuid()::text);
    existing_id UUID;
    row_obs ambient_observations%ROWTYPE;
BEGIN
    IF NULLIF(p_payload->>'responsibility_id', '') IS NOT NULL THEN
        rid := (p_payload->>'responsibility_id')::uuid;
    END IF;
    IF NULLIF(p_payload->>'source_item_id', '') IS NOT NULL THEN
        source_item := (p_payload->>'source_item_id')::uuid;
    END IF;
    IF NULLIF(p_payload->>'source_document_id', '') IS NOT NULL THEN
        source_doc := (p_payload->>'source_document_id')::uuid;
    END IF;

    SELECT id INTO existing_id
    FROM ambient_observations
    WHERE responsibility_id IS NOT DISTINCT FROM rid
      AND connector_id IS NOT DISTINCT FROM connector
      AND account_key IS NOT DISTINCT FROM account
      AND item_kind = kind
      AND provider_item_id = provider_id
    LIMIT 1;

    INSERT INTO ambient_observations (
        responsibility_id, connector_id, account_key, item_kind,
        provider_item_id, provider_thread_id, observed_at, title, content,
        participants, labels, source_item_id, source_document_id, raw
    )
    VALUES (
        rid,
        connector,
        account,
        kind,
        provider_id,
        NULLIF(p_payload->>'provider_thread_id', ''),
        COALESCE(NULLIF(p_payload->>'observed_at', '')::timestamptz, CURRENT_TIMESTAMP),
        NULLIF(p_payload->>'title', ''),
        NULLIF(p_payload->>'content', ''),
        COALESCE(p_payload->'participants', '[]'::jsonb),
        COALESCE(ARRAY(SELECT jsonb_array_elements_text(CASE WHEN jsonb_typeof(p_payload->'labels') = 'array' THEN p_payload->'labels' ELSE '[]'::jsonb END)), ARRAY[]::TEXT[]),
        source_item,
        source_doc,
        COALESCE(p_payload->'raw', '{}'::jsonb)
    )
    ON CONFLICT (responsibility_id, connector_id, account_key, item_kind, provider_item_id)
    DO UPDATE SET
        provider_thread_id = EXCLUDED.provider_thread_id,
        observed_at = GREATEST(ambient_observations.observed_at, EXCLUDED.observed_at),
        title = COALESCE(EXCLUDED.title, ambient_observations.title),
        content = COALESCE(EXCLUDED.content, ambient_observations.content),
        participants = EXCLUDED.participants,
        labels = EXCLUDED.labels,
        source_item_id = COALESCE(EXCLUDED.source_item_id, ambient_observations.source_item_id),
        source_document_id = COALESCE(EXCLUDED.source_document_id, ambient_observations.source_document_id),
        raw = ambient_observations.raw || EXCLUDED.raw,
        updated_at = CURRENT_TIMESTAMP
    RETURNING * INTO row_obs;

    RETURN jsonb_build_object(
        'observation_id', row_obs.id::text,
        'created', existing_id IS NULL,
        'responsibility_id', CASE WHEN row_obs.responsibility_id IS NULL THEN NULL ELSE row_obs.responsibility_id::text END,
        'connector_id', row_obs.connector_id,
        'account_key', row_obs.account_key,
        'item_kind', row_obs.item_kind,
        'provider_item_id', row_obs.provider_item_id,
        'source_item_id', CASE WHEN row_obs.source_item_id IS NULL THEN NULL ELSE row_obs.source_item_id::text END,
        'source_document_id', CASE WHEN row_obs.source_document_id IS NULL THEN NULL ELSE row_obs.source_document_id::text END,
        'title', row_obs.title,
        'observed_at', row_obs.observed_at
    );
END;
$$;

CREATE OR REPLACE FUNCTION claim_due_ambient_responsibilities(
    p_limit INT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    lim INT := GREATEST(COALESCE(p_limit, get_config_int('ambient.batch_size'), 20), 1);
    claim_timeout_s INT := GREATEST(COALESCE(get_config_int('ambient.claim_timeout_seconds'), 300), 30);
    now_ts TIMESTAMPTZ := CURRENT_TIMESTAMP;
    row_resp ambient_responsibilities%ROWTYPE;
    run_id UUID;
    result JSONB := '[]'::jsonb;
BEGIN
    IF NOT COALESCE(get_config_bool('ambient.enabled'), TRUE) THEN
        RETURN '[]'::jsonb;
    END IF;

    FOR row_resp IN
        SELECT *
        FROM ambient_responsibilities
        WHERE status = 'active'
          AND next_check_at IS NOT NULL
          AND next_check_at <= now_ts
          AND (cooldown_until IS NULL OR cooldown_until <= now_ts)
          AND (expires_at IS NULL OR expires_at > now_ts)
        ORDER BY
          CASE priority WHEN 'urgent' THEN 0 WHEN 'high' THEN 1 WHEN 'normal' THEN 2 ELSE 3 END,
          next_check_at ASC,
          created_at ASC
        LIMIT lim
        FOR UPDATE SKIP LOCKED
    LOOP
        INSERT INTO ambient_responsibility_runs (
            responsibility_id, status, due_at, trigger_snapshot
        )
        VALUES (
            row_resp.id,
            'checking',
            row_resp.next_check_at,
            jsonb_build_object(
                'trigger', row_resp.trigger,
                'evaluator', row_resp.evaluator,
                'sources', row_resp.sources,
                'claimed_at', now_ts
            )
        )
        RETURNING id INTO run_id;

        UPDATE ambient_responsibilities
        SET last_run_id = run_id,
            next_check_at = now_ts + make_interval(secs => claim_timeout_s),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = row_resp.id;

        result := result || jsonb_build_array(jsonb_build_object(
            'run_id', run_id::text,
            'due_at', row_resp.next_check_at,
            'responsibility', jsonb_build_object(
                'id', row_resp.id::text,
                'title', row_resp.title,
                'description', row_resp.description,
                'kind', row_resp.kind,
                'status', row_resp.status,
                'priority', row_resp.priority,
                'user_intent', row_resp.user_intent,
                'trigger', row_resp.trigger,
                'evaluator', row_resp.evaluator,
                'sources', row_resp.sources,
                'actions', row_resp.actions,
                'delivery', row_resp.delivery,
                'memory_policy', row_resp.memory_policy,
                'timezone', row_resp.timezone,
                'last_checked_at', row_resp.last_checked_at,
                'last_fired_at', row_resp.last_fired_at,
                'created_at', row_resp.created_at
            )
        ));
    END LOOP;

    UPDATE ambient_responsibilities
    SET status = 'expired',
        next_check_at = NULL,
        updated_at = CURRENT_TIMESTAMP
    WHERE status = 'active'
      AND expires_at IS NOT NULL
      AND expires_at <= now_ts;

    RETURN result;
END;
$$;

CREATE OR REPLACE FUNCTION complete_ambient_responsibility_run(
    p_run_id UUID,
    p_status TEXT,
    p_decision JSONB DEFAULT '{}'::jsonb,
    p_observations JSONB DEFAULT '[]'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    run_row ambient_responsibility_runs%ROWTYPE;
    row_resp ambient_responsibilities%ROWTYPE;
    normalized_status TEXT := lower(COALESCE(NULLIF(p_status, ''), 'silent'));
    decision_doc JSONB := COALESCE(p_decision, '{}'::jsonb);
    observations_doc JSONB := CASE WHEN jsonb_typeof(COALESCE(p_observations, '[]'::jsonb)) = 'array'
                                   THEN COALESCE(p_observations, '[]'::jsonb)
                                   ELSE '[]'::jsonb END;
    now_ts TIMESTAMPTZ := CURRENT_TIMESTAMP;
    next_check TIMESTAMPTZ;
    retry_base INT := GREATEST(COALESCE(get_config_int('ambient.failure_retry_base_seconds'), 60), 5);
    notify_message TEXT;
    outbox_doc JSONB := '[]'::jsonb;
    delivery_doc JSONB;
    resp_status TEXT;
    last_obs UUID := NULL;
BEGIN
    IF normalized_status NOT IN ('silent', 'fired', 'blocked', 'failed', 'skipped') THEN
        normalized_status := 'failed';
        decision_doc := decision_doc || jsonb_build_object('error', format('Invalid ambient run status %L', p_status));
    END IF;

    SELECT * INTO run_row
    FROM ambient_responsibility_runs
    WHERE id = p_run_id
    FOR UPDATE;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('success', false, 'error', format('Ambient run %s not found', p_run_id), 'error_type', 'invalid_params');
    END IF;

    SELECT * INTO row_resp
    FROM ambient_responsibilities
    WHERE id = run_row.responsibility_id
    FOR UPDATE;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('success', false, 'error', 'Ambient responsibility missing for run', 'error_type', 'execution_failed');
    END IF;

    notify_message := NULLIF(btrim(COALESCE(decision_doc->>'notify_message', '')), '');
    delivery_doc := COALESCE(row_resp.delivery, '{"mode":"outbox"}'::jsonb);
    IF normalized_status = 'fired'
       AND notify_message IS NOT NULL
       AND COALESCE(delivery_doc->>'mode', 'outbox') <> 'silent' THEN
        outbox_doc := outbox_doc || jsonb_build_array(
            build_user_message(
                notify_message,
                COALESCE(NULLIF(decision_doc->>'intent', ''), 'ambient_responsibility'),
                jsonb_build_object(
                    'responsibility_id', row_resp.id::text,
                    'responsibility_title', row_resp.title,
                    'run_id', p_run_id::text,
                    'decision', decision_doc
                )
            ) || jsonb_build_object('delivery', delivery_doc, 'task_name', row_resp.title)
        );
    END IF;

    UPDATE ambient_responsibility_runs
    SET status = normalized_status,
        finished_at = now_ts,
        observations = observations_doc,
        decision = decision_doc,
        outbox_messages = outbox_doc,
        error = NULLIF(decision_doc->>'error', '')
    WHERE id = p_run_id
    RETURNING * INTO run_row;

    IF jsonb_array_length(observations_doc) > 0 AND NULLIF(observations_doc->0->>'observation_id', '') IS NOT NULL THEN
        last_obs := (observations_doc->0->>'observation_id')::uuid;
    END IF;

    IF normalized_status IN ('silent', 'fired', 'skipped') THEN
        next_check := ambient_compute_next_check(row_resp.trigger, row_resp.timezone, now_ts);
        resp_status := CASE WHEN row_resp.expires_at IS NOT NULL AND row_resp.expires_at <= now_ts
                            THEN 'expired'
                            ELSE 'active' END;
    ELSIF normalized_status = 'blocked' THEN
        next_check := NULL;
        resp_status := 'blocked';
    ELSE
        next_check := now_ts + make_interval(secs => LEAST(retry_base * power(2, LEAST(row_resp.consecutive_errors, 6))::int, 3600));
        resp_status := 'active';
    END IF;

    UPDATE ambient_responsibilities
    SET status = resp_status,
        next_check_at = next_check,
        last_checked_at = now_ts,
        last_fired_at = CASE WHEN normalized_status = 'fired' THEN now_ts ELSE last_fired_at END,
        last_observation_id = COALESCE(last_obs, last_observation_id),
        consecutive_errors = CASE WHEN normalized_status = 'failed' THEN consecutive_errors + 1 ELSE 0 END,
        consecutive_silent = CASE
            WHEN normalized_status = 'silent' THEN consecutive_silent + 1
            WHEN normalized_status = 'fired' THEN 0
            ELSE consecutive_silent
        END,
        cooldown_until = CASE WHEN normalized_status = 'failed' THEN next_check ELSE NULL END,
        last_error = CASE WHEN normalized_status = 'failed' THEN COALESCE(decision_doc->>'error', 'ambient check failed') ELSE NULL END,
        metadata = metadata || jsonb_build_object('last_decision', decision_doc),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = row_resp.id
    RETURNING * INTO row_resp;

    RETURN jsonb_build_object(
        'success', true,
        'run_id', run_row.id::text,
        'responsibility_id', row_resp.id::text,
        'status', run_row.status,
        'responsibility_status', row_resp.status,
        'next_check_at', row_resp.next_check_at,
        'outbox_messages', outbox_doc,
        'observations', observations_doc,
        'decision', decision_doc
    );
END;
$$;

CREATE OR REPLACE FUNCTION ambient_responsibility_status()
RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    RETURN jsonb_build_object(
        'enabled', COALESCE(get_config_bool('ambient.enabled'), TRUE),
        'active', (SELECT COUNT(*) FROM ambient_responsibilities WHERE status = 'active'),
        'blocked', (SELECT COUNT(*) FROM ambient_responsibilities WHERE status = 'blocked'),
        'paused', (SELECT COUNT(*) FROM ambient_responsibilities WHERE status = 'paused'),
        'due_now', (SELECT COUNT(*) FROM ambient_responsibilities WHERE status = 'active' AND next_check_at <= CURRENT_TIMESTAMP),
        'latest_runs', COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
                'run_id', r.id::text,
                'responsibility_id', r.responsibility_id::text,
                'title', ar.title,
                'status', r.status,
                'started_at', r.started_at,
                'finished_at', r.finished_at,
                'decision', r.decision
            ) ORDER BY r.started_at DESC)
            FROM (
                SELECT *
                FROM ambient_responsibility_runs
                ORDER BY started_at DESC
                LIMIT 10
            ) r
            JOIN ambient_responsibilities ar ON ar.id = r.responsibility_id
        ), '[]'::jsonb)
    );
END;
$$;

UPDATE prompt_modules
SET content = replace(
    content,
    '- When asked to carry something forward ("next time, tell them...", "remind me about..."): `remember` the errand or `schedule` it with `manage_schedule` — a promise to carry a message is a commitment, and commitments live in memory, not in hope.',
    '- When asked to carry something forward, choose the right durable substrate before replying. Use `manage_schedule` for explicit one-shot or recurring timed reminders. Use `manage_responsibility` for ambient responsibilities: condition monitors, "let me know whenever...", recurring check-ins, "tell me if...", or anything that requires observing a source over time. A promise to watch, remind, or report is a commitment; store it before claiming it is handled.'
)
WHERE key = 'conversation'
  AND content LIKE '%When asked to carry something forward%manage_schedule%';

SET check_function_bodies = on;
