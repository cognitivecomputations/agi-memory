-- Finish ambient responsibilities: blocker refresh, detail payloads, and safer tool status.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

INSERT INTO config_defaults (key, value, description) VALUES
    ('ambient.generic_source_page_size', '25'::jsonb, 'Connector source items checked per generic ambient monitor run'),
    ('ambient.importance_llm_enabled', 'true'::jsonb, 'Use the connector importance LLM detector for ambient importance monitors when available')
ON CONFLICT (key) DO NOTHING;

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
        connector := replace(lower(NULLIF(btrim(COALESCE(source_doc->>'connector_id', source_doc->>'connector', '')), '')), '-', '_');
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

CREATE OR REPLACE FUNCTION refresh_ambient_responsibility_blockers()
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    row_resp ambient_responsibilities%ROWTYPE;
    missing JSONB;
    next_check TIMESTAMPTZ;
    blocked_count INT := 0;
    unblocked_count INT := 0;
BEGIN
    FOR row_resp IN
        SELECT *
        FROM ambient_responsibilities
        WHERE status IN ('active', 'blocked')
        FOR UPDATE SKIP LOCKED
    LOOP
        missing := ambient_missing_connectors(row_resp.sources);
        IF row_resp.status = 'active' AND jsonb_array_length(missing) > 0 THEN
            UPDATE ambient_responsibilities
            SET status = 'blocked',
                next_check_at = NULL,
                metadata = metadata || jsonb_build_object('missing_connectors', missing),
                last_error = 'missing connector setup',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = row_resp.id;
            blocked_count := blocked_count + 1;
        ELSIF row_resp.status = 'blocked' AND jsonb_array_length(missing) = 0 THEN
            next_check := ambient_compute_next_check(row_resp.trigger, row_resp.timezone, CURRENT_TIMESTAMP - INTERVAL '1 second');
            UPDATE ambient_responsibilities
            SET status = 'active',
                next_check_at = next_check,
                cooldown_until = NULL,
                last_error = NULL,
                metadata = metadata || jsonb_build_object('missing_connectors', '[]'::jsonb),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = row_resp.id;
            unblocked_count := unblocked_count + 1;
        ELSIF jsonb_array_length(missing) = 0 THEN
            UPDATE ambient_responsibilities
            SET metadata = metadata || jsonb_build_object('missing_connectors', '[]'::jsonb),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = row_resp.id
              AND COALESCE(metadata->'missing_connectors', 'null'::jsonb) <> '[]'::jsonb;
        END IF;
    END LOOP;

    RETURN jsonb_build_object('blocked', blocked_count, 'unblocked', unblocked_count);
END;
$$;

CREATE OR REPLACE FUNCTION ambient_responsibility_detail(
    p_responsibility_id UUID
) RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    row_resp ambient_responsibilities%ROWTYPE;
    missing JSONB;
    runs_doc JSONB;
    observations_doc JSONB;
    checkins_doc JSONB;
BEGIN
    SELECT *
    INTO row_resp
    FROM ambient_responsibilities
    WHERE id = p_responsibility_id;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('success', false, 'error', format('Ambient responsibility %s not found', p_responsibility_id));
    END IF;

    missing := ambient_missing_connectors(row_resp.sources);

    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'run_id', r.id::text,
        'status', r.status,
        'due_at', r.due_at,
        'started_at', r.started_at,
        'finished_at', r.finished_at,
        'decision', r.decision,
        'observations', r.observations,
        'outbox_messages', r.outbox_messages,
        'error', r.error
    ) ORDER BY r.started_at DESC), '[]'::jsonb)
    INTO runs_doc
    FROM (
        SELECT *
        FROM ambient_responsibility_runs
        WHERE responsibility_id = p_responsibility_id
        ORDER BY started_at DESC
        LIMIT 30
    ) r;

    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'observation_id', o.id::text,
        'connector_id', o.connector_id,
        'account_key', o.account_key,
        'item_kind', o.item_kind,
        'provider_item_id', o.provider_item_id,
        'provider_thread_id', o.provider_thread_id,
        'observed_at', o.observed_at,
        'title', o.title,
        'content_preview', left(COALESCE(o.content, ''), 500),
        'participants', o.participants,
        'labels', o.labels,
        'source_item_id', CASE WHEN o.source_item_id IS NULL THEN NULL ELSE o.source_item_id::text END,
        'source_document_id', CASE WHEN o.source_document_id IS NULL THEN NULL ELSE o.source_document_id::text END,
        'raw', o.raw
    ) ORDER BY o.observed_at DESC), '[]'::jsonb)
    INTO observations_doc
    FROM (
        SELECT *
        FROM ambient_observations
        WHERE responsibility_id = p_responsibility_id
        ORDER BY observed_at DESC
        LIMIT 30
    ) o;

    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'checkin_id', c.id::text,
        'label', c.label,
        'occurred_at', c.occurred_at,
        'note', c.note,
        'source', c.source,
        'metadata', c.metadata
    ) ORDER BY c.occurred_at DESC), '[]'::jsonb)
    INTO checkins_doc
    FROM (
        SELECT *
        FROM ambient_checkins
        WHERE responsibility_id = p_responsibility_id
        ORDER BY occurred_at DESC
        LIMIT 30
    ) c;

    RETURN jsonb_build_object(
        'success', true,
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
            'next_check_at', row_resp.next_check_at,
            'last_checked_at', row_resp.last_checked_at,
            'last_fired_at', row_resp.last_fired_at,
            'last_observation_id', CASE WHEN row_resp.last_observation_id IS NULL THEN NULL ELSE row_resp.last_observation_id::text END,
            'last_run_id', CASE WHEN row_resp.last_run_id IS NULL THEN NULL ELSE row_resp.last_run_id::text END,
            'consecutive_errors', row_resp.consecutive_errors,
            'consecutive_silent', row_resp.consecutive_silent,
            'cooldown_until', row_resp.cooldown_until,
            'expires_at', row_resp.expires_at,
            'created_by', row_resp.created_by,
            'source_session_id', row_resp.source_session_id,
            'metadata', row_resp.metadata,
            'last_error', row_resp.last_error,
            'created_at', row_resp.created_at,
            'updated_at', row_resp.updated_at,
            'missing_connectors', missing
        ),
        'latest_runs', runs_doc,
        'latest_observations', observations_doc,
        'latest_checkins', checkins_doc
    );
END;
$$;

CREATE OR REPLACE FUNCTION manage_ambient_responsibility_tool(
    p_args JSONB
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    action TEXT := lower(COALESCE(p_args->>'action', ''));
    rid UUID;
    missing JSONB;
    row_resp ambient_responsibilities%ROWTYPE;
    rows_doc JSONB;
    next_check TIMESTAMPTZ;
    detail_doc JSONB;
BEGIN
    IF action NOT IN ('create', 'list', 'pause', 'resume', 'cancel', 'status', 'checkin', 'evaluate_now') THEN
        RETURN jsonb_build_object('success', false, 'error', format('Invalid action %L', action), 'error_type', 'invalid_params');
    END IF;

    PERFORM refresh_ambient_responsibility_blockers();

    IF action = 'create' THEN
        RETURN create_ambient_responsibility(p_args);
    END IF;

    IF action = 'list' THEN
        rows_doc := list_ambient_responsibilities(NULLIF(p_args->>'status', ''), COALESCE(NULLIF(p_args->>'limit', '')::int, 50));
        RETURN jsonb_build_object('success', true, 'output', jsonb_build_object(
            'responsibilities', rows_doc,
            'count', jsonb_array_length(rows_doc),
            'status', ambient_responsibility_status()
        ), 'display_output', format('Found %s ambient responsibility(s)', jsonb_array_length(rows_doc)));
    END IF;

    rid := ambient_responsibility_id_from_args(p_args);
    IF rid IS NULL THEN
        RETURN jsonb_build_object('success', false, 'error', 'responsibility_id or title is required', 'error_type', 'invalid_params');
    END IF;

    IF action = 'status' THEN
        detail_doc := ambient_responsibility_detail(rid);
        IF NOT COALESCE((detail_doc->>'success')::boolean, false) THEN
            RETURN jsonb_build_object('success', false, 'error', detail_doc->>'error', 'error_type', 'invalid_params');
        END IF;
        RETURN jsonb_build_object('success', true, 'output', detail_doc);
    ELSIF action = 'pause' THEN
        UPDATE ambient_responsibilities
        SET status = 'paused',
            next_check_at = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = rid
        RETURNING * INTO row_resp;
        IF NOT FOUND THEN
            RETURN jsonb_build_object('success', false, 'error', format('Ambient responsibility %s not found', rid), 'error_type', 'invalid_params');
        END IF;
        RETURN jsonb_build_object('success', true, 'output', ambient_responsibility_detail(rid), 'display_output', format('Paused ambient responsibility: %s', row_resp.title));
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
        RETURN jsonb_build_object('success', true, 'output', ambient_responsibility_detail(rid), 'display_output', format('Resumed ambient responsibility: %s (%s)', row_resp.title, row_resp.status));
    ELSIF action = 'cancel' THEN
        UPDATE ambient_responsibilities
        SET status = 'disabled',
            next_check_at = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = rid
        RETURNING * INTO row_resp;
        IF NOT FOUND THEN
            RETURN jsonb_build_object('success', false, 'error', format('Ambient responsibility %s not found', rid), 'error_type', 'invalid_params');
        END IF;
        RETURN jsonb_build_object('success', true, 'output', ambient_responsibility_detail(rid), 'display_output', format('Cancelled ambient responsibility: %s', row_resp.title));
    ELSIF action = 'evaluate_now' THEN
        SELECT * INTO row_resp FROM ambient_responsibilities WHERE id = rid FOR UPDATE;
        IF NOT FOUND THEN
            RETURN jsonb_build_object('success', false, 'error', format('Ambient responsibility %s not found', rid), 'error_type', 'invalid_params');
        END IF;
        missing := ambient_missing_connectors(row_resp.sources);
        IF jsonb_array_length(missing) = 0 THEN
            UPDATE ambient_responsibilities
            SET status = 'active',
                next_check_at = CURRENT_TIMESTAMP,
                cooldown_until = NULL,
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
        RETURN jsonb_build_object('success', true, 'output', ambient_responsibility_detail(rid), 'display_output', format('Queued ambient check: %s', row_resp.title));
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
        IF NOT FOUND THEN
            RETURN jsonb_build_object('success', false, 'error', format('Ambient responsibility %s not found', rid), 'error_type', 'invalid_params');
        END IF;
        RETURN jsonb_build_object('success', true, 'output', ambient_responsibility_detail(rid), 'display_output', format('Recorded check-in: %s', row_resp.title));
    END IF;
EXCEPTION WHEN OTHERS THEN
    RETURN jsonb_build_object('success', false, 'error', SQLERRM, 'error_type', 'execution_failed');
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

    PERFORM refresh_ambient_responsibility_blockers();

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
        'disabled', (SELECT COUNT(*) FROM ambient_responsibilities WHERE status = 'disabled'),
        'due_now', (SELECT COUNT(*) FROM ambient_responsibilities WHERE status = 'active' AND next_check_at <= CURRENT_TIMESTAMP),
        'needs_setup', (
            SELECT COUNT(*)
            FROM ambient_responsibilities
            WHERE status = 'blocked'
              AND jsonb_array_length(ambient_missing_connectors(sources)) > 0
        ),
        'next_due_at', (
            SELECT MIN(next_check_at)
            FROM ambient_responsibilities
            WHERE status = 'active'
              AND next_check_at IS NOT NULL
        ),
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

SET check_function_bodies = on;
