-- Ambient responsibility polish after live verification.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

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

UPDATE prompt_modules
SET content = replace(
    content,
    '- When asked to carry something forward, choose the right durable substrate before replying. Use `manage_schedule` for explicit one-shot or recurring timed reminders. Use `manage_responsibility` for ambient responsibilities: condition monitors, "let me know whenever...", recurring check-ins, "tell me if...", or anything that requires observing a source over time. A promise to watch, remind, or report is a commitment; store it before claiming it is handled.',
    '- When asked to carry something forward, choose the right durable substrate before replying. Use `manage_schedule` for explicit one-shot timed reminders. Use `manage_responsibility` for ambient responsibilities: "let me know whenever Hope emails me", "watch Slack for anything urgent", "remind me to take pills twice daily", "tell me if I have not checked in", "notify me if my steps are low", or anything that observes a source over time. A promise to watch, remind, or report is a commitment; store it before claiming it is handled.
- For ambient responsibilities, translate the user''s words into trigger/evaluator/source/action fields yourself. Gmail monitors use `sources:[{"connector_id":"gmail","query":"..."}]`; important-only monitors use an importance evaluator; check-ins use `kind:"checkin"` plus a missing-checkin evaluator; wearable/health thresholds use `kind:"threshold"` with the relevant metric. Ask a short clarifying question only if the trigger, source, or action is genuinely missing.'
)
WHERE key = 'conversation'
  AND content LIKE '%Use `manage_responsibility` for ambient responsibilities%';

SET check_function_bodies = on;
