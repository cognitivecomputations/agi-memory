-- DB-owned tool catalog, policy, workflow bookkeeping, and schedule parsing.
SET search_path = public, ag_catalog, "$user";

CREATE OR REPLACE FUNCTION upsert_tool_definition(
    p_name TEXT,
    p_category TEXT,
    p_schema JSONB DEFAULT '{}'::jsonb,
    p_description TEXT DEFAULT '',
    p_energy_cost INT DEFAULT 1,
    p_allowed_contexts TEXT[] DEFAULT ARRAY[]::TEXT[],
    p_requires_approval BOOLEAN DEFAULT FALSE,
    p_supports_parallel BOOLEAN DEFAULT TRUE,
    p_optional BOOLEAN DEFAULT FALSE,
    p_read_only BOOLEAN DEFAULT TRUE,
    p_execution_kind TEXT DEFAULT 'python_driver',
    p_driver TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    meta JSONB;
BEGIN
    IF NULLIF(btrim(p_name), '') IS NULL THEN
        RAISE EXCEPTION 'tool name is required';
    END IF;
    IF NULLIF(btrim(p_category), '') IS NULL THEN
        RAISE EXCEPTION 'tool category is required';
    END IF;

    meta := COALESCE(p_metadata, '{}'::jsonb)
        || jsonb_build_object(
            'description', COALESCE(p_description, ''),
            'optional', COALESCE(p_optional, false),
            'is_read_only', COALESCE(p_read_only, true)
        );

    INSERT INTO tool_definitions (
        name, category, schema, default_energy_cost, allowed_contexts,
        requires_approval, supports_parallel, execution_kind, driver, metadata, updated_at
    )
    VALUES (
        p_name,
        p_category,
        COALESCE(p_schema, '{}'::jsonb),
        GREATEST(COALESCE(p_energy_cost, 1), 0),
        COALESCE(p_allowed_contexts, ARRAY[]::TEXT[]),
        COALESCE(p_requires_approval, false),
        COALESCE(p_supports_parallel, true),
        COALESCE(NULLIF(p_execution_kind, ''), 'python_driver'),
        p_driver,
        meta,
        CURRENT_TIMESTAMP
    )
    ON CONFLICT (name) DO UPDATE SET
        category = EXCLUDED.category,
        schema = EXCLUDED.schema,
        default_energy_cost = EXCLUDED.default_energy_cost,
        allowed_contexts = EXCLUDED.allowed_contexts,
        requires_approval = EXCLUDED.requires_approval,
        supports_parallel = EXCLUDED.supports_parallel,
        execution_kind = EXCLUDED.execution_kind,
        driver = EXCLUDED.driver,
        metadata = EXCLUDED.metadata,
        updated_at = CURRENT_TIMESTAMP;

    RETURN jsonb_build_object('name', p_name, 'status', 'upserted');
END;
$$;

CREATE OR REPLACE FUNCTION sync_tool_definitions(
    p_tools JSONB
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    tool JSONB;
    count_synced INT := 0;
BEGIN
    IF jsonb_typeof(COALESCE(p_tools, '[]'::jsonb)) <> 'array' THEN
        RAISE EXCEPTION 'p_tools must be an array';
    END IF;

    FOR tool IN SELECT * FROM jsonb_array_elements(COALESCE(p_tools, '[]'::jsonb))
    LOOP
        PERFORM upsert_tool_definition(
            tool->>'name',
            tool->>'category',
            COALESCE(tool->'schema', '{}'::jsonb),
            COALESCE(tool->>'description', ''),
            COALESCE(NULLIF(tool->>'energy_cost', '')::int, 1),
            COALESCE(ARRAY(SELECT jsonb_array_elements_text(COALESCE(tool->'allowed_contexts', '[]'::jsonb))), ARRAY[]::TEXT[]),
            COALESCE((tool->>'requires_approval')::boolean, false),
            COALESCE((tool->>'supports_parallel')::boolean, true),
            COALESCE((tool->>'optional')::boolean, false),
            COALESCE((tool->>'is_read_only')::boolean, true),
            COALESCE(tool->>'execution_kind', 'python_driver'),
            tool->>'driver',
            COALESCE(tool->'metadata', '{}'::jsonb)
        );
        count_synced := count_synced + 1;
    END LOOP;

    RETURN jsonb_build_object('synced', count_synced);
END;
$$;

CREATE OR REPLACE FUNCTION tool_config_enabled(
    p_tool_name TEXT,
    p_category TEXT,
    p_context TEXT,
    p_optional BOOLEAN DEFAULT FALSE
) RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    cfg JSONB := COALESCE(get_config('tools'), '{}'::jsonb);
    ctx_cfg JSONB := '{}'::jsonb;
    enabled JSONB;
    disabled JSONB;
    disabled_categories JSONB;
    ctx_enabled JSONB;
    ctx_disabled JSONB;
    allowed_optional JSONB;
    allowed_optional_groups JSONB;
BEGIN
    enabled := cfg->'enabled';
    disabled := COALESCE(cfg->'disabled', '[]'::jsonb);
    disabled_categories := COALESCE(cfg->'disabled_categories', '[]'::jsonb);
    ctx_cfg := COALESCE(cfg #> ARRAY['context_overrides', p_context], '{}'::jsonb);
    ctx_enabled := COALESCE(ctx_cfg->'enabled', '[]'::jsonb);
    ctx_disabled := COALESCE(ctx_cfg->'disabled', '[]'::jsonb);

    IF disabled ? p_tool_name OR disabled_categories ? p_category THEN
        RETURN FALSE;
    END IF;
    IF enabled IS NOT NULL AND jsonb_typeof(enabled) = 'array' AND NOT (enabled ? p_tool_name) THEN
        RETURN FALSE;
    END IF;
    IF ctx_disabled ? p_tool_name THEN
        RETURN FALSE;
    END IF;
    IF COALESCE((ctx_cfg->>'allow_all')::boolean, false) THEN
        RETURN TRUE;
    END IF;
    IF jsonb_typeof(ctx_enabled) = 'array' AND jsonb_array_length(ctx_enabled) > 0 AND NOT (ctx_enabled ? p_tool_name) THEN
        RETURN FALSE;
    END IF;

    IF COALESCE(p_optional, false) THEN
        allowed_optional := COALESCE(cfg->'allowed_optional', '[]'::jsonb);
        allowed_optional_groups := COALESCE(cfg->'allowed_optional_groups', '[]'::jsonb);
        IF NOT (allowed_optional ? p_tool_name OR allowed_optional_groups ? p_category OR allowed_optional_groups ? 'plugins') THEN
            RETURN FALSE;
        END IF;
    END IF;

    RETURN TRUE;
END;
$$;

CREATE OR REPLACE FUNCTION get_tool_specs_for_context(
    p_context TEXT
) RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    specs JSONB;
BEGIN
    SELECT COALESCE(jsonb_agg(
        jsonb_build_object(
            'type', 'function',
            'function', jsonb_build_object(
                'name', name,
                'description', COALESCE(metadata->>'description', ''),
                'parameters', schema
            )
        )
        ORDER BY name
    ), '[]'::jsonb)
    INTO specs
    FROM tool_definitions
    WHERE (COALESCE(array_length(allowed_contexts, 1), 0) = 0 OR lower(p_context) = ANY(allowed_contexts))
      AND tool_config_enabled(name, category, lower(p_context), COALESCE((metadata->>'optional')::boolean, false));

    RETURN specs;
END;
$$;

CREATE OR REPLACE FUNCTION evaluate_tool_call(
    p_tool_name TEXT,
    p_arguments JSONB DEFAULT '{}'::jsonb,
    p_context JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    tool tool_definitions%ROWTYPE;
    ctx TEXT := lower(COALESCE(p_context->>'tool_context', p_context->>'context', 'chat'));
    energy_available INT;
    cfg JSONB := COALESCE(get_config('tools'), '{}'::jsonb);
    ctx_cfg JSONB;
    cost INT;
    max_per_tool INT;
    boundary TEXT;
    approval_ok BOOLEAN := FALSE;
    approval_request UUID;
    purpose_goal_id UUID;
    assigned_energy_multiplier DOUBLE PRECISION;
BEGIN
    SELECT * INTO tool FROM tool_definitions WHERE name = p_tool_name;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('allowed', false, 'reason', 'Unknown tool: ' || p_tool_name, 'error_type', 'unknown_tool');
    END IF;

    IF NOT tool_config_enabled(tool.name, tool.category, ctx, COALESCE((tool.metadata->>'optional')::boolean, false)) THEN
        RETURN jsonb_build_object('allowed', false, 'reason', format('Tool %L is disabled', tool.name), 'error_type', 'disabled');
    END IF;
    IF COALESCE(array_length(tool.allowed_contexts, 1), 0) > 0 AND NOT (ctx = ANY(tool.allowed_contexts)) THEN
        RETURN jsonb_build_object('allowed', false, 'reason', format('Tool %L not allowed in %s context', tool.name, ctx), 'error_type', 'context_denied');
    END IF;

    cost := COALESCE(NULLIF(cfg #>> ARRAY['costs', tool.name], '')::int, tool.default_energy_cost);
    -- A user-assigned goal is the durable permission slip for autonomous
    -- effort.  Only declared outbound tools can use this path; adding model
    -- arguments to an unrelated tool cannot manufacture a discount.
    IF tool.metadata ? 'outbound'
       AND lower(COALESCE(p_arguments->>'purpose_kind', '')) = 'goal' THEN
        BEGIN
            purpose_goal_id := NULLIF(p_arguments->>'purpose_reference', '')::uuid;
        EXCEPTION WHEN OTHERS THEN
            purpose_goal_id := NULL;
        END;
        IF purpose_goal_id IS NOT NULL AND EXISTS (
            SELECT 1 FROM memories
            WHERE id = purpose_goal_id
              AND type = 'goal'
              AND goal_origin = 'user_request'::goal_source
        ) THEN
            assigned_energy_multiplier := LEAST(
                GREATEST(
                    COALESCE(get_config_float('outbound.assigned_goal_energy_multiplier'), 0.25),
                    0
                ),
                1
            );
            cost := CEIL(cost * assigned_energy_multiplier)::integer;
        END IF;
    END IF;
    IF ctx = 'heartbeat' AND p_context ? 'energy_available' THEN
        BEGIN energy_available := NULLIF(p_context->>'energy_available', '')::int;
        EXCEPTION WHEN OTHERS THEN energy_available := NULL; END;
        ctx_cfg := COALESCE(cfg #> '{context_overrides,heartbeat}', '{}'::jsonb);
        BEGIN max_per_tool := NULLIF(ctx_cfg->>'max_energy_per_tool', '')::int;
        EXCEPTION WHEN OTHERS THEN max_per_tool := NULL; END;
        IF max_per_tool IS NOT NULL AND cost > max_per_tool THEN
            RETURN jsonb_build_object('allowed', false, 'reason', format('Tool %L cost (%s) exceeds max per tool (%s)', tool.name, cost, max_per_tool), 'error_type', 'insufficient_energy', 'energy_cost', cost);
        END IF;
        IF energy_available IS NOT NULL AND cost > energy_available THEN
            RETURN jsonb_build_object('allowed', false, 'reason', format('Insufficient energy: need %s, have %s', cost, energy_available), 'error_type', 'insufficient_energy', 'energy_cost', cost);
        END IF;
    END IF;

    boundary := tool_boundary_violation(tool.name, tool.category);
    IF boundary IS NOT NULL THEN
        RETURN jsonb_build_object('allowed', false, 'reason', 'Boundary restriction: ' || boundary, 'error_type', 'boundary_violation', 'energy_cost', cost);
    END IF;

    IF tool.requires_approval AND NULLIF(p_context->>'approval_request_id', '') IS NOT NULL THEN
        BEGIN
            approval_request := (p_context->>'approval_request_id')::uuid;
            approval_ok := consume_operator_tool_approval(
                approval_request, tool.name, p_arguments, p_context
            );
        EXCEPTION WHEN OTHERS THEN
            approval_ok := FALSE;
        END;
    END IF;

    IF tool.requires_approval
       AND NULLIF(p_context->>'approval_request_id', '') IS NOT NULL
       AND NOT approval_ok THEN
        RETURN jsonb_build_object(
            'allowed', false,
            'reason', format('The operator approval proof for tool %L is invalid, expired, mismatched, or already consumed', tool.name),
            'error_type', 'approval_required',
            'energy_cost', cost
        );
    END IF;

    IF tool.requires_approval
       AND ctx <> 'chat'
       AND NOT is_tool_approved(tool.name)
       AND NOT approval_ok THEN
        RETURN jsonb_build_object('allowed', false, 'reason', format('Tool %L requires approval for autonomous use', tool.name), 'error_type', 'approval_required', 'energy_cost', cost);
    END IF;

    RETURN jsonb_build_object(
        'allowed', true,
        'energy_cost', cost,
        'supports_parallel', tool.supports_parallel,
        'execution_kind', tool.execution_kind,
        'driver', tool.driver,
        'operator_approval_request_id', CASE WHEN approval_ok THEN approval_request::text ELSE NULL END
    );
END;
$$;

CREATE OR REPLACE FUNCTION plan_tool_batch(
    p_calls JSONB,
    p_context JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    call JSONB;
    idx INT := 0;
    planned JSONB := '[]'::jsonb;
    decision JSONB;
    remaining INT;
    ctx JSONB;
BEGIN
    BEGIN remaining := NULLIF(p_context->>'energy_available', '')::int;
    EXCEPTION WHEN OTHERS THEN remaining := NULL; END;

    FOR call IN SELECT * FROM jsonb_array_elements(COALESCE(p_calls, '[]'::jsonb))
    LOOP
        ctx := p_context;
        IF remaining IS NOT NULL THEN
            ctx := jsonb_set(ctx, '{energy_available}', to_jsonb(remaining), true);
        END IF;
        decision := evaluate_tool_call(call->>'name', COALESCE(call->'arguments', '{}'::jsonb), ctx);
        IF COALESCE((decision->>'allowed')::boolean, false) AND remaining IS NOT NULL THEN
            remaining := GREATEST(0, remaining - COALESCE((decision->>'energy_cost')::int, 0));
        END IF;
        planned := planned || jsonb_build_array(call || jsonb_build_object('index', idx, 'policy', decision));
        idx := idx + 1;
    END LOOP;
    RETURN jsonb_build_object('calls', planned, 'remaining_energy', remaining);
END;
$$;

CREATE OR REPLACE FUNCTION db_brain_is_cron_expression(
    p_value TEXT
) RETURNS BOOLEAN
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT array_length(regexp_split_to_array(btrim(COALESCE(p_value, '')), '\s+'), 1) IN (5, 6)
       AND NOT EXISTS (
           SELECT 1
           FROM unnest(regexp_split_to_array(btrim(COALESCE(p_value, '')), '\s+')) field
           WHERE field !~ '^[0-9*/,\-?LW#]+$'
       );
$$;

CREATE OR REPLACE FUNCTION parse_schedule_input(
    p_input JSONB
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    schedule_str TEXT := NULLIF(btrim(COALESCE(p_input->>'schedule', '')), '');
    schedule_kind TEXT := NULLIF(lower(btrim(COALESCE(p_input->>'schedule_kind', ''))), '');
    timezone_value TEXT := normalize_timezone(COALESCE(p_input->>'timezone', 'UTC'));
    parts TEXT[];
    schedule JSONB := '{}'::jsonb;
    offset_text TEXT;
    offset_value INT;
    offset_unit TEXT;
    run_at TIMESTAMPTZ;
BEGIN
    IF schedule_str IS NOT NULL THEN
        IF db_brain_is_cron_expression(schedule_str) THEN
            schedule_kind := 'cron';
            schedule := jsonb_build_object('cron', schedule_str);
        ELSIF schedule_str LIKE '{%' THEN
            schedule := schedule_str::jsonb;
        ELSIF position(':' in schedule_str) > 0 THEN
            parts := string_to_array(schedule_str, ':');
            CASE lower(parts[1])
                WHEN 'once' THEN
                    offset_text := regexp_replace(COALESCE(parts[2], ''), '^\+', '');
                    IF offset_text !~ '^\d+[hmd]$' THEN
                        RAISE EXCEPTION 'Invalid offset format: %', offset_text;
                    END IF;
                    offset_value := left(offset_text, length(offset_text) - 1)::int;
                    offset_unit := right(offset_text, 1);
                    run_at := CURRENT_TIMESTAMP
                        + CASE offset_unit
                            WHEN 'h' THEN offset_value * INTERVAL '1 hour'
                            WHEN 'm' THEN offset_value * INTERVAL '1 minute'
                            WHEN 'd' THEN offset_value * INTERVAL '1 day'
                          END;
                    schedule_kind := 'once';
                    schedule := jsonb_build_object('run_at', run_at);
                WHEN 'daily' THEN
                    schedule_kind := 'daily';
                    schedule := jsonb_build_object('time', parts[2] || ':' || COALESCE(parts[3], '00'));
                WHEN 'weekly' THEN
                    schedule_kind := 'weekly';
                    schedule := jsonb_build_object('weekday', parts[2], 'time', parts[3] || ':' || COALESCE(parts[4], '00'));
                WHEN 'every' THEN
                    offset_text := COALESCE(parts[2], '');
                    IF offset_text !~ '^\d+[hms]$' THEN
                        RAISE EXCEPTION 'Invalid interval format: %', offset_text;
                    END IF;
                    schedule_kind := 'interval';
                    offset_value := left(offset_text, length(offset_text) - 1)::int;
                    offset_unit := right(offset_text, 1);
                    schedule := CASE offset_unit
                        WHEN 'h' THEN jsonb_build_object('every_hours', offset_value)
                        WHEN 'm' THEN jsonb_build_object('every_minutes', offset_value)
                        ELSE jsonb_build_object('every_seconds', offset_value)
                    END;
                ELSE
                    IF length(schedule_str) <= 5 THEN
                        schedule_kind := COALESCE(schedule_kind, 'daily');
                        schedule := jsonb_build_object('time', schedule_str);
                    ELSE
                        RAISE EXCEPTION 'Could not parse schedule: %', schedule_str;
                    END IF;
            END CASE;
        ELSIF length(schedule_str) <= 5 THEN
            schedule_kind := COALESCE(schedule_kind, 'daily');
            schedule := jsonb_build_object('time', schedule_str);
        ELSE
            RAISE EXCEPTION 'Could not parse schedule: %', schedule_str;
        END IF;
    END IF;

    IF schedule_kind IS NULL THEN
        RAISE EXCEPTION 'schedule_kind is required';
    END IF;

    IF schedule_kind = 'cron' THEN
        IF NULLIF(schedule->>'cron', '') IS NULL THEN
            RAISE EXCEPTION 'cron schedule requires a "cron" expression';
        END IF;
        -- cron_next_fire validates the expression (raises on bad syntax,
        -- out-of-range values, or a never-firing schedule).
        schedule := schedule || jsonb_build_object(
            '_next_run', cron_next_fire(schedule->>'cron', timezone_value, CURRENT_TIMESTAMP)::text);
    END IF;

    IF schedule_kind = 'once' AND schedule ? '_offset' THEN
        offset_text := regexp_replace(schedule->>'_offset', '^\+', '');
        IF offset_text !~ '^\d+[hmd]$' THEN
            RAISE EXCEPTION 'Invalid offset format: %', offset_text;
        END IF;
        offset_value := left(offset_text, length(offset_text) - 1)::int;
        offset_unit := right(offset_text, 1);
        run_at := CURRENT_TIMESTAMP
            + CASE offset_unit
                WHEN 'h' THEN offset_value * INTERVAL '1 hour'
                WHEN 'm' THEN offset_value * INTERVAL '1 minute'
                WHEN 'd' THEN offset_value * INTERVAL '1 day'
              END;
        schedule := (schedule - '_offset') || jsonb_build_object('run_at', run_at);
    END IF;

    RETURN jsonb_build_object(
        'schedule_kind', schedule_kind,
        'schedule', schedule,
        'timezone', timezone_value,
        'next_run_at', compute_next_run_at(schedule_kind, schedule, timezone_value, CURRENT_TIMESTAMP)
    );
END;
$$;

CREATE OR REPLACE FUNCTION build_schedule_delivery(
    p_args JSONB
) RETURNS JSONB
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT CASE COALESCE(NULLIF(p_args->>'delivery_mode', ''), 'outbox')
        WHEN 'channel' THEN jsonb_strip_nulls(jsonb_build_object(
            'mode', 'channel',
            'channel', NULLIF(p_args->>'delivery_channel', ''),
            'target_id', NULLIF(p_args->>'delivery_target_id', ''),
            'topic', NULLIF(p_args->>'delivery_topic', '')
        ))
        WHEN 'webhook' THEN jsonb_strip_nulls(jsonb_build_object(
            'mode', 'webhook',
            'url', NULLIF(p_args->>'delivery_webhook_url', '')
        ))
        WHEN 'silent' THEN '{"mode":"silent"}'::jsonb
        ELSE '{"mode":"outbox"}'::jsonb
    END;
$$;

CREATE OR REPLACE FUNCTION manage_schedule_tool(
    p_args JSONB
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    action TEXT := COALESCE(p_args->>'action', '');
    parsed JSONB;
    delivery JSONB;
    action_kind TEXT;
    action_payload JSONB := '{}'::jsonb;
    task_id UUID;
    row_data JSONB;
    tasks JSONB;
BEGIN
    IF action NOT IN ('create', 'list', 'update', 'cancel', 'stats') THEN
        RETURN jsonb_build_object('success', false, 'error', format('Invalid action %L', action), 'error_type', 'invalid_params');
    END IF;

    IF action = 'create' THEN
        IF NULLIF(btrim(COALESCE(p_args->>'name', '')), '') IS NULL THEN
            RETURN jsonb_build_object('success', false, 'error', 'Name is required for create', 'error_type', 'invalid_params');
        END IF;
        action_kind := COALESCE(NULLIF(p_args->>'action_kind', ''), 'queue_user_message');
        IF action_kind = 'queue_user_message' THEN
            IF NULLIF(btrim(COALESCE(p_args->>'message', '')), '') IS NULL THEN
                RETURN jsonb_build_object('success', false, 'error', 'message is required for queue_user_message action_kind', 'error_type', 'invalid_params');
            END IF;
            action_payload := jsonb_build_object('message', p_args->>'message');
        ELSIF action_kind = 'create_goal' THEN
            action_payload := jsonb_build_object('title', COALESCE(NULLIF(p_args->>'goal_title', ''), p_args->>'name'), 'description', p_args->>'description');
        ELSE
            RETURN jsonb_build_object('success', false, 'error', format('Invalid action_kind %L', action_kind), 'error_type', 'invalid_params');
        END IF;
        delivery := build_schedule_delivery(p_args);
        IF delivery->>'mode' = 'channel' AND NULLIF(delivery->>'target_id', '') IS NULL THEN
            RETURN jsonb_build_object('success', false, 'error', 'delivery_target_id is required when delivery_mode is channel', 'error_type', 'invalid_params');
        END IF;
        IF delivery->>'mode' = 'webhook' AND NULLIF(delivery->>'url', '') IS NULL THEN
            RETURN jsonb_build_object('success', false, 'error', 'delivery_webhook_url is required when delivery_mode is webhook', 'error_type', 'invalid_params');
        END IF;
        parsed := parse_schedule_input(p_args);
        task_id := create_scheduled_task(
            p_args->>'name',
            parsed->>'schedule_kind',
            parsed->'schedule',
            action_kind,
            action_payload,
            parsed->>'timezone',
            p_args->>'description',
            'active',
            COALESCE(NULLIF(p_args->>'max_runs', '')::int, CASE WHEN parsed->>'schedule_kind' = 'once' THEN 1 ELSE NULL END),
            'agent',
            delivery
        );
        RETURN jsonb_build_object('success', true, 'output', jsonb_build_object(
            'task_id', task_id::text,
            'name', p_args->>'name',
            'schedule_kind', parsed->>'schedule_kind',
            'action_kind', action_kind,
            'delivery', delivery
        ), 'display_output', format('Created scheduled task: %s (%s)', p_args->>'name', parsed->>'schedule_kind'));
    ELSIF action = 'list' THEN
        SELECT COALESCE(jsonb_agg(
            jsonb_build_object(
                'id', t.id::text,
                'name', t.name,
                'description', t.description,
                'schedule_kind', t.schedule_kind,
                'status', t.status,
                'next_run_at', t.next_run_at::text,
                'last_run_at', t.last_run_at::text,
                'run_count', COALESCE(t.run_count, 0),
                'action_kind', t.action_kind,
                'last_error', t.last_error
            )
            -- default outbox delivery is implementation detail; only
            -- non-default routing is surfaced
            || CASE WHEN t.delivery IS NOT NULL AND (t.delivery->>'mode') IS DISTINCT FROM 'outbox'
                    THEN jsonb_build_object('delivery', t.delivery)
                    ELSE '{}'::jsonb END
        ), '[]'::jsonb)
        INTO tasks
        FROM list_scheduled_tasks(NULLIF(p_args->>'status', '')) t;
        RETURN jsonb_build_object('success', true, 'output', jsonb_build_object('tasks', tasks, 'count', jsonb_array_length(tasks)), 'display_output', format('Found %s scheduled task(s)', jsonb_array_length(tasks)));
    ELSIF action = 'update' THEN
        task_id := NULLIF(p_args->>'task_id', '')::uuid;
        IF task_id IS NULL THEN
            RETURN jsonb_build_object('success', false, 'error', 'task_id is required for update', 'error_type', 'invalid_params');
        END IF;
        parsed := CASE WHEN p_args ? 'schedule' OR p_args ? 'schedule_kind' THEN parse_schedule_input(p_args) ELSE NULL END;
        delivery := CASE WHEN p_args ? 'delivery_mode' OR p_args ? 'delivery_channel' OR p_args ? 'delivery_target_id' OR p_args ? 'delivery_webhook_url' THEN build_schedule_delivery(p_args) ELSE NULL END;
        IF delivery IS NOT NULL AND delivery->>'mode' = 'channel' AND NULLIF(delivery->>'target_id', '') IS NULL THEN
            RETURN jsonb_build_object('success', false, 'error', 'delivery_target_id is required when delivery_mode is channel', 'error_type', 'invalid_params');
        END IF;
        IF delivery IS NOT NULL AND delivery->>'mode' = 'webhook' AND NULLIF(delivery->>'url', '') IS NULL THEN
            RETURN jsonb_build_object('success', false, 'error', 'delivery_webhook_url is required when delivery_mode is webhook', 'error_type', 'invalid_params');
        END IF;
        action_payload := CASE
            WHEN p_args ? 'message' THEN jsonb_build_object('message', p_args->>'message')
            WHEN p_args ? 'goal_title' THEN jsonb_build_object('title', p_args->>'goal_title')
            ELSE NULL
        END;
        row_data := update_scheduled_task(
            task_id,
            p_args->>'name',
            p_args->>'description',
            COALESCE(parsed->>'schedule_kind', p_args->>'schedule_kind'),
            parsed->'schedule',
            COALESCE(parsed->>'timezone', p_args->>'timezone'),
            p_args->>'action_kind',
            action_payload,
            p_args->>'status',
            NULLIF(p_args->>'max_runs', '')::int,
            delivery
        );
        RETURN jsonb_build_object('success', true, 'output', jsonb_build_object('task_id', task_id::text, 'updated', true, 'task', row_data), 'display_output', format('Updated scheduled task %s...', left(task_id::text, 8)));
    ELSIF action = 'cancel' THEN
        task_id := NULLIF(p_args->>'task_id', '')::uuid;
        IF task_id IS NULL AND NULLIF(p_args->>'name', '') IS NOT NULL THEN
            SELECT id INTO task_id FROM scheduled_tasks WHERE name = p_args->>'name' AND status = 'active' LIMIT 1;
        END IF;
        IF task_id IS NULL THEN
            RETURN jsonb_build_object('success', false, 'error', 'task_id or name is required for cancel', 'error_type', 'invalid_params');
        END IF;
        IF delete_scheduled_task(task_id, false, COALESCE(p_args->>'description', 'Cancelled by agent')) THEN
            RETURN jsonb_build_object('success', true, 'output', jsonb_build_object('task_id', task_id::text, 'cancelled', true), 'display_output', format('Cancelled scheduled task %s...', left(task_id::text, 8)));
        END IF;
        RETURN jsonb_build_object('success', false, 'error', format('Task %s not found', task_id), 'error_type', 'invalid_params');
    ELSE
        IF NULLIF(p_args->>'task_id', '') IS NOT NULL THEN
            SELECT jsonb_build_object(
                'task_id', t.id::text,
                'name', t.name,
                'schedule_kind', t.schedule_kind,
                'status', t.status,
                'run_count', t.run_count,
                'max_runs', t.max_runs,
                'last_run_at', t.last_run_at::text,
                'next_run_at', t.next_run_at::text,
                'last_error', t.last_error,
                'created_at', t.created_at::text
            ) INTO row_data
            FROM scheduled_tasks t WHERE id = (p_args->>'task_id')::uuid;
            IF row_data IS NULL THEN
                RETURN jsonb_build_object('success', false, 'error', format('Task %s not found', p_args->>'task_id'), 'error_type', 'execution_failed');
            END IF;
            RETURN jsonb_build_object('success', true, 'output', row_data);
        END IF;
        SELECT jsonb_build_object(
            'active_tasks', COUNT(*) FILTER (WHERE status = 'active'),
            'paused_tasks', COUNT(*) FILTER (WHERE status = 'paused'),
            'disabled_tasks', COUNT(*) FILTER (WHERE status = 'disabled'),
            'total_executions', COALESCE(SUM(run_count), 0),
            'tasks_with_errors', COUNT(*) FILTER (WHERE last_error IS NOT NULL AND status = 'active'),
            'last_execution', MAX(last_run_at)::text,
            'next_execution', (MIN(next_run_at) FILTER (WHERE status = 'active'))::text
        ) INTO row_data
        FROM scheduled_tasks;
        row_data := row_data || jsonb_build_object('recent_runs', COALESCE(
            (SELECT jsonb_agg(jsonb_build_object(
                    'at', e.created_at::text,
                    'tasks_executed', e.payload->'executed'
                ) ORDER BY e.created_at DESC)
             FROM (SELECT created_at, payload FROM gateway_events
                   WHERE source = 'cron'
                   ORDER BY created_at DESC LIMIT 10) e),
            '[]'::jsonb));
        RETURN jsonb_build_object('success', true, 'output', row_data);
    END IF;
EXCEPTION WHEN OTHERS THEN
    RETURN jsonb_build_object('success', false, 'error', SQLERRM, 'error_type', 'execution_failed');
END;
$$;

CREATE OR REPLACE FUNCTION recompute_cron_next_runs(
    p_task_ids UUID[]
) RETURNS INT
LANGUAGE plpgsql
AS $$
DECLARE
    updated_count INT := 0;
    task_id UUID;
    schedule_value JSONB;
    next_run TIMESTAMPTZ;
    task_tz TEXT;
BEGIN
    IF p_task_ids IS NULL OR cardinality(p_task_ids) = 0 THEN
        RETURN 0;
    END IF;
    FOREACH task_id IN ARRAY p_task_ids LOOP
        SELECT schedule, COALESCE(timezone, 'UTC') INTO schedule_value, task_tz FROM scheduled_tasks WHERE id = task_id AND schedule_kind = 'cron';
        IF NOT FOUND THEN
            CONTINUE;
        END IF;
        next_run := compute_next_run_at('cron', schedule_value, task_tz, CURRENT_TIMESTAMP);
        schedule_value := COALESCE(schedule_value, '{}'::jsonb)
            || jsonb_build_object('_next_run', next_run::text);
        UPDATE scheduled_tasks
        SET schedule = schedule_value,
            next_run_at = next_run,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = task_id;
        updated_count := updated_count + 1;
    END LOOP;
    RETURN updated_count;
END;
$$;

CREATE OR REPLACE FUNCTION create_workflow_execution(
    p_plan JSONB,
    p_context JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    wf_id UUID;
    step JSONB;
BEGIN
    IF NULLIF(p_plan->>'name', '') IS NULL THEN
        RAISE EXCEPTION 'Workflow name is required';
    END IF;
    IF jsonb_typeof(COALESCE(p_plan->'steps', '[]'::jsonb)) <> 'array' OR jsonb_array_length(COALESCE(p_plan->'steps', '[]'::jsonb)) = 0 THEN
        RAISE EXCEPTION 'Workflow must have at least one step';
    END IF;

    INSERT INTO workflow_executions (name, plan, status, session_id)
    VALUES (p_plan->>'name', p_plan, 'running', p_context->>'session_id')
    RETURNING id INTO wf_id;

    FOR step IN SELECT * FROM jsonb_array_elements(p_plan->'steps')
    LOOP
        INSERT INTO workflow_step_runs (
            workflow_id, step_name, tool_name, arguments, depends_on, status, max_attempts
        )
        VALUES (
            wf_id,
            step->>'name',
            step->>'tool',
            COALESCE(step->'arguments', '{}'::jsonb),
            COALESCE(ARRAY(SELECT jsonb_array_elements_text(COALESCE(step->'depends_on', '[]'::jsonb))), ARRAY[]::TEXT[]),
            CASE WHEN jsonb_array_length(COALESCE(step->'depends_on', '[]'::jsonb)) = 0 THEN 'ready' ELSE 'pending' END,
            CASE WHEN COALESCE(step->>'on_error', 'stop') = 'retry' THEN GREATEST(COALESCE(NULLIF(step->>'max_retries', '')::int, 1), 1) ELSE 1 END
        )
        ON CONFLICT (workflow_id, step_name) DO NOTHING;
    END LOOP;

    RETURN jsonb_build_object('workflow_id', wf_id::text, 'status', 'running');
END;
$$;

CREATE OR REPLACE FUNCTION workflow_plan_layers(
    p_plan JSONB
) RETURNS JSONB
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    steps JSONB := COALESCE(p_plan->'steps', '[]'::jsonb);
    step JSONB;
    step_names TEXT[] := ARRAY[]::TEXT[];
    remaining TEXT[] := ARRAY[]::TEXT[];
    satisfied TEXT[] := ARRAY[]::TEXT[];
    layers JSONB := '[]'::jsonb;
    layer JSONB;
    name_value TEXT;
    dep TEXT;
    deps TEXT[];
    progress BOOLEAN;
BEGIN
    IF jsonb_typeof(steps) <> 'array' OR jsonb_array_length(steps) = 0 THEN
        RAISE EXCEPTION 'Workflow must have at least one step';
    END IF;

    FOR step IN SELECT * FROM jsonb_array_elements(steps)
    LOOP
        name_value := step->>'name';
        IF NULLIF(name_value, '') IS NULL THEN
            RAISE EXCEPTION 'Workflow step name is required';
        END IF;
        IF name_value = ANY(step_names) THEN
            RAISE EXCEPTION 'Workflow step names must be unique';
        END IF;
        step_names := step_names || name_value;
        remaining := remaining || name_value;
    END LOOP;

    FOR step IN SELECT * FROM jsonb_array_elements(steps)
    LOOP
        FOR dep IN SELECT * FROM jsonb_array_elements_text(COALESCE(step->'depends_on', '[]'::jsonb))
        LOOP
            IF NOT dep = ANY(step_names) THEN
                RAISE EXCEPTION 'Step % depends on unknown step %', step->>'name', dep;
            END IF;
        END LOOP;
    END LOOP;

    WHILE cardinality(remaining) > 0 LOOP
        layer := '[]'::jsonb;
        progress := FALSE;
        FOREACH name_value IN ARRAY remaining LOOP
            SELECT COALESCE(ARRAY(SELECT jsonb_array_elements_text(COALESCE(s->'depends_on', '[]'::jsonb))), ARRAY[]::TEXT[])
            INTO deps
            FROM jsonb_array_elements(steps) s
            WHERE s->>'name' = name_value;

            IF NOT EXISTS (SELECT 1 FROM unnest(deps) d WHERE NOT d = ANY(satisfied)) THEN
                SELECT layer || jsonb_build_array(s)
                INTO layer
                FROM jsonb_array_elements(steps) s
                WHERE s->>'name' = name_value;
                progress := TRUE;
            END IF;
        END LOOP;

        IF NOT progress THEN
            RAISE EXCEPTION 'Circular dependency detected among steps: %', remaining;
        END IF;

        layers := layers || jsonb_build_array(layer);
        SELECT COALESCE(ARRAY(
            SELECT r
            FROM unnest(remaining) r
            WHERE NOT EXISTS (
                SELECT 1
                FROM jsonb_array_elements(layer) l
                WHERE l->>'name' = r
            )
        ), ARRAY[]::TEXT[]) INTO remaining;
        SELECT COALESCE(ARRAY(
            SELECT DISTINCT value
            FROM unnest(satisfied || ARRAY(
                SELECT l->>'name'
                FROM jsonb_array_elements(layer) l
            )) v(value)
        ), ARRAY[]::TEXT[]) INTO satisfied;
    END LOOP;

    RETURN layers;
END;
$$;

CREATE OR REPLACE FUNCTION resolve_workflow_templates(
    p_value JSONB,
    p_step_outputs JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    key TEXT;
    value JSONB;
    out_obj JSONB := '{}'::jsonb;
    out_arr JSONB := '[]'::jsonb;
    text_value TEXT;
    match TEXT[];
    replacement JSONB;
    replacement_text TEXT;
BEGIN
    IF p_value IS NULL THEN
        RETURN 'null'::jsonb;
    END IF;

    IF jsonb_typeof(p_value) = 'object' THEN
        FOR key, value IN SELECT * FROM jsonb_each(p_value)
        LOOP
            out_obj := out_obj || jsonb_build_object(key, resolve_workflow_templates(value, p_step_outputs));
        END LOOP;
        RETURN out_obj;
    ELSIF jsonb_typeof(p_value) = 'array' THEN
        FOR value IN SELECT * FROM jsonb_array_elements(p_value)
        LOOP
            out_arr := out_arr || jsonb_build_array(resolve_workflow_templates(value, p_step_outputs));
        END LOOP;
        RETURN out_arr;
    ELSIF jsonb_typeof(p_value) <> 'string' THEN
        RETURN p_value;
    END IF;

    text_value := p_value #>> '{}';
    match := regexp_match(text_value, '^\{\{([A-Za-z0-9_]+)\.output(?:\.([A-Za-z0-9_]+))?\}\}$');
    IF match IS NOT NULL THEN
        replacement := p_step_outputs -> match[1];
        IF match[2] IS NOT NULL AND jsonb_typeof(replacement) = 'object' THEN
            replacement := replacement -> match[2];
        END IF;
        RETURN COALESCE(replacement, p_value);
    END IF;

    FOR match IN SELECT regexp_matches(text_value, '\{\{([A-Za-z0-9_]+)\.output(?:\.([A-Za-z0-9_]+))?\}\}', 'g')
    LOOP
        replacement := p_step_outputs -> match[1];
        IF match[2] IS NOT NULL AND jsonb_typeof(replacement) = 'object' THEN
            replacement := replacement -> match[2];
        END IF;
        replacement_text := COALESCE(replacement #>> '{}', '{{' || match[1] || '.output' || CASE WHEN match[2] IS NULL THEN '' ELSE '.' || match[2] END || '}}');
        text_value := replace(
            text_value,
            '{{' || match[1] || '.output' || CASE WHEN match[2] IS NULL THEN '' ELSE '.' || match[2] END || '}}',
            replacement_text
        );
    END LOOP;

    RETURN to_jsonb(text_value);
END;
$$;

CREATE OR REPLACE FUNCTION claim_workflow_steps(
    p_workflow_id UUID
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    claimed JSONB;
BEGIN
    WITH ready AS (
        SELECT s.id
        FROM workflow_step_runs s
        WHERE s.workflow_id = p_workflow_id
          AND s.status IN ('ready', 'pending')
          AND NOT EXISTS (
              SELECT 1
              FROM unnest(s.depends_on) dep(step_name)
              JOIN workflow_step_runs d ON d.workflow_id = s.workflow_id AND d.step_name = dep.step_name
              WHERE d.status <> 'completed'
          )
        ORDER BY s.created_at
        FOR UPDATE SKIP LOCKED
    ),
    updated AS (
        UPDATE workflow_step_runs s
        SET status = 'in_progress',
            attempts = attempts + 1,
            started_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        FROM ready
        WHERE s.id = ready.id
        RETURNING s.*
    )
    SELECT COALESCE(jsonb_agg(to_jsonb(updated)), '[]'::jsonb) INTO claimed
    FROM updated;

    RETURN claimed;
END;
$$;

CREATE OR REPLACE FUNCTION apply_workflow_step_result(
    p_step_id UUID,
    p_result JSONB
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    row_out workflow_step_runs%ROWTYPE;
BEGIN
    UPDATE workflow_step_runs
    SET status = CASE WHEN COALESCE((p_result->>'success')::boolean, false) THEN 'completed' ELSE 'failed' END,
        output = COALESCE(p_result->'output', p_result),
        error = p_result->>'error',
        completed_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_step_id
    RETURNING * INTO row_out;

    RETURN to_jsonb(row_out);
END;
$$;

CREATE OR REPLACE FUNCTION finalize_workflow_execution(
    p_workflow_id UUID
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    status_value TEXT;
    steps JSONB;
    total_energy INT;
    error_value TEXT;
BEGIN
    SELECT COALESCE(jsonb_agg(to_jsonb(s) ORDER BY s.created_at), '[]'::jsonb),
           COALESCE(SUM(COALESCE((s.output->>'energy_spent')::int, 0)), 0),
           CASE WHEN COUNT(*) FILTER (WHERE s.status = 'failed') > 0 THEN 'failed' ELSE 'completed' END,
           MIN(s.error) FILTER (WHERE s.status = 'failed')
    INTO steps, total_energy, status_value, error_value
    FROM workflow_step_runs s
    WHERE s.workflow_id = p_workflow_id;

    UPDATE workflow_executions
    SET status = status_value,
        step_results = steps,
        total_energy_spent = total_energy,
        error = error_value,
        completed_at = CURRENT_TIMESTAMP
    WHERE id = p_workflow_id;

    RETURN jsonb_build_object('workflow_id', p_workflow_id::text, 'status', status_value, 'steps', steps, 'total_energy_spent', total_energy, 'error', error_value);
END;
$$;

-- Bulk writer for services/capability_probe.py. The table is a current
-- snapshot, so a worker refresh replaces its prior observation in place.
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
