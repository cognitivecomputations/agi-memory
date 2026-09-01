-- Hexis schema: scheduling functions.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

CREATE OR REPLACE FUNCTION normalize_timezone(p_timezone TEXT)
RETURNS TEXT AS $$
DECLARE
    tz TEXT;
BEGIN
    tz := NULLIF(btrim(p_timezone), '');
    IF tz IS NULL THEN
        RETURN 'UTC';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_timezone_names WHERE name = tz) THEN
        RETURN tz;
    END IF;
    RAISE EXCEPTION 'Unknown timezone: %', tz;
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION normalize_weekday(p_weekday TEXT)
RETURNS INT AS $$
DECLARE
    wd TEXT;
    wd_int INT;
BEGIN
    wd := NULLIF(lower(btrim(p_weekday)), '');
    IF wd IS NULL THEN
        RAISE EXCEPTION 'Weekday is required';
    END IF;

    BEGIN
        wd_int := wd::int;
        IF wd_int = 0 THEN
            wd_int := 7;
        END IF;
        IF wd_int < 1 OR wd_int > 7 THEN
            RAISE EXCEPTION 'Weekday out of range: %', wd_int;
        END IF;
        RETURN wd_int;
    EXCEPTION
        WHEN invalid_text_representation THEN
            NULL;
    END;

    IF wd IN ('mon', 'monday') THEN RETURN 1; END IF;
    IF wd IN ('tue', 'tues', 'tuesday') THEN RETURN 2; END IF;
    IF wd IN ('wed', 'weds', 'wednesday') THEN RETURN 3; END IF;
    IF wd IN ('thu', 'thur', 'thurs', 'thursday') THEN RETURN 4; END IF;
    IF wd IN ('fri', 'friday') THEN RETURN 5; END IF;
    IF wd IN ('sat', 'saturday') THEN RETURN 6; END IF;
    IF wd IN ('sun', 'sunday') THEN RETURN 7; END IF;

    RAISE EXCEPTION 'Invalid weekday: %', p_weekday;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION parse_time_of_day(p_time TEXT)
RETURNS TIME AS $$
DECLARE
    t TIME;
BEGIN
    t := NULLIF(btrim(p_time), '')::time;
    IF t IS NULL THEN
        RAISE EXCEPTION 'Time of day is required';
    END IF;
    RETURN t;
EXCEPTION
    WHEN OTHERS THEN
        RAISE EXCEPTION 'Invalid time of day: %', p_time;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- One atom of a cron field: a number or a name (JAN..DEC / SUN..SAT).
CREATE OR REPLACE FUNCTION cron_field_atom(
    p_atom TEXT, p_min INT, p_max INT, p_names TEXT[] DEFAULT NULL
) RETURNS INT AS $$
DECLARE
    idx INT;
    val INT;
BEGIN
    IF p_names IS NOT NULL THEN
        idx := array_position(p_names, upper(btrim(p_atom)));
        IF idx IS NOT NULL THEN
            RETURN p_min + idx - 1;
        END IF;
    END IF;
    IF btrim(p_atom) !~ '^\d+$' THEN
        RAISE EXCEPTION 'invalid cron atom: %', p_atom;
    END IF;
    val := btrim(p_atom)::int;
    IF val < p_min OR val > p_max THEN
        RAISE EXCEPTION 'cron value % out of range %-%', val, p_min, p_max;
    END IF;
    RETURN val;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Expand one cron field ('*', 'a', 'a-b', '*/n', 'a-b/n', 'a/n', lists) into
-- its sorted value set. Raises on any invalid syntax or out-of-range value,
-- so this doubles as the cron validator.
CREATE OR REPLACE FUNCTION cron_field_values(
    p_field TEXT, p_min INT, p_max INT, p_names TEXT[] DEFAULT NULL
) RETURNS INT[] AS $$
DECLARE
    field TEXT := btrim(coalesce(p_field, ''));
    part TEXT;
    body TEXT;
    step_txt TEXT;
    step INT;
    lo INT;
    hi INT;
    seen BOOLEAN[] := array_fill(false, ARRAY[p_max - p_min + 1]);
    result INT[] := '{}';
    i INT;
BEGIN
    IF field = '' THEN
        RAISE EXCEPTION 'empty cron field';
    END IF;
    FOREACH part IN ARRAY string_to_array(field, ',') LOOP
        part := btrim(part);
        step := 1;
        body := part;
        IF position('/' in part) > 0 THEN
            body := split_part(part, '/', 1);
            step_txt := split_part(part, '/', 2);
            IF step_txt !~ '^\d+$' OR step_txt::int < 1 THEN
                RAISE EXCEPTION 'invalid cron step: %', part;
            END IF;
            step := step_txt::int;
        END IF;
        IF body = '*' THEN
            lo := p_min;
            hi := p_max;
        ELSIF position('-' in body) > 0 THEN
            lo := cron_field_atom(split_part(body, '-', 1), p_min, p_max, p_names);
            hi := cron_field_atom(split_part(body, '-', 2), p_min, p_max, p_names);
            IF lo > hi THEN
                RAISE EXCEPTION 'inverted cron range: %', body;
            END IF;
        ELSE
            lo := cron_field_atom(body, p_min, p_max, p_names);
            -- 'a/n' means a..max stepped by n; a bare 'a' is just a.
            hi := CASE WHEN step > 1 THEN p_max ELSE lo END;
        END IF;
        i := lo;
        WHILE i <= hi LOOP
            seen[i - p_min + 1] := true;
            i := i + step;
        END LOOP;
    END LOOP;
    FOR i IN p_min..p_max LOOP
        IF seen[i - p_min + 1] THEN
            result := result || i;
        END IF;
    END LOOP;
    RETURN result;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Real cron next-fire computation (minute resolution) in the task's
-- timezone. Fields: minute hour day-of-month month day-of-week; a 6-field
-- expression drops its trailing seconds field. Standard Vixie semantics: when
-- both day fields are restricted (not '*'), a day matches if EITHER matches;
-- day-of-week accepts 0-7 (0 and 7 are Sunday) and SUN-SAT names.
CREATE OR REPLACE FUNCTION cron_next_fire(
    p_cron TEXT,
    p_timezone TEXT DEFAULT 'UTC',
    p_after TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
) RETURNS TIMESTAMPTZ AS $$
DECLARE
    fields TEXT[] := regexp_split_to_array(btrim(coalesce(p_cron, '')), '\s+');
    minutes INT[];
    hours INT[];
    doms INT[];
    months INT[];
    dows INT[];
    dom_star BOOLEAN;
    dow_star BOOLEAN;
    tz TEXT := normalize_timezone(p_timezone);
    local_after TIMESTAMP;
    day0 DATE;
    day DATE;
    d INT;
    h INT;
    m INT;
    day_ok BOOLEAN;
    cursor_time TIME;
    result TIMESTAMPTZ;
BEGIN
    IF array_length(fields, 1) = 6 THEN
        fields := fields[1:5];
    END IF;
    IF array_length(fields, 1) <> 5 THEN
        RAISE EXCEPTION 'cron expression must have 5 fields: %', p_cron;
    END IF;
    minutes := cron_field_values(fields[1], 0, 59);
    hours := cron_field_values(fields[2], 0, 23);
    doms := cron_field_values(fields[3], 1, 31);
    months := cron_field_values(fields[4], 1, 12,
        ARRAY['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']);
    dows := ARRAY(SELECT DISTINCT v % 7 FROM unnest(cron_field_values(fields[5], 0, 7,
        ARRAY['SUN','MON','TUE','WED','THU','FRI','SAT','SUN'])) v ORDER BY v % 7);
    dom_star := btrim(fields[3]) = '*';
    dow_star := btrim(fields[5]) = '*';

    local_after := date_trunc('minute', p_after AT TIME ZONE tz) + INTERVAL '1 minute';
    day0 := local_after::date;
    FOR d IN 0..1500 LOOP
        day := day0 + d;
        IF NOT (EXTRACT(MONTH FROM day)::int = ANY(months)) THEN
            CONTINUE;
        END IF;
        IF dom_star AND dow_star THEN
            day_ok := true;
        ELSIF dom_star THEN
            day_ok := EXTRACT(DOW FROM day)::int = ANY(dows);
        ELSIF dow_star THEN
            day_ok := EXTRACT(DAY FROM day)::int = ANY(doms);
        ELSE
            day_ok := EXTRACT(DAY FROM day)::int = ANY(doms)
                   OR EXTRACT(DOW FROM day)::int = ANY(dows);
        END IF;
        IF NOT day_ok THEN
            CONTINUE;
        END IF;
        cursor_time := CASE WHEN d = 0 THEN local_after::time ELSE '00:00'::time END;
        FOREACH h IN ARRAY hours LOOP
            IF make_time(h, 59, 59.0) < cursor_time THEN
                CONTINUE;
            END IF;
            FOREACH m IN ARRAY minutes LOOP
                IF make_time(h, m, 0.0) >= cursor_time THEN
                    result := (day + make_time(h, m, 0.0)) AT TIME ZONE tz;
                    IF result > p_after THEN
                        RETURN result;
                    END IF;
                END IF;
            END LOOP;
        END LOOP;
    END LOOP;
    RAISE EXCEPTION 'cron expression never fires: %', p_cron;
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION compute_next_run_at(
    p_schedule_kind TEXT,
    p_schedule JSONB,
    p_timezone TEXT DEFAULT 'UTC',
    p_after TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
)
RETURNS TIMESTAMPTZ AS $$
DECLARE
    schedule_kind TEXT;
    tz TEXT;
    after_ts TIMESTAMPTZ;
    after_local TIMESTAMP;
    run_at TIMESTAMPTZ;
    target_time TIME;
    weekday INT;
    days_ahead INT;
    interval_seconds INT;
    anchor_ts TIMESTAMPTZ;
    elapsed_seconds DOUBLE PRECISION;
    next_ts TIMESTAMPTZ;
BEGIN
    schedule_kind := NULLIF(lower(btrim(p_schedule_kind)), '');
    IF schedule_kind IS NULL THEN
        RAISE EXCEPTION 'schedule_kind is required';
    END IF;

    tz := normalize_timezone(p_timezone);
    after_ts := COALESCE(p_after, CURRENT_TIMESTAMP);

    CASE schedule_kind
        WHEN 'once' THEN
            run_at := NULLIF(p_schedule->>'run_at', '')::timestamptz;
            IF run_at IS NULL THEN
                RAISE EXCEPTION 'schedule.run_at is required for once schedules';
            END IF;
            IF run_at <= after_ts THEN
                RETURN NULL;
            END IF;
            RETURN run_at;
        WHEN 'interval' THEN
            interval_seconds := COALESCE(
                NULLIF((p_schedule->>'every_seconds')::int, 0),
                NULLIF((p_schedule->>'every_minutes')::int, 0) * 60,
                NULLIF((p_schedule->>'every_hours')::int, 0) * 3600
            );
            IF interval_seconds IS NULL OR interval_seconds <= 0 THEN
                RAISE EXCEPTION 'schedule.every_seconds/every_minutes/every_hours required for interval schedules';
            END IF;
            anchor_ts := NULLIF(p_schedule->>'anchor_at', '')::timestamptz;
            IF anchor_ts IS NULL THEN
                anchor_ts := after_ts;
            END IF;
            IF after_ts < anchor_ts THEN
                RETURN anchor_ts;
            END IF;
            elapsed_seconds := EXTRACT(EPOCH FROM (after_ts - anchor_ts));
            next_ts := anchor_ts + ((floor(elapsed_seconds / interval_seconds) + 1) * interval_seconds) * INTERVAL '1 second';
            RETURN next_ts;
        WHEN 'daily' THEN
            target_time := parse_time_of_day(p_schedule->>'time');
            after_local := after_ts AT TIME ZONE tz;
            run_at := (date_trunc('day', after_local) + target_time) AT TIME ZONE tz;
            IF run_at <= after_ts THEN
                run_at := ((date_trunc('day', after_local) + target_time) + INTERVAL '1 day') AT TIME ZONE tz;
            END IF;
            RETURN run_at;
        WHEN 'weekly' THEN
            target_time := parse_time_of_day(p_schedule->>'time');
            weekday := normalize_weekday(p_schedule->>'weekday');
            after_local := after_ts AT TIME ZONE tz;
            days_ahead := (weekday - EXTRACT(ISODOW FROM after_local)::int + 7) % 7;
            run_at := (date_trunc('day', after_local) + (days_ahead || ' days')::interval + target_time) AT TIME ZONE tz;
            IF run_at <= after_ts THEN
                run_at := run_at + INTERVAL '7 days';
            END IF;
            RETURN run_at;
        WHEN 'cron' THEN
            -- Real cron math lives in cron_next_fire; the schedule JSONB
            -- stores {"cron": "..."} (a stale "_next_run" is ignored).
            -- Fail open to +1 minute only if evaluation fails at runtime —
            -- creation-time validation should make that unreachable.
            BEGIN
                RETURN cron_next_fire(p_schedule->>'cron', tz, after_ts);
            EXCEPTION WHEN OTHERS THEN
                RAISE WARNING 'cron_next_fire failed for %: %', p_schedule->>'cron', SQLERRM;
                run_at := NULLIF(p_schedule->>'_next_run', '')::timestamptz;
                IF run_at IS NOT NULL AND run_at > after_ts THEN
                    RETURN run_at;
                END IF;
                RETURN after_ts + INTERVAL '1 minute';
            END;
        ELSE
            RAISE EXCEPTION 'Unsupported schedule_kind: %', schedule_kind;
    END CASE;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION create_scheduled_task(
    p_name TEXT,
    p_schedule_kind TEXT,
    p_schedule JSONB,
    p_action_kind TEXT,
    p_action_payload JSONB DEFAULT '{}'::jsonb,
    p_timezone TEXT DEFAULT 'UTC',
    p_description TEXT DEFAULT NULL,
    p_status TEXT DEFAULT 'active',
    p_max_runs INT DEFAULT NULL,
    p_created_by TEXT DEFAULT 'agent',
    p_delivery JSONB DEFAULT '{"mode": "outbox"}'::jsonb
)
RETURNS UUID AS $$
DECLARE
    task_id UUID;
    next_run TIMESTAMPTZ;
    status_value TEXT;
    action_kind TEXT;
BEGIN
    IF p_name IS NULL OR btrim(p_name) = '' THEN
        RAISE EXCEPTION 'Scheduled task name is required';
    END IF;
    status_value := COALESCE(NULLIF(p_status, ''), 'active');
    IF status_value NOT IN ('active', 'paused', 'disabled') THEN
        RAISE EXCEPTION 'Invalid status: %', status_value;
    END IF;
    action_kind := COALESCE(NULLIF(p_action_kind, ''), '');
    IF action_kind NOT IN ('queue_user_message', 'create_goal') THEN
        RAISE EXCEPTION 'Invalid action_kind: %', action_kind;
    END IF;
    IF action_kind = 'queue_user_message' THEN
        IF p_action_payload IS NULL OR NULLIF(p_action_payload->>'message', '') IS NULL THEN
            RAISE EXCEPTION 'queue_user_message requires action_payload.message';
        END IF;
    ELSIF action_kind = 'create_goal' THEN
        IF p_action_payload IS NULL OR NULLIF(p_action_payload->>'title', '') IS NULL THEN
            RAISE EXCEPTION 'create_goal requires action_payload.title';
        END IF;
    END IF;

    next_run := compute_next_run_at(p_schedule_kind, p_schedule, p_timezone, CURRENT_TIMESTAMP);
    IF next_run IS NULL THEN
        RAISE EXCEPTION 'Schedule does not produce a future run';
    END IF;

    INSERT INTO scheduled_tasks (
        name,
        description,
        schedule_kind,
        schedule,
        timezone,
        action_kind,
        action_payload,
        delivery,
        status,
        next_run_at,
        max_runs,
        created_by,
        created_at,
        updated_at
    ) VALUES (
        p_name,
        p_description,
        lower(btrim(p_schedule_kind)),
        COALESCE(p_schedule, '{}'::jsonb),
        normalize_timezone(p_timezone),
        action_kind,
        COALESCE(p_action_payload, '{}'::jsonb),
        COALESCE(p_delivery, '{"mode": "outbox"}'::jsonb),
        status_value,
        next_run,
        p_max_runs,
        NULLIF(p_created_by, ''),
        CURRENT_TIMESTAMP,
        CURRENT_TIMESTAMP
    )
    RETURNING id INTO task_id;

    RETURN task_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION set_scheduled_task_status(
    p_task_id UUID,
    p_status TEXT,
    p_reason TEXT DEFAULT NULL
)
RETURNS BOOLEAN AS $$
DECLARE
    status_value TEXT;
BEGIN
    status_value := COALESCE(NULLIF(p_status, ''), 'active');
    IF status_value NOT IN ('active', 'paused', 'disabled') THEN
        RAISE EXCEPTION 'Invalid status: %', status_value;
    END IF;

    UPDATE scheduled_tasks
    SET status = status_value,
        last_error = CASE WHEN p_reason IS NULL OR btrim(p_reason) = '' THEN last_error ELSE p_reason END,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_task_id;

    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION list_scheduled_tasks(
    p_status TEXT DEFAULT NULL,
    p_due_before TIMESTAMPTZ DEFAULT NULL,
    p_limit INT DEFAULT 50
)
RETURNS TABLE (
    id UUID,
    name TEXT,
    description TEXT,
    schedule_kind TEXT,
    schedule JSONB,
    timezone TEXT,
    action_kind TEXT,
    action_payload JSONB,
    delivery JSONB,
    status TEXT,
    next_run_at TIMESTAMPTZ,
    last_run_at TIMESTAMPTZ,
    run_count INT,
    max_runs INT,
    created_by TEXT,
    last_error TEXT,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        t.id,
        t.name,
        t.description,
        t.schedule_kind,
        t.schedule,
        t.timezone,
        t.action_kind,
        t.action_payload,
        t.delivery,
        t.status,
        t.next_run_at,
        t.last_run_at,
        t.run_count,
        t.max_runs,
        t.created_by,
        t.last_error,
        t.created_at,
        t.updated_at
    FROM scheduled_tasks t
    WHERE (p_status IS NULL OR t.status = p_status)
      AND (p_due_before IS NULL OR t.next_run_at <= p_due_before)
    ORDER BY t.next_run_at ASC
    LIMIT GREATEST(1, LEAST(p_limit, 200));
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION update_scheduled_task(
    p_task_id UUID,
    p_name TEXT DEFAULT NULL,
    p_description TEXT DEFAULT NULL,
    p_schedule_kind TEXT DEFAULT NULL,
    p_schedule JSONB DEFAULT NULL,
    p_timezone TEXT DEFAULT NULL,
    p_action_kind TEXT DEFAULT NULL,
    p_action_payload JSONB DEFAULT NULL,
    p_status TEXT DEFAULT NULL,
    p_max_runs INT DEFAULT NULL,
    p_delivery JSONB DEFAULT NULL
)
RETURNS JSONB AS $$
DECLARE
    current_task scheduled_tasks%ROWTYPE;
    updated_row JSONB;
    new_schedule_kind TEXT;
    new_schedule JSONB;
    new_timezone TEXT;
    new_action_kind TEXT;
    new_action_payload JSONB;
    new_status TEXT;
    new_next_run TIMESTAMPTZ;
BEGIN
    SELECT * INTO current_task
    FROM scheduled_tasks
    WHERE id = p_task_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Scheduled task not found: %', p_task_id;
    END IF;

    new_schedule_kind := COALESCE(NULLIF(p_schedule_kind, ''), current_task.schedule_kind);
    new_schedule := COALESCE(p_schedule, current_task.schedule);
    new_timezone := normalize_timezone(COALESCE(NULLIF(p_timezone, ''), current_task.timezone));
    new_action_kind := COALESCE(NULLIF(p_action_kind, ''), current_task.action_kind);
    new_action_payload := COALESCE(p_action_payload, current_task.action_payload);
    new_status := COALESCE(NULLIF(p_status, ''), current_task.status);

    IF new_status NOT IN ('active', 'paused', 'disabled') THEN
        RAISE EXCEPTION 'Invalid status: %', new_status;
    END IF;
    IF new_action_kind NOT IN ('queue_user_message', 'create_goal') THEN
        RAISE EXCEPTION 'Invalid action_kind: %', new_action_kind;
    END IF;
    IF new_action_kind = 'queue_user_message' THEN
        IF new_action_payload IS NULL OR NULLIF(new_action_payload->>'message', '') IS NULL THEN
            RAISE EXCEPTION 'queue_user_message requires action_payload.message';
        END IF;
    ELSIF new_action_kind = 'create_goal' THEN
        IF new_action_payload IS NULL OR NULLIF(new_action_payload->>'title', '') IS NULL THEN
            RAISE EXCEPTION 'create_goal requires action_payload.title';
        END IF;
    END IF;

    IF new_schedule_kind IS DISTINCT FROM current_task.schedule_kind
        OR new_schedule IS DISTINCT FROM current_task.schedule
        OR new_timezone IS DISTINCT FROM current_task.timezone THEN
        new_next_run := compute_next_run_at(new_schedule_kind, new_schedule, new_timezone, CURRENT_TIMESTAMP);
        IF new_next_run IS NULL THEN
            RAISE EXCEPTION 'Schedule does not produce a future run';
        END IF;
    ELSE
        new_next_run := current_task.next_run_at;
    END IF;

    UPDATE scheduled_tasks
    SET name = COALESCE(NULLIF(p_name, ''), current_task.name),
        description = COALESCE(p_description, current_task.description),
        schedule_kind = new_schedule_kind,
        schedule = new_schedule,
        timezone = new_timezone,
        action_kind = new_action_kind,
        action_payload = new_action_payload,
        delivery = COALESCE(p_delivery, current_task.delivery),
        status = new_status,
        next_run_at = new_next_run,
        max_runs = COALESCE(p_max_runs, current_task.max_runs),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_task_id
    RETURNING to_jsonb(scheduled_tasks.*) INTO updated_row;

    RETURN updated_row;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION delete_scheduled_task(
    p_task_id UUID,
    p_hard_delete BOOLEAN DEFAULT FALSE,
    p_reason TEXT DEFAULT NULL
)
RETURNS BOOLEAN AS $$
BEGIN
    IF COALESCE(p_hard_delete, FALSE) THEN
        DELETE FROM scheduled_tasks WHERE id = p_task_id;
        RETURN FOUND;
    END IF;

    UPDATE scheduled_tasks
    SET status = 'disabled',
        last_error = COALESCE(NULLIF(p_reason, ''), last_error),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_task_id;

    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION run_scheduled_tasks(p_limit INT DEFAULT 25)
RETURNS JSONB AS $$
DECLARE
    task RECORD;
    now_ts TIMESTAMPTZ := CURRENT_TIMESTAMP;
    outbox_messages JSONB := '[]'::jsonb;
    ran_count INT := 0;
    next_run TIMESTAMPTZ;
    action_payload JSONB;
    delivery_info JSONB;
    goal_id UUID;
    task_status TEXT;
    cron_task_ids JSONB := '[]'::jsonb;
    ran_tasks JSONB := '[]'::jsonb;
BEGIN
    FOR task IN
        SELECT *
        FROM scheduled_tasks
        WHERE status = 'active'
          AND next_run_at <= now_ts
        ORDER BY next_run_at ASC
        LIMIT p_limit
        FOR UPDATE SKIP LOCKED
    LOOP
        BEGIN
            action_payload := COALESCE(task.action_payload, '{}'::jsonb);
            delivery_info := COALESCE(task.delivery, '{"mode": "outbox"}'::jsonb);

            IF task.action_kind = 'queue_user_message' THEN
                outbox_messages := outbox_messages || jsonb_build_array(
                    build_user_message(
                        NULLIF(action_payload->>'message', ''),
                        NULLIF(action_payload->>'intent', ''),
                        action_payload->'context'
                    ) || jsonb_build_object('delivery', delivery_info, 'task_name', task.name)
                );
            ELSIF task.action_kind = 'create_goal' THEN
                goal_id := create_goal(
                    NULLIF(action_payload->>'title', ''),
                    NULLIF(action_payload->>'description', ''),
                    COALESCE(NULLIF(action_payload->>'source', ''), 'user_request')::goal_source,
                    COALESCE(NULLIF(action_payload->>'priority', ''), 'queued')::goal_priority,
                    NULLIF(action_payload->>'parent_id', '')::uuid,
                    COALESCE(NULLIF(action_payload->>'due_at', '')::timestamptz, task.next_run_at),
                    CASE
                        WHEN action_payload->>'origin' IN (
                            'curiosity', 'user_request', 'identity', 'derived', 'external'
                        ) THEN (action_payload->>'origin')::goal_source
                        ELSE 'derived'::goal_source
                    END
                );
                IF COALESCE((action_payload->>'notify')::boolean, false) THEN
                    outbox_messages := outbox_messages || jsonb_build_array(
                        build_user_message(
                            format('Created goal: %s', COALESCE(action_payload->>'title', goal_id::text)),
                            'goal_created',
                            jsonb_build_object('goal_id', goal_id::text, 'task_id', task.id::text)
                        ) || jsonb_build_object('delivery', delivery_info, 'task_name', task.name)
                    );
                END IF;
            END IF;

            ran_count := ran_count + 1;
            ran_tasks := ran_tasks || jsonb_build_array(jsonb_build_object(
                'id', task.id::text,
                'name', task.name,
                'schedule_kind', task.schedule_kind,
                'action_kind', task.action_kind,
                'delivery', delivery_info
            ));

            -- Track cron tasks for Python-side next_run recomputation
            IF task.schedule_kind = 'cron' THEN
                cron_task_ids := cron_task_ids || jsonb_build_array(task.id::text);
            END IF;

            next_run := compute_next_run_at(task.schedule_kind, task.schedule, task.timezone, now_ts);
            IF task.max_runs IS NOT NULL AND (task.run_count + 1) >= task.max_runs THEN
                task_status := 'disabled';
            ELSIF next_run IS NULL THEN
                task_status := 'disabled';
            ELSE
                task_status := task.status;
            END IF;

            UPDATE scheduled_tasks
            SET last_run_at = now_ts,
                run_count = run_count + 1,
                next_run_at = COALESCE(next_run, next_run_at),
                status = task_status,
                last_error = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = task.id;
        EXCEPTION
            WHEN OTHERS THEN
                UPDATE scheduled_tasks
                SET last_error = SQLERRM,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = task.id;
        END;
    END LOOP;

    RETURN jsonb_build_object(
        'ran', ran_count,
        'outbox_messages', outbox_messages,
        'cron_task_ids', cron_task_ids,
        'ran_tasks', ran_tasks
    );
END;
$$ LANGUAGE plpgsql;

SET check_function_bodies = on;
