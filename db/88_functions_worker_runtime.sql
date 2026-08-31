-- DB-native worker runtime observability.
SET search_path = public, ag_catalog, "$user";

CREATE OR REPLACE FUNCTION register_worker_instance(
    p_mode TEXT,
    p_instance_name TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS UUID AS $$
DECLARE
    v_id UUID;
    v_mode TEXT := lower(btrim(COALESCE(p_mode, 'unknown')));
    v_metadata JSONB := COALESCE(p_metadata, '{}'::jsonb);
    v_process_id INT;
    v_host_name TEXT;
    v_build_id TEXT;
BEGIN
    IF v_mode NOT IN ('heartbeat', 'maintenance', 'both', 'channel', 'unknown') THEN
        v_mode := 'unknown';
    END IF;

    IF jsonb_typeof(v_metadata->'process_id') = 'number' THEN
        v_process_id := (v_metadata->>'process_id')::int;
    END IF;
    v_host_name := NULLIF(v_metadata->>'host_name', '');
    v_build_id := NULLIF(v_metadata->>'build_id', '');

    INSERT INTO worker_instances (
        mode,
        instance_name,
        process_id,
        host_name,
        status,
        build_id,
        metadata
    ) VALUES (
        v_mode,
        NULLIF(p_instance_name, ''),
        v_process_id,
        v_host_name,
        'running',
        v_build_id,
        v_metadata
    )
    RETURNING id INTO v_id;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION record_worker_start_and_check_storm(
    p_mode TEXT,
    p_instance_name TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb,
    p_max_starts INTEGER DEFAULT 5,
    p_window_seconds INTEGER DEFAULT 120,
    p_backoff_cap_seconds INTEGER DEFAULT 300
) RETURNS JSONB AS $$
DECLARE
    v_mode TEXT := lower(btrim(COALESCE(p_mode, 'unknown')));
    v_metadata JSONB := COALESCE(p_metadata, '{}'::jsonb);
    v_process_id INT;
    v_host_name TEXT;
    v_max_starts INTEGER := GREATEST(COALESCE(p_max_starts, 5), 1);
    v_window_seconds INTEGER := GREATEST(COALESCE(p_window_seconds, 120), 10);
    v_backoff_cap_seconds INTEGER := GREATEST(COALESCE(p_backoff_cap_seconds, 300), 10);
    v_recent_count INTEGER := 0;
    v_excess INTEGER := 0;
    v_backoff INTEGER := 0;
BEGIN
    IF v_mode NOT IN ('heartbeat', 'maintenance', 'both', 'channel', 'unknown') THEN
        v_mode := 'unknown';
    END IF;

    IF jsonb_typeof(v_metadata->'process_id') = 'number' THEN
        v_process_id := (v_metadata->>'process_id')::int;
    END IF;
    v_host_name := NULLIF(v_metadata->>'host_name', '');

    INSERT INTO worker_start_events (
        mode,
        instance_name,
        process_id,
        host_name,
        metadata
    ) VALUES (
        v_mode,
        NULLIF(p_instance_name, ''),
        v_process_id,
        v_host_name,
        v_metadata
    );

    DELETE FROM worker_start_events
    WHERE started_at < CURRENT_TIMESTAMP - INTERVAL '1 day';

    SELECT COUNT(*) INTO v_recent_count
    FROM worker_start_events
    WHERE mode = v_mode
      AND instance_name IS NOT DISTINCT FROM NULLIF(p_instance_name, '')
      AND started_at >= CURRENT_TIMESTAMP - make_interval(secs => v_window_seconds);

    IF v_recent_count > v_max_starts THEN
        v_excess := LEAST(v_recent_count - v_max_starts, 6);
        v_backoff := LEAST(v_backoff_cap_seconds, (5 * (2 ^ v_excess))::int);
    END IF;

    RETURN jsonb_build_object(
        'storm', v_backoff > 0,
        'count', v_recent_count,
        'max_starts', v_max_starts,
        'window_seconds', v_window_seconds,
        'backoff_seconds', v_backoff
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION mark_worker_instance_seen(
    p_worker_id UUID,
    p_status TEXT DEFAULT 'running',
    p_current_task_type TEXT DEFAULT NULL,
    p_current_task_run_id UUID DEFAULT NULL,
    p_metadata JSONB DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    v_status TEXT := lower(btrim(COALESCE(p_status, 'running')));
    v_updated INT := 0;
BEGIN
    IF v_status NOT IN ('starting', 'running', 'stopping', 'stopped', 'stale', 'terminated') THEN
        v_status := 'running';
    END IF;

    UPDATE worker_instances
    SET last_seen_at = CURRENT_TIMESTAMP,
        status = v_status,
        current_task_type = CASE
            WHEN p_current_task_type IS NULL AND p_current_task_run_id IS NULL THEN current_task_type
            ELSE p_current_task_type
        END,
        current_task_run_id = CASE
            WHEN p_current_task_type IS NULL AND p_current_task_run_id IS NULL THEN current_task_run_id
            ELSE p_current_task_run_id
        END,
        metadata = CASE
            WHEN p_metadata IS NULL THEN metadata
            ELSE metadata || p_metadata
        END
    WHERE id = p_worker_id;

    GET DIAGNOSTICS v_updated = ROW_COUNT;
    RETURN jsonb_build_object('updated', v_updated);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION mark_worker_instance_stopped(
    p_worker_id UUID,
    p_reason TEXT DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    v_updated INT := 0;
    v_metadata JSONB := CASE
        WHEN p_reason IS NULL OR btrim(p_reason) = '' THEN '{}'::jsonb
        ELSE jsonb_build_object('stop_reason', p_reason)
    END;
BEGIN
    UPDATE worker_instances
    SET status = 'stopped',
        stopped_at = CURRENT_TIMESTAMP,
        last_seen_at = CURRENT_TIMESTAMP,
        current_task_type = NULL,
        current_task_run_id = NULL,
        metadata = metadata || v_metadata
    WHERE id = p_worker_id;

    GET DIAGNOSTICS v_updated = ROW_COUNT;
    RETURN jsonb_build_object('updated', v_updated);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION start_worker_task_run(
    p_worker_id UUID,
    p_task_type TEXT,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS UUID AS $$
DECLARE
    v_id UUID;
    v_task_type TEXT := lower(btrim(COALESCE(p_task_type, 'unknown')));
BEGIN
    IF v_task_type = '' THEN
        v_task_type := 'unknown';
    END IF;

    INSERT INTO worker_task_runs (worker_id, task_type, status, metadata)
    VALUES (p_worker_id, v_task_type, 'running', COALESCE(p_metadata, '{}'::jsonb))
    RETURNING id INTO v_id;

    UPDATE worker_instances
    SET last_seen_at = CURRENT_TIMESTAMP,
        status = 'running',
        current_task_type = v_task_type,
        current_task_run_id = v_id
    WHERE id = p_worker_id;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION complete_worker_task_run(
    p_run_id UUID,
    p_result JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB AS $$
DECLARE
    v_worker_id UUID;
    v_updated INT := 0;
BEGIN
    UPDATE worker_task_runs
    SET status = 'completed',
        finished_at = CURRENT_TIMESTAMP,
        result = COALESCE(p_result, '{}'::jsonb),
        error = NULL,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_run_id
      AND status = 'running'
    RETURNING worker_id INTO v_worker_id;

    GET DIAGNOSTICS v_updated = ROW_COUNT;

    IF v_worker_id IS NOT NULL THEN
        UPDATE worker_instances
        SET last_seen_at = CURRENT_TIMESTAMP,
            last_success_at = CURRENT_TIMESTAMP,
            current_task_type = CASE WHEN current_task_run_id = p_run_id THEN NULL ELSE current_task_type END,
            current_task_run_id = CASE WHEN current_task_run_id = p_run_id THEN NULL ELSE current_task_run_id END
        WHERE id = v_worker_id;
    END IF;

    RETURN jsonb_build_object('updated', v_updated);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION fail_worker_task_run(
    p_run_id UUID,
    p_error TEXT,
    p_result JSONB DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    v_worker_id UUID;
    v_updated INT := 0;
BEGIN
    UPDATE worker_task_runs
    SET status = 'failed',
        finished_at = CURRENT_TIMESTAMP,
        result = p_result,
        error = LEFT(COALESCE(p_error, 'worker task failed'), 4000),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_run_id
      AND status = 'running'
    RETURNING worker_id INTO v_worker_id;

    GET DIAGNOSTICS v_updated = ROW_COUNT;

    IF v_worker_id IS NOT NULL THEN
        UPDATE worker_instances
        SET last_seen_at = CURRENT_TIMESTAMP,
            last_error_at = CURRENT_TIMESTAMP,
            current_task_type = CASE WHEN current_task_run_id = p_run_id THEN NULL ELSE current_task_type END,
            current_task_run_id = CASE WHEN current_task_run_id = p_run_id THEN NULL ELSE current_task_run_id END
        WHERE id = v_worker_id;
    END IF;

    RETURN jsonb_build_object('updated', v_updated);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION discard_worker_task_run(
    p_run_id UUID,
    p_result JSONB DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    v_worker_id UUID;
    v_deleted INT := 0;
BEGIN
    SELECT worker_id INTO v_worker_id
    FROM worker_task_runs
    WHERE id = p_run_id;

    DELETE FROM worker_task_runs
    WHERE id = p_run_id
      AND status = 'running';

    GET DIAGNOSTICS v_deleted = ROW_COUNT;

    IF v_worker_id IS NOT NULL THEN
        UPDATE worker_instances
        SET last_seen_at = CURRENT_TIMESTAMP,
            current_task_type = CASE WHEN current_task_run_id = p_run_id THEN NULL ELSE current_task_type END,
            current_task_run_id = CASE WHEN current_task_run_id = p_run_id THEN NULL ELSE current_task_run_id END,
            metadata = CASE
                WHEN p_result IS NULL THEN metadata
                ELSE metadata || jsonb_build_object('last_idle_result', p_result)
            END
        WHERE id = v_worker_id;
    END IF;

    RETURN jsonb_build_object('deleted', v_deleted);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION record_worker_task_outcome(
    p_worker_id UUID,
    p_task_type TEXT,
    p_status TEXT,
    p_started_at TIMESTAMPTZ,
    p_finished_at TIMESTAMPTZ,
    p_result JSONB DEFAULT NULL,
    p_error TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS UUID AS $$
DECLARE
    v_id UUID;
    v_status TEXT := lower(btrim(COALESCE(p_status, 'completed')));
    v_task_type TEXT := lower(btrim(COALESCE(p_task_type, 'unknown')));
BEGIN
    IF v_status NOT IN ('completed', 'failed', 'unknown') THEN
        RAISE EXCEPTION 'record_worker_task_outcome status must be completed, failed, or unknown';
    END IF;
    IF v_task_type = '' THEN
        v_task_type := 'unknown';
    END IF;

    INSERT INTO worker_task_runs (
        worker_id,
        task_type,
        status,
        started_at,
        finished_at,
        result,
        error,
        metadata
    ) VALUES (
        p_worker_id,
        v_task_type,
        v_status,
        COALESCE(p_started_at, CURRENT_TIMESTAMP),
        COALESCE(p_finished_at, CURRENT_TIMESTAMP),
        p_result,
        CASE WHEN v_status = 'failed' THEN LEFT(COALESCE(p_error, 'worker task failed'), 4000) ELSE p_error END,
        COALESCE(p_metadata, '{}'::jsonb)
    )
    RETURNING id INTO v_id;

    IF p_worker_id IS NOT NULL THEN
        UPDATE worker_instances
        SET last_seen_at = CURRENT_TIMESTAMP,
            last_success_at = CASE WHEN v_status = 'completed' THEN CURRENT_TIMESTAMP ELSE last_success_at END,
            last_error_at = CASE WHEN v_status = 'failed' THEN CURRENT_TIMESTAMP ELSE last_error_at END,
            current_task_type = NULL,
            current_task_run_id = NULL
        WHERE id = p_worker_id;
    END IF;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION recover_interrupted_worker_runs(
    p_stale_after INTERVAL DEFAULT INTERVAL '10 minutes'
) RETURNS JSONB AS $$
DECLARE
    v_stale_workers INT := 0;
    v_unknown_runs INT := 0;
BEGIN
    WITH stale AS (
        UPDATE worker_instances
        SET status = 'stale',
            current_task_type = NULL,
            current_task_run_id = NULL,
            metadata = metadata || jsonb_build_object('stale_marked_at', CURRENT_TIMESTAMP)
        WHERE status IN ('starting', 'running', 'stopping')
          AND last_seen_at < CURRENT_TIMESTAMP - p_stale_after
        RETURNING id
    )
    SELECT COUNT(*) INTO v_stale_workers FROM stale;

    WITH interrupted AS (
        UPDATE worker_task_runs r
        SET status = 'unknown',
            finished_at = CURRENT_TIMESTAMP,
            error = COALESCE(error, 'worker stopped before reporting task outcome'),
            updated_at = CURRENT_TIMESTAMP
        WHERE r.status = 'running'
          AND (
              r.started_at < CURRENT_TIMESTAMP - p_stale_after
              OR EXISTS (
                  SELECT 1
                  FROM worker_instances wi
                  WHERE wi.id = r.worker_id
                    AND wi.status = 'stale'
              )
          )
        RETURNING id
    )
    SELECT COUNT(*) INTO v_unknown_runs FROM interrupted;

    RETURN jsonb_build_object(
        'stale_workers', v_stale_workers,
        'unknown_runs', v_unknown_runs
    );
END;
$$ LANGUAGE plpgsql;

-- Code/schema version skew (#113): a long-running worker keeps talking to a
-- schema that has moved on without it -- migrations auto-apply on startup,
-- but nothing previously checked or surfaced that the *running* worker code
-- predates the migrations it's now sharing a database with. Each worker
-- records the highest migration version bundled in its OWN db/migrations/
-- at registration time (metadata.bundled_latest_migration); this compares
-- that against the schema's actual latest applied version for every
-- non-stale worker still running.
CREATE OR REPLACE FUNCTION worker_code_schema_skew_report()
RETURNS JSONB
LANGUAGE sql STABLE
AS $$
    WITH latest AS (
        SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1
    ),
    skewed AS (
        SELECT
            w.id AS worker_id,
            w.mode,
            w.instance_name,
            w.build_id,
            NULLIF(w.metadata->>'bundled_latest_migration', '') AS bundled_latest_migration,
            (SELECT version FROM latest) AS db_latest_migration
        FROM worker_runtime_status w
        WHERE NOT w.is_stale
          AND w.status IN ('starting', 'running', 'stopping')
          AND NULLIF(w.metadata->>'bundled_latest_migration', '') IS NOT NULL
          AND (SELECT version FROM latest) IS NOT NULL
          AND NULLIF(w.metadata->>'bundled_latest_migration', '') < (SELECT version FROM latest)
    )
    SELECT jsonb_build_object(
        'db_latest_migration', (SELECT version FROM latest),
        'skewed_workers', COALESCE(jsonb_agg(to_jsonb(skewed)), '[]'::jsonb)
    )
    FROM skewed;
$$;
