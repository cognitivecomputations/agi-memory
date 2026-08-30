-- Gmail sync recovery (#110): a one-shot backfill (default max_messages=100)
-- previously completed "truncated" and just sat there forever -- nothing
-- re-enqueued a successor, and no ongoing sync mechanism was ever populated
-- (the ambient_responsibilities Gmail evaluator has been fully implemented
-- since services/ambient_responsibilities.py landed, but zero rows ever
-- existed to invoke it). This closes both recovery paths named in the issue:
--   (a) auto-create the ongoing-sync ambient responsibility whenever a
--       Gmail backfill is queued, and
--   (b) re-enqueue a successor job when a completed job reports truncated.
-- True incremental sync via Gmail's history.list API (the issue's third,
-- larger suggested fix) is not attempted here -- it's a bigger design/testing
-- effort on its own. The ambient responsibility's poll query already scopes
-- to messages after the last check, so this closes the "stuck forever" bug
-- even without it.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

INSERT INTO config_defaults (key, value, description) VALUES
    (
        'integrations.gmail.ambient_poll_interval_seconds',
        '900'::jsonb,
        'How often the ongoing-sync ambient responsibility (#110) re-checks Gmail for new messages after a backfill runs.'
    )
ON CONFLICT (key) DO UPDATE SET
    value = EXCLUDED.value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

-- Ongoing sync (#110): a one-shot backfill only ever ingests the first page
-- it's asked for. This is the "keep watching after the backfill" half --
-- an ambient_responsibilities row with a gmail source, evaluated on the
-- normal ambient poll cadence (services/ambient_responsibilities.py's
-- _evaluate_gmail, already implemented, was just never given anything to
-- watch). Idempotent: returns the existing row if one already covers this
-- account. Never raises -- an ambient-responsibility hiccup must not block
-- the backfill that's actually being requested.
CREATE OR REPLACE FUNCTION ensure_gmail_ambient_responsibility(
    p_account_key TEXT
) RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    v_key TEXT := lower(COALESCE(NULLIF(btrim(p_account_key), ''), ''));
    v_id UUID;
    v_result JSONB;
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_proc WHERE proname = 'create_ambient_responsibility'
    ) THEN
        RETURN NULL;  -- ambient_responsibilities not migrated in yet
    END IF;

    SELECT id INTO v_id
    FROM ambient_responsibilities
    WHERE status IN ('active', 'proposed', 'paused', 'blocked')
      AND sources @> jsonb_build_array(
          jsonb_build_object('connector_id', 'gmail', 'account_key', v_key)
      )
    LIMIT 1;
    IF v_id IS NOT NULL THEN
        RETURN v_id;
    END IF;

    v_result := create_ambient_responsibility(jsonb_build_object(
        'title', 'Ongoing Gmail sync',
        'user_intent', 'Keep ingesting new Gmail messages as they arrive, so memory of email stays current without a manual re-backfill.',
        'kind', 'monitor',
        'created_by', 'system',
        'memory_policy', 'task_scoped',
        'sources', jsonb_build_array(
            jsonb_build_object('connector_id', 'gmail', 'account_key', v_key)
        ),
        'trigger', jsonb_build_object(
            'kind', 'interval',
            'every_seconds', COALESCE(get_config_int('integrations.gmail.ambient_poll_interval_seconds'), 900)
        )
    ));
    RETURN NULLIF(v_result->'output'->>'responsibility_id', '')::uuid;
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'ensure_gmail_ambient_responsibility failed for %: %', v_key, SQLERRM;
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION enqueue_connector_backfill_job(
    p_connector_id TEXT,
    p_account_key TEXT,
    p_cursor_key TEXT DEFAULT 'messages',
    p_requested_range JSONB DEFAULT '{}'::jsonb,
    p_metadata JSONB DEFAULT '{}'::jsonb,
    p_max_attempts INT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    row_connection integration_connections%ROWTYPE;
    row_job connector_backfill_jobs%ROWTYPE;
    normalized_cursor TEXT := COALESCE(NULLIF(btrim(p_cursor_key), ''), 'messages');
    existing_id UUID;
BEGIN
    row_connection := _connector_connection(p_connector_id, p_account_key);
    PERFORM ensure_connector_cursor(
        row_connection.connector_id,
        row_connection.account_key,
        normalized_cursor,
        COALESCE(p_metadata, '{}'::jsonb)
    );
    IF row_connection.connector_id = 'gmail' THEN
        PERFORM ensure_gmail_ambient_responsibility(row_connection.account_key);
    END IF;

    SELECT id
    INTO existing_id
    FROM connector_backfill_jobs
    WHERE connection_id = row_connection.id
      AND cursor_key = normalized_cursor
      AND status IN ('pending', 'in_progress', 'paused')
    ORDER BY created_at DESC
    LIMIT 1;

    IF existing_id IS NOT NULL THEN
        SELECT * INTO row_job FROM connector_backfill_jobs WHERE id = existing_id;
        RETURN jsonb_build_object(
            'job_id', row_job.id::text,
            'existing', TRUE,
            'status', row_job.status,
            'connector_id', row_job.connector_id,
            'account_key', row_job.account_key,
            'cursor_key', row_job.cursor_key,
            'requested_range', row_job.requested_range,
            'estimate', estimate_connector_backfill(row_job.connector_id, row_job.requested_range),
            'progress', row_job.progress
        );
    END IF;

    INSERT INTO connector_backfill_jobs (
        connection_id,
        connector_id,
        account_key,
        cursor_key,
        requested_range,
        metadata,
        max_attempts
    )
    VALUES (
        row_connection.id,
        row_connection.connector_id,
        row_connection.account_key,
        normalized_cursor,
        COALESCE(p_requested_range, '{}'::jsonb),
        COALESCE(p_metadata, '{}'::jsonb),
        GREATEST(COALESCE(p_max_attempts, 3), 1)
    )
    RETURNING * INTO row_job;

    RETURN jsonb_build_object(
        'job_id', row_job.id::text,
        'existing', FALSE,
        'status', row_job.status,
        'connector_id', row_job.connector_id,
        'account_key', row_job.account_key,
        'cursor_key', row_job.cursor_key,
        'requested_range', row_job.requested_range,
        'estimate', estimate_connector_backfill(row_job.connector_id, row_job.requested_range),
        'progress', row_job.progress,
        'next_attempt_at', row_job.next_attempt_at
    );
END;
$$;

CREATE OR REPLACE FUNCTION complete_connector_backfill_job(
    p_job_id UUID,
    p_result JSONB DEFAULT '{}'::jsonb,
    p_cursor_value JSONB DEFAULT NULL,
    p_high_watermark TIMESTAMPTZ DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    row_job connector_backfill_jobs%ROWTYPE;
BEGIN
    SELECT *
    INTO row_job
    FROM connector_backfill_jobs
    WHERE id = p_job_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('job_id', p_job_id::text, 'status', 'missing');
    END IF;

    IF p_cursor_value IS NOT NULL AND p_cursor_value <> 'null'::jsonb THEN
        PERFORM advance_connector_cursor(
            row_job.connector_id,
            row_job.account_key,
            row_job.cursor_key,
            p_cursor_value,
            p_high_watermark,
            jsonb_build_object('completed_by_job_id', p_job_id::text)
        );
    END IF;

    UPDATE connector_backfill_jobs
    SET status = 'completed',
        result = COALESCE(p_result, '{}'::jsonb),
        error = NULL,
        completed_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_job_id
    RETURNING * INTO row_job;

    UPDATE connector_sync_cursors
    SET status = 'active',
        last_completed_at = CURRENT_TIMESTAMP,
        last_error = NULL,
        updated_at = CURRENT_TIMESTAMP
    WHERE connection_id = row_job.connection_id
      AND cursor_key = row_job.cursor_key;

    -- Self-continuing backfill (#110): a capped job (max_messages) that hit
    -- its cap and still has a page_token left (result.truncated) previously
    -- just sat there "completed" forever -- nothing re-picked it up, so a
    -- one-shot 100-message backfill was permanent. The cursor above already
    -- carries the page_token forward; this just makes sure a job exists to
    -- consume it on the next maintenance tick. enqueue_connector_backfill_job
    -- is idempotent (one active job per connection+cursor_key), so this is
    -- safe even if something else already queued a successor.
    IF COALESCE((p_result->>'truncated')::boolean, FALSE) THEN
        PERFORM enqueue_connector_backfill_job(
            row_job.connector_id,
            row_job.account_key,
            row_job.cursor_key,
            row_job.requested_range,
            row_job.metadata,
            row_job.max_attempts
        );
    END IF;

    RETURN jsonb_build_object(
        'job_id', row_job.id::text,
        'status', row_job.status,
        'result', row_job.result
    );
END;
$$;
