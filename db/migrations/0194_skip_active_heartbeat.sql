-- Do not enqueue a second heartbeat while a previous heartbeat is still active.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

CREATE OR REPLACE FUNCTION should_run_heartbeat()
RETURNS BOOLEAN AS $$
DECLARE
    state_record RECORD;
    interval_minutes FLOAT;
BEGIN
    IF is_agent_terminated() THEN
        RETURN FALSE;
    END IF;
    IF NOT is_agent_configured() THEN
        RETURN FALSE;
    END IF;
    IF NOT is_init_complete() THEN
        RETURN FALSE;
    END IF;

    SELECT * INTO state_record FROM heartbeat_state WHERE id = 1;
    IF state_record.is_paused THEN
        RETURN FALSE;
    END IF;
    IF state_record.active_heartbeat_id IS NOT NULL THEN
        RETURN FALSE;
    END IF;
    IF state_record.last_heartbeat_at IS NULL THEN
        RETURN TRUE;
    END IF;
    interval_minutes := get_config_float('heartbeat.heartbeat_interval_minutes');

    RETURN CURRENT_TIMESTAMP >= state_record.last_heartbeat_at + (interval_minutes || ' minutes')::INTERVAL;
END;
$$ LANGUAGE plpgsql;

SET check_function_bodies = on;
