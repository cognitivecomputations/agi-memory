-- Structured Wave C host actions: fixed Apple automation and secret-safe 1Password.
SET search_path = public, ag_catalog, "$user";

ALTER TABLE node_invocations
    DROP CONSTRAINT IF EXISTS node_invocations_action_check;
ALTER TABLE node_invocations
    ADD CONSTRAINT node_invocations_action_check CHECK (action IN (
        'system.run', 'screen.capture',
        'apple.reminders.list', 'apple.reminders.create',
        'apple.notes.search', 'apple.notes.create',
        'apple.calendar.list', 'apple.calendar.create',
        'apple.shortcuts.list', 'apple.shortcuts.run',
        'onepassword.items', 'onepassword.copy'
    ));

CREATE OR REPLACE FUNCTION create_node_invocation(
    p_node_id TEXT,
    p_action TEXT,
    p_arguments JSONB DEFAULT '{}'::jsonb,
    p_requested_by TEXT DEFAULT 'agent',
    p_timeout_seconds INTEGER DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_node hexis_nodes%ROWTYPE;
    v_id UUID;
    v_timeout INTEGER := LEAST(GREATEST(COALESCE(
        p_timeout_seconds, get_config_int('node.invoke_timeout_seconds'), 120
    ), 5), 300);
BEGIN
    SELECT * INTO v_node FROM hexis_nodes WHERE node_id = p_node_id;
    IF NOT FOUND OR v_node.status = 'revoked' THEN
        RETURN jsonb_build_object('queued', FALSE, 'status', 'unavailable', 'reason', 'The requested node is not paired.');
    END IF;
    IF v_node.status <> 'online'
       OR v_node.last_seen_at IS NULL
       OR v_node.last_seen_at < CURRENT_TIMESTAMP - INTERVAL '30 seconds' THEN
        RETURN jsonb_build_object(
            'queued', FALSE,
            'status', 'offline',
            'reason', format('Node "%s" is offline. Start `hexis node run` on that device, then retry.', v_node.name)
        );
    END IF;
    IF p_action NOT IN (
        'system.run', 'screen.capture',
        'apple.reminders.list', 'apple.reminders.create',
        'apple.notes.search', 'apple.notes.create',
        'apple.calendar.list', 'apple.calendar.create',
        'apple.shortcuts.list', 'apple.shortcuts.run',
        'onepassword.items', 'onepassword.copy'
    )
       OR NOT (v_node.capabilities ? p_action) THEN
        RETURN jsonb_build_object(
            'queued', FALSE,
            'status', 'unsupported',
            'reason', format('Node "%s" did not advertise capability %s.', v_node.name, p_action)
        );
    END IF;

    INSERT INTO node_invocations (
        node_id, action, arguments, requested_by, expires_at, metadata
    ) VALUES (
        p_node_id, p_action, COALESCE(p_arguments, '{}'::jsonb),
        COALESCE(NULLIF(p_requested_by, ''), 'agent'),
        CURRENT_TIMESTAMP + make_interval(secs => v_timeout),
        COALESCE(p_metadata, '{}'::jsonb)
    ) RETURNING id INTO v_id;
    PERFORM pg_notify('node_invocations', p_node_id);
    RETURN jsonb_build_object(
        'queued', TRUE, 'status', 'queued', 'invocation_id', v_id,
        'timeout_seconds', v_timeout
    );
END;
$$;
