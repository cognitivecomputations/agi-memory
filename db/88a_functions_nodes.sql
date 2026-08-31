-- Signed, explicitly paired companion nodes and their invocation queue.
SET search_path = public, ag_catalog, "$user";

INSERT INTO config_defaults (key, value, description) VALUES
    ('node.enabled', 'true'::jsonb,
     'Allow signed companion nodes to file pairing requests and connect after approval'),
    ('node.pairing_ttl_hours', '24'::jsonb,
     'Hours before an unanswered node pairing request expires'),
    ('node.invoke_timeout_seconds', '120'::jsonb,
     'Default bounded wait for a paired node invocation')
ON CONFLICT (key) DO NOTHING;

CREATE OR REPLACE FUNCTION register_node_handshake(
    p_node_id TEXT,
    p_public_key TEXT,
    p_name TEXT,
    p_capabilities JSONB DEFAULT '[]'::jsonb,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_node hexis_nodes%ROWTYPE;
    v_request node_pairing_requests%ROWTYPE;
    v_code TEXT;
    v_outbox UUID;
    v_ttl INTEGER := GREATEST(COALESCE(get_config_int('node.pairing_ttl_hours'), 24), 1);
BEGIN
    IF NOT COALESCE(get_config_bool('node.enabled'), TRUE) THEN
        RETURN jsonb_build_object(
            'approved', FALSE,
            'status', 'disabled',
            'reason', 'Companion nodes are disabled by node.enabled.'
        );
    END IF;
    IF NULLIF(btrim(COALESCE(p_node_id, '')), '') IS NULL
       OR NULLIF(btrim(COALESCE(p_public_key, '')), '') IS NULL THEN
        RETURN jsonb_build_object(
            'approved', FALSE,
            'status', 'invalid_identity',
            'reason', 'The node did not present a complete signed identity.'
        );
    END IF;
    IF jsonb_typeof(COALESCE(p_capabilities, 'null'::jsonb)) <> 'array' THEN
        RETURN jsonb_build_object(
            'approved', FALSE,
            'status', 'invalid_capabilities',
            'reason', 'Node capabilities must be a JSON array.'
        );
    END IF;

    UPDATE node_pairing_requests
    SET status = 'expired', decided_at = CURRENT_TIMESTAMP,
        decision_note = 'Pairing request expired before a decision.'
    WHERE status = 'pending' AND expires_at <= CURRENT_TIMESTAMP;

    SELECT * INTO v_node FROM hexis_nodes WHERE node_id = btrim(p_node_id);
    IF FOUND THEN
        IF v_node.public_key <> btrim(p_public_key) THEN
            RETURN jsonb_build_object(
                'approved', FALSE,
                'status', 'identity_mismatch',
                'reason', 'This node id is already paired to a different signing key. Revoke the old node before pairing a replacement.'
            );
        END IF;
        IF v_node.status = 'revoked' THEN
            RETURN jsonb_build_object(
                'approved', FALSE,
                'status', 'revoked',
                'reason', 'This node identity was revoked. Generate a new identity and pair it explicitly.'
            );
        END IF;
        -- A paired signing key does not grant future capabilities. Removing a
        -- capability is safe; adding one files a fresh, visible approval.
        IF NOT (p_capabilities <@ v_node.capabilities) THEN
            SELECT * INTO v_request
            FROM node_pairing_requests
            WHERE node_id = v_node.node_id AND status = 'pending'
            ORDER BY requested_at DESC LIMIT 1;
            IF NOT FOUND THEN
                v_code := upper(substr(replace(gen_random_uuid()::text, '-', ''), 1, 8));
                INSERT INTO node_pairing_requests (
                    code, node_id, public_key, name, capabilities, expires_at, metadata
                ) VALUES (
                    v_code, v_node.node_id, v_node.public_key,
                    COALESCE(NULLIF(btrim(p_name), ''), v_node.name), p_capabilities,
                    CURRENT_TIMESTAMP + make_interval(hours => v_ttl),
                    COALESCE(p_metadata, '{}'::jsonb) || jsonb_build_object(
                        'capability_escalation', TRUE,
                        'previous_capabilities', v_node.capabilities
                    )
                ) RETURNING * INTO v_request;
                BEGIN
                    v_outbox := queue_outbox_message(
                        format(
                            E'The paired node "%s" wants additional host capabilities: %s.\n\nIts existing access does not expand until you approve this exact change. Pairing code: %s',
                            v_request.name,
                            COALESCE(array_to_string(ARRAY(SELECT jsonb_array_elements_text(v_request.capabilities)), ', '), 'none'),
                            v_request.code
                        ),
                        'node_pairing_request',
                        'node_pairing',
                        jsonb_build_object(
                            'mode', 'web_inbox',
                            'pairing_request_id', v_request.id,
                            'pairing_code', v_request.code,
                            'capability_escalation', TRUE
                        )
                    );
                    UPDATE node_pairing_requests
                    SET outbox_message_id = v_outbox
                    WHERE id = v_request.id;
                EXCEPTION WHEN OTHERS THEN
                    RAISE WARNING 'Could not queue node capability approval: %', SQLERRM;
                END;
            ELSIF v_request.capabilities <> p_capabilities THEN
                RETURN jsonb_build_object(
                    'approved', FALSE,
                    'status', 'capability_change_pending',
                    'reason', 'A different capability change is already awaiting approval. Decide it first, then reconnect.'
                );
            END IF;
            RETURN jsonb_build_object(
                'approved', FALSE,
                'status', 'pairing_required',
                'request_id', v_request.id,
                'code', v_request.code,
                'expires_at', v_request.expires_at,
                'next_step', 'Approve or deny the added node capabilities. Existing access was not expanded.'
            );
        END IF;
        UPDATE hexis_nodes
        SET name = COALESCE(NULLIF(btrim(p_name), ''), name),
            capabilities = p_capabilities,
            updated_at = CURRENT_TIMESTAMP,
            metadata = metadata || COALESCE(p_metadata, '{}'::jsonb)
        WHERE node_id = v_node.node_id;
        RETURN jsonb_build_object(
            'approved', TRUE,
            'status', 'paired',
            'node_id', v_node.node_id
        );
    END IF;

    SELECT * INTO v_request
    FROM node_pairing_requests
    WHERE node_id = btrim(p_node_id) AND status = 'pending'
    ORDER BY requested_at DESC LIMIT 1;

    IF NOT FOUND THEN
        v_code := upper(substr(replace(gen_random_uuid()::text, '-', ''), 1, 8));
        INSERT INTO node_pairing_requests (
            code, node_id, public_key, name, capabilities, expires_at, metadata
        ) VALUES (
            v_code, btrim(p_node_id), btrim(p_public_key),
            COALESCE(NULLIF(btrim(p_name), ''), 'Unnamed node'), p_capabilities,
            CURRENT_TIMESTAMP + make_interval(hours => v_ttl),
            COALESCE(p_metadata, '{}'::jsonb)
        ) RETURNING * INTO v_request;

        BEGIN
            v_outbox := queue_outbox_message(
                format(
                    E'A companion node named "%s" wants access to host capabilities: %s.\n\nNothing can run until you approve this exact signed identity. Pairing code: %s',
                    v_request.name,
                    COALESCE(array_to_string(ARRAY(SELECT jsonb_array_elements_text(v_request.capabilities)), ', '), 'none'),
                    v_request.code
                ),
                'node_pairing_request',
                'node_pairing',
                jsonb_build_object(
                    'mode', 'web_inbox',
                    'pairing_request_id', v_request.id,
                    'pairing_code', v_request.code
                )
            );
            UPDATE node_pairing_requests
            SET outbox_message_id = v_outbox
            WHERE id = v_request.id;
        EXCEPTION WHEN OTHERS THEN
            RAISE WARNING 'Could not queue node pairing notification: %', SQLERRM;
        END;
    ELSIF v_request.public_key <> btrim(p_public_key) THEN
        RETURN jsonb_build_object(
            'approved', FALSE,
            'status', 'identity_mismatch',
            'reason', 'A pending request for this node id has a different signing key.'
        );
    END IF;

    RETURN jsonb_build_object(
        'approved', FALSE,
        'status', 'pairing_required',
        'request_id', v_request.id,
        'code', v_request.code,
        'expires_at', v_request.expires_at,
        'next_step', 'Approve or deny this node from the Hexis inbox. Keep `hexis node run` open; it connects automatically after approval.'
    );
END;
$$;

CREATE OR REPLACE FUNCTION decide_node_pairing(
    p_request TEXT,
    p_decision TEXT,
    p_actor TEXT DEFAULT 'operator',
    p_note TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_request node_pairing_requests%ROWTYPE;
    v_decision TEXT := lower(btrim(COALESCE(p_decision, '')));
BEGIN
    SELECT * INTO v_request
    FROM node_pairing_requests
    WHERE id::text = btrim(COALESCE(p_request, ''))
       OR upper(code) = upper(btrim(COALESCE(p_request, '')))
    ORDER BY requested_at DESC LIMIT 1
    FOR UPDATE;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('ok', FALSE, 'status', 'not_found', 'reason', 'Node pairing request was not found.');
    END IF;
    IF v_request.status <> 'pending' THEN
        RETURN jsonb_build_object(
            'ok', v_request.status = 'approved',
            'status', v_request.status,
            'request_id', v_request.id,
            'node_id', v_request.node_id,
            'reason', 'This pairing request was already decided.'
        );
    END IF;
    IF v_request.expires_at <= CURRENT_TIMESTAMP THEN
        UPDATE node_pairing_requests
        SET status = 'expired', decided_at = CURRENT_TIMESTAMP,
            decided_by = COALESCE(NULLIF(p_actor, ''), 'operator'),
            decision_note = 'Pairing request expired before a decision.'
        WHERE id = v_request.id;
        RETURN jsonb_build_object('ok', FALSE, 'status', 'expired', 'request_id', v_request.id);
    END IF;
    IF v_decision NOT IN ('approve', 'deny') THEN
        RETURN jsonb_build_object('ok', FALSE, 'status', 'invalid_decision', 'reason', 'Decision must be approve or deny.');
    END IF;

    IF v_decision = 'approve' THEN
        INSERT INTO hexis_nodes (
            node_id, public_key, name, capabilities, status, approved_by, metadata
        ) VALUES (
            v_request.node_id, v_request.public_key, v_request.name,
            v_request.capabilities, 'offline',
            COALESCE(NULLIF(p_actor, ''), 'operator'),
            jsonb_build_object('pairing_request_id', v_request.id)
        )
        ON CONFLICT (node_id) DO UPDATE
        SET public_key = CASE
                WHEN hexis_nodes.status = 'revoked' THEN EXCLUDED.public_key
                ELSE hexis_nodes.public_key
            END,
            name = EXCLUDED.name,
            capabilities = EXCLUDED.capabilities,
            status = 'offline',
            revoked_at = NULL,
            approved_at = CURRENT_TIMESTAMP,
            approved_by = EXCLUDED.approved_by,
            updated_at = CURRENT_TIMESTAMP;
    END IF;

    UPDATE node_pairing_requests
    SET status = CASE WHEN v_decision = 'approve' THEN 'approved' ELSE 'denied' END,
        decided_at = CURRENT_TIMESTAMP,
        decided_by = COALESCE(NULLIF(p_actor, ''), 'operator'),
        decision_note = NULLIF(p_note, '')
    WHERE id = v_request.id;
    PERFORM pg_notify('node_pairing_decisions', v_request.id::text);

    RETURN jsonb_build_object(
        'ok', v_decision = 'approve',
        'status', CASE WHEN v_decision = 'approve' THEN 'approved' ELSE 'denied' END,
        'request_id', v_request.id,
        'node_id', v_request.node_id,
        'name', v_request.name,
        'next_step', CASE
            WHEN v_decision = 'approve' THEN 'The signed identity is approved. A waiting node connects automatically; otherwise start `hexis node run`.'
            ELSE 'The request was denied. The node has no access.'
        END
    );
END;
$$;

CREATE OR REPLACE FUNCTION list_node_pairing_requests(
    p_status TEXT DEFAULT 'pending',
    p_limit INTEGER DEFAULT 50
) RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE v_result JSONB;
BEGIN
    SELECT COALESCE(jsonb_agg(to_jsonb(r) ORDER BY r.requested_at DESC), '[]'::jsonb)
    INTO v_result
    FROM (
        SELECT id, code, node_id, name, capabilities, status, requested_at,
               expires_at, decided_at, decided_by, decision_note
        FROM node_pairing_requests
        WHERE NULLIF(lower(btrim(COALESCE(p_status, ''))), '') IS NULL
           OR status = lower(btrim(p_status))
        ORDER BY requested_at DESC
        LIMIT LEAST(GREATEST(COALESCE(p_limit, 50), 1), 200)
    ) r;
    RETURN v_result;
END;
$$;

CREATE OR REPLACE FUNCTION list_hexis_nodes() RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    SELECT COALESCE(jsonb_agg(to_jsonb(n) ORDER BY n.name, n.node_id), '[]'::jsonb)
    FROM (
        SELECT node_id, name, capabilities, status, approved_at, approved_by,
               revoked_at, last_seen_at, metadata
        FROM hexis_nodes
    ) n
$$;

CREATE OR REPLACE FUNCTION mark_node_connection(
    p_node_id TEXT,
    p_public_key TEXT,
    p_connection_id UUID,
    p_online BOOLEAN,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE v_updated INTEGER;
BEGIN
    IF p_connection_id IS NULL THEN
        RETURN jsonb_build_object('updated', FALSE, 'online', p_online,
            'reason', 'A connection id is required.');
    END IF;
    IF p_online THEN
        UPDATE hexis_nodes
        SET status = 'online', connection_id = p_connection_id,
            last_seen_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP,
            metadata = metadata || COALESCE(p_metadata, '{}'::jsonb)
        WHERE node_id = p_node_id
          AND public_key = p_public_key
          AND status <> 'revoked'
          AND (
              status <> 'online'
              OR connection_id IS NULL
              OR connection_id = p_connection_id
              OR last_seen_at < CURRENT_TIMESTAMP - INTERVAL '30 seconds'
          );
    ELSE
        UPDATE hexis_nodes
        SET status = 'offline', connection_id = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE node_id = p_node_id
          AND public_key = p_public_key
          AND status <> 'revoked'
          AND connection_id = p_connection_id;
    END IF;
    GET DIAGNOSTICS v_updated = ROW_COUNT;
    RETURN jsonb_build_object('updated', v_updated > 0, 'online', p_online);
END;
$$;

CREATE OR REPLACE FUNCTION revoke_hexis_node(
    p_node_id TEXT,
    p_actor TEXT DEFAULT 'operator',
    p_reason TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE v_updated INTEGER;
BEGIN
    UPDATE hexis_nodes
    SET status = 'revoked', revoked_at = CURRENT_TIMESTAMP,
        connection_id = NULL,
        updated_at = CURRENT_TIMESTAMP,
        metadata = metadata || jsonb_build_object(
            'revoked_by', COALESCE(NULLIF(p_actor, ''), 'operator'),
            'revoke_reason', p_reason
        )
    WHERE node_id = p_node_id AND status <> 'revoked';
    GET DIAGNOSTICS v_updated = ROW_COUNT;
    UPDATE node_invocations
    SET status = 'cancelled', completed_at = CURRENT_TIMESTAMP,
        error = 'Node access was revoked before execution.'
    WHERE node_id = p_node_id AND status IN ('queued', 'dispatched');
    RETURN jsonb_build_object('revoked', v_updated > 0, 'node_id', p_node_id);
END;
$$;

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

CREATE OR REPLACE FUNCTION claim_node_invocation(p_node_id TEXT) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE v_row node_invocations%ROWTYPE;
BEGIN
    UPDATE node_invocations
    SET status = 'expired', completed_at = CURRENT_TIMESTAMP,
        error = 'Node invocation expired before dispatch.'
    WHERE node_id = p_node_id AND status = 'queued' AND expires_at <= CURRENT_TIMESTAMP;

    SELECT * INTO v_row
    FROM node_invocations
    WHERE node_id = p_node_id AND status = 'queued' AND expires_at > CURRENT_TIMESTAMP
    ORDER BY requested_at
    FOR UPDATE SKIP LOCKED LIMIT 1;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('claimed', FALSE);
    END IF;
    UPDATE node_invocations
    SET status = 'dispatched', dispatched_at = CURRENT_TIMESTAMP
    WHERE id = v_row.id;
    RETURN jsonb_build_object(
        'claimed', TRUE,
        'invocation_id', v_row.id,
        'action', v_row.action,
        'arguments', v_row.arguments,
        'expires_at', v_row.expires_at
    );
END;
$$;

CREATE OR REPLACE FUNCTION complete_node_invocation(
    p_invocation_id UUID,
    p_node_id TEXT,
    p_success BOOLEAN,
    p_result JSONB DEFAULT NULL,
    p_error TEXT DEFAULT NULL,
    p_result_signature TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE v_updated INTEGER;
BEGIN
    UPDATE node_invocations
    SET status = CASE WHEN p_success THEN 'succeeded' ELSE 'failed' END,
        result = p_result,
        error = CASE WHEN p_success THEN NULL ELSE COALESCE(NULLIF(p_error, ''), 'Node action failed.') END,
        result_signature = p_result_signature,
        completed_at = CURRENT_TIMESTAMP
    WHERE id = p_invocation_id AND node_id = p_node_id AND status = 'dispatched';
    GET DIAGNOSTICS v_updated = ROW_COUNT;
    IF v_updated > 0 THEN
        PERFORM pg_notify('node_invocation_results', p_invocation_id::text);
    END IF;
    RETURN jsonb_build_object('updated', v_updated > 0);
END;
$$;

CREATE OR REPLACE FUNCTION get_node_invocation(p_invocation_id UUID) RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    SELECT CASE WHEN i.id IS NULL THEN NULL ELSE jsonb_build_object(
        'invocation_id', i.id,
        'node_id', i.node_id,
        'action', i.action,
        'status', i.status,
        'result', i.result,
        'error', i.error,
        'requested_at', i.requested_at,
        'completed_at', i.completed_at,
        'expires_at', i.expires_at
    ) END
    FROM (SELECT 1) seed
    LEFT JOIN node_invocations i ON i.id = p_invocation_id
$$;
