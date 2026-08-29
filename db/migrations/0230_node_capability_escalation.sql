-- A paired signing key must not silently acquire newly advertised capabilities.
SET search_path = public, ag_catalog, "$user";

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
            'approved', FALSE, 'status', 'disabled',
            'reason', 'Companion nodes are disabled by node.enabled.'
        );
    END IF;
    IF NULLIF(btrim(COALESCE(p_node_id, '')), '') IS NULL
       OR NULLIF(btrim(COALESCE(p_public_key, '')), '') IS NULL THEN
        RETURN jsonb_build_object(
            'approved', FALSE, 'status', 'invalid_identity',
            'reason', 'The node did not present a complete signed identity.'
        );
    END IF;
    IF jsonb_typeof(COALESCE(p_capabilities, 'null'::jsonb)) <> 'array' THEN
        RETURN jsonb_build_object(
            'approved', FALSE, 'status', 'invalid_capabilities',
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
                'approved', FALSE, 'status', 'identity_mismatch',
                'reason', 'This node id is already paired to a different signing key. Revoke the old node before pairing a replacement.'
            );
        END IF;
        IF v_node.status = 'revoked' THEN
            RETURN jsonb_build_object(
                'approved', FALSE, 'status', 'revoked',
                'reason', 'This node identity was revoked. Generate a new identity and pair it explicitly.'
            );
        END IF;
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
                        'node_pairing_request', 'node_pairing',
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
            'approved', TRUE, 'status', 'paired', 'node_id', v_node.node_id
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
                'node_pairing_request', 'node_pairing',
                jsonb_build_object(
                    'mode', 'web_inbox',
                    'pairing_request_id', v_request.id,
                    'pairing_code', v_request.code
                )
            );
            UPDATE node_pairing_requests SET outbox_message_id = v_outbox
            WHERE id = v_request.id;
        EXCEPTION WHEN OTHERS THEN
            RAISE WARNING 'Could not queue node pairing notification: %', SQLERRM;
        END;
    ELSIF v_request.public_key <> btrim(p_public_key) THEN
        RETURN jsonb_build_object(
            'approved', FALSE, 'status', 'identity_mismatch',
            'reason', 'A pending request for this node id has a different signing key.'
        );
    END IF;
    RETURN jsonb_build_object(
        'approved', FALSE, 'status', 'pairing_required',
        'request_id', v_request.id, 'code', v_request.code,
        'expires_at', v_request.expires_at,
        'next_step', 'Approve or deny this node from the Hexis inbox. Keep `hexis node run` open; it connects automatically after approval.'
    );
END;
$$;
