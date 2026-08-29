-- Durable, exact-once operator approval: Slack actions -> optional iMessage
-- escalation -> one protected tool invocation. This uses the current
-- transactional outbox; it deliberately has no dependency on legacy
-- external_calls state.
SET search_path = public, ag_catalog, "$user";

INSERT INTO config_defaults (key, value, description) VALUES
    ('operator.approval.enabled', 'true'::jsonb,
     'File exact protected-tool approval requests for phone response when no local approver is present.'),
    ('operator.approval.slack_interactive_enabled', 'true'::jsonb,
     'Attach identity-checked Approve and Deny controls to Slack approval DMs.'),
    ('operator.approval.escalate_after_seconds', '300'::jsonb,
     'Seconds to wait for a Slack decision before escalating a still-pending approval to iMessage.'),
    ('operator.approval.wait_seconds', '900'::jsonb,
     'Maximum seconds a live tool call waits for a phone approval before expiring.'),
    ('operator.approval.max_escalation_attempts', '3'::jsonb,
     'Maximum failed iMessage delivery attempts for one approval request.')
ON CONFLICT (key) DO UPDATE SET
    value = EXCLUDED.value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

CREATE OR REPLACE FUNCTION create_operator_tool_approval_request(
    p_request_id UUID,
    p_tool_name TEXT,
    p_arguments JSONB,
    p_arguments_preview JSONB,
    p_tool_context TEXT,
    p_session_id TEXT,
    p_heartbeat_id TEXT,
    p_surface TEXT,
    p_message TEXT,
    p_presentation JSONB,
    p_wait_seconds INTEGER DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_slack_user TEXT;
    v_wait_seconds INTEGER;
    v_escalate_seconds INTEGER;
    v_outbox_id UUID;
    v_envelope JSONB;
    v_status TEXT;
BEGIN
    IF NOT COALESCE(get_config_bool('operator.approval.enabled'), TRUE) THEN
        RETURN jsonb_build_object(
            'created', FALSE,
            'routed', FALSE,
            'status', 'disabled',
            'reason', 'Phone approvals are disabled.',
            'next_step', 'Enable operator.approval.enabled or use an interactive terminal approver.'
        );
    END IF;
    IF p_request_id IS NULL OR NULLIF(btrim(p_tool_name), '') IS NULL THEN
        RAISE EXCEPTION 'request id and tool name are required';
    END IF;
    IF lower(COALESCE(p_tool_context, '')) NOT IN ('chat', 'heartbeat', 'mcp') THEN
        RAISE EXCEPTION 'invalid tool context: %', p_tool_context;
    END IF;

    v_wait_seconds := LEAST(GREATEST(COALESCE(
        p_wait_seconds,
        get_config_int('operator.approval.wait_seconds'),
        900
    ), 60), 3600);
    v_escalate_seconds := LEAST(GREATEST(COALESCE(
        get_config_int('operator.approval.escalate_after_seconds'), 300
    ), 60), v_wait_seconds);
    v_slack_user := NULLIF(btrim(get_config_text('channel.slack.operator_user_id')), '');
    v_status := CASE WHEN v_slack_user IS NULL THEN 'unrouted' ELSE 'pending' END;

    INSERT INTO operator_tool_approval_requests (
        id, tool_name, arguments_hash, arguments_preview, tool_context,
        session_id, heartbeat_id, surface, status, slack_user_id,
        escalate_after, expires_at, delivery_error, metadata
    ) VALUES (
        p_request_id,
        btrim(p_tool_name),
        encode(digest(convert_to(COALESCE(p_arguments, '{}'::jsonb)::text, 'UTF8'), 'sha256'), 'hex'),
        COALESCE(p_arguments_preview, '{}'::jsonb),
        lower(p_tool_context),
        NULLIF(btrim(p_session_id), ''),
        NULLIF(btrim(p_heartbeat_id), ''),
        COALESCE(NULLIF(btrim(p_surface), ''), lower(p_tool_context)),
        v_status,
        v_slack_user,
        CURRENT_TIMESTAMP + make_interval(secs => v_escalate_seconds),
        CURRENT_TIMESTAMP + make_interval(secs => v_wait_seconds),
        CASE WHEN v_slack_user IS NULL
             THEN 'channel.slack.operator_user_id is not configured'
             ELSE NULL END,
        jsonb_build_object('approval_contract', 'exact_once_v1')
    );

    IF v_slack_user IS NOT NULL THEN
        v_outbox_id := gen_random_uuid();
        v_envelope := build_outbox_message(
            'operator_approval',
            jsonb_build_object(
                'message', p_message,
                'intent', 'approval',
                'context', jsonb_build_object(
                    'requires_response', TRUE,
                    'approval_request_id', p_request_id::text,
                    'tool_name', p_tool_name
                ),
                'presentation', COALESCE(p_presentation, '{}'::jsonb),
                'delivery_mode', 'direct',
                'target_channel', 'slack',
                'target_id', v_slack_user,
                'approval_request_id', p_request_id::text
            )
        );
        INSERT INTO outbox_messages (id, envelope, source)
        VALUES (v_outbox_id, v_envelope, 'operator_approval');
        UPDATE operator_tool_approval_requests
        SET outbox_message_id = v_outbox_id, updated_at = CURRENT_TIMESTAMP
        WHERE id = p_request_id;
    END IF;

    RETURN jsonb_build_object(
        'created', TRUE,
        'routed', v_slack_user IS NOT NULL,
        'request_id', p_request_id,
        'status', v_status,
        'expires_at', CURRENT_TIMESTAMP + make_interval(secs => v_wait_seconds),
        'next_step', CASE WHEN v_slack_user IS NULL THEN
            'Run hexis channels setup slack and provide the operator Slack user ID, then restart hexis-channels.'
            ELSE NULL END
    );
END;
$$;

CREATE OR REPLACE FUNCTION mark_operator_tool_approval_slack_delivered(
    p_request_id UUID,
    p_channel_id TEXT,
    p_message_ts TEXT
) RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE operator_tool_approval_requests
    SET status = 'slack_delivered',
        slack_channel_id = NULLIF(btrim(p_channel_id), ''),
        slack_message_ts = NULLIF(btrim(p_message_ts), ''),
        slack_delivered_at = CURRENT_TIMESTAMP,
        delivery_error = NULL,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_request_id
      AND status = 'pending'
      AND expires_at > CURRENT_TIMESTAMP;
    RETURN FOUND;
END;
$$;

CREATE OR REPLACE FUNCTION record_operator_tool_approval_decision(
    p_request_id UUID,
    p_decision TEXT,
    p_channel TEXT,
    p_actor TEXT,
    p_text TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_decision TEXT := lower(btrim(COALESCE(p_decision, '')));
    v_channel TEXT := lower(btrim(COALESCE(p_channel, '')));
    v_actor TEXT := btrim(COALESCE(p_actor, ''));
    v_expected TEXT;
    v_status TEXT;
    v_row operator_tool_approval_requests%ROWTYPE;
BEGIN
    IF NOT COALESCE(get_config_bool('operator.approval.enabled'), TRUE) THEN
        RETURN jsonb_build_object('ok', FALSE, 'error', 'approval_disabled');
    END IF;
    IF v_decision NOT IN ('approve', 'deny') THEN
        RETURN jsonb_build_object('ok', FALSE, 'error', 'invalid_decision');
    END IF;
    IF v_channel = 'slack' THEN
        v_expected := NULLIF(btrim(get_config_text('channel.slack.operator_user_id')), '');
    ELSIF v_channel = 'imessage' THEN
        v_expected := NULLIF(btrim(get_config_text('channel.imessage.operator_recipient')), '');
    ELSE
        RETURN jsonb_build_object('ok', FALSE, 'error', 'unsupported_channel');
    END IF;
    IF v_expected IS NULL OR lower(v_actor) <> lower(v_expected) THEN
        RETURN jsonb_build_object('ok', FALSE, 'error', 'unauthorized_actor');
    END IF;

    v_status := CASE v_decision WHEN 'approve' THEN 'approved' ELSE 'denied' END;
    UPDATE operator_tool_approval_requests
    SET status = v_status,
        decision_channel = v_channel,
        decision_actor = v_actor,
        decision_at = CURRENT_TIMESTAMP,
        delivery_error = NULL,
        metadata = metadata || jsonb_build_object(
            'decision_text', left(COALESCE(p_text, ''), 200)
        ),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_request_id
      AND status IN ('pending', 'slack_delivered', 'escalating', 'escalated')
      AND expires_at > CURRENT_TIMESTAMP
    RETURNING * INTO v_row;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('ok', FALSE, 'error', 'not_pending_or_expired');
    END IF;
    PERFORM pg_notify('operator_approval_decisions', p_request_id::text);
    RETURN jsonb_build_object(
        'ok', TRUE,
        'request_id', p_request_id,
        'status', v_status,
        'tool_name', v_row.tool_name
    );
END;
$$;

CREATE OR REPLACE FUNCTION get_operator_tool_approval_status(p_request_id UUID)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_row operator_tool_approval_requests%ROWTYPE;
BEGIN
    UPDATE operator_tool_approval_requests
    SET status = 'expired', updated_at = CURRENT_TIMESTAMP
    WHERE id = p_request_id
      AND status IN ('unrouted', 'pending', 'slack_delivered', 'escalating', 'escalated')
      AND expires_at <= CURRENT_TIMESTAMP;

    SELECT * INTO v_row FROM operator_tool_approval_requests WHERE id = p_request_id;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('found', FALSE);
    END IF;
    RETURN jsonb_build_object(
        'found', TRUE,
        'request_id', v_row.id,
        'status', v_row.status,
        'tool_name', v_row.tool_name,
        'decision_channel', v_row.decision_channel,
        'expires_at', v_row.expires_at,
        'delivery_error', v_row.delivery_error
    );
END;
$$;

CREATE OR REPLACE FUNCTION consume_operator_tool_approval(
    p_request_id UUID,
    p_tool_name TEXT,
    p_arguments JSONB,
    p_context JSONB
) RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    v_hash TEXT;
BEGIN
    IF p_request_id IS NULL THEN
        RETURN FALSE;
    END IF;
    v_hash := encode(
        digest(convert_to(COALESCE(p_arguments, '{}'::jsonb)::text, 'UTF8'), 'sha256'),
        'hex'
    );
    UPDATE operator_tool_approval_requests
    SET status = 'consumed',
        consumed_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_request_id
      AND status = 'approved'
      AND consumed_at IS NULL
      AND expires_at > CURRENT_TIMESTAMP
      AND tool_name = p_tool_name
      AND arguments_hash = v_hash
      AND tool_context = lower(COALESCE(p_context->>'tool_context', p_context->>'context', 'chat'))
      AND (session_id IS NULL OR session_id = NULLIF(p_context->>'session_id', ''))
      AND (heartbeat_id IS NULL OR heartbeat_id = NULLIF(p_context->>'heartbeat_id', ''));
    RETURN FOUND;
END;
$$;

CREATE OR REPLACE FUNCTION expire_operator_tool_approval(p_request_id UUID)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE operator_tool_approval_requests
    SET status = 'expired', updated_at = CURRENT_TIMESTAMP
    WHERE id = p_request_id
      AND status IN ('unrouted', 'pending', 'slack_delivered', 'escalating', 'escalated');
    IF FOUND THEN
        PERFORM pg_notify('operator_approval_decisions', p_request_id::text);
    END IF;
    RETURN FOUND;
END;
$$;

CREATE OR REPLACE FUNCTION claim_operator_tool_approval_escalations(p_limit INTEGER DEFAULT 20)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_recipient TEXT;
    v_max_attempts INTEGER;
    v_result JSONB;
BEGIN
    UPDATE operator_tool_approval_requests
    SET status = 'expired', updated_at = CURRENT_TIMESTAMP
    WHERE status IN ('unrouted', 'pending', 'slack_delivered', 'escalating', 'escalated')
      AND expires_at <= CURRENT_TIMESTAMP;

    UPDATE operator_tool_approval_requests
    SET status = CASE WHEN slack_delivered_at IS NULL THEN 'pending' ELSE 'slack_delivered' END,
        delivery_error = 'Recovered an interrupted iMessage escalation; retrying.',
        updated_at = CURRENT_TIMESTAMP
    WHERE status = 'escalating'
      AND updated_at <= CURRENT_TIMESTAMP - INTERVAL '2 minutes'
      AND expires_at > CURRENT_TIMESTAMP;

    v_recipient := NULLIF(btrim(get_config_text('channel.imessage.operator_recipient')), '');
    IF v_recipient IS NULL THEN
        RETURN '[]'::jsonb;
    END IF;
    v_max_attempts := GREATEST(COALESCE(
        get_config_int('operator.approval.max_escalation_attempts'), 3
    ), 1);
    WITH candidate AS (
        SELECT id
        FROM operator_tool_approval_requests
        WHERE status IN ('pending', 'slack_delivered')
          AND escalate_after <= CURRENT_TIMESTAMP
          AND expires_at > CURRENT_TIMESTAMP
          AND escalation_attempts < v_max_attempts
        ORDER BY escalate_after, created_at
        FOR UPDATE SKIP LOCKED
        LIMIT GREATEST(1, LEAST(COALESCE(p_limit, 20), 100))
    ), claimed AS (
        UPDATE operator_tool_approval_requests r
        SET status = 'escalating',
            escalation_attempts = escalation_attempts + 1,
            imessage_recipient = v_recipient,
            updated_at = CURRENT_TIMESTAMP
        FROM candidate
        WHERE r.id = candidate.id
        RETURNING r.id, r.tool_name, r.arguments_preview, r.surface,
                  r.expires_at, r.imessage_recipient
    )
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', id,
        'tool_name', tool_name,
        'arguments_preview', arguments_preview,
        'surface', surface,
        'expires_at', expires_at,
        'recipient', imessage_recipient
    )), '[]'::jsonb)
    INTO v_result
    FROM claimed;
    RETURN v_result;
END;
$$;

CREATE OR REPLACE FUNCTION complete_operator_tool_approval_escalation(
    p_request_id UUID,
    p_message_id TEXT
) RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE operator_tool_approval_requests
    SET status = 'escalated',
        imessage_message_id = NULLIF(btrim(p_message_id), ''),
        escalated_at = CURRENT_TIMESTAMP,
        delivery_error = NULL,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_request_id AND status = 'escalating';
    RETURN FOUND;
END;
$$;

CREATE OR REPLACE FUNCTION fail_operator_tool_approval_escalation(
    p_request_id UUID,
    p_error TEXT
) RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE operator_tool_approval_requests
    SET status = CASE WHEN slack_delivered_at IS NULL THEN 'pending' ELSE 'slack_delivered' END,
        escalate_after = CURRENT_TIMESTAMP + INTERVAL '60 seconds',
        delivery_error = left(COALESCE(p_error, 'iMessage delivery failed'), 1000),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_request_id AND status = 'escalating';
    RETURN FOUND;
END;
$$;

CREATE OR REPLACE FUNCTION try_resolve_operator_tool_approval_from_inbound(
    p_channel TEXT,
    p_actor TEXT,
    p_text TEXT
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_match TEXT[];
    v_decision TEXT;
    v_code TEXT;
    v_request_id UUID;
    v_count INTEGER;
    v_result JSONB;
    v_expected TEXT;
BEGIN
    v_match := regexp_match(
        lower(btrim(COALESCE(p_text, ''))),
        '^(approve|approved|yes|y|👍|deny|denied|no|n)(?:[[:space:]]+([0-9a-f]{8}))?[[:space:]]*$'
    );
    IF v_match IS NULL THEN
        RETURN jsonb_build_object(
            'recognized', FALSE, 'matched', FALSE, 'reason', 'not_a_decision'
        );
    END IF;
    v_decision := CASE WHEN v_match[1] IN ('deny', 'denied', 'no', 'n')
                       THEN 'deny' ELSE 'approve' END;
    v_code := NULLIF(v_match[2], '');

    IF lower(COALESCE(p_channel, '')) = 'slack' THEN
        v_expected := NULLIF(btrim(get_config_text('channel.slack.operator_user_id')), '');
    ELSIF lower(COALESCE(p_channel, '')) = 'imessage' THEN
        v_expected := NULLIF(btrim(get_config_text('channel.imessage.operator_recipient')), '');
    ELSE
        RETURN jsonb_build_object(
            'recognized', FALSE, 'matched', FALSE, 'reason', 'unsupported_channel'
        );
    END IF;
    IF v_expected IS NULL OR lower(btrim(COALESCE(p_actor, ''))) <> lower(v_expected) THEN
        RETURN jsonb_build_object(
            'recognized', FALSE, 'matched', FALSE, 'reason', 'unauthorized_actor'
        );
    END IF;

    IF v_code IS NOT NULL THEN
        SELECT count(*) INTO v_count
        FROM operator_tool_approval_requests
        WHERE status IN ('pending', 'slack_delivered', 'escalating', 'escalated')
          AND expires_at > CURRENT_TIMESTAMP
          AND replace(id::text, '-', '') LIKE v_code || '%';
        IF v_count <> 1 THEN
            RETURN jsonb_build_object(
                'recognized', TRUE,
                'matched', FALSE,
                'reason', CASE WHEN v_count = 0 THEN 'request_not_found' ELSE 'ambiguous_code' END,
                'pending_count', v_count,
                'message', CASE WHEN v_count = 0 THEN
                    'That approval code is not pending. Use the eight-character code from the latest approval message.'
                    ELSE 'That short code matches more than one pending action. Use the Slack buttons to decide safely.' END
            );
        END IF;
        SELECT id INTO v_request_id
        FROM operator_tool_approval_requests
        WHERE status IN ('pending', 'slack_delivered', 'escalating', 'escalated')
          AND expires_at > CURRENT_TIMESTAMP
          AND replace(id::text, '-', '') LIKE v_code || '%'
        ORDER BY created_at DESC
        LIMIT 1;
    ELSE
        SELECT count(*) INTO v_count
        FROM operator_tool_approval_requests
        WHERE status IN ('pending', 'slack_delivered', 'escalating', 'escalated')
          AND expires_at > CURRENT_TIMESTAMP;
        IF v_count <> 1 THEN
            RETURN jsonb_build_object(
                'recognized', TRUE,
                'matched', FALSE,
                'reason', CASE WHEN v_count = 0 THEN 'no_pending_request' ELSE 'ambiguous_without_code' END,
                'pending_count', v_count,
                'message', CASE WHEN v_count = 0 THEN
                    'There is no pending protected action to approve or deny.'
                    ELSE 'More than one protected action is pending. Reply with approve CODE or deny CODE from the approval message.' END
            );
        END IF;
        SELECT id INTO v_request_id
        FROM operator_tool_approval_requests
        WHERE status IN ('pending', 'slack_delivered', 'escalating', 'escalated')
          AND expires_at > CURRENT_TIMESTAMP
        ORDER BY created_at DESC
        LIMIT 1;
    END IF;
    IF v_request_id IS NULL THEN
        RETURN jsonb_build_object(
            'recognized', TRUE,
            'matched', FALSE,
            'reason', 'request_not_found',
            'message', 'That approval code is not pending. Use the eight-character code from the latest approval message.'
        );
    END IF;

    v_result := record_operator_tool_approval_decision(
        v_request_id, v_decision, p_channel, p_actor, p_text
    );
    IF NOT COALESCE((v_result->>'ok')::boolean, FALSE) THEN
        RETURN v_result || jsonb_build_object(
            'recognized', TRUE,
            'matched', FALSE,
            'message', 'That protected action is no longer pending; ask Hexis to try it again if you still want it.'
        );
    END IF;
    RETURN v_result || jsonb_build_object(
        'recognized', TRUE,
        'matched', TRUE,
        'message', initcap(v_result->>'status') || ' ' || (v_result->>'tool_name') ||
            ' (' || left(replace(v_request_id::text, '-', ''), 8) || ').'
    );
END;
$$;

CREATE OR REPLACE FUNCTION list_pending_operator_tool_approvals(p_limit INTEGER DEFAULT 50)
RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', id,
        'code', left(replace(id::text, '-', ''), 8),
        'tool_name', tool_name,
        'arguments_preview', arguments_preview,
        'tool_context', tool_context,
        'surface', surface,
        'status', status,
        'created_at', created_at,
        'expires_at', expires_at,
        'delivery_error', delivery_error
    ) ORDER BY created_at DESC), '[]'::jsonb)
    FROM (
        SELECT * FROM operator_tool_approval_requests
        WHERE status IN ('unrouted', 'pending', 'slack_delivered', 'escalating', 'escalated', 'approved')
          AND expires_at > CURRENT_TIMESTAMP
        ORDER BY created_at DESC
        LIMIT GREATEST(1, LEAST(COALESCE(p_limit, 50), 200))
    ) pending;
$$;
