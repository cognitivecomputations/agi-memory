-- Phase 3: durable phone approvals for exact, one-shot protected tool calls.
SET search_path = public, ag_catalog, "$user";

CREATE TABLE IF NOT EXISTS operator_tool_approval_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tool_name TEXT NOT NULL,
    arguments_hash TEXT NOT NULL,
    arguments_preview JSONB NOT NULL DEFAULT '{}'::jsonb,
    tool_context TEXT NOT NULL CHECK (tool_context IN ('chat', 'heartbeat', 'mcp')),
    session_id TEXT,
    heartbeat_id TEXT,
    surface TEXT NOT NULL DEFAULT 'chat',
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'unrouted', 'pending', 'slack_delivered', 'escalating', 'escalated',
        'approved', 'denied', 'consumed', 'expired'
    )),
    slack_user_id TEXT,
    slack_channel_id TEXT,
    slack_message_ts TEXT,
    slack_delivered_at TIMESTAMPTZ,
    escalate_after TIMESTAMPTZ,
    escalation_attempts INTEGER NOT NULL DEFAULT 0,
    imessage_recipient TEXT,
    imessage_message_id TEXT,
    escalated_at TIMESTAMPTZ,
    decision_channel TEXT,
    decision_actor TEXT,
    decision_at TIMESTAMPTZ,
    consumed_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ NOT NULL,
    outbox_message_id UUID,
    delivery_error TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_operator_tool_approvals_pending
    ON operator_tool_approval_requests (expires_at, created_at)
    WHERE status IN ('unrouted', 'pending', 'slack_delivered', 'escalating', 'escalated');
CREATE INDEX IF NOT EXISTS idx_operator_tool_approvals_escalation_due
    ON operator_tool_approval_requests (escalate_after)
    WHERE status IN ('pending', 'slack_delivered');
CREATE INDEX IF NOT EXISTS idx_operator_tool_approvals_session
    ON operator_tool_approval_requests (session_id, created_at DESC)
    WHERE session_id IS NOT NULL;

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

CREATE OR REPLACE FUNCTION channel_setting_names(
    p_channel TEXT
) RETURNS TEXT[] AS $$
DECLARE
    catalog JSONB := '{
        "discord":  ["bot_token", "allowed_guilds"],
        "telegram": ["bot_token", "allowed_chat_ids"],
        "slack":    ["bot_token", "app_token", "signing_secret", "operator_user_id", "allowed_channels"],
        "signal":   ["phone_number", "api_url", "allowed_numbers"],
        "whatsapp": ["access_token", "phone_number_id", "verify_token", "webhook_port", "allowed_numbers"],
        "imessage": ["api_url", "password", "operator_recipient", "allowed_handles"],
        "matrix":   ["homeserver", "user_id", "access_token", "allowed_rooms"]
    }'::jsonb;
BEGIN
    IF NOT catalog ? COALESCE(p_channel, '') THEN
        RAISE EXCEPTION 'Unknown channel type: %; expected one of %',
            COALESCE(p_channel, '(null)'),
            (SELECT string_agg(key, ', ' ORDER BY key) FROM jsonb_object_keys(catalog) key);
    END IF;
    RETURN ARRAY(SELECT jsonb_array_elements_text(catalog->p_channel));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

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
            'created', FALSE, 'routed', FALSE, 'status', 'disabled',
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
        p_wait_seconds, get_config_int('operator.approval.wait_seconds'), 900
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
        p_request_id, btrim(p_tool_name),
        encode(digest(convert_to(COALESCE(p_arguments, '{}'::jsonb)::text, 'UTF8'), 'sha256'), 'hex'),
        COALESCE(p_arguments_preview, '{}'::jsonb), lower(p_tool_context),
        NULLIF(btrim(p_session_id), ''), NULLIF(btrim(p_heartbeat_id), ''),
        COALESCE(NULLIF(btrim(p_surface), ''), lower(p_tool_context)), v_status,
        v_slack_user,
        CURRENT_TIMESTAMP + make_interval(secs => v_escalate_seconds),
        CURRENT_TIMESTAMP + make_interval(secs => v_wait_seconds),
        CASE WHEN v_slack_user IS NULL
             THEN 'channel.slack.operator_user_id is not configured' ELSE NULL END,
        jsonb_build_object('approval_contract', 'exact_once_v1')
    );

    IF v_slack_user IS NOT NULL THEN
        v_outbox_id := gen_random_uuid();
        v_envelope := build_outbox_message(
            'operator_approval',
            jsonb_build_object(
                'message', p_message, 'intent', 'approval',
                'context', jsonb_build_object(
                    'requires_response', TRUE,
                    'approval_request_id', p_request_id::text,
                    'tool_name', p_tool_name
                ),
                'presentation', COALESCE(p_presentation, '{}'::jsonb),
                'delivery_mode', 'direct', 'target_channel', 'slack',
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
        'created', TRUE, 'routed', v_slack_user IS NOT NULL,
        'request_id', p_request_id, 'status', v_status,
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

CREATE OR REPLACE FUNCTION evaluate_connector_action_call(
    p_tool_name TEXT,
    p_arguments JSONB DEFAULT '{}'::jsonb,
    p_context JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    action JSONB;
    ctx TEXT := lower(COALESCE(p_context->>'tool_context', p_context->>'context', 'chat'));
    row_policy connector_action_policies%ROWTYPE;
    constraint_decision JSONB;
    target TEXT;
    account TEXT;
    explicit_action_approved BOOLEAN := FALSE;
BEGIN
    action := connector_action_for_tool(p_tool_name, COALESCE(p_arguments, '{}'::jsonb));
    IF NOT COALESCE((action->>'action_required')::boolean, FALSE) THEN
        RETURN jsonb_build_object('allowed', TRUE, 'action_required', FALSE);
    END IF;

    target := action->>'target';
    account := NULLIF(lower(btrim(COALESCE(action->>'account_key', ''))), '');
    BEGIN
        explicit_action_approved := COALESCE((p_context->>'action_approved')::boolean, FALSE);
    EXCEPTION WHEN OTHERS THEN
        explicit_action_approved := FALSE;
    END;

    IF explicit_action_approved THEN
        RETURN jsonb_build_object(
            'allowed', TRUE,
            'action_required', TRUE,
            'authorization_kind', 'operator_exact_once_approval',
            'connector_id', action->>'connector_id',
            'account_key', account,
            'action_kind', action->>'action_kind',
            'target', target,
            'sensitivity', action->>'sensitivity',
            'reason', 'identity-checked exact tool arguments approved by the operator'
        );
    END IF;

    FOR row_policy IN
        SELECT *
        FROM connector_action_policies
        WHERE status = 'active'
          AND connector_id = action->>'connector_id'
          AND action_kind = action->>'action_kind'
          AND (account_key IS NULL OR account IS NULL OR account_key = account)
          AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
          AND (COALESCE(array_length(contexts, 1), 0) = 0 OR ctx = ANY(contexts))
        ORDER BY
          CASE WHEN account_key IS NULL THEN 1 ELSE 0 END,
          updated_at DESC
    LOOP
        IF ctx <> 'chat' AND NOT row_policy.allow_autonomous THEN
            CONTINUE;
        END IF;
        IF ctx <> 'chat'
           AND row_policy.requires_per_action_approval
           AND NOT explicit_action_approved THEN
            CONTINUE;
        END IF;

        constraint_decision := connector_action_constraints_match(
            row_policy.constraints,
            target,
            COALESCE(p_arguments, '{}'::jsonb),
            row_policy.id
        );
        IF COALESCE((constraint_decision->>'allowed')::boolean, FALSE) THEN
            RETURN jsonb_build_object(
                'allowed', TRUE,
                'action_required', TRUE,
                'authorization_kind', CASE WHEN ctx = 'chat' THEN 'policy' ELSE 'preauthorized_policy' END,
                'policy_id', row_policy.id::text,
                'connector_id', action->>'connector_id',
                'account_key', account,
                'action_kind', action->>'action_kind',
                'target', target,
                'sensitivity', action->>'sensitivity'
            );
        END IF;
    END LOOP;

    IF ctx = 'chat' THEN
        RETURN jsonb_build_object(
            'allowed', TRUE,
            'action_required', TRUE,
            'authorization_kind', 'interactive_chat_approval',
            'connector_id', action->>'connector_id',
            'account_key', account,
            'action_kind', action->>'action_kind',
            'target', target,
            'sensitivity', action->>'sensitivity',
            'reason', 'interactive chat context supplies per-action approval'
        );
    END IF;

    RETURN jsonb_build_object(
        'allowed', FALSE,
        'action_required', TRUE,
        'error_type', 'approval_required',
        'reason', format(
            'Connector action %s/%s requires a matching preauthorized policy for %s context',
            action->>'connector_id',
            action->>'action_kind',
            ctx
        ),
        'connector_id', action->>'connector_id',
        'account_key', account,
        'action_kind', action->>'action_kind',
        'target', target,
        'sensitivity', action->>'sensitivity'
    );
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
    action_policy JSONB;
    approval_ok BOOLEAN := FALSE;
    approval_request UUID;
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

    action_policy := evaluate_connector_action_call(
        p_tool_name,
        COALESCE(p_arguments, '{}'::jsonb),
        p_context || jsonb_build_object('action_approved', approval_ok)
    );
    IF NOT COALESCE((action_policy->>'allowed')::boolean, FALSE) THEN
        RETURN jsonb_build_object(
            'allowed', false,
            'reason', action_policy->>'reason',
            'error_type', COALESCE(action_policy->>'error_type', 'approval_required'),
            'energy_cost', cost,
            'connector_action', action_policy
        );
    END IF;

    RETURN jsonb_build_object(
        'allowed', true,
        'energy_cost', cost,
        'supports_parallel', tool.supports_parallel,
        'execution_kind', tool.execution_kind,
        'driver', tool.driver,
        'connector_action', action_policy,
        'operator_approval_request_id', CASE WHEN approval_ok THEN approval_request::text ELSE NULL END
    );
END;
$$;

UPDATE integration_connectors
SET capability_manifest = jsonb_set(
        capability_manifest,
        '{send}',
        '{"label":"Send Slack messages and private approval DMs","status":"available","scopes":["chat:write","im:write"]}'::jsonb,
        TRUE
    ),
    setup_manifest = jsonb_set(
        jsonb_set(
            jsonb_set(
                setup_manifest,
                '{scope_order}',
                '["app_mentions:read","channels:history","chat:write","im:write","files:read","groups:history","im:history","mpim:history"]'::jsonb,
                TRUE
            ),
            '{config_keys}',
            '["channel.slack.bot_token","channel.slack.app_token","channel.slack.signing_secret","channel.slack.operator_user_id","channel.slack.allowed_channels"]'::jsonb,
            TRUE
        ),
        '{env_vars}',
        '["SLACK_BOT_TOKEN","SLACK_APP_TOKEN","SLACK_SIGNING_SECRET"]'::jsonb,
        TRUE
    ),
    updated_at = CURRENT_TIMESTAMP
WHERE id = 'slack';
