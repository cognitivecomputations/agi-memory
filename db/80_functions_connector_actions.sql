-- Connector action authorization policy.
--
-- Python tools perform external side effects. Postgres owns which connector
-- actions are permitted, under which constraints, in which context, and keeps
-- the audit trail.
SET search_path = public, ag_catalog, "$user";

INSERT INTO connector_action_tool_map (
    tool_name,
    connector_id,
    action_kind,
    target_argument,
    account_argument,
    sensitivity,
    metadata
) VALUES
    ('email_send', 'email', 'send', 'to', NULL, 'external_message',
     '{"provider_kind": "smtp"}'::jsonb),
    ('email_send_sendgrid', 'email', 'send', 'to', NULL, 'external_message',
     '{"provider_kind": "sendgrid"}'::jsonb),
    ('email_read', 'gmail', 'mark_read', 'message_id', NULL, 'provider_state_change',
     '{"when": {"mark_read": true}}'::jsonb),
    ('discord_send', 'discord', 'send', 'channel_id', NULL, 'external_message',
     '{"fallback_target_argument": "webhook_url"}'::jsonb),
    ('slack_send', 'slack', 'send', 'channel', NULL, 'external_message',
     '{}'::jsonb),
    ('telegram_send', 'telegram', 'send', 'chat_id', NULL, 'external_message',
     '{}'::jsonb),
    ('signal_send', 'signal', 'send', 'recipient', NULL, 'external_message',
     '{"tool_module": "core.tools.messaging"}'::jsonb),
    ('gmail_send', 'gmail', 'send', 'to', 'account_key', 'external_message',
     '{"tool_module": "core.tools.gmail_actions"}'::jsonb),
    ('gmail_reply', 'gmail', 'reply', 'thread_id', 'account_key', 'external_message',
     '{"tool_module": "core.tools.gmail_actions"}'::jsonb),
    ('gmail_label', 'gmail', 'label', 'message_id', 'account_key', 'provider_state_change',
     '{"tool_module": "core.tools.gmail_actions"}'::jsonb),
    ('gmail_spam_triage', 'gmail', 'spam_triage', 'message_id', 'account_key', 'provider_state_change',
     '{"tool_module": "core.tools.gmail_actions"}'::jsonb),
    ('twitter_x_post', 'twitter_x', 'post', 'text', 'account_key', 'external_message',
     '{"tool_module": "core.tools.twitter_x_actions", "provider_endpoint": "POST /2/tweets", "cost_basis": "per_request"}'::jsonb),
    ('twitter_x_reply', 'twitter_x', 'reply', 'reply_to_tweet_id', 'account_key', 'external_message',
     '{"tool_module": "core.tools.twitter_x_actions", "provider_endpoint": "POST /2/tweets", "cost_basis": "per_request"}'::jsonb),
    ('twitter_x_dm_send', 'twitter_x', 'dm_send', 'participant_id', 'account_key', 'external_message',
     '{"tool_module": "core.tools.twitter_x_actions", "provider_endpoint": "POST /2/dm_conversations/with/:participant_id/messages", "cost_basis": "per_request"}'::jsonb),
    ('gmail_delete', 'gmail', 'delete', 'message_id', 'account_key', 'destructive',
     '{"tool_module": "core.tools.gmail_actions"}'::jsonb)
ON CONFLICT (tool_name) DO UPDATE SET
    connector_id = EXCLUDED.connector_id,
    action_kind = EXCLUDED.action_kind,
    target_argument = EXCLUDED.target_argument,
    account_argument = EXCLUDED.account_argument,
    sensitivity = EXCLUDED.sensitivity,
    enabled = TRUE,
    metadata = connector_action_tool_map.metadata || EXCLUDED.metadata,
    updated_at = CURRENT_TIMESTAMP;

CREATE OR REPLACE FUNCTION _connector_action_arg(
    p_arguments JSONB,
    p_path TEXT
) RETURNS TEXT
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    parts TEXT[];
    value TEXT;
BEGIN
    IF NULLIF(btrim(COALESCE(p_path, '')), '') IS NULL THEN
        RETURN NULL;
    END IF;
    parts := string_to_array(p_path, '.');
    value := p_arguments #>> parts;
    RETURN NULLIF(btrim(COALESCE(value, '')), '');
END;
$$;

CREATE OR REPLACE FUNCTION _connector_action_when_matches(
    p_when JSONB,
    p_arguments JSONB
) RETURNS BOOLEAN
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    pair RECORD;
BEGIN
    IF p_when IS NULL OR p_when = 'null'::jsonb OR p_when = '{}'::jsonb THEN
        RETURN TRUE;
    END IF;
    IF jsonb_typeof(p_when) <> 'object' THEN
        RETURN FALSE;
    END IF;

    FOR pair IN SELECT key, value FROM jsonb_each(p_when)
    LOOP
        IF NOT p_arguments ? pair.key THEN
            RETURN FALSE;
        END IF;
        IF p_arguments->pair.key <> pair.value THEN
            RETURN FALSE;
        END IF;
    END LOOP;

    RETURN TRUE;
END;
$$;

CREATE OR REPLACE FUNCTION connector_action_for_tool(
    p_tool_name TEXT,
    p_arguments JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    row_map connector_action_tool_map%ROWTYPE;
    target TEXT;
    fallback_target TEXT;
    account TEXT;
BEGIN
    SELECT *
    INTO row_map
    FROM connector_action_tool_map
    WHERE tool_name = p_tool_name
      AND enabled;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('action_required', FALSE);
    END IF;

    IF NOT _connector_action_when_matches(row_map.metadata->'when', COALESCE(p_arguments, '{}'::jsonb)) THEN
        RETURN jsonb_build_object('action_required', FALSE);
    END IF;

    target := _connector_action_arg(COALESCE(p_arguments, '{}'::jsonb), row_map.target_argument);
    IF target IS NULL AND NULLIF(row_map.metadata->>'fallback_target_argument', '') IS NOT NULL THEN
        fallback_target := _connector_action_arg(
            COALESCE(p_arguments, '{}'::jsonb),
            row_map.metadata->>'fallback_target_argument'
        );
        IF fallback_target IS NOT NULL THEN
            target := '[configured-or-redacted]';
        END IF;
    END IF;
    account := _connector_action_arg(COALESCE(p_arguments, '{}'::jsonb), row_map.account_argument);

    RETURN jsonb_build_object(
        'action_required', TRUE,
        'tool_name', row_map.tool_name,
        'connector_id', row_map.connector_id,
        'action_kind', row_map.action_kind,
        'target', target,
        'account_key', account,
        'sensitivity', row_map.sensitivity,
        'metadata', row_map.metadata
    );
END;
$$;

CREATE OR REPLACE FUNCTION _connector_action_jsonb_text_has(
    p_values JSONB,
    p_value TEXT
) RETURNS BOOLEAN
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    normalized TEXT := lower(btrim(COALESCE(p_value, '')));
BEGIN
    IF normalized = '' OR p_values IS NULL OR jsonb_typeof(p_values) <> 'array' THEN
        RETURN FALSE;
    END IF;

    RETURN EXISTS (
        SELECT 1
        FROM jsonb_array_elements_text(p_values) item(value)
        WHERE lower(btrim(value)) = normalized
    );
END;
$$;

CREATE OR REPLACE FUNCTION connector_action_constraints_match(
    p_constraints JSONB,
    p_target TEXT,
    p_arguments JSONB,
    p_policy_id UUID DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    constraints JSONB := COALESCE(p_constraints, '{}'::jsonb);
    target TEXT := NULLIF(btrim(COALESCE(p_target, '')), '');
    max_per_day INT;
    used_today INT;
BEGIN
    IF constraints ? 'denied_targets'
       AND _connector_action_jsonb_text_has(constraints->'denied_targets', target) THEN
        RETURN jsonb_build_object('allowed', FALSE, 'reason', 'target is denied by policy constraints');
    END IF;

    IF constraints ? 'denied_recipients'
       AND _connector_action_jsonb_text_has(constraints->'denied_recipients', target) THEN
        RETURN jsonb_build_object('allowed', FALSE, 'reason', 'recipient is denied by policy constraints');
    END IF;

    IF constraints ? 'allowed_targets'
       AND NOT _connector_action_jsonb_text_has(constraints->'allowed_targets', target) THEN
        RETURN jsonb_build_object('allowed', FALSE, 'reason', 'target is not in the policy allowlist');
    END IF;

    IF constraints ? 'allowed_recipients'
       AND NOT _connector_action_jsonb_text_has(constraints->'allowed_recipients', target) THEN
        RETURN jsonb_build_object('allowed', FALSE, 'reason', 'recipient is not in the policy allowlist');
    END IF;

    IF constraints ? 'max_per_day' AND p_policy_id IS NOT NULL THEN
        BEGIN
            max_per_day := NULLIF(constraints->>'max_per_day', '')::int;
        EXCEPTION WHEN OTHERS THEN
            max_per_day := NULL;
        END;
        IF max_per_day IS NOT NULL THEN
            SELECT COUNT(*)::int
            INTO used_today
            FROM connector_action_audit
            WHERE policy_id = p_policy_id
              AND decision = 'allowed'
              AND created_at >= CURRENT_TIMESTAMP - INTERVAL '1 day';
            IF used_today >= max_per_day THEN
                RETURN jsonb_build_object('allowed', FALSE, 'reason', 'policy daily action limit reached');
            END IF;
        END IF;
    END IF;

    RETURN jsonb_build_object('allowed', TRUE);
END;
$$;

CREATE OR REPLACE FUNCTION grant_connector_action_policy(
    p_connector_id TEXT,
    p_action_kind TEXT,
    p_account_key TEXT DEFAULT NULL,
    p_constraints JSONB DEFAULT '{}'::jsonb,
    p_allow_autonomous BOOLEAN DEFAULT FALSE,
    p_requires_per_action_approval BOOLEAN DEFAULT TRUE,
    p_contexts TEXT[] DEFAULT NULL,
    p_expires_at TIMESTAMPTZ DEFAULT NULL,
    p_source_session_id TEXT DEFAULT NULL,
    p_rationale TEXT DEFAULT NULL,
    p_granted_by TEXT DEFAULT 'user'
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    new_id UUID;
    contexts TEXT[];
BEGIN
    IF NULLIF(btrim(COALESCE(p_connector_id, '')), '') IS NULL THEN
        RAISE EXCEPTION 'connector_id is required';
    END IF;
    IF NULLIF(btrim(COALESCE(p_action_kind, '')), '') IS NULL THEN
        RAISE EXCEPTION 'action_kind is required';
    END IF;

    contexts := COALESCE(
        p_contexts,
        CASE WHEN COALESCE(p_allow_autonomous, FALSE)
             THEN ARRAY['chat', 'heartbeat']::TEXT[]
             ELSE ARRAY['chat']::TEXT[]
        END
    );

    INSERT INTO connector_action_policies (
        connector_id,
        account_key,
        action_kind,
        contexts,
        allow_autonomous,
        requires_per_action_approval,
        constraints,
        granted_by,
        source_session_id,
        rationale,
        expires_at
    )
    VALUES (
        lower(btrim(p_connector_id)),
        NULLIF(lower(btrim(COALESCE(p_account_key, ''))), ''),
        lower(btrim(p_action_kind)),
        contexts,
        COALESCE(p_allow_autonomous, FALSE),
        COALESCE(p_requires_per_action_approval, TRUE),
        COALESCE(p_constraints, '{}'::jsonb),
        COALESCE(NULLIF(btrim(p_granted_by), ''), 'user'),
        NULLIF(btrim(COALESCE(p_source_session_id, '')), ''),
        NULLIF(btrim(COALESCE(p_rationale, '')), ''),
        p_expires_at
    )
    RETURNING id INTO new_id;

    RETURN jsonb_build_object(
        'policy_id', new_id::text,
        'status', 'active',
        'connector_id', lower(btrim(p_connector_id)),
        'account_key', NULLIF(lower(btrim(COALESCE(p_account_key, ''))), ''),
        'action_kind', lower(btrim(p_action_kind)),
        'contexts', to_jsonb(contexts),
        'allow_autonomous', COALESCE(p_allow_autonomous, FALSE),
        'requires_per_action_approval', COALESCE(p_requires_per_action_approval, TRUE),
        'constraints', COALESCE(p_constraints, '{}'::jsonb),
        'expires_at', p_expires_at
    );
END;
$$;

CREATE OR REPLACE FUNCTION revoke_connector_action_policy(
    p_policy_id UUID,
    p_reason TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    row_policy connector_action_policies%ROWTYPE;
BEGIN
    UPDATE connector_action_policies
    SET status = 'revoked',
        revoked_at = CURRENT_TIMESTAMP,
        revoke_reason = NULLIF(btrim(COALESCE(p_reason, '')), ''),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_policy_id
      AND status = 'active'
    RETURNING * INTO row_policy;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('policy_id', p_policy_id::text, 'status', 'missing_or_not_active');
    END IF;

    RETURN jsonb_build_object(
        'policy_id', row_policy.id::text,
        'status', row_policy.status,
        'revoked_at', row_policy.revoked_at
    );
END;
$$;

CREATE OR REPLACE FUNCTION list_connector_action_policies(
    p_connector_id TEXT DEFAULT NULL,
    p_account_key TEXT DEFAULT NULL,
    p_include_inactive BOOLEAN DEFAULT FALSE
) RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    SELECT COALESCE(jsonb_agg(
        jsonb_build_object(
            'policy_id', id::text,
            'connector_id', connector_id,
            'account_key', account_key,
            'action_kind', action_kind,
            'status', status,
            'contexts', contexts,
            'allow_autonomous', allow_autonomous,
            'requires_per_action_approval', requires_per_action_approval,
            'constraints', constraints,
            'source_session_id', source_session_id,
            'rationale', rationale,
            'expires_at', expires_at,
            'revoked_at', revoked_at,
            'created_at', created_at,
            'updated_at', updated_at
        )
        ORDER BY updated_at DESC, created_at DESC
    ), '[]'::jsonb)
    FROM connector_action_policies
    WHERE (p_include_inactive OR status = 'active')
      AND (p_connector_id IS NULL OR connector_id = lower(btrim(p_connector_id)))
      AND (p_account_key IS NULL OR account_key = lower(btrim(p_account_key)));
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

CREATE OR REPLACE FUNCTION record_connector_action_audit_from_tool_execution(
    p_tool_execution_id UUID,
    p_record JSONB
) RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    action JSONB;
    decision JSONB;
    audit_id UUID;
    tool_context TEXT := lower(COALESCE(p_record->>'tool_context', 'unknown'));
    success BOOLEAN := COALESCE((p_record->>'success')::boolean, FALSE);
    decision_label TEXT;
BEGIN
    action := connector_action_for_tool(
        p_record->>'tool_name',
        COALESCE(p_record->'arguments', '{}'::jsonb)
    );
    IF NOT COALESCE((action->>'action_required')::boolean, FALSE) THEN
        RETURN NULL;
    END IF;

    decision := evaluate_connector_action_call(
        p_record->>'tool_name',
        COALESCE(p_record->'arguments', '{}'::jsonb),
        jsonb_build_object(
            'tool_context', tool_context,
            'call_id', p_record->>'call_id',
            'session_id', p_record->>'session_id'
        )
    );
    IF NOT COALESCE((decision->>'allowed')::boolean, FALSE) THEN
        decision_label := 'denied';
    ELSIF success THEN
        decision_label := 'allowed';
    ELSE
        decision_label := 'failed';
    END IF;

    INSERT INTO connector_action_audit (
        policy_id,
        tool_execution_id,
        connector_id,
        account_key,
        action_kind,
        target,
        tool_name,
        tool_context,
        decision,
        reason,
        arguments,
        context,
        external_receipt
    )
    VALUES (
        NULLIF(decision->>'policy_id', '')::uuid,
        p_tool_execution_id,
        action->>'connector_id',
        NULLIF(action->>'account_key', ''),
        action->>'action_kind',
        NULLIF(action->>'target', ''),
        p_record->>'tool_name',
        tool_context,
        decision_label,
        COALESCE(decision->>'reason', p_record->>'error'),
        COALESCE(p_record->'arguments', '{}'::jsonb),
        jsonb_build_object(
            'authorization_kind', decision->>'authorization_kind',
            'tool_context', tool_context,
            'call_id', p_record->>'call_id',
            'session_id', p_record->>'session_id'
        ),
        jsonb_build_object(
            'output', p_record->'output',
            'error', p_record->>'error',
            'error_type', p_record->>'error_type'
        )
    )
    RETURNING id INTO audit_id;

    RETURN audit_id;
END;
$$;

-- DB tool policy with connector action policy folded in.
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

-- Tool audit with connector action audit folded in.
CREATE OR REPLACE FUNCTION record_tool_execution(
    p_record JSONB
) RETURNS UUID AS $$
DECLARE
    rec_id UUID;
BEGIN
    INSERT INTO tool_executions (
        tool_name, arguments, tool_context, call_id, session_id,
        success, output, error, error_type, energy_spent, duration_seconds
    )
    VALUES (
        p_record->>'tool_name',
        COALESCE(p_record->'arguments', 'null'::jsonb),
        p_record->>'tool_context',
        p_record->>'call_id',
        p_record->>'session_id',
        COALESCE((p_record->>'success')::boolean, false),
        p_record->'output',
        p_record->>'error',
        p_record->>'error_type',
        COALESCE((p_record->>'energy_spent')::float, 0.0)::int,
        (p_record->>'duration_seconds')::float
    )
    RETURNING id INTO rec_id;

    BEGIN
        PERFORM record_connector_action_audit_from_tool_execution(rec_id, p_record);
    EXCEPTION WHEN OTHERS THEN
        RAISE WARNING 'connector action audit failed for tool execution %: %', rec_id, SQLERRM;
    END;

    RETURN rec_id;
END;
$$ LANGUAGE plpgsql;
