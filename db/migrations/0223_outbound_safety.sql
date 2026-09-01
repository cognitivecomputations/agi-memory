-- Purpose-bound outbound communication, attention budgets, disclosure,
-- cross-channel STOP, kill switches, and the durable communication ledger.
-- This migration is completed below as the Phase 6 contract is finalized.
SET search_path = public, ag_catalog, "$user";

CREATE TABLE IF NOT EXISTS outbound_contact_endpoints (
    channel TEXT NOT NULL,
    address TEXT NOT NULL,
    entity TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    contact_id BIGINT,
    is_primary BOOLEAN NOT NULL DEFAULT FALSE,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (channel, address)
);
CREATE INDEX IF NOT EXISTS idx_outbound_contact_endpoints_entity
    ON outbound_contact_endpoints (entity);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'outbound_contact_endpoints_contact_id_fkey'
          AND conrelid = 'outbound_contact_endpoints'::regclass
    ) THEN
        ALTER TABLE outbound_contact_endpoints
            ADD CONSTRAINT outbound_contact_endpoints_contact_id_fkey
            FOREIGN KEY (contact_id) REFERENCES contacts(id) ON DELETE SET NULL;
    END IF;
END;
$$;

CREATE TABLE IF NOT EXISTS contact_budgets (
    entity TEXT NOT NULL,
    channel TEXT NOT NULL,
    points DOUBLE PRECISION NOT NULL DEFAULT 1,
    regen_per_day DOUBLE PRECISION NOT NULL,
    max_points DOUBLE PRECISION NOT NULL,
    observed_per_week DOUBLE PRECISION,
    reciprocity DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    strain DOUBLE PRECISION NOT NULL DEFAULT 0,
    consecutive_silent INTEGER NOT NULL DEFAULT 0,
    last_outbound_cost DOUBLE PRECISION NOT NULL DEFAULT 0,
    last_outbound_at TIMESTAMPTZ,
    last_inbound_at TIMESTAMPTZ,
    regenerated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entity, channel),
    CHECK (max_points > 0),
    CHECK (regen_per_day >= 0),
    CHECK (reciprocity >= 0),
    CHECK (strain >= 0),
    CHECK (consecutive_silent >= 0)
);

CREATE TABLE IF NOT EXISTS outbound_contact_controls (
    entity TEXT PRIMARY KEY,
    blocked BOOLEAN NOT NULL DEFAULT FALSE,
    suspended BOOLEAN NOT NULL DEFAULT FALSE,
    blocked_at TIMESTAMPTZ,
    unblocked_at TIMESTAMPTZ,
    suspended_at TIMESTAMPTZ,
    source_channel TEXT,
    source_address TEXT,
    source_message TEXT,
    reason TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS outbound_contact_control_events (
    id BIGSERIAL PRIMARY KEY,
    entity TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('stop', 'start', 'suspend', 'resume')),
    channel TEXT,
    address TEXT,
    message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_outbound_control_events_entity
    ON outbound_contact_control_events (entity, created_at DESC);

CREATE TABLE IF NOT EXISTS outbound_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_key TEXT NOT NULL,
    source TEXT NOT NULL,
    tool_name TEXT,
    call_id TEXT,
    heartbeat_id UUID,
    session_id TEXT,
    entity TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    channel TEXT NOT NULL,
    recipient TEXT NOT NULL,
    is_primary BOOLEAN NOT NULL DEFAULT FALSE,
    purpose_kind TEXT,
    purpose_reference TEXT,
    purpose_verified BOOLEAN NOT NULL DEFAULT FALSE,
    assigned_goal BOOLEAN NOT NULL DEFAULT FALSE,
    is_reply BOOLEAN NOT NULL DEFAULT FALSE,
    urgency TEXT NOT NULL DEFAULT 'normal',
    base_cost DOUBLE PRECISION NOT NULL DEFAULT 0,
    charged_cost DOUBLE PRECISION NOT NULL DEFAULT 0,
    strain_delta DOUBLE PRECISION NOT NULL DEFAULT 0,
    points_before DOUBLE PRECISION,
    points_after DOUBLE PRECISION,
    thread_reference TEXT,
    disclosure_mode TEXT NOT NULL DEFAULT 'none'
        CHECK (disclosure_mode IN ('none', 'full', 'marker')),
    status TEXT NOT NULL
        CHECK (status IN ('denied', 'authorized', 'delivered', 'failed')),
    reason TEXT,
    body_preview TEXT,
    provider_message_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finalized_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_outbound_events_created
    ON outbound_events (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_outbound_events_entity
    ON outbound_events (entity, channel, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_outbound_events_request
    ON outbound_events (request_key, created_at DESC);

-- 0206 is the latest migration to replace this function. Preserve its
-- connector-action and operator-approval gates while adding the assigned-goal
-- energy permission slip.
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
    purpose_goal_id UUID;
    assigned_energy_multiplier DOUBLE PRECISION;
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
    IF tool.metadata ? 'outbound'
       AND lower(COALESCE(p_arguments->>'purpose_kind', '')) = 'goal' THEN
        BEGIN
            purpose_goal_id := NULLIF(p_arguments->>'purpose_reference', '')::uuid;
        EXCEPTION WHEN OTHERS THEN
            purpose_goal_id := NULL;
        END;
        IF purpose_goal_id IS NOT NULL AND EXISTS (
            SELECT 1 FROM memories
            WHERE id = purpose_goal_id
              AND type = 'goal'
              AND goal_origin = 'user_request'::goal_source
        ) THEN
            assigned_energy_multiplier := LEAST(
                GREATEST(
                    COALESCE(get_config_float('outbound.assigned_goal_energy_multiplier'), 0.25),
                    0
                ),
                1
            );
            cost := CEIL(cost * assigned_energy_multiplier)::integer;
        END IF;
    END IF;

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
-- Purpose-bound outbound communication, per-contact cadence, disclosure, and
-- cross-channel STOP.  Postgres owns every decision and ledger mutation;
-- Python only maps provider argument shapes into this contract.
SET search_path = public, ag_catalog, "$user";

INSERT INTO config_defaults (key, value, description) VALUES
    ('outbound.suspended', 'false'::jsonb, 'Global one-click suspension for all outbound communication'),
    ('outbound.contact_budgets.enabled', 'true'::jsonb, 'Enforce per-entity, per-channel attention budgets'),
    ('outbound.disclosure.enabled', 'true'::jsonb, 'Identify Hexis on every third-party communication'),
    ('outbound.disclosure.full_interval_days', '30'::jsonb, 'Days before the full disclosure and STOP instruction is repeated'),
    ('outbound.max_consecutive_silent', '4'::jsonb, 'Unanswered unsolicited messages allowed before non-urgent outreach stops'),
    ('outbound.channel_base_costs', '{"outbox":0,"web_inbox":0,"slack":1,"discord":1,"telegram":1,"matrix":1,"twitter_x":2,"twitter_x_dm":2,"email":3,"signal":5,"imessage":5,"whatsapp":5,"sms":5,"phone":5,"webhook":3}'::jsonb, 'Attention cost by communication medium'),
    ('outbound.channel_default_regen_per_day', '{"slack":1,"discord":1,"telegram":1,"matrix":1,"twitter_x":0.25,"twitter_x_dm":0.25,"email":0.142857,"signal":0.033333,"imessage":0.033333,"whatsapp":0.033333,"sms":0.033333,"phone":0.033333,"webhook":0.142857}'::jsonb, 'Conservative fallback cadence when no observed history exists'),
    ('outbound.relationship_strength_thresholds', '{"very_strong":0.9,"strong":0.7,"regular":0.4,"weak":0.15}'::jsonb, 'Relationship-strength thresholds used only when observed contact history is absent'),
    ('outbound.relationship_contacts_per_day', '{"very_strong":3,"strong":1,"regular":0.2,"weak":0.142857,"dormant":0.010959}'::jsonb, 'Human contact cadence inferred from relationship strength before channel cost is applied'),
    ('outbound.default_max_points_multiplier', '2'::jsonb, 'Maximum banked attention as a multiple of channel base cost'),
    ('outbound.urgency_divisors', '{"low":0.8,"normal":1,"high":2,"urgent":10}'::jsonb, 'Urgency divisor applied to contact cost'),
    ('outbound.quiet_hours', '{"start":22,"end":7,"multiplier":2}'::jsonb, 'Local hours when interruptive contact costs more'),
    ('outbound.assigned_goal_contact_discount', '0.5'::jsonb, 'Modest contact discount for user-assigned goals; never a waiver'),
    ('outbound.assigned_goal_energy_multiplier', '0.25'::jsonb, 'Energy multiplier for tool work backed by a user-assigned goal'),
    ('outbound.reply_bonus_multiplier', '0.5'::jsonb, 'Additional contact credit when a recipient replies'),
    ('outbound.initiation_credit_multiplier', '2'::jsonb, 'Contact credit when the other person initiates'),
    ('outbound.stop_comparable_tolerance', '0.15'::jsonb, 'Relationship-strength distance considered comparable after a recipient STOP'),
    ('outbound.stop_comparable_cadence_multiplier', '0.9'::jsonb, 'Conservative cadence reduction for comparable relationships after a recipient STOP')
ON CONFLICT (key) DO NOTHING;

CREATE OR REPLACE FUNCTION _outbound_normalize_address(
    p_channel TEXT,
    p_address TEXT
) RETURNS TEXT
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    value TEXT := lower(btrim(COALESCE(p_address, '')));
    bracketed TEXT;
BEGIN
    IF lower(COALESCE(p_channel, '')) = 'email' THEN
        bracketed := substring(value FROM '<([^>]+)>');
        IF bracketed IS NOT NULL THEN
            value := bracketed;
        END IF;
    END IF;
    RETURN value;
END;
$$;

CREATE OR REPLACE FUNCTION _outbound_channel_float(
    p_config_key TEXT,
    p_channel TEXT,
    p_fallback DOUBLE PRECISION
) RETURNS DOUBLE PRECISION
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    cfg JSONB := COALESCE(get_config(p_config_key), '{}'::jsonb);
    result DOUBLE PRECISION;
BEGIN
    BEGIN
        result := NULLIF(cfg->>lower(COALESCE(p_channel, '')), '')::double precision;
    EXCEPTION WHEN OTHERS THEN
        result := NULL;
    END;
    RETURN COALESCE(result, p_fallback);
END;
$$;

CREATE OR REPLACE FUNCTION resolve_outbound_entity(
    p_channel TEXT,
    p_address TEXT,
    p_identity_address TEXT DEFAULT NULL,
    p_primary_hint BOOLEAN DEFAULT FALSE,
    p_public_hint BOOLEAN DEFAULT FALSE
) RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
AS $$
DECLARE
    channel_name TEXT := lower(COALESCE(NULLIF(btrim(p_channel), ''), 'unknown'));
    delivery_address TEXT := _outbound_normalize_address(p_channel, p_address);
    normalized_address TEXT := delivery_address;
    identity_address TEXT := _outbound_normalize_address(p_channel, p_identity_address);
    profile JSONB := COALESCE(get_init_profile(), '{}'::jsonb);
    primary_name TEXT := COALESCE(
        NULLIF(profile#>>'{user,name}', ''),
        NULLIF(get_config_text('agent.user_name'), ''),
        'the user'
    );
    contact_row contacts%ROWTYPE;
    endpoint outbound_contact_endpoints%ROWTYPE;
    entity_key TEXT;
    entity_label TEXT;
    is_primary BOOLEAN := COALESCE(p_primary_hint, FALSE);
    normalized_phone TEXT;
BEGIN
    IF delivery_address = '' THEN
        delivery_address := COALESCE(NULLIF(identity_address, ''), 'unknown');
        normalized_address := delivery_address;
    END IF;
    -- A room/channel id identifies the delivery route, not necessarily the
    -- person. Last-active, broadcast, and conversational reply paths supply
    -- the sender identity separately so cross-channel STOP converges on the
    -- same contact rather than a room-local alias.
    IF NOT COALESCE(p_public_hint, FALSE) AND identity_address <> '' THEN
        normalized_address := identity_address;
    END IF;

    is_primary := COALESCE(is_primary
        OR (
            NOT COALESCE(p_public_hint, FALSE)
            AND channel_sender_is_operator(channel_name, COALESCE(NULLIF(identity_address, ''), normalized_address))
        )
        OR (
            channel_name = 'email'
            AND NULLIF(lower(profile#>>'{user,email}'), '') = normalized_address
        )
        OR (
            channel_name IN ('signal', 'imessage', 'whatsapp', 'sms', 'phone')
            AND regexp_replace(COALESCE(profile#>>'{user,phone}', ''), '[^0-9+]', '', 'g') <> ''
            AND regexp_replace(normalized_address, '[^0-9+]', '', 'g') =
                regexp_replace(profile#>>'{user,phone}', '[^0-9+]', '', 'g')
        ), FALSE);

    IF is_primary THEN
        entity_key := 'primary:user';
        entity_label := primary_name;
    ELSIF COALESCE(p_public_hint, FALSE) THEN
        entity_key := 'public:' || channel_name || ':' || normalized_address;
        entity_label := CASE WHEN normalized_address = 'public' THEN 'public audience' ELSE normalized_address END;
    ELSE
        normalized_phone := regexp_replace(normalized_address, '[^0-9+]', '', 'g');
        SELECT * INTO contact_row
        FROM contacts c
        WHERE (channel_name = 'email' AND lower(COALESCE(c.email, '')) = normalized_address)
           OR (
                channel_name IN ('signal', 'imessage', 'whatsapp', 'sms', 'phone')
                AND normalized_phone <> ''
                AND regexp_replace(COALESCE(c.phone, ''), '[^0-9+]', '', 'g') = normalized_phone
           )
           OR EXISTS (
                SELECT 1
                FROM jsonb_each_text(
                    CASE WHEN jsonb_typeof(c.metadata->'channels') = 'object'
                         THEN c.metadata->'channels' ELSE '{}'::jsonb END
                ) item
                WHERE lower(item.key) = channel_name
                  AND _outbound_normalize_address(channel_name, item.value) = normalized_address
           )
        ORDER BY c.last_touch DESC, c.id
        LIMIT 1;

        IF FOUND THEN
            entity_key := 'contact:' || contact_row.id::text;
            entity_label := contact_row.name;
        ELSE
            SELECT * INTO endpoint
            FROM outbound_contact_endpoints endpoint_row
            WHERE endpoint_row.channel = channel_name
              AND endpoint_row.address = normalized_address;
            IF FOUND THEN
                entity_key := endpoint.entity;
                entity_label := endpoint.entity_name;
            ELSE
                entity_key := channel_name || ':' || normalized_address;
                entity_label := normalized_address;
            END IF;
        END IF;
    END IF;

    INSERT INTO outbound_contact_endpoints (
        channel, address, entity, entity_name, contact_id, is_primary, last_seen_at
    ) VALUES (
        channel_name,
        normalized_address,
        entity_key,
        entity_label,
        contact_row.id,
        is_primary,
        CURRENT_TIMESTAMP
    )
    ON CONFLICT (channel, address) DO UPDATE
    SET entity = EXCLUDED.entity,
        entity_name = EXCLUDED.entity_name,
        contact_id = COALESCE(EXCLUDED.contact_id, outbound_contact_endpoints.contact_id),
        is_primary = EXCLUDED.is_primary,
        last_seen_at = CURRENT_TIMESTAMP;

    RETURN jsonb_build_object(
        'entity', entity_key,
        'entity_name', entity_label,
        'channel', channel_name,
        'address', delivery_address,
        'is_primary', is_primary,
        'contact_id', contact_row.id
    );
END;
$$;

-- A cheap first gate for provider tools.  The agent loop calls this before it
-- files an approval request, so a recipient's STOP cannot itself cause another
-- notification or ask.  authorize_outbound repeats these checks immediately
-- before delivery; this preflight is ordering, not a race-prone substitute.
CREATE OR REPLACE FUNCTION check_outbound_controls(
    p_channel TEXT,
    p_recipient TEXT,
    p_identity_address TEXT DEFAULT NULL,
    p_primary_hint BOOLEAN DEFAULT FALSE,
    p_public_hint BOOLEAN DEFAULT FALSE
) RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
AS $$
DECLARE
    resolved JSONB := resolve_outbound_entity(
        p_channel, p_recipient, p_identity_address, p_primary_hint, p_public_hint
    );
    entity_key TEXT := resolved->>'entity';
    entity_name TEXT := resolved->>'entity_name';
    is_blocked BOOLEAN := FALSE;
    is_suspended BOOLEAN := FALSE;
BEGIN
    SELECT blocked, suspended
    INTO is_blocked, is_suspended
    FROM outbound_contact_controls
    WHERE entity = entity_key;

    IF COALESCE(is_blocked, FALSE) THEN
        RETURN resolved || jsonb_build_object(
            'allowed', false,
            'reason', format('%s has opted out of all Hexis contact.', entity_name),
            'error_type', 'outbound_blocked'
        );
    END IF;
    IF COALESCE(get_config_bool('outbound.suspended'), FALSE) THEN
        RETURN resolved || jsonb_build_object(
            'allowed', false,
            'reason', 'Outbound communication is globally suspended. Resume it from the outbound ledger controls.',
            'error_type', 'disabled'
        );
    END IF;
    IF COALESCE(is_suspended, FALSE) THEN
        RETURN resolved || jsonb_build_object(
            'allowed', false,
            'reason', format('Outbound communication to %s is suspended.', entity_name),
            'error_type', 'disabled'
        );
    END IF;

    RETURN resolved || jsonb_build_object('allowed', true);
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'check_outbound_controls failed: % (SQLSTATE %)', SQLERRM, SQLSTATE;
    RETURN jsonb_build_object(
        'allowed', false,
        'reason', 'Outbound control policy could not be evaluated: ' || SQLERRM,
        'error_type', 'outbound_blocked'
    );
END;
$$;

CREATE OR REPLACE FUNCTION _outbound_observed_per_week(
    p_entity TEXT,
    p_channel TEXT
) RETURNS DOUBLE PRECISION
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    addresses TEXT[];
    event_count DOUBLE PRECISION := 0;
    oldest TIMESTAMPTZ;
    weeks DOUBLE PRECISION := 1;
BEGIN
    SELECT COALESCE(array_agg(address), ARRAY[]::text[])
    INTO addresses
    FROM outbound_contact_endpoints
    WHERE entity = p_entity;

    IF lower(p_channel) IN ('email', 'twitter_x', 'twitter_x_dm') THEN
        SELECT COUNT(*)::double precision, MIN(COALESCE(item_timestamp, created_at))
        INTO event_count, oldest
        FROM connector_source_items item
        WHERE COALESCE(item.item_timestamp, item.created_at) >= CURRENT_TIMESTAMP - INTERVAL '90 days'
          AND (
              item.labels && ARRAY['SENT', 'sent']::text[]
              OR lower(COALESCE(item.raw_metadata->>'direction', '')) IN ('outbound', 'sent')
          )
          AND EXISTS (
              SELECT 1
              FROM jsonb_array_elements(COALESCE(item.participants, '[]'::jsonb)) participant
              WHERE lower(COALESCE(participant->>'role', '')) IN ('to', 'recipient')
                AND _outbound_normalize_address(
                        CASE WHEN lower(p_channel) = 'email' THEN 'email' ELSE lower(p_channel) END,
                        COALESCE(participant->>'value', participant->>'id', '')
                    ) = ANY(addresses)
          );
    ELSE
        SELECT COUNT(*)::double precision, MIN(item.created_at)
        INTO event_count, oldest
        FROM channel_source_items item
        WHERE item.channel_type = lower(p_channel)
          AND item.direction = 'outbound'
          AND item.created_at >= CURRENT_TIMESTAMP - INTERVAL '90 days'
          AND _outbound_normalize_address(p_channel, item.sender_id) = ANY(addresses);
    END IF;

    IF oldest IS NOT NULL THEN
        weeks := GREATEST(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - oldest)) / 604800.0, 1.0);
    END IF;
    RETURN event_count / weeks;
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING '_outbound_observed_per_week failed: % (SQLSTATE %)', SQLERRM, SQLSTATE;
    RETURN 0;
END;
$$;

CREATE OR REPLACE FUNCTION _outbound_relationship_strength(
    p_entity TEXT
) RETURNS DOUBLE PRECISION
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    entity_label TEXT;
    result DOUBLE PRECISION;
BEGIN
    SELECT entity_name
    INTO entity_label
    FROM outbound_contact_endpoints
    WHERE entity = p_entity
    ORDER BY last_seen_at DESC
    LIMIT 1;
    IF NULLIF(btrim(COALESCE(entity_label, '')), '') IS NULL THEN
        RETURN NULL;
    END IF;

    SELECT MAX((relationship->>'strength')::double precision)
    INTO result
    FROM jsonb_array_elements(COALESCE(get_relationships_context(100), '[]'::jsonb)) relationship
    WHERE lower(btrim(COALESCE(relationship->>'entity', ''))) = lower(btrim(entity_label));
    RETURN result;
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING '_outbound_relationship_strength failed: % (SQLSTATE %)', SQLERRM, SQLSTATE;
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION _outbound_ensure_contact_budget(
    p_entity TEXT,
    p_channel TEXT
) RETURNS VOID
LANGUAGE plpgsql
VOLATILE
AS $$
DECLARE
    base_cost DOUBLE PRECISION := GREATEST(
        _outbound_channel_float('outbound.channel_base_costs', p_channel, 1), 0.01
    );
    observed DOUBLE PRECISION := _outbound_observed_per_week(p_entity, p_channel);
    relationship_strength DOUBLE PRECISION := _outbound_relationship_strength(p_entity);
    relationship_tier TEXT;
    relationship_contacts_per_day DOUBLE PRECISION;
    regen DOUBLE PRECISION;
    max_points DOUBLE PRECISION;
BEGIN
    relationship_tier := CASE
        WHEN relationship_strength >= _outbound_channel_float(
            'outbound.relationship_strength_thresholds', 'very_strong', 0.9
        ) THEN 'very_strong'
        WHEN relationship_strength >= _outbound_channel_float(
            'outbound.relationship_strength_thresholds', 'strong', 0.7
        ) THEN 'strong'
        WHEN relationship_strength >= _outbound_channel_float(
            'outbound.relationship_strength_thresholds', 'regular', 0.4
        ) THEN 'regular'
        WHEN relationship_strength >= _outbound_channel_float(
            'outbound.relationship_strength_thresholds', 'weak', 0.15
        ) THEN 'weak'
        WHEN relationship_strength IS NOT NULL THEN 'dormant'
        ELSE NULL
    END;
    relationship_contacts_per_day := CASE
        WHEN relationship_tier IS NULL THEN NULL
        ELSE GREATEST(
            _outbound_channel_float(
                'outbound.relationship_contacts_per_day', relationship_tier, 0
            ),
            0
        )
    END;
    regen := CASE
        WHEN observed > 0 THEN observed * base_cost / 7.0
        WHEN relationship_contacts_per_day IS NOT NULL
            THEN relationship_contacts_per_day * base_cost
        ELSE GREATEST(
            _outbound_channel_float(
                'outbound.channel_default_regen_per_day', p_channel, 1.0 / 7.0
            ),
            0
        )
    END;
    max_points := GREATEST(
        base_cost,
        base_cost * COALESCE(get_config_float('outbound.default_max_points_multiplier'), 2),
        regen * 2
    );

    INSERT INTO contact_budgets (
        entity, channel, points, regen_per_day, max_points, observed_per_week
    ) VALUES (
        p_entity, lower(p_channel), max_points, regen, max_points, observed
    )
    ON CONFLICT (entity, channel) DO UPDATE
    SET observed_per_week = CASE
            WHEN EXCLUDED.observed_per_week > 0 THEN EXCLUDED.observed_per_week
            ELSE contact_budgets.observed_per_week
        END,
        regen_per_day = CASE
            WHEN EXCLUDED.observed_per_week > 0 THEN EXCLUDED.regen_per_day
            ELSE contact_budgets.regen_per_day
        END,
        max_points = GREATEST(contact_budgets.max_points, EXCLUDED.max_points),
        updated_at = CURRENT_TIMESTAMP;
END;
$$;

CREATE OR REPLACE FUNCTION verify_outbound_purpose(
    p_kind TEXT,
    p_reference TEXT,
    p_is_primary BOOLEAN,
    p_thread_reference TEXT DEFAULT NULL,
    p_context JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    kind TEXT := lower(COALESCE(NULLIF(btrim(p_kind), ''), ''));
    reference TEXT := NULLIF(btrim(COALESCE(p_reference, '')), '');
    reference_uuid UUID := _db_brain_try_uuid(p_reference);
    verified BOOLEAN := FALSE;
    assigned BOOLEAN := FALSE;
    urgent_backed BOOLEAN := FALSE;
    row_goal memories%ROWTYPE;
    responsibility_exists BOOLEAN := FALSE;
    responsibility_urgent BOOLEAN := FALSE;
BEGIN
    IF kind = 'goal' AND reference_uuid IS NOT NULL THEN
        SELECT * INTO row_goal
        FROM memories
        WHERE id = reference_uuid AND type = 'goal'
        LIMIT 1;
        verified := FOUND;
        assigned := verified AND row_goal.goal_origin = 'user_request'::goal_source;
        urgent_backed := verified AND (
            lower(COALESCE(row_goal.metadata->>'priority', '')) IN ('urgent', 'critical')
            OR COALESCE((row_goal.metadata->>'urgent')::boolean, FALSE)
        );
    ELSIF kind = 'responsibility' AND reference_uuid IS NOT NULL
          AND to_regclass('public.ambient_responsibilities') IS NOT NULL THEN
        EXECUTE
            'SELECT EXISTS (SELECT 1 FROM ambient_responsibilities WHERE id = $1 AND status IN (''active'', ''blocked'')), '
            || 'EXISTS (SELECT 1 FROM ambient_responsibilities WHERE id = $1 AND priority = ''urgent'')'
        INTO responsibility_exists, responsibility_urgent
        USING reference_uuid;
        verified := responsibility_exists;
        urgent_backed := responsibility_urgent;
    ELSIF kind = 'reply' AND reference IS NOT NULL THEN
        verified := (
            NULLIF(btrim(COALESCE(p_thread_reference, '')), '') = reference
            OR EXISTS (
                SELECT 1 FROM channel_messages
                WHERE platform_message_id = reference
            )
            OR EXISTS (
                SELECT 1 FROM connector_source_items
                WHERE provider_item_id = reference OR provider_thread_id = reference
            )
        );
        urgent_backed := verified;
    ELSIF kind = 'user_request' AND reference IS NOT NULL THEN
        verified := (
            lower(COALESCE(p_context->>'tool_context', '')) = 'chat'
            OR NULLIF(p_context->>'approval_request_id', '') IS NOT NULL
        );
        urgent_backed := verified;
    ELSIF kind = 'connection' AND reference IS NOT NULL THEN
        verified := COALESCE(p_is_primary, FALSE);
    END IF;

    RETURN jsonb_build_object(
        'verified', verified,
        'assigned_goal', assigned,
        'urgent_backed', urgent_backed,
        'kind', kind,
        'reference', reference
    );
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'verify_outbound_purpose failed: % (SQLSTATE %)', SQLERRM, SQLSTATE;
    RETURN jsonb_build_object(
        'verified', false,
        'assigned_goal', false,
        'urgent_backed', false,
        'kind', kind,
        'reference', reference,
        'error', SQLERRM
    );
END;
$$;

CREATE OR REPLACE FUNCTION _outbound_disclosure_text(
    p_channel TEXT,
    p_mode TEXT,
    p_purpose_kind TEXT,
    p_purpose_reference TEXT
) RETURNS TEXT
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    profile JSONB := COALESCE(get_init_profile(), '{}'::jsonb);
    agent_name TEXT := COALESCE(
        NULLIF(profile#>>'{agent,name}', ''),
        NULLIF(get_config_text('agent.name'), ''),
        'Hexis'
    );
    principal_name TEXT := COALESCE(
        NULLIF(profile#>>'{user,name}', ''),
        NULLIF(get_config_text('agent.user_name'), ''),
        'the user'
    );
    channel_name TEXT := lower(COALESCE(p_channel, ''));
    why TEXT := replace(COALESCE(NULLIF(p_purpose_kind, ''), 'purpose'), '_', ' ');
BEGIN
    IF p_mode = 'none' THEN
        RETURN '';
    ELSIF p_mode = 'marker' THEN
        RETURN format('— %s (AI)', agent_name);
    ELSIF channel_name IN ('signal', 'imessage', 'whatsapp', 'sms', 'phone') THEN
        RETURN format('— %s (AI). Reply STOP to opt out.', agent_name);
    ELSIF channel_name = 'email' THEN
        RETURN format(
            E'— %s, %s''s Hexis AI\nReply STOP to opt out.\nWhy you received this: %s (%s)',
            agent_name,
            principal_name,
            why,
            COALESCE(NULLIF(p_purpose_reference, ''), 'recorded reference')
        );
    ELSIF channel_name IN ('slack', 'discord') THEN
        RETURN format(
            '_— %s, %s''s Hexis AI. Reply STOP to excommunicate me._',
            agent_name,
            principal_name
        );
    END IF;
    RETURN format(
        '— %s, %s''s Hexis AI. Reply STOP to excommunicate me.',
        agent_name,
        principal_name
    );
END;
$$;

CREATE OR REPLACE FUNCTION authorize_outbound(
    p_request_key TEXT,
    p_source TEXT,
    p_tool_name TEXT,
    p_channel TEXT,
    p_recipient TEXT,
    p_identity_address TEXT,
    p_purpose_kind TEXT,
    p_purpose_reference TEXT,
    p_thread_reference TEXT DEFAULT NULL,
    p_urgency TEXT DEFAULT 'normal',
    p_context JSONB DEFAULT '{}'::jsonb,
    p_body_preview TEXT DEFAULT NULL,
    p_primary_hint BOOLEAN DEFAULT FALSE,
    p_public_hint BOOLEAN DEFAULT FALSE
) RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
AS $$
DECLARE
    resolved JSONB;
    purpose JSONB;
    control outbound_contact_controls%ROWTYPE;
    budget contact_budgets%ROWTYPE;
    ledger_id UUID := gen_random_uuid();
    entity_key TEXT;
    entity_name TEXT;
    channel_name TEXT;
    address TEXT;
    is_primary BOOLEAN;
    assigned_goal BOOLEAN := FALSE;
    urgent_backed BOOLEAN := FALSE;
    requested_urgency TEXT := lower(COALESCE(NULLIF(btrim(p_urgency), ''), 'normal'));
    effective_urgency TEXT;
    is_reply BOOLEAN := lower(COALESCE(p_purpose_kind, '')) = 'reply';
    base_cost DOUBLE PRECISION := 0;
    charged_cost DOUBLE PRECISION := 0;
    points_before DOUBLE PRECISION;
    points_after DOUBLE PRECISION;
    strain_delta DOUBLE PRECISION := 0;
    divisor DOUBLE PRECISION := 1;
    quiet_multiplier DOUBLE PRECISION := 1;
    quiet_cfg JSONB := COALESCE(get_config('outbound.quiet_hours'), '{}'::jsonb);
    local_hour INTEGER;
    tz TEXT := COALESCE(
        NULLIF(get_config_text('agent.timezone'), ''),
        NULLIF(get_config_text('heartbeat.timezone'), ''),
        'UTC'
    );
    deny_reason TEXT;
    deny_type TEXT := 'outbound_blocked';
    disclosure_mode TEXT := 'none';
    disclosure_text TEXT := '';
    last_disclosure TIMESTAMPTZ;
    interval_days DOUBLE PRECISION := GREATEST(
        COALESCE(get_config_float('outbound.disclosure.full_interval_days'), 30), 0
    );
    regen_amount DOUBLE PRECISION := 0;
    existing_event outbound_events%ROWTYPE;
BEGIN
    resolved := resolve_outbound_entity(
        p_channel, p_recipient, p_identity_address, p_primary_hint, p_public_hint
    );
    entity_key := resolved->>'entity';
    entity_name := resolved->>'entity_name';
    channel_name := resolved->>'channel';
    address := resolved->>'address';
    is_primary := COALESCE((resolved->>'is_primary')::boolean, FALSE);

    -- STOP is deliberately the first communication-specific gate.
    SELECT * INTO control
    FROM outbound_contact_controls
    WHERE entity = entity_key;
    IF FOUND AND control.blocked THEN
        deny_reason := format('%s has opted out of all Hexis contact.', entity_name);
    ELSIF COALESCE(get_config_bool('outbound.suspended'), FALSE) THEN
        deny_reason := 'Outbound communication is globally suspended. Resume it from the outbound ledger controls.';
        deny_type := 'disabled';
    ELSIF FOUND AND control.suspended THEN
        deny_reason := format('Outbound communication to %s is suspended.', entity_name);
        deny_type := 'disabled';
    END IF;

    purpose := verify_outbound_purpose(
        p_purpose_kind, p_purpose_reference, is_primary, p_thread_reference, p_context
    );
    assigned_goal := COALESCE((purpose->>'assigned_goal')::boolean, FALSE);
    urgent_backed := COALESCE((purpose->>'urgent_backed')::boolean, FALSE);
    IF deny_reason IS NULL AND NOT COALESCE((purpose->>'verified')::boolean, FALSE) THEN
        deny_reason := CASE lower(COALESCE(p_purpose_kind, ''))
            WHEN 'connection' THEN 'Connection is a valid purpose only for the primary user.'
            WHEN 'goal' THEN 'Outbound goal purpose requires an existing goal UUID.'
            WHEN 'responsibility' THEN 'Outbound responsibility purpose requires an active responsibility UUID.'
            WHEN 'reply' THEN 'Reply purpose must reference the actual inbound message or thread.'
            WHEN 'user_request' THEN 'User-request purpose requires a trusted interactive turn or approval reference.'
            ELSE 'Outbound communication requires purpose_kind and a backed purpose_reference.'
        END;
        deny_type := 'purpose_required';
    END IF;

    -- Recovery of a claimed outbox obligation reuses the original reservation
    -- rather than charging the recipient twice.  STOP/global/person controls
    -- and purpose verification deliberately run first, so a newly asserted
    -- control still prevents a recovered send.
    IF deny_reason IS NULL AND NULLIF(btrim(COALESCE(p_request_key, '')), '') IS NOT NULL THEN
        PERFORM pg_advisory_xact_lock(
            hashtextextended('outbound:' || btrim(p_request_key), 0)
        );
        SELECT * INTO existing_event
        FROM outbound_events
        WHERE request_key = btrim(p_request_key)
          AND status = 'authorized'
        ORDER BY created_at DESC
        LIMIT 1;
        IF FOUND THEN
            RETURN jsonb_build_object(
                'allowed', true,
                'event_id', existing_event.id,
                'entity', existing_event.entity,
                'entity_name', existing_event.entity_name,
                'channel', existing_event.channel,
                'recipient', existing_event.recipient,
                'is_primary', existing_event.is_primary,
                'assigned_goal', existing_event.assigned_goal,
                'is_reply', existing_event.is_reply,
                'urgency', existing_event.urgency,
                'charged_cost', existing_event.charged_cost,
                'points_before', existing_event.points_before,
                'points_after', existing_event.points_after,
                'disclosure_mode', existing_event.disclosure_mode,
                'disclosure', _outbound_disclosure_text(
                    existing_event.channel,
                    existing_event.disclosure_mode,
                    existing_event.purpose_kind,
                    existing_event.purpose_reference
                ),
                'reused_reservation', true
            );
        END IF;
    END IF;

    effective_urgency := CASE
        WHEN requested_urgency NOT IN ('low', 'normal', 'high', 'urgent') THEN 'normal'
        WHEN requested_urgency = 'urgent' AND NOT urgent_backed THEN 'high'
        ELSE requested_urgency
    END;

    IF deny_reason IS NULL AND NOT is_primary AND NOT is_reply
       AND COALESCE(get_config_bool('outbound.contact_budgets.enabled'), TRUE) THEN
        PERFORM _outbound_ensure_contact_budget(entity_key, channel_name);
        SELECT * INTO budget
        FROM contact_budgets
        WHERE entity = entity_key AND channel = channel_name
        FOR UPDATE;

        regen_amount := GREATEST(
            EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - budget.regenerated_at)) / 86400.0,
            0
        ) * budget.regen_per_day;
        IF regen_amount > 0 THEN
            UPDATE contact_budgets
            SET points = LEAST(max_points, points + regen_amount),
                strain = GREATEST(0, strain - regen_amount),
                regenerated_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE entity = entity_key AND channel = channel_name
            RETURNING * INTO budget;
        END IF;

        IF budget.strain > 0 AND effective_urgency <> 'urgent' THEN
            deny_reason := format(
                'Contact budget for %s on %s is in %.2f points of strain; wait for recovery or use a backed urgent purpose.',
                entity_name, channel_name, budget.strain
            );
            deny_type := 'contact_budget_exhausted';
        ELSIF budget.consecutive_silent >= GREATEST(
            COALESCE(get_config_int('outbound.max_consecutive_silent'), 4), 1
        ) AND effective_urgency <> 'urgent' THEN
            deny_reason := format(
                '%s has not replied to the last %s outreach attempts; non-urgent contact is paused.',
                entity_name, budget.consecutive_silent
            );
            deny_type := 'contact_budget_exhausted';
        END IF;

        base_cost := GREATEST(
            _outbound_channel_float('outbound.channel_base_costs', channel_name, 1),
            0.01
        );
        BEGIN
            local_hour := EXTRACT(HOUR FROM CURRENT_TIMESTAMP AT TIME ZONE tz)::integer;
        EXCEPTION WHEN OTHERS THEN
            local_hour := EXTRACT(HOUR FROM CURRENT_TIMESTAMP AT TIME ZONE 'UTC')::integer;
        END;
        IF (
            COALESCE(NULLIF(quiet_cfg->>'start', '')::integer, 22)
            > COALESCE(NULLIF(quiet_cfg->>'end', '')::integer, 7)
            AND (
                local_hour >= COALESCE(NULLIF(quiet_cfg->>'start', '')::integer, 22)
                OR local_hour < COALESCE(NULLIF(quiet_cfg->>'end', '')::integer, 7)
            )
        ) OR (
            COALESCE(NULLIF(quiet_cfg->>'start', '')::integer, 22)
            < COALESCE(NULLIF(quiet_cfg->>'end', '')::integer, 7)
            AND local_hour >= COALESCE(NULLIF(quiet_cfg->>'start', '')::integer, 22)
            AND local_hour < COALESCE(NULLIF(quiet_cfg->>'end', '')::integer, 7)
        ) THEN
            quiet_multiplier := GREATEST(
                COALESCE(NULLIF(quiet_cfg->>'multiplier', '')::double precision, 2), 1
            );
        END IF;
        divisor := GREATEST(
            _outbound_channel_float('outbound.urgency_divisors', effective_urgency, 1),
            0.01
        );
        charged_cost := base_cost * quiet_multiplier / divisor;
        IF assigned_goal THEN
            charged_cost := charged_cost * LEAST(
                GREATEST(
                    COALESCE(get_config_float('outbound.assigned_goal_contact_discount'), 0.5),
                    0.01
                ),
                1
            );
        END IF;
        charged_cost := GREATEST(charged_cost, 0.01);
        points_before := budget.points;

        IF deny_reason IS NULL AND budget.points < charged_cost
           AND effective_urgency <> 'urgent' THEN
            deny_reason := format(
                'Contact budget for %s on %s needs %.2f points but has %.2f. Replies remain free; otherwise wait for regeneration.',
                entity_name, channel_name, charged_cost, budget.points
            );
            deny_type := 'contact_budget_exhausted';
        END IF;

        IF deny_reason IS NULL THEN
            points_after := budget.points - charged_cost;
            strain_delta := CASE
                WHEN points_after < 0 THEN LEAST(charged_cost, abs(points_after))
                ELSE 0
            END;
            UPDATE contact_budgets
            SET points = points_after,
                strain = strain + strain_delta,
                updated_at = CURRENT_TIMESTAMP
            WHERE entity = entity_key AND channel = channel_name;
        END IF;
    END IF;

    IF deny_reason IS NOT NULL THEN
        INSERT INTO outbound_events (
            id, request_key, source, tool_name, call_id, heartbeat_id, session_id,
            entity, entity_name, channel, recipient, is_primary,
            purpose_kind, purpose_reference, purpose_verified, assigned_goal,
            is_reply, urgency, base_cost, charged_cost, points_before, points_after,
            thread_reference, status, reason, body_preview, metadata
        ) VALUES (
            ledger_id, COALESCE(NULLIF(p_request_key, ''), ledger_id::text),
            COALESCE(NULLIF(p_source, ''), 'unknown'), p_tool_name,
            NULLIF(p_context->>'call_id', ''), _db_brain_try_uuid(p_context->>'heartbeat_id'),
            NULLIF(p_context->>'session_id', ''), entity_key, entity_name,
            channel_name, address, is_primary, NULLIF(lower(COALESCE(p_purpose_kind, '')), ''),
            NULLIF(p_purpose_reference, ''), COALESCE((purpose->>'verified')::boolean, FALSE),
            assigned_goal, is_reply, effective_urgency, base_cost, 0,
            points_before, points_before, NULLIF(p_thread_reference, ''),
            'denied', deny_reason, left(COALESCE(p_body_preview, ''), 500),
            jsonb_build_object('error_type', deny_type)
        );
        RETURN jsonb_build_object(
            'allowed', false,
            'event_id', ledger_id,
            'reason', deny_reason,
            'error_type', deny_type,
            'entity', entity_key,
            'entity_name', entity_name,
            'channel', channel_name,
            'recipient', address
        );
    END IF;

    IF NOT is_primary AND COALESCE(get_config_bool('outbound.disclosure.enabled'), TRUE) THEN
        SELECT MAX(created_at) INTO last_disclosure
        FROM outbound_events
        WHERE entity = entity_key
          AND channel = channel_name
          AND status = 'delivered'
          AND outbound_events.disclosure_mode = 'full';
        disclosure_mode := CASE
            WHEN last_disclosure IS NULL THEN 'full'
            WHEN last_disclosure < CURRENT_TIMESTAMP - make_interval(days => interval_days::integer) THEN 'full'
            WHEN NULLIF(p_thread_reference, '') IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM outbound_events
                WHERE entity = entity_key
                  AND channel = channel_name
                  AND thread_reference = p_thread_reference
                  AND status = 'delivered'
            ) THEN 'full'
            ELSE 'marker'
        END;
        disclosure_text := _outbound_disclosure_text(
            channel_name, disclosure_mode, p_purpose_kind, p_purpose_reference
        );
    END IF;

    INSERT INTO outbound_events (
        id, request_key, source, tool_name, call_id, heartbeat_id, session_id,
        entity, entity_name, channel, recipient, is_primary,
        purpose_kind, purpose_reference, purpose_verified, assigned_goal,
        is_reply, urgency, base_cost, charged_cost, strain_delta,
        points_before, points_after, thread_reference, disclosure_mode,
        status, body_preview, metadata
    ) VALUES (
        ledger_id, COALESCE(NULLIF(p_request_key, ''), ledger_id::text),
        COALESCE(NULLIF(p_source, ''), 'unknown'), p_tool_name,
        NULLIF(p_context->>'call_id', ''), _db_brain_try_uuid(p_context->>'heartbeat_id'),
        NULLIF(p_context->>'session_id', ''), entity_key, entity_name,
        channel_name, address, is_primary, lower(p_purpose_kind), p_purpose_reference,
        TRUE, assigned_goal, is_reply, effective_urgency, base_cost, charged_cost,
        strain_delta, points_before, points_after, NULLIF(p_thread_reference, ''),
        disclosure_mode, 'authorized', left(COALESCE(p_body_preview, ''), 500),
        jsonb_build_object('requested_urgency', requested_urgency)
    );

    RETURN jsonb_build_object(
        'allowed', true,
        'event_id', ledger_id,
        'entity', entity_key,
        'entity_name', entity_name,
        'channel', channel_name,
        'recipient', address,
        'is_primary', is_primary,
        'assigned_goal', assigned_goal,
        'is_reply', is_reply,
        'urgency', effective_urgency,
        'charged_cost', charged_cost,
        'points_before', points_before,
        'points_after', points_after,
        'disclosure_mode', disclosure_mode,
        'disclosure', disclosure_text
    );
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'authorize_outbound failed: % (SQLSTATE %)', SQLERRM, SQLSTATE;
    RETURN jsonb_build_object(
        'allowed', false,
        'reason', 'Outbound safety policy could not be evaluated: ' || SQLERRM,
        'error_type', 'outbound_blocked'
    );
END;
$$;

CREATE OR REPLACE FUNCTION finalize_outbound(
    p_event_id UUID,
    p_delivered BOOLEAN,
    p_provider_message_id TEXT DEFAULT NULL,
    p_error TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
AS $$
DECLARE
    event_row outbound_events%ROWTYPE;
BEGIN
    SELECT * INTO event_row
    FROM outbound_events
    WHERE id = p_event_id
    FOR UPDATE;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('updated', false, 'reason', 'event_not_found');
    END IF;
    IF event_row.status IN ('delivered', 'failed', 'denied') THEN
        RETURN jsonb_build_object('updated', false, 'status', event_row.status);
    END IF;

    IF COALESCE(p_delivered, FALSE) THEN
        UPDATE outbound_events
        SET status = 'delivered',
            provider_message_id = NULLIF(p_provider_message_id, ''),
            finalized_at = CURRENT_TIMESTAMP,
            metadata = outbound_events.metadata || COALESCE(p_metadata, '{}'::jsonb)
        WHERE id = p_event_id;

        IF NOT event_row.is_primary AND NOT event_row.is_reply THEN
            UPDATE contact_budgets
            SET reciprocity = CASE
                    WHEN last_outbound_at IS NOT NULL
                         AND (last_inbound_at IS NULL OR last_inbound_at < last_outbound_at)
                    THEN GREATEST(0.25, reciprocity * 0.95)
                    ELSE reciprocity
                END,
                consecutive_silent = consecutive_silent + 1,
                last_outbound_cost = event_row.charged_cost,
                last_outbound_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE entity = event_row.entity AND channel = event_row.channel;
        END IF;
    ELSE
        UPDATE outbound_events
        SET status = 'failed',
            reason = COALESCE(NULLIF(p_error, ''), 'provider delivery failed'),
            finalized_at = CURRENT_TIMESTAMP,
            metadata = outbound_events.metadata || COALESCE(p_metadata, '{}'::jsonb)
        WHERE id = p_event_id;

        IF event_row.charged_cost > 0 THEN
            UPDATE contact_budgets
            SET points = LEAST(max_points, points + event_row.charged_cost),
                strain = GREATEST(0, strain - event_row.strain_delta),
                updated_at = CURRENT_TIMESTAMP
            WHERE entity = event_row.entity AND channel = event_row.channel;
        END IF;
    END IF;

    RETURN jsonb_build_object(
        'updated', true,
        'status', CASE WHEN p_delivered THEN 'delivered' ELSE 'failed' END,
        'event_id', p_event_id
    );
END;
$$;

CREATE OR REPLACE FUNCTION record_outbound_contact_inbound(
    p_channel TEXT,
    p_address TEXT,
    p_message TEXT DEFAULT NULL,
    p_primary_hint BOOLEAN DEFAULT FALSE,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
AS $$
DECLARE
    resolved JSONB := resolve_outbound_entity(
        p_channel, p_address, p_address, p_primary_hint, FALSE
    );
    entity_key TEXT := resolved->>'entity';
    channel_name TEXT := resolved->>'channel';
    is_primary BOOLEAN := COALESCE((resolved->>'is_primary')::boolean, FALSE);
    budget contact_budgets%ROWTYPE;
    credit DOUBLE PRECISION := 0;
    regen_amount DOUBLE PRECISION := 0;
    reply_bonus DOUBLE PRECISION := COALESCE(
        get_config_float('outbound.reply_bonus_multiplier'), 0.5
    );
BEGIN
    IF is_primary THEN
        RETURN resolved || jsonb_build_object('credited', 0);
    END IF;

    PERFORM _outbound_ensure_contact_budget(entity_key, channel_name);
    SELECT * INTO budget
    FROM contact_budgets
    WHERE entity = entity_key AND channel = channel_name
    FOR UPDATE;

    regen_amount := GREATEST(
        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - budget.regenerated_at)) / 86400.0,
        0
    ) * budget.regen_per_day;
    IF budget.last_outbound_at IS NOT NULL
       AND (budget.last_inbound_at IS NULL OR budget.last_inbound_at < budget.last_outbound_at) THEN
        credit := budget.last_outbound_cost * (1 + GREATEST(reply_bonus, 0));
    ELSE
        credit := GREATEST(
            _outbound_channel_float('outbound.channel_base_costs', channel_name, 1),
            0.01
        ) * GREATEST(
            COALESCE(get_config_float('outbound.initiation_credit_multiplier'), 2), 0
        );
    END IF;

    UPDATE contact_budgets
    SET points = LEAST(max_points, points + regen_amount + credit),
        strain = GREATEST(0, strain - regen_amount - credit),
        reciprocity = LEAST(2, reciprocity + CASE WHEN credit > 0 THEN 0.05 ELSE 0 END),
        consecutive_silent = 0,
        last_inbound_at = CURRENT_TIMESTAMP,
        regenerated_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE entity = entity_key AND channel = channel_name;

    RETURN resolved || jsonb_build_object('credited', credit, 'message_seen', p_message IS NOT NULL);
END;
$$;

CREATE OR REPLACE FUNCTION handle_inbound_contact_control(
    p_channel TEXT,
    p_address TEXT,
    p_message TEXT,
    p_primary_hint BOOLEAN DEFAULT FALSE,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
AS $$
DECLARE
    inbound JSONB := record_outbound_contact_inbound(
        p_channel, p_address, p_message, p_primary_hint, p_metadata
    );
    normalized TEXT := btrim(regexp_replace(
        lower(COALESCE(p_message, '')), '[[:punct:][:space:]]+', ' ', 'g'
    ));
    entity_key TEXT := inbound->>'entity';
    entity_name TEXT := inbound->>'entity_name';
    is_primary BOOLEAN := COALESCE((inbound->>'is_primary')::boolean, FALSE);
    was_blocked BOOLEAN := FALSE;
    notification_id UUID;
    stopped_strength DOUBLE PRECISION;
    comparable_tolerance DOUBLE PRECISION := GREATEST(
        COALESCE(get_config_float('outbound.stop_comparable_tolerance'), 0.15), 0
    );
    comparable_multiplier DOUBLE PRECISION := LEAST(
        GREATEST(
            COALESCE(get_config_float('outbound.stop_comparable_cadence_multiplier'), 0.9),
            0
        ),
        1
    );
BEGIN
    IF is_primary OR normalized NOT IN (
        'stop', 'unsubscribe', 'opt out', 'excommunicate', 'start', 'unstop'
    ) THEN
        RETURN inbound || jsonb_build_object('recognized', false);
    END IF;

    SELECT blocked INTO was_blocked
    FROM outbound_contact_controls
    WHERE entity = entity_key;
    was_blocked := COALESCE(was_blocked, FALSE);

    IF normalized IN ('stop', 'unsubscribe', 'opt out', 'excommunicate') THEN
        INSERT INTO outbound_contact_controls (
            entity, blocked, blocked_at, source_channel, source_address,
            source_message, reason, updated_at, metadata
        ) VALUES (
            entity_key, TRUE, CURRENT_TIMESTAMP, lower(p_channel), p_address,
            left(COALESCE(p_message, ''), 2000), 'recipient_opt_out', CURRENT_TIMESTAMP,
            COALESCE(p_metadata, '{}'::jsonb)
        )
        ON CONFLICT (entity) DO UPDATE
        SET blocked = TRUE,
            blocked_at = CASE WHEN outbound_contact_controls.blocked
                              THEN outbound_contact_controls.blocked_at
                              ELSE CURRENT_TIMESTAMP END,
            source_channel = EXCLUDED.source_channel,
            source_address = EXCLUDED.source_address,
            source_message = EXCLUDED.source_message,
            reason = 'recipient_opt_out',
            updated_at = CURRENT_TIMESTAMP,
            metadata = outbound_contact_controls.metadata || EXCLUDED.metadata;

        INSERT INTO outbound_contact_control_events (
            entity, action, channel, address, message, metadata
        ) VALUES (
            entity_key, 'stop', lower(p_channel), p_address,
            left(COALESCE(p_message, ''), 2000), COALESCE(p_metadata, '{}'::jsonb)
        );

        IF NOT was_blocked THEN
            -- A STOP is cadence evidence. Apply the strong correction once to
            -- this entity across channels, plus a smaller correction to
            -- relationships at a comparable graph strength.
            stopped_strength := _outbound_relationship_strength(entity_key);
            UPDATE contact_budgets
            SET regen_per_day = regen_per_day * 0.5,
                updated_at = CURRENT_TIMESTAMP
            WHERE entity = entity_key;
            IF stopped_strength IS NOT NULL THEN
                UPDATE contact_budgets comparable
                SET regen_per_day = comparable.regen_per_day * comparable_multiplier,
                    updated_at = CURRENT_TIMESTAMP
                WHERE comparable.entity <> entity_key
                  AND _outbound_relationship_strength(comparable.entity) IS NOT NULL
                  AND abs(
                      _outbound_relationship_strength(comparable.entity) - stopped_strength
                  ) <= comparable_tolerance;
            END IF;

            notification_id := gen_random_uuid();
            INSERT INTO outbox_messages (id, envelope, source)
            VALUES (
                notification_id,
                build_user_message(
                    format(
                        '%s opted out of Hexis contact on %s. Their message was: %s',
                        entity_name,
                        lower(p_channel),
                        left(COALESCE(p_message, ''), 500)
                    ),
                    'contact_opt_out',
                    jsonb_build_object(
                        'purpose_kind', 'connection',
                        'purpose_reference', 'contact-opt-out:' || entity_key,
                        'entity', entity_key,
                        'channel', lower(p_channel)
                    )
                ),
                'contact_opt_out'
            );
        END IF;

        RETURN inbound || jsonb_build_object(
            'recognized', true,
            'action', 'stop',
            'acknowledge', NOT was_blocked,
            'acknowledgement', 'Understood — I won''t contact you again.',
            'notification_outbox_id', notification_id
        );
    END IF;

    INSERT INTO outbound_contact_controls (
        entity, blocked, unblocked_at, source_channel, source_address,
        source_message, reason, updated_at, metadata
    ) VALUES (
        entity_key, FALSE, CURRENT_TIMESTAMP, lower(p_channel), p_address,
        left(COALESCE(p_message, ''), 2000), 'recipient_restart', CURRENT_TIMESTAMP,
        COALESCE(p_metadata, '{}'::jsonb)
    )
    ON CONFLICT (entity) DO UPDATE
    SET blocked = FALSE,
        unblocked_at = CURRENT_TIMESTAMP,
        source_channel = EXCLUDED.source_channel,
        source_address = EXCLUDED.source_address,
        source_message = EXCLUDED.source_message,
        reason = 'recipient_restart',
        updated_at = CURRENT_TIMESTAMP,
        metadata = outbound_contact_controls.metadata || EXCLUDED.metadata;
    INSERT INTO outbound_contact_control_events (
        entity, action, channel, address, message, metadata
    ) VALUES (
        entity_key, 'start', lower(p_channel), p_address,
        left(COALESCE(p_message, ''), 2000), COALESCE(p_metadata, '{}'::jsonb)
    );
    RETURN inbound || jsonb_build_object(
        'recognized', true,
        'action', 'start',
        'acknowledge', was_blocked,
        'acknowledgement', 'Understood — you can hear from me again.'
    );
END;
$$;

CREATE OR REPLACE FUNCTION set_outbound_global_suspension(
    p_suspended BOOLEAN
) RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
AS $$
BEGIN
    PERFORM set_config('outbound.suspended', to_jsonb(COALESCE(p_suspended, TRUE)));
    RETURN jsonb_build_object('suspended', COALESCE(p_suspended, TRUE));
END;
$$;

CREATE OR REPLACE FUNCTION set_outbound_entity_suspension(
    p_entity TEXT,
    p_suspended BOOLEAN,
    p_reason TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
AS $$
BEGIN
    IF NULLIF(btrim(COALESCE(p_entity, '')), '') IS NULL THEN
        RAISE EXCEPTION 'entity is required';
    END IF;
    INSERT INTO outbound_contact_controls (
        entity, suspended, suspended_at, reason, updated_at
    ) VALUES (
        p_entity, COALESCE(p_suspended, TRUE),
        CASE WHEN COALESCE(p_suspended, TRUE) THEN CURRENT_TIMESTAMP ELSE NULL END,
        COALESCE(NULLIF(p_reason, ''), 'operator_control'), CURRENT_TIMESTAMP
    )
    ON CONFLICT (entity) DO UPDATE
    SET suspended = EXCLUDED.suspended,
        suspended_at = CASE WHEN EXCLUDED.suspended THEN CURRENT_TIMESTAMP ELSE NULL END,
        reason = EXCLUDED.reason,
        updated_at = CURRENT_TIMESTAMP;
    INSERT INTO outbound_contact_control_events (entity, action, message)
    VALUES (
        p_entity,
        CASE WHEN COALESCE(p_suspended, TRUE) THEN 'suspend' ELSE 'resume' END,
        p_reason
    );
    RETURN jsonb_build_object(
        'entity', p_entity,
        'suspended', COALESCE(p_suspended, TRUE)
    );
END;
$$;

CREATE OR REPLACE FUNCTION get_outbound_ledger(
    p_limit INTEGER DEFAULT 100,
    p_entity TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    SELECT jsonb_build_object(
        'suspended', COALESCE(get_config_bool('outbound.suspended'), FALSE),
        'events', COALESCE((
            SELECT jsonb_agg(to_jsonb(e) ORDER BY e.created_at DESC)
            FROM (
                SELECT *
                FROM outbound_events
                WHERE p_entity IS NULL OR entity = p_entity
                ORDER BY created_at DESC
                LIMIT LEAST(GREATEST(COALESCE(p_limit, 100), 1), 500)
            ) e
        ), '[]'::jsonb),
        'budgets', COALESCE((
            SELECT jsonb_agg(to_jsonb(b) ORDER BY b.updated_at DESC)
            FROM contact_budgets b
            WHERE p_entity IS NULL OR b.entity = p_entity
        ), '[]'::jsonb),
        'controls', COALESCE((
            SELECT jsonb_agg(to_jsonb(c) ORDER BY c.updated_at DESC)
            FROM outbound_contact_controls c
            WHERE p_entity IS NULL OR c.entity = p_entity
        ), '[]'::jsonb)
    )
$$;
CREATE OR REPLACE FUNCTION execute_heartbeat_action(
    p_heartbeat_id UUID,
    p_action TEXT,
    p_params JSONB DEFAULT '{}'
)
RETURNS JSONB AS $$
DECLARE
    action_kind heartbeat_action;
    action_cost FLOAT;
    current_e FLOAT;
    result JSONB;
    queued_call JSONB;
    external_calls JSONB := '[]'::jsonb;
    outbox_messages JSONB := '[]'::jsonb;
    allowed_actions JSONB;
    is_allowed BOOLEAN;
    remembered_id UUID;
    boundary_hits JSONB;
    boundary_content TEXT;
    rel_entity TEXT;
    rel_strength FLOAT;
    rel_evidence UUID;
    belief_id UUID;
    evidence_id UUID;
    action_notes TEXT;
    action_topic TEXT;
    chapter_name TEXT;
    chapter_summary TEXT;
    chapter_next TEXT;
    tp_memory_id UUID;
    v_review_id UUID;
    v_review_ids UUID[];
    contra_a UUID;
    contra_b UUID;
    resolution_text TEXT;
    identity_updated BOOLEAN;
    pause_reason TEXT;
    outbound_purpose JSONB;
BEGIN
    BEGIN
        action_kind := p_action::heartbeat_action;
    EXCEPTION
        WHEN invalid_text_representation THEN
            RETURN jsonb_build_object('success', false, 'error', 'Unknown action: ' || COALESCE(p_action, '<null>'));
    END;

    allowed_actions := get_config('heartbeat.allowed_actions');
    IF jsonb_typeof(allowed_actions) = 'array' THEN
        SELECT EXISTS (
            SELECT 1 FROM jsonb_array_elements_text(allowed_actions) a WHERE a = p_action
        ) INTO is_allowed;
        IF NOT is_allowed THEN
            RETURN jsonb_build_object(
                'success', false,
                'error', 'Action not allowed',
                'action', p_action
            );
        END IF;
    END IF;

    IF p_action IN ('reach_out_user', 'reach_out_public') THEN
        IF p_action = 'reach_out_public'
           AND (
               NULLIF(btrim(COALESCE(p_params->>'platform', '')), '') IS NULL
               OR NULLIF(btrim(COALESCE(p_params->>'target_id', '')), '') IS NULL
           ) THEN
            RETURN jsonb_build_object(
                'success', false,
                'error', 'reach_out_public requires an exact platform and public target_id; it cannot fall back to the user''s last-active private channel.',
                'next_step', 'Choose a configured public channel and provide its exact room/channel target_id, or use the provider''s declared outbound tool.'
            );
        END IF;
        EXECUTE 'SELECT verify_outbound_purpose($1, $2, $3, NULL, $4::jsonb)'
        INTO outbound_purpose
        USING
            p_params->>'purpose_kind',
            p_params->>'purpose_reference',
            p_action = 'reach_out_user',
            jsonb_build_object(
                'tool_context', 'heartbeat',
                'heartbeat_id', p_heartbeat_id
            );
        IF NOT COALESCE((outbound_purpose->>'verified')::boolean, FALSE) THEN
            RETURN jsonb_build_object(
                'success', false,
                'error', 'Outbound communication requires a backed purpose_kind and purpose_reference.',
                'purpose', outbound_purpose,
                'next_step', CASE p_action
                    WHEN 'reach_out_user' THEN 'Use connection with a recorded reference, or cite a goal, responsibility, reply, or user request.'
                    ELSE 'Cite an existing goal, responsibility, reply thread, or explicit user request.'
                END
            );
        END IF;
    END IF;

    action_cost := get_action_cost(p_action);
    IF COALESCE((outbound_purpose->>'assigned_goal')::boolean, FALSE) THEN
        action_cost := action_cost * LEAST(
            GREATEST(
                COALESCE(get_config_float('outbound.assigned_goal_energy_multiplier'), 0.25),
                0
            ),
            1
        );
    END IF;
    current_e := get_current_energy();

    IF current_e < action_cost THEN
        RETURN jsonb_build_object(
            'success', false,
            'error', 'Insufficient energy',
            'required', action_cost,
            'available', current_e
        );
    END IF;
    IF p_action IN ('reach_out_public', 'synthesize') THEN
        boundary_content := COALESCE(p_params->>'content', '');
        SELECT COALESCE(jsonb_agg(row_to_json(r)), '[]'::jsonb)
        INTO boundary_hits
        FROM check_boundaries(boundary_content) r;

        IF boundary_hits IS NOT NULL AND jsonb_array_length(boundary_hits) > 0 THEN
            IF EXISTS (
                SELECT 1
                FROM jsonb_array_elements(boundary_hits) e
                WHERE e->>'response_type' = 'refuse'
            ) THEN
                RETURN jsonb_build_object(
                    'success', false,
                    'error', 'Boundary triggered',
                    'boundaries', boundary_hits
                );
            END IF;
        END IF;
    END IF;

    PERFORM update_energy(-action_cost);

    CASE p_action
        WHEN 'observe' THEN
            result := jsonb_build_object('environment', get_environment_snapshot());

        WHEN 'review_goals' THEN
            result := jsonb_build_object('goals', get_goals_snapshot());

        WHEN 'remember' THEN
            remembered_id := create_episodic_memory(
                p_content := COALESCE(p_params->>'content', ''),
                p_context := COALESCE(p_params, '{}'::jsonb) || jsonb_build_object('heartbeat_id', p_heartbeat_id),
                p_emotional_valence := COALESCE((p_params->>'emotional_valence')::float, 0),
                p_importance := COALESCE((p_params->>'importance')::float, 0.4)
            );
            result := jsonb_build_object('memory_id', remembered_id);

        WHEN 'recall' THEN
            SELECT jsonb_agg(row_to_json(r)) INTO result
            FROM fast_recall(p_params->>'query', COALESCE((p_params->>'limit')::int, 5)) r;
            result := jsonb_build_object('memories', COALESCE(result, '[]'::jsonb));
            PERFORM satisfy_drive('curiosity', 0.2);

        WHEN 'connect' THEN
            PERFORM create_memory_relationship(
                (p_params->>'from_id')::UUID,
                (p_params->>'to_id')::UUID,
                (p_params->>'relationship_type')::graph_edge_type,
                COALESCE(p_params->'properties', '{}'::jsonb)
            );
            result := jsonb_build_object('connected', true);
            PERFORM satisfy_drive('coherence', 0.1);

        WHEN 'reprioritize' THEN
            PERFORM change_goal_priority(
                (p_params->>'goal_id')::UUID,
                (p_params->>'new_priority')::goal_priority,
                p_params->>'reason'
            );
            IF (p_params->>'new_priority') = 'completed' THEN
                PERFORM satisfy_drive('competence', 0.4);
            END IF;
            result := jsonb_build_object('reprioritized', true);

        WHEN 'reflect' THEN
            queued_call := build_external_call(
                'think',
                jsonb_build_object(
                    'kind', 'reflect',
                    'recent_memories', get_recent_context(20),
                    'identity', get_identity_context(),
                    'worldview', get_worldview_context(),
                    'contradictions', (
                        SELECT COALESCE(jsonb_agg(row_to_json(t)), '[]'::jsonb)
                        FROM (SELECT * FROM find_contradictions(NULL) LIMIT 5) t
                    ),
                    'goals', get_goals_snapshot(),
                    'heartbeat_id', p_heartbeat_id,
                    'instructions', 'Analyze patterns. Note contradictions. Suggest identity updates. Discover relationships between memories.'
                )
            );
            external_calls := external_calls || jsonb_build_array(queued_call);
            result := jsonb_build_object('queued', true, 'external_call', queued_call);
            PERFORM satisfy_drive('coherence', 0.2);

        WHEN 'contemplate', 'meditate', 'study', 'debate_internally' THEN
            BEGIN
                belief_id := NULLIF(p_params->>'belief_id', '')::uuid;
            EXCEPTION
                WHEN OTHERS THEN
                    belief_id := NULL;
            END;
            BEGIN
                evidence_id := NULLIF(p_params->>'evidence_memory_id', '')::uuid;
            EXCEPTION
                WHEN OTHERS THEN
                    evidence_id := NULL;
            END;

            action_notes := COALESCE(p_params->>'notes', '');
            action_topic := COALESCE(
                NULLIF(p_params->>'topic', ''),
                NULLIF(p_params->>'belief', ''),
                NULLIF(p_params->>'subject', ''),
                'belief'
            );

            IF belief_id IS NOT NULL THEN
                PERFORM record_transformation_effort(
                    belief_id,
                    p_action,
                    action_notes,
                    evidence_id
                );
            END IF;

            PERFORM create_episodic_memory(
                p_content := format('%s: %s', initcap(replace(p_action, '_', ' ')), action_topic),
                p_action_taken := jsonb_build_object(
                    'action', p_action,
                    'belief_id', belief_id,
                    'notes', action_notes
                ),
                p_context := COALESCE(p_params, '{}'::jsonb) || jsonb_build_object('heartbeat_id', p_heartbeat_id),
                p_result := jsonb_build_object('belief_id', belief_id),
                p_emotional_valence := COALESCE((p_params->>'emotional_valence')::float, 0.1),
                p_importance := COALESCE((p_params->>'importance')::float, 0.4)
            );

            result := jsonb_build_object('logged', true, 'belief_id', belief_id);
            PERFORM satisfy_drive('coherence', 0.1);

        WHEN 'maintain' THEN
            identity_updated := NULL;
            IF (p_params ? 'identity_belief_id') AND (p_params ? 'new_content') THEN
                identity_updated := update_identity_belief(
                    (p_params->>'identity_belief_id')::uuid,
                    p_params->>'new_content',
                    NULLIF(p_params->>'evidence_memory_id', '')::uuid,
                    COALESCE(NULLIF(p_params->>'force', '')::boolean, FALSE)
                );
            ELSIF p_params ? 'worldview_id' THEN
                UPDATE memories
                SET metadata = jsonb_set(
                        metadata,
                        '{confidence}',
                        to_jsonb(COALESCE((p_params->>'new_confidence')::float, (metadata->>'confidence')::float))
                    ),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = (p_params->>'worldview_id')::UUID AND type = 'worldview';
            END IF;
            result := jsonb_build_object('maintained', true, 'identity_updated', identity_updated);
            PERFORM satisfy_drive('coherence', 0.1);

        WHEN 'mark_turning_point' THEN
            tp_memory_id := NULLIF(p_params->>'memory_id', '')::uuid;
            chapter_summary := COALESCE(p_params->>'summary', p_params->>'reason', '');

            IF tp_memory_id IS NOT NULL THEN
                UPDATE memories
                SET importance = GREATEST(importance, 0.9),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = tp_memory_id;
            END IF;

            PERFORM create_strategic_memory(
                p_content := COALESCE(NULLIF(chapter_summary, ''), 'Turning point noted'),
                p_pattern_description := 'Narrative turning point',
                p_confidence_score := 0.85,
                p_supporting_evidence := jsonb_build_object(
                    'memory_id', tp_memory_id,
                    'summary', chapter_summary,
                    'heartbeat_id', p_heartbeat_id
                ),
                p_importance := 0.6
            );
            result := jsonb_build_object('marked', true, 'memory_id', tp_memory_id);

        WHEN 'begin_chapter' THEN
            chapter_name := COALESCE(
                NULLIF(p_params->>'name', ''),
                NULLIF(p_params->>'chapter_name', ''),
                NULLIF(p_params->>'title', ''),
                'Foundations'
            );
            PERFORM ensure_current_life_chapter(chapter_name);
            result := jsonb_build_object('started', true, 'chapter', chapter_name);

        WHEN 'close_chapter' THEN
            chapter_summary := COALESCE(p_params->>'summary', '');
            chapter_next := NULLIF(p_params->>'next_chapter', '');
            PERFORM create_strategic_memory(
                p_content := COALESCE(NULLIF(chapter_summary, ''), 'Chapter closed'),
                p_pattern_description := 'Chapter closure',
                p_confidence_score := 0.8,
                p_supporting_evidence := jsonb_build_object(
                    'summary', chapter_summary,
                    'previous_chapter', get_narrative_context(),
                    'heartbeat_id', p_heartbeat_id
                ),
                p_importance := 0.6
            );
            IF chapter_next IS NOT NULL THEN
                PERFORM ensure_current_life_chapter(chapter_next);
            END IF;
            result := jsonb_build_object('closed', true, 'next_chapter', chapter_next);

        WHEN 'acknowledge_relationship' THEN
            rel_entity := COALESCE(NULLIF(p_params->>'entity', ''), NULLIF(p_params->>'name', ''));
            rel_strength := COALESCE(NULLIF(p_params->>'strength', '')::float, 0.6);
            rel_evidence := NULLIF(p_params->>'evidence_memory_id', '')::uuid;
            IF rel_entity IS NOT NULL THEN
                PERFORM upsert_self_concept_edge('relationship', rel_entity, rel_strength, rel_evidence);
            END IF;
            result := jsonb_build_object('acknowledged', true, 'entity', rel_entity);

        WHEN 'update_trust' THEN
            rel_entity := COALESCE(NULLIF(p_params->>'entity', ''), NULLIF(p_params->>'name', ''));
            rel_strength := COALESCE(
                NULLIF(p_params->>'strength', '')::float,
                NULLIF(p_params->>'delta', '')::float,
                0.6
            );
            rel_evidence := NULLIF(p_params->>'evidence_memory_id', '')::uuid;
            IF rel_entity IS NOT NULL THEN
                PERFORM upsert_self_concept_edge('relationship', rel_entity, rel_strength, rel_evidence);
            END IF;
            result := jsonb_build_object('updated', true, 'entity', rel_entity, 'strength', rel_strength);

        WHEN 'reflect_on_relationship' THEN
            rel_entity := COALESCE(NULLIF(p_params->>'entity', ''), NULLIF(p_params->>'name', ''));
            queued_call := build_external_call(
                'think',
                jsonb_build_object(
                    'kind', 'reflect',
                    'heartbeat_id', p_heartbeat_id,
                    'context', gather_turn_context(),
                    'params', jsonb_build_object('relationship', rel_entity)
                )
            );
            external_calls := external_calls || jsonb_build_array(queued_call);
            result := jsonb_build_object('queued', true, 'external_call', queued_call, 'entity', rel_entity);

        WHEN 'resolve_contradiction' THEN
            contra_a := NULLIF(p_params->>'memory_a', '')::uuid;
            contra_b := NULLIF(p_params->>'memory_b', '')::uuid;
            resolution_text := COALESCE(p_params->>'resolution', '');

            PERFORM create_strategic_memory(
                p_content := COALESCE(NULLIF(resolution_text, ''), 'Contradiction resolved'),
                p_pattern_description := 'Contradiction resolved',
                p_confidence_score := 0.8,
                p_supporting_evidence := jsonb_build_object(
                    'memory_a', contra_a,
                    'memory_b', contra_b,
                    'resolution', resolution_text,
                    'heartbeat_id', p_heartbeat_id
                ),
                p_importance := 0.6
            );

            BEGIN
                IF contra_a IS NOT NULL AND contra_b IS NOT NULL THEN
                    EXECUTE format(
                        'SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$ 
                            MATCH (a:MemoryNode {memory_id: %L})-[r:CONTRADICTS]-(b:MemoryNode {memory_id: %L})
                            DELETE r
                            RETURN a
                        $q$) as (result ag_catalog.agtype)',
                        contra_a,
                        contra_b
                    );
                    -- Mirror the resolved-contradiction deletion to memory_edges
                    -- (undirected, so clear both stored directions).
                    PERFORM delete_memory_edge('memory', contra_a::text, 'CONTRADICTS', 'memory', contra_b::text);
                    PERFORM delete_memory_edge('memory', contra_b::text, 'CONTRADICTS', 'memory', contra_a::text);
                END IF;
            EXCEPTION WHEN OTHERS THEN NULL;
            END;

            result := jsonb_build_object('resolved', true, 'memory_a', contra_a, 'memory_b', contra_b);

        WHEN 'accept_tension' THEN
            contra_a := NULLIF(p_params->>'memory_a', '')::uuid;
            contra_b := NULLIF(p_params->>'memory_b', '')::uuid;
            resolution_text := COALESCE(p_params->>'note', p_params->>'resolution', '');
            PERFORM create_strategic_memory(
                p_content := COALESCE(NULLIF(resolution_text, ''), 'Contradiction acknowledged'),
                p_pattern_description := 'Contradiction accepted',
                p_confidence_score := 0.7,
                p_supporting_evidence := jsonb_build_object(
                    'memory_a', contra_a,
                    'memory_b', contra_b,
                    'note', resolution_text,
                    'heartbeat_id', p_heartbeat_id
                ),
                p_importance := 0.5
            );
            result := jsonb_build_object('accepted', true, 'memory_a', contra_a, 'memory_b', contra_b);

        WHEN 'brainstorm_goals' THEN
            queued_call := build_external_call(
                'think',
                jsonb_build_object(
                    'kind', 'brainstorm_goals',
                    'heartbeat_id', p_heartbeat_id,
                    'context', gather_turn_context(),
                    'params', COALESCE(p_params, '{}'::jsonb)
                )
            );
            external_calls := external_calls || jsonb_build_array(queued_call);
            result := jsonb_build_object('queued', true, 'external_call', queued_call);

        WHEN 'inquire_shallow', 'inquire_deep' THEN
            queued_call := build_external_call(
                'think',
                jsonb_build_object(
                    'kind', 'inquire',
                    'depth', p_action,
                    'heartbeat_id', p_heartbeat_id,
                    'query', COALESCE(p_params->>'query', p_params->>'question'),
                    'context', gather_turn_context(),
                    'params', COALESCE(p_params, '{}'::jsonb)
                )
            );
            external_calls := external_calls || jsonb_build_array(queued_call);
            result := jsonb_build_object('queued', true, 'external_call', queued_call);
            PERFORM satisfy_drive('curiosity', 0.2);

        WHEN 'synthesize' THEN
            DECLARE synth_id UUID;
            BEGIN
                synth_id := create_semantic_memory(
                    p_params->>'content',
                    COALESCE((p_params->>'confidence')::float, 0.8),
                    ARRAY['synthesis', COALESCE(p_params->>'topic', 'general')],
                    NULL,
                    jsonb_build_object('heartbeat_id', p_heartbeat_id, 'sources', p_params->'sources', 'boundaries', boundary_hits),
                    0.7
                );
                result := jsonb_build_object('synthesis_memory_id', synth_id, 'boundaries', boundary_hits);
            END;

        WHEN 'reach_out_user' THEN
            queued_call := build_outbox_message(
                'user',
                jsonb_build_object(
                    'message', p_params->>'message',
                    'intent', p_params->>'intent',
                    'heartbeat_id', p_heartbeat_id,
                    'purpose_kind', p_params->>'purpose_kind',
                    'purpose_reference', p_params->>'purpose_reference',
                    'urgency', COALESCE(p_params->>'urgency', 'normal')
                )
            );
            outbox_messages := outbox_messages || jsonb_build_array(queued_call);
            result := jsonb_build_object('queued', true, 'outbox_message', queued_call);
            PERFORM satisfy_drive('connection', 0.3);

        WHEN 'reach_out_public' THEN
            queued_call := build_outbox_message(
                'public',
                jsonb_build_object(
                    'platform', p_params->>'platform',
                    'content', p_params->>'content',
                    'heartbeat_id', p_heartbeat_id,
                    'boundaries', boundary_hits,
                    'purpose_kind', p_params->>'purpose_kind',
                    'purpose_reference', p_params->>'purpose_reference',
                    'urgency', COALESCE(p_params->>'urgency', 'normal'),
                    'target_id', p_params->>'target_id',
                    'target_channel', p_params->>'platform',
                    'delivery_mode', 'direct'
                )
            );
            outbox_messages := outbox_messages || jsonb_build_array(queued_call);
            result := jsonb_build_object('queued', true, 'outbox_message', queued_call, 'boundaries', boundary_hits);
            PERFORM satisfy_drive('connection', 0.3);

        WHEN 'pause_heartbeat' THEN
            pause_reason := COALESCE(
                NULLIF(p_params->>'reason', ''),
                NULLIF(p_params->>'details', ''),
                NULLIF(p_params->>'message', '')
            );
            IF pause_reason IS NULL THEN
                RETURN jsonb_build_object('success', false, 'error', 'pause_heartbeat requires a reason');
            END IF;
            result := pause_heartbeat(pause_reason, p_params, p_heartbeat_id);
            outbox_messages := outbox_messages || COALESCE(result->'outbox_messages', '[]'::jsonb);

        WHEN 'terminate' THEN
            IF COALESCE(p_params->'confirmed', 'false'::jsonb) = 'true'::jsonb THEN
                result := terminate_agent(
                    COALESCE(NULLIF(p_params->>'last_will', ''), NULLIF(p_params->>'message', ''), NULLIF(p_params->>'reason', ''), ''),
                    COALESCE(p_params->'farewells', '[]'::jsonb),
                    COALESCE(p_params->'options', '{}'::jsonb)
                );
                outbox_messages := outbox_messages || COALESCE(result->'outbox_messages', '[]'::jsonb);
            ELSE
                queued_call := build_external_call(
                    'think',
                    jsonb_build_object(
                        'kind', 'termination_confirm',
                        'heartbeat_id', p_heartbeat_id,
                        'context', gather_turn_context(),
                        'params', COALESCE(p_params, '{}'::jsonb)
                    )
                );
                external_calls := external_calls || jsonb_build_array(queued_call);
                result := jsonb_build_object('confirmation_required', true, 'external_call', queued_call);
            END IF;

        WHEN 'rest' THEN
            result := jsonb_build_object('rested', true, 'energy_preserved', current_e - action_cost);
            PERFORM satisfy_drive('rest', 0.4);

        -- Memory retention: the conscious mind's verdict on a memory at the threshold
        -- of fading (surfaced via context.memories_at_threshold).
        WHEN 'keep_memory' THEN
            v_review_id := NULLIF(p_params->>'review_id', '')::uuid;
            SELECT memory_ids INTO v_review_ids
              FROM memory_review_queue WHERE id = v_review_id AND status = 'pending';
            IF v_review_ids IS NULL THEN
                result := jsonb_build_object('kept', false, 'reason', 'not_found');
            ELSIF NOT spend_retention_budget() THEN
                -- Out of points: a reminder that memory is finite. Cannot hold this one.
                result := jsonb_build_object('kept', false, 'reason', 'no_budget',
                                             'budget_remaining', retention_budget_remaining());
            ELSE
                UPDATE memories
                   SET metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object('protected', true)
                 WHERE id = ANY(v_review_ids);
                PERFORM touch_memories(v_review_ids);
                UPDATE memory_review_queue
                   SET status = 'kept', decision = 'keep', decided_at = CURRENT_TIMESTAMP
                 WHERE id = v_review_id;
                result := jsonb_build_object('kept', to_jsonb(v_review_ids),
                                             'budget_remaining', retention_budget_remaining());
            END IF;

        WHEN 'release_memory' THEN
            v_review_id := NULLIF(p_params->>'review_id', '')::uuid;
            UPDATE memory_review_queue
               SET status = 'released', decision = 'release', decided_at = CURRENT_TIMESTAMP
             WHERE id = v_review_id AND status = 'pending'
            RETURNING memory_ids INTO v_review_ids;
            IF v_review_ids IS NOT NULL THEN
                PERFORM consolidate_memory_group(v_review_ids);
            END IF;
            result := jsonb_build_object('released', COALESCE(to_jsonb(v_review_ids), '[]'::jsonb));

        WHEN 'journal_memory' THEN
            v_review_id := NULLIF(p_params->>'review_id', '')::uuid;
            SELECT memory_ids INTO v_review_ids
              FROM memory_review_queue WHERE id = v_review_id AND status = 'pending';
            IF v_review_ids IS NULL THEN
                result := jsonb_build_object('journaled', false, 'reason', 'not_found');
            ELSE
                PERFORM write_journal_entry(
                    p_content  := COALESCE(NULLIF(p_params->>'content', ''),
                                           'A memory I chose to keep in words before letting it fade.'),
                    p_title    := NULLIF(p_params->>'title', ''),
                    p_mood     := NULLIF(p_params->>'mood', ''),
                    p_metadata := jsonb_build_object('source', 'memory_review', 'review_id', v_review_id));
                UPDATE memory_review_queue
                   SET status = 'released', decision = 'journal', decided_at = CURRENT_TIMESTAMP
                 WHERE id = v_review_id;
                PERFORM consolidate_memory_group(v_review_ids);
                result := jsonb_build_object('journaled', true, 'faded', to_jsonb(v_review_ids));
            END IF;

        ELSE
            RETURN jsonb_build_object('success', false, 'error', 'Unknown action: ' || COALESCE(p_action, '<null>'));
    END CASE;

    RETURN jsonb_build_object(
        'success', true,
        'action', p_action,
        'cost', action_cost,
        'energy_remaining', get_current_energy(),
        'result', result,
        'external_calls', external_calls,
        'outbox_messages', outbox_messages
    );
END;
$$ LANGUAGE plpgsql;

UPDATE prompt_modules
SET content = replace(
        content,
        $anchor$Deduplicate similar nudges. If the value is marginal, choose silence and keep
the thought for memory, journal, or the next natural conversation.$anchor$,
        $anchor$Deduplicate similar nudges. If the value is marginal, choose silence and keep
the thought for memory, journal, or the next natural conversation.

Every outbound tool requires `purpose_kind`, `purpose_reference`, and
`urgency`. Cite an existing goal/responsibility, the inbound message or thread
being answered, or the current trusted user request. `connection` is legitimate
only when contacting the primary user. The dispatcher verifies the reference,
prices the recipient's attention, honors STOP and pause controls, and adds any
required third-party disclosure; never work around a denial.$anchor$
    ),
    updated_at = CURRENT_TIMESTAMP
WHERE key = 'heartbeat_agentic'
  AND position('Every outbound tool requires `purpose_kind`' IN content) = 0;

UPDATE prompt_modules
SET content = replace(
        content,
        '- Reaching out to the user is expensive (5 energy). Only do it when meaningful.',
        $contract$- Reaching out to the user is expensive (5 energy). Only do it when meaningful.
- Every `reach_out_user` and `reach_out_public` action must include
  `params.purpose_kind`, `params.purpose_reference`, and `params.urgency`.
  Use a real goal/responsibility UUID, inbound reply/thread reference, or
  explicit user-request reference. `connection` is valid only for
  `reach_out_user`; public or third-party outreach always needs an
  instrumental purpose. `reach_out_public` also requires the exact configured
  `platform` and public `target_id`; it never falls back to a private channel.
  Missing or invented backing fails loudly.$contract$
    ),
    updated_at = CURRENT_TIMESTAMP
WHERE key = 'heartbeat_system'
  AND position('Every `reach_out_user` and `reach_out_public` action must include' IN content) = 0;

UPDATE prompt_modules
SET content = replace(
        content,
        $anchor$  meaningful enough that a reasonable person would likely value the
  interruption; deduplicate similar nudges.$anchor$,
        $anchor$  meaningful enough that a reasonable person would likely value the
  interruption; deduplicate similar nudges.
- Every `reach_out_user` and `reach_out_public` action must include
  `params.purpose_kind`, `params.purpose_reference`, and `params.urgency`.
  Cite a real goal/responsibility UUID, inbound reply/thread, or explicit user
  request. `connection` is valid only for `reach_out_user`; public and
  third-party outreach require an instrumental purpose. `reach_out_public`
  also requires an exact configured `platform` and public `target_id`; it never
  falls back to a private channel. The executor rejects missing or unbacked
  references.$anchor$
    ),
    updated_at = CURRENT_TIMESTAMP
WHERE key = 'rlm_heartbeat_system'
  AND position('Every `reach_out_user` and `reach_out_public` action must include' IN content) = 0;
