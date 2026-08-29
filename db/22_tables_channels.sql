-- ============================================================================
-- Channel System Tables
--
-- Stores conversation sessions and message logs for channel adapters
-- (Discord, Telegram, etc.)
-- ============================================================================

-- Channel sessions: per-sender conversation state
-- Each baseline file runs as its own psql session at initdb, so it must pin
-- its own search_path: under an ag_catalog-first cluster default, unqualified
-- CREATEs would land Hexis objects in ag_catalog (the #77 fossil bug).
SET search_path = public, ag_catalog, "$user";

CREATE TABLE IF NOT EXISTS channel_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel_type TEXT NOT NULL,            -- 'discord', 'telegram', etc.
    channel_id TEXT NOT NULL,              -- platform chat/channel ID
    sender_id TEXT NOT NULL,               -- platform user ID
    sender_name TEXT,                      -- display name (informational)
    history JSONB DEFAULT '[]'::jsonb,     -- conversation messages array
    last_active TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(channel_type, channel_id, sender_id)
);

-- Channel message log: audit trail for all channel messages
CREATE TABLE IF NOT EXISTS channel_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES channel_sessions(id) ON DELETE CASCADE,
    direction TEXT NOT NULL CHECK (direction IN ('inbound', 'outbound')),
    content TEXT NOT NULL,
    platform_message_id TEXT,              -- platform's message ID
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_channel_sessions_lookup
    ON channel_sessions(channel_type, channel_id, sender_id);

CREATE INDEX IF NOT EXISTS idx_channel_sessions_active
    ON channel_sessions(last_active DESC);

CREATE INDEX IF NOT EXISTS idx_channel_messages_session
    ON channel_messages(session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_channel_messages_created
    ON channel_messages(created_at DESC);

-- Central, auditable reply/observe/wake/drop decisions. The master feature
-- flag is dark by default; rows appear only after the operator enables it.
CREATE TABLE IF NOT EXISTS inbound_disposition_events (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    channel_type TEXT NOT NULL,
    channel_id TEXT,
    sender_id TEXT,
    session_id UUID REFERENCES channel_sessions(id) ON DELETE SET NULL,
    platform_message_id TEXT,
    disposition TEXT NOT NULL
        CHECK (disposition IN ('engage', 'observe', 'wake', 'drop')),
    reason TEXT NOT NULL,
    ambiguous BOOLEAN NOT NULL DEFAULT FALSE,
    classifier_used BOOLEAN NOT NULL DEFAULT FALSE,
    classifier_label TEXT,
    is_operator BOOLEAN NOT NULL DEFAULT FALSE,
    reply_allowed BOOLEAN NOT NULL DEFAULT FALSE,
    preview TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    wake_processed_at TIMESTAMPTZ,
    wake_heartbeat_id UUID,
    wake_outcome TEXT CHECK (
        wake_outcome IS NULL OR wake_outcome IN ('started', 'stale')
    )
);

CREATE INDEX IF NOT EXISTS idx_inbound_disposition_events_ts
    ON inbound_disposition_events (ts DESC);
CREATE INDEX IF NOT EXISTS idx_inbound_disposition_events_channel_ts
    ON inbound_disposition_events (channel_type, ts DESC);
CREATE INDEX IF NOT EXISTS idx_inbound_disposition_pending_wake
    ON inbound_disposition_events (ts, id)
    WHERE disposition = 'wake' AND wake_processed_at IS NULL;

-- Canonical identity aliases used by contact cadence and cross-channel STOP.
-- Known contacts converge on contact:<id>; otherwise an endpoint remains
-- channel-scoped until the operator links it through the contacts table.
CREATE TABLE IF NOT EXISTS outbound_contact_endpoints (
    channel TEXT NOT NULL,
    address TEXT NOT NULL,
    entity TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    -- Logical reference: contacts is loaded later in the baseline sequence.
    -- Resolution verifies it before writing; a FK here would make db/22
    -- depend on db/30 during clean bootstrap.
    contact_id BIGINT,
    is_primary BOOLEAN NOT NULL DEFAULT FALSE,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (channel, address)
);
CREATE INDEX IF NOT EXISTS idx_outbound_contact_endpoints_entity
    ON outbound_contact_endpoints (entity);

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

-- Channel deliveries: log of outbox-initiated (proactive) messages
CREATE TABLE IF NOT EXISTS channel_deliveries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    outbox_message_id TEXT,                -- RabbitMQ message ID
    channel_type TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    sender_id TEXT,                        -- target sender (if known)
    content TEXT NOT NULL,
    delivery_mode TEXT NOT NULL,           -- 'direct', 'last_active', 'broadcast'
    success BOOLEAN NOT NULL,
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_channel_deliveries_created
    ON channel_deliveries(created_at DESC);

CREATE TABLE IF NOT EXISTS channel_unreachable_targets (
    channel_type TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    error_kind TEXT NOT NULL DEFAULT 'unreachable',
    failure_count INTEGER NOT NULL DEFAULT 1,
    marked_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    suppress_until TIMESTAMPTZ NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (channel_type, channel_id)
);

CREATE INDEX IF NOT EXISTS idx_channel_unreachable_targets_suppress_until
    ON channel_unreachable_targets(suppress_until);

CREATE TABLE IF NOT EXISTS channel_delivery_obligations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    obligation_key TEXT NOT NULL UNIQUE,
    source_outbox_message_id TEXT,
    channel_type TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    sender_id TEXT,
    thread_id TEXT,
    content TEXT NOT NULL,
    message JSONB NOT NULL DEFAULT '{}'::jsonb,
    delivery_mode TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending'
        CHECK (state IN ('pending', 'attempting', 'delivered', 'failed', 'abandoned')),
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    attempting_at TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_channel_delivery_obligations_recoverable
    ON channel_delivery_obligations (state, next_attempt_at, updated_at)
    WHERE state IN ('pending', 'attempting', 'failed');

CREATE TABLE IF NOT EXISTS channel_presence_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel_type TEXT NOT NULL,
    channel_id TEXT,
    presence_kind TEXT NOT NULL
        CHECK (presence_kind IN ('online', 'offline', 'typing', 'processing', 'idle')),
    direction TEXT NOT NULL DEFAULT 'system'
        CHECK (direction IN ('system', 'inbound', 'outbound')),
    sender_id TEXT,
    session_key TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_channel_presence_events_recent
    ON channel_presence_events (channel_type, channel_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_channel_presence_events_live
    ON channel_presence_events (expires_at)
    WHERE expires_at IS NOT NULL;
