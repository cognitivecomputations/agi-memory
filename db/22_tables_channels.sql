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
