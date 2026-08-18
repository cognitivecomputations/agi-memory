-- =============================================================
-- Sub-Agent Sessions
-- =============================================================
-- Tracks background tasks spawned by the agent during heartbeat
-- or chat. Each session runs its own AgentLoop with an energy
-- budget and stores results back here.
-- =============================================================

-- Each baseline file runs as its own psql session at initdb, so it must pin
-- its own search_path: under an ag_catalog-first cluster default, unqualified
-- CREATEs would land Hexis objects in ag_catalog (the #77 fossil bug).
SET search_path = public, ag_catalog, "$user";

CREATE TABLE IF NOT EXISTS sub_agent_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task TEXT NOT NULL,                          -- What the sub-agent should do
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    energy_budget INT NOT NULL DEFAULT 5,
    energy_spent INT NOT NULL DEFAULT 0,
    result TEXT,                                 -- Summary of what was accomplished
    error TEXT,                                  -- Error message if failed
    parent_heartbeat_id UUID,                    -- Heartbeat that spawned this (if any)
    parent_session_id TEXT,                      -- Chat session that spawned this (if any)
    tool_context TEXT NOT NULL DEFAULT 'heartbeat',  -- 'heartbeat' or 'chat'
    transcript JSONB DEFAULT '[]'::jsonb,        -- Messages exchanged during execution
    notify_on_complete BOOLEAN DEFAULT true,
    memory_id UUID,                              -- Episodic memory created from result
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_sub_agent_sessions_status
    ON sub_agent_sessions (status)
    WHERE status IN ('pending', 'running');

CREATE INDEX IF NOT EXISTS idx_sub_agent_sessions_created
    ON sub_agent_sessions (created_at DESC);
