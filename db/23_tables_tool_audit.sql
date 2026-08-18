-- ============================================================================
-- Tool Audit Trail Tables
--
-- Persistent execution log for all tool calls across heartbeat, chat, and MCP
-- contexts. Workflow execution tracking for multi-step tool orchestration.
-- ============================================================================

-- Tool executions: audit log for every tool call
-- Each baseline file runs as its own psql session at initdb, so it must pin
-- its own search_path: under an ag_catalog-first cluster default, unqualified
-- CREATEs would land Hexis objects in ag_catalog (the #77 fossil bug).
SET search_path = public, ag_catalog, "$user";

CREATE TABLE IF NOT EXISTS tool_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tool_name TEXT NOT NULL,
    arguments JSONB DEFAULT '{}'::jsonb,
    tool_context TEXT NOT NULL,              -- 'heartbeat', 'chat', 'mcp'
    call_id TEXT NOT NULL,
    session_id TEXT,
    success BOOLEAN NOT NULL,
    output JSONB,                            -- truncated to ~10KB
    error TEXT,
    error_type TEXT,
    energy_spent INTEGER DEFAULT 0,
    duration_seconds FLOAT DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tool_exec_name
    ON tool_executions(tool_name);

CREATE INDEX IF NOT EXISTS idx_tool_exec_ctx
    ON tool_executions(tool_context);

CREATE INDEX IF NOT EXISTS idx_tool_exec_session
    ON tool_executions(session_id);

CREATE INDEX IF NOT EXISTS idx_tool_exec_created
    ON tool_executions(created_at DESC);

-- Workflow executions: tracks multi-step workflow plans
CREATE TABLE IF NOT EXISTS workflow_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    plan JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',   -- running, completed, failed, cancelled
    step_results JSONB DEFAULT '[]'::jsonb,
    total_energy_spent INTEGER DEFAULT 0,
    session_id TEXT,
    error TEXT,
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_workflow_exec_status
    ON workflow_executions(status);

CREATE INDEX IF NOT EXISTS idx_workflow_exec_created
    ON workflow_executions(started_at DESC);
