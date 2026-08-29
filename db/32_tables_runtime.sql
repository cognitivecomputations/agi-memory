-- Hexis DB-owned runtime tables.
SET search_path = public, ag_catalog, "$user";

CREATE TABLE IF NOT EXISTS prompt_modules (
    key TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    description TEXT,
    source_path TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS llm_task_kinds (
    task_kind TEXT PRIMARY KEY,
    provider_config_key TEXT NOT NULL,
    prompt_module_keys JSONB NOT NULL DEFAULT '[]'::jsonb,
    response_schema JSONB NOT NULL DEFAULT '{}'::jsonb,
    defaults JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS external_driver_calls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    driver TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'in_progress', 'completed', 'failed', 'dropped')),
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    result JSONB,
    error TEXT,
    attempts INT NOT NULL DEFAULT 0,
    max_attempts INT NOT NULL DEFAULT 3,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    claimed_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_external_driver_calls_pending
    ON external_driver_calls (driver, next_attempt_at, created_at)
    WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS idx_external_driver_calls_in_progress
    ON external_driver_calls (claimed_at)
    WHERE status = 'in_progress';

CREATE TABLE IF NOT EXISTS tool_definitions (
    name TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    schema JSONB NOT NULL DEFAULT '{}'::jsonb,
    default_energy_cost INT NOT NULL DEFAULT 1,
    allowed_contexts TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    requires_approval BOOLEAN NOT NULL DEFAULT FALSE,
    supports_parallel BOOLEAN NOT NULL DEFAULT FALSE,
    execution_kind TEXT NOT NULL DEFAULT 'python_driver'
        CHECK (execution_kind IN ('db_function', 'python_driver', 'external_driver')),
    driver TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Live truth for the tool surface in each worker process. Unlike the catalog,
-- this records whether registration, configuration, and skill reachability agree.
CREATE TABLE IF NOT EXISTS worker_capabilities (
    worker_name TEXT NOT NULL,
    worker_id UUID,
    tool_name TEXT NOT NULL,
    tool_context TEXT NOT NULL CHECK (tool_context IN ('heartbeat', 'chat', 'mcp')),
    available BOOLEAN NOT NULL,
    reason_code TEXT,
    reason_if_missing TEXT,
    registry_kind TEXT NOT NULL DEFAULT 'default',
    last_checked_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (worker_name, tool_context, tool_name)
);

CREATE INDEX IF NOT EXISTS idx_worker_capabilities_checked
    ON worker_capabilities (last_checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_worker_capabilities_gaps
    ON worker_capabilities (worker_name, reason_code, last_checked_at DESC)
    WHERE available = FALSE;

CREATE TABLE IF NOT EXISTS tool_surface_decision_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID,
    surface TEXT NOT NULL DEFAULT 'chat',
    tool_context TEXT NOT NULL,
    decision_kind TEXT NOT NULL DEFAULT 'selection'
        CHECK (decision_kind IN ('selection', 'skill_activation')),
    input_text_hash TEXT NOT NULL,
    selected_skills TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    considered JSONB NOT NULL DEFAULT '[]'::jsonb,
    allowed_tools TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    reachable_tools TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    unreachable_tools TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    available_skill_count INT NOT NULL DEFAULT 0,
    registry_kind TEXT NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tool_surface_decisions_created
    ON tool_surface_decision_events (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tool_surface_decisions_session
    ON tool_surface_decision_events (session_id, created_at DESC)
    WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tool_surface_decisions_gaps
    ON tool_surface_decision_events (created_at DESC)
    WHERE cardinality(unreachable_tools) > 0;

-- The heartbeat economy is auditable state, not a timer-side calculation.
-- One singleton anchors time-proportional regeneration; each beat and each
-- useful outcome signal remains inspectable after scheduling decisions.
CREATE TABLE IF NOT EXISTS heartbeat_economy_state (
    id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    last_regenerated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO heartbeat_economy_state (id, last_regenerated_at)
SELECT 1, COALESCE(
    (SELECT NULLIF(value->>'last_heartbeat_at', '')::timestamptz
     FROM state WHERE key = 'heartbeat_state'),
    CURRENT_TIMESTAMP
)
ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS heartbeat_outcomes (
    heartbeat_id UUID PRIMARY KEY,
    heartbeat_number INT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'error', 'cancelled')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    stopped_reason TEXT,
    energy_before FLOAT NOT NULL DEFAULT 0,
    elapsed_regen_hours FLOAT NOT NULL DEFAULT 0,
    surplus_decayed FLOAT NOT NULL DEFAULT 0,
    energy_regenerated FLOAT NOT NULL DEFAULT 0,
    regen_multiplier FLOAT NOT NULL DEFAULT 1,
    energy_after_regen FLOAT NOT NULL DEFAULT 0,
    energy_spent FLOAT NOT NULL DEFAULT 0,
    durable_memories_created INT NOT NULL DEFAULT 0,
    contradictions_resolved INT NOT NULL DEFAULT 0,
    goals_advanced INT NOT NULL DEFAULT 0,
    proactive_contact BOOLEAN NOT NULL DEFAULT FALSE,
    user_feedback_score FLOAT NOT NULL DEFAULT 0,
    outcome_score FLOAT NOT NULL DEFAULT 0,
    outcome_tier TEXT NOT NULL DEFAULT 'none'
        CHECK (outcome_tier IN ('none', 'useful', 'high_value')),
    urgency_ratio FLOAT NOT NULL DEFAULT 0,
    cadence_minutes FLOAT,
    next_heartbeat_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_heartbeat_outcomes_completed
    ON heartbeat_outcomes (completed_at DESC)
    WHERE completed_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS heartbeat_outcome_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    heartbeat_id UUID NOT NULL REFERENCES heartbeat_outcomes(heartbeat_id) ON DELETE CASCADE,
    signal_kind TEXT NOT NULL CHECK (signal_kind IN (
        'durable_memory', 'contradiction_resolved', 'goal_advanced',
        'proactive_contact', 'user_feedback', 'tool_success', 'tool_failure'
    )),
    signal_key TEXT NOT NULL,
    amount FLOAT NOT NULL DEFAULT 1,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (heartbeat_id, signal_key)
);
CREATE INDEX IF NOT EXISTS idx_heartbeat_outcome_signals_kind
    ON heartbeat_outcome_signals (heartbeat_id, signal_kind, created_at);

-- Durable, advisory internal deliberation. These records preserve the
-- inspectable reasons, challenges, dissent, and invalidation conditions for a
-- council run. They never authorize or gate an action.
CREATE TABLE IF NOT EXISTS deliberation_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed')),
    topic TEXT NOT NULL CHECK (btrim(topic) <> ''),
    stakes TEXT NOT NULL DEFAULT 'material'
        CHECK (stakes IN ('routine', 'material', 'high')),
    source_context TEXT NOT NULL DEFAULT 'chat',
    source_session_id TEXT,
    heartbeat_id UUID,
    call_id TEXT,
    persona_keys JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(persona_keys) = 'array'),
    signal_count INT NOT NULL DEFAULT 0 CHECK (signal_count >= 0),
    input_context JSONB NOT NULL DEFAULT '{}'::jsonb,
    error TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_deliberation_sessions_recent
    ON deliberation_sessions (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_deliberation_sessions_status
    ON deliberation_sessions (status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_deliberation_sessions_heartbeat
    ON deliberation_sessions (heartbeat_id, started_at DESC)
    WHERE heartbeat_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS deliberation_moves (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES deliberation_sessions(id) ON DELETE CASCADE,
    move_key TEXT NOT NULL,
    round INT NOT NULL DEFAULT 1 CHECK (round >= 1),
    ordinal INT NOT NULL DEFAULT 0 CHECK (ordinal >= 0),
    role TEXT NOT NULL CHECK (role IN ('perspective', 'challenge', 'synthesis')),
    persona_key TEXT,
    content TEXT NOT NULL CHECK (btrim(content) <> ''),
    target_move_id UUID REFERENCES deliberation_moves(id) ON DELETE SET NULL,
    evidence_memory_ids UUID[] NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (session_id, move_key)
);

CREATE INDEX IF NOT EXISTS idx_deliberation_moves_session
    ON deliberation_moves (session_id, round, ordinal, created_at);

CREATE TABLE IF NOT EXISTS deliberation_verdicts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL UNIQUE
        REFERENCES deliberation_sessions(id) ON DELETE CASCADE,
    recommendation TEXT NOT NULL CHECK (btrim(recommendation) <> ''),
    report TEXT NOT NULL CHECK (btrim(report) <> ''),
    agreements JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(agreements) = 'array'),
    disagreements JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(disagreements) = 'array'),
    risks JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(risks) = 'array'),
    missing_evidence JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(missing_evidence) = 'array'),
    dissent JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(dissent) = 'array'),
    invalidation_conditions JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(invalidation_conditions) = 'array'),
    evidence_memory_ids UUID[] NOT NULL DEFAULT '{}',
    summary_memory_id UUID,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_deliberation_verdicts_recent
    ON deliberation_verdicts (created_at DESC);

CREATE TABLE IF NOT EXISTS agent_turns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mode TEXT NOT NULL,
    session_id UUID,
    heartbeat_id UUID,
    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'waiting_external', 'completed', 'failed', 'cancelled')),
    phase TEXT NOT NULL DEFAULT 'execute',
    user_message TEXT,
    messages JSONB NOT NULL DEFAULT '[]'::jsonb,
    runtime_state JSONB NOT NULL DEFAULT '{}'::jsonb,
    stopped_reason TEXT,
    result JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_agent_turns_status_created
    ON agent_turns (status, created_at DESC);

CREATE TABLE IF NOT EXISTS agent_turn_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    turn_id UUID NOT NULL REFERENCES agent_turns(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_agent_turn_events_turn_created
    ON agent_turn_events (turn_id, created_at);

-- DB-owned chat session history. This is the portable short-term
-- conversation substrate for app/API/TUI chat; UI-local history is rendering
-- state, not continuity.
CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    surface TEXT NOT NULL DEFAULT 'chat',
    external_id TEXT,
    title TEXT,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'archived')),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_active_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    cleared_at TIMESTAMPTZ
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_sessions_external
    ON chat_sessions (surface, external_id)
    WHERE external_id IS NOT NULL AND status = 'active';
CREATE INDEX IF NOT EXISTS idx_chat_sessions_active
    ON chat_sessions (surface, status, last_active_at DESC);

CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    ordinal INT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
    content TEXT NOT NULL,
    visible_in_context BOOLEAN NOT NULL DEFAULT TRUE,
    source_message_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (session_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_ordinal
    ON chat_messages (session_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_chat_messages_context
    ON chat_messages (session_id, visible_in_context, ordinal DESC);
CREATE INDEX IF NOT EXISTS idx_chat_messages_metadata
    ON chat_messages USING GIN (metadata);

CREATE TABLE IF NOT EXISTS workflow_step_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES workflow_executions(id) ON DELETE CASCADE,
    step_name TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    arguments JSONB NOT NULL DEFAULT '{}'::jsonb,
    depends_on TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'ready', 'in_progress', 'completed', 'failed', 'skipped')),
    output JSONB,
    error TEXT,
    attempts INT NOT NULL DEFAULT 0,
    max_attempts INT NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    UNIQUE (workflow_id, step_name)
);

CREATE INDEX IF NOT EXISTS idx_workflow_step_runs_status
    ON workflow_step_runs (workflow_id, status, created_at);

-- Change legibility (#93): the substrate-change journal the agent reads.
CREATE TABLE IF NOT EXISTS change_journal (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kind TEXT NOT NULL CHECK (kind IN ('migration', 'code', 'prompt_module', 'config_flip', 'self_extension')),
    summary TEXT NOT NULL,
    detail JSONB NOT NULL DEFAULT '{}'::jsonb,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_change_journal_occurred
    ON change_journal (occurred_at DESC);

-- Per-section ingestion receipts (#85/#90): completion, not intent.
CREATE TABLE IF NOT EXISTS ingestion_receipts (
    doc_ref TEXT NOT NULL,
    section_hash TEXT NOT NULL,
    memory_id UUID,
    memories_created INT NOT NULL DEFAULT 0,
    source_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (doc_ref, section_hash)
);

-- Durable ingestion jobs (#87): background ingestion survives restarts.
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- 'artifact' jobs carry no inline content: payload.artifact_id points at
    -- preserved original bytes in source_artifacts (uploads, binary files).
    kind TEXT NOT NULL CHECK (kind IN ('text', 'url', 'artifact')),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'in_progress', 'completed', 'failed', 'cancelled')),
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    content TEXT,
    content_hash TEXT,
    progress JSONB NOT NULL DEFAULT '{}'::jsonb,
    result JSONB,
    error TEXT,
    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
    attempts INT NOT NULL DEFAULT 0,
    max_attempts INT NOT NULL DEFAULT 3,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    claimed_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_pending
    ON ingestion_jobs (next_attempt_at, created_at) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_in_progress
    ON ingestion_jobs (claimed_at) WHERE status = 'in_progress';
CREATE UNIQUE INDEX IF NOT EXISTS idx_ingestion_jobs_active_hash
    ON ingestion_jobs (content_hash) WHERE status IN ('pending', 'in_progress');

-- Durable raw source artifacts: ingestion extracts memories, but the exact
-- source text stays available for deliberate document search/open later.
CREATE TABLE IF NOT EXISTS source_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_ingested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    path TEXT,
    file_type TEXT,
    content TEXT NOT NULL,
    word_count INT NOT NULL DEFAULT 0,
    size_bytes INT NOT NULL DEFAULT 0,
    source_attribution JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'redacted', 'archived'))
);

CREATE INDEX IF NOT EXISTS idx_source_documents_status_updated
    ON source_documents (status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_source_documents_path
    ON source_documents (path) WHERE path IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_source_documents_source_type
    ON source_documents (source_type);
CREATE INDEX IF NOT EXISTS idx_source_documents_content_fts
    ON source_documents USING GIN (to_tsvector('english', title || ' ' || COALESCE(path, '') || ' ' || content))
    WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_source_documents_source_attribution
    ON source_documents USING GIN (source_attribution);

-- Durable source-document chunks: stable, citable slices of a source
-- document with locators (page/section/sheet row/slide/message) and their
-- own embeddings for hybrid retrieval. Keyed by (document, chunk_index);
-- ids and embeddings survive re-ingestion when content is unchanged.
-- Privacy/status stay single-source on source_documents — every read joins.
CREATE TABLE IF NOT EXISTS source_document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_document_id UUID NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    locator_kind TEXT NOT NULL DEFAULT 'char'
        CHECK (locator_kind IN ('char', 'page', 'section', 'sheet_row', 'slide', 'message')),
    locator JSONB NOT NULL DEFAULT '{}'::jsonb,
    heading_path TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    token_count INT,
    char_start INT NOT NULL DEFAULT 0,
    char_end INT NOT NULL DEFAULT 0,
    page_start INT,
    page_end INT,
    sheet_name TEXT,
    row_start INT,
    row_end INT,
    column_start INT,
    column_end INT,
    embedding vector(768),
    embedded_at TIMESTAMPTZ,
    embedding_model TEXT,
    embedding_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (embedding_status IN ('pending', 'in_progress', 'embedded', 'failed', 'skipped')),
    embedding_claimed_at TIMESTAMPTZ,
    embedding_attempts INT NOT NULL DEFAULT 0,
    chunker_version TEXT NOT NULL DEFAULT 'v2',
    access_count INT NOT NULL DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (source_document_id, chunk_index)
);

-- Keep the chunk embedding column in step with the configured dimension
-- (mirrors the db/00 DO block; this file runs after embedding_dimension()
-- and sync_embedding_dimension_config() exist).
DO $$
DECLARE
    dim INT;
BEGIN
    dim := embedding_dimension();
    IF dim IS NOT NULL AND dim <> 768 THEN
        EXECUTE format(
            'ALTER TABLE source_document_chunks ALTER COLUMN embedding TYPE vector(%s) USING embedding::vector(%s)',
            dim,
            dim
        );
    END IF;
END;
$$;

CREATE INDEX IF NOT EXISTS idx_source_chunks_document
    ON source_document_chunks (source_document_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_source_chunks_fts
    ON source_document_chunks USING GIN (to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_source_chunks_embedding
    ON source_document_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_source_chunks_hash
    ON source_document_chunks (content_hash);
CREATE INDEX IF NOT EXISTS idx_source_chunks_embed_queue
    ON source_document_chunks (embedding_status, created_at)
    WHERE embedding_status IN ('pending', 'in_progress');

-- Original source artifacts: the exact bytes (or a stable reference) a
-- source document was extracted from, preserved BEFORE extraction so a
-- failed parse never loses the source and a better extractor can re-run
-- later. Bytes live in-DB up to ingest.artifact_max_db_bytes (rides
-- pg_dump backups); larger artifacts live in a content-addressed managed
-- directory with the hash recorded here.
CREATE TABLE IF NOT EXISTS source_artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_document_id UUID REFERENCES source_documents(id) ON DELETE SET NULL,
    storage_kind TEXT NOT NULL
        CHECK (storage_kind IN ('database', 'filesystem', 'connector', 'url', 'external')),
    storage_ref TEXT,
    bytes BYTEA,
    original_filename TEXT,
    mime_type TEXT,
    byte_size BIGINT NOT NULL DEFAULT 0,
    sha256 TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'redacted', 'archived')),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_source_artifacts_doc
    ON source_artifacts (source_document_id) WHERE source_document_id IS NOT NULL;

-- The artifact hash a document's normalized content was extracted from.
ALTER TABLE source_documents ADD COLUMN IF NOT EXISTS original_hash TEXT;

-- Extraction runs: which extractor produced a document's normalized text,
-- with structured warnings (OCR used, rows truncated, unsupported features)
-- and errors. Failed runs may carry an artifact but no document.
CREATE TABLE IF NOT EXISTS source_extraction_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_document_id UUID REFERENCES source_documents(id) ON DELETE CASCADE,
    artifact_id UUID REFERENCES source_artifacts(id) ON DELETE SET NULL,
    extractor_name TEXT NOT NULL,
    extractor_version TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL
        CHECK (status IN ('completed', 'completed_with_warnings', 'failed')),
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    errors JSONB NOT NULL DEFAULT '[]'::jsonb,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_source_extraction_runs_doc
    ON source_extraction_runs (source_document_id, created_at DESC)
    WHERE source_document_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_source_extraction_runs_artifact
    ON source_extraction_runs (artifact_id, created_at DESC)
    WHERE artifact_id IS NOT NULL;

-- Raw channel-message source artifacts. Channel adapters write
-- channel_messages; Postgres owns the exact source document, ingestion job
-- link, provenance, and sensitivity classification for every message.
CREATE TABLE IF NOT EXISTS channel_source_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel_message_id UUID NOT NULL REFERENCES channel_messages(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES channel_sessions(id) ON DELETE CASCADE,
    channel_type TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    sender_id TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('inbound', 'outbound')),
    platform_message_id TEXT,
    source_document_id UUID REFERENCES source_documents(id) ON DELETE SET NULL,
    ingestion_job_id UUID REFERENCES ingestion_jobs(id) ON DELETE SET NULL,
    content_hash TEXT NOT NULL,
    sensitivity TEXT NOT NULL DEFAULT 'private'
        CHECK (sensitivity IN ('private', 'shared', 'public')),
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'redacted', 'archived', 'error')),
    raw_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (channel_message_id)
);

CREATE INDEX IF NOT EXISTS idx_channel_source_items_session
    ON channel_source_items (session_id, direction, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_channel_source_items_channel
    ON channel_source_items (channel_type, channel_id, sender_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_channel_source_items_document
    ON channel_source_items (source_document_id) WHERE source_document_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_channel_source_items_metadata
    ON channel_source_items USING GIN (raw_metadata);

-- Channel adapter runtime visibility. Workers own the heartbeat writes;
-- Postgres owns the state surface consumed by chat/CLI/UI setup flows.
CREATE TABLE IF NOT EXISTS channel_adapter_runtime (
    channel_type TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'unknown'
        CHECK (status IN ('unknown', 'not_configured', 'configured', 'starting', 'running', 'stopped', 'error', 'missing_dependency')),
    configured BOOLEAN NOT NULL DEFAULT FALSE,
    running BOOLEAN NOT NULL DEFAULT FALSE,
    worker_id TEXT,
    pid INT,
    last_checked_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_started_at TIMESTAMPTZ,
    last_stopped_at TIMESTAMPTZ,
    last_error TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_channel_adapter_runtime_status
    ON channel_adapter_runtime (status, updated_at DESC);

-- First-class personal-data connector setup. Long-lived secrets live in
-- ~/.hexis/auth; the database owns connector identity, grants, setup state,
-- provenance, and revocation status.
CREATE TABLE IF NOT EXISTS integration_connectors (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    category TEXT NOT NULL,
    auth_type TEXT NOT NULL
        CHECK (auth_type IN ('oauth2', 'api_key', 'device_code', 'pairing', 'manual', 'local_export')),
    status TEXT NOT NULL DEFAULT 'available'
        CHECK (status IN ('available', 'planned', 'disabled')),
    capability_manifest JSONB NOT NULL DEFAULT '{}'::jsonb,
    setup_manifest JSONB NOT NULL DEFAULT '{}'::jsonb,
    docs_url TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_integration_connectors_status
    ON integration_connectors (status, category, id);

CREATE TABLE IF NOT EXISTS integration_connections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    connector_id TEXT NOT NULL REFERENCES integration_connectors(id) ON DELETE CASCADE,
    account_key TEXT NOT NULL,
    display_name TEXT,
    status TEXT NOT NULL DEFAULT 'connected'
        CHECK (status IN ('pending', 'connected', 'needs_reauth', 'revoked', 'error')),
    credential_ref TEXT,
    granted_scopes TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    capabilities JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_channel TEXT,
    source_session_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    last_error TEXT,
    connected_at TIMESTAMPTZ,
    last_verified_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (connector_id, account_key)
);

CREATE INDEX IF NOT EXISTS idx_integration_connections_status
    ON integration_connections (connector_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_integration_connections_metadata
    ON integration_connections USING GIN (metadata);

CREATE TABLE IF NOT EXISTS connection_attempts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    connector_id TEXT NOT NULL REFERENCES integration_connectors(id) ON DELETE CASCADE,
    account_key TEXT,
    status TEXT NOT NULL DEFAULT 'pending_user'
        CHECK (status IN ('pending_user', 'awaiting_input', 'exchanging', 'complete', 'error', 'expired', 'cancelled')),
    requested_capabilities JSONB NOT NULL DEFAULT '[]'::jsonb,
    requested_scopes TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    flow_state JSONB NOT NULL DEFAULT '{}'::jsonb,
    authorization_url TEXT,
    user_next_step TEXT,
    source_channel TEXT,
    source_session_id TEXT,
    credential_ref TEXT,
    error TEXT,
    expires_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_connection_attempts_status
    ON connection_attempts (connector_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_connection_attempts_session
    ON connection_attempts (source_channel, source_session_id, created_at DESC);

-- DB-owned connector backfill substrate. Provider adapters fetch pages and
-- bodies; Postgres owns cursor state, retry/pause lifecycle, provider-item
-- receipts, and the link from raw channel items to source_documents.
CREATE TABLE IF NOT EXISTS connector_sync_cursors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    connection_id UUID NOT NULL REFERENCES integration_connections(id) ON DELETE CASCADE,
    connector_id TEXT NOT NULL,
    account_key TEXT NOT NULL,
    cursor_key TEXT NOT NULL DEFAULT 'default',
    cursor_value JSONB NOT NULL DEFAULT '{}'::jsonb,
    high_watermark TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'paused', 'error')),
    last_started_at TIMESTAMPTZ,
    last_completed_at TIMESTAMPTZ,
    last_error TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (connection_id, cursor_key)
);

CREATE INDEX IF NOT EXISTS idx_connector_sync_cursors_status
    ON connector_sync_cursors (connector_id, account_key, status, updated_at DESC);

CREATE TABLE IF NOT EXISTS connector_backfill_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    connection_id UUID NOT NULL REFERENCES integration_connections(id) ON DELETE CASCADE,
    connector_id TEXT NOT NULL,
    account_key TEXT NOT NULL,
    cursor_key TEXT NOT NULL DEFAULT 'default',
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'in_progress', 'paused', 'completed', 'failed', 'cancelled')),
    requested_range JSONB NOT NULL DEFAULT '{}'::jsonb,
    progress JSONB NOT NULL DEFAULT '{}'::jsonb,
    result JSONB,
    error TEXT,
    attempts INT NOT NULL DEFAULT 0,
    max_attempts INT NOT NULL DEFAULT 3,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    claimed_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
    pause_requested BOOLEAN NOT NULL DEFAULT FALSE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_connector_backfill_jobs_active
    ON connector_backfill_jobs (connection_id, cursor_key)
    WHERE status IN ('pending', 'in_progress', 'paused');
CREATE INDEX IF NOT EXISTS idx_connector_backfill_jobs_pending
    ON connector_backfill_jobs (status, next_attempt_at, created_at)
    WHERE status IN ('pending', 'in_progress');

CREATE TABLE IF NOT EXISTS connector_source_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    connection_id UUID NOT NULL REFERENCES integration_connections(id) ON DELETE CASCADE,
    connector_id TEXT NOT NULL,
    account_key TEXT NOT NULL,
    provider_item_id TEXT NOT NULL,
    provider_thread_id TEXT,
    item_kind TEXT NOT NULL DEFAULT 'message',
    source_document_id UUID REFERENCES source_documents(id) ON DELETE SET NULL,
    content_hash TEXT NOT NULL,
    item_timestamp TIMESTAMPTZ,
    labels TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    participants JSONB NOT NULL DEFAULT '[]'::jsonb,
    attachments JSONB NOT NULL DEFAULT '[]'::jsonb,
    ingestion_job_id UUID REFERENCES ingestion_jobs(id) ON DELETE SET NULL,
    sensitivity TEXT NOT NULL DEFAULT 'private'
        CHECK (sensitivity IN ('private', 'shared', 'public')),
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'redacted', 'archived')),
    raw_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (connection_id, provider_item_id)
);

CREATE INDEX IF NOT EXISTS idx_connector_source_items_provider
    ON connector_source_items (connector_id, account_key, item_kind, provider_item_id);
CREATE INDEX IF NOT EXISTS idx_connector_source_items_time
    ON connector_source_items (connector_id, account_key, item_timestamp DESC)
    WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_connector_source_items_metadata
    ON connector_source_items USING GIN (raw_metadata);

-- DB-owned connector action authorization. Provider adapters execute effects;
-- Postgres owns durable grants, constraints, decisions, and audit.
CREATE TABLE IF NOT EXISTS connector_action_tool_map (
    tool_name TEXT PRIMARY KEY,
    connector_id TEXT NOT NULL,
    action_kind TEXT NOT NULL,
    target_argument TEXT,
    account_argument TEXT,
    sensitivity TEXT NOT NULL DEFAULT 'external_action',
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_connector_action_tool_map_connector
    ON connector_action_tool_map (connector_id, action_kind)
    WHERE enabled;

CREATE TABLE IF NOT EXISTS connector_action_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    connector_id TEXT NOT NULL,
    account_key TEXT,
    action_kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'revoked', 'expired')),
    contexts TEXT[] NOT NULL DEFAULT ARRAY['chat']::TEXT[],
    allow_autonomous BOOLEAN NOT NULL DEFAULT FALSE,
    requires_per_action_approval BOOLEAN NOT NULL DEFAULT TRUE,
    constraints JSONB NOT NULL DEFAULT '{}'::jsonb,
    granted_by TEXT NOT NULL DEFAULT 'user',
    source_session_id TEXT,
    rationale TEXT,
    expires_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    revoke_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_connector_action_policies_active
    ON connector_action_policies (connector_id, action_kind, account_key, updated_at DESC)
    WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_connector_action_policies_constraints
    ON connector_action_policies USING GIN (constraints);

CREATE TABLE IF NOT EXISTS connector_action_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_id UUID REFERENCES connector_action_policies(id) ON DELETE SET NULL,
    tool_execution_id UUID REFERENCES tool_executions(id) ON DELETE SET NULL,
    connector_id TEXT NOT NULL,
    account_key TEXT,
    action_kind TEXT NOT NULL,
    target TEXT,
    tool_name TEXT NOT NULL,
    tool_context TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('allowed', 'denied', 'failed', 'pending')),
    reason TEXT,
    arguments JSONB NOT NULL DEFAULT '{}'::jsonb,
    context JSONB NOT NULL DEFAULT '{}'::jsonb,
    external_receipt JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_connector_action_audit_policy
    ON connector_action_audit (policy_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_connector_action_audit_connector
    ON connector_action_audit (connector_id, account_key, action_kind, created_at DESC);

-- Exact, one-shot approvals for protected tool calls. Arguments themselves are
-- never stored: only their canonical JSON hash and a redacted human preview.
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

-- Consent-first automation proposals. A proposal is inert until the user
-- accepts it; dedup_key is deliberately permanent so "Not for me" latches.
CREATE TABLE IF NOT EXISTS automation_suggestions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source TEXT NOT NULL
        CHECK (source IN ('catalog', 'blueprint', 'usage', 'connector')),
    dedup_key TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    rationale TEXT NOT NULL,
    task_spec JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'accepted', 'dismissed')),
    scheduled_task_id UUID REFERENCES scheduled_tasks(id) ON DELETE SET NULL,
    outbox_message_id UUID,
    decision_channel TEXT,
    decision_actor TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    decided_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_automation_suggestions_status_created
    ON automation_suggestions (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_automation_suggestions_task
    ON automation_suggestions (scheduled_task_id)
    WHERE scheduled_task_id IS NOT NULL;

-- Curated proposals are data, not branches in application code. Preconditions
-- are evaluated against the live connector registry before a suggestion is
-- filed; catalog rows never create schedules themselves.
CREATE TABLE IF NOT EXISTS automation_suggestion_catalog (
    dedup_key TEXT PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'catalog'
        CHECK (source IN ('catalog', 'connector')),
    title TEXT NOT NULL,
    rationale TEXT NOT NULL,
    task_spec JSONB NOT NULL,
    precondition TEXT NOT NULL DEFAULT 'none'
        CHECK (precondition IN ('none', 'gmail_connected', 'calendar_connected')),
    sort_order INTEGER NOT NULL DEFAULT 100,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_automation_suggestion_catalog_enabled
    ON automation_suggestion_catalog (enabled, sort_order, dedup_key);

-- A clarification is durable because the turn that asks it may outlive its
-- transport connection. Interactive surfaces wait on the same row that
-- heartbeat/outbox answers resume on a later beat.
CREATE TABLE IF NOT EXISTS agent_questions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID,
    heartbeat_id UUID,
    surface TEXT NOT NULL,
    prompt TEXT NOT NULL,
    choices JSONB NOT NULL DEFAULT '[]'::jsonb,
    allow_free_text BOOLEAN NOT NULL DEFAULT TRUE,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'answered', 'timed_out', 'superseded')),
    answer TEXT,
    answer_choice_index INTEGER,
    answer_channel TEXT,
    answer_actor TEXT,
    asked_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ,
    answered_at TIMESTAMPTZ,
    resumed_at TIMESTAMPTZ,
    outbox_message_id UUID,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_agent_questions_pending_session
    ON agent_questions (session_id, asked_at DESC)
    WHERE status = 'pending' AND session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_agent_questions_answered_resume
    ON agent_questions (answered_at, asked_at)
    WHERE status = 'answered' AND resumed_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_agent_questions_heartbeat
    ON agent_questions (heartbeat_id, asked_at DESC)
    WHERE heartbeat_id IS NOT NULL;

-- Metadata-only, append-only record of inbound voice transcription attempts.
-- Transcript content deliberately remains in the conversation path, not here.
CREATE TABLE IF NOT EXISTS voice_note_stt_events (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    channel_type TEXT NOT NULL,
    channel_id TEXT,
    sender_id TEXT,
    message_id TEXT,
    attachment_id TEXT,
    mime_type TEXT,
    filename TEXT,
    provider TEXT NOT NULL,
    model TEXT,
    outcome TEXT NOT NULL CHECK (
        outcome = 'transcribed'
        OR outcome LIKE 'skipped\_%' ESCAPE '\'
        OR outcome LIKE 'failed\_%' ESCAPE '\'
    ),
    transcript_chars INTEGER CHECK (transcript_chars IS NULL OR transcript_chars >= 0),
    error_detail TEXT,
    duration_ms INTEGER CHECK (duration_ms IS NULL OR duration_ms >= 0),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_voice_note_stt_events_created
    ON voice_note_stt_events (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_voice_note_stt_events_channel
    ON voice_note_stt_events (channel_type, created_at DESC);

-- Browser Web Push endpoints are explicit per-device grants. Revocation is
-- soft so a browser can re-enable the same endpoint without duplicating it.
CREATE TABLE IF NOT EXISTS web_push_subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    endpoint TEXT NOT NULL UNIQUE,
    p256dh TEXT NOT NULL,
    auth TEXT NOT NULL,
    expiration_time BIGINT,
    user_agent TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    failure_count INTEGER NOT NULL DEFAULT 0 CHECK (failure_count >= 0),
    last_error TEXT,
    last_delivered_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_web_push_subscriptions_active
    ON web_push_subscriptions (updated_at DESC)
    WHERE revoked_at IS NULL;

-- Headless companion nodes connect outward to the API. Their Ed25519 public
-- key is the durable identity; approval is explicit and revocation is kept as
-- evidence rather than deleting the device.
CREATE TABLE IF NOT EXISTS hexis_nodes (
    node_id TEXT PRIMARY KEY,
    public_key TEXT NOT NULL,
    name TEXT NOT NULL,
    capabilities JSONB NOT NULL DEFAULT '[]'::jsonb,
    status TEXT NOT NULL DEFAULT 'offline'
        CHECK (status IN ('offline', 'online', 'revoked')),
    approved_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    approved_by TEXT NOT NULL DEFAULT 'operator',
    revoked_at TIMESTAMPTZ,
    last_seen_at TIMESTAMPTZ,
    connection_id UUID,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS node_pairing_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code TEXT NOT NULL UNIQUE,
    node_id TEXT NOT NULL,
    public_key TEXT NOT NULL,
    name TEXT NOT NULL,
    capabilities JSONB NOT NULL DEFAULT '[]'::jsonb,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'approved', 'denied', 'expired')),
    requested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP + INTERVAL '1 day',
    decided_at TIMESTAMPTZ,
    decided_by TEXT,
    decision_note TEXT,
    outbox_message_id UUID,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_node_pairing_one_pending
    ON node_pairing_requests (node_id) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_node_pairing_status
    ON node_pairing_requests (status, requested_at DESC);

CREATE TABLE IF NOT EXISTS node_invocations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id TEXT NOT NULL REFERENCES hexis_nodes(node_id),
    action TEXT NOT NULL CHECK (action IN (
        'system.run', 'screen.capture',
        'apple.reminders.list', 'apple.reminders.create',
        'apple.notes.search', 'apple.notes.create',
        'apple.calendar.list', 'apple.calendar.create',
        'apple.shortcuts.list', 'apple.shortcuts.run',
        'onepassword.items', 'onepassword.copy'
    )),
    arguments JSONB NOT NULL DEFAULT '{}'::jsonb,
    requested_by TEXT NOT NULL DEFAULT 'agent',
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'dispatched', 'succeeded', 'failed', 'expired', 'cancelled')),
    result JSONB,
    error TEXT,
    result_signature TEXT,
    requested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    dispatched_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_node_invocations_queued
    ON node_invocations (node_id, requested_at)
    WHERE status = 'queued';
CREATE INDEX IF NOT EXISTS idx_node_invocations_recent
    ON node_invocations (requested_at DESC);
