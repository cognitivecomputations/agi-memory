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
