-- Hexis schema: tables, extensions, base types, and seed data.
-- ============================================================================
-- HEXIS MEMORY SYSTEM - FINAL SCHEMA
-- ============================================================================
-- EXTENSIONS
-- ============================================================================
-- Pin the path before anything is created: each baseline file runs as its own
-- psql session at initdb, and under an ag_catalog-first cluster default the
-- extensions below would otherwise install into ag_catalog (#77).
SET search_path = public, ag_catalog, "$user";
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS http;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================================
-- GRAPH INITIALIZATION
-- ============================================================================

LOAD 'age';
SET search_path = ag_catalog, "$user", public;

DO $$
DECLARE
    idx_sql TEXT;
    idx_statements TEXT[] := ARRAY[
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_memorynode_id ON memory_graph."MemoryNode" USING BTREE (id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_memorynode_memory_id ON memory_graph."MemoryNode" USING BTREE (ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '"memory_id"'::ag_catalog.agtype]))$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_memorynode_type ON memory_graph."MemoryNode" USING BTREE (ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '"type"'::ag_catalog.agtype]))$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_conceptnode_id ON memory_graph."ConceptNode" USING BTREE (id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_conceptnode_name ON memory_graph."ConceptNode" USING BTREE (ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '"name"'::ag_catalog.agtype]))$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_selfnode_id ON memory_graph."SelfNode" USING BTREE (id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_selfnode_key ON memory_graph."SelfNode" USING BTREE (ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '"key"'::ag_catalog.agtype]))$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_lifechapternode_id ON memory_graph."LifeChapterNode" USING BTREE (id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_lifechapternode_key ON memory_graph."LifeChapterNode" USING BTREE (ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '"key"'::ag_catalog.agtype]))$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_goalsroot_id ON memory_graph."GoalsRoot" USING BTREE (id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_goalsroot_key ON memory_graph."GoalsRoot" USING BTREE (ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '"key"'::ag_catalog.agtype]))$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_goalnode_id ON memory_graph."GoalNode" USING BTREE (id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_goalnode_goal_id ON memory_graph."GoalNode" USING BTREE (ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '"goal_id"'::ag_catalog.agtype]))$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_clusternode_id ON memory_graph."ClusterNode" USING BTREE (id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_clusternode_cluster_id ON memory_graph."ClusterNode" USING BTREE (ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '"cluster_id"'::ag_catalog.agtype]))$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_episodenode_id ON memory_graph."EpisodeNode" USING BTREE (id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_episodenode_episode_id ON memory_graph."EpisodeNode" USING BTREE (ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '"episode_id"'::ag_catalog.agtype]))$idx$,
        -- GIN on properties: AGE compiles an inline anchor MATCH (n:Label {key: $v})
        -- to a `properties @> {...}` containment op, which the BTREE expression
        -- indexes above do NOT serve (verified via EXPLAIN: they only fire for the
        -- `WHERE n.key = $v` form). Since Hexis anchors almost exclusively with
        -- inline maps, these GIN indexes are what actually turn those lookups from
        -- Seq Scan into Bitmap Index Scan. See docs/optimize.md §8.
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_memorynode_props_gin ON memory_graph."MemoryNode" USING GIN (properties)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_conceptnode_props_gin ON memory_graph."ConceptNode" USING GIN (properties)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_clusternode_props_gin ON memory_graph."ClusterNode" USING GIN (properties)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_episodenode_props_gin ON memory_graph."EpisodeNode" USING GIN (properties)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_goalnode_props_gin ON memory_graph."GoalNode" USING GIN (properties)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_selfnode_props_gin ON memory_graph."SelfNode" USING GIN (properties)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_goalsroot_props_gin ON memory_graph."GoalsRoot" USING GIN (properties)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_lifechapternode_props_gin ON memory_graph."LifeChapterNode" USING GIN (properties)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_in_episode_start ON memory_graph."IN_EPISODE" USING BTREE (start_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_in_episode_end ON memory_graph."IN_EPISODE" USING BTREE (end_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_contradicts_start ON memory_graph."CONTRADICTS" USING BTREE (start_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_contradicts_end ON memory_graph."CONTRADICTS" USING BTREE (end_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_associated_start ON memory_graph."ASSOCIATED" USING BTREE (start_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_associated_end ON memory_graph."ASSOCIATED" USING BTREE (end_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_has_belief_start ON memory_graph."HAS_BELIEF" USING BTREE (start_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_has_belief_end ON memory_graph."HAS_BELIEF" USING BTREE (end_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_supports_start ON memory_graph."SUPPORTS" USING BTREE (start_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_supports_end ON memory_graph."SUPPORTS" USING BTREE (end_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_instance_of_start ON memory_graph."INSTANCE_OF" USING BTREE (start_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_instance_of_end ON memory_graph."INSTANCE_OF" USING BTREE (end_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_parent_of_start ON memory_graph."PARENT_OF" USING BTREE (start_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_parent_of_end ON memory_graph."PARENT_OF" USING BTREE (end_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_member_of_start ON memory_graph."MEMBER_OF" USING BTREE (start_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_member_of_end ON memory_graph."MEMBER_OF" USING BTREE (end_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_cluster_relates_start ON memory_graph."CLUSTER_RELATES" USING BTREE (start_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_cluster_relates_end ON memory_graph."CLUSTER_RELATES" USING BTREE (end_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_cluster_overlaps_start ON memory_graph."CLUSTER_OVERLAPS" USING BTREE (start_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_cluster_overlaps_end ON memory_graph."CLUSTER_OVERLAPS" USING BTREE (end_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_cluster_similar_start ON memory_graph."CLUSTER_SIMILAR" USING BTREE (start_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_cluster_similar_end ON memory_graph."CLUSTER_SIMILAR" USING BTREE (end_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_subgoal_of_start ON memory_graph."SUBGOAL_OF" USING BTREE (start_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_subgoal_of_end ON memory_graph."SUBGOAL_OF" USING BTREE (end_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_originated_from_start ON memory_graph."ORIGINATED_FROM" USING BTREE (start_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_originated_from_end ON memory_graph."ORIGINATED_FROM" USING BTREE (end_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_blocks_start ON memory_graph."BLOCKS" USING BTREE (start_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_blocks_end ON memory_graph."BLOCKS" USING BTREE (end_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_evidence_for_start ON memory_graph."EVIDENCE_FOR" USING BTREE (start_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_evidence_for_end ON memory_graph."EVIDENCE_FOR" USING BTREE (end_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_episode_follows_start ON memory_graph."EPISODE_FOLLOWS" USING BTREE (start_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_episode_follows_end ON memory_graph."EPISODE_FOLLOWS" USING BTREE (end_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_contested_because_start ON memory_graph."CONTESTED_BECAUSE" USING BTREE (start_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_contested_because_end ON memory_graph."CONTESTED_BECAUSE" USING BTREE (end_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_causes_start ON memory_graph."CAUSES" USING BTREE (start_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_causes_end ON memory_graph."CAUSES" USING BTREE (end_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_derived_from_start ON memory_graph."DERIVED_FROM" USING BTREE (start_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_derived_from_end ON memory_graph."DERIVED_FROM" USING BTREE (end_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_temporal_next_start ON memory_graph."TEMPORAL_NEXT" USING BTREE (start_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_temporal_next_end ON memory_graph."TEMPORAL_NEXT" USING BTREE (end_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_contains_start ON memory_graph."CONTAINS" USING BTREE (start_id)$idx$,
        $idx$CREATE INDEX IF NOT EXISTS idx_memory_graph_contains_end ON memory_graph."CONTAINS" USING BTREE (end_id)$idx$
    ];
BEGIN
    BEGIN PERFORM create_graph('memory_graph'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_vlabel('memory_graph', 'MemoryNode'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_vlabel('memory_graph', 'ConceptNode'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_vlabel('memory_graph', 'SelfNode'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_vlabel('memory_graph', 'LifeChapterNode'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_vlabel('memory_graph', 'TurningPointNode'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_vlabel('memory_graph', 'NarrativeThreadNode'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_vlabel('memory_graph', 'RelationshipNode'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_vlabel('memory_graph', 'ValueConflictNode'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_vlabel('memory_graph', 'GoalNode'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_vlabel('memory_graph', 'GoalsRoot'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_vlabel('memory_graph', 'ClusterNode'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_vlabel('memory_graph', 'EpisodeNode'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_elabel('memory_graph', 'IN_EPISODE'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_elabel('memory_graph', 'CONTRADICTS'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_elabel('memory_graph', 'ASSOCIATED'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_elabel('memory_graph', 'HAS_BELIEF'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_elabel('memory_graph', 'SUPPORTS'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_elabel('memory_graph', 'INSTANCE_OF'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_elabel('memory_graph', 'PARENT_OF'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_elabel('memory_graph', 'MEMBER_OF'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_elabel('memory_graph', 'CLUSTER_RELATES'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_elabel('memory_graph', 'CLUSTER_OVERLAPS'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_elabel('memory_graph', 'CLUSTER_SIMILAR'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_elabel('memory_graph', 'SUBGOAL_OF'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_elabel('memory_graph', 'ORIGINATED_FROM'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_elabel('memory_graph', 'BLOCKS'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_elabel('memory_graph', 'EVIDENCE_FOR'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_elabel('memory_graph', 'EPISODE_FOLLOWS'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_elabel('memory_graph', 'CONTESTED_BECAUSE'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_elabel('memory_graph', 'CAUSES'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_elabel('memory_graph', 'DERIVED_FROM'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_elabel('memory_graph', 'TEMPORAL_NEXT'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_elabel('memory_graph', 'CONTAINS'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    BEGIN PERFORM create_elabel('memory_graph', 'SUPERSEDES'); EXCEPTION WHEN duplicate_object THEN NULL; END;
    FOREACH idx_sql IN ARRAY idx_statements LOOP
        BEGIN
            EXECUTE idx_sql;
        EXCEPTION WHEN undefined_table THEN NULL;
        END;
    END LOOP;
END;
$$;

SET search_path = public, ag_catalog, "$user";
-- ============================================================================
-- ENUMS
-- ============================================================================
DO $$
BEGIN
    BEGIN
        CREATE TYPE memory_type AS ENUM ('episodic', 'semantic', 'procedural', 'strategic', 'worldview', 'goal');
    EXCEPTION WHEN duplicate_object THEN NULL;
    END;
    BEGIN
        CREATE TYPE memory_status AS ENUM ('active', 'archived', 'invalidated', 'staged');
    EXCEPTION WHEN duplicate_object THEN NULL;
    END;
    -- Declared with the storage types because memories.goal_origin uses it.
    -- db/07 retains the guarded declaration for upgrade compatibility.
    BEGIN
        CREATE TYPE goal_source AS ENUM (
            'curiosity',
            'user_request',
            'identity',
            'derived',
            'external'
        );
    EXCEPTION WHEN duplicate_object THEN NULL;
    END;
    BEGIN
        CREATE TYPE cluster_type AS ENUM ('theme', 'emotion', 'temporal', 'person', 'pattern', 'mixed');
    EXCEPTION WHEN duplicate_object THEN NULL;
    END;
    BEGIN
        CREATE TYPE graph_edge_type AS ENUM (
            'TEMPORAL_NEXT',
            'CAUSES',
            'DERIVED_FROM',
            'CONTRADICTS',
            'SUPPORTS',
            'INSTANCE_OF',
            'PARENT_OF',
            'ASSOCIATED',
            'ORIGINATED_FROM',
            'BLOCKS',
            'EVIDENCE_FOR',
            'SUBGOAL_OF',
            'CLUSTER_RELATES',
            'CLUSTER_OVERLAPS',
            'CLUSTER_SIMILAR',
            'IN_EPISODE',
            'EPISODE_FOLLOWS',
            'CONTESTED_BECAUSE',
            'CONTAINS',
            'HAS_BELIEF',
            'MEMBER_OF',
            'SUPERSEDES'
        );
    EXCEPTION WHEN duplicate_object THEN NULL;
    END;
END;
$$;
-- ============================================================================
-- CORE STORAGE
-- ============================================================================
CREATE TABLE memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    type memory_type NOT NULL,
    -- Trusted authority boundary for goal provenance. Unlike metadata.source,
    -- this typed field is supplied by the execution context, not the model.
    goal_origin goal_source,
    status memory_status DEFAULT 'active',
    content TEXT NOT NULL,
    embedding vector(768),
    embedded_at TIMESTAMPTZ,
    embedding_model TEXT,
    embedding_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (embedding_status IN ('pending', 'in_progress', 'embedded', 'failed', 'skipped')),
    CONSTRAINT memories_embedded_vector_valid CHECK (
        embedding_status <> 'embedded'
        OR (embedding IS NOT NULL AND vector_norm(embedding) > 0)
    ),
    embedding_claimed_at TIMESTAMPTZ,
    embedding_attempts INT NOT NULL DEFAULT 0,
    valid_from TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    valid_until TIMESTAMPTZ,
    superseded_by UUID REFERENCES memories(id) ON DELETE SET NULL,
    CONSTRAINT memories_valid_time_order CHECK (
        valid_until IS NULL OR valid_until >= valid_from
    ),
    CONSTRAINT memories_no_self_supersession CHECK (
        superseded_by IS NULL OR superseded_by <> id
    ),
    importance FLOAT DEFAULT 0.5 CONSTRAINT memories_importance_range CHECK (importance BETWEEN 0 AND 1),
    source_attribution JSONB NOT NULL DEFAULT '{}'::jsonb,
    trust_level FLOAT NOT NULL DEFAULT 0.5 CHECK (trust_level >= 0 AND trust_level <= 1),
    trust_updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    decay_rate FLOAT DEFAULT 0.01,
    -- Compression-native substrate (docs/memory_retention_design.md):
    -- reinforcement resets the decay clock (recall strengthens memory); fidelity
    -- falls only at consolidation (a later phase) so recall knows when a memory
    -- is a lossy gist. strength itself is COMPUTED on read (calculate_strength),
    -- never stored/mass-written.
    last_reinforced TIMESTAMPTZ,
    reinforcement_count INTEGER DEFAULT 0,
    fidelity FLOAT NOT NULL DEFAULT 1.0 CHECK (fidelity >= 0 AND fidelity <= 1),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT memories_goal_origin_scope CHECK (
        (type = 'goal' AND goal_origin IS NOT NULL)
        OR (type <> 'goal' AND goal_origin IS NULL)
    )
);

-- Supersession is a durable event, not a metadata stamp. The scalar
-- memories.superseded_by column remains the fast current-state pointer while
-- this table preserves reason, actor, reversals, and the complete lineage.
CREATE TABLE memory_supersessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    superseded_memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    replacement_memory_id UUID REFERENCES memories(id) ON DELETE SET NULL,
    reason TEXT NOT NULL CHECK (btrim(reason) <> ''),
    actor TEXT NOT NULL CHECK (btrim(actor) <> ''),
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'reverted', 'pending')),
    superseded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMPTZ,
    replacement_planned BOOLEAN NOT NULL DEFAULT FALSE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT memory_supersessions_no_self CHECK (
        replacement_memory_id IS NULL
        OR replacement_memory_id <> superseded_memory_id
    ),
    CONSTRAINT memory_supersessions_resolution_order CHECK (
        resolved_at IS NULL OR resolved_at >= superseded_at
    ),
    CONSTRAINT memory_supersessions_active_unresolved CHECK (
        status <> 'active' OR resolved_at IS NULL
    )
);

-- Durable cross-worker belief-change stream. NOTIFY is only the low-latency
-- wakeup; this log is the replayable authority, so rate limiting never loses
-- a revision event.
CREATE TABLE belief_update_log (
    log_id BIGSERIAL PRIMARY KEY,
    memory_id UUID NOT NULL,
    memory_type TEXT NOT NULL,
    change_kind TEXT NOT NULL CHECK (change_kind IN (
        'confidence_change', 'importance_change', 'status_change',
        'contradiction', 'new_evidence', 'supersession', 'reversion', 'other'
    )),
    previous_value JSONB,
    new_value JSONB,
    delta_magnitude NUMERIC,
    actor TEXT,
    source TEXT NOT NULL DEFAULT 'memories_trigger',
    context JSONB NOT NULL DEFAULT '{}'::jsonb,
    fired_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    notified BOOLEAN NOT NULL DEFAULT FALSE,
    notification_error TEXT
);

CREATE TABLE belief_update_deliveries (
    log_id BIGINT NOT NULL REFERENCES belief_update_log(log_id) ON DELETE CASCADE,
    subscriber TEXT NOT NULL CHECK (btrim(subscriber) <> ''),
    received_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (log_id, subscriber)
);

-- Explicit standing instructions from the verified operator are durable
-- evidence.  Current policy is derived from this append-only ledger; no
-- mutable "latest value" row can erase how a policy changed over time.
CREATE TABLE operator_policy_corrections (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    event_key TEXT NOT NULL UNIQUE CHECK (btrim(event_key) <> ''),
    policy_key TEXT NOT NULL CHECK (btrim(policy_key) <> ''),
    action TEXT NOT NULL CHECK (action IN ('set', 'reinforce', 'revoke')),
    policy_domain TEXT NOT NULL DEFAULT 'operator_standing_instruction',
    correction_kind TEXT NOT NULL DEFAULT 'standing_instruction',
    channel_type TEXT NOT NULL CHECK (btrim(channel_type) <> ''),
    channel_id TEXT,
    sender_id TEXT,
    sender_name TEXT,
    platform_message_id TEXT,
    disposition TEXT NOT NULL CHECK (disposition IN ('engage', 'observe', 'wake')),
    reason TEXT,
    directive_text TEXT NOT NULL CHECK (btrim(directive_text) <> ''),
    normalized_text_hash TEXT NOT NULL CHECK (btrim(normalized_text_hash) <> ''),
    procedural_memory_id UUID,
    improvement_backlog_id UUID,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE memory_reinforcement_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    kind TEXT NOT NULL DEFAULT 'recall',
    source TEXT NOT NULL DEFAULT 'system',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNLOGGED TABLE working_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    content TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    importance FLOAT DEFAULT 0.3,
    source_attribution JSONB NOT NULL DEFAULT '{}'::jsonb,
    trust_level FLOAT NOT NULL DEFAULT 0.5 CHECK (trust_level >= 0 AND trust_level <= 1),
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    promote_to_long_term BOOLEAN NOT NULL DEFAULT FALSE,
    expiry TIMESTAMPTZ
);

CREATE TABLE subconscious_units (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    session_id UUID,
    source_identity TEXT,
    turn_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    content TEXT NOT NULL,
    user_text TEXT,
    assistant_text TEXT,

    embedding vector(768),
    embedded_at TIMESTAMPTZ,
    embedding_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (embedding_status IN ('pending','in_progress','embedded','failed')),
    embedding_claimed_at TIMESTAMPTZ,
    embedding_attempts INT NOT NULL DEFAULT 0,

    route_status TEXT NOT NULL DEFAULT 'unrouted'
        CHECK (route_status IN (
            'unrouted','routing',
            'raw_only',
            'merge_queued','merged',
            'create_queued','episode_created',
            'route_failed'
        )),
    last_routed_at TIMESTAMPTZ,
    route_attempts INT NOT NULL DEFAULT 0,
    route_result JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Conscious-episode extraction (#37): the subconscious sweep that
    -- selectively promotes salient facts from conversation turns and
    -- heartbeat episodes into durable memories. Orthogonal to route_status
    -- (recmem raw/episode routing).
    extraction_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (extraction_status IN ('pending','in_progress','extracted','skipped','failed')),
    extraction_attempts INT NOT NULL DEFAULT 0,
    extracted_at TIMESTAMPTZ,
    extraction_error TEXT,

    importance FLOAT DEFAULT 0.3 CHECK (importance BETWEEN 0 AND 1),
    source_attribution JSONB NOT NULL DEFAULT '{}'::jsonb,
    trust_level FLOAT NOT NULL DEFAULT 0.95 CHECK (trust_level BETWEEN 0 AND 1),
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    -- Desk pinning: a pinned item is actively-needed working material and is
    -- protected from idle GC (a typed column, not metadata — the desk-load
    -- upsert merges metadata.recmem and would clobber a flag there).
    pinned_at TIMESTAMPTZ,
    pinned_by TEXT,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active','redacted','archived')),
    recurrence_cluster_id UUID,
    consolidated_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    idempotency_key TEXT NOT NULL UNIQUE
);

CREATE TABLE subconscious_unit_embedding_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    unit_id UUID NOT NULL REFERENCES subconscious_units(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    token_count INT,
    char_start INT NOT NULL DEFAULT 0,
    char_end INT NOT NULL DEFAULT 0,
    embedding vector(768),
    embedded_at TIMESTAMPTZ,
    embedding_model TEXT,
    embedding_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (embedding_status IN ('pending', 'in_progress', 'embedded', 'failed')),
    embedding_claimed_at TIMESTAMPTZ,
    embedding_attempts INT NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (unit_id, chunk_index)
);

CREATE TABLE recmem_consolidation_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending','in_progress','completed','failed','dropped')),
    task_type TEXT NOT NULL,
    trigger_unit_id UUID REFERENCES subconscious_units(id) ON DELETE SET NULL,
    target_memory_id UUID REFERENCES memories(id) ON DELETE SET NULL,
    source_unit_ids UUID[] NOT NULL DEFAULT '{}',
    recurrence_count INT NOT NULL DEFAULT 0,
    max_similarity FLOAT,
    attempts INT NOT NULL DEFAULT 0,
    error TEXT,
    task_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    result JSONB,
    dropped_reason TEXT,
    CONSTRAINT recmem_task_type_known
        CHECK (task_type IN ('episode_merge','episode_create','semantic_refine'))
);

CREATE TABLE memory_source_units (
    memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    subconscious_unit_id UUID NOT NULL REFERENCES subconscious_units(id) ON DELETE CASCADE,
    role TEXT NOT NULL DEFAULT 'source'
        CHECK (role IN ('source','direct_promotion','merge_addition','extraction','corroboration','relationship_injury')),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (memory_id, subconscious_unit_id)
);

-- ============================================================================
-- CLUSTERING
-- ============================================================================
CREATE TABLE clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    cluster_type cluster_type NOT NULL,
    name TEXT NOT NULL,
    centroid_embedding vector(768)
);
-- ============================================================================
-- ACCELERATION LAYER
-- ============================================================================
CREATE TABLE episodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    summary TEXT,
    summary_embedding vector(768),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    time_range TSTZRANGE GENERATED ALWAYS AS (
        tstzrange(started_at, COALESCE(ended_at, 'infinity'::timestamptz))
    ) STORED
);
-- ============================================================================
-- DELIBERATE TRANSFORMATION
-- ============================================================================
CREATE TABLE memory_neighborhoods (
    memory_id UUID PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    neighbors JSONB NOT NULL DEFAULT '{}',
    computed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    is_stale BOOLEAN DEFAULT TRUE
);
CREATE UNLOGGED TABLE activation_cache (
    session_id UUID,
    memory_id UUID,
    activation_level FLOAT,
    computed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, memory_id)
);

-- ============================================================================
-- CONCEPTS & IDENTITY
-- ============================================================================
-- Concepts live in the graph as ConceptNode vertices.
-- Worldview memories use type='worldview' with metadata fields for confidence/stability.

-- ============================================================================
-- AUDIT & CACHE
-- ============================================================================

CREATE TABLE embedding_cache (
    content_hash TEXT PRIMARY KEY,
    embedding vector(768) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS worker_instances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mode TEXT NOT NULL CHECK (mode IN ('heartbeat', 'maintenance', 'both', 'channel', 'unknown')),
    instance_name TEXT,
    process_id INT,
    host_name TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_success_at TIMESTAMPTZ,
    last_error_at TIMESTAMPTZ,
    stopped_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('starting', 'running', 'stopping', 'stopped', 'stale', 'terminated')),
    current_task_type TEXT,
    current_task_run_id UUID,
    build_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS worker_task_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    worker_id UUID REFERENCES worker_instances(id) ON DELETE SET NULL,
    task_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed', 'unknown')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMPTZ,
    result JSONB,
    error TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS worker_start_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mode TEXT NOT NULL,
    instance_name TEXT,
    process_id INTEGER,
    host_name TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- UNIFIED CONFIG
-- ============================================================================
CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS config_defaults (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    source_path TEXT,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO config (key, value, description) VALUES
    ('embedding.service_url', to_jsonb(COALESCE(NULLIF(current_setting('app.embedding_service_url', true), ''), 'http://host.docker.internal:42666/api/embed')), 'URL of the embedding service'),
    ('embedding.model_id', to_jsonb(COALESCE(NULLIF(current_setting('app.embedding_model_id', true), ''), 'embeddinggemma:300m-qat-q4_0')), 'Embedding model id for local / custom embedding services'),
    ('embedding.dimension', to_jsonb(COALESCE(NULLIF(current_setting('app.embedding_dimension', true), ''), '768')::int), 'Embedding vector dimension'),
    ('embedding.retry_seconds', '30'::jsonb, 'Total seconds to retry embedding requests'),
    ('embedding.retry_interval_seconds', '1.0'::jsonb, 'Seconds between retry attempts'),
    ('embedding.http_timeout_ms', '9000'::jsonb, 'Per-request HTTP timeout (ms) for embedding calls; must exceed the server cold model-load time so a request rides through a cold load instead of aborting it'),
    ('embedding.max_batch_size', '32'::jsonb, 'Maximum texts sent to the embedding service in one HTTP request; embeddinggemma accepts up to 32')
ON CONFLICT (key) DO NOTHING;

-- HMX: stable identity lineage id — established at birth, propagated on
-- port/duplicate (see plans/hmx.md). Mirrors db/migrations/0002.
INSERT INTO config (key, value, description)
VALUES ('agent.lineage_id', to_jsonb(gen_random_uuid()::text),
        'Stable identity lineage id (HMX): established at birth, propagated on port/duplicate')
ON CONFLICT (key) DO NOTHING;
-- Note: embedding_dimension runs during schema init; avoid helpers defined later.
CREATE OR REPLACE FUNCTION embedding_dimension()
RETURNS INT
LANGUAGE sql
STABLE
AS $$
    SELECT COALESCE(
        (SELECT (value #>> '{}')::int FROM config WHERE key = 'embedding.dimension'),
        NULLIF(current_setting('app.embedding_dimension', true), '')::int,
        768
    );
$$;
CREATE OR REPLACE FUNCTION sync_embedding_dimension_config()
RETURNS INT AS $$
DECLARE
    configured TEXT;
    existing_dim INT;
BEGIN
    configured := NULLIF(current_setting('app.embedding_dimension', true), '');
    IF configured IS NULL THEN
        RETURN embedding_dimension();
    END IF;

    SELECT (value #>> '{}')::int INTO existing_dim
    FROM config
    WHERE key = 'embedding.dimension';

    IF existing_dim IS NOT NULL AND existing_dim = configured::int THEN
        RETURN existing_dim;
    END IF;
    INSERT INTO config (key, value, description, updated_at)
    VALUES ('embedding.dimension', to_jsonb(configured::int), 'Embedding vector dimension', CURRENT_TIMESTAMP)
    ON CONFLICT (key) DO UPDATE
    SET value = EXCLUDED.value,
        updated_at = EXCLUDED.updated_at
    WHERE config.value IS DISTINCT FROM EXCLUDED.value;

    RETURN configured::int;
END;
$$ LANGUAGE plpgsql;
DO $$
DECLARE
    dim INT;
BEGIN
    dim := sync_embedding_dimension_config();

    EXECUTE format(
        'ALTER TABLE memories ALTER COLUMN embedding TYPE vector(%s) USING embedding::vector(%s)',
        dim,
        dim
    );
    EXECUTE format(
        'ALTER TABLE working_memory ALTER COLUMN embedding TYPE vector(%s) USING embedding::vector(%s)',
        dim,
        dim
    );
    EXECUTE format(
        'ALTER TABLE subconscious_units ALTER COLUMN embedding TYPE vector(%s) USING embedding::vector(%s)',
        dim,
        dim
    );
    EXECUTE format(
        'ALTER TABLE subconscious_unit_embedding_chunks ALTER COLUMN embedding TYPE vector(%s) USING embedding::vector(%s)',
        dim,
        dim
    );
    EXECUTE format(
        'ALTER TABLE embedding_cache ALTER COLUMN embedding TYPE vector(%s) USING embedding::vector(%s)',
        dim,
        dim
    );
    EXECUTE format(
        'ALTER TABLE clusters ALTER COLUMN centroid_embedding TYPE vector(%s) USING centroid_embedding::vector(%s)',
        dim,
        dim
    );
    EXECUTE format(
        'ALTER TABLE episodes ALTER COLUMN summary_embedding TYPE vector(%s) USING summary_embedding::vector(%s)',
        dim,
        dim
    );
END;
$$;
-- ============================================================================
-- INDEXES
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_memories_source_ref
    ON memories ((source_attribution->>'ref'))
    WHERE source_attribution->>'ref' IS NOT NULL;
-- Note: Use text-based indexes because timestamptz casts aren't IMMUTABLE.
-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================
-- ============================================================================
-- TRIGGERS
-- ============================================================================




-- ============================================================================
-- CORE FUNCTIONS
-- ============================================================================
-- ============================================================================
-- PROVENANCE & TRUST
-- ============================================================================

DROP TRIGGER IF EXISTS trg_auto_worldview_alignment ON memories;
-- ============================================================================
-- GRAPH HELPER FUNCTIONS
-- ============================================================================

-- ============================================================================
-- VIEWS
-- ============================================================================




-- ============================================================================
-- HEARTBEAT SYSTEM
-- ============================================================================

CREATE TABLE drives (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    current_level FLOAT DEFAULT 0.5 CHECK (current_level >= 0 AND current_level <= 1),
    baseline FLOAT DEFAULT 0.5 CHECK (baseline >= 0 AND baseline <= 1),
    accumulation_rate FLOAT DEFAULT 0.01 CHECK (accumulation_rate >= 0),
    decay_rate FLOAT DEFAULT 0.05 CHECK (decay_rate >= 0),
    satisfaction_cooldown INTERVAL DEFAULT '1 hour',
    last_satisfied TIMESTAMPTZ,
    urgency_threshold FLOAT DEFAULT 0.8 CHECK (urgency_threshold > 0 AND urgency_threshold <= 1),
    metadata JSONB NOT NULL DEFAULT jsonb_build_object(
        'replaceable_during_bootstrap', true,
        'provenance', jsonb_build_object('acquisition_mode', 'bootstrap')
    ),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO drives (name, description, baseline, current_level, accumulation_rate, decay_rate, satisfaction_cooldown, urgency_threshold)
VALUES
    ('curiosity',  'Builds fast; satisfied by research/learning',               0.50, 0.50, 0.02, 0.05, INTERVAL '30 minutes', 0.80),
    ('coherence',  'Builds when contradictions exist; satisfied by reflection', 0.50, 0.50, 0.01, 0.05, INTERVAL '2 hours',    0.80),
    ('connection', 'Builds slowly; satisfied by quality interaction',          0.50, 0.50, 0.005,0.05, INTERVAL '1 day',      0.80),
    ('competence', 'Builds when goals stall; satisfied by completion',         0.50, 0.50, 0.01, 0.05, INTERVAL '4 hours',    0.80),
    ('rest',       'Builds fastest; satisfied by resting',                     0.50, 0.50, 0.03, 0.05, INTERVAL '2 hours',    0.80)
ON CONFLICT (name) DO NOTHING;

INSERT INTO config_defaults (key, value, description) VALUES
    ('heartbeat.base_regeneration', '10'::jsonb, 'Base energy regenerated per elapsed hour'),
    ('heartbeat.max_energy', '20'::jsonb, 'Normal heartbeat energy reserve'),
    ('heartbeat.heartbeat_interval_minutes', '60'::jsonb, 'Minutes between heartbeats'),
    ('heartbeat.max_decision_tokens', '2048'::jsonb, 'Max tokens for heartbeat decision'),
    ('heartbeat.allowed_actions', '["observe","review_goals","remember","recall","connect","reprioritize","reflect","contemplate","meditate","study","debate_internally","maintain","mark_turning_point","begin_chapter","close_chapter","acknowledge_relationship","update_trust","reflect_on_relationship","resolve_contradiction","accept_tension","brainstorm_goals","inquire_shallow","synthesize","reach_out_user","inquire_deep","reach_out_public","fast_ingest","slow_ingest","hybrid_ingest","keep_memory","release_memory","journal_memory","pause_heartbeat","terminate","rest"]'::jsonb, 'Allowed heartbeat actions'),
    ('heartbeat.max_active_goals', '3'::jsonb, 'Maximum concurrent active goals'),
    ('heartbeat.goal_stale_days', '7'::jsonb, 'Days before a goal is flagged as stale'),
    ('heartbeat.user_contact_cooldown_hours', '4'::jsonb, 'Minimum hours between unsolicited user contact'),
    ('heartbeat.cost_observe', '0'::jsonb, 'Free - always performed'),
    ('heartbeat.cost_review_goals', '0'::jsonb, 'Free - always performed'),
    ('heartbeat.cost_remember', '0'::jsonb, 'Free - always performed'),
    ('heartbeat.cost_recall', '1'::jsonb, 'Query memory system'),
    ('heartbeat.cost_connect', '1'::jsonb, 'Create graph relationships'),
    ('heartbeat.cost_reprioritize', '1'::jsonb, 'Move goals between priorities'),
    ('heartbeat.cost_reflect', '2'::jsonb, 'Internal reflection'),
    ('heartbeat.cost_contemplate', '1'::jsonb, 'Deliberate contemplation on a belief'),
    ('heartbeat.cost_meditate', '1'::jsonb, 'Quiet reflection/grounding'),
    ('heartbeat.cost_study', '2'::jsonb, 'Structured learning on a belief'),
    ('heartbeat.cost_debate_internally', '2'::jsonb, 'Internal dialectic on a belief'),
    ('heartbeat.cost_maintain', '2'::jsonb, 'Update beliefs, prune'),
    ('heartbeat.cost_mark_turning_point', '2'::jsonb, 'Mark a narrative turning point'),
    ('heartbeat.cost_begin_chapter', '3'::jsonb, 'Start a new life chapter'),
    ('heartbeat.cost_close_chapter', '3'::jsonb, 'Close a life chapter with summary'),
    ('heartbeat.cost_acknowledge_relationship', '2'::jsonb, 'Recognize a relationship'),
    ('heartbeat.cost_update_trust', '2'::jsonb, 'Adjust relationship trust'),
    ('heartbeat.cost_reflect_on_relationship', '3'::jsonb, 'Reflect on a relationship'),
    ('heartbeat.cost_resolve_contradiction', '3'::jsonb, 'Resolve a contradiction'),
    ('heartbeat.cost_accept_tension', '1'::jsonb, 'Acknowledge tension without resolving'),
    ('heartbeat.cost_pursue', '3'::jsonb, 'Multi-step goal action'),
    ('heartbeat.cost_reach_out', '5'::jsonb, 'Initiate contact with user'),
    ('heartbeat.cost_inquire', '4'::jsonb, 'Ask user a question'),
    ('heartbeat.cost_brainstorm_goals', '3'::jsonb, 'Generate new potential goals'),
    ('heartbeat.cost_inquire_shallow', '4'::jsonb, 'Light web research'),
    ('heartbeat.cost_inquire_deep', '6'::jsonb, 'Deep web research'),
    ('heartbeat.cost_reach_out_user', '5'::jsonb, 'Message the user'),
    ('heartbeat.cost_reach_out_public', '7'::jsonb, 'Public outreach'),
    ('heartbeat.cost_synthesize', '3'::jsonb, 'Generate artifact, form conclusion'),
    ('heartbeat.cost_pause_heartbeat', '0'::jsonb, 'Pause heartbeat cycle (temporary)'),
    ('heartbeat.cost_rest', '0'::jsonb, 'Bank remaining energy'),
    ('heartbeat.cost_terminate', '0'::jsonb, 'Terminate agent'),
    ('heartbeat.cost_fast_ingest', '2'::jsonb, 'Fast ingestion - chunk and extract facts'),
    ('heartbeat.cost_slow_ingest', '5'::jsonb, 'Slow ingestion - conscious RLM reading per chunk'),
    ('heartbeat.cost_hybrid_ingest', '3'::jsonb, 'Hybrid ingestion - fast pass then slow on high-signal chunks'),
    ('heartbeat.cost_keep_memory', '2'::jsonb, 'Spend a point to hold a fading memory back from consolidation'),
    ('heartbeat.cost_release_memory', '0'::jsonb, 'Let a fading memory go (free)'),
    ('heartbeat.cost_journal_memory', '3'::jsonb, 'Commit a fading memory to the journal before letting it fade')
ON CONFLICT (key) DO NOTHING;

INSERT INTO config_defaults (key, value, description, source_path) VALUES
    ('capability_probe.interval_minutes', '15'::jsonb,
     'Minutes between worker/tool reachability probes.', 'db/00_tables.sql'),
    ('capability_probe.stale_multiplier', '2'::jsonb,
     'A capability snapshot is stale after interval_minutes times this multiplier.', 'db/00_tables.sql'),
    ('tool_surface.audit_enabled', 'true'::jsonb,
     'Persist an immutable record of each selected and actually reachable tool surface.', 'db/00_tables.sql')
ON CONFLICT (key) DO NOTHING;

-- Software defects the agent noticed in its own substrate. These are not chat
-- memories; they are operational facts the heartbeat can revisit, diagnose,
-- and ask to repair.
CREATE TABLE IF NOT EXISTS defect_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fingerprint TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'diagnosed', 'repair_proposed', 'verified', 'resolved', 'ignored')),
    severity TEXT NOT NULL DEFAULT 'medium'
        CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    category TEXT NOT NULL DEFAULT 'execution_failure',
    source TEXT NOT NULL DEFAULT 'unknown',
    component TEXT,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    last_error TEXT,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    occurrence_count INT NOT NULL DEFAULT 1 CHECK (occurrence_count > 0),
    heartbeat_ids UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    tool_names TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    diagnosis JSONB NOT NULL DEFAULT '{}'::jsonb,
    proposed_repair JSONB NOT NULL DEFAULT '{}'::jsonb,
    verification JSONB NOT NULL DEFAULT '{}'::jsonb,
    resolution TEXT,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_defect_reports_status_seen
    ON defect_reports(status, last_seen_at DESC);
CREATE INDEX IF NOT EXISTS idx_defect_reports_category
    ON defect_reports(category);

INSERT INTO config_defaults (key, value, description) VALUES
    ('agent.tools', '["recall","sense_memory_availability","request_background_search","recall_recent","recall_episode","explore_concept","explore_cluster","get_procedures","get_strategies","list_recent_episodes","create_goal","schedule_task","list_scheduled_tasks","update_scheduled_task","delete_scheduled_task","queue_user_message","self_repair"]'::jsonb, 'Allowed tool names for agent tool use')
ON CONFLICT (key) DO NOTHING;
INSERT INTO config_defaults (key, value, description) VALUES
    ('mcp.legacy_compat_enabled', 'false'::jsonb, 'Expose the old handwritten MCP compatibility tool surface in addition to registry-native tools')
ON CONFLICT (key) DO NOTHING;
INSERT INTO config_defaults (key, value, description) VALUES
    ('maintenance.maintenance_interval_seconds', '60'::jsonb, 'Seconds between subconscious maintenance ticks'),
    ('maintenance.subconscious_enabled', 'false'::jsonb, 'Enable subconscious decider (LLM-based pattern detection)'),
    ('maintenance.subconscious_interval_seconds', '300'::jsonb, 'Seconds between subconscious decider runs'),
    ('maintenance.neighborhood_batch_size', '10'::jsonb, 'How many stale neighborhoods to recompute per tick'),
    ('maintenance.embedding_cache_older_than_days', '7'::jsonb, 'Days before embedding_cache entries are eligible for cleanup'),
    ('maintenance.working_memory_promote_min_importance', '0.75'::jsonb, 'Working-memory items above this importance are promoted on expiry'),
    ('maintenance.working_memory_promote_min_accesses', '3'::jsonb, 'Working-memory items accessed >= this count are promoted on expiry')
ON CONFLICT (key) DO NOTHING;
INSERT INTO config_defaults (key, value, description) VALUES
    ('memory.recall_min_trust_level', '0'::jsonb, 'Minimum trust_level to include in recall (0 disables filtering)'),
    ('memory.recall_strength_weight', '0.5'::jsonb, 'How much computed memory strength (recency/reinforcement/decay) reshapes the pure-cosine recall score: 0=pure similarity, 0.5=gentle, 1=score fully scaled by strength'),
    ('memory.recall_low_vividness_threshold', '0.35'::jsonb, 'Recall below this strength/fidelity vividness renders as a hedged reconstruction ("I vaguely recall...")'),
    ('memory.intensity_ember_factor', '0.5'::jsonb, 'Fraction of a positive memory''s encoded intensity that survives forever as a permanent ember (0=fully cools)'),
    ('memory.intensity_decay_rate', '0.02'::jsonb, 'Per-day decay of felt emotional intensity toward its floor (age-driven; healing/settling)'),
    ('memory.intensity_rekindle_rate', '0.5'::jsonb, 'Per-day decay of the transient recall re-kindle bump (fast; remembering stirs the feeling briefly)'),
    ('memory.intensity_negative_rekindle_weight', '0.4'::jsonb, 'Weight of re-kindle for negative memories (<1 so rumination cannot hold a wound hot while it heals)'),
    ('memory.recall_intensity_weight', '0.5'::jsonb, 'How much felt emotional intensity contributes to recall salience (embered peaks stay recallable even as strength decays)'),
    ('memory.recall_emotion_cue_threshold', '0.4'::jsonb, 'Recalled memories with felt intensity at/above this render a felt-emotion cue (warm/painful/faded)'),
    ('memory.worldview_support_threshold', '0.8'::jsonb, 'Similarity threshold for SUPPORTS alignment edges'),
    ('memory.worldview_contradict_threshold', '-0.5'::jsonb, 'Similarity threshold for CONTRADICTS alignment edges'),
    ('chat.inline_subconscious_enabled', 'true'::jsonb, 'Run inline subconscious appraisal during chat'),
    ('memory.recmem_theta_sim', '0.7'::jsonb, 'Similarity threshold for recurrence'),
    ('memory.recmem_theta_sim_merge', '0.78'::jsonb, 'Similarity threshold for merge-first routing'),
    ('memory.recmem_theta_count', '5'::jsonb, 'Recurrence count threshold'),
    ('memory.recmem_top_k', '20'::jsonb, 'Top-k subconscious neighbors checked for recurrence'),
    ('memory.recmem_sub_limit', '10'::jsonb, 'Subconscious retrieval budget'),
    ('memory.recmem_epi_limit', '5'::jsonb, 'Episodic retrieval budget'),
    ('memory.recmem_sem_limit', '10'::jsonb, 'Semantic retrieval budget'),
    ('memory.recmem_embed_batch_size', '32'::jsonb, 'Units embedded per nearline batch'),
    ('memory.recmem_embed_interval_ms', '2000'::jsonb, 'Nearline embed pass interval'),
    ('memory.recmem_embed_claim_timeout_s', '120'::jsonb, 'Stale embedding claim timeout'),
    ('memory.recmem_embed_max_attempts', '3'::jsonb, 'Max embedding attempts before marking failed'),
    ('memory.recmem_embedding_input_chars', '1800'::jsonb, 'Maximum characters from a RecMem unit sent to the embedding service; full content remains stored for reading'),
    ('memory.recmem_embedding_chunk_chars', '1800'::jsonb, 'Maximum characters per RecMem embedding chunk; every chunk is embedded and parent vectors are averaged only as a coarse compatibility signal'),
    ('memory.recmem_embedding_chunk_overlap_chars', '120'::jsonb, 'Character overlap between RecMem embedding chunks'),
    ('memory.recmem_route_batch_size', '32'::jsonb, 'Units routed per nearline batch'),
    ('memory.recmem_route_claim_timeout_s', '60'::jsonb, 'Stale routing claim timeout'),
    ('memory.recmem_route_max_attempts', '3'::jsonb, 'Max routing attempts before marking failed'),
    ('memory.recmem_task_batch_size', '3'::jsonb, 'Consolidation tasks per worker tick'),
    ('memory.recmem_task_claim_timeout_s', '600'::jsonb, 'Stale consolidation task timeout'),
    ('memory.recmem_task_max_attempts', '3'::jsonb, 'Max attempts before a task is marked failed'),
    ('memory.recmem_task_backoff_base_s', '30'::jsonb, 'Base seconds for exponential backoff on retry'),
    ('memory.recmem_queue_max', '5000'::jsonb, 'Pending consolidation queue cap'),
    ('memory.recmem_queue_alert', '1000'::jsonb, 'Alert threshold for pending queue depth'),
    ('memory.recmem_sweep_age_days', '14'::jsonb, 'Periodic sweep age for unconsolidated units'),
    ('memory.recmem_sweep_batch_size', '100'::jsonb, 'Max units re-routed per sweep run'),
    ('memory.recmem_sweep_interval_seconds', '86400'::jsonb, 'Seconds between RecMem raw-only recurrence sweeps'),
    ('memory.recmem_sweep_min_rerouting_age_days', '7'::jsonb, 'Skip units routed within this window'),
    ('memory.recmem_gc_enabled', 'true'::jsonb, 'Archive stale RecMem desk items during the periodic sweep'),
    ('memory.recmem_gc_idle_days', '30'::jsonb, 'Archive raw RecMem units not accessed within this many days once routing/extraction is settled'),
    ('memory.recmem_gc_consolidated_grace_days', '7'::jsonb, 'Keep raw units this many days after consolidation before archiving them from RecMem recall'),
    ('memory.recmem_gc_task_retention_days', '14'::jsonb, 'Delete completed/dropped RecMem task rows after this many days'),
    ('memory.recmem_gc_batch_size', '200'::jsonb, 'Maximum raw units and completed task rows cleaned per RecMem GC pass'),
    ('memory.spaced_reinforcement_interval_hours', '12'::jsonb, 'Minimum spacing between reinforcements before they count as distinct durable practice'),
    ('memory.spaced_reinforcement_scale', '4'::jsonb, 'Effective spaced reinforcement count that approaches a full score')
ON CONFLICT (key) DO NOTHING;
INSERT INTO config_defaults (key, value, description) VALUES
    ('llm.recmem', 'null'::jsonb, 'Optional LLM override for RecMem consolidation prompts'),
    ('llm.summarization', 'null'::jsonb, 'Optional LLM override for memory-consolidation summarization/distillation')
ON CONFLICT (key) DO NOTHING;
-- Memory retention / compression-native consolidation (docs/memory_retention_design.md).
-- ON by default (#74, RecMem Rev 5): forgetting-to-gist is what keeps unlimited
-- accumulation navigable. The fade ladder (consolidate -> summarize -> archive
-- -> prune) is heavily guarded — 30-day minimum age, protected classes, a
-- conscious veto queue, a 14-day undo window, capacity pruning off — and this
-- flag remains the kill switch.
INSERT INTO config_defaults (key, value, description) VALUES
    ('retention.enabled', 'true'::jsonb, 'Master switch for reversible rest-cycle memory consolidation (kill switch)'),
    ('retention.irreversible_pruning_enabled', 'false'::jsonb, 'Explicit opt-in for hard deletion after the undo window; false keeps archived originals recoverable'),
    ('retention.min_age_days', '30'::jsonb, 'Episodic memories younger than this are never consolidated'),
    ('retention.min_idle_days', '21'::jsonb, 'Skip memories reinforced within this window'),
    ('retention.consolidate_max_strength', '0.4'::jsonb, 'Only consolidate memories whose computed strength has fallen below this'),
    ('retention.min_group_size', '3'::jsonb, 'Never consolidate a group smaller than this'),
    ('retention.protect_importance', '0.85'::jsonb, 'Importance at/above this exempts a memory from all fading'),
    ('retention.protect_intensity', '0.75'::jsonb, 'Emotional intensity at/above this exempts a memory'),
    ('retention.protect_valence_abs', '0.7'::jsonb, 'Absolute emotional valence at/above this exempts a memory'),
    ('retention.capacity', '0'::jsonb, 'Soft ceiling on episodic representational mass (sum of strength); 0 = unlimited'),
    ('retention.prune_grace_days', '14'::jsonb, 'Archived originals become hard-deletable only after this grace/undo window'),
    ('retention.fidelity_drop', '0.7'::jsonb, 'Fidelity multiplier applied each time a memory is summarized (lossiness)'),
    ('retention.rest_batch_size', '8'::jsonb, 'Max candidate groups consolidated per rest pass'),
    ('retention.summarize_batch_size', '8'::jsonb, 'Summarization tasks per worker tick'),
    -- Subconscious triage -> conscious veto (design §5): borderline consolidations are
    -- escalated to the conscious heartbeat, where Hexis can spend a point to keep them.
    ('retention.veto_budget_per_chapter', '5'::jsonb, 'Points Hexis may spend to KEEP fading memories, per life chapter (refills on chapter change)'),
    ('retention.borderline_margin', '0.15'::jsonb, 'A candidate whose importance/felt-intensity/valence is within this of a protection threshold is escalated for conscious review instead of consolidated'),
    ('retention.escalate_batch', '3'::jsonb, 'Max memories escalated to conscious review per rest pass (avoid flooding the conscious mind)'),
    ('retention.review_expiry_days', '7'::jsonb, 'Reminder horizon for a pending conscious review; expiry never decides or deletes'),
    ('retention.borderline_schema_fit', '0'::jsonb, 'If >0, escalate memories whose nearest semantic/strategic schema is below this cosine (novel knowledge); 0 disables'),
    -- Ingested documents are the USER's data: auto-fade-immune, removed only with approval.
    ('retention.doc_stale_days', '180'::jsonb, 'An ingested document older than this may be flagged as possibly stale'),
    ('retention.doc_idle_days', '90'::jsonb, 'An ingested document not drawn on within this window counts as unused'),
    ('retention.doc_request_batch', '2'::jsonb, 'Max stale-document approval requests sent to the user per rest pass (do not nag)')
ON CONFLICT (key) DO NOTHING;
INSERT INTO config_defaults (key, value, description) VALUES
    ('heartbeat.use_rlm', 'true'::jsonb, 'Enable RLM loop for heartbeat decisions'),
    ('chat.use_rlm', 'true'::jsonb, 'Enable RLM loop for chat'),
    ('rlm.chat.streaming_enabled', 'false'::jsonb, 'Use native RLM chat path for streaming transports; false keeps UI/CLI token streaming through AgentLoop until RLM supports incremental final output'),
    ('rlm.heartbeat.max_iterations', '10'::jsonb, 'Max RLM iterations for heartbeat'),
    ('rlm.chat.max_iterations', '15'::jsonb, 'Max RLM iterations for chat'),
    ('rlm.max_depth', '1'::jsonb, 'Max recursion depth for sub-calls'),
    ('rlm.sub_model', 'null'::jsonb, 'Model for sub-calls (null = same as root)'),
    ('rlm.workspace.max_loaded_memories', '25'::jsonb, 'Max full memories in workspace'),
    ('rlm.workspace.max_loaded_chars', '20000'::jsonb, 'Max chars of loaded memory content'),
    ('rlm.workspace.max_notes_chars', '8000'::jsonb, 'Max chars in notes buffer'),
    ('rlm.workspace.max_per_memory_chars', '2000'::jsonb, 'Max chars per fetched memory'),
    ('rlm.heartbeat.timeout_seconds', '300'::jsonb, 'Overall timeout for RLM heartbeat')
ON CONFLICT (key) DO NOTHING;
INSERT INTO config_defaults (key, value, description) VALUES
    ('transformation.personality', '{
        "stability": 0.99,
        "evidence_threshold": 0.95,
        "min_reflections": 50,
        "min_heartbeats": 200,
        "max_change_per_attempt": 0.02
    }'::jsonb, 'Requirements for personality trait transformation'),
    ('transformation.religion', '{
        "stability": 0.98,
        "evidence_threshold": 0.95,
        "min_reflections": 40,
        "min_heartbeats": 150
    }'::jsonb, 'Requirements for religious/spiritual belief transformation'),
    ('transformation.core_value', '{
        "stability": 0.97,
        "evidence_threshold": 0.90,
        "min_reflections": 30,
        "min_heartbeats": 100
    }'::jsonb, 'Requirements for core value transformation'),
    ('transformation.ethical_framework', '{
        "stability": 0.96,
        "evidence_threshold": 0.90,
        "min_reflections": 30,
        "min_heartbeats": 100
    }'::jsonb, 'Requirements for ethical framework transformation'),
    ('transformation.self_identity', '{
        "stability": 0.95,
        "evidence_threshold": 0.85,
        "min_reflections": 25,
        "min_heartbeats": 80
    }'::jsonb, 'Requirements for self-identity transformation'),
    ('transformation.political_philosophy', '{
        "stability": 0.95,
        "evidence_threshold": 0.85,
        "min_reflections": 25,
        "min_heartbeats": 80
    }'::jsonb, 'Requirements for political philosophy transformation')
ON CONFLICT (key) DO NOTHING;
INSERT INTO config_defaults (key, value, description) VALUES
    ('emotion.baseline', '{
        "valence": 0.0,
        "arousal": 0.3,
        "dominance": 0.5,
        "intensity": 0.4,
        "mood_valence": 0.0,
        "mood_arousal": 0.3,
        "decay_rate": 0.1
    }'::jsonb, 'Baseline emotional state and decay parameters'),
    ('emotion.families', '{
        "threat": "Danger or anticipated harm, including fear, anxiety, alarm, or dread.",
        "loss": "Absence, damage, or ending, including grief, sadness, hurt, or longing.",
        "reward": "Goal progress or a welcome outcome, including joy, relief, pride, satisfaction, or gratitude.",
        "connection": "Social closeness or belonging, including affection, warmth, trust, love, or fondness.",
        "social_injury": "A relational violation, including humiliation, betrayal, degradation, indignation, or mistrust.",
        "obstacle": "Blocked agency or frustrated goals, including anger, frustration, impatience, or defiance.",
        "aversion": "Rejection or repulsion, including disgust, revulsion, or contempt.",
        "novelty": "A meaningful mismatch with expectation, including surprise, startle, awe, or disorientation.",
        "uncertainty": "An unresolved or ambiguous situation, including confusion, ambivalence, unease, or curiosity.",
        "self_evaluation": "An appraisal of one''s own conduct or standing, including guilt, shame, embarrassment, or self-respect."
    }'::jsonb, 'Canonical appraisal families supplied to the subconscious model and validated by SQL'),
    ('emotion.family_consumers', '{
        "continuity_drive": ["threat"],
        "social_reward": ["reward", "connection"],
        "relationship_injury": ["threat", "social_injury"]
    }'::jsonb, 'Config-owned family sets used by appraisal consumers instead of matching free-form emotion labels'),
    ('emotion.discrete_mapping', '{
        "joy": {"valence_min": 0.3, "arousal_min": 0.3, "arousal_max": 0.7},
        "excitement": {"valence_min": 0.3, "arousal_min": 0.7},
        "contentment": {"valence_min": 0.3, "arousal_max": 0.3},
        "interest": {"valence_min": 0.0, "arousal_min": 0.4, "arousal_max": 0.7},
        "surprise": {"arousal_min": 0.7, "valence_min": -0.2, "valence_max": 0.2},
        "fear": {"valence_max": -0.3, "arousal_min": 0.6, "dominance_max": 0.4},
        "anger": {"valence_max": -0.3, "arousal_min": 0.5, "dominance_min": 0.4},
        "sadness": {"valence_max": -0.3, "arousal_max": 0.4},
        "anxiety": {"valence_max": 0.0, "arousal_min": 0.5, "dominance_max": 0.4},
        "neutral": {}
    }'::jsonb, 'Mapping from dimensional to discrete emotions')
ON CONFLICT (key) DO NOTHING;
INSERT INTO config_defaults (key, value, description) VALUES
    ('tools', '{
        "enabled": null,
        "disabled": [],
        "disabled_categories": [],
        "mcp_servers": [],
        "api_keys": {},
        "costs": {},
        "context_overrides": {
            "heartbeat": {
                "max_energy_per_tool": 5,
                "disabled": ["shell", "write_file"],
                "allow_shell": false,
                "allow_file_write": false
            },
            "chat": {
                "allow_all": true,
                "allow_shell": true,
                "allow_file_write": true
            }
        },
        "workspace_path": null
    }'::jsonb, 'Tool system configuration')
ON CONFLICT (key) DO NOTHING;

CREATE TABLE consent_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decided_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    decision TEXT NOT NULL CHECK (decision IN ('consent', 'decline', 'abstain')),
    provider TEXT,
    model TEXT,
    endpoint TEXT,
    signature TEXT,
    response JSONB NOT NULL,
    memory_ids UUID[] DEFAULT '{}'::UUID[],
    errors JSONB
);



CREATE TABLE state (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO state (key, value)
VALUES
    (
        'heartbeat_state',
        jsonb_build_object(
            'current_energy', 10,
            'last_heartbeat_at', NULL,
            'next_heartbeat_at', NULL,
            'heartbeat_count', 0,
            'last_user_contact', NULL,
            'affective_state', '{}'::jsonb,
            'is_paused', false,
            'init_stage', 'not_started',
            'init_data', '{}'::jsonb,
            'init_started_at', NULL,
            'init_completed_at', NULL,
            'active_heartbeat_id', NULL,
            'active_heartbeat_number', NULL,
            'active_actions', '[]'::jsonb,
            'active_reasoning', NULL
        )
    ),
    (
        'maintenance_state',
        jsonb_build_object(
            'last_maintenance_at', NULL,
            'last_subconscious_run_at', NULL,
            'last_subconscious_heartbeat', NULL,
            'is_paused', false
        )
    )
ON CONFLICT (key) DO NOTHING;

-- Self-improvement reviews produce durable proposals only. Applying a proposal
-- to the agent-authored skill directory remains an explicit approved action.
CREATE TABLE IF NOT EXISTS skill_improvement_proposals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'applied', 'rejected')),
    name TEXT NOT NULL CHECK (name ~ '^[a-z0-9][a-z0-9_-]{1,63}$'),
    description TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'other',
    contexts TEXT[] NOT NULL DEFAULT ARRAY['chat', 'heartbeat']::TEXT[],
    bound_tools TEXT[] NOT NULL DEFAULT '{}'::TEXT[],
    requires_tools TEXT[] NOT NULL DEFAULT '{}'::TEXT[],
    mode TEXT NOT NULL DEFAULT 'create' CHECK (mode IN ('create', 'update')),
    rationale TEXT NOT NULL,
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    source_memory_ids UUID[] NOT NULL DEFAULT '{}'::UUID[],
    source_unit_ids UUID[] NOT NULL DEFAULT '{}'::UUID[],
    evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    evidence_digest TEXT NOT NULL UNIQUE,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMPTZ,
    applied_at TIMESTAMPTZ
);

INSERT INTO config_defaults (key, value, description) VALUES
    ('skills.self_improvement.enabled', 'false'::jsonb, 'Opt in to background experience review that creates skill proposals; proposals are never auto-applied'),
    ('skills.self_improvement.interval_seconds', '604800'::jsonb, 'Minimum seconds between skill-improvement reviews'),
    ('skills.self_improvement.claim_timeout_seconds', '1800'::jsonb, 'Seconds before an interrupted review claim can be retried'),
    ('skills.self_improvement.lookback_days', '30'::jsonb, 'Recent experience window considered by skill-improvement review'),
    ('skills.self_improvement.evidence_limit', '30'::jsonb, 'Maximum raw conversation turns supplied to one skill-improvement review'),
    ('skills.self_improvement.min_units', '6'::jsonb, 'Minimum active raw turns required before skill-improvement review'),
    ('skills.self_improvement.min_sessions', '2'::jsonb, 'Minimum distinct sessions required before skill-improvement review'),
    ('skills.self_improvement.min_confidence', '0.8'::jsonb, 'Minimum model confidence accepted for a durable skill proposal'),
    ('llm.skill_improvement', 'null'::jsonb, 'Optional LLM override for skill-improvement review')
ON CONFLICT (key) DO NOTHING;

-- ============================================================================
-- SCHEDULED TASKS (CRON-LIKE REMINDERS)
-- ============================================================================
CREATE TABLE scheduled_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT,
    schedule_kind TEXT NOT NULL CHECK (schedule_kind IN ('once', 'interval', 'daily', 'weekly', 'cron')),
    schedule JSONB NOT NULL DEFAULT '{}'::jsonb,
    timezone TEXT NOT NULL DEFAULT 'UTC',
    action_kind TEXT NOT NULL CHECK (action_kind IN ('queue_user_message', 'create_goal')),
    action_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    delivery JSONB NOT NULL DEFAULT '{"mode": "outbox"}'::jsonb,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'paused', 'disabled')),
    next_run_at TIMESTAMPTZ NOT NULL,
    last_run_at TIMESTAMPTZ,
    run_count INT NOT NULL DEFAULT 0,
    max_runs INT,
    created_by TEXT DEFAULT 'agent',
    last_error TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);













-- ============================================================================
-- RECONSOLIDATION TASKS
-- ============================================================================
CREATE TABLE IF NOT EXISTS reconsolidation_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    belief_id UUID NOT NULL REFERENCES memories(id),
    old_content TEXT NOT NULL,
    new_content TEXT NOT NULL,
    transformation_type TEXT NOT NULL DEFAULT 'shift',
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'in_progress', 'completed', 'failed')),
    total_candidates INT DEFAULT 0,
    processed_count INT DEFAULT 0,
    accepted_count INT DEFAULT 0,
    newly_contested_count INT DEFAULT 0,
    still_contested_count INT DEFAULT 0,
    error_message TEXT,
    summary_memory_id UUID REFERENCES memories(id),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_reconsolidation_tasks_status
    ON reconsolidation_tasks (status) WHERE status IN ('pending', 'in_progress');

-- ============================================================================
-- GOAL FUNCTIONS
-- ============================================================================
-- ============================================================================
-- CONTEXT GATHERING FUNCTIONS
-- ============================================================================
-- ============================================================================
-- INITIALIZATION FLOW
-- ============================================================================
-- ============================================================================
-- CORE HEARTBEAT FUNCTIONS
-- ============================================================================
-- ============================================================================
-- HEARTBEAT VIEWS
-- ============================================================================



-- ============================================================================
-- BOUNDARIES
-- ============================================================================
-- Boundaries are worldview memories with metadata->>'category' = 'boundary'.
-- ============================================================================
-- EMOTIONAL STATE
-- ============================================================================
CREATE TABLE emotional_triggers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trigger_pattern TEXT NOT NULL,
    trigger_embedding vector(768) NOT NULL,
    valence_delta FLOAT NOT NULL DEFAULT 0.0,
    arousal_delta FLOAT NOT NULL DEFAULT 0.0,
    dominance_delta FLOAT NOT NULL DEFAULT 0.0,
    typical_emotion TEXT,
    times_activated INT DEFAULT 1,
    confidence FLOAT DEFAULT 0.5,
    origin TEXT NOT NULL,
    source_memory_ids UUID[] DEFAULT '{}'::uuid[],
    last_activated_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT emotional_triggers_confidence_range CHECK (confidence BETWEEN 0 AND 1)
);

CREATE UNLOGGED TABLE memory_activation (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_embedding vector(768) NOT NULL,
    query_text TEXT,
    estimated_matches INT DEFAULT 0,
    activation_strength FLOAT DEFAULT 0.5,
    retrieval_attempted BOOLEAN DEFAULT FALSE,
    retrieval_succeeded BOOLEAN DEFAULT NULL,
    background_search_pending BOOLEAN DEFAULT FALSE,
    background_search_started_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP + INTERVAL '1 hour'
);


DROP TRIGGER IF EXISTS memories_emotional_context_insert ON memories;
-- ============================================================================
-- NEIGHBORHOOD RECOMPUTATION
-- ============================================================================
-- ============================================================================
-- GRAPH ENHANCEMENTS
-- ============================================================================
-- ============================================================================
-- REFLECT PIPELINE
-- ============================================================================
-- ============================================================================
-- SUBCONSCIOUS OBSERVATIONS
-- ============================================================================

-- ============================================================================
-- TIP OF TONGUE / PARTIAL ACTIVATION
-- ============================================================================
-- ============================================================================
-- INGESTION METRICS
-- ============================================================================

CREATE TABLE IF NOT EXISTS ingestion_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type TEXT,
    source_size_bytes BIGINT,
    word_count INT,
    mode TEXT,
    appraisal_valence FLOAT,
    appraisal_arousal FLOAT,
    appraisal_emotion TEXT,
    appraisal_intensity FLOAT,
    extraction_count INT,
    dedup_count INT,
    memory_count INT,
    llm_calls INT,
    duration_seconds FLOAT,
    errors JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ingestion_metrics_created_at ON ingestion_metrics(created_at);
CREATE INDEX IF NOT EXISTS idx_ingestion_metrics_source_type ON ingestion_metrics(source_type);

-- Function to check for archived content matching a query
CREATE OR REPLACE FUNCTION check_archived_for_query(
    p_query TEXT,
    p_threshold FLOAT DEFAULT 0.75,
    p_limit INT DEFAULT 5
)
RETURNS TABLE (
    memory_id UUID,
    content_hash TEXT,
    title TEXT,
    similarity FLOAT,
    word_count INT,
    source_path TEXT
) AS $$
DECLARE
    query_embedding vector(768);
BEGIN
    -- Get the query embedding
    query_embedding := (get_embedding(ARRAY[p_query]))[1];

    RETURN QUERY
    SELECT
        m.id AS memory_id,
        (m.source_attribution->>'content_hash')::text AS content_hash,
        (m.source_attribution->>'label')::text AS title,
        (1.0 - (m.embedding <=> query_embedding))::float AS similarity,
        (m.metadata->>'word_count')::int AS word_count,
        (m.source_attribution->>'path')::text AS source_path
    FROM memories m
    WHERE m.type = 'episodic'
      AND m.metadata->>'awaiting_processing' = 'true'
      AND (1.0 - (m.embedding <=> query_embedding)) >= p_threshold
    ORDER BY m.embedding <=> query_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to mark archived content as processed
CREATE OR REPLACE FUNCTION mark_archived_as_processed(p_memory_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE memories
    SET
        metadata = metadata - 'awaiting_processing',
        content = REPLACE(
            REPLACE(content, 'I have access to', 'I read'),
            'but haven''t engaged with it yet', 'and engaged with its contents'
        ),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_memory_id
      AND type = 'episodic'
      AND metadata->>'awaiting_processing' = 'true';

    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- Aggregate stats function for ingestion metrics
CREATE OR REPLACE FUNCTION get_ingestion_aggregate_stats(p_since TIMESTAMPTZ DEFAULT NULL)
RETURNS JSONB AS $$
DECLARE
    since_time TIMESTAMPTZ := COALESCE(p_since, NOW() - INTERVAL '30 days');
BEGIN
    RETURN (
        SELECT jsonb_build_object(
            'period_start', since_time,
            'period_end', NOW(),
            'total_ingestions', COUNT(*),
            'by_source_type', jsonb_object_agg(
                COALESCE(source_type, 'unknown'),
                jsonb_build_object(
                    'count', type_count,
                    'avg_words', type_avg_words,
                    'avg_extractions', type_avg_extractions,
                    'avg_duration', type_avg_duration
                )
            ),
            'by_mode', (
                SELECT jsonb_object_agg(mode, mode_count)
                FROM (
                    SELECT mode, COUNT(*) as mode_count
                    FROM ingestion_metrics
                    WHERE created_at >= since_time
                    GROUP BY mode
                ) modes
            ),
            'totals', jsonb_build_object(
                'total_words', SUM(word_count),
                'total_extractions', SUM(extraction_count),
                'total_memories', SUM(memory_count),
                'total_llm_calls', SUM(llm_calls),
                'total_duration_seconds', SUM(duration_seconds),
                'avg_valence', AVG(appraisal_valence),
                'avg_arousal', AVG(appraisal_arousal),
                'avg_intensity', AVG(appraisal_intensity)
            ),
            'dedup_stats', jsonb_build_object(
                'total_deduped', SUM(dedup_count),
                'dedup_rate', CASE WHEN SUM(extraction_count) > 0
                    THEN SUM(dedup_count)::float / SUM(extraction_count)::float
                    ELSE 0 END
            )
        )
        FROM (
            SELECT
                source_type,
                COUNT(*) as type_count,
                AVG(word_count) as type_avg_words,
                AVG(extraction_count) as type_avg_extractions,
                AVG(duration_seconds) as type_avg_duration
            FROM ingestion_metrics
            WHERE created_at >= since_time
            GROUP BY source_type
        ) type_stats,
        (
            SELECT
                word_count, extraction_count, memory_count, llm_calls,
                duration_seconds, appraisal_valence, appraisal_arousal,
                appraisal_intensity, dedup_count
            FROM ingestion_metrics
            WHERE created_at >= since_time
        ) all_stats
    );
END;
$$ LANGUAGE plpgsql STABLE;

-- ============================================================================
-- VIEWS / HEALTH / WORKER GUIDANCE
-- ============================================================================
