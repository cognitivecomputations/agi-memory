-- Hexis schema: indexes.
-- Each baseline file runs as its own psql session at initdb, so it must pin
-- its own search_path: under an ag_catalog-first cluster default, unqualified
-- CREATEs would land Hexis objects in ag_catalog (the #77 fossil bug).
SET search_path = public, ag_catalog, "$user";

CREATE INDEX IF NOT EXISTS idx_memories_source_content_hash
    ON memories ((source_attribution->>'content_hash'))
    WHERE source_attribution->>'content_hash' IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_episodes_ended_at ON episodes (ended_at);
CREATE INDEX IF NOT EXISTS idx_config_key_pattern ON config (key text_pattern_ops);
CREATE INDEX idx_memories_embedding ON memories USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL AND embedding_status = 'embedded';
CREATE INDEX IF NOT EXISTS idx_memories_embedding_queue
    ON memories (embedding_status, created_at)
    WHERE embedding_status IN ('pending', 'in_progress');
CREATE INDEX idx_memories_status ON memories (status);
CREATE INDEX idx_memories_type ON memories (type);
CREATE INDEX IF NOT EXISTS idx_memories_validity
    ON memories (valid_until)
    WHERE valid_until IS NOT NULL;
CREATE INDEX idx_memories_content ON memories USING GIN (content gin_trgm_ops);
CREATE INDEX idx_memories_content_fts ON memories USING GIN (to_tsvector('english', content));
CREATE INDEX idx_memories_importance ON memories (importance DESC) WHERE status = 'active';
CREATE INDEX idx_memories_created ON memories (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_tool_write_session_created
    ON memories ((metadata#>>'{tool_write,session_id}'), created_at DESC)
    WHERE status = 'active' AND metadata ? 'tool_write';
CREATE INDEX idx_memories_last_accessed ON memories (last_accessed DESC NULLS LAST);
CREATE INDEX idx_memories_updated ON memories (updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_reinforcement_events_memory_created
    ON memory_reinforcement_events (memory_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_supersessions_superseded
    ON memory_supersessions (superseded_memory_id, superseded_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_supersessions_replacement
    ON memory_supersessions (replacement_memory_id, superseded_at DESC)
    WHERE replacement_memory_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_supersessions_active_per_memory
    ON memory_supersessions (superseded_memory_id)
    WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_belief_update_log_recent
    ON belief_update_log (fired_at DESC, log_id DESC);
CREATE INDEX IF NOT EXISTS idx_belief_update_log_memory
    ON belief_update_log (memory_id, fired_at DESC);
CREATE INDEX IF NOT EXISTS idx_belief_update_log_kind
    ON belief_update_log (change_kind, fired_at DESC);
CREATE INDEX IF NOT EXISTS idx_belief_update_deliveries_subscriber
    ON belief_update_deliveries (subscriber, received_at DESC);
CREATE INDEX IF NOT EXISTS idx_operator_policy_corrections_policy_created
    ON operator_policy_corrections (policy_key, created_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_operator_policy_corrections_action_created
    ON operator_policy_corrections (action, created_at DESC, id DESC);
CREATE INDEX idx_memories_activation_boost ON memories (((metadata->>'activation_boost')::float))
    WHERE metadata ? 'activation_boost';
CREATE INDEX idx_memories_metadata ON memories USING GIN (metadata);
CREATE INDEX idx_memories_emotional_valence ON memories ((metadata->>'emotional_valence')) WHERE type = 'episodic';
CREATE INDEX idx_memories_confidence ON memories ((metadata->>'confidence')) WHERE type = 'semantic';
CREATE INDEX idx_memories_worldview_confidence ON memories (((metadata->>'confidence')::float)) WHERE type = 'worldview';
CREATE INDEX idx_memories_worldview_active_exploration ON memories (updated_at DESC)
    WHERE type = 'worldview'
      AND COALESCE((metadata->'transformation_state'->>'active_exploration')::boolean, false) = true;
CREATE INDEX idx_memories_emotional_pattern_created ON memories (created_at DESC)
    WHERE type = 'strategic'
      AND metadata->'supporting_evidence'->>'kind' = 'emotional_pattern';
CREATE INDEX idx_working_memory_expiry ON working_memory (expiry);
CREATE INDEX idx_working_memory_embedding ON working_memory USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_subconscious_units_embedding
    ON subconscious_units USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL AND status = 'active';
CREATE INDEX IF NOT EXISTS idx_subconscious_unit_chunks_unit
    ON subconscious_unit_embedding_chunks (unit_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_subconscious_unit_chunks_fts
    ON subconscious_unit_embedding_chunks USING GIN (to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_subconscious_unit_chunks_embedding
    ON subconscious_unit_embedding_chunks USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_subconscious_unit_chunks_embed_queue
    ON subconscious_unit_embedding_chunks (embedding_status, created_at)
    WHERE embedding_status IN ('pending', 'in_progress');
CREATE INDEX IF NOT EXISTS idx_subconscious_units_embed_pending
    ON subconscious_units (created_at)
    WHERE embedding_status = 'pending';
CREATE INDEX IF NOT EXISTS idx_subconscious_units_embed_claimed
    ON subconscious_units (embedding_claimed_at)
    WHERE embedding_status = 'in_progress';
CREATE INDEX IF NOT EXISTS idx_subconscious_units_route_pending
    ON subconscious_units (last_routed_at NULLS FIRST, created_at)
    WHERE embedding_status = 'embedded' AND route_status = 'unrouted';
CREATE INDEX IF NOT EXISTS idx_subconscious_units_route_claimed
    ON subconscious_units (last_routed_at)
    WHERE route_status = 'routing';
CREATE INDEX IF NOT EXISTS idx_subconscious_units_raw_only
    ON subconscious_units (last_routed_at)
    WHERE route_status = 'raw_only' AND consolidated_at IS NULL AND status = 'active';
CREATE INDEX IF NOT EXISTS idx_subconscious_units_extraction_pending
    ON subconscious_units (turn_at)
    WHERE extraction_status = 'pending' AND status = 'active';
CREATE INDEX IF NOT EXISTS idx_subconscious_units_status_created
    ON subconscious_units (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_subconscious_units_last_accessed
    ON subconscious_units (last_accessed DESC NULLS LAST)
    WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_subconscious_units_gc_candidates
    ON subconscious_units (route_status, COALESCE(last_accessed, consolidated_at, last_routed_at, created_at))
    WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_subconscious_units_pinned
    ON subconscious_units (pinned_at)
    WHERE pinned_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_subconscious_units_session_created
    ON subconscious_units (session_id, created_at DESC)
    WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_subconscious_units_content_fts
    ON subconscious_units USING GIN (to_tsvector('english', content))
    WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_subconscious_units_metadata
    ON subconscious_units USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_recmem_tasks_pending
    ON recmem_consolidation_tasks (next_attempt_at, created_at)
    WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_recmem_tasks_in_progress
    ON recmem_consolidation_tasks (started_at)
    WHERE status = 'in_progress';
CREATE INDEX IF NOT EXISTS idx_recmem_tasks_open_create_sources
    ON recmem_consolidation_tasks USING GIN (source_unit_ids)
    WHERE status IN ('pending','in_progress') AND task_type = 'episode_create';
CREATE INDEX IF NOT EXISTS idx_recmem_tasks_status_type
    ON recmem_consolidation_tasks (status, task_type, next_attempt_at);
CREATE INDEX IF NOT EXISTS idx_memory_source_units_source
    ON memory_source_units (subconscious_unit_id);
CREATE INDEX idx_clusters_centroid ON clusters USING hnsw (centroid_embedding vector_cosine_ops);
CREATE INDEX idx_clusters_type ON clusters (cluster_type);
CREATE INDEX idx_episodes_time_range ON episodes USING GIST (time_range);
CREATE INDEX idx_episodes_summary_embedding ON episodes USING hnsw (summary_embedding vector_cosine_ops);
CREATE INDEX idx_episodes_started ON episodes (started_at DESC);
CREATE INDEX idx_neighborhoods_stale ON memory_neighborhoods (is_stale) WHERE is_stale = TRUE;
CREATE INDEX idx_neighborhoods_neighbors ON memory_neighborhoods USING GIN (neighbors);
CREATE INDEX idx_neighborhoods_stale_computed ON memory_neighborhoods (computed_at ASC NULLS FIRST)
    WHERE is_stale = TRUE;
CREATE INDEX idx_memories_worldview_category ON memories ((metadata->>'category'))
    WHERE type = 'worldview';
CREATE INDEX idx_memories_goal_priority ON memories ((metadata->>'priority'))
    WHERE type = 'goal';
CREATE INDEX idx_embedding_cache_created ON embedding_cache (created_at);
CREATE INDEX IF NOT EXISTS idx_worker_instances_status_seen
    ON worker_instances (status, last_seen_at DESC);
CREATE INDEX IF NOT EXISTS idx_worker_instances_mode_seen
    ON worker_instances (mode, last_seen_at DESC);
CREATE INDEX IF NOT EXISTS idx_worker_task_runs_task_started
    ON worker_task_runs (task_type, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_worker_task_runs_status_started
    ON worker_task_runs (status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_worker_task_runs_worker_started
    ON worker_task_runs (worker_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_worker_start_events_mode_started
    ON worker_start_events (mode, instance_name, started_at DESC);
CREATE INDEX idx_consent_log_model_endpoint ON consent_log (provider, model, endpoint);
CREATE UNIQUE INDEX idx_emotional_triggers_pattern ON emotional_triggers (trigger_pattern);
CREATE INDEX idx_emotional_triggers_embedding ON emotional_triggers USING hnsw (trigger_embedding vector_cosine_ops);
CREATE INDEX idx_memory_activation_embedding ON memory_activation USING hnsw (query_embedding vector_cosine_ops);
CREATE INDEX idx_memory_activation_pending ON memory_activation (background_search_pending)
    WHERE background_search_pending = TRUE;
CREATE INDEX idx_memory_activation_pending_started ON memory_activation (background_search_started_at, created_at)
    WHERE background_search_pending = TRUE;
CREATE INDEX idx_memory_activation_expires_at ON memory_activation (expires_at);
CREATE INDEX idx_scheduled_tasks_due ON scheduled_tasks (next_run_at)
    WHERE status = 'active';
CREATE INDEX idx_scheduled_tasks_status ON scheduled_tasks (status);
CREATE INDEX IF NOT EXISTS idx_skill_improvement_proposals_status_created
    ON skill_improvement_proposals (status, created_at DESC);
