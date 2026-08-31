-- Hexis schema: views.
SET search_path = public, ag_catalog, "$user";
CREATE VIEW memory_health AS
SELECT 
    type,
    COUNT(*) as total_memories,
    AVG(importance) as avg_importance,
    AVG(access_count) as avg_access_count,
    COUNT(*) FILTER (WHERE last_accessed > CURRENT_TIMESTAMP - INTERVAL '1 day') as accessed_last_day,
    AVG(calculate_relevance(importance, decay_rate, created_at, last_accessed)) as avg_relevance
FROM memories
WHERE status = 'active'
GROUP BY type;
CREATE VIEW cluster_insights AS
SELECT
    mc.id,
    mc.name,
    mc.cluster_type,
    (SELECT COUNT(*) FROM get_cluster_members_graph(mc.id)) as memory_count
FROM clusters mc
ORDER BY memory_count DESC, mc.name ASC;
CREATE VIEW episode_summary AS
SELECT
    e.id,
    e.started_at,
    e.ended_at,
    e.metadata->>'episode_type' as episode_type,
    e.summary,
    (SELECT COUNT(*) FROM find_episode_memories_graph(e.id)) as memory_count,
    (SELECT MIN(m.created_at) FROM find_episode_memories_graph(e.id) fem JOIN memories m ON fem.memory_id = m.id) as first_memory_at,
    (SELECT MAX(m.created_at) FROM find_episode_memories_graph(e.id) fem JOIN memories m ON fem.memory_id = m.id) as last_memory_at
FROM episodes e
ORDER BY e.started_at DESC;
CREATE VIEW stale_neighborhoods AS
SELECT 
    mn.memory_id,
    m.content,
    m.type,
    mn.computed_at,
    AGE(CURRENT_TIMESTAMP, mn.computed_at) as staleness
FROM memory_neighborhoods mn
JOIN memories m ON mn.memory_id = m.id
WHERE mn.is_stale = TRUE
ORDER BY mn.computed_at ASC;
CREATE OR REPLACE VIEW drive_status AS
SELECT
    name,
    current_level,
    baseline,
    urgency_threshold,
    (current_level >= urgency_threshold) as is_urgent,
    ROUND((current_level / NULLIF(urgency_threshold, 0) * 100)::numeric, 1) as urgency_percent,
    last_satisfied,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_satisfied)) / 3600 as hours_since_satisfied
FROM drives
ORDER BY current_level DESC;
CREATE VIEW heartbeat_state AS
SELECT
    1 as id,
    COALESCE((s.value->>'current_energy')::float, 10) as current_energy,
    (s.value->>'last_heartbeat_at')::timestamptz as last_heartbeat_at,
    (s.value->>'next_heartbeat_at')::timestamptz as next_heartbeat_at,
    COALESCE((s.value->>'heartbeat_count')::int, 0) as heartbeat_count,
    (s.value->>'last_user_contact')::timestamptz as last_user_contact,
    COALESCE(s.value->'affective_state', '{}'::jsonb) as affective_state,
    COALESCE((s.value->>'is_paused')::boolean, false) as is_paused,
    COALESCE((s.value->>'init_stage')::init_stage, 'not_started'::init_stage) as init_stage,
    COALESCE(s.value->'init_data', '{}'::jsonb) as init_data,
    (s.value->>'init_started_at')::timestamptz as init_started_at,
    (s.value->>'init_completed_at')::timestamptz as init_completed_at,
    NULLIF(s.value->>'active_heartbeat_id', '')::uuid as active_heartbeat_id,
    (s.value->>'active_heartbeat_number')::int as active_heartbeat_number,
    COALESCE(s.value->'active_actions', '[]'::jsonb) as active_actions,
    NULLIF(s.value->>'active_reasoning', '') as active_reasoning,
    s.updated_at
FROM state s
WHERE s.key = 'heartbeat_state';
CREATE VIEW maintenance_state AS
SELECT
    1 as id,
    (s.value->>'last_maintenance_at')::timestamptz as last_maintenance_at,
    (s.value->>'last_subconscious_run_at')::timestamptz as last_subconscious_run_at,
    (s.value->>'last_subconscious_heartbeat')::int as last_subconscious_heartbeat,
    COALESCE((s.value->>'is_paused')::boolean, false) as is_paused,
    s.updated_at
FROM state s
WHERE s.key = 'maintenance_state';
CREATE VIEW recmem_state AS
SELECT
    1 as id,
    (s.value->>'last_sweep_at')::timestamptz as last_sweep_at,
    COALESCE(s.value->'last_sweep_result', '{}'::jsonb) as last_sweep_result,
    s.updated_at
FROM state s
WHERE s.key = 'recmem_state';
CREATE VIEW active_goals AS
SELECT
    id,
    metadata->>'title' as title,
    metadata->>'description' as description,
    metadata->>'source' as source,
    (metadata->>'last_touched')::timestamptz as last_touched,
    jsonb_array_length(COALESCE(metadata->'progress', '[]'::jsonb)) as progress_count,
    (metadata->'blocked_by' IS NOT NULL AND metadata->'blocked_by' <> 'null'::jsonb) as is_blocked,
    created_at,
    goal_origin::text as origin
FROM memories
WHERE type = 'goal' AND status = 'active' AND metadata->>'priority' = 'active'
ORDER BY (metadata->>'last_touched')::timestamptz DESC;
CREATE VIEW goal_backlog AS
SELECT
    metadata->>'priority' as priority,
    COUNT(*) as count,
    jsonb_agg(jsonb_build_object(
        'id', id,
        'title', metadata->>'title',
        'source', metadata->>'source',
        'origin', goal_origin::text
    ) ORDER BY (metadata->>'last_touched')::timestamptz DESC) as goals
FROM memories
WHERE type = 'goal' AND status = 'active' AND metadata->>'priority' IN ('active', 'queued', 'backburner')
GROUP BY metadata->>'priority';
CREATE VIEW heartbeat_health AS
SELECT
    (SELECT heartbeat_count FROM heartbeat_state WHERE id = 1) as total_heartbeats,
    (SELECT current_energy FROM heartbeat_state WHERE id = 1) as current_energy,
    (SELECT last_heartbeat_at FROM heartbeat_state WHERE id = 1) as last_heartbeat,
    (SELECT next_heartbeat_at FROM heartbeat_state WHERE id = 1) as next_heartbeat,
    (SELECT is_paused FROM heartbeat_state WHERE id = 1) as is_paused,
    (SELECT COUNT(*) FROM memories WHERE type = 'goal' AND status = 'active' AND metadata->>'priority' = 'active') as active_goals,
    (SELECT COUNT(*) FROM memories WHERE type = 'goal' AND status = 'active' AND metadata->>'priority' = 'queued') as queued_goals,
    0::int as pending_calls,
    NULL::float as avg_energy_delta_24h,
    0::int as reach_outs_24h;
CREATE VIEW recent_heartbeats AS
SELECT
    m.id as memory_id,
    NULLIF(m.metadata#>>'{context,heartbeat_id}', '')::uuid as heartbeat_id,
    (m.metadata#>>'{context,heartbeat_number}')::int as heartbeat_number,
    COALESCE((m.metadata->>'event_time')::timestamptz, m.created_at) as started_at,
    COALESCE((m.metadata->>'event_time')::timestamptz, m.created_at) as ended_at,
    NULL::float as energy_start,
    NULL::float as energy_end,
    jsonb_array_length(COALESCE(m.metadata#>'{context,actions_taken}', '[]'::jsonb)) as action_count,
    m.content as narrative,
    NULLIF(m.metadata->>'emotional_valence', '')::float as emotional_valence
FROM memories m
WHERE m.type = 'episodic'
  AND m.metadata#>>'{context,heartbeat_id}' IS NOT NULL
ORDER BY COALESCE((m.metadata->>'event_time')::timestamptz, m.created_at) DESC
LIMIT 20;
CREATE OR REPLACE VIEW boundary_status AS
SELECT
    content as name,
    COALESCE(metadata->>'subcategory', 'ethical') as boundary_type,
    metadata->>'response_type' as response_type,
    importance,
    1.0 - COALESCE((metadata->>'stability')::float, 0.9) as flexibility,
    0::bigint as times_triggered
FROM memories
WHERE type = 'worldview' AND metadata->>'category' = 'boundary' AND status = 'active'
ORDER BY importance DESC;
CREATE OR REPLACE VIEW current_emotional_state AS
SELECT
    gen_random_uuid() as id,
    COALESCE((affective_state->>'updated_at')::timestamptz, updated_at) as recorded_at,
    (affective_state->>'valence')::float as valence,
    (affective_state->>'arousal')::float as arousal,
    COALESCE((affective_state->>'dominance')::float, 0.5::float) as dominance,
    affective_state->>'primary_emotion' as primary_emotion,
    NULL::text[] as secondary_emotions,
    (affective_state->>'intensity')::float as intensity,
    NULL::text as triggered_by_type,
    NULL::uuid as triggered_by_id,
    NULL::text as trigger_description,
    NULL::uuid as heartbeat_id
FROM heartbeat_state
WHERE id = 1;
CREATE OR REPLACE VIEW emotional_trend AS
WITH base AS (
    SELECT
        date_trunc(
            'hour',
            COALESCE((m.metadata->>'event_time')::timestamptz, m.created_at)
        ) as hour,
        NULLIF(m.metadata->>'emotional_valence', '')::float as emotional_valence,
        NULLIF(m.metadata#>>'{emotional_context,arousal}', '')::float as emotional_arousal,
        NULLIF(m.metadata#>>'{emotional_context,primary_emotion}', '') as emotional_primary_emotion
    FROM memories m
    WHERE m.type = 'episodic'
      AND m.metadata#>>'{context,heartbeat_id}' IS NOT NULL
      AND COALESCE((m.metadata->>'event_time')::timestamptz, m.created_at)
            > CURRENT_TIMESTAMP - INTERVAL '24 hours'
      AND m.metadata->>'emotional_valence' IS NOT NULL
)
SELECT
    base.hour,
    AVG(base.emotional_valence) as avg_valence,
    COALESCE(AVG(base.emotional_arousal), 0.5)::float as avg_arousal,
    COALESCE(
        (
            SELECT b2.emotional_primary_emotion
            FROM base b2
            WHERE b2.hour = base.hour
              AND b2.emotional_primary_emotion IS NOT NULL
            GROUP BY b2.emotional_primary_emotion
            ORDER BY COUNT(*) DESC
            LIMIT 1
        ),
        CASE
            WHEN AVG(base.emotional_valence) > 0.2 THEN 'content'
            WHEN AVG(base.emotional_valence) < -0.2 THEN 'down'
            ELSE 'neutral'
        END
    ) as dominant_emotion,
    COUNT(*) as state_changes
FROM base
GROUP BY base.hour
ORDER BY base.hour DESC;
CREATE OR REPLACE VIEW cognitive_health AS
SELECT
    (SELECT current_energy FROM heartbeat_state WHERE id = 1) as energy,
    get_config_float('heartbeat.max_energy') as max_energy,
    (SELECT COUNT(*) FROM drives WHERE current_level >= urgency_threshold) as urgent_drives,
    (SELECT AVG(current_level) FROM drives) as avg_drive_level,
    (SELECT COUNT(*) FROM memories WHERE type = 'goal' AND status = 'active' AND metadata->>'priority' = 'active') as active_goals,
    (SELECT COUNT(*) FROM memories WHERE type = 'goal' AND status = 'active' AND metadata->>'priority' = 'active'
        AND metadata->'blocked_by' IS NOT NULL AND metadata->'blocked_by' <> 'null'::jsonb) as blocked_goals,
    (SELECT COUNT(*) FROM memories WHERE status = 'active') as total_memories,
    (SELECT COUNT(*) FROM memory_neighborhoods WHERE is_stale = TRUE) as stale_neighborhoods,
    (SELECT valence FROM current_emotional_state) as current_valence,
    (SELECT primary_emotion FROM current_emotional_state) as current_emotion,
    (
        SELECT COUNT(*)
        FROM memories m
        WHERE m.type = 'episodic'
          AND m.metadata#>>'{context,heartbeat_id}' IS NOT NULL
          AND COALESCE((m.metadata->>'event_time')::timestamptz, m.created_at)
                > CURRENT_TIMESTAMP - INTERVAL '24 hours'
    ) as heartbeats_24h,
    0::bigint as pending_calls,
    0::bigint as relationships_discovered_24h;
CREATE OR REPLACE VIEW recmem_health AS
SELECT
    (SELECT COUNT(*) FROM subconscious_units WHERE status = 'active') AS active_raw_units,
    (SELECT COUNT(*) FROM subconscious_units WHERE embedding_status = 'pending') AS pending_embeddings,
    (SELECT COUNT(*) FROM subconscious_units WHERE embedding_status = 'failed') AS failed_embeddings,
    (SELECT COUNT(*) FROM subconscious_units WHERE route_status = 'unrouted' AND embedding_status = 'embedded') AS pending_routes,
    (SELECT COUNT(*) FROM subconscious_units WHERE route_status = 'route_failed') AS failed_routes,
    (SELECT COUNT(*) FROM recmem_consolidation_tasks WHERE status = 'pending' AND next_attempt_at <= CURRENT_TIMESTAMP) AS pending_tasks,
    (SELECT COUNT(*) FROM recmem_consolidation_tasks WHERE status = 'failed') AS failed_tasks,
    (SELECT COUNT(*) FROM recmem_consolidation_tasks WHERE status = 'dropped') AS dropped_tasks,
    (SELECT AVG(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at))) FROM subconscious_units WHERE embedding_status = 'pending') AS avg_pending_embedding_age_s,
    (SELECT AVG(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - embedded_at))) FROM subconscious_units WHERE embedding_status = 'embedded' AND route_status = 'unrouted') AS avg_pending_route_age_s;
CREATE OR REPLACE VIEW worker_tasks AS
SELECT
    'heartbeat'::text AS task_type,
    CASE WHEN should_run_heartbeat() THEN 1 ELSE 0 END AS pending_count,
    'Run heartbeat if due'::text AS description
UNION ALL
SELECT
    'subconscious_maintenance'::text AS task_type,
    CASE WHEN should_run_maintenance() THEN 1 ELSE 0 END AS pending_count,
    'Run subconscious maintenance tick (consolidate + prune)'::text AS description
UNION ALL
SELECT
    'recmem_embedding'::text AS task_type,
    COUNT(*)::int AS pending_count,
    'Embed pending RecMem raw units'::text AS description
FROM subconscious_units
WHERE embedding_status = 'pending'
UNION ALL
SELECT
    'recmem_routing'::text AS task_type,
    COUNT(*)::int AS pending_count,
    'Route embedded RecMem raw units'::text AS description
FROM subconscious_units
WHERE embedding_status = 'embedded'
  AND route_status = 'unrouted'
UNION ALL
SELECT
    'recmem_consolidation'::text AS task_type,
    COUNT(*)::int AS pending_count,
    'Process pending RecMem consolidation tasks'::text AS description
FROM recmem_consolidation_tasks
WHERE status = 'pending'
  AND next_attempt_at <= CURRENT_TIMESTAMP
UNION ALL
SELECT
    'recmem_sweep'::text AS task_type,
    CASE WHEN should_run_recmem_sweep() THEN 1 ELSE 0 END AS pending_count,
    'Re-route old RecMem raw-only units'::text AS description;

CREATE OR REPLACE VIEW worker_runtime_status AS
SELECT
    id,
    mode,
    instance_name,
    process_id,
    host_name,
    started_at,
    last_seen_at,
    last_success_at,
    last_error_at,
    stopped_at,
    status,
    current_task_type,
    current_task_run_id,
    build_id,
    metadata,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_seen_at))::int AS last_seen_age_s,
    (
        status IN ('starting', 'running', 'stopping')
        AND last_seen_at < CURRENT_TIMESTAMP - INTERVAL '3 minutes'
    ) AS is_stale
FROM worker_instances
ORDER BY last_seen_at DESC;

CREATE OR REPLACE VIEW worker_start_storm_status AS
SELECT
    mode,
    instance_name,
    COUNT(*) FILTER (WHERE started_at >= CURRENT_TIMESTAMP - INTERVAL '2 minutes')::int AS starts_last_2m,
    COUNT(*) FILTER (WHERE started_at >= CURRENT_TIMESTAMP - INTERVAL '10 minutes')::int AS starts_last_10m,
    MAX(started_at) AS latest_start_at,
    (
        COUNT(*) FILTER (WHERE started_at >= CURRENT_TIMESTAMP - INTERVAL '2 minutes') > 5
    ) AS is_storming
FROM worker_start_events
GROUP BY mode, instance_name
ORDER BY latest_start_at DESC;

CREATE OR REPLACE VIEW worker_task_status AS
WITH task_catalog AS (
    SELECT task_type, max(description) AS description
    FROM (
        SELECT task_type, description FROM worker_tasks
        UNION ALL
        SELECT DISTINCT task_type, NULL::text AS description FROM worker_task_runs
    ) catalog
    GROUP BY task_type
)
SELECT
    tc.task_type,
    COALESCE(wt.pending_count, 0)::int AS pending_count,
    tc.description,
    latest.status AS latest_status,
    latest.started_at AS latest_started_at,
    latest.finished_at AS latest_finished_at,
    latest.result AS latest_result,
    latest.error AS latest_error,
    success.finished_at AS last_success_at,
    COALESCE(failures.failures_since_success, 0)::int AS failures_since_success,
    COALESCE(running.running_count, 0)::int AS running_count
FROM task_catalog tc
LEFT JOIN worker_tasks wt ON wt.task_type = tc.task_type
LEFT JOIN LATERAL (
    SELECT status, started_at, finished_at, result, error
    FROM worker_task_runs wr
    WHERE wr.task_type = tc.task_type
    ORDER BY wr.started_at DESC
    LIMIT 1
) latest ON true
LEFT JOIN LATERAL (
    SELECT finished_at
    FROM worker_task_runs wr
    WHERE wr.task_type = tc.task_type
      AND wr.status = 'completed'
    ORDER BY wr.finished_at DESC NULLS LAST
    LIMIT 1
) success ON true
LEFT JOIN LATERAL (
    SELECT COUNT(*)::int AS failures_since_success
    FROM worker_task_runs wr
    WHERE wr.task_type = tc.task_type
      AND wr.status = 'failed'
      AND (success.finished_at IS NULL OR wr.started_at > success.finished_at)
) failures ON true
LEFT JOIN LATERAL (
    SELECT COUNT(*)::int AS running_count
    FROM worker_task_runs wr
    WHERE wr.task_type = tc.task_type
      AND wr.status = 'running'
) running ON true
ORDER BY tc.task_type;
