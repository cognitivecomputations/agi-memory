-- Phase 6 prerequisite: typed, execution-context-owned goal provenance.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS goal_origin goal_source;

-- Old initialization goals came directly from the user's init flow. For all
-- other legacy goals, preserve a valid historical origin/source when one
-- exists and use the conservative autonomous default otherwise.
UPDATE memories
SET goal_origin = CASE
    WHEN metadata->>'origin' = 'initialization'
        THEN 'user_request'::goal_source
    WHEN metadata->>'origin' IN (
        'curiosity', 'user_request', 'identity', 'derived', 'external'
    ) THEN (metadata->>'origin')::goal_source
    WHEN metadata->>'source' IN (
        'curiosity', 'user_request', 'identity', 'derived', 'external'
    ) THEN (metadata->>'source')::goal_source
    ELSE 'derived'::goal_source
END
WHERE type = 'goal'::memory_type
  AND goal_origin IS NULL;

UPDATE memories
SET goal_origin = NULL
WHERE type <> 'goal'::memory_type
  AND goal_origin IS NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'public.memories'::regclass
          AND conname = 'memories_goal_origin_scope'
    ) THEN
        ALTER TABLE memories
            ADD CONSTRAINT memories_goal_origin_scope CHECK (
                (type = 'goal' AND goal_origin IS NOT NULL)
                OR (type <> 'goal' AND goal_origin IS NULL)
            ) NOT VALID;
    END IF;
END;
$$;

ALTER TABLE memories VALIDATE CONSTRAINT memories_goal_origin_scope;

CREATE OR REPLACE FUNCTION normalize_memory_goal_origin()
RETURNS TRIGGER AS $$
DECLARE
    candidate TEXT;
BEGIN
    IF NEW.type = 'goal'::memory_type THEN
        IF NEW.metadata->>'origin' = 'initialization' THEN
            NEW.goal_origin := 'user_request'::goal_source;
        ELSIF NEW.goal_origin IS NULL THEN
            candidate := NULLIF(NEW.metadata->>'origin', '');
            IF candidate = 'initialization' THEN
                candidate := 'user_request';
            ELSIF candidate NOT IN (
                'curiosity', 'user_request', 'identity', 'derived', 'external'
            ) THEN
                candidate := NULLIF(NEW.metadata->>'source', '');
            END IF;
            IF candidate NOT IN (
                'curiosity', 'user_request', 'identity', 'derived', 'external'
            ) THEN
                candidate := 'derived';
            END IF;
            NEW.goal_origin := candidate::goal_source;
        END IF;
    ELSE
        NEW.goal_origin := NULL;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_memory_goal_origin ON memories;
CREATE TRIGGER trg_memory_goal_origin
BEFORE INSERT OR UPDATE OF type, goal_origin, metadata ON memories
FOR EACH ROW
EXECUTE FUNCTION normalize_memory_goal_origin();

CREATE OR REPLACE FUNCTION create_goal(
    p_title TEXT,
    p_description TEXT,
    p_source goal_source,
    p_priority goal_priority,
    p_parent_id UUID,
    p_due_at TIMESTAMPTZ,
    p_origin goal_source
)
RETURNS UUID AS $$
DECLARE
    new_goal_id UUID;
    active_count INT;
    max_active INT;
    goal_embedding vector;
    goal_metadata JSONB;
    resolved_origin goal_source := COALESCE(p_origin, 'derived'::goal_source);
BEGIN
    SELECT id INTO new_goal_id
    FROM memories
    WHERE type = 'goal' AND content = p_title AND status = 'active'
    LIMIT 1;
    IF new_goal_id IS NOT NULL THEN
        IF resolved_origin = 'user_request'::goal_source THEN
            UPDATE memories
            SET goal_origin = 'user_request'::goal_source,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = new_goal_id
              AND goal_origin IS DISTINCT FROM 'user_request'::goal_source;
        END IF;
        RETURN new_goal_id;
    END IF;

    IF p_priority = 'active' THEN
        SELECT COUNT(*) INTO active_count
        FROM memories
        WHERE type = 'goal' AND status = 'active' AND metadata->>'priority' = 'active';
        max_active := get_config_int('heartbeat.max_active_goals');
        IF active_count >= max_active THEN
            p_priority := 'queued';
        END IF;
    END IF;

    goal_embedding := (get_embedding(ARRAY[p_title]))[1];
    goal_metadata := jsonb_build_object(
        'title', p_title,
        'description', p_description,
        'priority', p_priority::text,
        'source', p_source::text,
        'due_at', p_due_at,
        'progress', '[]'::jsonb,
        'blocked_by', NULL,
        'emotional_valence', 0.0,
        'last_touched', CURRENT_TIMESTAMP,
        'parent_goal_id', p_parent_id
    );
    INSERT INTO memories (type, goal_origin, content, embedding, importance, metadata)
    VALUES (
        'goal'::memory_type,
        resolved_origin,
        p_title,
        goal_embedding,
        0.7,
        goal_metadata
    )
    RETURNING id INTO new_goal_id;

    BEGIN
        PERFORM ensure_goals_root();
        PERFORM sync_goal_node(new_goal_id);
        EXECUTE format('SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
            MATCH (root:GoalsRoot {key: ''goals''})
            MATCH (g:GoalNode {goal_id: %L})
            CREATE (root)-[:CONTAINS {priority: %L}]->(g)
            RETURN g
        $q$) as (result ag_catalog.agtype)', new_goal_id, p_priority::text);
        PERFORM upsert_memory_edge('goals_root', 'goals', 'CONTAINS', 'goal', new_goal_id::text,
                                   1.0, NULL, NULL, jsonb_build_object('priority', p_priority::text));
        IF p_parent_id IS NOT NULL THEN
            PERFORM link_goal_subgoal(p_parent_id, new_goal_id);
        END IF;
    EXCEPTION WHEN OTHERS THEN
        NULL;
    END;

    RETURN new_goal_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION create_goal(
    p_title TEXT,
    p_description TEXT DEFAULT NULL,
    p_source goal_source DEFAULT 'curiosity',
    p_priority goal_priority DEFAULT 'queued',
    p_parent_id UUID DEFAULT NULL,
    p_due_at TIMESTAMPTZ DEFAULT NULL
)
RETURNS UUID AS $$
    SELECT create_goal(
        p_title,
        p_description,
        COALESCE(p_source, 'curiosity'::goal_source),
        COALESCE(p_priority, 'queued'::goal_priority),
        p_parent_id,
        p_due_at,
        'derived'::goal_source
    );
$$ LANGUAGE sql;

CREATE OR REPLACE FUNCTION execute_goals_tool(
    p_args JSONB,
    p_goal_origin goal_source
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    action TEXT := COALESCE(p_args->>'action', '');
    title TEXT;
    goal_id UUID;
    priority TEXT;
    source_value TEXT;
    snapshot JSONB;
    rows_json JSONB;
BEGIN
    IF action NOT IN ('create', 'update_priority', 'add_progress', 'list') THEN
        RETURN tool_error(format('Invalid action %L', action), 'invalid_params');
    END IF;
    IF action = 'create' THEN
        title := NULLIF(btrim(COALESCE(p_args->>'title', '')), '');
        IF title IS NULL THEN
            RETURN tool_error('Title is required for create', 'invalid_params');
        END IF;
        priority := COALESCE(NULLIF(p_args->>'priority', ''), 'queued');
        IF priority NOT IN ('active', 'queued', 'backburner', 'completed', 'abandoned') THEN
            priority := 'queued';
        END IF;
        source_value := COALESCE(NULLIF(p_args->>'source', ''), 'curiosity');
        IF source_value NOT IN ('curiosity', 'user_request', 'identity', 'derived', 'external') THEN
            source_value := 'curiosity';
        END IF;
        goal_id := create_goal(
            title,
            p_args->>'description',
            source_value::goal_source,
            priority::goal_priority,
            NULL,
            NULL,
            COALESCE(p_goal_origin, 'derived'::goal_source)
        );
        RETURN tool_success(
            jsonb_build_object(
                'goal_id', goal_id::text,
                'title', title,
                'priority', priority,
                'origin', COALESCE(p_goal_origin, 'derived'::goal_source)::text
            ),
            format('Created goal: %s (%s)', title, priority)
        );
    ELSIF action = 'update_priority' THEN
        priority := COALESCE(p_args->>'priority', '');
        IF NULLIF(p_args->>'goal_id', '') IS NULL THEN
            RETURN tool_error('goal_id is required for update_priority', 'invalid_params');
        END IF;
        IF priority NOT IN ('active', 'queued', 'backburner', 'completed', 'abandoned') THEN
            RETURN tool_error(format('Invalid priority %L', priority), 'invalid_params');
        END IF;
        BEGIN
            goal_id := (p_args->>'goal_id')::uuid;
        EXCEPTION WHEN invalid_text_representation THEN
            RETURN tool_error(format('Invalid goal_id: %s', p_args->>'goal_id'), 'invalid_params');
        END;
        PERFORM change_goal_priority(goal_id, priority::goal_priority, COALESCE(p_args->>'reason', ''));
        RETURN tool_success(
            jsonb_build_object(
                'goal_id', goal_id::text,
                'new_priority', priority,
                'reason', COALESCE(p_args->>'reason', '')
            ),
            format('Updated goal %s... to %s', left(goal_id::text, 8), priority)
        );
    ELSIF action = 'add_progress' THEN
        IF NULLIF(p_args->>'goal_id', '') IS NULL THEN
            RETURN tool_error('goal_id is required for add_progress', 'invalid_params');
        END IF;
        IF NULLIF(btrim(COALESCE(p_args->>'note', '')), '') IS NULL THEN
            RETURN tool_error('note is required for add_progress', 'invalid_params');
        END IF;
        BEGIN
            goal_id := (p_args->>'goal_id')::uuid;
        EXCEPTION WHEN invalid_text_representation THEN
            RETURN tool_error(format('Invalid goal_id: %s', p_args->>'goal_id'), 'invalid_params');
        END;
        PERFORM add_goal_progress(goal_id, p_args->>'note');
        RETURN tool_success(
            jsonb_build_object('goal_id', goal_id::text, 'note', p_args->>'note'),
            format('Added progress to goal %s...', left(goal_id::text, 8))
        );
    ELSE
        priority := NULLIF(p_args->>'priority', '');
        IF priority IS NOT NULL AND priority IN ('active', 'queued', 'backburner', 'completed', 'abandoned') THEN
            SELECT COALESCE(jsonb_agg(to_jsonb(g)), '[]'::jsonb) INTO rows_json
            FROM get_goals_by_priority(priority::goal_priority) g;
            RETURN tool_success(jsonb_build_object('goals', rows_json, 'count', jsonb_array_length(rows_json)));
        END IF;
        snapshot := get_goals_snapshot();
        RETURN tool_success(COALESCE(snapshot, '{}'::jsonb));
    END IF;
EXCEPTION WHEN OTHERS THEN
    RETURN tool_error(SQLERRM);
END;
$$;

CREATE OR REPLACE FUNCTION execute_goals_tool(p_args JSONB)
RETURNS JSONB
LANGUAGE sql
AS $$
    SELECT execute_goals_tool(p_args, 'derived'::goal_source);
$$;

CREATE OR REPLACE FUNCTION get_goals_snapshot()
RETURNS JSONB AS $$
DECLARE
    active_goals JSONB;
    queued_goals JSONB;
    issues JSONB;
    stale_days FLOAT;
BEGIN
    stale_days := get_config_float('heartbeat.goal_stale_days');
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', id,
        'title', metadata->>'title',
        'description', metadata->>'description',
        'origin', goal_origin::text,
        'due_at', (metadata->>'due_at')::timestamptz,
        'last_touched', (metadata->>'last_touched')::timestamptz,
        'progress_count', jsonb_array_length(COALESCE(metadata->'progress', '[]'::jsonb)),
        'blocked_by', metadata->'blocked_by'
    )), '[]'::jsonb)
    INTO active_goals
    FROM memories
    WHERE type = 'goal' AND status = 'active' AND metadata->>'priority' = 'active';

    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', id,
        'title', metadata->>'title',
        'source', metadata->>'source',
        'origin', goal_origin::text,
        'due_at', (metadata->>'due_at')::timestamptz
    )), '[]'::jsonb)
    INTO queued_goals
    FROM (
        SELECT * FROM memories
        WHERE type = 'goal' AND status = 'active' AND metadata->>'priority' = 'queued'
        ORDER BY (metadata->>'due_at')::timestamptz NULLS LAST,
                 (metadata->>'last_touched')::timestamptz DESC
        LIMIT 5
    ) q;

    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'goal_id', id,
        'title', metadata->>'title',
        'issue', CASE
            WHEN metadata->'blocked_by' IS NOT NULL
                 AND metadata->'blocked_by' <> 'null'::jsonb THEN 'blocked'
            WHEN (metadata->>'due_at')::timestamptz IS NOT NULL
                 AND (metadata->>'due_at')::timestamptz < CURRENT_TIMESTAMP THEN 'overdue'
            WHEN (metadata->>'last_touched')::timestamptz
                 < CURRENT_TIMESTAMP - (stale_days || ' days')::INTERVAL THEN 'stale'
            ELSE 'unknown'
        END,
        'due_at', (metadata->>'due_at')::timestamptz,
        'days_since_touched', EXTRACT(
            EPOCH FROM (CURRENT_TIMESTAMP - (metadata->>'last_touched')::timestamptz)
        ) / 86400
    )), '[]'::jsonb)
    INTO issues
    FROM memories
    WHERE type = 'goal'
      AND status = 'active'
      AND metadata->>'priority' = 'active'
      AND (
          (metadata->'blocked_by' IS NOT NULL AND metadata->'blocked_by' <> 'null'::jsonb)
          OR ((metadata->>'due_at')::timestamptz IS NOT NULL
              AND (metadata->>'due_at')::timestamptz < CURRENT_TIMESTAMP)
          OR (metadata->>'last_touched')::timestamptz
              < CURRENT_TIMESTAMP - (stale_days || ' days')::INTERVAL
      );

    RETURN jsonb_build_object(
        'active', active_goals,
        'queued', queued_goals,
        'issues', issues,
        'counts', jsonb_build_object(
            'active', (SELECT COUNT(*) FROM memories WHERE type = 'goal' AND status = 'active' AND metadata->>'priority' = 'active'),
            'queued', (SELECT COUNT(*) FROM memories WHERE type = 'goal' AND status = 'active' AND metadata->>'priority' = 'queued'),
            'backburner', (SELECT COUNT(*) FROM memories WHERE type = 'goal' AND status = 'active' AND metadata->>'priority' = 'backburner')
        )
    );
END;
$$ LANGUAGE plpgsql;

DROP FUNCTION IF EXISTS get_goals_by_priority(goal_priority);
CREATE OR REPLACE FUNCTION get_goals_by_priority(
    p_priority goal_priority DEFAULT NULL
) RETURNS TABLE (
    id UUID,
    title TEXT,
    description TEXT,
    priority TEXT,
    source TEXT,
    due_at TIMESTAMPTZ,
    last_touched TIMESTAMPTZ,
    progress JSONB,
    blocked_by JSONB,
    emotional_valence FLOAT,
    created_at TIMESTAMPTZ,
    origin TEXT
) AS $$
BEGIN
    IF p_priority IS NULL THEN
        RETURN QUERY
        SELECT
            m.id,
            m.metadata->>'title',
            m.metadata->>'description',
            m.metadata->>'priority',
            m.metadata->>'source',
            (m.metadata->>'due_at')::timestamptz,
            (m.metadata->>'last_touched')::timestamptz,
            m.metadata->'progress',
            m.metadata->'blocked_by',
            (m.metadata->>'emotional_valence')::float,
            m.created_at,
            m.goal_origin::text
        FROM memories m
        WHERE m.type = 'goal'
          AND m.status = 'active'
          AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
          AND m.metadata->>'priority' IN ('active', 'queued')
        ORDER BY m.metadata->>'priority',
                 (m.metadata->>'last_touched')::timestamptz DESC;
    ELSE
        RETURN QUERY
        SELECT
            m.id,
            m.metadata->>'title',
            m.metadata->>'description',
            m.metadata->>'priority',
            m.metadata->>'source',
            (m.metadata->>'due_at')::timestamptz,
            (m.metadata->>'last_touched')::timestamptz,
            m.metadata->'progress',
            m.metadata->'blocked_by',
            (m.metadata->>'emotional_valence')::float,
            m.created_at,
            m.goal_origin::text
        FROM memories m
        WHERE m.type = 'goal'
          AND m.status = 'active'
          AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
          AND m.metadata->>'priority' = p_priority::text
        ORDER BY (m.metadata->>'last_touched')::timestamptz DESC;
    END IF;
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE VIEW active_goals AS
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

CREATE OR REPLACE VIEW goal_backlog AS
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
WHERE type = 'goal'
  AND status = 'active'
  AND metadata->>'priority' IN ('active', 'queued', 'backburner')
GROUP BY metadata->>'priority';
