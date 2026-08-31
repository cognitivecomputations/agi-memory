-- Hexis schema: goal functions.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

CREATE OR REPLACE FUNCTION touch_goal(p_goal_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE memories
    SET metadata = jsonb_set(metadata, '{last_touched}', to_jsonb(CURRENT_TIMESTAMP)),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_goal_id AND type = 'goal';
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION add_goal_progress(p_goal_id UUID, p_note TEXT)
RETURNS VOID AS $$
BEGIN
    UPDATE memories
    SET metadata = jsonb_set(
            jsonb_set(metadata, '{last_touched}', to_jsonb(CURRENT_TIMESTAMP)),
            '{progress}',
            COALESCE(metadata->'progress', '[]'::jsonb) || jsonb_build_array(jsonb_build_object(
                'timestamp', CURRENT_TIMESTAMP,
                'note', p_note
            ))
        ),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_goal_id AND type = 'goal';
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION change_goal_priority(
    p_goal_id UUID,
    p_new_priority goal_priority,
    p_reason TEXT DEFAULT NULL
)
RETURNS VOID AS $$
DECLARE
    old_priority TEXT;
BEGIN
    SELECT metadata->>'priority' INTO old_priority
    FROM memories WHERE id = p_goal_id AND type = 'goal';

    IF old_priority IS NULL THEN
        RAISE NOTICE 'Goal % not found', p_goal_id;
        RETURN;
    END IF;

    UPDATE memories
    SET metadata = metadata
        || jsonb_build_object('priority', p_new_priority::text)
        || jsonb_build_object('last_touched', CURRENT_TIMESTAMP)
        || CASE WHEN p_new_priority::text = 'completed'
                THEN jsonb_build_object('completed_at', CURRENT_TIMESTAMP)
                ELSE '{}'::jsonb END
        || CASE WHEN p_new_priority::text = 'abandoned'
                THEN jsonb_build_object('abandoned_at', CURRENT_TIMESTAMP, 'abandonment_reason', p_reason)
                ELSE '{}'::jsonb END,
        updated_at = CURRENT_TIMESTAMP,
        status = CASE WHEN p_new_priority::text IN ('completed', 'abandoned')
                      THEN 'archived'::memory_status
                      ELSE status END
    WHERE id = p_goal_id AND type = 'goal';
    PERFORM add_goal_progress(p_goal_id,
        format('Priority changed from %s to %s%s',
            old_priority, p_new_priority,
            CASE WHEN p_reason IS NOT NULL THEN ': ' || p_reason ELSE '' END
        )
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION normalize_memory_goal_origin()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.type = 'goal'::memory_type THEN
        IF NEW.metadata->>'origin' = 'initialization' THEN
            NEW.goal_origin := 'user_request'::goal_source;
        ELSIF NEW.goal_origin IS NULL THEN
            -- Raw/legacy inserts have no authenticated execution context.
            -- metadata.source is model-authored description, never authority.
            NEW.goal_origin := 'derived'::goal_source;
        END IF;
    ELSE
        NEW.goal_origin := NULL;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Seven-argument authority-aware implementation. Callers crossing a tool or
-- transport boundary must pass p_origin from trusted execution context.
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
    -- Dedup: return existing active goal with same title
    SELECT id INTO new_goal_id
    FROM memories
    WHERE type = 'goal' AND content = p_title AND status = 'active'
    LIMIT 1;
    IF new_goal_id IS NOT NULL THEN
        -- A direct user assignment upgrades an existing autonomous goal; less
        -- authoritative duplicate observations never downgrade it.
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

-- Compatibility entry point for legacy database callers. It deliberately
-- defaults to derived: metadata.source is descriptive model output, not proof
-- that the user assigned the goal. Trusted callers use the seven-argument form.
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
CREATE OR REPLACE FUNCTION sync_goal_node(p_goal_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
    EXECUTE format('SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
        MERGE (g:GoalNode {goal_id: %L})
        RETURN g
    $q$) as (result ag_catalog.agtype)', p_goal_id);
    RETURN TRUE;
EXCEPTION WHEN OTHERS THEN
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION link_goal_subgoal(p_parent_id UUID, p_child_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
    PERFORM sync_goal_node(p_parent_id);
    PERFORM sync_goal_node(p_child_id);

    EXECUTE format('SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
        MATCH (parent:GoalNode {goal_id: %L})
        MATCH (child:GoalNode {goal_id: %L})
        MERGE (child)-[:SUBGOAL_OF]->(parent)
        RETURN child
    $q$) as (result ag_catalog.agtype)', p_parent_id, p_child_id);
    PERFORM upsert_memory_edge('goal', p_child_id::text, 'SUBGOAL_OF', 'goal', p_parent_id::text, 1.0);
    RETURN TRUE;
EXCEPTION WHEN OTHERS THEN
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION link_goal_to_memory(
    p_goal_id UUID,
    p_memory_id UUID,
    p_link_type TEXT DEFAULT 'evidence'
)
RETURNS BOOLEAN AS $$
DECLARE
    edge_type TEXT;
BEGIN
    edge_type := CASE p_link_type
        WHEN 'origin' THEN 'ORIGINATED_FROM'
        WHEN 'blocker' THEN 'BLOCKS'
        ELSE 'EVIDENCE_FOR'
    END;
    PERFORM sync_goal_node(p_goal_id);
    PERFORM sync_memory_node(p_memory_id);
    IF edge_type = 'ORIGINATED_FROM' THEN
        EXECUTE format('SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
            MATCH (g:GoalNode {goal_id: %L})
            MATCH (m:MemoryNode {memory_id: %L})
            CREATE (g)-[:ORIGINATED_FROM]->(m)
            RETURN g
        $q$) as (result ag_catalog.agtype)', p_goal_id, p_memory_id);
    ELSIF edge_type = 'BLOCKS' THEN
        EXECUTE format('SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
            MATCH (m:MemoryNode {memory_id: %L})
            MATCH (g:GoalNode {goal_id: %L})
            CREATE (m)-[:BLOCKS]->(g)
            RETURN m
        $q$) as (result ag_catalog.agtype)', p_memory_id, p_goal_id);
    ELSE
        EXECUTE format('SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
            MATCH (m:MemoryNode {memory_id: %L})
            MATCH (g:GoalNode {goal_id: %L})
            CREATE (m)-[:EVIDENCE_FOR]->(g)
            RETURN m
        $q$) as (result ag_catalog.agtype)', p_memory_id, p_goal_id);
    END IF;

    IF edge_type = 'ORIGINATED_FROM' THEN
        PERFORM upsert_memory_edge('goal', p_goal_id::text, 'ORIGINATED_FROM', 'memory', p_memory_id::text, 1.0);
    ELSIF edge_type = 'BLOCKS' THEN
        PERFORM upsert_memory_edge('memory', p_memory_id::text, 'BLOCKS', 'goal', p_goal_id::text, 1.0);
    ELSE
        PERFORM upsert_memory_edge('memory', p_memory_id::text, 'EVIDENCE_FOR', 'goal', p_goal_id::text, 1.0);
    END IF;

    RETURN TRUE;
EXCEPTION WHEN OTHERS THEN
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION find_goal_memories(p_goal_id UUID, p_link_type TEXT DEFAULT NULL)
RETURNS TABLE (
    memory_id UUID,
    link_type TEXT
) AS $$
DECLARE
    rec RECORD;
    label_clean TEXT;
BEGIN
    FOR rec IN EXECUTE format(
        'SELECT memory_id::text as memory_id, label::text as label
         FROM ag_catalog.cypher(''memory_graph'', $q$
             MATCH (g:GoalNode {goal_id: %L})-[e]-(m:MemoryNode)
             RETURN m.memory_id, label(e)
         $q$) as (memory_id ag_catalog.agtype, label ag_catalog.agtype)',
        p_goal_id
    )
    LOOP
        label_clean := replace(rec.label, '"', '');

        IF p_link_type IS NULL OR
           (p_link_type = 'origin' AND label_clean = 'ORIGINATED_FROM') OR
           (p_link_type = 'blocker' AND label_clean = 'BLOCKS') OR
           (p_link_type IN ('evidence', 'progress', 'completion') AND label_clean = 'EVIDENCE_FOR') THEN
            memory_id := regexp_replace(rec.memory_id, '[^0-9a-fA-F-]', '', 'g')::uuid;
            link_type := CASE
                WHEN label_clean = 'ORIGINATED_FROM' THEN 'origin'
                WHEN label_clean = 'BLOCKS' THEN 'blocker'
                ELSE 'evidence'
            END;
            RETURN NEXT;
        END IF;
    END LOOP;
EXCEPTION WHEN OTHERS THEN
    RETURN;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION sync_memory_node(p_memory_id UUID)
RETURNS BOOLEAN AS $$
DECLARE
    mem_type TEXT;
BEGIN
    SELECT type::text INTO mem_type FROM memories WHERE id = p_memory_id;
    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;
    EXECUTE format('SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
        MERGE (m:MemoryNode {memory_id: %L})
        SET m.type = %L, m.created_at = %L
        RETURN m
    $q$) as (result ag_catalog.agtype)', p_memory_id, mem_type, CURRENT_TIMESTAMP::text);
    RETURN TRUE;
EXCEPTION WHEN OTHERS THEN
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION sync_cluster_node(p_cluster_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
    EXECUTE format('SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
        MERGE (c:ClusterNode {cluster_id: %L})
        RETURN c
    $q$) as (result ag_catalog.agtype)', p_cluster_id);
    RETURN TRUE;
EXCEPTION WHEN OTHERS THEN
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION link_cluster_relationship(
    p_from_cluster_id UUID,
    p_to_cluster_id UUID,
    p_relationship_type TEXT DEFAULT 'relates',
    p_strength FLOAT DEFAULT 0.5
)
RETURNS BOOLEAN AS $$
DECLARE
    edge_type TEXT;
BEGIN
    edge_type := CASE p_relationship_type
        WHEN 'overlaps' THEN 'CLUSTER_OVERLAPS'
        WHEN 'similar' THEN 'CLUSTER_SIMILAR'
        ELSE 'CLUSTER_RELATES'
    END;
    PERFORM sync_cluster_node(p_from_cluster_id);
    PERFORM sync_cluster_node(p_to_cluster_id);

    EXECUTE format('SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
        MATCH (from:ClusterNode {cluster_id: %L})
        MATCH (to:ClusterNode {cluster_id: %L})
        CREATE (from)-[:%s {strength: %s}]->(to)
        RETURN from
    $q$) as (result ag_catalog.agtype)', p_from_cluster_id, p_to_cluster_id, edge_type, p_strength);

    PERFORM upsert_memory_edge('cluster', p_from_cluster_id::text, edge_type,
                               'cluster', p_to_cluster_id::text, p_strength, NULL, NULL,
                               jsonb_build_object('strength', p_strength));
    RETURN TRUE;
EXCEPTION WHEN OTHERS THEN
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION find_related_clusters(p_cluster_id UUID)
RETURNS TABLE (
    related_cluster_id UUID,
    relationship_type TEXT,
    strength FLOAT
) AS $$
DECLARE
    rec RECORD;
BEGIN
    FOR rec IN EXECUTE format('SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
        MATCH (from:ClusterNode {cluster_id: %L})-[e]->(to:ClusterNode)
        RETURN to.cluster_id, label(e), e.strength
    $q$) as (cluster_id ag_catalog.agtype, label ag_catalog.agtype, str ag_catalog.agtype)', p_cluster_id)
    LOOP
        related_cluster_id := (rec.cluster_id::text)::uuid;
        relationship_type := CASE rec.label::text
            WHEN 'CLUSTER_OVERLAPS' THEN 'overlaps'
            WHEN 'CLUSTER_SIMILAR' THEN 'similar'
            ELSE 'relates'
        END;
        strength := COALESCE((rec.str::text)::float, 0.5);
        RETURN NEXT;
    END LOOP;
EXCEPTION WHEN OTHERS THEN
    RETURN;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION link_memory_to_cluster_graph(
    p_memory_id UUID,
    p_cluster_id UUID,
    p_strength FLOAT DEFAULT 1.0
)
RETURNS BOOLEAN AS $$
BEGIN
    PERFORM sync_memory_node(p_memory_id);
    PERFORM sync_cluster_node(p_cluster_id);

    EXECUTE format('SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
        MATCH (m:MemoryNode {memory_id: %L})
        MATCH (c:ClusterNode {cluster_id: %L})
        MERGE (m)-[r:MEMBER_OF]->(c)
        SET r.strength = %s, r.added_at = %L
        RETURN m
    $q$) as (result ag_catalog.agtype)', p_memory_id, p_cluster_id, p_strength, CURRENT_TIMESTAMP::text);

    PERFORM upsert_memory_edge('memory', p_memory_id::text, 'MEMBER_OF', 'cluster', p_cluster_id::text,
                               p_strength, NULL, NULL, jsonb_build_object('strength', p_strength));
    RETURN TRUE;
EXCEPTION WHEN OTHERS THEN
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION sync_episode_node(p_episode_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
    EXECUTE format('SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
        MERGE (e:EpisodeNode {episode_id: %L})
        RETURN e
    $q$) as (result ag_catalog.agtype)', p_episode_id);
    RETURN TRUE;
EXCEPTION WHEN OTHERS THEN
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;

SET check_function_bodies = on;
