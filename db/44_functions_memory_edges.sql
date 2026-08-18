-- Relational typed-edge substrate for dynamic sub-knowledge-graph reasoning.
--
-- Mirrors the Apache AGE `memory_graph` edges into a plain relational table so
-- multi-hop, seeded subgraph assembly runs as a Postgres WITH RECURSIVE over
-- btree indexes (no agtype) and joins directly to `memories` for scoring. AGE
-- stays authoritative/written for ad-hoc Cypher; memory_edges is the primary
-- substrate for the reasoning path (build_context_subgraph, db/44 Phase 2).
--
-- Node addressing is generic (node_type, node_key) so heterogeneous endpoints
-- fit one table:
--   memory/goal   -> memories.id::text
--   cluster       -> clusters.id::text
--   episode       -> episodes.id::text
--   concept       -> concept name
--   self/goals_root/life_chapter -> singleton key ('self'/'goals'/'current')
-- Each baseline file runs as its own psql session at initdb, so it must pin
-- its own search_path: under an ag_catalog-first cluster default, unqualified
-- CREATEs would land Hexis objects in ag_catalog (the #77 fossil bug).
SET search_path = public, ag_catalog, "$user";

SET search_path = public, ag_catalog, "$user";

CREATE TABLE IF NOT EXISTS memory_edges (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    src_type   TEXT NOT NULL,
    src_id     TEXT NOT NULL,
    -- TEXT, not the graph_edge_type enum: the graph's edge vocabulary is
    -- open-ended (MEMBER_OF/CONTAINS/HAS_BELIEF are real labels the enum omits).
    -- The enum stays as create_memory_relationship's typed contract only.
    rel_type   TEXT NOT NULL,
    dst_type   TEXT NOT NULL,
    dst_id     TEXT NOT NULL,
    weight     FLOAT NOT NULL DEFAULT 1.0,
    kind       TEXT,
    source     TEXT,
    properties JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (src_type, src_id, rel_type, dst_type, dst_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_edges_src ON memory_edges (src_type, src_id, rel_type);
CREATE INDEX IF NOT EXISTS idx_memory_edges_dst ON memory_edges (dst_type, dst_id, rel_type);

-- Dual-write hook: called beside each AGE edge write. Upsert on the natural key.
CREATE OR REPLACE FUNCTION upsert_memory_edge(
    p_src_type TEXT,
    p_src_id   TEXT,
    p_rel_type TEXT,
    p_dst_type TEXT,
    p_dst_id   TEXT,
    p_weight   FLOAT DEFAULT 1.0,
    p_kind     TEXT DEFAULT NULL,
    p_source   TEXT DEFAULT NULL,
    p_properties JSONB DEFAULT '{}'::jsonb
) RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    IF NULLIF(btrim(COALESCE(p_src_id, '')), '') IS NULL
       OR NULLIF(btrim(COALESCE(p_dst_id, '')), '') IS NULL THEN
        RETURN;  -- never store a dangling edge
    END IF;
    INSERT INTO memory_edges (src_type, src_id, rel_type, dst_type, dst_id, weight, kind, source, properties)
    VALUES (
        p_src_type, p_src_id, p_rel_type, p_dst_type, p_dst_id,
        COALESCE(p_weight, 1.0), p_kind, p_source, COALESCE(p_properties, '{}'::jsonb)
    )
    ON CONFLICT (src_type, src_id, rel_type, dst_type, dst_id) DO UPDATE SET
        weight     = COALESCE(EXCLUDED.weight, memory_edges.weight),
        kind       = COALESCE(EXCLUDED.kind, memory_edges.kind),
        source     = COALESCE(EXCLUDED.source, memory_edges.source),
        properties = memory_edges.properties || EXCLUDED.properties,
        updated_at = CURRENT_TIMESTAMP;
END;
$$;

-- Convenience overload for the common memory->memory case (UUID endpoints),
-- pulling weight from strength/confidence in the properties (matches the AGE
-- create_memory_relationship contract).
CREATE OR REPLACE FUNCTION upsert_memory_edge(
    p_from_id UUID,
    p_to_id   UUID,
    p_rel_type graph_edge_type,
    p_properties JSONB DEFAULT '{}'::jsonb
) RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    PERFORM upsert_memory_edge(
        'memory', p_from_id::text, p_rel_type::text, 'memory', p_to_id::text,
        COALESCE((p_properties->>'strength')::float, (p_properties->>'confidence')::float, 1.0),
        p_properties->>'kind',
        p_properties->>'source',
        COALESCE(p_properties, '{}'::jsonb)
    );
END;
$$;

CREATE OR REPLACE FUNCTION delete_memory_edge(
    p_src_type TEXT, p_src_id TEXT, p_rel_type TEXT, p_dst_type TEXT, p_dst_id TEXT
) RETURNS VOID
LANGUAGE sql
AS $$
    DELETE FROM memory_edges
    WHERE src_type = p_src_type AND src_id = p_src_id AND rel_type = p_rel_type
      AND dst_type = p_dst_type AND dst_id = p_dst_id;
$$;

-- Map an AGE node label + its property map to (node_type, node_key).
CREATE OR REPLACE FUNCTION _memory_edge_node_ref(p_label TEXT, p_props JSONB)
RETURNS TEXT[]   -- [node_type, node_key]
LANGUAGE sql IMMUTABLE
AS $$
    SELECT CASE p_label
        WHEN 'MemoryNode'      THEN ARRAY['memory',       p_props->>'memory_id']
        WHEN 'GoalNode'        THEN ARRAY['goal',         p_props->>'goal_id']
        WHEN 'ConceptNode'     THEN ARRAY['concept',      p_props->>'name']
        WHEN 'ClusterNode'     THEN ARRAY['cluster',      p_props->>'cluster_id']
        WHEN 'EpisodeNode'     THEN ARRAY['episode',      p_props->>'episode_id']
        WHEN 'SelfNode'        THEN ARRAY['self',         COALESCE(p_props->>'key', 'self')]
        WHEN 'GoalsRoot'       THEN ARRAY['goals_root',   COALESCE(p_props->>'key', 'goals')]
        WHEN 'LifeChapterNode' THEN ARRAY['life_chapter', COALESCE(p_props->>'key', 'current')]
        ELSE ARRAY[lower(p_label), COALESCE(p_props->>'memory_id', p_props->>'name', p_props->>'key')]
    END;
$$;

-- One-time migration: populate memory_edges from the live AGE graph. Iterates
-- every edge (all labels at once), resolving endpoint types/keys from node
-- labels. Idempotent (upsert). Returns the number of edges backfilled.
CREATE OR REPLACE FUNCTION backfill_memory_edges()
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    rec RECORD;
    n INTEGER := 0;
    src_ref TEXT[];
    dst_ref TEXT[];
    props JSONB;
    rt TEXT;
BEGIN
    FOR rec IN
        SELECT * FROM ag_catalog.cypher('memory_graph', $q$
            MATCH (a)-[r]->(b)
            RETURN label(a), properties(a), type(r), properties(r), label(b), properties(b)
        $q$) AS (la agtype, pa agtype, rt agtype, pr agtype, lb agtype, pb agtype)
    LOOP
        rt := btrim(rec.rt::text, '"');
        src_ref := _memory_edge_node_ref(btrim(rec.la::text, '"'), (rec.pa::text)::jsonb);
        dst_ref := _memory_edge_node_ref(btrim(rec.lb::text, '"'), (rec.pb::text)::jsonb);
        props   := COALESCE((rec.pr::text)::jsonb, '{}'::jsonb);
        PERFORM upsert_memory_edge(
            src_ref[1], src_ref[2], rt, dst_ref[1], dst_ref[2],
            COALESCE((props->>'strength')::float, (props->>'confidence')::float, 1.0),
            props->>'kind', props->>'source', props
        );
        n := n + 1;
    END LOOP;
    RETURN n;
END;
$$;

-- Cast text->uuid, returning NULL instead of erroring on non-uuid keys (concept
-- names, singleton keys). Lets node-label joins run over heterogeneous node ids.
CREATE OR REPLACE FUNCTION _safe_uuid(p TEXT)
RETURNS UUID
LANGUAGE plpgsql IMMUTABLE
AS $$
BEGIN
    RETURN p::uuid;
EXCEPTION WHEN OTHERS THEN
    RETURN NULL;
END;
$$;

-- Dynamic sub-knowledge-graph: given seed memories, expand outward over
-- memory_edges (both directions, bounded depth, weighted, budgeted) and return
-- a focused subgraph as JSONB {nodes, edges}. This is the primary graph-
-- reasoning path -- it reveals the *structure* (supports/contradicts/causes/...)
-- among + around what recall surfaced, which flat vector recall cannot.
--   p_rel_types: restrict expansion to these edge types (NULL = all).
--   p_depth:     max hops from a seed (clamped 0..6).
--   p_budget:    max nodes in the result (clamped 1..500); seeds are kept first.
CREATE OR REPLACE FUNCTION build_context_subgraph(
    p_seed_ids UUID[],
    p_depth INT DEFAULT 2,
    p_rel_types TEXT[] DEFAULT NULL,
    p_budget INT DEFAULT 40
) RETURNS JSONB
LANGUAGE plpgsql STABLE
AS $$
DECLARE
    result JSONB;
    v_depth  INT := LEAST(GREATEST(COALESCE(p_depth, 2), 0), 6);
    v_budget INT := LEAST(GREATEST(COALESCE(p_budget, 40), 1), 500);
BEGIN
    IF p_seed_ids IS NULL OR array_length(p_seed_ids, 1) IS NULL THEN
        RETURN jsonb_build_object('nodes', '[]'::jsonb, 'edges', '[]'::jsonb);
    END IF;

    WITH RECURSIVE
    -- Live memories only. memory_edges is a DERIVED edge store, not the source of
    -- truth: `memories` is. As maintenance consolidates/prunes/invalidates
    -- memories (soft state changes), those memories -- and every edge touching
    -- them -- drop out here, so the subgraph always reflects the current memory
    -- set without needing to delete edge rows.
    live_mem AS (
        SELECT id FROM memories
        WHERE status = 'active' AND (valid_until IS NULL OR valid_until > CURRENT_TIMESTAMP)
    ),
    -- Edges whose memory/goal endpoints are live (concept/cluster/episode/self
    -- endpoints carry no memory status and pass through). rel-filtered here so
    -- both traversal and output see the same live edge set.
    edges_live AS (
        SELECT e.src_type, e.src_id, e.dst_type, e.dst_id, e.weight, e.rel_type
        FROM memory_edges e
        WHERE (p_rel_types IS NULL OR e.rel_type = ANY(p_rel_types))
          AND (e.src_type NOT IN ('memory', 'goal') OR _safe_uuid(e.src_id) IN (SELECT id FROM live_mem))
          AND (e.dst_type NOT IN ('memory', 'goal') OR _safe_uuid(e.dst_id) IN (SELECT id FROM live_mem))
    ),
    -- Undirected view (each live edge usable from either endpoint).
    adj AS (
        SELECT src_type AS from_type, src_id AS from_id,
               dst_type AS to_type,   dst_id AS to_id, weight, rel_type
        FROM edges_live
        UNION ALL
        SELECT dst_type, dst_id, src_type, src_id, weight, rel_type
        FROM edges_live
    ),
    frontier AS (
        SELECT 'memory'::text AS node_type, s::text AS node_id, 0 AS depth,
               1.0::float AS path_weight, ARRAY['memory:' || s::text] AS visited
        FROM unnest(p_seed_ids) s
        WHERE s IN (SELECT id FROM live_mem)   -- only live seeds enter
        UNION ALL
        SELECT a.to_type, a.to_id, f.depth + 1, f.path_weight * COALESCE(a.weight, 1.0),
               f.visited || (a.to_type || ':' || a.to_id)
        FROM frontier f
        JOIN adj a ON a.from_type = f.node_type AND a.from_id = f.node_id
        WHERE f.depth < v_depth
          AND NOT ((a.to_type || ':' || a.to_id) = ANY(f.visited))  -- cycle guard
    ),
    reached AS (
        SELECT node_type, node_id, MIN(depth) AS depth, MAX(path_weight) AS relevance
        FROM frontier
        GROUP BY node_type, node_id
    ),
    kept AS (
        SELECT node_type, node_id, depth, relevance
        FROM reached
        ORDER BY depth ASC, relevance DESC, node_id
        LIMIT v_budget
    ),
    node_json AS (
        SELECT jsonb_agg(jsonb_build_object(
            'type', k.node_type,
            'id',   k.node_id,
            'label', left(COALESCE(m.content, cl.name, ep.summary, k.node_id), 200),
            'memory_type', m.type,
            'importance', m.importance,
            'strength', CASE WHEN m.id IS NULL THEN NULL
                             ELSE round(calculate_strength(m.importance, m.decay_rate, m.created_at, m.last_reinforced)::numeric, 4) END,
            'fidelity', m.fidelity,
            'depth', k.depth,
            'relevance', round(k.relevance::numeric, 4)
        ) ORDER BY k.depth, k.relevance DESC) AS arr
        FROM kept k
        LEFT JOIN memories m ON k.node_type IN ('memory', 'goal') AND m.id = _safe_uuid(k.node_id)
        LEFT JOIN clusters cl ON k.node_type = 'cluster'  AND cl.id = _safe_uuid(k.node_id)
        LEFT JOIN episodes ep ON k.node_type = 'episode'  AND ep.id = _safe_uuid(k.node_id)
    ),
    edge_json AS (
        SELECT jsonb_agg(DISTINCT jsonb_build_object(
            'src_type', e.src_type, 'src_id', e.src_id,
            'rel', e.rel_type::text,
            'dst_type', e.dst_type, 'dst_id', e.dst_id,
            'weight', round(e.weight::numeric, 4)
        )) AS arr
        FROM edges_live e
        WHERE (e.src_type, e.src_id) IN (SELECT node_type, node_id FROM kept)
          AND (e.dst_type, e.dst_id) IN (SELECT node_type, node_id FROM kept)
    )
    SELECT jsonb_build_object(
        'nodes', COALESCE(node_json.arr, '[]'::jsonb),
        'edges', COALESCE(edge_json.arr, '[]'::jsonb)
    )
    INTO result
    FROM node_json, edge_json;

    RETURN COALESCE(result, jsonb_build_object('nodes', '[]'::jsonb, 'edges', '[]'::jsonb));
END;
$$;

CREATE TABLE IF NOT EXISTS graph_reconciliation_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repair BOOLEAN NOT NULL DEFAULT TRUE,
    dangling_edges INT NOT NULL DEFAULT 0,
    deleted_edges INT NOT NULL DEFAULT 0,
    age_backfilled_edges INT,
    result JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE OR REPLACE FUNCTION reconcile_graph(
    p_repair BOOLEAN DEFAULT TRUE
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    dangling_count INT := 0;
    deleted_count INT := 0;
    backfilled_count INT := NULL;
    result JSONB;
BEGIN
    SELECT count(*)::int
    INTO dangling_count
    FROM memory_edges e
    WHERE (e.src_type = 'memory' AND NOT EXISTS (
              SELECT 1 FROM memories m
              WHERE m.id = _safe_uuid(e.src_id)
                AND m.status = 'active'
                AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
          ))
       OR (e.dst_type = 'memory' AND NOT EXISTS (
              SELECT 1 FROM memories m
              WHERE m.id = _safe_uuid(e.dst_id)
                AND m.status = 'active'
                AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
          ));

    IF COALESCE(p_repair, TRUE) AND dangling_count > 0 THEN
        WITH deleted AS (
            DELETE FROM memory_edges e
            WHERE (e.src_type = 'memory' AND NOT EXISTS (
                      SELECT 1 FROM memories m
                      WHERE m.id = _safe_uuid(e.src_id)
                        AND m.status = 'active'
                        AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
                  ))
               OR (e.dst_type = 'memory' AND NOT EXISTS (
                      SELECT 1 FROM memories m
                      WHERE m.id = _safe_uuid(e.dst_id)
                        AND m.status = 'active'
                        AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
                  ))
            RETURNING 1
        )
        SELECT count(*)::int INTO deleted_count FROM deleted;
    END IF;

    BEGIN
        IF COALESCE(p_repair, TRUE)
           AND to_regproc('public.backfill_memory_edges') IS NOT NULL THEN
            SELECT backfill_memory_edges() INTO backfilled_count;
        END IF;
    EXCEPTION WHEN OTHERS THEN
        backfilled_count := NULL;
    END;

    result := jsonb_build_object(
        'repair', COALESCE(p_repair, TRUE),
        'dangling_edges', dangling_count,
        'deleted_edges', deleted_count,
        'age_backfilled_edges', backfilled_count,
        'status', CASE WHEN dangling_count = 0 OR COALESCE(p_repair, TRUE) THEN 'ok' ELSE 'needs_repair' END
    );

    INSERT INTO graph_reconciliation_runs (
        repair, dangling_edges, deleted_edges, age_backfilled_edges, result
    )
    VALUES (
        COALESCE(p_repair, TRUE), dangling_count, deleted_count, backfilled_count, result
    );

    RETURN result;
END;
$$;

CREATE OR REPLACE FUNCTION memory_graph_paths(
    p_seed_id UUID,
    p_rel_types TEXT[] DEFAULT ARRAY['CAUSES','CONTRADICTS','CONTESTED_BECAUSE','SUPPORTS','SUPERSEDES'],
    p_depth INT DEFAULT 3,
    p_limit INT DEFAULT 25
) RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    WITH RECURSIVE walk AS (
        SELECT
            e.id AS edge_id,
            e.src_type,
            e.src_id,
            e.rel_type,
            e.dst_type,
            e.dst_id,
            e.weight,
            1 AS depth,
            ARRAY[e.src_type || ':' || e.src_id, e.dst_type || ':' || e.dst_id] AS visited,
            jsonb_build_array(jsonb_build_object(
                'src_type', e.src_type,
                'src_id', e.src_id,
                'rel', e.rel_type,
                'dst_type', e.dst_type,
                'dst_id', e.dst_id,
                'weight', e.weight
            )) AS edges
        FROM memory_edges e
        WHERE ((e.src_type = 'memory' AND e.src_id = p_seed_id::text)
           OR (e.dst_type = 'memory' AND e.dst_id = p_seed_id::text))
          AND (p_rel_types IS NULL OR e.rel_type = ANY(p_rel_types))
        UNION ALL
        SELECT
            e.id,
            e.src_type,
            e.src_id,
            e.rel_type,
            e.dst_type,
            e.dst_id,
            e.weight,
            w.depth + 1,
            w.visited || CASE
                WHEN e.src_type || ':' || e.src_id = w.visited[array_length(w.visited, 1)]
                THEN e.dst_type || ':' || e.dst_id
                ELSE e.src_type || ':' || e.src_id
            END,
            w.edges || jsonb_build_array(jsonb_build_object(
                'src_type', e.src_type,
                'src_id', e.src_id,
                'rel', e.rel_type,
                'dst_type', e.dst_type,
                'dst_id', e.dst_id,
                'weight', e.weight
            ))
        FROM walk w
        JOIN memory_edges e
          ON (e.src_type || ':' || e.src_id = w.visited[array_length(w.visited, 1)]
              OR e.dst_type || ':' || e.dst_id = w.visited[array_length(w.visited, 1)])
        WHERE w.depth < LEAST(GREATEST(COALESCE(p_depth, 3), 1), 6)
          AND (p_rel_types IS NULL OR e.rel_type = ANY(p_rel_types))
          AND NOT (
              CASE
                WHEN e.src_type || ':' || e.src_id = w.visited[array_length(w.visited, 1)]
                THEN e.dst_type || ':' || e.dst_id
                ELSE e.src_type || ':' || e.src_id
              END = ANY(w.visited)
          )
    ),
    ranked AS (
        SELECT *
        FROM walk
        ORDER BY depth ASC, weight DESC
        LIMIT LEAST(GREATEST(COALESCE(p_limit, 25), 1), 100)
    )
    SELECT jsonb_build_object(
        'seed_id', p_seed_id::text,
        'paths', COALESCE(jsonb_agg(jsonb_build_object(
            'depth', depth,
            'terminal', visited[array_length(visited, 1)],
            'visited', to_jsonb(visited),
            'edges', edges
        ) ORDER BY depth), '[]'::jsonb)
    )
    FROM ranked;
$$;

CREATE OR REPLACE FUNCTION memory_context_paths(
    p_seed_ids UUID[],
    p_depth INT DEFAULT 2
) RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    SELECT jsonb_build_object(
        'seeds', COALESCE((SELECT jsonb_agg(s::text) FROM unnest(COALESCE(p_seed_ids, ARRAY[]::uuid[])) s), '[]'::jsonb),
        'paths', COALESCE(jsonb_agg(path_doc), '[]'::jsonb)
    )
    FROM (
        SELECT memory_graph_paths(seed_id, ARRAY['CAUSES','CONTRADICTS','CONTESTED_BECAUSE','SUPPORTS','SUPERSEDES'], p_depth, 10) AS path_doc
        FROM unnest(COALESCE(p_seed_ids, ARRAY[]::uuid[])) seed_id
    ) q;
$$;
