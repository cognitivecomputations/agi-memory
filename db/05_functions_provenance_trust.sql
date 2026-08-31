-- Hexis schema: provenance and trust functions.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

CREATE OR REPLACE FUNCTION normalize_source_reference(p_source JSONB)
RETURNS JSONB AS $$
DECLARE
    kind TEXT;
    ref TEXT;
    label TEXT;
    author TEXT;
    observed_at TIMESTAMPTZ;
    trust FLOAT;
    content_hash TEXT;
    source_document_id TEXT;
    document_id TEXT;
    chunk_id TEXT;
    chunk_index INT;
    sensitivity TEXT;
BEGIN
    IF p_source IS NULL OR jsonb_typeof(p_source) <> 'object' THEN
        RETURN '{}'::jsonb;
    END IF;

    kind := NULLIF(p_source->>'kind', '');
    ref := COALESCE(NULLIF(p_source->>'ref', ''), NULLIF(p_source->>'uri', ''));
    label := NULLIF(p_source->>'label', '');
    author := NULLIF(p_source->>'author', '');
    content_hash := NULLIF(p_source->>'content_hash', '');
    source_document_id := COALESCE(NULLIF(p_source->>'source_document_id', ''), NULLIF(p_source->>'document_id', ''));
    document_id := COALESCE(NULLIF(p_source->>'document_id', ''), source_document_id);
    -- Chunk-grain provenance survives normalization: memories extracted from
    -- a source chunk keep the handle needed to cite the exact passage.
    chunk_id := NULLIF(p_source->>'chunk_id', '');
    BEGIN
        chunk_index := NULLIF(p_source->>'chunk_index', '')::int;
    EXCEPTION WHEN OTHERS THEN
        chunk_index := NULL;
    END;
    -- Sensitivity survives normalization (#92): 'private' is the one defined
    -- level; it keeps the memory out of group recall and default export.
    sensitivity := CASE WHEN p_source->>'sensitivity' = 'private' THEN 'private' END;

    BEGIN
        observed_at := (p_source->>'observed_at')::timestamptz;
    EXCEPTION WHEN OTHERS THEN
        observed_at := CURRENT_TIMESTAMP;
    END;
    IF observed_at IS NULL THEN
        observed_at := CURRENT_TIMESTAMP;
    END IF;

    trust := COALESCE(NULLIF(p_source->>'trust', '')::float, 0.5);
    trust := LEAST(1.0, GREATEST(0.0, trust));

    RETURN jsonb_strip_nulls(
        jsonb_build_object(
            'kind', kind,
            'ref', ref,
            'label', label,
            'author', author,
            'observed_at', observed_at,
            'trust', trust,
            'content_hash', content_hash,
            'source_document_id', source_document_id,
            'document_id', document_id,
            'chunk_id', chunk_id,
            'chunk_index', chunk_index,
            'sensitivity', sensitivity
        )
    );
    END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION recall_memories_filtered(
    p_query_text TEXT,
    p_limit INT DEFAULT 10,
    p_memory_types memory_type[] DEFAULT NULL,
    p_min_importance FLOAT DEFAULT 0.0
) RETURNS TABLE (
    memory_id UUID,
    content TEXT,
    memory_type memory_type,
    score FLOAT,
    source TEXT,
    importance FLOAT,
    trust_level FLOAT,
    source_attribution JSONB,
    created_at TIMESTAMPTZ,
    emotional_valence FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH hits AS (
        SELECT * FROM fast_recall(p_query_text, p_limit * 2)
    )
    SELECT
        h.memory_id,
        h.content,
        h.memory_type,
        h.score,
        h.source,
        m.importance,
        m.trust_level,
        m.source_attribution,
        m.created_at,
        (m.metadata->>'emotional_valence')::float AS emotional_valence
    FROM hits h
    JOIN memories m ON m.id = h.memory_id
    WHERE (p_memory_types IS NULL OR h.memory_type = ANY(p_memory_types))
      AND m.status = 'active'
      AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
      AND m.importance >= COALESCE(p_min_importance, 0.0)
    ORDER BY h.score DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- Structured recall with rich filters: source path/kind, date range, concept, metadata
CREATE OR REPLACE FUNCTION recall_memories_structured(
    p_query_text TEXT,
    p_limit INT DEFAULT 10,
    p_memory_types memory_type[] DEFAULT NULL,
    p_min_importance FLOAT DEFAULT 0.0,
    p_source_path TEXT DEFAULT NULL,
    p_source_kind TEXT DEFAULT NULL,
    p_created_after TIMESTAMPTZ DEFAULT NULL,
    p_created_before TIMESTAMPTZ DEFAULT NULL,
    p_concept TEXT DEFAULT NULL,
    p_metadata_filter JSONB DEFAULT NULL
) RETURNS TABLE (
    memory_id UUID,
    content TEXT,
    memory_type memory_type,
    score FLOAT,
    source TEXT,
    importance FLOAT,
    trust_level FLOAT,
    source_attribution JSONB,
    created_at TIMESTAMPTZ,
    emotional_valence FLOAT
) AS $$
DECLARE
    use_vector BOOLEAN;
BEGIN
    -- If no query text, we can't use semantic/lexical search; fall back to
    -- direct filters. Query+filter recall still needs hybrid candidates so
    -- just-written async-embedding memories remain visible before the worker
    -- vectorizes them.
    use_vector := (p_query_text IS NOT NULL AND trim(p_query_text) <> '');

    IF use_vector THEN
        RETURN QUERY
        WITH hits AS (
            SELECT * FROM recall_hybrid(p_query_text, GREATEST(p_limit * 5, 20))
        )
        SELECT
            h.memory_id,
            h.content,
            h.memory_type,
            h.score,
            h.source,
            m.importance,
            m.trust_level,
            m.source_attribution,
            m.created_at,
            (m.metadata->>'emotional_valence')::float AS emotional_valence
        FROM hits h
        JOIN memories m ON m.id = h.memory_id
        WHERE (p_memory_types IS NULL OR h.memory_type = ANY(p_memory_types))
          AND m.importance >= COALESCE(p_min_importance, 0.0)
          AND (p_source_path IS NULL OR m.source_attribution->>'path' ILIKE '%' || p_source_path || '%')
          AND (p_source_kind IS NULL OR m.source_attribution->>'kind' = p_source_kind)
          AND (p_created_after IS NULL OR m.created_at >= p_created_after)
          AND (p_created_before IS NULL OR m.created_at <= p_created_before)
          AND (p_metadata_filter IS NULL OR m.metadata @> p_metadata_filter)
        ORDER BY h.score DESC
        LIMIT p_limit;
    ELSE
        -- Filter-only mode (no semantic search)
        RETURN QUERY
        SELECT
            m.id AS memory_id,
            m.content,
            m.type AS memory_type,
            m.importance::float AS score,
            'filter'::text AS source,
            m.importance,
            m.trust_level,
            m.source_attribution,
            m.created_at,
            (m.metadata->>'emotional_valence')::float AS emotional_valence
        FROM memories m
        WHERE m.status = 'active'
          AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
          AND (p_memory_types IS NULL OR m.type = ANY(p_memory_types))
          AND m.importance >= COALESCE(p_min_importance, 0.0)
          AND (p_source_path IS NULL OR m.source_attribution->>'path' ILIKE '%' || p_source_path || '%')
          AND (p_source_kind IS NULL OR m.source_attribution->>'kind' = p_source_kind)
          AND (p_created_after IS NULL OR m.created_at >= p_created_after)
          AND (p_created_before IS NULL OR m.created_at <= p_created_before)
          AND (p_metadata_filter IS NULL OR m.metadata @> p_metadata_filter)
        ORDER BY m.importance DESC, m.created_at DESC
        LIMIT p_limit;
    END IF;

    -- Concept filter: if p_concept is provided, additionally filter via graph
    -- (Handled in Python layer via knowledge graph lookup when concept != NULL)
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION recall_memories_stub(
    p_query_text TEXT,
    p_limit INT DEFAULT 10,
    p_memory_types memory_type[] DEFAULT NULL,
    p_min_importance FLOAT DEFAULT 0.0,
    p_preview_chars INT DEFAULT 256
) RETURNS TABLE (
    memory_id UUID,
    preview TEXT,
    memory_type memory_type,
    score FLOAT,
    source TEXT,
    importance FLOAT,
    trust_level FLOAT,
    source_attribution JSONB,
    created_at TIMESTAMPTZ,
    emotional_valence FLOAT,
    content_length INT
) AS $$
BEGIN
    RETURN QUERY
    WITH hits AS (
        SELECT * FROM fast_recall(p_query_text, p_limit * 2)
    )
    SELECT
        h.memory_id,
        LEFT(h.content, p_preview_chars) AS preview,
        h.memory_type,
        h.score,
        h.source,
        m.importance,
        m.trust_level,
        m.source_attribution,
        m.created_at,
        (m.metadata->>'emotional_valence')::float AS emotional_valence,
        length(h.content) AS content_length
    FROM hits h
    JOIN memories m ON m.id = h.memory_id
    WHERE (p_memory_types IS NULL OR h.memory_type = ANY(p_memory_types))
      AND m.importance >= COALESCE(p_min_importance, 0.0)
    ORDER BY h.score DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION record_memory_reinforcement(
    p_memory_id UUID,
    p_kind TEXT DEFAULT 'recall',
    p_source TEXT DEFAULT 'system',
    p_metadata JSONB DEFAULT '{}'::jsonb,
    p_created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
) RETURNS UUID AS $$
DECLARE
    event_id UUID;
BEGIN
    IF p_memory_id IS NULL THEN
        RAISE EXCEPTION 'memory_id is required';
    END IF;

    INSERT INTO memory_reinforcement_events (memory_id, kind, source, metadata, created_at)
    VALUES (
        p_memory_id,
        COALESCE(NULLIF(trim(p_kind), ''), 'recall'),
        COALESCE(NULLIF(trim(p_source), ''), 'system'),
        COALESCE(p_metadata, '{}'::jsonb),
        COALESCE(p_created_at, CURRENT_TIMESTAMP)
    )
    RETURNING id INTO event_id;

    RETURN event_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION memory_spaced_reinforcement_score(
    p_memory_id UUID,
    p_window INTERVAL DEFAULT INTERVAL '180 days',
    p_min_interval INTERVAL DEFAULT NULL
) RETURNS FLOAT AS $$
DECLARE
    rec RECORD;
    total_count INT := 0;
    spaced_count INT := 0;
    last_counted_at TIMESTAMPTZ := NULL;
    effective_min_interval INTERVAL;
    scale FLOAT;
BEGIN
    IF p_memory_id IS NULL THEN
        RETURN 0.0;
    END IF;

    effective_min_interval := COALESCE(
        p_min_interval,
        (GREATEST(COALESCE(get_config_float('memory.spaced_reinforcement_interval_hours'), 12.0), 0.01) || ' hours')::interval
    );
    scale := GREATEST(COALESCE(get_config_float('memory.spaced_reinforcement_scale'), 4.0), 1.0);

    FOR rec IN
        SELECT created_at
        FROM memory_reinforcement_events
        WHERE memory_id = p_memory_id
          AND created_at >= CURRENT_TIMESTAMP - COALESCE(p_window, INTERVAL '180 days')
        ORDER BY created_at ASC
    LOOP
        total_count := total_count + 1;
        IF last_counted_at IS NULL OR rec.created_at - last_counted_at >= effective_min_interval THEN
            spaced_count := spaced_count + 1;
            last_counted_at := rec.created_at;
        END IF;
    END LOOP;

    IF total_count = 0 THEN
        RETURN 0.0;
    END IF;

    -- Saturating score: repeated practice matters, but packed repetition
    -- collapses into one effective reinforcement until enough time has passed.
    RETURN LEAST(1.0, GREATEST(0.0, 1.0 - exp(-spaced_count::float / scale)));
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION touch_memories(p_ids UUID[])
RETURNS INT AS $$
DECLARE
    updated_count INT;
BEGIN
    IF p_ids IS NULL OR array_length(p_ids, 1) IS NULL THEN
        RETURN 0;
    END IF;
    -- Access IS reinforcement: recalling a memory strengthens it (resets the
    -- decay clock via last_reinforced) -- the "up-ladder" of the compression-
    -- native substrate (docs/memory_retention_design.md §2). We also keep a
    -- reinforcement event log so spaced practice remains observable.
    WITH updated AS (
        UPDATE memories
        SET access_count = access_count + 1,
            last_accessed = CURRENT_TIMESTAMP,
            last_reinforced = CURRENT_TIMESTAMP,
            reinforcement_count = reinforcement_count + 1
        WHERE id = ANY(p_ids)
        RETURNING id
    ),
    logged AS (
        INSERT INTO memory_reinforcement_events (memory_id, kind, source, metadata)
        SELECT id, 'recall', 'touch_memories', '{}'::jsonb
        FROM updated
        RETURNING 1
    )
    SELECT COUNT(*)::int INTO updated_count FROM logged;

    RETURN COALESCE(updated_count, 0);
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION get_memory_by_id(p_memory_id UUID)
RETURNS TABLE (
    id UUID,
    type memory_type,
    content TEXT,
    importance FLOAT,
    trust_level FLOAT,
    source_attribution JSONB,
    created_at TIMESTAMPTZ,
    emotional_valence FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.type,
        m.content,
        m.importance,
        m.trust_level,
        m.source_attribution,
        m.created_at,
        (m.metadata->>'emotional_valence')::float
    FROM memories m
    WHERE m.id = p_memory_id
      AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP);
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION get_memories_summary(p_ids UUID[])
RETURNS TABLE (
    id UUID,
    type memory_type,
    content TEXT,
    importance FLOAT
) AS $$
BEGIN
    IF p_ids IS NULL OR array_length(p_ids, 1) IS NULL THEN
        RETURN;
    END IF;
    RETURN QUERY
    SELECT
        m.id,
        m.type,
        m.content,
        m.importance
    FROM memories m
    WHERE m.id = ANY(p_ids);
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION get_memories_by_ids(
    p_ids UUID[],
    p_max_chars INT DEFAULT 2000
) RETURNS TABLE (
    id UUID,
    type memory_type,
    content TEXT,
    importance FLOAT,
    trust_level FLOAT,
    source_attribution JSONB,
    created_at TIMESTAMPTZ,
    emotional_valence FLOAT
) AS $$
BEGIN
    IF p_ids IS NULL OR array_length(p_ids, 1) IS NULL THEN
        RETURN;
    END IF;
    RETURN QUERY
    SELECT
        m.id,
        m.type,
        LEFT(m.content, p_max_chars) AS content,
        m.importance,
        m.trust_level,
        m.source_attribution,
        m.created_at,
        (m.metadata->>'emotional_valence')::float
    FROM memories m
    WHERE m.id = ANY(p_ids);
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION list_recent_memories(
    p_limit INT DEFAULT 10,
    p_memory_types memory_type[] DEFAULT NULL,
    p_by_access BOOLEAN DEFAULT FALSE
) RETURNS TABLE (
    memory_id UUID,
    content TEXT,
    memory_type memory_type,
    importance FLOAT,
    created_at TIMESTAMPTZ,
    last_accessed TIMESTAMPTZ,
    trust_level FLOAT,
    source_attribution JSONB,
    emotional_valence FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.content,
        m.type,
        m.importance,
        m.created_at,
        m.last_accessed,
        m.trust_level,
        m.source_attribution,
        (m.metadata->>'emotional_valence')::float
    FROM memories m
    WHERE m.status = 'active'
      AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
      AND (p_memory_types IS NULL OR m.type = ANY(p_memory_types))
    ORDER BY
        CASE WHEN p_by_access THEN m.last_accessed ELSE m.created_at END DESC NULLS LAST
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION get_episode_details(p_episode_id UUID)
RETURNS TABLE (
    id UUID,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    episode_type TEXT,
    summary TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.id,
        e.started_at,
        e.ended_at,
        e.metadata->>'episode_type' as episode_type,
        e.summary
    FROM episodes e
    WHERE e.id = p_episode_id;
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION get_episode_memories(p_episode_id UUID)
RETURNS TABLE (
    memory_id UUID,
    content TEXT,
    memory_type memory_type,
    importance FLOAT,
    trust_level FLOAT,
    source_attribution JSONB,
    created_at TIMESTAMPTZ,
    emotional_valence FLOAT,
    sequence_order INT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.content,
        m.type,
        m.importance,
        m.trust_level,
        m.source_attribution,
        m.created_at,
        (m.metadata->>'emotional_valence')::float,
        fem.sequence_order
    FROM find_episode_memories_graph(p_episode_id) fem
    JOIN memories m ON fem.memory_id = m.id
    ORDER BY fem.sequence_order ASC;
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION list_recent_episodes(p_limit INT DEFAULT 5)
RETURNS TABLE (
    id UUID,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    episode_type TEXT,
    summary TEXT,
    memory_count INT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.id,
        e.started_at,
        e.ended_at,
        e.metadata->>'episode_type' as episode_type,
        e.summary,
        (SELECT COUNT(*)::int FROM find_episode_memories_graph(e.id)) as memory_count
    FROM episodes e
    ORDER BY e.started_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION search_clusters_by_query(
    p_query TEXT,
    p_limit INT DEFAULT 3
) RETURNS TABLE (
    id UUID,
    name TEXT,
    cluster_type cluster_type,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH query_embedding AS (
        SELECT (get_embedding(ARRAY[ensure_embedding_prefix(p_query, 'search_query')]))[1] as emb
    )
    SELECT
        c.id,
        c.name,
        c.cluster_type,
        1 - (c.centroid_embedding <=> (SELECT emb FROM query_embedding)) as similarity
    FROM clusters c
    WHERE c.centroid_embedding IS NOT NULL
    ORDER BY c.centroid_embedding <=> (SELECT emb FROM query_embedding)
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION get_cluster_sample_memories(
    p_cluster_id UUID,
    p_limit INT DEFAULT 3
) RETURNS TABLE (
    memory_id UUID,
    content TEXT,
    memory_type memory_type,
    membership_strength FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.content,
        m.type,
        gcm.membership_strength
    FROM get_cluster_members_graph(p_cluster_id) gcm
    JOIN memories m ON gcm.memory_id = m.id
    WHERE m.status = 'active'
      AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
    ORDER BY gcm.membership_strength DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION explore_clusters_with_samples(
    p_query TEXT,
    p_cluster_limit INT DEFAULT 3,
    p_sample_limit INT DEFAULT 3
) RETURNS TABLE (
    cluster_id UUID,
    cluster_name TEXT,
    cluster_type cluster_type,
    cluster_similarity FLOAT,
    memory_id UUID,
    content TEXT,
    memory_type memory_type,
    membership_strength FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        sc.id,
        sc.name,
        sc.cluster_type,
        sc.similarity,
        sm.memory_id,
        sm.content,
        sm.memory_type,
        sm.membership_strength
    FROM search_clusters_by_query(p_query, p_cluster_limit) sc
    LEFT JOIN LATERAL get_cluster_sample_memories(sc.id, p_sample_limit) sm ON true;
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION find_related_concepts_for_memories(
    p_memory_ids UUID[],
    p_exclude TEXT DEFAULT '',
    p_limit INT DEFAULT 10
) RETURNS TABLE (
    name TEXT,
    shared_memories INT
) AS $$
DECLARE
    ids_sql TEXT;
    sql TEXT;
BEGIN
    IF p_memory_ids IS NULL OR array_length(p_memory_ids, 1) IS NULL THEN
        RETURN;
    END IF;

    SELECT array_to_string(ARRAY(
        SELECT quote_literal(mid::text)
        FROM unnest(p_memory_ids) as mid
    ), ',') INTO ids_sql;

    IF ids_sql IS NULL OR btrim(ids_sql) = '' THEN
        RETURN;
    END IF;

    sql := format($sql$
        SELECT
            replace(name_raw::text, '"', '') as name,
            (shared_raw::text)::int as shared_memories
        FROM ag_catalog.cypher('memory_graph', $q$
            MATCH (m:MemoryNode)-[:INSTANCE_OF]->(c:ConceptNode)
            WHERE m.memory_id IN [%s] AND c.name <> %L
            RETURN c.name, COUNT(m) as shared
            ORDER BY COUNT(m) DESC
            LIMIT %s
        $q$) as (name_raw ag_catalog.agtype, shared_raw ag_catalog.agtype)
    $sql$, ids_sql, COALESCE(p_exclude, ''), p_limit);

    RETURN QUERY EXECUTE sql;
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION search_procedural_memories(
    p_task TEXT,
    p_limit INT DEFAULT 3
) RETURNS TABLE (
    memory_id UUID,
    content TEXT,
    steps JSONB,
    prerequisites JSONB,
    success_rate FLOAT,
    average_duration FLOAT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH query_embedding AS (
        SELECT (get_embedding(ARRAY[ensure_embedding_prefix(p_task, 'search_query')]))[1] as emb
    )
    SELECT
        m.id,
        m.content,
        m.metadata->'steps' as steps,
        m.metadata->'prerequisites' as prerequisites,
        CASE
            WHEN COALESCE((m.metadata->>'total_attempts')::int, 0) > 0 THEN
                (m.metadata->>'success_count')::float / NULLIF((m.metadata->>'total_attempts')::float, 0)
            ELSE NULL
        END as success_rate,
        (m.metadata->>'average_duration_seconds')::float as average_duration,
        1 - (m.embedding <=> (SELECT emb FROM query_embedding)) as similarity
    FROM memories m
    WHERE m.status = 'active'
      AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
      AND m.type = 'procedural'
    ORDER BY m.embedding <=> (SELECT emb FROM query_embedding)
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION search_strategic_memories(
    p_situation TEXT,
    p_limit INT DEFAULT 3
) RETURNS TABLE (
    memory_id UUID,
    content TEXT,
    pattern_description TEXT,
    confidence_score FLOAT,
    context_applicability JSONB,
    success_metrics JSONB,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH query_embedding AS (
        SELECT (get_embedding(ARRAY[ensure_embedding_prefix(p_situation, 'search_query')]))[1] as emb
    )
    SELECT
        m.id,
        m.content,
        COALESCE(m.metadata->>'pattern_description', m.content) as pattern_description,
        (m.metadata->>'confidence_score')::float as confidence_score,
        m.metadata->'context_applicability' as context_applicability,
        m.metadata->'success_metrics' as success_metrics,
        1 - (m.embedding <=> (SELECT emb FROM query_embedding)) as similarity
    FROM memories m
    WHERE m.status = 'active'
      AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
      AND m.type = 'strategic'
    ORDER BY m.embedding <=> (SELECT emb FROM query_embedding)
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION normalize_source_references(p_sources JSONB)
RETURNS JSONB AS $$
DECLARE
    elem JSONB;
    out_arr JSONB := '[]'::jsonb;
BEGIN
    IF p_sources IS NULL THEN
        RETURN '[]'::jsonb;
    END IF;

    IF jsonb_typeof(p_sources) = 'array' THEN
        FOR elem IN SELECT * FROM jsonb_array_elements(p_sources)
        LOOP
            out_arr := out_arr || jsonb_build_array(normalize_source_reference(elem));
        END LOOP;
    ELSIF jsonb_typeof(p_sources) = 'object' THEN
        out_arr := jsonb_build_array(normalize_source_reference(p_sources));
    ELSE
        RETURN '[]'::jsonb;
    END IF;

    RETURN COALESCE(
        (SELECT jsonb_agg(e) FROM jsonb_array_elements(out_arr) e WHERE e <> '{}'::jsonb),
        '[]'::jsonb
    );
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION dedupe_source_references(p_sources JSONB)
RETURNS JSONB AS $$
BEGIN
    RETURN COALESCE((
        SELECT jsonb_agg(d.elem)
        FROM (
            SELECT DISTINCT ON (d.key) d.elem
            FROM (
                SELECT
                    COALESCE(NULLIF(e->>'ref', ''), NULLIF(e->>'label', ''), md5(e::text)) AS key,
                    e AS elem,
                    COALESCE(e->>'observed_at', '') AS observed_at
                FROM jsonb_array_elements(normalize_source_references(p_sources)) e
            ) d
            ORDER BY d.key, d.observed_at DESC
        ) d
    ), '[]'::jsonb);
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION source_reinforcement_score(p_source_references JSONB)
RETURNS FLOAT AS $$
DECLARE
    unique_sources INT;
    avg_trust FLOAT;
BEGIN
    WITH elems AS (
        SELECT
            COALESCE(NULLIF(e->>'ref', ''), NULLIF(e->>'label', ''), md5(e::text)) AS key,
            COALESCE((e->>'trust')::float, 0.5) AS trust
        FROM jsonb_array_elements(dedupe_source_references(p_source_references)) e
    )
    SELECT COUNT(DISTINCT key), AVG(trust) INTO unique_sources, avg_trust
    FROM elems;

    IF unique_sources IS NULL OR unique_sources = 0 THEN
        RETURN 0.0;
    END IF;

    avg_trust := COALESCE(avg_trust, 0.5);
    RETURN 1.0 - exp(-0.8 * unique_sources * avg_trust);
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION compute_worldview_alignment(p_memory_id UUID)
RETURNS FLOAT AS $$
DECLARE
    supports_score FLOAT := 0;
    contradicts_score FLOAT := 0;
    alignment FLOAT;
    sql TEXT;
BEGIN
    BEGIN
        sql := format($sql$
            SELECT COALESCE(SUM((strength::text)::float), 0)
            FROM ag_catalog.cypher('memory_graph', $q$
                MATCH (m:MemoryNode {memory_id: %L})-[r:SUPPORTS]->(w:MemoryNode)
                WHERE w.type = 'worldview'
                RETURN r.strength
            $q$) as (strength ag_catalog.agtype)
        $sql$, p_memory_id);
        EXECUTE sql INTO supports_score;
    EXCEPTION WHEN OTHERS THEN supports_score := 0; END;
    BEGIN
        sql := format($sql$
            SELECT COALESCE(SUM((strength::text)::float), 0)
            FROM ag_catalog.cypher('memory_graph', $q$
                MATCH (m:MemoryNode {memory_id: %L})-[r:CONTRADICTS]->(w:MemoryNode)
                WHERE w.type = 'worldview'
                RETURN r.strength
            $q$) as (strength ag_catalog.agtype)
        $sql$, p_memory_id);
        EXECUTE sql INTO contradicts_score;
    EXCEPTION WHEN OTHERS THEN contradicts_score := 0; END;
    supports_score := COALESCE(supports_score, 0);
    contradicts_score := COALESCE(contradicts_score, 0);

    IF (supports_score + contradicts_score) = 0 THEN
        RETURN 0.0;
    END IF;

    alignment := (supports_score - contradicts_score) / (supports_score + contradicts_score);
    RETURN LEAST(1.0, GREATEST(-1.0, alignment));
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION compute_semantic_trust(
    p_confidence FLOAT,
    p_source_references JSONB,
    p_worldview_alignment FLOAT DEFAULT 0.0
)
RETURNS FLOAT AS $$
DECLARE
    base_confidence FLOAT;
    reinforcement FLOAT;
    cap FLOAT;
    effective FLOAT;
    alignment FLOAT;
BEGIN
    base_confidence := LEAST(1.0, GREATEST(0.0, COALESCE(p_confidence, 0.5)));
    reinforcement := source_reinforcement_score(p_source_references);
    cap := 0.15 + 0.85 * reinforcement;
    effective := LEAST(base_confidence, cap);

    alignment := LEAST(1.0, GREATEST(-1.0, COALESCE(p_worldview_alignment, 0.0)));
    IF alignment < 0 THEN
        effective := effective * (1.0 + alignment);
    ELSE
        effective := LEAST(1.0, effective + 0.10 * alignment);
    END IF;

    RETURN LEAST(1.0, GREATEST(0.0, effective));
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION sync_memory_trust(p_memory_id UUID)
RETURNS VOID AS $$
DECLARE
    mtype memory_type;
    conf FLOAT;
    sources JSONB;
    alignment FLOAT;
    computed FLOAT;
    mem_metadata JSONB;
BEGIN
    SELECT type, metadata INTO mtype, mem_metadata FROM memories WHERE id = p_memory_id;
    IF NOT FOUND THEN
        RETURN;
    END IF;

    -- Protected memories (e.g. origin documents) keep their seeded trust:
    -- confidence may still be revised, but derived trust is pinned.
    IF COALESCE((mem_metadata->>'protected')::boolean, FALSE) THEN
        RETURN;
    END IF;

    -- Sources are authority (#83) for worldview too: a belief's trust is its
    -- source's trust — a row may never assert more trust than its provenance
    -- carries.
    IF mtype = 'worldview' THEN
        UPDATE memories
        SET trust_level = LEAST(1.0, GREATEST(0.0, COALESCE(
                (source_attribution->>'trust')::float, 0.5))),
            trust_updated_at = CURRENT_TIMESTAMP
        WHERE id = p_memory_id;
        RETURN;
    END IF;

    IF mtype <> 'semantic' THEN
        RETURN;
    END IF;
    conf := COALESCE((mem_metadata->>'confidence')::float, 0.5);
    sources := mem_metadata->'source_references';

    sources := dedupe_source_references(sources);
    alignment := compute_worldview_alignment(p_memory_id);
    computed := compute_semantic_trust(conf, sources, alignment);

    UPDATE memories
    SET trust_level = computed,
        trust_updated_at = CURRENT_TIMESTAMP,
        source_attribution = CASE
            WHEN (source_attribution = '{}'::jsonb OR source_attribution IS NULL)
                 AND jsonb_typeof(sources) = 'array'
                 AND jsonb_array_length(sources) > 0
            THEN normalize_source_reference(sources->0)
            ELSE source_attribution
        END
    WHERE id = p_memory_id;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION add_semantic_source_reference(
    p_memory_id UUID,
    p_source JSONB
)
RETURNS VOID AS $$
DECLARE
    normalized JSONB;
BEGIN
    normalized := normalize_source_reference(p_source);
    IF normalized = '{}'::jsonb THEN
        RETURN;
    END IF;
    UPDATE memories
    SET metadata = jsonb_set(
            jsonb_set(
                metadata,
                '{source_references}',
                dedupe_source_references(
                    COALESCE(metadata->'source_references', '[]'::jsonb) || jsonb_build_array(normalized)
                )
            ),
            '{last_validated}',
            to_jsonb(CURRENT_TIMESTAMP)
        )
    WHERE id = p_memory_id AND type = 'semantic';

    PERFORM sync_memory_trust(p_memory_id);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION record_memory_correction(
    p_memory_id UUID,
    p_correction TEXT,
    p_scope TEXT DEFAULT 'behavior',
    p_source JSONB DEFAULT '{}'::jsonb,
    p_invalid_precedent BOOLEAN DEFAULT FALSE
)
RETURNS JSONB AS $$
DECLARE
    normalized_source JSONB := normalize_source_reference(COALESCE(p_source, '{}'::jsonb));
    correction_text TEXT := NULLIF(btrim(COALESCE(p_correction, '')), '');
    scope_text TEXT := COALESCE(NULLIF(btrim(COALESCE(p_scope, '')), ''), 'behavior');
    correction JSONB;
    updated memories%ROWTYPE;
BEGIN
    IF p_memory_id IS NULL THEN
        RAISE EXCEPTION 'memory_id is required';
    END IF;
    IF correction_text IS NULL THEN
        RAISE EXCEPTION 'correction is required';
    END IF;

    correction := jsonb_build_object(
        'correction', correction_text,
        'scope', scope_text,
        'invalid_precedent', COALESCE(p_invalid_precedent, FALSE),
        'recorded_at', CURRENT_TIMESTAMP,
        'source', normalized_source
    );

    UPDATE memories
    SET metadata = jsonb_set(
            jsonb_set(
                jsonb_set(
                    COALESCE(metadata, '{}'::jsonb),
                    '{corrections}',
                    COALESCE(metadata->'corrections', '[]'::jsonb) || jsonb_build_array(correction),
                    true
                ),
                '{latest_correction}',
                correction,
                true
            ),
            '{invalid_precedent}',
            to_jsonb(COALESCE((metadata->>'invalid_precedent')::boolean, FALSE) OR COALESCE(p_invalid_precedent, FALSE)),
            true
        ),
        updated_at = CURRENT_TIMESTAMP,
        last_reinforced = CURRENT_TIMESTAMP,
        reinforcement_count = COALESCE(reinforcement_count, 0) + 1
    WHERE id = p_memory_id
      AND status = 'active'
    RETURNING * INTO updated;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'memory % not found', p_memory_id;
    END IF;

    IF COALESCE(p_invalid_precedent, FALSE) THEN
        UPDATE subconscious_units s
        SET metadata = jsonb_set(
                jsonb_set(
                    COALESCE(s.metadata, '{}'::jsonb),
                    '{latest_correction}',
                    correction,
                    true
                ),
                '{invalid_precedent}',
                'true'::jsonb,
                true
            ),
            updated_at = CURRENT_TIMESTAMP
        WHERE EXISTS (
            SELECT 1
            FROM memory_source_units msu
            WHERE msu.memory_id = p_memory_id
              AND msu.subconscious_unit_id = s.id
        );
    END IF;

    PERFORM sync_memory_trust(p_memory_id);

    RETURN jsonb_build_object(
        'memory_id', updated.id::text,
        'status', 'corrected',
        'invalid_precedent', COALESCE((updated.metadata->>'invalid_precedent')::boolean, FALSE),
        'latest_correction', updated.metadata->'latest_correction'
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_memory_truth_profile(p_memory_id UUID)
RETURNS JSONB AS $$
DECLARE
    mtype memory_type;
    base_conf FLOAT;
    sources JSONB;
    reinforcement FLOAT;
    alignment FLOAT;
    trust FLOAT;
    source_count INT;
    mem_metadata JSONB;
BEGIN
    SELECT type, trust_level, metadata INTO mtype, trust, mem_metadata
    FROM memories
    WHERE id = p_memory_id;

    IF NOT FOUND THEN
        RETURN '{}'::jsonb;
    END IF;

    IF mtype = 'semantic' THEN
        base_conf := COALESCE((mem_metadata->>'confidence')::float, 0.5);
        sources := mem_metadata->'source_references';

        sources := dedupe_source_references(sources);
        reinforcement := source_reinforcement_score(sources);
        alignment := compute_worldview_alignment(p_memory_id);
        source_count := COALESCE(jsonb_array_length(sources), 0);

        RETURN jsonb_build_object(
            'type', 'semantic',
            'base_confidence', COALESCE(base_conf, 0.5),
            'trust_level', trust,
            'source_count', source_count,
            'source_reinforcement', reinforcement,
            'worldview_alignment', alignment,
            'sources', sources
        );
    END IF;

    RETURN jsonb_build_object(
        'type', mtype::text,
        'trust_level', trust
    );
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION update_worldview_confidence_from_influences(
    p_worldview_memory_id UUID,
    p_window INTERVAL DEFAULT INTERVAL '30 days',
    p_learning_rate FLOAT DEFAULT 0.05
)
RETURNS VOID AS $$
DECLARE
    delta FLOAT := 0;
    base_conf FLOAT;
    mem_meta JSONB;
BEGIN
    IF p_worldview_memory_id IS NULL THEN
        RETURN;
    END IF;
    SELECT metadata INTO mem_meta FROM memories WHERE id = p_worldview_memory_id AND type = 'worldview';
    IF NOT FOUND THEN RETURN; END IF;

    base_conf := COALESCE((mem_meta->>'confidence')::float, 0.5);
    BEGIN
        EXECUTE format($sql$
            SELECT COALESCE(AVG((strength::text)::float * 0.5), 0)
            FROM ag_catalog.cypher('memory_graph', $q$
                MATCH (m:MemoryNode)-[r:SUPPORTS]->(w:MemoryNode {memory_id: %L})
                RETURN r.strength
            $q$) as (strength ag_catalog.agtype)
        $sql$, p_worldview_memory_id) INTO delta;
    EXCEPTION WHEN OTHERS THEN delta := 0; END;
    UPDATE memories
    SET metadata = jsonb_set(
            metadata,
            '{confidence}',
            to_jsonb(LEAST(1.0, GREATEST(0.0, base_conf + COALESCE(p_learning_rate, 0.05) * COALESCE(delta, 0))))
        ),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_worldview_memory_id;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION create_memory(
    p_type memory_type,
    p_content TEXT,
    p_importance FLOAT DEFAULT 0.5,
    p_source_attribution JSONB DEFAULT NULL,
    p_trust_level FLOAT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS UUID AS $$
DECLARE
    new_memory_id UUID;
    normalized_source JSONB;
    effective_trust FLOAT;
BEGIN
    normalized_source := normalize_source_reference(p_source_attribution);
    IF normalized_source = '{}'::jsonb THEN
        normalized_source := jsonb_build_object(
            'kind',
            CASE
                WHEN p_type = 'semantic' THEN 'unattributed'
                ELSE 'internal'
            END,
            'observed_at', CURRENT_TIMESTAMP
        );
    END IF;

    effective_trust := p_trust_level;
    IF effective_trust IS NULL THEN
        effective_trust := CASE
            WHEN p_type = 'episodic' THEN 0.95
            WHEN p_type = 'semantic' THEN 0.20
            WHEN p_type = 'procedural' THEN 0.70
            WHEN p_type = 'strategic' THEN 0.70
            ELSE 0.50
        END;
    END IF;
    effective_trust := LEAST(1.0, GREATEST(0.0, effective_trust));
    INSERT INTO memories (
        type,
        content,
        embedding,
        embedding_status,
        embedding_attempts,
        importance,
        source_attribution,
        trust_level,
        trust_updated_at,
        metadata
    )
    VALUES (
        p_type,
        p_content,
        NULL,
        'pending',
        0,
        p_importance,
        normalized_source,
        effective_trust,
        CURRENT_TIMESTAMP,
        COALESCE(p_metadata, '{}'::jsonb)
    )
    RETURNING id INTO new_memory_id;
    EXECUTE format(
        'SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
            MERGE (n:MemoryNode {memory_id: %L})
            SET n.type = %L, n.created_at = %L
            RETURN n
        $q$) as (result ag_catalog.agtype)',
        new_memory_id,
        p_type,
        CURRENT_TIMESTAMP
    );

    RETURN new_memory_id;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION create_episodic_memory(
    p_content TEXT,
    p_action_taken JSONB DEFAULT NULL,
    p_context JSONB DEFAULT NULL,
    p_result JSONB DEFAULT NULL,
    p_emotional_valence FLOAT DEFAULT 0.0,
    p_event_time TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    p_importance FLOAT DEFAULT 0.5,
    p_source_attribution JSONB DEFAULT NULL,
    p_trust_level FLOAT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    new_memory_id UUID;
    normalized_source JSONB;
    effective_trust FLOAT;
    meta JSONB;
BEGIN
    normalized_source := normalize_source_reference(p_source_attribution);
    IF normalized_source = '{}'::jsonb THEN
        normalized_source := jsonb_build_object('kind', 'internal', 'observed_at', CURRENT_TIMESTAMP);
    END IF;
    effective_trust := COALESCE(p_trust_level, 0.95);
    meta := jsonb_build_object(
        'action_taken', p_action_taken,
        'context', p_context,
        'result', p_result,
        'emotional_valence', LEAST(1.0, GREATEST(-1.0, COALESCE(p_emotional_valence, 0.0))),
        'event_time', COALESCE(p_event_time, CURRENT_TIMESTAMP),
        'verification_status', NULL
    );

    new_memory_id := create_memory('episodic', p_content, p_importance, normalized_source, effective_trust, meta);

    RETURN new_memory_id;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION create_semantic_memory(
    p_content TEXT,
    p_confidence FLOAT,
    p_category TEXT[] DEFAULT NULL,
    p_related_concepts TEXT[] DEFAULT NULL,
    p_source_references JSONB DEFAULT NULL,
    p_importance FLOAT DEFAULT 0.5,
    p_source_attribution JSONB DEFAULT NULL,
    p_trust_level FLOAT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    new_memory_id UUID;
    normalized_sources JSONB;
    primary_source JSONB;
    base_confidence FLOAT;
    effective_trust FLOAT;
    meta JSONB;
BEGIN
    normalized_sources := dedupe_source_references(p_source_references);
    base_confidence := LEAST(1.0, GREATEST(0.0, COALESCE(p_confidence, 0.5)));

    primary_source := normalize_source_reference(p_source_attribution);
    IF primary_source = '{}'::jsonb AND jsonb_typeof(normalized_sources) = 'array' AND jsonb_array_length(normalized_sources) > 0 THEN
        primary_source := normalize_source_reference(normalized_sources->0);
    END IF;
    IF primary_source = '{}'::jsonb THEN
        primary_source := jsonb_build_object('kind', 'unattributed', 'observed_at', CURRENT_TIMESTAMP);
    END IF;

    effective_trust := COALESCE(p_trust_level, compute_semantic_trust(base_confidence, normalized_sources, 0.0));
    meta := jsonb_build_object(
        'confidence', base_confidence,
        'last_validated', CURRENT_TIMESTAMP,
        'source_references', normalized_sources,
        'contradictions', NULL,
        'category', to_jsonb(p_category),
        'related_concepts', to_jsonb(p_related_concepts)
    );

    new_memory_id := create_memory('semantic', p_content, p_importance, primary_source, effective_trust, meta);

    PERFORM sync_memory_trust(new_memory_id);

    RETURN new_memory_id;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION create_procedural_memory(
    p_content TEXT,
    p_steps JSONB,
    p_prerequisites JSONB DEFAULT NULL,
    p_importance FLOAT DEFAULT 0.5,
    p_source_attribution JSONB DEFAULT NULL,
    p_trust_level FLOAT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    new_memory_id UUID;
    normalized_source JSONB;
    effective_trust FLOAT;
    meta JSONB;
BEGIN
    normalized_source := normalize_source_reference(p_source_attribution);
    IF normalized_source = '{}'::jsonb THEN
        normalized_source := jsonb_build_object('kind', 'internal', 'observed_at', CURRENT_TIMESTAMP);
    END IF;
    effective_trust := COALESCE(p_trust_level, 0.70);
    meta := jsonb_build_object(
        'steps', p_steps,
        'prerequisites', p_prerequisites,
        'success_count', 0,
        'total_attempts', 0,
        'average_duration_seconds', NULL,
        'failure_points', NULL
    );

    new_memory_id := create_memory('procedural', p_content, p_importance, normalized_source, effective_trust, meta);

    RETURN new_memory_id;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION create_strategic_memory(
    p_content TEXT,
    p_pattern_description TEXT,
    p_confidence_score FLOAT,
    p_supporting_evidence JSONB DEFAULT NULL,
    p_context_applicability JSONB DEFAULT NULL,
    p_importance FLOAT DEFAULT 0.5,
    p_source_attribution JSONB DEFAULT NULL,
    p_trust_level FLOAT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    new_memory_id UUID;
    normalized_source JSONB;
    effective_trust FLOAT;
    meta JSONB;
BEGIN
    normalized_source := normalize_source_reference(p_source_attribution);
    IF normalized_source = '{}'::jsonb THEN
        normalized_source := jsonb_build_object('kind', 'internal', 'observed_at', CURRENT_TIMESTAMP);
    END IF;
    effective_trust := COALESCE(p_trust_level, 0.70);
    meta := jsonb_build_object(
        'pattern_description', p_pattern_description,
        'confidence_score', LEAST(1.0, GREATEST(0.0, COALESCE(p_confidence_score, 0.5))),
        'supporting_evidence', p_supporting_evidence,
        'success_metrics', NULL,
        'adaptation_history', NULL,
        'context_applicability', p_context_applicability
    );

    new_memory_id := create_memory('strategic', p_content, p_importance, normalized_source, effective_trust, meta);

    RETURN new_memory_id;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION create_worldview_memory(
    p_content TEXT,
    p_category TEXT DEFAULT 'belief',
    p_confidence FLOAT DEFAULT 0.8,
    p_stability FLOAT DEFAULT 0.7,
    p_importance FLOAT DEFAULT 0.8,
    p_origin TEXT DEFAULT 'discovered',
    p_trigger_patterns JSONB DEFAULT NULL,
    p_response_type TEXT DEFAULT NULL,
    p_response_template TEXT DEFAULT NULL,
    p_emotional_valence FLOAT DEFAULT 0.0,
    p_extra_metadata JSONB DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    new_memory_id UUID;
    normalized_source JSONB;
    effective_trust FLOAT;
    meta JSONB;
BEGIN
    -- Dedup: return existing active worldview memory with same content
    SELECT id INTO new_memory_id
    FROM memories
    WHERE type = 'worldview' AND content = p_content AND status = 'active'
    LIMIT 1;
    IF new_memory_id IS NOT NULL THEN
        RETURN new_memory_id;
    END IF;

    normalized_source := jsonb_build_object('kind', 'internal', 'observed_at', CURRENT_TIMESTAMP);
    effective_trust := LEAST(1.0, GREATEST(0.0, COALESCE(p_stability, 0.7)));
    meta := jsonb_build_object(
        'category', p_category,
        'confidence', LEAST(1.0, GREATEST(0.0, COALESCE(p_confidence, 0.8))),
        'stability', LEAST(1.0, GREATEST(0.0, COALESCE(p_stability, 0.7))),
        'origin', COALESCE(p_origin, 'discovered'),
        'emotional_valence', LEAST(1.0, GREATEST(-1.0, COALESCE(p_emotional_valence, 0.0))),
        'evidence_threshold', 0.9,
        'trigger_patterns', p_trigger_patterns,
        'response_type', p_response_type,
        'response_template', p_response_template
    );
    IF p_extra_metadata IS NOT NULL THEN
        meta := meta || p_extra_metadata;
    END IF;

    new_memory_id := create_memory('worldview', p_content, p_importance, normalized_source, effective_trust, meta);
    BEGIN
        EXECUTE format(
            'SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
                MATCH (s:SelfNode)
                MATCH (m:MemoryNode {memory_id: %L})
                CREATE (s)-[:HAS_BELIEF {category: %L, stability: %s}]->(m)
                RETURN m
            $q$) as (result ag_catalog.agtype)',
            new_memory_id,
            p_category,
            p_stability
        );
        PERFORM upsert_memory_edge('self', 'self', 'HAS_BELIEF', 'memory', new_memory_id::text,
                                   p_stability, p_category, NULL,
                                   jsonb_build_object('category', p_category, 'stability', p_stability));
    EXCEPTION WHEN OTHERS THEN NULL;
    END;

    RETURN new_memory_id;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION create_worldview_belief(
    p_content TEXT,
    p_category TEXT DEFAULT 'belief',
    p_confidence FLOAT DEFAULT 0.8,
    p_stability FLOAT DEFAULT 0.7,
    p_importance FLOAT DEFAULT 0.8,
    p_origin TEXT DEFAULT 'discovered',
    p_evidence_threshold FLOAT DEFAULT 0.7,
    p_emotional_valence FLOAT DEFAULT 0.0,
    p_trigger_patterns TEXT[] DEFAULT NULL,
    p_response_type TEXT DEFAULT NULL,
    p_source_references JSONB DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    new_memory_id UUID;
    normalized_sources JSONB;
    trigger_json JSONB;
    meta_patch JSONB := '{}'::jsonb;
BEGIN
    trigger_json := CASE
        WHEN p_trigger_patterns IS NULL THEN NULL
        ELSE to_jsonb(p_trigger_patterns)
    END;

    new_memory_id := create_worldview_memory(
        p_content,
        p_category,
        p_confidence,
        p_stability,
        p_importance,
        p_origin,
        trigger_json,
        p_response_type,
        NULL,
        p_emotional_valence
    );

    IF p_evidence_threshold IS NOT NULL THEN
        meta_patch := meta_patch || jsonb_build_object(
            'evidence_threshold',
            LEAST(1.0, GREATEST(0.0, p_evidence_threshold))
        );
    END IF;

    IF p_source_references IS NOT NULL THEN
        normalized_sources := dedupe_source_references(p_source_references);
        meta_patch := meta_patch || jsonb_build_object('source_references', normalized_sources);
    END IF;

    IF meta_patch <> '{}'::jsonb THEN
        UPDATE memories
        SET metadata = metadata || meta_patch,
            source_attribution = CASE
                WHEN normalized_sources IS NOT NULL
                     AND jsonb_typeof(normalized_sources) = 'array'
                     AND jsonb_array_length(normalized_sources) > 0
                THEN normalize_source_reference(normalized_sources->0)
                ELSE source_attribution
            END,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = new_memory_id;
    END IF;

    RETURN new_memory_id;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION update_identity_belief(
    p_worldview_id UUID,
    p_new_content TEXT,
    p_evidence_memory_id UUID,
    p_force BOOLEAN DEFAULT FALSE
)
RETURNS BOOLEAN AS $$
DECLARE
    current_stability FLOAT;
    stable_threshold FLOAT := 0.8;
BEGIN
    SELECT COALESCE((metadata->>'stability')::float, 0.7)
    INTO current_stability
    FROM memories
    WHERE id = p_worldview_id
      AND type = 'worldview'
      AND metadata->>'category' = 'self';

    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;
    IF current_stability > stable_threshold AND NOT COALESCE(p_force, FALSE) THEN
        PERFORM create_strategic_memory(
            'Identity belief challenged but stable',
            'Identity stability check',
            0.7,
            jsonb_build_object(
                'worldview_id', p_worldview_id,
                'evidence_memory_id', p_evidence_memory_id
            )
        );
        RETURN FALSE;
    END IF;

    UPDATE memories
    SET content = p_new_content,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_worldview_id AND type = 'worldview';

    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION batch_create_memories(p_items JSONB)
RETURNS UUID[] AS $$
DECLARE
    ids UUID[] := ARRAY[]::UUID[];
    item JSONB;
    mtype memory_type;
    content TEXT;
    importance FLOAT;
    new_id UUID;
    idx INT := 0;
BEGIN
    IF p_items IS NULL OR jsonb_typeof(p_items) <> 'array' THEN
        RETURN ids;
    END IF;

    FOR item IN SELECT * FROM jsonb_array_elements(p_items)
    LOOP
        idx := idx + 1;
        mtype := NULLIF(item->>'type', '')::memory_type;
        content := NULLIF(item->>'content', '');
        IF content IS NULL OR mtype IS NULL THEN
            RAISE EXCEPTION 'batch_create_memories: item % missing required fields', idx;
        END IF;
        importance := COALESCE(NULLIF(item->>'importance', '')::float, 0.5);

        IF mtype = 'episodic' THEN
            new_id := create_episodic_memory(
                content,
                item->'action_taken',
                item->'context',
                item->'result',
                COALESCE(NULLIF(item->>'emotional_valence', '')::float, 0.0),
                COALESCE(NULLIF(item->>'event_time', '')::timestamptz, CURRENT_TIMESTAMP),
                importance,
                item->'source_attribution',
                NULLIF(item->>'trust_level', '')::float
            );
        ELSIF mtype = 'semantic' THEN
            new_id := create_semantic_memory(
                content,
                COALESCE(NULLIF(item->>'confidence', '')::float, 0.8),
                CASE WHEN item ? 'category' THEN ARRAY(SELECT jsonb_array_elements_text(item->'category')) ELSE NULL END,
                CASE WHEN item ? 'related_concepts' THEN ARRAY(SELECT jsonb_array_elements_text(item->'related_concepts')) ELSE NULL END,
                item->'source_references',
                importance,
                item->'source_attribution',
                NULLIF(item->>'trust_level', '')::float
            );
        ELSIF mtype = 'procedural' THEN
            new_id := create_procedural_memory(
                content,
                COALESCE(item->'steps', jsonb_build_object('steps', '[]'::jsonb)),
                item->'prerequisites',
                importance,
                item->'source_attribution',
                NULLIF(item->>'trust_level', '')::float
            );
        ELSIF mtype = 'strategic' THEN
            new_id := create_strategic_memory(
                content,
                COALESCE(NULLIF(item->>'pattern_description', ''), content),
                COALESCE(NULLIF(item->>'confidence_score', '')::float, 0.8),
                item->'supporting_evidence',
                item->'context_applicability',
                importance,
                item->'source_attribution',
                NULLIF(item->>'trust_level', '')::float
            );
        ELSE
            RAISE EXCEPTION 'batch_create_memories: item % invalid type %', idx, mtype::text;
        END IF;

        IF new_id IS NULL THEN
            RAISE EXCEPTION 'batch_create_memories: item % failed to create memory', idx;
        END IF;
        ids := array_append(ids, new_id);
    END LOOP;

    RETURN ids;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION create_memory_with_embedding(
    p_type memory_type,
    p_content TEXT,
    p_embedding vector,
    p_importance FLOAT DEFAULT 0.5,
    p_source_attribution JSONB DEFAULT NULL,
    p_trust_level FLOAT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS UUID AS $$
DECLARE
    new_memory_id UUID;
    normalized_source JSONB;
    effective_trust FLOAT;
BEGIN
    IF p_embedding IS NULL THEN
        RAISE EXCEPTION 'embedding must not be NULL';
    END IF;

    normalized_source := normalize_source_reference(p_source_attribution);
    IF normalized_source = '{}'::jsonb THEN
        normalized_source := jsonb_build_object(
            'kind',
            CASE
                WHEN p_type = 'semantic' THEN 'unattributed'
                ELSE 'internal'
            END,
            'observed_at', CURRENT_TIMESTAMP
        );
    END IF;

    effective_trust := p_trust_level;
    IF effective_trust IS NULL THEN
        effective_trust := CASE
            WHEN p_type = 'episodic' THEN 0.95
            WHEN p_type = 'semantic' THEN 0.20
            WHEN p_type = 'procedural' THEN 0.70
            WHEN p_type = 'strategic' THEN 0.70
            ELSE 0.50
        END;
    END IF;
    effective_trust := LEAST(1.0, GREATEST(0.0, effective_trust));

    INSERT INTO memories (
        type,
        content,
        embedding,
        embedded_at,
        embedding_model,
        embedding_status,
        embedding_attempts,
        importance,
        source_attribution,
        trust_level,
        trust_updated_at,
        metadata
    )
    VALUES (
        p_type,
        p_content,
        p_embedding,
        CURRENT_TIMESTAMP,
        get_config_text('embedding.model_id'),
        'embedded',
        1,
        p_importance,
        normalized_source,
        effective_trust,
        CURRENT_TIMESTAMP,
        COALESCE(p_metadata, '{}'::jsonb)
    )
    RETURNING id INTO new_memory_id;

    EXECUTE format(
        'SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
            CREATE (n:MemoryNode {memory_id: %L, type: %L, created_at: %L})
            RETURN n
        $q$) as (result ag_catalog.agtype)',
        new_memory_id,
        p_type,
        CURRENT_TIMESTAMP
    );

    RETURN new_memory_id;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION batch_create_memories_with_embeddings(
    p_type memory_type,
    p_contents TEXT[],
    p_embeddings JSONB,
    p_importance FLOAT DEFAULT 0.5
)
RETURNS UUID[] AS $$
DECLARE
    ids UUID[] := ARRAY[]::UUID[];
    n INT;
    i INT;
    expected_dim INT;
    emb_vec vector;
    emb_json JSONB;
    emb_arr FLOAT4[];
    new_id UUID;
    default_meta JSONB;
BEGIN
    n := COALESCE(array_length(p_contents, 1), 0);
    IF n = 0 THEN
        RETURN ids;
    END IF;

    IF p_embeddings IS NULL OR jsonb_typeof(p_embeddings) <> 'array' THEN
        RAISE EXCEPTION 'embeddings must be a JSON array';
    END IF;
    IF jsonb_array_length(p_embeddings) <> n THEN
        RAISE EXCEPTION 'contents and embeddings length mismatch';
    END IF;

    expected_dim := embedding_dimension();

    FOR i IN 1..n LOOP
        IF p_contents[i] IS NULL OR p_contents[i] = '' THEN
            CONTINUE;
        END IF;

        emb_json := p_embeddings->(i - 1);
        IF emb_json IS NULL OR jsonb_typeof(emb_json) <> 'array' THEN
            RAISE EXCEPTION 'embedding % must be a JSON array', i;
        END IF;

        SELECT ARRAY_AGG(value::float4) INTO emb_arr
        FROM jsonb_array_elements_text(emb_json) value;

        IF COALESCE(array_length(emb_arr, 1), 0) <> expected_dim THEN
            RAISE EXCEPTION 'embedding dimension mismatch: expected %, got %', expected_dim, COALESCE(array_length(emb_arr, 1), 0);
        END IF;

        emb_vec := (emb_arr::float4[])::vector;
        IF p_type = 'episodic' THEN
            default_meta := jsonb_build_object(
                'action_taken', NULL,
                'context', jsonb_build_object('type', 'raw_batch'),
                'result', NULL,
                'emotional_valence', 0.0,
                'verification_status', NULL,
                'event_time', CURRENT_TIMESTAMP
            );
        ELSIF p_type = 'semantic' THEN
            default_meta := jsonb_build_object(
                'confidence', 0.8,
                'last_validated', CURRENT_TIMESTAMP,
                'source_references', '[]'::jsonb,
                'contradictions', NULL,
                'category', NULL,
                'related_concepts', NULL
            );
        ELSIF p_type = 'procedural' THEN
            default_meta := jsonb_build_object(
                'steps', '[]'::jsonb,
                'prerequisites', NULL,
                'success_count', 0,
                'total_attempts', 0,
                'average_duration_seconds', NULL,
                'failure_points', NULL
            );
        ELSIF p_type = 'strategic' THEN
            default_meta := jsonb_build_object(
                'pattern_description', p_contents[i],
                'supporting_evidence', NULL,
                'confidence_score', 0.8,
                'success_metrics', NULL,
                'adaptation_history', NULL,
                'context_applicability', NULL
            );
        ELSE
            default_meta := '{}'::jsonb;
        END IF;

        new_id := create_memory_with_embedding(p_type, p_contents[i], emb_vec, p_importance, NULL, NULL, default_meta);

        IF p_type = 'semantic' THEN
            PERFORM sync_memory_trust(new_id);
        END IF;

        ids := array_append(ids, new_id);
    END LOOP;

    RETURN ids;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION search_similar_memories(
    p_query_text TEXT,
    p_limit INT DEFAULT 10,
    p_memory_types memory_type[] DEFAULT NULL,
    p_min_importance FLOAT DEFAULT 0.0
) RETURNS TABLE (
    memory_id UUID,
    content TEXT,
    type memory_type,
    similarity FLOAT,
    importance FLOAT
) AS $$
DECLARE
    query_embedding vector;
BEGIN
    query_embedding := (get_embedding(ARRAY[ensure_embedding_prefix(p_query_text, 'search_query')]))[1];
    
    RETURN QUERY
    WITH candidates AS MATERIALIZED (
        SELECT m.id, m.content, m.type, m.embedding, m.importance
        FROM memories m
        WHERE m.status = 'active'
          AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
          AND m.embedding_status = 'embedded'
          AND m.embedding IS NOT NULL
          AND (p_memory_types IS NULL OR m.type = ANY(p_memory_types))
          AND m.importance >= p_min_importance
    )
    SELECT
        c.id,
        c.content,
        c.type,
        1 - (c.embedding <=> query_embedding) as similarity,
        c.importance
    FROM candidates c
    ORDER BY c.embedding <=> query_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION assign_memory_to_clusters(
    p_memory_id UUID,
    p_max_clusters INT DEFAULT 3
) RETURNS VOID AS $$
DECLARE
    memory_embedding vector;
    cluster_record RECORD;
    similarity_threshold FLOAT := 0.7;
    assigned_count INT := 0;
    zero_vec vector := array_fill(0.0::float, ARRAY[embedding_dimension()])::vector;
BEGIN
    SELECT embedding INTO memory_embedding
    FROM memories
    WHERE id = p_memory_id
      AND embedding_status = 'embedded';
    IF memory_embedding IS NULL THEN
        RETURN;
    END IF;

    FOR cluster_record IN
        SELECT id, 1 - (centroid_embedding <=> memory_embedding) as similarity
        FROM clusters
        WHERE centroid_embedding IS NOT NULL
          AND centroid_embedding <> zero_vec
        ORDER BY centroid_embedding <=> memory_embedding
        LIMIT 50
    LOOP
        IF cluster_record.similarity >= similarity_threshold AND assigned_count < p_max_clusters THEN
            PERFORM link_memory_to_cluster_graph(p_memory_id, cluster_record.id, cluster_record.similarity);
            assigned_count := assigned_count + 1;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION recalculate_cluster_centroid(p_cluster_id UUID)
RETURNS VOID AS $$
DECLARE
    new_centroid vector;
BEGIN
    SELECT AVG(m.embedding)::vector
    INTO new_centroid
    FROM memories m
    JOIN get_cluster_members_graph(p_cluster_id) gcm ON m.id = gcm.memory_id
    WHERE m.status = 'active'
      AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
      AND gcm.membership_strength > 0.3;

    UPDATE clusters
    SET centroid_embedding = new_centroid,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_cluster_id;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION create_memory_relationship(
    p_from_id UUID,
    p_to_id UUID,
    p_relationship_type graph_edge_type,
    p_properties JSONB DEFAULT '{}'
) RETURNS VOID AS $$
BEGIN
    EXECUTE format(
        'SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
            MATCH (a:MemoryNode {memory_id: %L}), (b:MemoryNode {memory_id: %L})
            CREATE (a)-[r:%s %s]->(b)
            RETURN r
        $q$) as (result ag_catalog.agtype)',
        p_from_id,
        p_to_id,
        p_relationship_type,
        CASE WHEN p_properties = '{}'::jsonb 
             THEN '' 
             ELSE format('{%s}',
                  (SELECT string_agg(format('%I: %s', key, value), ', ')
                   FROM jsonb_each(p_properties)))
        END
    );
    -- Dual-write to the relational edge substrate (db/44) for subgraph reasoning.
    PERFORM upsert_memory_edge(p_from_id, p_to_id, p_relationship_type, p_properties);
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION auto_check_worldview_alignment()
RETURNS TRIGGER AS $$
DECLARE
    min_support FLOAT;
    min_contradict FLOAT;
    sim FLOAT;
    w RECORD;
BEGIN
    IF NEW.type <> 'semantic' THEN
        RETURN NEW;
    END IF;
    IF NEW.embedding IS NULL OR NEW.embedding_status <> 'embedded' THEN
        RETURN NEW;
    END IF;

    min_support := COALESCE(get_config_float('memory.worldview_support_threshold'), 0.8);
    min_contradict := COALESCE(get_config_float('memory.worldview_contradict_threshold'), -0.5);

    BEGIN
        FOR w IN
            SELECT id, embedding
            FROM memories
            WHERE type = 'worldview'
              AND status = 'active'
              AND embedding_status = 'embedded'
              AND embedding IS NOT NULL
            ORDER BY embedding <=> NEW.embedding
            LIMIT 10
        LOOP
            sim := 1 - (w.embedding <=> NEW.embedding);
            IF sim >= min_support THEN
                PERFORM create_memory_relationship(
                    NEW.id,
                    w.id,
                    'SUPPORTS',
                    jsonb_build_object('strength', sim, 'source', 'auto_alignment')
                );
            ELSIF sim <= min_contradict THEN
                PERFORM create_memory_relationship(
                    NEW.id,
                    w.id,
                    'CONTRADICTS',
                    jsonb_build_object('strength', ABS(sim), 'source', 'auto_alignment')
                );
            END IF;
        END LOOP;
    EXCEPTION
        WHEN OTHERS THEN
            NULL;
    END;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION link_memory_to_concept(
    p_memory_id UUID,
    p_concept_name TEXT,
    p_strength FLOAT DEFAULT 1.0
) RETURNS BOOLEAN AS $$
BEGIN
    EXECUTE format(
        'SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
            MERGE (c:ConceptNode {name: %L})
            RETURN c
        $q$) as (result ag_catalog.agtype)',
        p_concept_name
    );
    EXECUTE format(
        'SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
            MATCH (m:MemoryNode {memory_id: %L})
            MATCH (c:ConceptNode {name: %L})
            CREATE (m)-[:INSTANCE_OF {strength: %s}]->(c)
            RETURN m
        $q$) as (result ag_catalog.agtype)',
        p_memory_id,
        p_concept_name,
        p_strength
    );
    PERFORM upsert_memory_edge('memory', p_memory_id::text, 'INSTANCE_OF', 'concept', p_concept_name,
                               p_strength, NULL, NULL, jsonb_build_object('strength', p_strength));
    RETURN TRUE;
EXCEPTION
    WHEN OTHERS THEN
        RETURN FALSE;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION create_concept(
    p_name TEXT,
    p_description TEXT DEFAULT NULL,
    p_depth INT DEFAULT NULL
)
RETURNS BOOLEAN AS $$
DECLARE
    desc_literal TEXT;
    depth_literal TEXT;
BEGIN
    IF p_name IS NULL OR btrim(p_name) = '' THEN
        RETURN FALSE;
    END IF;

    desc_literal := CASE WHEN p_description IS NULL THEN 'NULL' ELSE quote_literal(p_description) END;
    depth_literal := CASE WHEN p_depth IS NULL THEN 'NULL' ELSE p_depth::text END;

    EXECUTE format('SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
        MERGE (c:ConceptNode {name: %L})
        SET c.description = COALESCE(%s, c.description),
            c.depth = COALESCE(%s, c.depth)
        RETURN c
    $q$) as (result ag_catalog.agtype)', p_name, desc_literal, depth_literal);

    RETURN TRUE;
EXCEPTION WHEN OTHERS THEN
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION link_concept_parent(
    p_child_name TEXT,
    p_parent_name TEXT
)
RETURNS BOOLEAN AS $$
BEGIN
    IF p_child_name IS NULL OR btrim(p_child_name) = ''
       OR p_parent_name IS NULL OR btrim(p_parent_name) = '' THEN
        RETURN FALSE;
    END IF;

    PERFORM create_concept(p_child_name);
    PERFORM create_concept(p_parent_name);

    EXECUTE format('SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
        MATCH (child:ConceptNode {name: %L})
        MATCH (parent:ConceptNode {name: %L})
        MERGE (parent)-[:PARENT_OF]->(child)
        RETURN parent
    $q$) as (result ag_catalog.agtype)', p_child_name, p_parent_name);

    PERFORM upsert_memory_edge('concept', p_parent_name, 'PARENT_OF', 'concept', p_child_name, 1.0);
    RETURN TRUE;
EXCEPTION WHEN OTHERS THEN
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION touch_working_memory(p_ids UUID[])
RETURNS VOID AS $$
BEGIN
    IF p_ids IS NULL OR array_length(p_ids, 1) IS NULL THEN
        RETURN;
    END IF;

    UPDATE working_memory
    SET access_count = access_count + 1,
        last_accessed = CURRENT_TIMESTAMP
    WHERE id = ANY(p_ids);
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION promote_working_memory_to_episodic(
    p_working_memory_id UUID,
    p_importance FLOAT DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    wm RECORD;
    new_id UUID;
    affect JSONB;
    v_valence FLOAT;
    meta JSONB;
BEGIN
    SELECT * INTO wm FROM working_memory WHERE id = p_working_memory_id;
    IF NOT FOUND THEN
        RETURN NULL;
    END IF;

    affect := get_current_affective_state();
    BEGIN
        v_valence := NULLIF(affect->>'valence', '')::float;
    EXCEPTION
        WHEN OTHERS THEN
            v_valence := 0.0;
    END;
    v_valence := LEAST(1.0, GREATEST(-1.0, COALESCE(v_valence, 0.0)));
    meta := jsonb_build_object(
        'action_taken', NULL,
        'context', jsonb_build_object(
            'from_working_memory_id', wm.id,
            'promoted_at', CURRENT_TIMESTAMP,
            'working_memory_created_at', wm.created_at,
            'working_memory_expiry', wm.expiry,
            'source_attribution', wm.source_attribution
        ),
        'result', NULL,
        'emotional_valence', v_valence,
        'verification_status', NULL,
        'event_time', wm.created_at
    );

    new_id := create_memory_with_embedding(
        'episodic'::memory_type,
        wm.content,
        wm.embedding,
        COALESCE(p_importance, wm.importance, 0.4),
        wm.source_attribution,
        wm.trust_level,
        meta
    );

    RETURN new_id;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION cleanup_working_memory(
    p_min_importance_to_promote FLOAT DEFAULT 0.75,
    p_min_accesses_to_promote INT DEFAULT 3
)
RETURNS JSONB AS $$
DECLARE
    promoted UUID[] := ARRAY[]::uuid[];
    rec RECORD;
    deleted_count INT := 0;
BEGIN
    FOR rec IN
        SELECT id, importance, access_count, promote_to_long_term
        FROM working_memory
        WHERE expiry < CURRENT_TIMESTAMP
    LOOP
        IF COALESCE(rec.promote_to_long_term, false)
           OR COALESCE(rec.importance, 0) >= COALESCE(p_min_importance_to_promote, 0.75)
           OR COALESCE(rec.access_count, 0) >= COALESCE(p_min_accesses_to_promote, 3)
        THEN
            promoted := array_append(promoted, promote_working_memory_to_episodic(rec.id, rec.importance));
        END IF;
    END LOOP;

    WITH deleted AS (
        DELETE FROM working_memory
        WHERE expiry < CURRENT_TIMESTAMP
        RETURNING 1
    )
    SELECT COUNT(*) INTO deleted_count FROM deleted;

    RETURN jsonb_build_object(
        'deleted_count', COALESCE(deleted_count, 0),
        'promoted_count', COALESCE(array_length(promoted, 1), 0),
        'promoted_ids', COALESCE(to_jsonb(promoted), '[]'::jsonb)
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION add_to_working_memory(
    p_content TEXT,
    p_expiry INTERVAL DEFAULT INTERVAL '1 hour',
    p_importance FLOAT DEFAULT 0.3,
    p_source_attribution JSONB DEFAULT NULL,
    p_trust_level FLOAT DEFAULT NULL,
    p_promote_to_long_term BOOLEAN DEFAULT FALSE
) RETURNS UUID AS $$
	DECLARE
	    new_id UUID;
	    embedding_vec vector;
	    normalized_source JSONB;
	    effective_trust FLOAT;
	BEGIN
	    embedding_vec := (get_embedding(ARRAY[p_content]))[1];

	    normalized_source := normalize_source_reference(p_source_attribution);
	    IF normalized_source = '{}'::jsonb THEN
	        normalized_source := jsonb_build_object('kind', 'internal', 'observed_at', CURRENT_TIMESTAMP);
	    END IF;
	    effective_trust := p_trust_level;
	    IF effective_trust IS NULL THEN
	        effective_trust := 0.8;
	    END IF;
	    effective_trust := LEAST(1.0, GREATEST(0.0, effective_trust));

	    INSERT INTO working_memory (content, embedding, importance, source_attribution, trust_level, promote_to_long_term, expiry)
	    VALUES (
	        p_content,
	        embedding_vec,
	        LEAST(1.0, GREATEST(0.0, COALESCE(p_importance, 0.3))),
	        normalized_source,
	        effective_trust,
	        COALESCE(p_promote_to_long_term, false),
	        CURRENT_TIMESTAMP + p_expiry
	    )
	    RETURNING id INTO new_id;
	    
	    RETURN new_id;
	END;
	$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION search_working_memory(
    p_query_text TEXT,
    p_limit INT DEFAULT 5
) RETURNS TABLE (
    memory_id UUID,
    content TEXT,
    similarity FLOAT,
    created_at TIMESTAMPTZ
) AS $$
	DECLARE
	    query_embedding vector;
	    zero_vec vector;
	BEGIN
	    query_embedding := (get_embedding(ARRAY[ensure_embedding_prefix(p_query_text, 'search_query')]))[1];
	    zero_vec := array_fill(0.0::float, ARRAY[embedding_dimension()])::vector;
	    PERFORM cleanup_working_memory();
	    
	    RETURN QUERY
	    WITH ranked AS (
	        SELECT
	            wm.id,
	            wm.content AS content_text,
	            1 - (wm.embedding <=> query_embedding) as similarity,
	            wm.created_at,
	            (wm.embedding <=> query_embedding) as dist
	        FROM working_memory wm
	        WHERE wm.embedding IS NOT NULL
	          AND wm.embedding <> zero_vec
	        ORDER BY wm.embedding <=> query_embedding
	        LIMIT p_limit
	    ),
	    touched AS (
	        UPDATE working_memory wm
	        SET access_count = access_count + 1,
	            last_accessed = CURRENT_TIMESTAMP
	        WHERE wm.id IN (SELECT id FROM ranked)
	        RETURNING wm.id
	    )
	    SELECT ranked.id AS memory_id, ranked.content_text AS content, ranked.similarity, ranked.created_at
	    FROM ranked
	    ORDER BY ranked.dist;
	END;
	$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION cleanup_embedding_cache(
    p_older_than INTERVAL DEFAULT INTERVAL '7 days'
) RETURNS INT AS $$
DECLARE
    deleted_count INT;
BEGIN
    WITH deleted AS (
        DELETE FROM embedding_cache
        WHERE created_at < CURRENT_TIMESTAMP - p_older_than
        RETURNING 1
    )
    SELECT COUNT(*) INTO deleted_count FROM deleted;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Hybrid recall: combines vector similarity (fast_recall) with Postgres full-text search (tsvector)
-- Returns merged, deduplicated results ranked by a combined score.
CREATE OR REPLACE FUNCTION recall_hybrid(
    p_query_text TEXT,
    p_limit INT DEFAULT 10,
    p_vector_weight FLOAT DEFAULT 0.6,
    p_fts_weight FLOAT DEFAULT 0.4
) RETURNS TABLE (
    memory_id UUID,
    content TEXT,
    memory_type memory_type,
    score FLOAT,
    source TEXT,
    importance FLOAT,
    trust_level FLOAT,
    source_attribution JSONB
) AS $$
DECLARE
    fts_query tsquery;
BEGIN
    -- Build tsquery from natural language
    BEGIN
        fts_query := websearch_to_tsquery('english', p_query_text);
    EXCEPTION WHEN OTHERS THEN
        fts_query := plainto_tsquery('english', p_query_text);
    END;

    RETURN QUERY
    WITH
    -- Vector search results via fast_recall
    vector_hits AS (
        SELECT fr.memory_id, fr.content, fr.memory_type, fr.score AS vector_score, fr.source
        FROM fast_recall(p_query_text, p_limit * 2) fr
    ),
    -- Full-text search results
    fts_hits AS (
        SELECT
            m.id AS memory_id,
            m.content,
            m.type AS memory_type,
            ts_rank_cd(to_tsvector('english', m.content), fts_query)::float AS fts_score,
            'fts'::text AS source
        FROM memories m
        WHERE m.status = 'active'
          AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
          AND to_tsvector('english', m.content) @@ fts_query
        ORDER BY fts_score DESC
        LIMIT p_limit * 2
    ),
    -- Merge and deduplicate
    merged AS (
        SELECT
            COALESCE(v.memory_id, f.memory_id) AS mem_id,
            COALESCE(v.content, f.content) AS mem_content,
            COALESCE(v.memory_type, f.memory_type) AS mem_type,
            -- Normalize and combine scores. FTS-only hits keep newly written
            -- async-embedding memories visible before the worker vectorizes
            -- them; the match still has to satisfy Postgres' tsquery.
            CASE
                WHEN v.memory_id IS NULL AND f.memory_id IS NOT NULL THEN
                    LEAST(1.0, GREATEST(0.35, COALESCE(f.fts_score, 0.0) + p_fts_weight))
                ELSE
                    COALESCE(v.vector_score, 0.0) * p_vector_weight
                    + COALESCE(f.fts_score, 0.0) * p_fts_weight
            END AS combined_score,
            CASE
                WHEN v.memory_id IS NOT NULL AND f.memory_id IS NOT NULL THEN 'hybrid'
                WHEN v.memory_id IS NOT NULL THEN v.source
                ELSE 'fts'
            END AS hit_source
        FROM vector_hits v
        FULL OUTER JOIN fts_hits f ON v.memory_id = f.memory_id
    )
    SELECT
        mg.mem_id,
        mg.mem_content,
        mg.mem_type,
        mg.combined_score,
        mg.hit_source,
        m.importance,
        m.trust_level,
        m.source_attribution
    FROM merged mg
    JOIN memories m ON m.id = mg.mem_id
    WHERE m.status = 'active'
      AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
    ORDER BY mg.combined_score DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- ============================================================================
-- HMX Slice 0: bootstrap provenance (plans/hmx.md)
--
-- HMX distinguishes what the agent lived from what was seeded at init or
-- imported. `metadata->'provenance'->>'acquisition_mode'` records this;
-- bootstrap entries additionally carry `metadata.replaceable_during_bootstrap`
-- so the empty-target check can treat a freshly-initialized instance as
-- replaceable during port/duplicate.
-- ============================================================================

-- The "created by initialization" predicate. Must stay in sync with
-- reset_persona() (db/10_functions_initialization.sql) and the WHEN clause of
-- trg_hmx_bootstrap_provenance (db/91_triggers.sql), which inlines it because
-- trigger WHEN expressions cannot reference other tables and benefit from
-- staying function-call-free on the hot path.
CREATE OR REPLACE FUNCTION is_initialization_memory(p_metadata JSONB, p_source_attribution JSONB)
RETURNS BOOLEAN AS $$
    SELECT p_metadata->>'origin' = 'initialization'
        OR p_metadata->>'type' = 'initialization'
        OR COALESCE(p_source_attribution->>'source', '') = 'initialization';
$$ LANGUAGE sql IMMUTABLE;

-- Trigger body: materialize bootstrap provenance on init-created rows the
-- moment they are created (or first marked as init-created). Rows already
-- deliberately transformed (non-empty change_history) are earned state.
CREATE OR REPLACE FUNCTION hmx_tag_bootstrap_provenance()
RETURNS TRIGGER AS $$
BEGIN
    IF COALESCE(jsonb_typeof(NEW.metadata->'change_history'), '') = 'array'
       AND jsonb_array_length(NEW.metadata->'change_history') > 0 THEN
        NEW.metadata := NEW.metadata || jsonb_build_object(
            'provenance', jsonb_build_object('acquisition_mode', 'experienced'));
    ELSE
        NEW.metadata := NEW.metadata || jsonb_build_object(
            'provenance', jsonb_build_object('acquisition_mode', 'bootstrap'),
            'replaceable_during_bootstrap', true);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- One-shot backfill for rows that predate the trigger. Safe to re-run: only
-- touches rows with no provenance. Returns per-class counts for diagnostics.
CREATE OR REPLACE FUNCTION hmx_backfill_provenance()
RETURNS JSONB AS $$
DECLARE
    n_bootstrap INT;
    n_experienced_init INT;
    n_experienced INT;
BEGIN
    -- Init-created, never deliberately transformed -> bootstrap (replaceable).
    UPDATE memories
    SET metadata = metadata || jsonb_build_object(
            'provenance', jsonb_build_object('acquisition_mode', 'bootstrap'),
            'replaceable_during_bootstrap', true)
    WHERE metadata->'provenance' IS NULL
      AND is_initialization_memory(metadata, source_attribution)
      AND NOT (COALESCE(jsonb_typeof(metadata->'change_history'), '') = 'array'
               AND jsonb_array_length(metadata->'change_history') > 0);
    GET DIAGNOSTICS n_bootstrap = ROW_COUNT;

    -- Init-created but transformed since -> the agent has made it its own.
    UPDATE memories
    SET metadata = metadata || jsonb_build_object(
            'provenance', jsonb_build_object('acquisition_mode', 'experienced'))
    WHERE metadata->'provenance' IS NULL
      AND is_initialization_memory(metadata, source_attribution);
    GET DIAGNOSTICS n_experienced_init = ROW_COUNT;

    -- Everything else was lived.
    UPDATE memories
    SET metadata = metadata || jsonb_build_object(
            'provenance', jsonb_build_object('acquisition_mode', 'experienced'))
    WHERE metadata->'provenance' IS NULL;
    GET DIAGNOSTICS n_experienced = ROW_COUNT;

    RETURN jsonb_build_object(
        'bootstrap', n_bootstrap,
        'experienced_from_init', n_experienced_init,
        'experienced', n_experienced);
END;
$$ LANGUAGE plpgsql;

SET check_function_bodies = on;
