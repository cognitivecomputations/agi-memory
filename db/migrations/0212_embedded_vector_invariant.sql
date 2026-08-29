-- Make embedding_status='embedded' the sole read-path validity predicate.
-- This restores the partial HNSW index path and enforces the invariant at writes.
SET search_path = public, ag_catalog, "$user";

ALTER TABLE memories
    DROP CONSTRAINT IF EXISTS memories_embedded_vector_valid;
ALTER TABLE memories
    ADD CONSTRAINT memories_embedded_vector_valid CHECK (
        embedding_status <> 'embedded'
        OR (embedding IS NOT NULL AND vector_norm(embedding) > 0)
    );

CREATE OR REPLACE FUNCTION recompute_neighborhood(
    p_memory_id UUID,
    p_neighbor_count INT DEFAULT 20,
    p_min_similarity FLOAT DEFAULT 0.5
)
RETURNS VOID AS $$
DECLARE
    memory_emb vector;
    neighbors JSONB;
BEGIN
    SELECT embedding INTO memory_emb
    FROM memories
    WHERE id = p_memory_id
      AND status = 'active'
      AND embedding_status = 'embedded';

    IF memory_emb IS NULL THEN
        RETURN;
    END IF;

    SELECT jsonb_object_agg(id::text, round(similarity::numeric, 4))
    INTO neighbors
    FROM (
        SELECT m.id, 1 - (m.embedding <=> memory_emb) as similarity
        FROM memories m
        WHERE m.id != p_memory_id
          AND m.status = 'active'
          AND m.embedding_status = 'embedded'
          AND m.embedding IS NOT NULL
        ORDER BY m.embedding <=> memory_emb
        LIMIT p_neighbor_count
    ) sub
    WHERE similarity >= p_min_similarity;

    INSERT INTO memory_neighborhoods (memory_id, neighbors, computed_at, is_stale)
    VALUES (p_memory_id, COALESCE(neighbors, '{}'::jsonb), CURRENT_TIMESTAMP, FALSE)
    ON CONFLICT (memory_id) DO UPDATE SET
        neighbors = EXCLUDED.neighbors,
        computed_at = EXCLUDED.computed_at,
        is_stale = FALSE;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION sense_memory_availability(
    p_query TEXT,
    p_query_embedding vector DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    query_emb vector;
    estimated_count INT;
    top_similarity FLOAT;
    activation_id UUID;
BEGIN
    query_emb := COALESCE(p_query_embedding, (get_embedding(ARRAY[ensure_embedding_prefix(p_query, 'search_query')]))[1]);

    SELECT
        COUNT(*),
        MAX(1 - (embedding <=> query_emb))
    INTO estimated_count, top_similarity
    FROM memories
    WHERE status = 'active'
      AND (valid_until IS NULL OR valid_until > CURRENT_TIMESTAMP)
      AND embedding_status = 'embedded'
      AND embedding IS NOT NULL
      AND (1 - (embedding <=> query_emb)) > 0.5
    LIMIT 100;

    INSERT INTO memory_activation (
        query_embedding,
        query_text,
        estimated_matches,
        activation_strength
    ) VALUES (
        query_emb,
        p_query,
        estimated_count,
        COALESCE(top_similarity, 0)
    )
    RETURNING id INTO activation_id;

    RETURN jsonb_build_object(
        'feeling', CASE
            WHEN estimated_count = 0 THEN 'nothing'
            WHEN estimated_count <= 2 THEN 'vague'
            WHEN estimated_count <= 5 THEN 'something'
            WHEN estimated_count <= 10 THEN 'familiar'
            ELSE 'rich'
        END,
        'estimated_count', estimated_count,
        'strongest_match', top_similarity,
        'activation_id', activation_id,
        'description', CASE
            WHEN estimated_count = 0 THEN 'I don''t think I know anything about this'
            WHEN top_similarity > 0.8 THEN 'I know this well - let me recall'
            WHEN top_similarity > 0.6 THEN 'This feels familiar - I should be able to remember'
            WHEN estimated_count > 0 THEN 'I might know something about this - it''s not coming immediately'
            ELSE 'I don''t think I know anything about this'
        END
    );
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

CREATE OR REPLACE FUNCTION recmem_subconscious_vector_hits(
    p_query_embedding vector,
    p_limit INT DEFAULT 10,
    p_exclude_sensitive BOOLEAN DEFAULT FALSE,
    p_zero_vec vector DEFAULT NULL
) RETURNS TABLE (
    tier TEXT,
    item_id UUID,
    content TEXT,
    memory_type TEXT,
    score FLOAT,
    source_unit_ids UUID[],
    source_attribution JSONB,
    created_at TIMESTAMPTZ,
    trust_level FLOAT,
    fidelity FLOAT,
    strength FLOAT,
    emotional_intensity FLOAT,
    confidence FLOAT,
    retrieval_source TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH chunk_best AS (
        SELECT DISTINCT ON (s.id)
            s.id AS item_id,
            CASE
                WHEN length(s.content) > length(c.content) + 200 THEN
                    concat_ws(E'\n',
                        '[Matching RecMem span: chunk ' || c.chunk_index::text
                            || ', chars ' || c.char_start::text || '-' || c.char_end::text
                            || ' of unit ' || s.id::text || ']',
                        '',
                        c.content
                    )
                ELSE s.content
            END AS content,
            (1 - (c.embedding <=> p_query_embedding))::float AS score,
            ARRAY[s.id]::uuid[] AS source_unit_ids,
            COALESCE(s.source_attribution, '{}'::jsonb)
                || jsonb_build_object(
                    'recmem_embedding_chunk',
                    jsonb_build_object(
                        'chunk_id', c.id::text,
                        'unit_id', s.id::text,
                        'chunk_index', c.chunk_index,
                        'char_start', c.char_start,
                        'char_end', c.char_end,
                        'chunk_count', chunk_stats.chunk_count
                    )
                ) AS source_attribution,
            s.created_at,
            s.trust_level,
            s.metadata,
            'chunk_vector'::text AS retrieval_source
        FROM subconscious_unit_embedding_chunks c
        JOIN subconscious_units s ON s.id = c.unit_id
        CROSS JOIN LATERAL (
            SELECT COUNT(*)::int AS chunk_count
            FROM subconscious_unit_embedding_chunks all_chunks
            WHERE all_chunks.unit_id = s.id
        ) chunk_stats
        WHERE s.status = 'active'
          AND s.embedding_status = 'embedded'
          AND c.embedding_status = 'embedded'
          AND c.embedding IS NOT NULL
          AND (NOT p_exclude_sensitive
               OR COALESCE(s.source_attribution->>'sensitivity', '') <> 'private')
        ORDER BY s.id, c.embedding <=> p_query_embedding, c.chunk_index
    ),
    parent_hits AS (
        SELECT
            s.id AS item_id,
            s.content,
            (1 - (s.embedding <=> p_query_embedding))::float AS score,
            ARRAY[s.id]::uuid[] AS source_unit_ids,
            s.source_attribution,
            s.created_at,
            s.trust_level,
            s.metadata,
            'vector'::text AS retrieval_source
        FROM subconscious_units s
        WHERE s.status = 'active'
          AND s.embedding_status = 'embedded'
          AND s.embedding IS NOT NULL
          AND (NOT p_exclude_sensitive
               OR COALESCE(s.source_attribution->>'sensitivity', '') <> 'private')
    ),
    best AS (
        SELECT DISTINCT ON (candidate_rows.item_id)
            candidate_rows.item_id,
            candidate_rows.content,
            candidate_rows.score,
            candidate_rows.source_unit_ids,
            candidate_rows.source_attribution,
            candidate_rows.created_at,
            candidate_rows.trust_level,
            candidate_rows.metadata,
            candidate_rows.retrieval_source
        FROM (
            SELECT * FROM chunk_best
            UNION ALL
            SELECT * FROM parent_hits
        ) candidate_rows
        ORDER BY candidate_rows.item_id, candidate_rows.score DESC, candidate_rows.retrieval_source
    )
    SELECT
        'subconscious'::text AS tier,
        b.item_id,
        CASE
            WHEN b.metadata->>'invalid_precedent' = 'true' THEN
                '[INVALID PRECEDENT - do not imitate'
                || CASE WHEN NULLIF(b.metadata#>>'{latest_correction,correction}', '') IS NOT NULL
                        THEN '; correction: ' || (b.metadata#>>'{latest_correction,correction}')
                        ELSE '' END
                || '] '
                || b.content
            ELSE b.content
        END AS content,
        NULL::text AS memory_type,
        GREATEST(
            0.001,
            b.score - CASE WHEN b.metadata->>'invalid_precedent' = 'true' THEN 0.35 ELSE 0.0 END
        )::float AS score,
        b.source_unit_ids,
        b.source_attribution,
        b.created_at,
        b.trust_level,
        1.0::float AS fidelity,
        1.0::float AS strength,
        NULL::float AS emotional_intensity,
        NULL::float AS confidence,
        b.retrieval_source
    FROM best b
    ORDER BY b.score DESC, b.created_at DESC
    LIMIT GREATEST(COALESCE(p_limit, 10), 0);
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION recmem_recall_context(
    p_query TEXT,
    p_k_sub INT DEFAULT 10,
    p_k_epi INT DEFAULT 5,
    p_k_sem INT DEFAULT 10,
    p_session_id UUID DEFAULT NULL,
    -- Sensitivity enforcement (#92): group channels and other shared
    -- surfaces recall with this TRUE; the agent's own 1:1 recall keeps
    -- everything. The prompt's privacy promise, made mechanical.
    p_exclude_sensitive BOOLEAN DEFAULT FALSE,
    -- Knowledge tier budget (#96 fusion): procedural, strategic, worldview,
    -- and goal memories join recall — one mind, one retrieval mechanism.
    p_k_know INT DEFAULT 5
) RETURNS TABLE (
    tier TEXT,
    item_id UUID,
    content TEXT,
    memory_type TEXT,
    score FLOAT,
    source_unit_ids UUID[],
    source_attribution JSONB,
    created_at TIMESTAMPTZ,
    trust_level FLOAT,
    fidelity FLOAT,
    strength FLOAT,
    emotional_intensity FLOAT,
    confidence FLOAT,
    retrieval_source TEXT
) AS $$
DECLARE
    query_embedding vector;
    strength_weight FLOAT;
    intensity_weight FLOAT;
    recency_weight FLOAT;
    recency_halflife FLOAT;
    boost_weight FLOAT;
    graph_weight FLOAT;
    min_trust FLOAT;
    current_valence FLOAT;
    current_arousal FLOAT;
    current_primary TEXT;
    affective_state JSONB;
BEGIN
    query_embedding := (get_embedding(ARRAY[ensure_embedding_prefix(p_query, 'search_query')]))[1];
    -- The unified ranker (#96, completing #57's "unification, first step"):
    -- recmem's tier skeleton with fast_recall's full scoring transplanted —
    -- associations, episode-temporal binding, mood congruence, trust floor,
    -- and the activation-boost term that lets incubation and reward actually
    -- change what comes to mind.
    recency_weight := COALESCE(get_config_float('memory.recency_weight'), 0.1);
    recency_halflife := GREATEST(COALESCE(get_config_float('memory.recency_halflife_days'), 7.0), 0.01);
    strength_weight := LEAST(1.0, GREATEST(0.0, COALESCE(get_config_float('memory.recall_strength_weight'), 0.5)));
    intensity_weight := LEAST(1.0, GREATEST(0.0, COALESCE(get_config_float('memory.recall_intensity_weight'), 0.5)));
    boost_weight := LEAST(1.0, GREATEST(0.0, COALESCE(get_config_float('memory.recall_activation_boost_weight'), 0.3)));
    graph_weight := LEAST(1.0, GREATEST(0.0, COALESCE(get_config_float('memory.recall_graph_adjacency_weight'), 0.12)));
    min_trust := COALESCE(get_config_float('memory.recall_min_trust_level'), 0.0);

    -- Mood-congruent recall: the current affective state colors what
    -- surfaces, exactly as it did in fast_recall.
    affective_state := get_current_affective_state();
    BEGIN
        current_valence := NULLIF(affective_state->>'valence', '')::float;
    EXCEPTION WHEN OTHERS THEN current_valence := NULL; END;
    BEGIN
        current_arousal := NULLIF(affective_state->>'arousal', '')::float;
    EXCEPTION WHEN OTHERS THEN current_arousal := NULL; END;
    BEGIN
        current_primary := NULLIF(affective_state->>'primary_emotion', '');
    EXCEPTION WHEN OTHERS THEN current_primary := NULL; END;
    current_valence := COALESCE(current_valence, 0.0);
    current_arousal := COALESCE(current_arousal, 0.5);
    current_primary := COALESCE(current_primary, 'neutral');

    RETURN QUERY
    WITH raw_hits AS (
        SELECT *
        FROM recmem_subconscious_vector_hits(
            query_embedding,
            GREATEST(COALESCE(p_k_sub, 10), 0),
            p_exclude_sensitive
        )
    ),
    recent_unembedded AS (
        SELECT
            'subconscious'::text AS tier,
            s.id AS item_id,
            s.content,
            NULL::text AS memory_type,
            0.2::float AS score,
            ARRAY[s.id]::uuid[] AS source_unit_ids,
            s.source_attribution,
            s.created_at,
            s.trust_level,
            1.0::float AS fidelity,
            1.0::float AS strength,
            NULL::float AS emotional_intensity,
            NULL::float AS confidence,
            'temporal'::text AS retrieval_source
        FROM subconscious_units s
        WHERE p_session_id IS NOT NULL
          AND s.session_id = p_session_id
          AND s.status = 'active'
          AND s.embedding_status <> 'embedded'
          AND (NOT p_exclude_sensitive
               OR COALESCE(s.source_attribution->>'sensitivity', '') <> 'private')
        ORDER BY s.created_at DESC
        LIMIT 3
    ),
    -- Shared candidate machinery: ONE ANN scan seeds all memory tiers, and
    -- the association/temporal expansions run once over that shared pool —
    -- never per tier (#96 hot-path requirement).
    -- Per-type-group seed scans: each tier is GUARANTEED candidates of its
    -- own type (a type-blind shared pool lets the episodic bulk crowd rare
    -- types out entirely). The expensive shared machinery — association
    -- expansion and episode binding — still runs once over the union.
    mem_seeds AS (
        (SELECT m.id, (1 - (m.embedding <=> query_embedding))::float AS sim
         FROM memories m
         WHERE m.status = 'active'
           AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
           AND m.type = 'episodic'
           AND m.embedding_status = 'embedded'
           AND m.embedding IS NOT NULL
           AND (NOT p_exclude_sensitive
                OR COALESCE(m.source_attribution->>'sensitivity', '') <> 'private')
         ORDER BY m.embedding <=> query_embedding
         LIMIT GREATEST(COALESCE(p_k_epi, 5), 0) * 2)
        UNION ALL
        (SELECT m.id, (1 - (m.embedding <=> query_embedding))::float AS sim
         FROM memories m
         WHERE m.status = 'active'
           AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
           AND m.type = 'semantic'
           AND m.embedding_status = 'embedded'
           AND m.embedding IS NOT NULL
           AND (NOT p_exclude_sensitive
                OR COALESCE(m.source_attribution->>'sensitivity', '') <> 'private')
         ORDER BY m.embedding <=> query_embedding
         LIMIT GREATEST(COALESCE(p_k_sem, 10), 0) * 2)
        UNION ALL
        (SELECT m.id, (1 - (m.embedding <=> query_embedding))::float AS sim
         FROM memories m
         WHERE m.status = 'active'
           AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
           AND m.type::text IN ('procedural', 'strategic', 'worldview', 'goal')
           AND m.embedding_status = 'embedded'
           AND m.embedding IS NOT NULL
           AND (NOT p_exclude_sensitive
                OR COALESCE(m.source_attribution->>'sensitivity', '') <> 'private')
         ORDER BY m.embedding <=> query_embedding
         LIMIT GREATEST(COALESCE(p_k_know, 5), 0) * 2)
    ),
    associations AS (
        -- Spreading activation through precomputed neighborhoods.
        SELECT (n.key)::uuid AS mem_id, MAX((n.value)::float * s.sim) AS assoc_score
        FROM mem_seeds s
        JOIN memory_neighborhoods mn ON s.id = mn.memory_id,
        LATERAL jsonb_each_text(mn.neighbors) n
        WHERE NOT mn.is_stale
        GROUP BY (n.key)::uuid
    ),
    temporal AS (
        -- Episode binding: what belongs to the open or just-closed episode
        -- stays near the surface.
        SELECT DISTINCT fem.memory_id AS mem_id, 0.15::float AS temp_score
        FROM episodes e
        CROSS JOIN LATERAL find_episode_memories_graph(e.id) fem
        WHERE e.ended_at IS NULL
           OR e.ended_at > CURRENT_TIMESTAMP - INTERVAL '1 hour'
        LIMIT 20
    ),
    graph_adj AS (
        -- Typed graph adjacency: if vector recall catches one memory in a
        -- causal/contradictory/supporting cluster, its immediate typed
        -- neighbors receive a small candidate signal. This is distinct from
        -- embedding neighborhoods and preserves deliberate graph structure.
        SELECT neighbor_id::uuid AS mem_id, MAX(edge_signal) AS graph_score
        FROM (
            SELECT e.dst_id AS neighbor_id, COALESCE(e.weight, 1.0) * s.sim AS edge_signal
            FROM mem_seeds s
            JOIN memory_edges e
              ON e.src_type = 'memory'
             AND e.src_id = s.id::text
             AND e.dst_type = 'memory'
            WHERE e.rel_type IN ('SUPPORTS','CONTRADICTS','CAUSES','CONTESTED_BECAUSE','RELATED_TO','SUPERSEDES')
              AND _safe_uuid(e.dst_id) IS NOT NULL
            UNION ALL
            SELECT e.src_id AS neighbor_id, COALESCE(e.weight, 1.0) * s.sim AS edge_signal
            FROM mem_seeds s
            JOIN memory_edges e
              ON e.dst_type = 'memory'
             AND e.dst_id = s.id::text
             AND e.src_type = 'memory'
            WHERE e.rel_type IN ('SUPPORTS','CONTRADICTS','CAUSES','CONTESTED_BECAUSE','RELATED_TO','SUPERSEDES')
              AND _safe_uuid(e.src_id) IS NOT NULL
        ) g
        GROUP BY neighbor_id::uuid
    ),
    candidate_ids AS (
        SELECT s.id AS mem_id, s.sim AS vector_score, NULL::float AS assoc_score, NULL::float AS temp_score, NULL::float AS graph_score
        FROM mem_seeds s
        UNION
        SELECT a.mem_id, NULL, a.assoc_score, NULL, NULL FROM associations a
        UNION
        SELECT tp.mem_id, NULL, NULL, tp.temp_score, NULL FROM temporal tp
        UNION
        SELECT ga.mem_id, NULL, NULL, NULL, ga.graph_score FROM graph_adj ga
    ),
    candidates AS (
        SELECT c.mem_id,
               MAX(c.vector_score) AS vector_score,
               MAX(c.assoc_score) AS assoc_score,
               MAX(c.temp_score) AS temp_score,
               MAX(c.graph_score) AS graph_score
        FROM candidate_ids c
        GROUP BY c.mem_id
    ),
    scored AS (
        SELECT
            m.id AS item_id,
            CASE
                WHEN m.metadata->>'invalid_precedent' = 'true' THEN
                    '[INVALID PRECEDENT - do not imitate'
                    || CASE WHEN NULLIF(m.metadata#>>'{latest_correction,correction}', '') IS NOT NULL
                            THEN '; correction: ' || (m.metadata#>>'{latest_correction,correction}')
                            ELSE '' END
                    || '] '
                    || m.content
                ELSE m.content
            END AS content,
            m.type::text AS memory_type,
            m.type AS mtype,
            GREATEST(
                COALESCE(c.vector_score, (1 - (m.embedding <=> query_embedding)))
                  * (1.0 - strength_weight + strength_weight
                     * GREATEST(
                         calculate_strength(m.importance, m.decay_rate, m.created_at, m.last_reinforced),
                         intensity_weight * current_emotional_intensity(
                             (m.metadata->'emotional_context'->>'intensity')::float,
                             (m.metadata->>'emotional_valence')::float, m.created_at, m.last_reinforced)))
                + COALESCE(c.assoc_score, 0) * 0.2
                + COALESCE(c.temp_score, 0)
                + COALESCE(c.graph_score, 0) * graph_weight
                + recency_weight * exp(-ln(2.0) * GREATEST(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - m.created_at)), 0)
                                       / (86400.0 * recency_halflife))
                + COALESCE(m.trust_level, 0.5) * 0.1
                -- Reward/incubation salience: boosted memories genuinely come
                -- to mind more easily until the boost decays.
                + LEAST(1.0, GREATEST(0.0, COALESCE((m.metadata->>'activation_boost')::float, 0.0))) * boost_weight
                -- Corrected memories remain auditable but should not act as
                -- behavioral precedents when a similar situation recurs.
                - CASE WHEN m.metadata->>'invalid_precedent' = 'true' THEN 0.35 ELSE 0.0 END
                -- Mood congruence (transplanted from fast_recall, weight 0.05).
                + (CASE
                       WHEN m.metadata ? 'emotional_context' THEN
                           (COALESCE(
                                CASE WHEN (m.metadata->'emotional_context'->>'valence') IS NULL THEN NULL
                                     ELSE 1.0 - (ABS((m.metadata->'emotional_context'->>'valence')::float - current_valence) / 2.0)
                                END, 0.5) * 0.6
                            + COALESCE(
                                CASE WHEN (m.metadata->'emotional_context'->>'arousal') IS NULL THEN NULL
                                     ELSE 1.0 - ABS((m.metadata->'emotional_context'->>'arousal')::float - current_arousal)
                                END, 0.5) * 0.3
                            + (CASE
                                   WHEN (m.metadata->'emotional_context'->>'primary_emotion') IS NULL THEN 0.5
                                   WHEN (m.metadata->'emotional_context'->>'primary_emotion') = current_primary THEN 1.0
                                   ELSE 0.7
                               END) * 0.1)
                       ELSE
                           CASE WHEN (m.metadata->>'emotional_valence') IS NULL THEN 0.5
                                ELSE 1.0 - (ABS((m.metadata->>'emotional_valence')::float - current_valence) / 2.0)
                           END
                   END) * 0.05,
                0.001)::float AS score,
            m.source_attribution,
            m.created_at,
            m.trust_level,
            m.fidelity,
            calculate_strength(m.importance, m.decay_rate, m.created_at, m.last_reinforced)::float AS strength,
            (current_emotional_intensity((m.metadata->'emotional_context'->>'intensity')::float,
                (m.metadata->>'emotional_valence')::float, m.created_at, m.last_reinforced)
             * SIGN(COALESCE((m.metadata->>'emotional_valence')::float, 0)))::float AS emotional_intensity,
            (m.metadata->>'confidence')::float AS confidence,
            CASE
                WHEN c.vector_score IS NOT NULL THEN 'vector'
                WHEN c.assoc_score IS NOT NULL THEN 'association'
                WHEN c.temp_score IS NOT NULL THEN 'temporal'
                WHEN c.graph_score IS NOT NULL THEN 'graph'
                ELSE 'fallback'
            END AS retrieval_source
        FROM candidates c
        JOIN memories m ON m.id = c.mem_id
        WHERE m.status = 'active'
          AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
          AND m.embedding_status = 'embedded'
          AND m.embedding IS NOT NULL
          AND m.trust_level >= min_trust
          AND (NOT p_exclude_sensitive
               OR COALESCE(m.source_attribution->>'sensitivity', '') <> 'private')
    ),
    with_units AS (
        SELECT sc.*, COALESCE(
                   (SELECT array_agg(msu.subconscious_unit_id)
                    FROM memory_source_units msu
                    WHERE msu.memory_id = sc.item_id), '{}'::uuid[]) AS source_unit_ids
        FROM scored sc
    ),
    epi_hits AS (
        SELECT 'episodic'::text AS tier, w.item_id, w.content, w.memory_type, w.score,
               w.source_unit_ids, w.source_attribution, w.created_at, w.trust_level,
               w.fidelity, w.strength, w.emotional_intensity, w.confidence,
               w.retrieval_source
        FROM with_units w WHERE w.mtype = 'episodic'
        ORDER BY w.score DESC LIMIT GREATEST(COALESCE(p_k_epi, 5), 0)
    ),
    sem_hits AS (
        SELECT 'semantic'::text AS tier, w.item_id, w.content, w.memory_type, w.score,
               w.source_unit_ids, w.source_attribution, w.created_at, w.trust_level,
               w.fidelity, w.strength, w.emotional_intensity, w.confidence,
               w.retrieval_source
        FROM with_units w WHERE w.mtype = 'semantic'
        ORDER BY w.score DESC LIMIT GREATEST(COALESCE(p_k_sem, 10), 0)
    ),
    know_hits AS (
        SELECT 'knowledge'::text AS tier, w.item_id, w.content, w.memory_type, w.score,
               w.source_unit_ids, w.source_attribution, w.created_at, w.trust_level,
               w.fidelity, w.strength, w.emotional_intensity, w.confidence,
               w.retrieval_source
        FROM with_units w WHERE w.mtype::text IN ('procedural', 'strategic', 'worldview', 'goal')
        ORDER BY w.score DESC LIMIT GREATEST(COALESCE(p_k_know, 5), 0)
    ),
    spontaneous_hits AS (
        -- What's on her mind arrives unbidden (#98): strongly boosted
        -- memories (incubation resolutions, reward spikes) join recall even
        -- when the query didn't ask for them — then fade with boost decay.
        SELECT
            'spontaneous'::text AS tier,
            sm.id AS item_id,
            sm.content,
            sm.type::text AS memory_type,
            LEAST(1.0, COALESCE((sm.metadata->>'activation_boost')::float, 0.0))::float AS score,
            COALESCE((SELECT array_agg(msu.subconscious_unit_id)
                      FROM memory_source_units msu WHERE msu.memory_id = sm.id), '{}'::uuid[]) AS source_unit_ids,
            sm.source_attribution,
            sm.created_at,
            sm.trust_level,
            sm.fidelity,
            calculate_strength(sm.importance, sm.decay_rate, sm.created_at, sm.last_reinforced)::float AS strength,
            NULL::float AS emotional_intensity,
            (sm.metadata->>'confidence')::float AS confidence,
            'spontaneous'::text AS retrieval_source
        FROM get_spontaneous_memories(2) sm
        WHERE (NOT p_exclude_sensitive
               OR COALESCE(sm.source_attribution->>'sensitivity', '') <> 'private')
          AND sm.id NOT IN (
              SELECT h.item_id FROM epi_hits h
              UNION ALL SELECT h.item_id FROM sem_hits h
              UNION ALL SELECT h.item_id FROM know_hits h)
    )
    SELECT * FROM raw_hits
    UNION ALL
    SELECT * FROM recent_unembedded
    UNION ALL
    SELECT * FROM epi_hits
    UNION ALL
    SELECT * FROM sem_hits
    UNION ALL
    SELECT * FROM know_hits
    UNION ALL
    SELECT * FROM spontaneous_hits
    ORDER BY tier, score DESC, created_at DESC;
END;
$$ LANGUAGE plpgsql;

