-- Structured query recall should not lose just-written async-embedding
-- memories. Use hybrid candidates first, then apply the structured filters.

SET search_path = public, ag_catalog, "$user";

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
END;
$$ LANGUAGE plpgsql STABLE;
