-- Hexis schema: neighborhood recomputation functions.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

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
CREATE OR REPLACE FUNCTION batch_recompute_neighborhoods(
    p_batch_size INT DEFAULT 50
)
RETURNS INT AS $$
DECLARE
    recomputed INT := 0;
    mem_id UUID;
BEGIN
    FOR mem_id IN
        SELECT memory_id
        FROM memory_neighborhoods
        WHERE is_stale = TRUE
        ORDER BY computed_at ASC NULLS FIRST
        LIMIT p_batch_size
    LOOP
        PERFORM recompute_neighborhood(mem_id);
        recomputed := recomputed + 1;
    END LOOP;

    RETURN recomputed;
END;
$$ LANGUAGE plpgsql;

SET check_function_bodies = on;
