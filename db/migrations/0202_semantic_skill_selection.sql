-- Rank skills by what a request means, not by which words it shares.
--
-- Selection scored literal token overlap against a skill's *name*, so "email"
-- scored zero against `gmail-actions` and "book time with Sarah next week"
-- scored zero against `calendar`. Seven of ten ordinary requests reached
-- `core-memory` and nothing else. Alias lists were a stopgap and behaved like
-- one: adding `decide` to the council made it fire on "what did we decide last
-- time", which is recall, not deliberation.
--
-- Embeddings answer "most like which" directly. This is one round trip and, in
-- the steady state, no model calls at all: get_embedding() caches by content
-- hash, and skill descriptions do not change between turns.
SET search_path = public, ag_catalog, "$user";

INSERT INTO config_defaults (key, value, description) VALUES
    ('skills.semantic_selection_enabled', 'true'::jsonb,
     'Rank skills by embedding similarity instead of literal token overlap'),
    ('skills.semantic_threshold', '0.45'::jsonb,
     'Minimum cosine similarity for a skill to auto-activate')
ON CONFLICT (key) DO NOTHING;

-- Names and texts are parallel arrays; the query is embedded alongside them so
-- the whole ranking costs one get_embedding() call, most of it cache hits.
CREATE OR REPLACE FUNCTION rank_skills_by_similarity(
    p_names TEXT[],
    p_texts TEXT[],
    p_query TEXT
) RETURNS TABLE (skill_name TEXT, similarity FLOAT)
LANGUAGE plpgsql
AS $$
DECLARE
    n INT := COALESCE(array_length(p_names, 1), 0);
    vecs vector[];
    query_vec vector;
BEGIN
    IF n = 0 OR COALESCE(btrim(p_query), '') = '' THEN
        RETURN;
    END IF;
    IF COALESCE(array_length(p_texts, 1), 0) <> n THEN
        RAISE EXCEPTION 'names and texts must be the same length (% vs %)',
            n, COALESCE(array_length(p_texts, 1), 0);
    END IF;

    -- One call: every skill text plus the query. Repeat turns hit the cache.
    vecs := get_embedding(array_append(p_texts, p_query));
    query_vec := vecs[n + 1];

    IF query_vec IS NULL THEN
        RETURN;
    END IF;

    RETURN QUERY
    SELECT p_names[i],
           (1.0 - (vecs[i] <=> query_vec))::float
    FROM generate_series(1, n) AS i
    WHERE vecs[i] IS NOT NULL
    ORDER BY 2 DESC;
END;
$$;
