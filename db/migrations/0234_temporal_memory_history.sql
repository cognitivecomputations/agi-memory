-- Point-in-time recall and explainable memory diffs over the durable validity
-- and supersession record.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

CREATE OR REPLACE FUNCTION memory_was_valid_at(
    p_memory_id UUID,
    p_as_of TIMESTAMPTZ
)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
STRICT
AS $$
    SELECT EXISTS (
        SELECT 1
        FROM memories m
        WHERE m.id = p_memory_id
          AND m.status <> 'staged'
          -- Inactive legacy/imported rows did not always receive valid_until.
          -- Their final updated_at is the conservative historical close: they
          -- remain reconstructable before it but can never look current.
          AND (
              m.status = 'active'
              OR (
                  m.status IN ('archived', 'invalidated')
                  AND COALESCE(m.valid_until, m.updated_at) > p_as_of
              )
          )
          AND COALESCE(m.valid_from, m.created_at) <= p_as_of
          AND (m.valid_until IS NULL OR m.valid_until > p_as_of)
          AND NOT EXISTS (
              SELECT 1
              FROM memory_supersessions s
              WHERE s.superseded_memory_id = m.id
                AND s.superseded_at <= p_as_of
                AND (
                    s.status = 'active'
                    OR (s.status = 'reverted' AND s.resolved_at > p_as_of)
                )
          )
    );
$$;

-- Belief confidence and trust are updated in place, but the append-only audit
-- stores the exact prior values. The first revision after a historical instant
-- therefore reconstructs the state that existed at that instant.
CREATE OR REPLACE FUNCTION memory_epistemic_state_as_of(
    p_memory_id UUID,
    p_as_of TIMESTAMPTZ
)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
STRICT
AS $$
DECLARE
    v_memory memories%ROWTYPE;
    v_next belief_revision_audit%ROWTYPE;
BEGIN
    SELECT * INTO v_memory FROM memories WHERE id = p_memory_id;
    IF NOT FOUND THEN
        RETURN '{}'::jsonb;
    END IF;
    SELECT * INTO v_next
    FROM belief_revision_audit
    WHERE memory_id = p_memory_id
      AND created_at > p_as_of
    ORDER BY created_at, audit_id
    LIMIT 1;
    RETURN jsonb_strip_nulls(jsonb_build_object(
        'confidence', CASE
            WHEN v_next.audit_id IS NOT NULL THEN v_next.prior_confidence
            ELSE NULLIF(v_memory.metadata->>'confidence', '')::float
        END,
        'trust_level', CASE
            WHEN v_next.audit_id IS NOT NULL AND v_next.prior_trust IS NOT NULL
                THEN v_next.prior_trust
            ELSE v_memory.trust_level
        END
    ));
END;
$$;

CREATE OR REPLACE FUNCTION temporal_memory_snapshot(
    p_query TEXT,
    p_as_of TIMESTAMPTZ,
    p_limit INT DEFAULT NULL,
    p_memory_types memory_type[] DEFAULT NULL,
    p_min_score FLOAT DEFAULT NULL,
    p_exclude_sensitive BOOLEAN DEFAULT FALSE
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_query TEXT := NULLIF(btrim(COALESCE(p_query, '')), '');
    v_as_of TIMESTAMPTZ := LEAST(p_as_of, CURRENT_TIMESTAMP);
    v_limit INT := LEAST(GREATEST(COALESCE(
        p_limit, get_config_int('memory.recall_default_limit'), 5
    ), 1), COALESCE(get_config_int('memory.recall_max_limit'), 50));
    v_min_score FLOAT := LEAST(1.0, GREATEST(0.0, COALESCE(
        p_min_score, get_config_float('memory.recall_min_score'), 0.35
    )));
    v_query_embedding vector;
    v_fts_query tsquery;
    v_embedding_degraded BOOLEAN := FALSE;
    v_rows JSONB;
BEGIN
    IF p_as_of IS NULL THEN
        RAISE EXCEPTION 'as_of is required';
    END IF;
    IF p_as_of > CURRENT_TIMESTAMP + INTERVAL '5 minutes' THEN
        RAISE EXCEPTION 'as_of cannot be in the future; choose now or an earlier instant';
    END IF;
    IF v_query IS NOT NULL THEN
        BEGIN
            v_fts_query := websearch_to_tsquery('english', v_query);
        EXCEPTION WHEN OTHERS THEN
            v_fts_query := plainto_tsquery('english', v_query);
        END;
        BEGIN
            v_query_embedding := (
                get_embedding(ARRAY[ensure_embedding_prefix(v_query, 'search_query')])
            )[1];
        EXCEPTION WHEN OTHERS THEN
            v_query_embedding := NULL;
            v_embedding_degraded := TRUE;
        END;
    END IF;

    WITH scored AS (
        SELECT
            m.*,
            CASE
                WHEN v_query_embedding IS NOT NULL
                 AND m.embedding_status = 'embedded'
                 AND m.embedding IS NOT NULL
                THEN (1.0 - (m.embedding <=> v_query_embedding))::float
                ELSE NULL::float
            END AS semantic_score,
            CASE
                WHEN v_query IS NOT NULL
                THEN ts_rank_cd(to_tsvector('english', m.content), v_fts_query, 32)::float
                ELSE 0.0::float
            END AS lexical_rank,
            CASE
                WHEN v_query IS NOT NULL THEN similarity(lower(m.content), lower(v_query))::float
                ELSE 0.0::float
            END AS trigram_score
        FROM memories m
        WHERE memory_was_valid_at(m.id, v_as_of)
          AND (p_memory_types IS NULL OR m.type = ANY(p_memory_types))
          AND (
              NOT COALESCE(p_exclude_sensitive, FALSE)
              OR COALESCE(m.source_attribution->>'sensitivity', '') <> 'private'
          )
    ), ranked AS (
        SELECT s.*,
               CASE
                   WHEN v_query IS NULL THEN s.importance::float
                   ELSE GREATEST(
                       COALESCE(s.semantic_score, 0.0),
                       CASE WHEN s.lexical_rank > 0
                            THEN LEAST(1.0, 0.35 + s.lexical_rank * 4.0)
                            ELSE 0.0 END,
                       s.trigram_score
                   )
               END AS final_score
        FROM scored s
        WHERE v_query IS NULL
           OR COALESCE(s.semantic_score >= 0.15, FALSE)
           OR s.lexical_rank > 0
           OR s.trigram_score >= 0.10
           OR s.content ILIKE '%' || v_query || '%'
    ), selected AS (
        SELECT *
        FROM ranked
        WHERE final_score >= CASE WHEN v_query IS NULL THEN 0.0 ELSE v_min_score END
        ORDER BY final_score DESC, valid_from DESC, created_at DESC, id
        LIMIT v_limit
    )
    SELECT COALESCE(jsonb_agg(jsonb_strip_nulls(jsonb_build_object(
        'memory_id', s.id,
        'content', s.content,
        'type', s.type,
        'score', s.final_score,
        'importance', s.importance,
        'confidence', epistemic.value->'confidence',
        'trust_level', epistemic.value->'trust_level',
        'source_attribution', s.source_attribution,
        'citation_id', citation.value->>'citation_id',
        'citation', citation.value || jsonb_build_object(
            'trust_level', epistemic.value->'trust_level',
            'low_trust', COALESCE((epistemic.value->>'trust_level')::float, 0.0)
                < COALESCE(get_config_float('memory.low_trust_threshold'), 0.5)
        ),
        'valid_from', s.valid_from,
        'valid_until', s.valid_until,
        'superseded_by', s.superseded_by,
        'created_at', s.created_at,
        'current_status', s.status
    )) ORDER BY s.final_score DESC, s.valid_from DESC, s.created_at DESC, s.id), '[]'::jsonb)
    INTO v_rows
    FROM selected s
    CROSS JOIN LATERAL (
        SELECT memory_epistemic_state_as_of(s.id, v_as_of) AS value
    ) epistemic
    CROSS JOIN LATERAL (
        SELECT memory_citation_envelope(s.id) AS value
    ) citation;

    RETURN jsonb_build_object(
        'query', COALESCE(v_query, ''),
        'as_of', v_as_of,
        'count', jsonb_array_length(v_rows),
        'retrieval_mode', CASE
            WHEN v_query IS NULL THEN 'chronological'
            WHEN v_query_embedding IS NULL THEN 'lexical'
            ELSE 'hybrid'
        END,
        'degraded', v_embedding_degraded,
        'degraded_reason', CASE WHEN v_embedding_degraded
            THEN 'Embedding search was unavailable; exact lexical history still ran.' END,
        'memories', v_rows
    );
END;
$$;

CREATE OR REPLACE FUNCTION diff_memory_history(
    p_query TEXT,
    p_from TIMESTAMPTZ,
    p_to TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    p_limit INT DEFAULT NULL,
    p_memory_types memory_type[] DEFAULT NULL,
    p_min_score FLOAT DEFAULT NULL,
    p_exclude_sensitive BOOLEAN DEFAULT FALSE
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_query TEXT := NULLIF(btrim(COALESCE(p_query, '')), '');
    v_to TIMESTAMPTZ := LEAST(COALESCE(p_to, CURRENT_TIMESTAMP), CURRENT_TIMESTAMP);
    v_from_snapshot JSONB;
    v_to_snapshot JSONB;
    v_from_ids UUID[] := ARRAY[]::UUID[];
    v_to_ids UUID[] := ARRAY[]::UUID[];
    v_all_ids UUID[] := ARRAY[]::UUID[];
    v_added JSONB;
    v_expired JSONB;
    v_supersessions JSONB;
    v_revisions JSONB;
    v_decisions JSONB;
BEGIN
    IF v_query IS NULL THEN
        RAISE EXCEPTION 'query is required';
    END IF;
    IF p_from IS NULL THEN
        RAISE EXCEPTION 'from_time is required';
    END IF;
    IF p_from >= v_to THEN
        RAISE EXCEPTION 'from_time must be earlier than to_time';
    END IF;
    IF p_to > CURRENT_TIMESTAMP + INTERVAL '5 minutes' THEN
        RAISE EXCEPTION 'to_time cannot be in the future; choose now or an earlier instant';
    END IF;

    v_from_snapshot := temporal_memory_snapshot(
        v_query, p_from, p_limit, p_memory_types, p_min_score, p_exclude_sensitive
    );
    v_to_snapshot := temporal_memory_snapshot(
        v_query, v_to, p_limit, p_memory_types, p_min_score, p_exclude_sensitive
    );

    SELECT COALESCE(array_agg((item->>'memory_id')::uuid), ARRAY[]::UUID[])
    INTO v_from_ids
    FROM jsonb_array_elements(v_from_snapshot->'memories') item;
    SELECT COALESCE(array_agg((item->>'memory_id')::uuid), ARRAY[]::UUID[])
    INTO v_to_ids
    FROM jsonb_array_elements(v_to_snapshot->'memories') item;
    SELECT COALESCE(array_agg(DISTINCT id), ARRAY[]::UUID[])
    INTO v_all_ids
    FROM unnest(v_from_ids || v_to_ids) id;

    SELECT COALESCE(jsonb_agg(item), '[]'::jsonb) INTO v_added
    FROM jsonb_array_elements(v_to_snapshot->'memories') item
    WHERE NOT ((item->>'memory_id')::uuid = ANY(v_from_ids));
    SELECT COALESCE(jsonb_agg(item), '[]'::jsonb) INTO v_expired
    FROM jsonb_array_elements(v_from_snapshot->'memories') item
    WHERE NOT ((item->>'memory_id')::uuid = ANY(v_to_ids));

    SELECT COALESCE(jsonb_agg(jsonb_strip_nulls(jsonb_build_object(
        'event', 'supersession',
        'at', s.superseded_at,
        'status', s.status,
        'resolved_at', s.resolved_at,
        'reason', s.reason,
        'actor', s.actor,
        'superseded_memory_id', s.superseded_memory_id,
        'superseded_content', old_memory.content,
        'replacement_memory_id', s.replacement_memory_id,
        'replacement_content', replacement.content,
        'metadata', s.metadata
    )) ORDER BY s.superseded_at, s.id), '[]'::jsonb)
    INTO v_supersessions
    FROM memory_supersessions s
    JOIN memories old_memory ON old_memory.id = s.superseded_memory_id
    LEFT JOIN memories replacement ON replacement.id = s.replacement_memory_id
    WHERE s.superseded_at > p_from
      AND s.superseded_at <= v_to
      AND (
          s.superseded_memory_id = ANY(v_all_ids)
          OR s.replacement_memory_id = ANY(v_all_ids)
      );

    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'event', 'belief_revision',
        'at', a.created_at,
        'memory_id', a.memory_id,
        'stance', a.stance,
        'prior_confidence', a.prior_confidence,
        'posterior_confidence', a.posterior_confidence,
        'applied', a.applied,
        'reason', a.reason,
        'evidence', a.evidence,
        'policy_context', a.policy_context
    ) ORDER BY a.created_at, a.audit_id), '[]'::jsonb)
    INTO v_revisions
    FROM belief_revision_audit a
    WHERE a.memory_id = ANY(v_all_ids)
      AND a.created_at > p_from
      AND a.created_at <= v_to;

    SELECT COALESCE(jsonb_agg(jsonb_strip_nulls(jsonb_build_object(
        'event', 'contradiction_decision',
        'at', c.resolved_at,
        'case_id', c.id,
        'code', contradiction_case_code(c.id),
        'outcome', c.outcome,
        'note', c.resolution_note,
        'winner_memory_id', c.winner_memory_id,
        'loser_memory_id', c.loser_memory_id,
        'memory_a', c.memory_a,
        'memory_b', c.memory_b
    )) ORDER BY c.resolved_at, c.id), '[]'::jsonb)
    INTO v_decisions
    FROM contradiction_cases c
    WHERE c.resolved_at > p_from
      AND c.resolved_at <= v_to
      AND (c.memory_a = ANY(v_all_ids) OR c.memory_b = ANY(v_all_ids));

    RETURN jsonb_build_object(
        'query', v_query,
        'from_time', p_from,
        'to_time', v_to,
        'from_snapshot', v_from_snapshot,
        'to_snapshot', v_to_snapshot,
        'added', v_added,
        'expired', v_expired,
        'supersessions', v_supersessions,
        'belief_revisions', v_revisions,
        'contradiction_decisions', v_decisions,
        'summary', jsonb_build_object(
            'from_count', jsonb_array_length(v_from_snapshot->'memories'),
            'to_count', jsonb_array_length(v_to_snapshot->'memories'),
            'added', jsonb_array_length(v_added),
            'expired', jsonb_array_length(v_expired),
            'supersessions', jsonb_array_length(v_supersessions),
            'belief_revisions', jsonb_array_length(v_revisions),
            'contradiction_decisions', jsonb_array_length(v_decisions)
        )
    );
END;
$$;

-- Extend the DB-owned memory tool dispatcher without duplicating its mature
-- recall/remember implementation.
DO $rename$
BEGIN
    IF to_regprocedure('_execute_memory_tool_dispatch_legacy_temporal(text,jsonb)') IS NULL THEN
        ALTER FUNCTION _execute_memory_tool_dispatch(TEXT, JSONB)
            RENAME TO _execute_memory_tool_dispatch_legacy_temporal;
    END IF;
END;
$rename$;

CREATE OR REPLACE FUNCTION _execute_memory_tool_dispatch(
    p_tool_name TEXT,
    p_args JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_query TEXT := NULLIF(btrim(COALESCE(p_args->>'query', '')), '');
    v_as_of TIMESTAMPTZ;
    v_from TIMESTAMPTZ;
    v_to TIMESTAMPTZ;
    v_types memory_type[];
    v_limit INT;
    v_min_score FLOAT;
    v_exclude_sensitive BOOLEAN;
    v_result JSONB;
BEGIN
    IF p_tool_name NOT IN ('recall_at_time', 'diff_memory_history') THEN
        RETURN _execute_memory_tool_dispatch_legacy_temporal(p_tool_name, p_args);
    END IF;
    IF v_query IS NULL THEN
        RETURN tool_error('query is required', 'invalid_params');
    END IF;
    IF p_args ? 'memory_types'
       AND p_args->'memory_types' <> 'null'::jsonb
       AND jsonb_typeof(p_args->'memory_types') <> 'array' THEN
        RETURN tool_error('memory_types must be an array', 'invalid_params');
    END IF;
    IF jsonb_typeof(p_args->'memory_types') = 'array'
       AND jsonb_array_length(p_args->'memory_types') > 0 THEN
        BEGIN
            SELECT array_agg(value::memory_type)
            INTO v_types
            FROM jsonb_array_elements_text(p_args->'memory_types') t(value);
        EXCEPTION WHEN OTHERS THEN
            RETURN tool_error('memory_types contains an unsupported memory type', 'invalid_params');
        END;
    END IF;
    BEGIN
        v_limit := NULLIF(p_args->>'limit', '')::int;
        v_min_score := NULLIF(p_args->>'min_score', '')::float;
        v_exclude_sensitive := COALESCE(
            NULLIF(p_args->>'exclude_sensitive', '')::boolean,
            FALSE
        );
        IF p_tool_name = 'recall_at_time' THEN
            v_as_of := NULLIF(p_args->>'as_of', '')::timestamptz;
        ELSE
            v_from := NULLIF(p_args->>'from_time', '')::timestamptz;
            v_to := COALESCE(NULLIF(p_args->>'to_time', '')::timestamptz, CURRENT_TIMESTAMP);
        END IF;
    EXCEPTION WHEN OTHERS THEN
        RETURN tool_error(
            'Times must be ISO-8601 timestamps; limit, min_score, and exclude_sensitive must have the documented types',
            'invalid_params'
        );
    END;
    IF v_limit IS NOT NULL AND v_limit < 1 THEN
        RETURN tool_error('limit must be at least 1', 'invalid_params');
    END IF;
    IF v_min_score IS NOT NULL AND (v_min_score < 0 OR v_min_score > 1) THEN
        RETURN tool_error('min_score must be between 0 and 1', 'invalid_params');
    END IF;

    IF p_tool_name = 'recall_at_time' THEN
        IF v_as_of IS NULL THEN
            RETURN tool_error('as_of is required', 'invalid_params');
        END IF;
        IF v_as_of > CURRENT_TIMESTAMP + INTERVAL '5 minutes' THEN
            RETURN tool_error('as_of cannot be in the future; choose now or an earlier instant', 'invalid_params');
        END IF;
        v_as_of := LEAST(v_as_of, CURRENT_TIMESTAMP);
        v_result := temporal_memory_snapshot(
            v_query,
            v_as_of,
            v_limit,
            v_types,
            v_min_score,
            v_exclude_sensitive
        );
        RETURN tool_success(
            v_result,
            format('Found %s memories that were valid at %s for %L',
                v_result->>'count', v_as_of, v_query)
        );
    END IF;

    IF v_from IS NULL THEN
        RETURN tool_error('from_time is required', 'invalid_params');
    END IF;
    IF v_from >= v_to THEN
        RETURN tool_error('from_time must be earlier than to_time', 'invalid_params');
    END IF;
    IF v_to > CURRENT_TIMESTAMP + INTERVAL '5 minutes' THEN
        RETURN tool_error('to_time cannot be in the future; choose now or an earlier instant', 'invalid_params');
    END IF;
    v_to := LEAST(v_to, CURRENT_TIMESTAMP);
    v_result := diff_memory_history(
        v_query,
        v_from,
        v_to,
        v_limit,
        v_types,
        v_min_score,
        v_exclude_sensitive
    );
    RETURN tool_success(
        v_result,
        format('Compared memory for %L between %s and %s: %s added, %s expired',
            v_query, v_from, v_to,
            v_result#>>'{summary,added}', v_result#>>'{summary,expired}')
    );
EXCEPTION WHEN OTHERS THEN
    RETURN tool_error(SQLERRM);
END;
$$;

SET check_function_bodies = on;

-- Existing installations already have the conversation module. Append the
-- structural situational cue once; fresh schemas receive the same paragraph
-- from db/40_seed_prompt_modules.sql.
UPDATE prompt_modules
SET content = content || $cue$

**When the question is temporally framed:** phrases such as “as of,” “back then,” “at that point,” or “what did you know on” are a situational cue to use `recall_at_time`, after resolving the requested instant against the Temporal Context. “Has that changed?”, “what changed between,” and “why is that different now?” cue `diff_memory_history`. Do not answer these from current recall and do not infer an old state from present wording: use the validity and supersession record. Cite the returned historical memories with their exact `citation_id`, and distinguish “the record contains no matching memory” from “the record says the opposite.”
$cue$,
    updated_at = CURRENT_TIMESTAMP
WHERE key = 'conversation'
  AND position('When the question is temporally framed' in content) = 0;
