-- Provenance-by-default: source-aware trust and stable citation envelopes.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

INSERT INTO config_defaults (key, value, description) VALUES
    ('memory.source_trust_defaults',
     '{
       "$default": 0.45,
       "unattributed": 0.20,
       "inference": 0.45,
       "internal": 0.65,
       "self_observation": 0.75,
       "conversation": 0.80,
       "chat": 0.80,
       "user": 0.80,
       "user_testimony": 0.80,
       "document": 0.75,
       "documentation": 0.80,
       "repository_document": 0.80,
       "paper": 0.85,
       "web": 0.55,
       "web_page": 0.55,
       "email": 0.70,
       "connector": 0.70,
       "recmem": 0.75,
       "origin_document": 0.90,
       "consent": 0.95
     }'::jsonb,
     'Default source trust by provenance kind when a writer does not provide trust explicitly.'),
    ('memory.low_trust_threshold', '0.5'::jsonb,
     'Trust below this value is visibly marked as low-trust in citation surfaces.')
ON CONFLICT (key) DO NOTHING;

CREATE OR REPLACE FUNCTION source_kind_default_trust(p_kind TEXT)
RETURNS FLOAT
LANGUAGE sql
STABLE
AS $$
    WITH policy AS (
        SELECT COALESCE(get_config('memory.source_trust_defaults'), '{}'::jsonb) AS value
    )
    SELECT LEAST(1.0, GREATEST(0.0, COALESCE(
        NULLIF(policy.value->>lower(NULLIF(btrim(p_kind), '')), '')::float,
        NULLIF(policy.value->>'$default', '')::float,
        0.45
    )))
    FROM policy;
$$;

-- Preserve the source's complete locator-bearing object. Earlier normalization
-- rebuilt a short allowlist, which silently discarded path/page/row metadata,
-- and manufactured {trust: 0.5, observed_at: now} for an empty source. An empty
-- object is absence of provenance, not corroborating evidence.
CREATE OR REPLACE FUNCTION normalize_source_reference(p_source JSONB)
RETURNS JSONB AS $$
DECLARE
    v_kind TEXT;
    v_ref TEXT;
    v_label TEXT;
    v_author TEXT;
    v_observed_at TIMESTAMPTZ;
    v_trust FLOAT;
    v_content_hash TEXT;
    v_source_document_id TEXT;
    v_document_id TEXT;
    v_chunk_id TEXT;
    v_chunk_index INT;
    v_sensitivity TEXT;
    v_has_identity BOOLEAN;
    v_trust_explicit BOOLEAN;
    v_result JSONB;
BEGIN
    IF p_source IS NULL OR jsonb_typeof(p_source) <> 'object' THEN
        RETURN '{}'::jsonb;
    END IF;

    v_kind := NULLIF(btrim(p_source->>'kind'), '');
    v_ref := COALESCE(NULLIF(btrim(p_source->>'ref'), ''), NULLIF(btrim(p_source->>'uri'), ''));
    v_label := NULLIF(btrim(p_source->>'label'), '');
    v_author := NULLIF(btrim(p_source->>'author'), '');
    v_content_hash := NULLIF(btrim(p_source->>'content_hash'), '');
    v_source_document_id := COALESCE(
        NULLIF(btrim(p_source->>'source_document_id'), ''),
        NULLIF(btrim(p_source->>'document_id'), '')
    );
    v_document_id := COALESCE(NULLIF(btrim(p_source->>'document_id'), ''), v_source_document_id);
    v_chunk_id := NULLIF(btrim(p_source->>'chunk_id'), '');
    BEGIN
        v_chunk_index := NULLIF(p_source->>'chunk_index', '')::int;
    EXCEPTION WHEN OTHERS THEN
        v_chunk_index := NULL;
    END;
    v_sensitivity := CASE WHEN p_source->>'sensitivity' = 'private' THEN 'private' END;
    v_has_identity := v_kind IS NOT NULL
        OR v_ref IS NOT NULL
        OR v_label IS NOT NULL
        OR v_author IS NOT NULL
        OR v_content_hash IS NOT NULL
        OR v_document_id IS NOT NULL
        OR v_chunk_id IS NOT NULL
        OR NULLIF(btrim(p_source->>'path'), '') IS NOT NULL;
    IF NOT v_has_identity THEN
        RETURN '{}'::jsonb;
    END IF;

    BEGIN
        v_observed_at := NULLIF(p_source->>'observed_at', '')::timestamptz;
    EXCEPTION WHEN OTHERS THEN
        v_observed_at := NULL;
    END;
    v_observed_at := COALESCE(v_observed_at, CURRENT_TIMESTAMP);
    v_trust_explicit := p_source ? 'trust'
        AND NULLIF(btrim(p_source->>'trust'), '') IS NOT NULL
        AND COALESCE(p_source->>'trust_origin', '') <> 'default';
    BEGIN
        v_trust := CASE
            WHEN v_trust_explicit THEN NULLIF(p_source->>'trust', '')::float
            ELSE source_kind_default_trust(COALESCE(v_kind, 'unattributed'))
        END;
    EXCEPTION WHEN OTHERS THEN
        v_trust := source_kind_default_trust(COALESCE(v_kind, 'unattributed'));
        v_trust_explicit := FALSE;
    END;
    v_trust := LEAST(1.0, GREATEST(0.0, COALESCE(v_trust, 0.0)));

    v_result := p_source
        - 'uri'
        - 'trust'
        - 'trust_origin'
        - 'observed_at'
        - 'sensitivity'
        - 'source_document_id'
        - 'document_id'
        - 'chunk_id'
        - 'chunk_index';
    RETURN jsonb_strip_nulls(v_result || jsonb_build_object(
        'kind', v_kind,
        'ref', v_ref,
        'label', v_label,
        'author', v_author,
        'observed_at', v_observed_at,
        'trust', v_trust,
        'trust_origin', CASE WHEN v_trust_explicit THEN 'explicit' ELSE 'default' END,
        'content_hash', v_content_hash,
        'source_document_id', v_source_document_id,
        'document_id', v_document_id,
        'chunk_id', v_chunk_id,
        'chunk_index', v_chunk_index,
        'sensitivity', v_sensitivity
    ));
END;
$$ LANGUAGE plpgsql STABLE;

-- Confidence and provenance both matter. Multiplication avoids the old plateau
-- where every sufficiently confident one-source belief collapsed to the same
-- cap, while keeping unsupported claims visibly weak.
CREATE OR REPLACE FUNCTION compute_semantic_trust(
    p_confidence FLOAT,
    p_source_references JSONB,
    p_worldview_alignment FLOAT DEFAULT 0.0
)
RETURNS FLOAT AS $$
DECLARE
    v_confidence FLOAT;
    v_reinforcement FLOAT;
    v_evidence_factor FLOAT;
    v_effective FLOAT;
    v_alignment FLOAT;
BEGIN
    v_confidence := LEAST(1.0, GREATEST(0.0, COALESCE(p_confidence, 0.5)));
    v_reinforcement := source_reinforcement_score(p_source_references);
    v_evidence_factor := 0.15 + 0.85 * v_reinforcement;
    v_effective := v_confidence * v_evidence_factor;

    v_alignment := LEAST(1.0, GREATEST(-1.0, COALESCE(p_worldview_alignment, 0.0)));
    IF v_alignment < 0 THEN
        v_effective := v_effective * (1.0 + v_alignment);
    ELSE
        v_effective := LEAST(v_confidence, v_effective + 0.10 * v_alignment);
    END IF;
    RETURN LEAST(1.0, GREATEST(0.0, v_effective));
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION memory_citation_envelope(p_memory_id UUID)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_memory RECORD;
    v_chunk_id UUID;
    v_document_id UUID;
    v_chunk source_document_chunks%ROWTYPE;
    v_document source_documents%ROWTYPE;
    v_label TEXT;
    v_href TEXT;
    v_locator JSONB;
    v_threshold FLOAT := COALESCE(get_config_float('memory.low_trust_threshold'), 0.5);
BEGIN
    SELECT id, type, source_attribution, metadata, trust_level
    INTO v_memory
    FROM memories
    WHERE id = p_memory_id;
    IF NOT FOUND THEN
        RETURN '{}'::jsonb;
    END IF;

    v_chunk_id := _db_brain_try_uuid(v_memory.source_attribution->>'chunk_id');
    IF v_chunk_id IS NULL THEN
        SELECT _db_brain_try_uuid(src->>'chunk_id') INTO v_chunk_id
        FROM jsonb_array_elements(CASE
            WHEN jsonb_typeof(v_memory.metadata->'source_references') = 'array'
            THEN v_memory.metadata->'source_references' ELSE '[]'::jsonb END) src
        WHERE _db_brain_try_uuid(src->>'chunk_id') IS NOT NULL
        LIMIT 1;
    END IF;

    IF v_chunk_id IS NOT NULL THEN
        SELECT c.id, c.source_document_id, c.chunk_index, c.locator_kind, c.locator,
               c.heading_path, c.page_start, c.page_end, c.sheet_name,
               c.row_start, c.row_end
        INTO v_chunk
        FROM source_document_chunks c
        WHERE c.id = v_chunk_id;
        IF FOUND THEN
            v_document_id := v_chunk.source_document_id;
            v_locator := jsonb_strip_nulls(jsonb_build_object(
                'kind', v_chunk.locator_kind,
                'locator', v_chunk.locator,
                'chunk_index', v_chunk.chunk_index,
                'heading_path', to_jsonb(v_chunk.heading_path),
                'page_start', v_chunk.page_start,
                'page_end', v_chunk.page_end,
                'sheet_name', v_chunk.sheet_name,
                'row_start', v_chunk.row_start,
                'row_end', v_chunk.row_end
            ));
        END IF;
    END IF;

    v_document_id := COALESCE(
        v_document_id,
        _db_brain_try_uuid(v_memory.source_attribution->>'source_document_id'),
        _db_brain_try_uuid(v_memory.source_attribution->>'document_id')
    );
    IF v_document_id IS NULL THEN
        SELECT COALESCE(
            _db_brain_try_uuid(src->>'source_document_id'),
            _db_brain_try_uuid(src->>'document_id')
        ) INTO v_document_id
        FROM jsonb_array_elements(CASE
            WHEN jsonb_typeof(v_memory.metadata->'source_references') = 'array'
            THEN v_memory.metadata->'source_references' ELSE '[]'::jsonb END) src
        WHERE COALESCE(
            _db_brain_try_uuid(src->>'source_document_id'),
            _db_brain_try_uuid(src->>'document_id')
        ) IS NOT NULL
        LIMIT 1;
    END IF;
    IF v_document_id IS NULL THEN
        SELECT d.id INTO v_document_id
        FROM source_documents d
        WHERE d.status = 'active'
          AND (
              d.content_hash = NULLIF(v_memory.source_attribution->>'content_hash', '')
              OR d.content_hash = NULLIF(v_memory.source_attribution->>'ref', '')
              OR EXISTS (
                  SELECT 1
                  FROM jsonb_array_elements(CASE
                      WHEN jsonb_typeof(v_memory.metadata->'source_references') = 'array'
                      THEN v_memory.metadata->'source_references' ELSE '[]'::jsonb END) src
                  WHERE d.content_hash = NULLIF(src->>'content_hash', '')
                     OR d.content_hash = NULLIF(src->>'ref', '')
              )
          )
        LIMIT 1;
    END IF;

    IF v_document_id IS NOT NULL THEN
        SELECT id, title, path, source_type, source_attribution
        INTO v_document
        FROM source_documents
        WHERE id = v_document_id AND status = 'active';
    END IF;

    v_label := COALESCE(
        NULLIF(v_document.title, ''),
        NULLIF(v_document.path, ''),
        NULLIF(v_memory.source_attribution->>'label', ''),
        NULLIF(v_memory.source_attribution->>'path', ''),
        NULLIF(v_memory.source_attribution->>'ref', ''),
        initcap(v_memory.type::text) || ' memory ' || left(v_memory.id::text, 8)
    );
    v_href := CASE
        WHEN v_document_id IS NOT NULL AND v_chunk_id IS NOT NULL
            THEN '/documents?document=' || v_document_id::text || '&chunk=' || v_chunk_id::text
        WHEN v_document_id IS NOT NULL
            THEN '/documents?document=' || v_document_id::text
        ELSE '/memories?memory=' || v_memory.id::text
    END;

    RETURN jsonb_strip_nulls(jsonb_build_object(
        'citation_id', 'mem-' || v_memory.id::text,
        'label', v_label,
        'href', v_href,
        'memory_id', v_memory.id::text,
        'document_id', v_document_id::text,
        'chunk_id', v_chunk_id::text,
        'source_kind', COALESCE(v_document.source_type, v_memory.source_attribution->>'kind'),
        'source_attribution', v_memory.source_attribution,
        'source_references', v_memory.metadata->'source_references',
        'trust_level', v_memory.trust_level,
        'low_trust', COALESCE(v_memory.trust_level, 0.0) < v_threshold,
        'locator', v_locator
    ));
END;
$$;

CREATE OR REPLACE FUNCTION source_citation_envelope(
    p_document_id UUID,
    p_chunk_id UUID DEFAULT NULL
)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_document source_documents%ROWTYPE;
    v_chunk source_document_chunks%ROWTYPE;
    v_trust FLOAT;
    v_threshold FLOAT := COALESCE(get_config_float('memory.low_trust_threshold'), 0.5);
    v_label TEXT;
    v_href TEXT;
    v_locator JSONB;
BEGIN
    SELECT * INTO v_document
    FROM source_documents
    WHERE id = p_document_id AND status = 'active';
    IF NOT FOUND THEN
        RETURN '{}'::jsonb;
    END IF;
    IF p_chunk_id IS NOT NULL THEN
        SELECT * INTO v_chunk
        FROM source_document_chunks
        WHERE id = p_chunk_id AND source_document_id = p_document_id;
        IF FOUND THEN
            v_locator := jsonb_strip_nulls(jsonb_build_object(
                'kind', v_chunk.locator_kind,
                'locator', v_chunk.locator,
                'chunk_index', v_chunk.chunk_index,
                'heading_path', to_jsonb(v_chunk.heading_path),
                'page_start', v_chunk.page_start,
                'page_end', v_chunk.page_end,
                'sheet_name', v_chunk.sheet_name,
                'row_start', v_chunk.row_start,
                'row_end', v_chunk.row_end
            ));
        END IF;
    END IF;
    v_trust := COALESCE(
        NULLIF(v_document.source_attribution->>'trust', '')::float,
        source_kind_default_trust(COALESCE(
            v_document.source_attribution->>'kind',
            v_document.source_type
        ))
    );
    v_label := COALESCE(NULLIF(v_document.title, ''), NULLIF(v_document.path, ''), 'Source document');
    v_href := '/documents?document=' || v_document.id::text
        || CASE WHEN p_chunk_id IS NOT NULL THEN '&chunk=' || p_chunk_id::text ELSE '' END;
    RETURN jsonb_strip_nulls(jsonb_build_object(
        'citation_id', CASE
            WHEN p_chunk_id IS NOT NULL THEN 'chunk-' || p_chunk_id::text
            ELSE 'doc-' || v_document.id::text
        END,
        'label', v_label,
        'href', v_href,
        'document_id', v_document.id::text,
        'chunk_id', p_chunk_id::text,
        'source_kind', COALESCE(v_document.source_attribution->>'kind', v_document.source_type),
        'source_attribution', v_document.source_attribution,
        'trust_level', v_trust,
        'low_trust', v_trust < v_threshold,
        'locator', v_locator
    ));
END;
$$;

CREATE OR REPLACE FUNCTION enrich_source_document_payload(p_payload JSONB)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_document_id UUID;
BEGIN
    IF p_payload IS NULL OR p_payload ? 'error' THEN
        RETURN p_payload;
    END IF;
    v_document_id := _db_brain_try_uuid(p_payload->>'document_id');
    IF v_document_id IS NULL THEN
        RETURN p_payload;
    END IF;
    RETURN p_payload || jsonb_build_object(
        'citation_id', source_citation_envelope(v_document_id)->>'citation_id',
        'citation', source_citation_envelope(v_document_id),
        'source_attribution', COALESCE(
            (SELECT d.source_attribution FROM source_documents d WHERE d.id = v_document_id),
            '{}'::jsonb
        )
    );
END;
$$;

CREATE OR REPLACE FUNCTION enrich_source_chunk_payload(p_payload JSONB)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_item JSONB;
    v_items JSONB := '[]'::jsonb;
    v_document_id UUID;
    v_chunk_id UUID;
    v_citation JSONB;
BEGIN
    IF p_payload IS NULL OR p_payload ? 'error'
       OR jsonb_typeof(p_payload->'chunks') <> 'array' THEN
        RETURN p_payload;
    END IF;
    FOR v_item IN SELECT value FROM jsonb_array_elements(p_payload->'chunks')
    LOOP
        v_document_id := _db_brain_try_uuid(v_item->>'document_id');
        v_chunk_id := _db_brain_try_uuid(v_item->>'chunk_id');
        v_citation := source_citation_envelope(v_document_id, v_chunk_id);
        v_items := v_items || jsonb_build_array(jsonb_strip_nulls(v_item || jsonb_build_object(
            'citation_id', v_citation->>'citation_id',
            'citation', v_citation,
            'source_attribution', v_citation->'source_attribution',
            'trust_level', v_citation->'trust_level'
        )));
    END LOOP;
    RETURN jsonb_set(p_payload, '{chunks}', v_items, TRUE);
END;
$$;

-- Post-process DB-native memory tool output without moving shaping logic into
-- Python. The existing dispatcher stays authoritative; this function adds the
-- stable provenance contract at its boundary.
CREATE OR REPLACE FUNCTION enrich_memory_tool_result(p_tool_name TEXT, p_result JSONB)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_item JSONB;
    v_items JSONB := '[]'::jsonb;
    v_citation JSONB;
    v_memory_id UUID;
BEGIN
    IF p_result IS NULL OR NOT COALESCE((p_result->>'success')::boolean, FALSE) THEN
        RETURN p_result;
    END IF;

    IF p_tool_name = 'recall' AND jsonb_typeof(p_result #> '{output,memories}') = 'array' THEN
        FOR v_item IN SELECT value FROM jsonb_array_elements(p_result #> '{output,memories}')
        LOOP
            v_memory_id := _db_brain_try_uuid(v_item->>'memory_id');
            v_citation := memory_citation_envelope(v_memory_id);
            v_item := v_item || jsonb_build_object(
                'citation_id', v_citation->>'citation_id',
                'citation', v_citation,
                'provenance', v_citation,
                'source_attribution', COALESCE(
                    (SELECT m.source_attribution FROM memories m WHERE m.id = v_memory_id),
                    '{}'::jsonb
                ),
                'chunk_locator', v_citation->'locator'
            );
            v_items := v_items || jsonb_build_array(jsonb_strip_nulls(v_item));
        END LOOP;
        RETURN jsonb_set(p_result, '{output,memories}', v_items, TRUE);
    ELSIF p_tool_name IN ('open_memory', 'belief_history') THEN
        v_memory_id := _db_brain_try_uuid(COALESCE(
            p_result #>> '{output,memory,id}',
            p_result #>> '{output,memory,memory_id}'
        ));
        IF v_memory_id IS NOT NULL THEN
            v_citation := memory_citation_envelope(v_memory_id);
            RETURN jsonb_set(p_result, '{output,citation}', v_citation, TRUE);
        END IF;
    END IF;
    RETURN p_result;
END;
$$;

CREATE OR REPLACE FUNCTION memory_trust_distribution()
RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    SELECT jsonb_build_object(
        'active_memories', count(*),
        'distinct_trust_levels', count(DISTINCT round(m.trust_level::numeric, 6)),
        'low_trust', count(*) FILTER (
            WHERE m.trust_level < COALESCE(get_config_float('memory.low_trust_threshold'), 0.5)
        ),
        'minimum', min(m.trust_level),
        'maximum', max(m.trust_level),
        'average', avg(m.trust_level),
        'by_source_kind', COALESCE((
            SELECT jsonb_object_agg(kind, amount)
            FROM (
                SELECT COALESCE(NULLIF(source_attribution->>'kind', ''), 'unattributed') AS kind,
                       count(*) AS amount
                FROM memories
                WHERE status = 'active'
                GROUP BY 1
                ORDER BY 1
            ) kinds
        ), '{}'::jsonb)
    )
    FROM memories m
    WHERE m.status = 'active';
$$;

-- Repair only the recognizable legacy plateau: these rows all came through
-- the former implicit 0.5 source default and therefore lost the distinction
-- between confidence and provenance. Explicit non-plateau trust is untouched.
DO $$
DECLARE
    v_memory_id UUID;
BEGIN
    WITH candidates AS (
        SELECT m.id,
               CASE
                   WHEN jsonb_typeof(m.metadata->'source_references') = 'array'
                   THEN m.metadata->'source_references'
                   ELSE '[]'::jsonb
               END AS refs
        FROM memories m
        WHERE m.type = 'semantic'
          AND abs(m.trust_level - 0.4302279608697066) < 0.000000001
    ), rebuilt AS (
        SELECT c.id,
               COALESCE(jsonb_agg(n.normalized) FILTER (WHERE n.normalized <> '{}'::jsonb), '[]'::jsonb) AS refs
        FROM candidates c
        LEFT JOIN LATERAL jsonb_array_elements(c.refs) src ON TRUE
        LEFT JOIN LATERAL (
            SELECT normalize_source_reference(
                CASE
                    WHEN NULLIF(src->>'kind', '') IS NULL
                     AND NULLIF(src->>'ref', '') IS NULL
                     AND NULLIF(src->>'label', '') IS NULL
                     AND NULLIF(src->>'path', '') IS NULL
                     AND NULLIF(src->>'content_hash', '') IS NULL
                     AND NULLIF(src->>'document_id', '') IS NULL
                     AND NULLIF(src->>'source_document_id', '') IS NULL
                     AND NULLIF(src->>'chunk_id', '') IS NULL
                    THEN (src - 'trust') || jsonb_build_object('kind', 'inference')
                    WHEN COALESCE(src->>'trust_origin', '') = ''
                     AND NULLIF(src->>'trust', '')::float = 0.5
                    THEN src - 'trust'
                    ELSE src
                END
            ) AS normalized
        ) n ON TRUE
        GROUP BY c.id
    )
    UPDATE memories m
    SET metadata = jsonb_set(m.metadata, '{source_references}', r.refs, TRUE),
        source_attribution = normalize_source_reference(
            CASE
                WHEN NULLIF(m.source_attribution->>'kind', '') IS NULL
                 AND NULLIF(m.source_attribution->>'ref', '') IS NULL
                 AND NULLIF(m.source_attribution->>'label', '') IS NULL
                 AND NULLIF(m.source_attribution->>'path', '') IS NULL
                 AND NULLIF(m.source_attribution->>'content_hash', '') IS NULL
                 AND NULLIF(m.source_attribution->>'document_id', '') IS NULL
                 AND NULLIF(m.source_attribution->>'source_document_id', '') IS NULL
                 AND NULLIF(m.source_attribution->>'chunk_id', '') IS NULL
                THEN (m.source_attribution - 'trust') || jsonb_build_object('kind', 'inference')
                WHEN COALESCE(m.source_attribution->>'trust_origin', '') = ''
                 AND NULLIF(m.source_attribution->>'trust', '')::float = 0.5
                THEN m.source_attribution - 'trust'
                ELSE m.source_attribution
            END
        ),
        updated_at = CURRENT_TIMESTAMP
    FROM rebuilt r
    WHERE m.id = r.id;

    FOR v_memory_id IN
        SELECT id
        FROM memories
        WHERE type = 'semantic'
          AND abs(trust_level - 0.4302279608697066) < 0.000000001
    LOOP
        PERFORM sync_memory_trust(v_memory_id);
    END LOOP;
END;
$$;

UPDATE prompt_modules
SET content = replace(
        content,
        $old$- Use and cite relevant memories naturally.
- If nothing found, say so honestly. Do not invent memories.
- Prefer higher-trust, better-sourced memories when uncertain.$old$,
        $new$- Use and cite relevant memories naturally.
- If nothing found, say so honestly. Do not invent memories.
- Prefer higher-trust, better-sourced memories when uncertain.
- Recall and source-document results carry a stable `citation_id`, full
  `source_attribution`, trust, and an exact source locator. Every factual claim
  drawn from one of those results must end with its exact footnote marker,
  `[^citation_id]` (for example `[^mem-…]` or `[^chunk-…]`). Never invent or
  shorten an ID, and never cite a result that does not support the claim.
- Treat trust below the result's low-trust threshold as weak ground: qualify the
  claim in plain language rather than hiding the uncertainty. The renderer will
  also mark that citation as low trust.$new$
    ),
    updated_at = CURRENT_TIMESTAMP
WHERE key = 'conversation'
  AND position('stable `citation_id`' IN content) = 0;
