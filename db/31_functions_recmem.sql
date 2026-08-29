-- RecMem: recurrence-based memory consolidation.

-- Each baseline file runs as its own psql session at initdb, so it must pin
-- its own search_path: under an ag_catalog-first cluster default, unqualified
-- CREATEs would land Hexis objects in ag_catalog (the #77 fossil bug).
SET search_path = public, ag_catalog, "$user";

INSERT INTO config_defaults (key, value, description) VALUES
    ('memory.history_browse_max', '200'::jsonb,
     'Row ceiling for keyword-less time-window browsing in search_history (preview-grain rows)')
    ,
    ('memory.recall_graph_adjacency_weight', '0.12'::jsonb,
     'How much typed memory_edges adjacency contributes to fused recall scoring')
    ,
    ('memory.recmem_embedding_input_chars', '1800'::jsonb,
     'Maximum characters from a RecMem unit sent to the embedding service; full content remains stored for reading')
    ,
    ('memory.recmem_embedding_chunk_chars', '1800'::jsonb,
     'Maximum characters per RecMem embedding chunk; every chunk is embedded and parent vectors are averaged only as a coarse compatibility signal')
    ,
    ('memory.recmem_embedding_chunk_overlap_chars', '120'::jsonb,
     'Character overlap between RecMem embedding chunks')
ON CONFLICT (key) DO NOTHING;

-- Turns are labeled with the real names when configured (#56): "User:" as a
-- speaker label leaks into extracted memories ("The user is becoming a
-- leader") and splits one person across two identities in recall.
-- The label is the system's standing ASSUMPTION, not verified identity (#61):
-- channels that know who is talking (platform sender names) pass p_user_label;
-- the owner-name default covers the single-user paths, and the extraction
-- prompt treats either as overridable by the conversation's own evidence.
-- The one place turn labels are resolved (#56/#82): the agent's own name and
-- the user's, from config with init-profile fallback. Everything that names
-- the participants — turn rendering, extraction context, source labels —
-- reads this.
CREATE OR REPLACE FUNCTION get_turn_labels()
RETURNS JSONB AS $$
    SELECT jsonb_build_object(
        'user_label', COALESCE(
            NULLIF(get_config_text('agent.user_name'), ''),
            NULLIF(get_init_profile()#>>'{user,name}', ''),
            'User'),
        'agent_label', COALESCE(
            NULLIF(get_config_text('agent.name'), ''),
            NULLIF(get_init_profile()#>>'{agent,name}', ''),
            'Assistant'));
$$ LANGUAGE sql STABLE;

CREATE OR REPLACE FUNCTION format_recmem_turn(
    p_user_text TEXT,
    p_assistant_text TEXT,
    p_user_label TEXT DEFAULT NULL
) RETURNS TEXT AS $$
DECLARE
    labels JSONB := get_turn_labels();
    user_label TEXT := COALESCE(
        NULLIF(trim(COALESCE(p_user_label, '')), ''),
        labels->>'user_label');
    agent_label TEXT := labels->>'agent_label';
BEGIN
    RETURN format(
        '%s: %s%s%s: %s',
        user_label,
        COALESCE(p_user_text, ''),
        E'\n\n',
        agent_label,
        COALESCE(p_assistant_text, '')
    );
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION normalize_recmem_text(
    p_text TEXT
) RETURNS TEXT AS $$
    SELECT regexp_replace(
        regexp_replace(
            regexp_replace(
                replace(replace(COALESCE($1, ''), E'\r\n', E'\n'), E'\r', E'\n'),
                '[ \t]+$', '', 'gm'
            ),
            '^\n+', ''
        ),
        '\n+$', ''
    );
$$ LANGUAGE sql IMMUTABLE;

CREATE OR REPLACE FUNCTION recmem_embedding_input(
    p_content TEXT
) RETURNS TEXT AS $$
DECLARE
    source_text TEXT := COALESCE(p_content, '');
    max_chars INT := GREATEST(COALESCE(get_config_int('memory.recmem_embedding_input_chars'), 1800), 500);
    omitted_marker TEXT := E'\n\n[... full RecMem unit remains stored; embedding input was shortened ...]\n\n';
    available_chars INT;
    head_chars INT;
    tail_chars INT;
BEGIN
    IF length(source_text) <= max_chars THEN
        RETURN source_text;
    END IF;

    available_chars := max_chars - length(omitted_marker);
    IF available_chars < 100 THEN
        RETURN left(source_text, max_chars);
    END IF;

    head_chars := CEIL(available_chars * 0.7)::INT;
    tail_chars := available_chars - head_chars;
    RETURN left(source_text, head_chars) || omitted_marker || right(source_text, tail_chars);
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION ensure_recmem_embedding_chunks(
    p_unit_id UUID,
    p_chunk_chars INT DEFAULT NULL,
    p_overlap_chars INT DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    unit_row subconscious_units%ROWTYPE;
    chunk_chars INT := GREATEST(COALESCE(p_chunk_chars, get_config_int('memory.recmem_embedding_chunk_chars'), 1800), 500);
    overlap_chars INT := GREATEST(COALESCE(p_overlap_chars, get_config_int('memory.recmem_embedding_chunk_overlap_chars'), 120), 0);
    step_chars INT;
    chunk_count INT := 0;
    existing_count INT := 0;
    unchanged_count INT := 0;
    trimmed_count INT := 0;
BEGIN
    SELECT * INTO unit_row
    FROM subconscious_units
    WHERE id = p_unit_id;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('error', 'unit_not_found', 'unit_id', p_unit_id);
    END IF;

    overlap_chars := LEAST(overlap_chars, chunk_chars / 2);
    step_chars := GREATEST(chunk_chars - overlap_chars, 1);

    WITH source AS (
        SELECT COALESCE(unit_row.content, '') AS content
    ),
    starts AS (
        SELECT generate_series(0, GREATEST(length(content) - 1, 0), step_chars) AS char_start,
               content
        FROM source
        WHERE length(content) > 0
    ),
    incoming AS MATERIALIZED (
        SELECT
            (row_number() OVER (ORDER BY char_start) - 1)::int AS chunk_index,
            substring(content FROM char_start + 1 FOR chunk_chars) AS chunk_content,
            char_start,
            LEAST(char_start + chunk_chars, length(content)) AS char_end
        FROM starts
    ),
    shaped AS MATERIALIZED (
        SELECT
            chunk_index,
            chunk_content,
            encode(sha256(convert_to(chunk_content, 'UTF8')), 'hex') AS content_hash,
            GREATEST(1, length(chunk_content) / 4) AS token_count,
            char_start,
            char_end
        FROM incoming
    ),
    pre AS (
        SELECT count(*) AS existing_total,
               count(*) FILTER (WHERE c.content_hash = s.content_hash) AS unchanged_total
        FROM shaped s
        JOIN subconscious_unit_embedding_chunks c
          ON c.unit_id = p_unit_id AND c.chunk_index = s.chunk_index
    ),
    upserted AS (
        INSERT INTO subconscious_unit_embedding_chunks (
            unit_id,
            chunk_index,
            content,
            content_hash,
            token_count,
            char_start,
            char_end
        )
        SELECT
            p_unit_id,
            chunk_index,
            chunk_content,
            content_hash,
            token_count,
            char_start,
            char_end
        FROM shaped
        ON CONFLICT (unit_id, chunk_index) DO UPDATE SET
            content = CASE
                WHEN subconscious_unit_embedding_chunks.content_hash = EXCLUDED.content_hash
                    THEN subconscious_unit_embedding_chunks.content
                ELSE EXCLUDED.content
            END,
            content_hash = EXCLUDED.content_hash,
            token_count = EXCLUDED.token_count,
            char_start = EXCLUDED.char_start,
            char_end = EXCLUDED.char_end,
            embedding = CASE
                WHEN subconscious_unit_embedding_chunks.content_hash = EXCLUDED.content_hash
                    THEN subconscious_unit_embedding_chunks.embedding
                ELSE NULL
            END,
            embedded_at = CASE
                WHEN subconscious_unit_embedding_chunks.content_hash = EXCLUDED.content_hash
                    THEN subconscious_unit_embedding_chunks.embedded_at
                ELSE NULL
            END,
            embedding_model = CASE
                WHEN subconscious_unit_embedding_chunks.content_hash = EXCLUDED.content_hash
                    THEN subconscious_unit_embedding_chunks.embedding_model
                ELSE NULL
            END,
            embedding_status = CASE
                WHEN subconscious_unit_embedding_chunks.content_hash = EXCLUDED.content_hash
                    THEN subconscious_unit_embedding_chunks.embedding_status
                ELSE 'pending'
            END,
            embedding_attempts = CASE
                WHEN subconscious_unit_embedding_chunks.content_hash = EXCLUDED.content_hash
                    THEN subconscious_unit_embedding_chunks.embedding_attempts
                ELSE 0
            END,
            embedding_claimed_at = CASE
                WHEN subconscious_unit_embedding_chunks.content_hash = EXCLUDED.content_hash
                    THEN subconscious_unit_embedding_chunks.embedding_claimed_at
                ELSE NULL
            END,
            updated_at = CURRENT_TIMESTAMP
        RETURNING 1
    )
    SELECT count(*), COALESCE(pre.existing_total, 0), COALESCE(pre.unchanged_total, 0)
    INTO chunk_count, existing_count, unchanged_count
    FROM shaped
    CROSS JOIN pre
    GROUP BY pre.existing_total, pre.unchanged_total;

    chunk_count := COALESCE(chunk_count, 0);
    existing_count := COALESCE(existing_count, 0);
    unchanged_count := COALESCE(unchanged_count, 0);

    DELETE FROM subconscious_unit_embedding_chunks
    WHERE unit_id = p_unit_id
      AND chunk_index >= chunk_count;
    GET DIAGNOSTICS trimmed_count = ROW_COUNT;

    RETURN jsonb_build_object(
        'unit_id', p_unit_id,
        'count', chunk_count,
        'inserted', chunk_count - existing_count,
        'unchanged', unchanged_count,
        'reembedded', existing_count - unchanged_count,
        'trimmed', trimmed_count,
        'chunk_chars', chunk_chars,
        'overlap_chars', overlap_chars
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION claim_recmem_embedding_chunks(
    p_unit_id UUID,
    p_claim_timeout_s INT DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    timeout_s INT := COALESCE(p_claim_timeout_s, get_config_int('memory.recmem_embed_claim_timeout_s'), 120);
    payload JSONB;
BEGIN
    WITH candidate AS (
        SELECT c.id
        FROM subconscious_unit_embedding_chunks c
        WHERE c.unit_id = p_unit_id
          AND (
              c.embedding_status = 'pending'
              OR (
                  c.embedding_status = 'in_progress'
                  AND c.embedding_claimed_at < CURRENT_TIMESTAMP - (timeout_s * INTERVAL '1 second')
              )
          )
        ORDER BY c.chunk_index
        FOR UPDATE SKIP LOCKED
    ),
    claimed AS (
        UPDATE subconscious_unit_embedding_chunks c
        SET embedding_status = 'in_progress',
            embedding_claimed_at = CURRENT_TIMESTAMP,
            embedding_attempts = c.embedding_attempts + 1,
            updated_at = CURRENT_TIMESTAMP
        FROM candidate cand
        WHERE c.id = cand.id
        RETURNING c.id, c.unit_id, c.chunk_index, c.content, c.char_start, c.char_end, c.embedding_attempts
    )
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'chunk_id', id,
        'unit_id', unit_id,
        'chunk_index', chunk_index,
        'content', content,
        'char_start', char_start,
        'char_end', char_end,
        'attempts', embedding_attempts
    ) ORDER BY chunk_index), '[]'::jsonb)
    INTO payload
    FROM claimed;

    RETURN payload;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION fail_recmem_embedding_chunk(
    p_chunk_id UUID,
    p_error TEXT DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    max_attempts INT := COALESCE(get_config_int('memory.recmem_embed_max_attempts'), 3);
    final_status TEXT;
BEGIN
    UPDATE subconscious_unit_embedding_chunks
    SET embedding_status = CASE WHEN embedding_attempts >= max_attempts THEN 'failed' ELSE 'pending' END,
        embedding_claimed_at = NULL,
        metadata = COALESCE(metadata, '{}'::jsonb)
            || jsonb_build_object(
                'embedding_error',
                jsonb_build_object('error', p_error, 'at', CURRENT_TIMESTAMP)
            ),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_chunk_id
    RETURNING embedding_status INTO final_status;

    RETURN jsonb_build_object('chunk_id', p_chunk_id, 'embedding_status', final_status);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION embed_claimed_recmem_chunks(
    p_chunk_ids UUID[],
    p_contents TEXT[]
) RETURNS JSONB AS $$
DECLARE
    expected_count INT := COALESCE(array_length(p_chunk_ids, 1), 0);
    content_count INT := COALESCE(array_length(p_contents, 1), 0);
    embeddings vector[];
    embedding_count INT;
    updated_count INT := 0;
BEGIN
    IF expected_count = 0 THEN
        RETURN jsonb_build_object('updated', 0, 'skipped', true, 'reason', 'no_chunks');
    END IF;

    IF content_count <> expected_count THEN
        RAISE EXCEPTION 'RecMem chunk embedding input mismatch: % ids, % contents',
            expected_count,
            content_count;
    END IF;

    embeddings := get_embedding(p_contents);
    embedding_count := COALESCE(array_length(embeddings, 1), 0);

    IF embedding_count <> expected_count THEN
        RAISE EXCEPTION 'RecMem chunk embedding output mismatch: expected %, got %',
            expected_count,
            embedding_count;
    END IF;

    WITH payload AS (
        SELECT
            p_chunk_ids[i] AS chunk_id,
            embeddings[i] AS embedding
        FROM generate_subscripts(p_chunk_ids, 1) AS i
    )
    UPDATE subconscious_unit_embedding_chunks c
    SET embedding = payload.embedding,
        embedded_at = CURRENT_TIMESTAMP,
        embedding_model = get_config_text('embedding.model_id'),
        embedding_status = 'embedded',
        embedding_claimed_at = NULL,
        updated_at = CURRENT_TIMESTAMP
    FROM payload
    WHERE c.id = payload.chunk_id
      AND c.embedding_status = 'in_progress';

    GET DIAGNOSTICS updated_count = ROW_COUNT;

    IF updated_count <> expected_count THEN
        RAISE EXCEPTION 'RecMem chunk embedding update mismatch: expected %, updated %',
            expected_count,
            updated_count;
    END IF;

    RETURN jsonb_build_object('updated', updated_count);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION finalize_recmem_unit_embedding(
    p_unit_id UUID
) RETURNS JSONB AS $$
DECLARE
    total_chunks INT;
    embedded_chunks INT;
    failed_chunks INT;
    centroid vector;
BEGIN
    SELECT
        COUNT(*)::int,
        COUNT(*) FILTER (WHERE embedding_status = 'embedded' AND embedding IS NOT NULL)::int,
        COUNT(*) FILTER (WHERE embedding_status = 'failed')::int
    INTO total_chunks, embedded_chunks, failed_chunks
    FROM subconscious_unit_embedding_chunks
    WHERE unit_id = p_unit_id;

    IF COALESCE(total_chunks, 0) = 0 THEN
        RETURN jsonb_build_object('unit_id', p_unit_id, 'status', 'waiting', 'reason', 'no_chunks');
    END IF;

    IF COALESCE(failed_chunks, 0) > 0 THEN
        RETURN jsonb_build_object(
            'unit_id', p_unit_id,
            'status', 'waiting',
            'reason', 'failed_chunks',
            'chunks', jsonb_build_object('total', total_chunks, 'embedded', embedded_chunks, 'failed', failed_chunks)
        );
    END IF;

    IF embedded_chunks < total_chunks THEN
        RETURN jsonb_build_object(
            'unit_id', p_unit_id,
            'status', 'waiting',
            'reason', 'pending_chunks',
            'chunks', jsonb_build_object('total', total_chunks, 'embedded', embedded_chunks, 'failed', failed_chunks)
        );
    END IF;

    SELECT avg(embedding)
    INTO centroid
    FROM subconscious_unit_embedding_chunks
    WHERE unit_id = p_unit_id
      AND embedding_status = 'embedded'
      AND embedding IS NOT NULL;

    UPDATE subconscious_units
    SET embedding = centroid,
        embedded_at = CURRENT_TIMESTAMP,
        embedding_status = 'embedded',
        embedding_claimed_at = NULL,
        metadata = COALESCE(metadata, '{}'::jsonb)
            || jsonb_build_object(
                'recmem',
                COALESCE(metadata->'recmem', '{}'::jsonb)
                    || jsonb_build_object(
                        'embedding_chunks',
                        jsonb_build_object(
                            'count', total_chunks,
                            'embedded', embedded_chunks,
                            'finalized_at', CURRENT_TIMESTAMP,
                            'parent_embedding', 'average_chunk_centroid'
                        )
                    )
            ),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_unit_id
      AND status = 'active';

    RETURN jsonb_build_object(
        'unit_id', p_unit_id,
        'status', 'embedded',
        'chunks', jsonb_build_object('total', total_chunks, 'embedded', embedded_chunks)
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION compute_recmem_idempotency_key(
    p_user_text TEXT,
    p_assistant_text TEXT,
    p_session_id UUID DEFAULT NULL,
    p_source_identity TEXT DEFAULT NULL
) RETURNS TEXT AS $$
DECLARE
    normalized TEXT;
BEGIN
    IF NULLIF(trim(COALESCE(p_source_identity, '')), '') IS NOT NULL THEN
        RETURN 'src:' || trim(p_source_identity);
    END IF;

    normalized := normalize_recmem_text(p_user_text)
        || chr(30)
        || normalize_recmem_text(p_assistant_text)
        || chr(30)
        || COALESCE(p_session_id::text, '');

    RETURN 'hash:' || encode(digest(normalized, 'sha256'), 'hex');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION recmem_ingest_turn(
    p_user_text TEXT,
    p_assistant_text TEXT,
    p_session_id UUID DEFAULT NULL,
    p_source_identity TEXT DEFAULT NULL,
    p_turn_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    p_importance FLOAT DEFAULT 0.3,
    p_source_attribution JSONB DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb,
    p_user_label TEXT DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    unit_content TEXT;
    idem TEXT;
    new_id UUID;
    existing_id UUID;
BEGIN
    IF COALESCE(p_user_text, '') = '' AND COALESCE(p_assistant_text, '') = '' THEN
        RETURN jsonb_build_object('status', 'empty');
    END IF;

    unit_content := format_recmem_turn(p_user_text, p_assistant_text, p_user_label);
    idem := compute_recmem_idempotency_key(p_user_text, p_assistant_text, p_session_id, p_source_identity);

    INSERT INTO subconscious_units (
        session_id,
        source_identity,
        turn_at,
        content,
        user_text,
        assistant_text,
        importance,
        source_attribution,
        metadata,
        idempotency_key
    )
    VALUES (
        p_session_id,
        NULLIF(trim(COALESCE(p_source_identity, '')), ''),
        COALESCE(p_turn_at, CURRENT_TIMESTAMP),
        unit_content,
        COALESCE(p_user_text, ''),
        COALESCE(p_assistant_text, ''),
        LEAST(1.0, GREATEST(0.0, COALESCE(p_importance, 0.3))),
        COALESCE(p_source_attribution, '{}'::jsonb),
        COALESCE(p_metadata, '{}'::jsonb),
        idem
    )
    ON CONFLICT (idempotency_key) DO NOTHING
    RETURNING id INTO new_id;

    IF new_id IS NOT NULL THEN
        RETURN jsonb_build_object('unit_id', new_id, 'status', 'stored');
    END IF;

    SELECT id INTO existing_id
    FROM subconscious_units
    WHERE idempotency_key = idem;

    RETURN jsonb_build_object('unit_id', existing_id, 'status', 'duplicate');
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION touch_subconscious_units(p_ids UUID[])
RETURNS INT AS $$
DECLARE
    updated_count INT;
BEGIN
    IF p_ids IS NULL OR array_length(p_ids, 1) IS NULL THEN
        RETURN 0;
    END IF;

    UPDATE subconscious_units
    SET access_count = access_count + 1,
        last_accessed = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = ANY(p_ids)
      AND status = 'active';
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN COALESCE(updated_count, 0);
END;
$$ LANGUAGE plpgsql;

-- Free lexical recall across raw conversation turns, desk-loaded source
-- documents, and consolidated memory.
-- This deliberately avoids get_embedding(): it remains available when an
-- embedding provider is offline and is suitable for background review work.
CREATE OR REPLACE FUNCTION search_cross_session_history(
    p_query TEXT,
    p_limit INT DEFAULT 20,
    p_sources TEXT[] DEFAULT ARRAY['turn', 'memory']::TEXT[],
    p_created_after TIMESTAMPTZ DEFAULT NULL,
    p_created_before TIMESTAMPTZ DEFAULT NULL,
    p_exclude_session_id UUID DEFAULT NULL,
    -- Sensitivity enforcement (#92/#96): group contexts search with this
    -- TRUE; the operator and 1:1 recall keep everything.
    p_exclude_sensitive BOOLEAN DEFAULT FALSE
) RETURNS TABLE (
    source_kind TEXT,
    item_id UUID,
    session_id UUID,
    content TEXT,
    user_text TEXT,
    assistant_text TEXT,
    memory_type TEXT,
    occurred_at TIMESTAMPTZ,
    rank FLOAT,
    source_unit_ids UUID[],
    source_attribution JSONB,
    metadata JSONB
) AS $$
DECLARE
    -- Browse mode (#68): a time window with no keywords means "everything in
    -- the window, newest first" — '*' and '' count as no keywords. Without a
    -- window either, there is nothing to anchor on and we return empty.
    browse_mode BOOLEAN :=
        NULLIF(trim(COALESCE(p_query, '')), '') IS NULL
        OR trim(COALESCE(p_query, '')) = '*';
    -- Preview-grain rows are cheap, so browse affords a higher ceiling (#76).
    browse_cap INT := CASE
        WHEN NULLIF(trim(COALESCE(p_query, '')), '') IS NULL
          OR trim(COALESCE(p_query, '')) = '*'
        THEN GREATEST(COALESCE(get_config_int('memory.history_browse_max'), 200), 1)
        ELSE 100 END;
BEGIN
    IF browse_mode AND p_created_after IS NULL AND p_created_before IS NULL THEN
        RETURN;
    END IF;

    RETURN QUERY
    WITH query_doc AS (
        SELECT websearch_to_tsquery('english', CASE WHEN browse_mode THEN '' ELSE p_query END) AS query
    ),
    turn_hits AS (
        SELECT
            'turn'::TEXT AS source_kind,
            s.id AS item_id,
            s.session_id,
            -- Browse grain (#76): a timeline scan reads previews, not
            -- transcripts — open_memory / a keyword search fetch verbatim.
            CASE WHEN browse_mode AND length(s.content) > 280
                 THEN left(s.content, 280) || ' …'
                 ELSE s.content END AS content,
            -- The content preview IS the browse surface: the raw halves stay
            -- home, or a 200-row page still weighs a megabyte.
            CASE WHEN browse_mode THEN NULL ELSE s.user_text END AS user_text,
            CASE WHEN browse_mode THEN NULL ELSE s.assistant_text END AS assistant_text,
            NULL::TEXT AS memory_type,
            s.turn_at AS occurred_at,
            CASE WHEN browse_mode THEN 0.0 ELSE ts_rank_cd(to_tsvector('english', s.content), q.query, 32) END::FLOAT AS rank,
            ARRAY[s.id]::UUID[] AS source_unit_ids,
            s.source_attribution,
            s.metadata
        FROM subconscious_units s
        CROSS JOIN query_doc q
        WHERE 'turn' = ANY(COALESCE(p_sources, ARRAY['turn', 'memory']::TEXT[]))
          AND (browse_mode OR numnode(q.query) > 0)
          AND s.status = 'active'
          AND COALESCE(s.metadata#>>'{recmem,kind}', '') <> 'source_document_desk'
          AND (p_exclude_session_id IS NULL OR s.session_id IS DISTINCT FROM p_exclude_session_id)
          AND (p_created_after IS NULL OR s.turn_at >= p_created_after)
          AND (p_created_before IS NULL OR s.turn_at < p_created_before)
          AND (NOT p_exclude_sensitive
               OR COALESCE(s.source_attribution->>'sensitivity', '') <> 'private')
          AND (browse_mode OR to_tsvector('english', s.content) @@ q.query)
        ORDER BY rank DESC, occurred_at DESC, item_id
        LIMIT LEAST(GREATEST(COALESCE(p_limit, 20), 1), browse_cap)
    ),
    desk_hits AS (
        SELECT
            'desk'::TEXT AS source_kind,
            s.id AS item_id,
            s.session_id,
            CASE WHEN browse_mode AND length(s.content) > 500
                 THEN left(s.content, 500) || ' …'
                 ELSE s.content END AS content,
            NULL::TEXT AS user_text,
            NULL::TEXT AS assistant_text,
            NULL::TEXT AS memory_type,
            s.turn_at AS occurred_at,
            CASE WHEN browse_mode THEN 0.0 ELSE ts_rank_cd(to_tsvector('english', s.content), q.query, 32) END::FLOAT AS rank,
            ARRAY[s.id]::UUID[] AS source_unit_ids,
            s.source_attribution,
            s.metadata
        FROM subconscious_units s
        CROSS JOIN query_doc q
        WHERE 'desk' = ANY(COALESCE(p_sources, ARRAY['turn', 'memory']::TEXT[]))
          AND (browse_mode OR numnode(q.query) > 0)
          AND s.status = 'active'
          AND COALESCE(s.metadata#>>'{recmem,kind}', '') = 'source_document_desk'
          AND (p_created_after IS NULL OR s.turn_at >= p_created_after)
          AND (p_created_before IS NULL OR s.turn_at < p_created_before)
          AND (NOT p_exclude_sensitive
               OR COALESCE(s.source_attribution->>'sensitivity', '') <> 'private')
          AND (browse_mode OR to_tsvector('english', s.content) @@ q.query)
        ORDER BY rank DESC, occurred_at DESC, item_id
        LIMIT LEAST(GREATEST(COALESCE(p_limit, 20), 1), browse_cap)
    ),
    memory_hits AS (
        SELECT
            'memory'::TEXT AS source_kind,
            m.id AS item_id,
            (
                SELECT su.session_id
                FROM memory_source_units msu
                JOIN subconscious_units su ON su.id = msu.subconscious_unit_id
                WHERE msu.memory_id = m.id AND su.session_id IS NOT NULL
                ORDER BY su.turn_at DESC, su.id
                LIMIT 1
            ) AS session_id,
            m.content,
            NULL::TEXT AS user_text,
            NULL::TEXT AS assistant_text,
            m.type::TEXT AS memory_type,
            m.created_at AS occurred_at,
            CASE WHEN browse_mode THEN 0.0 ELSE ts_rank_cd(to_tsvector('english', m.content), q.query, 32) END::FLOAT AS rank,
            COALESCE(
                (
                    SELECT array_agg(msu.subconscious_unit_id ORDER BY msu.created_at, msu.subconscious_unit_id)
                    FROM memory_source_units msu
                    WHERE msu.memory_id = m.id
                ),
                '{}'::UUID[]
            ) AS source_unit_ids,
            m.source_attribution,
            m.metadata
        FROM memories m
        CROSS JOIN query_doc q
        WHERE 'memory' = ANY(COALESCE(p_sources, ARRAY['turn', 'memory']::TEXT[]))
          AND (browse_mode OR numnode(q.query) > 0)
          AND m.status = 'active'
          AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
          AND (p_created_after IS NULL OR m.created_at >= p_created_after)
          AND (p_created_before IS NULL OR m.created_at < p_created_before)
          AND (NOT p_exclude_sensitive
               OR COALESCE(m.source_attribution->>'sensitivity', '') <> 'private')
          AND (browse_mode OR to_tsvector('english', m.content) @@ q.query)
        ORDER BY rank DESC, occurred_at DESC, item_id
        LIMIT LEAST(GREATEST(COALESCE(p_limit, 20), 1), browse_cap)
    )
    SELECT hits.*
    FROM (
        SELECT * FROM turn_hits
        UNION ALL
        SELECT * FROM desk_hits
        UNION ALL
        SELECT * FROM memory_hits
    ) hits
    ORDER BY hits.rank DESC, hits.occurred_at DESC, hits.source_kind, hits.item_id
    LIMIT LEAST(GREATEST(COALESCE(p_limit, 20), 1), browse_cap);
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION claim_recmem_unembedded_batch(
    p_limit INT DEFAULT 32,
    p_claim_timeout_s INT DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    timeout_s INT := COALESCE(p_claim_timeout_s, get_config_int('memory.recmem_embed_claim_timeout_s'), 120);
    payload JSONB;
BEGIN
    WITH candidate AS (
        SELECT id
        FROM subconscious_units
        WHERE status = 'active'
          AND (
              embedding_status = 'pending'
              OR (
                  embedding_status = 'in_progress'
                  AND embedding_claimed_at < CURRENT_TIMESTAMP - (timeout_s * INTERVAL '1 second')
              )
          )
        ORDER BY created_at
        FOR UPDATE SKIP LOCKED
        LIMIT GREATEST(COALESCE(p_limit, 32), 1)
    ),
    claimed AS (
        UPDATE subconscious_units u
        SET embedding_status = 'in_progress',
            embedding_claimed_at = CURRENT_TIMESTAMP,
            embedding_attempts = embedding_attempts + 1,
            updated_at = CURRENT_TIMESTAMP
        FROM candidate c
        WHERE u.id = c.id
        RETURNING
            u.id,
            u.content,
            recmem_embedding_input(u.content) AS embedding_input,
            u.embedding_attempts
    )
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'unit_id', id,
        'content', content,
        'embedding_input', embedding_input,
        'embedding_input_chars', length(embedding_input),
        'attempts', embedding_attempts
    )), '[]'::jsonb)
    INTO payload
    FROM claimed;

    RETURN payload;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION apply_recmem_embeddings(
    p_payload JSONB
) RETURNS JSONB AS $$
DECLARE
    item JSONB;
    updated_count INT := 0;
    row_count INT := 0;
    emb_arr FLOAT4[];
BEGIN
    FOR item IN SELECT value FROM jsonb_array_elements(COALESCE(p_payload, '[]'::jsonb))
    LOOP
        SELECT array_agg(value::float4 ORDER BY ord)
        INTO emb_arr
        FROM jsonb_array_elements_text(item->'embedding') WITH ORDINALITY AS e(value, ord);

        IF emb_arr IS NULL OR array_length(emb_arr, 1) IS NULL THEN
            CONTINUE;
        END IF;

        UPDATE subconscious_units
        SET embedding = emb_arr::vector,
            embedded_at = CURRENT_TIMESTAMP,
            embedding_status = 'embedded',
            embedding_claimed_at = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = (item->>'unit_id')::uuid
          AND embedding_status = 'in_progress';

        GET DIAGNOSTICS row_count = ROW_COUNT;
        updated_count := updated_count + row_count;
    END LOOP;

    RETURN jsonb_build_object('updated', updated_count);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION fail_recmem_embedding(
    p_unit_id UUID,
    p_error TEXT DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    max_attempts INT := COALESCE(get_config_int('memory.recmem_embed_max_attempts'), 3);
    final_status TEXT;
BEGIN
    UPDATE subconscious_units
    SET embedding_status = CASE WHEN embedding_attempts >= max_attempts THEN 'failed' ELSE 'pending' END,
        embedding_claimed_at = NULL,
        metadata = COALESCE(metadata, '{}'::jsonb)
            || jsonb_build_object(
                'recmem',
                COALESCE(metadata->'recmem', '{}'::jsonb)
                    || jsonb_build_object(
                        'embedding_error',
                        jsonb_build_object('error', p_error, 'at', CURRENT_TIMESTAMP)
                    )
            ),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_unit_id
    RETURNING embedding_status INTO final_status;

    RETURN jsonb_build_object('unit_id', p_unit_id, 'embedding_status', final_status);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION claim_recmem_unrouted_batch(
    p_limit INT DEFAULT 32,
    p_claim_timeout_s INT DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    timeout_s INT := COALESCE(p_claim_timeout_s, get_config_int('memory.recmem_route_claim_timeout_s'), 60);
    payload JSONB;
BEGIN
    WITH candidate AS (
        SELECT id
        FROM subconscious_units
        WHERE status = 'active'
          AND embedding_status = 'embedded'
          AND (
              route_status = 'unrouted'
              OR (
                  route_status = 'routing'
                  AND last_routed_at < CURRENT_TIMESTAMP - (timeout_s * INTERVAL '1 second')
              )
          )
        ORDER BY last_routed_at NULLS FIRST, created_at
        FOR UPDATE SKIP LOCKED
        LIMIT GREATEST(COALESCE(p_limit, 32), 1)
    ),
    claimed AS (
        UPDATE subconscious_units u
        SET route_status = 'routing',
            route_attempts = route_attempts + 1,
            last_routed_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        FROM candidate c
        WHERE u.id = c.id
        RETURNING u.id, u.content, u.route_attempts
    )
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'unit_id', id,
        'content', content,
        'attempts', route_attempts
    )), '[]'::jsonb)
    INTO payload
    FROM claimed;

    RETURN payload;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION _recmem_pending_queue_depth()
RETURNS INT AS $$
    SELECT COUNT(*)::int
    FROM recmem_consolidation_tasks
    WHERE status IN ('pending','in_progress');
$$ LANGUAGE sql STABLE;

CREATE OR REPLACE FUNCTION recmem_route_unit(
    p_unit_id UUID
) RETURNS JSONB AS $$
DECLARE
    unit_row subconscious_units%ROWTYPE;
    theta_sim FLOAT := COALESCE(get_config_float('memory.recmem_theta_sim'), 0.7);
    theta_merge FLOAT := COALESCE(get_config_float('memory.recmem_theta_sim_merge'), 0.78);
    theta_count INT := COALESCE(get_config_int('memory.recmem_theta_count'), 5);
    top_k INT := COALESCE(get_config_int('memory.recmem_top_k'), 20);
    queue_max INT := COALESCE(get_config_int('memory.recmem_queue_max'), 5000);
    nearest_memory_id UUID;
    nearest_similarity FLOAT;
    source_ids UUID[];
    recurrence_count INT;
    max_neighbor_similarity FLOAT;
    task_id UUID;
    overlaps_open_create BOOLEAN;
BEGIN
    SELECT * INTO unit_row
    FROM subconscious_units
    WHERE id = p_unit_id
      AND status = 'active'
      AND embedding_status = 'embedded';

    IF NOT FOUND THEN
        RETURN jsonb_build_object('unit_id', p_unit_id, 'status', 'skipped', 'reason', 'not_embedded_or_inactive');
    END IF;

    SELECT m.id, 1 - (m.embedding <=> unit_row.embedding)
    INTO nearest_memory_id, nearest_similarity
    FROM memories m
    WHERE m.status = 'active'
      AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
      AND m.type = 'episodic'
      AND m.embedding IS NOT NULL
    ORDER BY m.embedding <=> unit_row.embedding
    LIMIT 1;

    IF nearest_memory_id IS NOT NULL
       AND nearest_similarity >= theta_merge
       AND COALESCE(unit_row.route_result->>'merge_rejected_target_memory_id', '') <> nearest_memory_id::text THEN
        IF _recmem_pending_queue_depth() >= queue_max THEN
            UPDATE subconscious_units
            SET route_status = 'raw_only',
                route_result = route_result || jsonb_build_object(
                    'decision', 'raw_only',
                    'reason', 'queue_full',
                    'nearest_memory_id', nearest_memory_id,
                    'similarity', nearest_similarity
                ),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = p_unit_id;
            RETURN jsonb_build_object('unit_id', p_unit_id, 'status', 'raw_only', 'reason', 'queue_full');
        END IF;

        SELECT id INTO task_id
        FROM recmem_consolidation_tasks
        WHERE task_type = 'episode_merge'
          AND status = 'pending'
          AND target_memory_id = nearest_memory_id
        ORDER BY created_at
        LIMIT 1
        FOR UPDATE;

        IF task_id IS NOT NULL THEN
            UPDATE recmem_consolidation_tasks
            SET source_unit_ids = (
                    SELECT array_agg(DISTINCT source_id)
                    FROM unnest(array_append(source_unit_ids, p_unit_id)) AS source_id
                ),
                task_payload = task_payload || jsonb_build_object('coalesced_at', CURRENT_TIMESTAMP),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = task_id;
        ELSE
            INSERT INTO recmem_consolidation_tasks (
                task_type,
                trigger_unit_id,
                target_memory_id,
                source_unit_ids,
                max_similarity,
                task_payload
            )
            VALUES (
                'episode_merge',
                p_unit_id,
                nearest_memory_id,
                ARRAY[p_unit_id],
                nearest_similarity,
                jsonb_build_object(
                    'unit_content', unit_row.content,
                    'target_memory_id', nearest_memory_id,
                    'similarity', nearest_similarity
                )
            )
            RETURNING id INTO task_id;
        END IF;

        UPDATE subconscious_units
        SET route_status = 'merge_queued',
            route_result = route_result || jsonb_build_object(
                'decision', 'merge_queued',
                'task_id', task_id,
                'target_memory_id', nearest_memory_id,
                'similarity', nearest_similarity
            ),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = p_unit_id;

        RETURN jsonb_build_object('unit_id', p_unit_id, 'status', 'merge_queued', 'task_id', task_id);
    END IF;

    WITH neighbors AS (
        SELECT s.id, 1 - (s.embedding <=> unit_row.embedding) AS similarity
        FROM subconscious_units s
        WHERE s.status = 'active'
          AND s.embedding_status = 'embedded'
          AND s.embedding IS NOT NULL
        ORDER BY s.embedding <=> unit_row.embedding
        LIMIT GREATEST(top_k, theta_count)
    ),
    recurrent AS (
        SELECT id, similarity
        FROM neighbors
        WHERE similarity >= theta_sim
    )
    SELECT array_agg(id ORDER BY id), COUNT(*)::int, MAX(similarity)
    INTO source_ids, recurrence_count, max_neighbor_similarity
    FROM recurrent;

    source_ids := COALESCE(source_ids, ARRAY[p_unit_id]);
    IF NOT p_unit_id = ANY(source_ids) THEN
        source_ids := source_ids || p_unit_id;
        recurrence_count := COALESCE(recurrence_count, 0) + 1;
    END IF;

    IF COALESCE(recurrence_count, 0) < theta_count THEN
        UPDATE subconscious_units
        SET route_status = 'raw_only',
            route_result = route_result || jsonb_build_object(
                'decision', 'raw_only',
                'recurrence_count', COALESCE(recurrence_count, 0),
                'max_similarity', max_neighbor_similarity
            ),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = p_unit_id;
        RETURN jsonb_build_object('unit_id', p_unit_id, 'status', 'raw_only', 'recurrence_count', COALESCE(recurrence_count, 0));
    END IF;

    SELECT EXISTS (
        SELECT 1
        FROM recmem_consolidation_tasks t
        WHERE t.task_type = 'episode_create'
          AND t.status IN ('pending','in_progress')
          AND t.source_unit_ids && source_ids
    ) INTO overlaps_open_create;

    IF overlaps_open_create THEN
        UPDATE subconscious_units
        SET route_status = 'raw_only',
            route_result = route_result || jsonb_build_object(
                'decision', 'raw_only',
                'reason', 'open_create_overlap',
                'recurrence_count', recurrence_count
            ),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = p_unit_id;
        RETURN jsonb_build_object('unit_id', p_unit_id, 'status', 'raw_only', 'reason', 'open_create_overlap');
    END IF;

    IF _recmem_pending_queue_depth() >= queue_max THEN
        UPDATE subconscious_units
        SET route_status = 'raw_only',
            route_result = route_result || jsonb_build_object(
                'decision', 'raw_only',
                'reason', 'queue_full_create_paused',
                'recurrence_count', recurrence_count
            ),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = p_unit_id;
        RETURN jsonb_build_object('unit_id', p_unit_id, 'status', 'raw_only', 'reason', 'queue_full_create_paused');
    END IF;

    INSERT INTO recmem_consolidation_tasks (
        task_type,
        trigger_unit_id,
        source_unit_ids,
        recurrence_count,
        max_similarity,
        task_payload
    )
    VALUES (
        'episode_create',
        p_unit_id,
        source_ids,
        recurrence_count,
        max_neighbor_similarity,
        jsonb_build_object('source_unit_ids', source_ids, 'recurrence_count', recurrence_count)
    )
    RETURNING id INTO task_id;

    UPDATE subconscious_units
    SET route_status = 'create_queued',
        route_result = route_result || jsonb_build_object(
            'decision', 'create_queued',
            'task_id', task_id,
            'recurrence_count', recurrence_count,
            'max_similarity', max_neighbor_similarity
        ),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = ANY(source_ids)
      AND route_status IN ('routing','raw_only','unrouted');

    RETURN jsonb_build_object('unit_id', p_unit_id, 'status', 'create_queued', 'task_id', task_id, 'recurrence_count', recurrence_count);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION fail_recmem_routing(
    p_unit_id UUID,
    p_error TEXT DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    max_attempts INT := COALESCE(get_config_int('memory.recmem_route_max_attempts'), 3);
    final_status TEXT;
BEGIN
    UPDATE subconscious_units
    SET route_status = CASE WHEN route_attempts >= max_attempts THEN 'route_failed' ELSE 'unrouted' END,
        route_result = jsonb_build_object('error', p_error, 'at', CURRENT_TIMESTAMP),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_unit_id
    RETURNING route_status INTO final_status;

    RETURN jsonb_build_object('unit_id', p_unit_id, 'route_status', final_status);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION claim_recmem_consolidation_task(
    p_claim_timeout_s INT DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    timeout_s INT := COALESCE(p_claim_timeout_s, get_config_int('memory.recmem_task_claim_timeout_s'), 600);
    task JSONB;
BEGIN
    WITH candidate AS (
        SELECT id
        FROM recmem_consolidation_tasks
        WHERE (status = 'pending' AND next_attempt_at <= CURRENT_TIMESTAMP)
           OR (status = 'in_progress' AND started_at < CURRENT_TIMESTAMP - (timeout_s * INTERVAL '1 second'))
        ORDER BY next_attempt_at, created_at
        FOR UPDATE SKIP LOCKED
        LIMIT 1
    ),
    claimed AS (
        UPDATE recmem_consolidation_tasks t
        SET status = 'in_progress',
            started_at = CURRENT_TIMESTAMP,
            attempts = attempts + 1,
            updated_at = CURRENT_TIMESTAMP
        FROM candidate c
        WHERE t.id = c.id
        RETURNING t.*
    )
    SELECT to_jsonb(claimed)
    INTO task
    FROM claimed;

    RETURN task;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION fail_recmem_consolidation_task(
    p_task_id UUID,
    p_error TEXT DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    max_attempts INT := COALESCE(get_config_int('memory.recmem_task_max_attempts'), 3);
    backoff_base INT := COALESCE(get_config_int('memory.recmem_task_backoff_base_s'), 30);
    task recmem_consolidation_tasks%ROWTYPE;
BEGIN
    SELECT * INTO task
    FROM recmem_consolidation_tasks
    WHERE id = p_task_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('task_id', p_task_id, 'status', 'missing');
    END IF;

    IF task.attempts >= max_attempts THEN
        UPDATE recmem_consolidation_tasks
        SET status = 'failed',
            error = p_error,
            completed_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = p_task_id;
        RETURN jsonb_build_object('task_id', p_task_id, 'status', 'failed');
    END IF;

    UPDATE recmem_consolidation_tasks
    SET status = 'pending',
        started_at = NULL,
        next_attempt_at = CURRENT_TIMESTAMP + (backoff_base * power(2, GREATEST(attempts - 1, 0))) * INTERVAL '1 second',
        error = p_error,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_task_id;

    RETURN jsonb_build_object('task_id', p_task_id, 'status', 'pending');
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION link_memory_to_source_unit(
    p_memory_id UUID,
    p_unit_id UUID,
    p_role TEXT DEFAULT 'source'
) RETURNS BOOLEAN AS $$
BEGIN
    INSERT INTO memory_source_units (memory_id, subconscious_unit_id, role)
    VALUES (p_memory_id, p_unit_id, COALESCE(p_role, 'source'))
    ON CONFLICT (memory_id, subconscious_unit_id) DO UPDATE
    SET role = EXCLUDED.role;

    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION apply_recmem_episode_merge(
    p_task_id UUID,
    p_merged_content TEXT DEFAULT NULL,
    p_should_merge BOOLEAN DEFAULT TRUE
) RETURNS JSONB AS $$
DECLARE
    task recmem_consolidation_tasks%ROWTYPE;
    old_content TEXT;
    new_embedding vector;
    unit_id UUID;
    queue_max INT := COALESCE(get_config_int('memory.recmem_queue_max'), 5000);
BEGIN
    SELECT * INTO task
    FROM recmem_consolidation_tasks
    WHERE id = p_task_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('task_id', p_task_id, 'status', 'missing');
    END IF;

    IF NOT COALESCE(p_should_merge, TRUE) THEN
        UPDATE subconscious_units
        SET route_status = 'routing',
            route_result = route_result || jsonb_build_object(
                'merge_rejected', true,
                'merge_rejected_target_memory_id', task.target_memory_id,
                'at', CURRENT_TIMESTAMP
            ),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ANY(task.source_unit_ids);

        FOREACH unit_id IN ARRAY task.source_unit_ids LOOP
            PERFORM recmem_route_unit(unit_id);
        END LOOP;

        UPDATE recmem_consolidation_tasks
        SET status = 'completed',
            completed_at = CURRENT_TIMESTAMP,
            result = jsonb_build_object('merged', false),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = p_task_id;

        RETURN jsonb_build_object('task_id', p_task_id, 'status', 'completed', 'merged', false);
    END IF;

    SELECT content INTO old_content
    FROM memories
    WHERE id = task.target_memory_id;

    IF task.target_memory_id IS NULL OR old_content IS NULL THEN
        PERFORM fail_recmem_consolidation_task(p_task_id, 'target memory missing');
        RETURN jsonb_build_object('task_id', p_task_id, 'status', 'failed', 'reason', 'target_missing');
    END IF;

    new_embedding := (get_embedding(ARRAY[COALESCE(NULLIF(p_merged_content, ''), old_content)]))[1];

    UPDATE memories
    SET content = COALESCE(NULLIF(p_merged_content, ''), old_content),
        embedding = new_embedding,
        metadata = COALESCE(metadata, '{}'::jsonb)
            || jsonb_build_object(
                'recmem',
                COALESCE(metadata->'recmem', '{}'::jsonb)
                    || jsonb_build_object(
                        'merge_history',
                        COALESCE(metadata#>'{recmem,merge_history}', '[]'::jsonb)
                            || jsonb_build_array(jsonb_build_object('content', old_content, 'merged_at', CURRENT_TIMESTAMP))
                    )
            ),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = task.target_memory_id;

    FOREACH unit_id IN ARRAY task.source_unit_ids LOOP
        PERFORM link_memory_to_source_unit(task.target_memory_id, unit_id, 'merge_addition');
    END LOOP;

    UPDATE subconscious_units
    SET consolidated_at = CURRENT_TIMESTAMP,
        route_status = 'merged',
        updated_at = CURRENT_TIMESTAMP
    WHERE id = ANY(task.source_unit_ids);

    -- semantic_refine retired (#57): conscious extraction (db/61) is the sole
    -- semantic-fact minter — it carries provenance and routes through the
    -- belief-revision policy; recmem's refinement minted unattributed
    -- near-duplicates. The handler remains only to drain legacy queued tasks.

    UPDATE recmem_consolidation_tasks
    SET status = 'completed',
        completed_at = CURRENT_TIMESTAMP,
        result = jsonb_build_object('merged', true, 'target_memory_id', task.target_memory_id),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_task_id;

    RETURN jsonb_build_object('task_id', p_task_id, 'status', 'completed', 'merged', true, 'target_memory_id', task.target_memory_id);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION apply_recmem_episode_create(
    p_task_id UUID,
    p_episodes JSONB
) RETURNS JSONB AS $$
DECLARE
    task recmem_consolidation_tasks%ROWTYPE;
    item JSONB;
    episode_content TEXT;
    new_embedding vector;
    memory_id UUID;
    created_ids UUID[] := ARRAY[]::UUID[];
    unit_id UUID;
    source_attr JSONB;
    queue_max INT := COALESCE(get_config_int('memory.recmem_queue_max'), 5000);
    span_from TIMESTAMPTZ;
    span_to TIMESTAMPTZ;
BEGIN
    SELECT * INTO task
    FROM recmem_consolidation_tasks
    WHERE id = p_task_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('task_id', p_task_id, 'status', 'missing');
    END IF;

    -- Scene metadata (#73): the memory carries when the experience happened
    -- (the units' time span), not just when consolidation ran — timeline
    -- queries and retention grouping key off lived time.
    SELECT min(turn_at), max(turn_at) INTO span_from, span_to
    FROM subconscious_units
    WHERE id = ANY(task.source_unit_ids);

    source_attr := jsonb_build_object(
        'kind', 'recmem',
        'ref', task.id::text,
        'label', 'RecMem episodic consolidation',
        'observed_at', CURRENT_TIMESTAMP,
        'trust', 0.9
    );
    -- Sensitivity propagates from source to derivation (#92): one private
    -- turn in a scene marks the whole scene memory private.
    IF EXISTS (
        SELECT 1 FROM subconscious_units u
        WHERE u.id = ANY(task.source_unit_ids)
          AND u.source_attribution->>'sensitivity' = 'private'
    ) THEN
        source_attr := source_attr || jsonb_build_object('sensitivity', 'private');
    END IF;

    FOR item IN SELECT value FROM jsonb_array_elements(COALESCE(p_episodes, '[]'::jsonb))
    LOOP
        episode_content := COALESCE(item->>'content', item->>'episode', item#>>'{}');
        IF NULLIF(trim(COALESCE(episode_content, '')), '') IS NULL THEN
            CONTINUE;
        END IF;

        new_embedding := (get_embedding(ARRAY[episode_content]))[1];
        memory_id := create_memory_with_embedding(
            'episodic',
            episode_content,
            new_embedding,
            COALESCE(NULLIF(item->>'importance', '')::float, 0.6),
            source_attr,
            0.9,
            jsonb_build_object('recmem', jsonb_strip_nulls(jsonb_build_object(
                'task_id', task.id,
                'source_unit_ids', task.source_unit_ids,
                'reason', task.task_payload->>'reason',
                'session_id', task.task_payload->>'session_id',
                'occurred_from', span_from,
                'occurred_to', span_to
            )))
        );
        created_ids := created_ids || memory_id;

        FOREACH unit_id IN ARRAY task.source_unit_ids LOOP
            PERFORM link_memory_to_source_unit(memory_id, unit_id, 'source');
        END LOOP;

        -- semantic_refine retired (#57): see apply_recmem_episode_merge.
    END LOOP;

    IF cardinality(created_ids) = 0 THEN
        UPDATE subconscious_units
        SET route_status = 'raw_only',
            route_result = route_result || jsonb_build_object(
                'episode_create_empty', true,
                'task_id', p_task_id,
                'at', CURRENT_TIMESTAMP
            ),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ANY(task.source_unit_ids);

        UPDATE recmem_consolidation_tasks
        SET status = 'completed',
            completed_at = CURRENT_TIMESTAMP,
            result = jsonb_build_object('memory_ids', created_ids, 'empty', true),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = p_task_id;

        RETURN jsonb_build_object('task_id', p_task_id, 'status', 'completed', 'memory_ids', created_ids, 'empty', true);
    END IF;

    UPDATE subconscious_units
    SET consolidated_at = CURRENT_TIMESTAMP,
        route_status = 'episode_created',
        updated_at = CURRENT_TIMESTAMP
    WHERE id = ANY(task.source_unit_ids);

    UPDATE recmem_consolidation_tasks
    SET status = 'completed',
        completed_at = CURRENT_TIMESTAMP,
        result = jsonb_build_object('memory_ids', created_ids),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_task_id;

    RETURN jsonb_build_object('task_id', p_task_id, 'status', 'completed', 'memory_ids', created_ids);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION apply_recmem_semantic_facts(
    p_task_id UUID,
    p_facts JSONB
) RETURNS JSONB AS $$
DECLARE
    task recmem_consolidation_tasks%ROWTYPE;
    item JSONB;
    fact_content TEXT;
    fact_embedding vector;
    duplicate_id UUID;
    memory_id UUID;
    created_ids UUID[] := ARRAY[]::UUID[];
    unit_id UUID;
BEGIN
    SELECT * INTO task
    FROM recmem_consolidation_tasks
    WHERE id = p_task_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('task_id', p_task_id, 'status', 'missing');
    END IF;

    FOR item IN SELECT value FROM jsonb_array_elements(COALESCE(p_facts, '[]'::jsonb))
    LOOP
        fact_content := COALESCE(item->>'content', item->>'fact', item#>>'{}');
        IF NULLIF(trim(COALESCE(fact_content, '')), '') IS NULL THEN
            CONTINUE;
        END IF;

        fact_embedding := (get_embedding(ARRAY[fact_content]))[1];

        SELECT m.id INTO duplicate_id
        FROM memories m
        WHERE m.status = 'active'
          AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
          AND m.type = 'semantic'
          AND 1 - (m.embedding <=> fact_embedding) >= 0.92
        ORDER BY m.embedding <=> fact_embedding
        LIMIT 1;

        IF duplicate_id IS NOT NULL THEN
            CONTINUE;
        END IF;

        memory_id := create_memory_with_embedding(
            'semantic',
            fact_content,
            fact_embedding,
            COALESCE(NULLIF(item->>'importance', '')::float, 0.55),
            jsonb_build_object(
                'kind', 'recmem',
                'ref', task.id::text,
                'label', 'RecMem semantic refinement',
                'observed_at', CURRENT_TIMESTAMP,
                'trust', 0.85
            ),
            0.85,
            jsonb_build_object('recmem', jsonb_build_object('task_id', task.id, 'episode_id', task.target_memory_id, 'source_unit_ids', task.source_unit_ids))
        );
        created_ids := created_ids || memory_id;

        FOREACH unit_id IN ARRAY task.source_unit_ids LOOP
            PERFORM link_memory_to_source_unit(memory_id, unit_id, 'source');
        END LOOP;

        IF task.target_memory_id IS NOT NULL THEN
            PERFORM create_memory_relationship(memory_id, task.target_memory_id, 'DERIVED_FROM', '{}'::jsonb);
        END IF;
    END LOOP;

    UPDATE recmem_consolidation_tasks
    SET status = 'completed',
        completed_at = CURRENT_TIMESTAMP,
        result = jsonb_build_object('memory_ids', created_ids),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_task_id;

    RETURN jsonb_build_object('task_id', p_task_id, 'status', 'completed', 'memory_ids', created_ids);
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

CREATE OR REPLACE FUNCTION recmem_gc(
    p_limit INT DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    gc_limit INT := GREATEST(COALESCE(p_limit, get_config_int('memory.recmem_gc_batch_size'), 200), 1);
    idle_days INT := GREATEST(COALESCE(get_config_int('memory.recmem_gc_idle_days'), 30), 1);
    consolidated_grace_days INT := GREATEST(COALESCE(get_config_int('memory.recmem_gc_consolidated_grace_days'), 7), 1);
    task_retention_days INT := GREATEST(COALESCE(get_config_int('memory.recmem_gc_task_retention_days'), 14), 1);
    archived_count INT := 0;
    redacted_source_count INT := 0;
    deleted_task_count INT := 0;
BEGIN
    IF NOT COALESCE(get_config_bool('memory.recmem_gc_enabled'), TRUE) THEN
        RETURN jsonb_build_object('skipped', true, 'reason', 'disabled');
    END IF;

    WITH candidates AS (
        SELECT
            u.id,
            CASE
                WHEN u.route_status IN ('merged','episode_created') THEN 'consolidated'
                ELSE 'idle_raw'
            END AS reason
        FROM subconscious_units u
        WHERE u.status = 'active'
          -- Pinned desk material is actively needed: idle GC never takes it.
          AND u.pinned_at IS NULL
          AND NOT EXISTS (
              SELECT 1
              FROM recmem_consolidation_tasks t
              WHERE t.status IN ('pending','in_progress')
                AND (t.trigger_unit_id = u.id OR u.id = ANY(t.source_unit_ids))
          )
          AND (
              (
                  u.route_status IN ('merged','episode_created')
                  AND u.consolidated_at IS NOT NULL
                  AND GREATEST(COALESCE(u.last_accessed, '-infinity'::timestamptz), u.consolidated_at)
                      < CURRENT_TIMESTAMP - (consolidated_grace_days * INTERVAL '1 day')
              )
              OR (
                  u.route_status IN ('raw_only','route_failed')
                  AND u.extraction_status IN ('extracted','skipped','failed')
                  AND COALESCE(u.last_routed_at, u.updated_at, u.created_at)
                      < CURRENT_TIMESTAMP - (idle_days * INTERVAL '1 day')
                  AND COALESCE(u.last_accessed, u.created_at)
                      < CURRENT_TIMESTAMP - (idle_days * INTERVAL '1 day')
              )
              OR (
                  u.embedding_status = 'failed'
                  AND u.extraction_status IN ('extracted','skipped','failed')
                  AND COALESCE(u.last_accessed, u.created_at)
                      < CURRENT_TIMESTAMP - (idle_days * INTERVAL '1 day')
              )
          )
        ORDER BY COALESCE(u.last_accessed, u.consolidated_at, u.last_routed_at, u.created_at), u.id
        LIMIT gc_limit
        FOR UPDATE SKIP LOCKED
    ),
    archived AS (
        UPDATE subconscious_units u
        SET status = 'archived',
            metadata = COALESCE(u.metadata, '{}'::jsonb)
                || jsonb_build_object(
                    'recmem',
                    COALESCE(u.metadata->'recmem', '{}'::jsonb)
                        || jsonb_build_object(
                            'gc',
                            jsonb_build_object(
                                'archived_at', CURRENT_TIMESTAMP,
                                'reason', c.reason
                            )
                        )
                ),
            updated_at = CURRENT_TIMESTAMP
        FROM candidates c
        WHERE u.id = c.id
        RETURNING 1
    )
    SELECT COUNT(*) INTO archived_count FROM archived;

    -- Privacy sweep: desk material whose source document was redacted or
    -- archived goes immediately, regardless of idle time — and pinning does
    -- NOT protect against redaction.
    WITH redacted_candidates AS (
        SELECT u.id
        FROM subconscious_units u
        JOIN source_documents d
          ON u.metadata #>> '{recmem,document_id}' ~ '^[0-9a-fA-F-]{36}$'
         AND d.id = (u.metadata #>> '{recmem,document_id}')::uuid
        WHERE u.status = 'active'
          AND u.metadata #>> '{recmem,kind}' = 'source_document_desk'
          AND d.status IN ('redacted', 'archived')
        ORDER BY u.id
        LIMIT gc_limit
        FOR UPDATE OF u SKIP LOCKED
    ),
    redacted_archived AS (
        UPDATE subconscious_units u
        SET status = 'archived',
            pinned_at = NULL,
            pinned_by = NULL,
            metadata = COALESCE(u.metadata, '{}'::jsonb)
                || jsonb_build_object(
                    'recmem',
                    COALESCE(u.metadata->'recmem', '{}'::jsonb)
                        || jsonb_build_object(
                            'gc',
                            jsonb_build_object(
                                'archived_at', CURRENT_TIMESTAMP,
                                'reason', 'source_redacted'
                            )
                        )
                ),
            updated_at = CURRENT_TIMESTAMP
        FROM redacted_candidates c
        WHERE u.id = c.id
        RETURNING 1
    )
    SELECT COUNT(*) INTO redacted_source_count FROM redacted_archived;

    WITH task_candidates AS (
        SELECT id
        FROM recmem_consolidation_tasks
        WHERE status IN ('completed','dropped')
          AND COALESCE(completed_at, updated_at, created_at)
              < CURRENT_TIMESTAMP - (task_retention_days * INTERVAL '1 day')
        ORDER BY COALESCE(completed_at, updated_at, created_at), id
        LIMIT gc_limit
        FOR UPDATE SKIP LOCKED
    ),
    deleted AS (
        DELETE FROM recmem_consolidation_tasks t
        USING task_candidates c
        WHERE t.id = c.id
        RETURNING 1
    )
    SELECT COUNT(*) INTO deleted_task_count FROM deleted;

    RETURN jsonb_build_object(
        'archived_units', archived_count,
        'redacted_source_units', redacted_source_count,
        'deleted_tasks', deleted_task_count,
        'idle_days', idle_days,
        'consolidated_grace_days', consolidated_grace_days,
        'task_retention_days', task_retention_days
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION recmem_periodic_sweep(
    p_limit INT DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    sweep_limit INT := COALESCE(p_limit, get_config_int('memory.recmem_sweep_batch_size'), 100);
    min_age_days INT := COALESCE(get_config_int('memory.recmem_sweep_min_rerouting_age_days'), 7);
    unit_id UUID;
    processed INT := 0;
    gc_result JSONB;
BEGIN
    FOR unit_id IN
        SELECT id
        FROM subconscious_units
        WHERE status = 'active'
          AND embedding_status = 'embedded'
          AND route_status = 'raw_only'
          AND consolidated_at IS NULL
          AND (last_routed_at IS NULL OR last_routed_at < CURRENT_TIMESTAMP - (min_age_days * INTERVAL '1 day'))
        ORDER BY created_at
        LIMIT sweep_limit
    LOOP
        UPDATE subconscious_units
        SET route_status = 'routing',
            last_routed_at = CURRENT_TIMESTAMP,
            route_attempts = route_attempts + 1,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = unit_id;
        PERFORM recmem_route_unit(unit_id);
        processed := processed + 1;
    END LOOP;

    gc_result := recmem_gc();
    RETURN jsonb_build_object('processed', processed, 'gc', gc_result);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION should_run_recmem_sweep()
RETURNS BOOLEAN AS $$
DECLARE
    state_doc JSONB := COALESCE(get_state('recmem_state'), '{}'::jsonb);
    last_run TIMESTAMPTZ := NULLIF(state_doc->>'last_sweep_at', '')::timestamptz;
    interval_seconds FLOAT := COALESCE(get_config_float('memory.recmem_sweep_interval_seconds'), 86400);
BEGIN
    IF last_run IS NULL THEN
        RETURN TRUE;
    END IF;

    RETURN CURRENT_TIMESTAMP >= last_run + (interval_seconds || ' seconds')::interval;
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION mark_recmem_sweep_run(
    p_result JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB AS $$
DECLARE
    merged JSONB;
BEGIN
    merged := COALESCE(get_state('recmem_state'), '{}'::jsonb)
        || jsonb_build_object(
            'last_sweep_at', CURRENT_TIMESTAMP,
            'last_sweep_result', COALESCE(p_result, '{}'::jsonb)
        );

    PERFORM set_state('recmem_state', merged);
    RETURN merged;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION recmem_unhealthy_items()
RETURNS TABLE (
    kind TEXT,
    item_id UUID,
    item_status TEXT,
    attempts INT,
    last_seen TIMESTAMPTZ,
    error TEXT,
    extra JSONB
) AS $$
    SELECT
        'embedding'::text,
        id,
        embedding_status,
        embedding_attempts,
        COALESCE(embedding_claimed_at, updated_at, created_at),
        metadata#>>'{recmem,embedding_error,error}',
        metadata
    FROM subconscious_units
    WHERE embedding_status = 'failed'
    UNION ALL
    SELECT
        'routing'::text,
        id,
        route_status,
        route_attempts,
        COALESCE(last_routed_at, updated_at, created_at),
        route_result->>'error',
        route_result
    FROM subconscious_units
    WHERE route_status = 'route_failed'
    UNION ALL
    SELECT
        'task'::text,
        id,
        status,
        attempts,
        COALESCE(completed_at, started_at, updated_at, created_at),
        error,
        task_payload
    FROM recmem_consolidation_tasks
    WHERE status = 'failed'
    UNION ALL
    SELECT
        'task'::text,
        id,
        status,
        attempts,
        COALESCE(completed_at, started_at, updated_at, created_at),
        dropped_reason,
        task_payload
    FROM recmem_consolidation_tasks
    WHERE status = 'dropped';
$$ LANGUAGE sql STABLE;

CREATE OR REPLACE FUNCTION recmem_redact_unit(
    p_unit_id UUID,
    p_reason TEXT DEFAULT NULL,
    p_cascade_invalidate BOOLEAN DEFAULT TRUE
) RETURNS JSONB AS $$
DECLARE
    invalidated_ids UUID[] := ARRAY[]::UUID[];
BEGIN
    UPDATE subconscious_units
    SET status = 'redacted',
        metadata = jsonb_set(
            COALESCE(metadata, '{}'::jsonb),
            '{redaction}',
            jsonb_build_object('reason', p_reason, 'at', CURRENT_TIMESTAMP),
            true
        ),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_unit_id;

    IF COALESCE(p_cascade_invalidate, TRUE) THEN
        WITH linked AS (
            SELECT DISTINCT memory_id
            FROM memory_source_units
            WHERE subconscious_unit_id = p_unit_id
        ),
        updated AS (
            UPDATE memories m
            SET valid_until = CURRENT_TIMESTAMP,
                metadata = COALESCE(m.metadata, '{}'::jsonb)
                    || jsonb_build_object(
                        'recmem',
                        COALESCE(m.metadata->'recmem', '{}'::jsonb)
                            || jsonb_build_object(
                                'invalidation',
                                jsonb_build_object(
                                    'reason', 'source_redacted',
                                    'source_unit_id', p_unit_id,
                                    'detail', p_reason,
                                    'at', CURRENT_TIMESTAMP
                                )
                            )
                    ),
                updated_at = CURRENT_TIMESTAMP
            FROM linked l
            WHERE m.id = l.memory_id
            RETURNING m.id
        )
        SELECT COALESCE(array_agg(id), '{}'::uuid[])
        INTO invalidated_ids
        FROM updated;
    END IF;

    RETURN jsonb_build_object('redacted_unit_id', p_unit_id, 'invalidated_memory_ids', invalidated_ids);
END;
$$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION has_pending_recmem_consolidation()
RETURNS BOOLEAN AS $$
    SELECT EXISTS (
        SELECT 1
        FROM recmem_consolidation_tasks
        WHERE status = 'pending'
          AND next_attempt_at <= CURRENT_TIMESTAMP
    );
$$ LANGUAGE sql STABLE;
