-- HMX Slice 6: accepted-memory re-embedding and raw-unit routing.
SET search_path = public, ag_catalog, "$user";

INSERT INTO config_defaults (key, value, description) VALUES
    ('memory.hmx_reembed_batch_size', '16'::jsonb,
     'Accepted HMX memories embedded per maintenance tick'),
    ('memory.hmx_reembed_claim_timeout_s', '300'::jsonb,
     'Seconds before an interrupted HMX embedding claim can be recovered'),
    ('memory.hmx_reembed_max_attempts', '3'::jsonb,
     'Maximum HMX embedding attempts before the import is marked failed')
ON CONFLICT (key) DO NOTHING;

CREATE OR REPLACE FUNCTION hmx_is_imported_memory(p_metadata JSONB)
RETURNS BOOLEAN AS $$
    SELECT jsonb_typeof(COALESCE(p_metadata#>'{provenance,import_chain}', '[]'::jsonb)) = 'array'
       AND jsonb_array_length(COALESCE(p_metadata#>'{provenance,import_chain}', '[]'::jsonb)) > 0;
$$ LANGUAGE sql IMMUTABLE;

CREATE OR REPLACE FUNCTION hmx_queue_reembed(p_memory_ids UUID[])
RETURNS JSONB AS $$
DECLARE
    queued_ids UUID[] := '{}'::uuid[];
    queued_count INT := 0;
    skipped_count INT := 0;
BEGIN
    IF COALESCE(cardinality(p_memory_ids), 0) = 0 THEN
        RETURN jsonb_build_object('queued', 0, 'skipped', 0, 'memory_ids', '[]'::jsonb);
    END IF;

    WITH updated AS (
        UPDATE memories m
        SET embedding = array_fill(0.0::float, ARRAY[embedding_dimension()])::vector,
            metadata = (COALESCE(m.metadata, '{}'::jsonb)
                - 'embedding_error'
                - 'embedding_claimed_at'
                - 'embedding_completed_at')
                || jsonb_build_object(
                    'embedding_status', 'pending_import',
                    'embedding_attempts', 0,
                    'embedding_queued_at', CURRENT_TIMESTAMP
                ),
            updated_at = CURRENT_TIMESTAMP
        WHERE m.id = ANY(p_memory_ids)
          AND m.status = 'active'
          AND hmx_is_imported_memory(m.metadata)
        RETURNING m.id
    )
    SELECT COALESCE(array_agg(id ORDER BY id), '{}'::uuid[])
    INTO queued_ids
    FROM updated;

    queued_count := cardinality(queued_ids);

    UPDATE memories m
    SET metadata = (COALESCE(m.metadata, '{}'::jsonb) - 'embedding_claimed_at')
        || jsonb_build_object('embedding_status', 'not_applicable'),
        updated_at = CURRENT_TIMESTAMP
    WHERE m.id = ANY(p_memory_ids)
      AND m.status <> 'active'
      AND hmx_is_imported_memory(m.metadata);

    skipped_count := cardinality(p_memory_ids) - queued_count;
    RETURN jsonb_build_object(
        'queued', queued_count,
        'skipped', GREATEST(skipped_count, 0),
        'memory_ids', to_jsonb(queued_ids)
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION hmx_claim_reembed_batch(
    p_limit INT DEFAULT NULL,
    p_claim_timeout_s INT DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    batch_size INT := COALESCE(
        p_limit,
        get_config_int('memory.hmx_reembed_batch_size'),
        16
    );
    timeout_s INT := COALESCE(
        p_claim_timeout_s,
        get_config_int('memory.hmx_reembed_claim_timeout_s'),
        300
    );
    payload JSONB;
BEGIN
    WITH candidate AS (
        SELECT m.id
        FROM memories m
        WHERE m.status = 'active'
          AND hmx_is_imported_memory(m.metadata)
          AND (
              m.metadata->>'embedding_status' = 'pending_import'
              OR (
                  m.metadata->>'embedding_status' = 'in_progress'
                  AND COALESCE(
                      NULLIF(m.metadata->>'embedding_claimed_at', '')::timestamptz,
                      '-infinity'::timestamptz
                  ) < CURRENT_TIMESTAMP - (GREATEST(timeout_s, 1) * INTERVAL '1 second')
              )
          )
        ORDER BY m.created_at, m.id
        FOR UPDATE SKIP LOCKED
        LIMIT GREATEST(batch_size, 1)
    ),
    claimed AS (
        UPDATE memories m
        SET metadata = COALESCE(m.metadata, '{}'::jsonb)
            || jsonb_build_object(
                'embedding_status', 'in_progress',
                'embedding_claimed_at', CURRENT_TIMESTAMP,
                'embedding_attempts', COALESCE((m.metadata->>'embedding_attempts')::int, 0) + 1
            ),
            updated_at = CURRENT_TIMESTAMP
        FROM candidate c
        WHERE m.id = c.id
        RETURNING m.id, m.content, (m.metadata->>'embedding_attempts')::int AS attempts
    )
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'memory_id', id,
        'content', content,
        'attempts', attempts
    ) ORDER BY id), '[]'::jsonb)
    INTO payload
    FROM claimed;

    RETURN payload;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION hmx_refresh_reembed_derivatives(p_memory_ids UUID[])
RETURNS JSONB AS $$
DECLARE
    valid_ids UUID[];
    current_memory_id UUID;
    cluster_id UUID;
    neighborhood_count INT := 0;
    cluster_ids UUID[] := '{}'::uuid[];
BEGIN
    SELECT COALESCE(array_agg(m.id ORDER BY m.id), '{}'::uuid[])
    INTO valid_ids
    FROM memories m
    WHERE m.id = ANY(COALESCE(p_memory_ids, '{}'::uuid[]))
      AND m.status = 'active'
      AND m.metadata->>'embedding_status' = 'embedded';

    IF cardinality(valid_ids) = 0 THEN
        RETURN jsonb_build_object('neighborhoods_recomputed', 0, 'clusters_recomputed', 0);
    END IF;

    UPDATE memory_neighborhoods mn
    SET is_stale = TRUE
    WHERE NOT (mn.memory_id = ANY(valid_ids));

    FOREACH current_memory_id IN ARRAY valid_ids LOOP
        INSERT INTO memory_neighborhoods (memory_id, is_stale)
        VALUES (current_memory_id, TRUE)
        ON CONFLICT ON CONSTRAINT memory_neighborhoods_pkey
        DO UPDATE SET is_stale = TRUE;
        PERFORM recompute_neighborhood(current_memory_id);
        neighborhood_count := neighborhood_count + 1;
    END LOOP;

    SELECT COALESCE(array_agg(DISTINCT c.id ORDER BY c.id), '{}'::uuid[])
    INTO cluster_ids
    FROM memory_edges e
    JOIN clusters c ON c.id::text = e.dst_id
    WHERE e.src_type = 'memory'
      AND e.src_id = ANY(ARRAY(SELECT id::text FROM unnest(valid_ids) AS id))
      AND e.rel_type = 'MEMBER_OF'
      AND e.dst_type = 'cluster';

    FOREACH cluster_id IN ARRAY cluster_ids LOOP
        PERFORM recalculate_cluster_centroid(cluster_id);
    END LOOP;

    FOREACH current_memory_id IN ARRAY valid_ids LOOP
        PERFORM assign_memory_to_clusters(current_memory_id);
    END LOOP;

    SELECT COALESCE(array_agg(DISTINCT c.id ORDER BY c.id), '{}'::uuid[])
    INTO cluster_ids
    FROM memory_edges e
    JOIN clusters c ON c.id::text = e.dst_id
    WHERE e.src_type = 'memory'
      AND e.src_id = ANY(ARRAY(SELECT id::text FROM unnest(valid_ids) AS id))
      AND e.rel_type = 'MEMBER_OF'
      AND e.dst_type = 'cluster';

    FOREACH cluster_id IN ARRAY cluster_ids LOOP
        PERFORM recalculate_cluster_centroid(cluster_id);
    END LOOP;

    RETURN jsonb_build_object(
        'neighborhoods_recomputed', neighborhood_count,
        'clusters_recomputed', cardinality(cluster_ids)
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION hmx_apply_reembed_batch(p_memory_ids UUID[])
RETURNS JSONB AS $$
DECLARE
    valid_ids UUID[];
    contents TEXT[];
    embeddings vector[];
    v_embedding_model TEXT;
    derivative_result JSONB;
    index INT;
    updated_count INT := 0;
    affected INT;
BEGIN
    SELECT array_agg(m.id ORDER BY requested.ordinality),
           array_agg(m.content ORDER BY requested.ordinality)
    INTO valid_ids, contents
    FROM unnest(COALESCE(p_memory_ids, '{}'::uuid[])) WITH ORDINALITY AS requested(id, ordinality)
    JOIN memories m ON m.id = requested.id
    WHERE m.status = 'active'
      AND m.metadata->>'embedding_status' = 'in_progress'
      AND hmx_is_imported_memory(m.metadata);

    IF COALESCE(cardinality(valid_ids), 0) = 0 THEN
        RETURN jsonb_build_object('embedded', 0, 'memory_ids', '[]'::jsonb);
    END IF;

    embeddings := get_embedding(contents);
    IF cardinality(embeddings) <> cardinality(valid_ids) THEN
        RAISE EXCEPTION 'HMX embedding response size mismatch: expected %, got %',
            cardinality(valid_ids), cardinality(embeddings);
    END IF;

    v_embedding_model := COALESCE(
        (SELECT value #>> '{}' FROM config WHERE key = 'embedding.model_id'),
        'unknown'
    );

    FOR index IN 1..cardinality(valid_ids) LOOP
        UPDATE memories m
        SET embedding = embeddings[index],
            metadata = (COALESCE(m.metadata, '{}'::jsonb)
                - 'embedding_error'
                - 'embedding_claimed_at')
                || jsonb_build_object(
                    'embedding_status', 'embedded',
                    'embedding_completed_at', CURRENT_TIMESTAMP,
                    'embedding_model', v_embedding_model
                ),
            updated_at = CURRENT_TIMESTAMP
        WHERE m.id = valid_ids[index]
          AND m.status = 'active'
          AND m.metadata->>'embedding_status' = 'in_progress';
        GET DIAGNOSTICS affected = ROW_COUNT;
        updated_count := updated_count + affected;
    END LOOP;

    derivative_result := hmx_refresh_reembed_derivatives(valid_ids);
    RETURN jsonb_build_object(
        'embedded', updated_count,
        'memory_ids', to_jsonb(valid_ids),
        'derivatives', derivative_result
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION hmx_fail_reembed(p_memory_id UUID, p_error TEXT DEFAULT NULL)
RETURNS JSONB AS $$
DECLARE
    max_attempts INT := COALESCE(get_config_int('memory.hmx_reembed_max_attempts'), 3);
    final_status TEXT;
BEGIN
    UPDATE memories m
    SET metadata = (COALESCE(m.metadata, '{}'::jsonb) - 'embedding_claimed_at')
        || jsonb_build_object(
            'embedding_status', CASE
                WHEN COALESCE((m.metadata->>'embedding_attempts')::int, 0) >= max_attempts
                    THEN 'failed_import'
                ELSE 'pending_import'
            END,
            'embedding_error', jsonb_build_object(
                'message', left(COALESCE(p_error, 'unknown embedding failure'), 1000),
                'at', CURRENT_TIMESTAMP
            )
        ),
        updated_at = CURRENT_TIMESTAMP
    WHERE m.id = p_memory_id
      AND m.metadata->>'embedding_status' = 'in_progress'
    RETURNING m.metadata->>'embedding_status' INTO final_status;

    RETURN jsonb_build_object(
        'memory_id', p_memory_id,
        'embedding_status', final_status
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION hmx_import_raw_units(
    p_records JSONB,
    p_export_id TEXT,
    p_ref_map JSONB
) RETURNS JSONB AS $$
DECLARE
    item JSONB;
    derived_ref JSONB;
    ingest_result JSONB;
    unit_id UUID;
    mapped_memory_id UUID;
    source_ref TEXT;
    source_identity TEXT;
    inserted_count INT := 0;
    duplicate_count INT := 0;
    warnings JSONB := '[]'::jsonb;
    ref_map JSONB := '{}'::jsonb;
BEGIN
    FOR item IN SELECT value FROM jsonb_array_elements(COALESCE(p_records, '[]'::jsonb))
    LOOP
        source_ref := item->>'ref';
        source_identity := 'import:' || p_export_id || ':' || COALESCE(
            NULLIF(item->>'source_identity', ''),
            source_ref,
            'raw-unit'
        );
        ingest_result := recmem_ingest_turn(
            COALESCE(item->>'user_text', ''),
            COALESCE(item->>'assistant_text', ''),
            NULL,
            source_identity,
            COALESCE(NULLIF(item->>'turn_at', '')::timestamptz, CURRENT_TIMESTAMP),
            COALESCE((item->>'importance')::float, 0.3),
            '{}'::jsonb,
            jsonb_build_object(
                'hmx', jsonb_build_object(
                    'export_id', p_export_id,
                    'source_ref', source_ref,
                    'source_identity', item->>'source_identity',
                    'source_idempotency_key', item->>'idempotency_key',
                    'source_route_status', item->>'route_status'
                )
            )
        );

        IF ingest_result->>'status' = 'empty' THEN
            warnings := warnings || jsonb_build_array(jsonb_build_object(
                'code', 'empty_raw_unit', 'section', 'raw_units', 'ref', source_ref
            ));
            CONTINUE;
        END IF;

        unit_id := (ingest_result->>'unit_id')::uuid;
        ref_map := ref_map || jsonb_build_object(source_ref, unit_id::text);
        IF ingest_result->>'status' = 'stored' THEN
            inserted_count := inserted_count + 1;
        ELSE
            duplicate_count := duplicate_count + 1;
        END IF;

        FOR derived_ref IN
            SELECT value FROM jsonb_array_elements(COALESCE(item->'derived_memory_refs', '[]'::jsonb))
        LOOP
            BEGIN
                mapped_memory_id := (p_ref_map->>trim(both '"' from derived_ref::text))::uuid;
            EXCEPTION WHEN OTHERS THEN
                mapped_memory_id := NULL;
            END;
            IF mapped_memory_id IS NULL THEN
                warnings := warnings || jsonb_build_array(jsonb_build_object(
                    'code', 'orphaned_reference',
                    'section', 'raw_units',
                    'ref', trim(both '"' from derived_ref::text)
                ));
            ELSE
                INSERT INTO memory_source_units (memory_id, subconscious_unit_id)
                VALUES (mapped_memory_id, unit_id)
                ON CONFLICT DO NOTHING;
            END IF;
        END LOOP;
    END LOOP;

    RETURN jsonb_build_object(
        'inserted', inserted_count,
        'duplicates', duplicate_count,
        'ref_map', ref_map,
        'warnings', warnings
    );
END;
$$ LANGUAGE plpgsql;
