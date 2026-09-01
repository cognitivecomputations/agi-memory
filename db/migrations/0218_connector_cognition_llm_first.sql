-- Phase 5: make connector cognition LLM-authoritative with a content cache.
SET search_path = public, ag_catalog, "$user";

INSERT INTO config_defaults (key, value, description) VALUES
    ('connector.user_model_synthesis_mode', '"llm"'::jsonb,
     'User-model synthesis mode; llm is authoritative, while rules is retained only as an explicit LLM-disabled fallback')
ON CONFLICT (key) DO UPDATE SET
    value = EXCLUDED.value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

CREATE TABLE IF NOT EXISTS connector_cognition_cache (
    task TEXT NOT NULL CHECK (task IN ('user_model_claims', 'item_importance')),
    content_hash TEXT NOT NULL,
    detector_version TEXT NOT NULL,
    result JSONB NOT NULL,
    provenance TEXT NOT NULL DEFAULT 'llm' CHECK (provenance = 'llm'),
    provider TEXT,
    model TEXT,
    hit_count BIGINT NOT NULL DEFAULT 0 CHECK (hit_count >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (task, content_hash, detector_version)
);

CREATE INDEX IF NOT EXISTS idx_connector_cognition_cache_last_used
    ON connector_cognition_cache (last_used_at DESC);

CREATE OR REPLACE FUNCTION claim_user_model_source_items(
    p_limit INT DEFAULT NULL,
    p_claim_timeout_s INT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    lim INT := GREATEST(COALESCE(p_limit, get_config_int('connector.user_model_synthesis_batch_size'), 10), 1);
    timeout_s INT := COALESCE(p_claim_timeout_s, get_config_int('connector.user_model_synthesis_claim_timeout_s'), 600);
    candidate RECORD;
    row_progress user_model_source_progress%ROWTYPE;
    item JSONB;
    result JSONB := '[]'::jsonb;
BEGIN
    FOR candidate IN
        SELECT csi.id
        FROM connector_source_items csi
        JOIN source_documents d ON d.id = csi.source_document_id AND d.status = 'active'
        LEFT JOIN user_model_source_progress p ON p.source_item_id = csi.id
        WHERE csi.status = 'active'
          AND csi.sensitivity IN ('private', 'shared')
          AND (
                p.source_item_id IS NULL
             OR p.status = 'pending'
             OR (p.status = 'failed' AND p.attempts < 3)
             OR (p.status = 'in_progress'
                 AND p.claimed_at < CURRENT_TIMESTAMP - make_interval(secs => timeout_s))
          )
        ORDER BY COALESCE(csi.item_timestamp, csi.created_at), csi.id
        LIMIT lim
        FOR UPDATE OF csi SKIP LOCKED
    LOOP
        INSERT INTO user_model_source_progress (source_item_id, status, attempts, claimed_at, last_error)
        VALUES (candidate.id, 'in_progress', 1, CURRENT_TIMESTAMP, NULL)
        ON CONFLICT (source_item_id) DO UPDATE SET
            status = 'in_progress',
            attempts = user_model_source_progress.attempts + 1,
            claimed_at = CURRENT_TIMESTAMP,
            last_error = NULL,
            updated_at = CURRENT_TIMESTAMP
        RETURNING * INTO row_progress;

        SELECT jsonb_build_object(
            'source_item_id', csi.id::text,
            'connector_id', csi.connector_id,
            'account_key', csi.account_key,
            'provider_item_id', csi.provider_item_id,
            'provider_thread_id', csi.provider_thread_id,
            'source_document_id', d.id::text,
            'content_hash', csi.content_hash,
            'title', d.title,
            'path', d.path,
            'content', d.content,
            'sensitivity', csi.sensitivity,
            'item_timestamp', csi.item_timestamp,
            'attempts', row_progress.attempts
        )
        INTO item
        FROM connector_source_items csi
        JOIN source_documents d ON d.id = csi.source_document_id
        WHERE csi.id = candidate.id;

        result := result || jsonb_build_array(item);
    END LOOP;

    RETURN result;
END;
$$;

CREATE OR REPLACE FUNCTION claim_connector_importance_items(
    p_limit INT DEFAULT NULL,
    p_claim_timeout_s INT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    lim INT := GREATEST(COALESCE(p_limit, get_config_int('connector.importance_detection_batch_size'), 20), 1);
    timeout_s INT := COALESCE(p_claim_timeout_s, get_config_int('connector.importance_detection_claim_timeout_s'), 600);
    candidate RECORD;
    row_importance connector_item_importance%ROWTYPE;
    item JSONB;
    result JSONB := '[]'::jsonb;
BEGIN
    FOR candidate IN
        SELECT csi.id
        FROM connector_source_items csi
        JOIN source_documents d ON d.id = csi.source_document_id AND d.status = 'active'
        LEFT JOIN connector_item_importance i ON i.source_item_id = csi.id
        WHERE csi.status = 'active'
          AND (
                i.source_item_id IS NULL
             OR i.status = 'pending'
             OR (i.status = 'failed' AND i.attempts < 3)
             OR (i.status = 'in_progress'
                 AND i.claimed_at < CURRENT_TIMESTAMP - make_interval(secs => timeout_s))
          )
        ORDER BY COALESCE(csi.item_timestamp, csi.created_at), csi.id
        LIMIT lim
        FOR UPDATE OF csi SKIP LOCKED
    LOOP
        INSERT INTO connector_item_importance (
            source_item_id, connector_id, account_key, source_document_id,
            status, attempts, claimed_at, last_error
        )
        SELECT csi.id, csi.connector_id, csi.account_key, csi.source_document_id,
               'in_progress', 1, CURRENT_TIMESTAMP, NULL
        FROM connector_source_items csi
        WHERE csi.id = candidate.id
        ON CONFLICT (source_item_id) DO UPDATE SET
            status = 'in_progress',
            attempts = connector_item_importance.attempts + 1,
            claimed_at = CURRENT_TIMESTAMP,
            last_error = NULL,
            updated_at = CURRENT_TIMESTAMP
        RETURNING * INTO row_importance;

        SELECT jsonb_build_object(
            'source_item_id', csi.id::text,
            'connector_id', csi.connector_id,
            'account_key', csi.account_key,
            'provider_item_id', csi.provider_item_id,
            'source_document_id', d.id::text,
            'content_hash', csi.content_hash,
            'title', d.title,
            'path', d.path,
            'content', d.content,
            'sensitivity', csi.sensitivity,
            'item_timestamp', csi.item_timestamp,
            'attempts', row_importance.attempts
        )
        INTO item
        FROM connector_source_items csi
        JOIN source_documents d ON d.id = csi.source_document_id
        WHERE csi.id = candidate.id;

        result := result || jsonb_build_array(item);
    END LOOP;

    RETURN result;
END;
$$;
