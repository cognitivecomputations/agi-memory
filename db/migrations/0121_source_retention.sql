-- 0121: Ownership-based source retention.
-- User-provided sources never auto-fade: approval now cascades to the
-- filing cabinet (doc archived, chunks removed, artifact bytes released).
-- Agent-acquired sources (acquisition='agent') gain a daily autonomous
-- pass: archive when idle, escalate to a user fade request when heavily
-- referenced. All gated on retention.enabled (enabled by default; still a kill switch).
SET search_path = public, ag_catalog, "$user";

INSERT INTO config_defaults (key, value, description) VALUES
    ('ingest.upload_max_bytes', '104857600'::jsonb,
     'Maximum file size accepted by the upload API; larger files use the synchronous CLI path')
ON CONFLICT (key) DO NOTHING;

-- ============================================================================
-- Agent-acquired source retention (ownership-based two-tier policy).
--
-- User-provided sources (CLI, uploads, chat drops, connectors) are the
-- USER's data: they never auto-fade — only the fade-request → approval flow
-- above touches them. Sources the AGENT chose to keep on its own
-- (source_attribution->>'acquisition' = 'agent': url_ingest/git_ingest from
-- a heartbeat) are the agent's working set, so the daily subconscious pass
-- may archive them autonomously once truly idle — reversibly (doc archived,
-- chunks kept, artifact bytes kept). Escalates to a user fade request
-- instead when the source is heavily referenced by memories.
-- ============================================================================

INSERT INTO config_defaults (key, value, description) VALUES
    ('retention.agent_source_idle_days', '60'::jsonb,
     'Archive agent-acquired sources untouched (chunks, desk, memories) for this many days'),
    ('retention.agent_source_escalate_memories', '5'::jsonb,
     'Agent-acquired sources cited by at least this many active memories escalate to a user fade request instead of auto-archiving'),
    ('retention.agent_source_batch', '5'::jsonb,
     'Agent-acquired sources processed per daily retention pass')
ON CONFLICT (key) DO NOTHING;

CREATE OR REPLACE FUNCTION run_agent_source_retention()
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_idle_days FLOAT := GREATEST(COALESCE(get_config_float('retention.agent_source_idle_days'), 60), 1);
    v_escalate INT := GREATEST(COALESCE(get_config_int('retention.agent_source_escalate_memories'), 5), 1);
    v_batch INT := GREATEST(COALESCE(get_config_int('retention.agent_source_batch'), 5), 1);
    v_archived INT := 0;
    v_escalated INT := 0;
    rec RECORD;
    v_memory_count INT;
BEGIN
    IF NOT COALESCE(get_config_bool('retention.enabled'), false) THEN
        RETURN jsonb_build_object('skipped', true);
    END IF;

    FOR rec IN
        SELECT d.id, d.content_hash, d.title
        FROM source_documents d
        WHERE d.status = 'active'
          AND d.source_attribution->>'acquisition' = 'agent'
          AND age_in_days(d.last_ingested_at) >= v_idle_days
          -- no recently-touched chunks
          AND NOT EXISTS (
              SELECT 1 FROM source_document_chunks c
              WHERE c.source_document_id = d.id
                AND c.last_accessed IS NOT NULL
                AND age_in_days(c.last_accessed) < v_idle_days
          )
          -- nothing of it still on the active desk
          AND NOT EXISTS (
              SELECT 1 FROM subconscious_units u
              WHERE u.status = 'active'
                AND u.metadata #>> '{recmem,kind}' = 'source_document_desk'
                AND u.metadata #>> '{recmem,document_id}' = d.id::text
          )
          -- no recently-reinforced memories citing it
          AND NOT EXISTS (
              SELECT 1 FROM memories m
              WHERE m.status = 'active'
                AND m.source_attribution->>'content_hash' = d.content_hash
                AND age_in_days(GREATEST(m.last_reinforced, m.last_accessed, m.created_at)) < v_idle_days
          )
          AND NOT EXISTS (
              SELECT 1 FROM document_fade_requests r
              WHERE r.content_hash = d.content_hash AND r.status = 'pending'
          )
        ORDER BY d.last_ingested_at
        LIMIT v_batch
    LOOP
        SELECT count(*) INTO v_memory_count
        FROM memories m
        WHERE m.status = 'active'
          AND m.source_attribution->>'content_hash' = rec.content_hash;

        IF v_memory_count >= v_escalate THEN
            -- Heavily referenced: this rose to user-attention level.
            INSERT INTO document_fade_requests (content_hash, label, memory_count)
            VALUES (rec.content_hash, rec.title, v_memory_count)
            ON CONFLICT (content_hash) DO NOTHING;
            IF FOUND THEN
                PERFORM queue_outbox_message(
                    'A while back I fetched "' || COALESCE(rec.title, 'a web source')
                    || '" on my own and built ' || v_memory_count || ' memories from it, '
                    || 'but I haven''t drawn on it lately. Want me to let it fade, or keep it?',
                    'document_fade', 'retention');
                v_escalated := v_escalated + 1;
            END IF;
        ELSE
            -- Low-stakes: archive reversibly (chunks and artifact bytes kept;
            -- re-ingesting or un-archiving restores full retrieval).
            UPDATE source_documents
            SET status = 'archived',
                metadata = metadata || jsonb_build_object(
                    'retention', jsonb_build_object(
                        'archived_at', CURRENT_TIMESTAMP,
                        'reason', 'agent_source_idle')),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = rec.id AND status = 'active';
            v_archived := v_archived + 1;
        END IF;
    END LOOP;

    RETURN jsonb_build_object('archived', v_archived, 'escalated', v_escalated);
END;
$$;

-- Approval cascade: archive the source document, drop chunks, release bytes.
CREATE OR REPLACE FUNCTION resolve_document_fade(p_ref TEXT, p_decision TEXT)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_hash  TEXT;
    v_label TEXT;
    v_ids   UUID[];
    v_affected INT := 0;
    v_id UUID;
BEGIN
    IF p_ref IS NULL OR btrim(p_ref) = '' THEN
        RETURN jsonb_build_object('error', 'no document reference given');
    END IF;
    SELECT content_hash, label INTO v_hash, v_label
    FROM document_fade_requests
    WHERE status = 'pending'
      AND (content_hash = p_ref OR label ILIKE p_ref OR label ILIKE '%' || p_ref || '%')
    ORDER BY (content_hash = p_ref) DESC, requested_at
    LIMIT 1;
    IF v_hash IS NULL THEN
        RETURN jsonb_build_object('error', 'no pending document approval matches', 'ref', p_ref);
    END IF;

    v_ids := ARRAY(SELECT id FROM memories
                   WHERE status = 'active' AND source_attribution->>'content_hash' = v_hash);

    IF lower(COALESCE(p_decision, '')) = 'approve' THEN
        FOREACH v_id IN ARRAY v_ids LOOP
            IF delete_memory_fully(v_id) THEN v_affected := v_affected + 1; END IF;
        END LOOP;
        -- Approval cascades to the filing cabinet: the source document is
        -- archived (tombstone kept, never silently deleted), its durable
        -- chunks are removed, and preserved artifact bytes are released.
        -- Desk copies are swept by recmem_gc's archived-source pass.
        DELETE FROM source_document_chunks c
        USING source_documents d
        WHERE c.source_document_id = d.id
          AND d.content_hash = v_hash;
        UPDATE source_artifacts a
        SET bytes = NULL,
            status = 'archived',
            metadata = a.metadata || jsonb_build_object(
                'retention', jsonb_build_object(
                    'bytes_released_at', CURRENT_TIMESTAMP,
                    'reason', 'document_fade_approved')),
            updated_at = CURRENT_TIMESTAMP
        FROM source_documents d
        WHERE a.source_document_id = d.id
          AND d.content_hash = v_hash
          AND a.status <> 'redacted';
        UPDATE source_documents
        SET status = 'archived',
            metadata = metadata || jsonb_build_object(
                'retention', jsonb_build_object(
                    'archived_at', CURRENT_TIMESTAMP,
                    'reason', 'document_fade_approved')),
            updated_at = CURRENT_TIMESTAMP
        WHERE content_hash = v_hash AND status = 'active';
        UPDATE document_fade_requests SET status = 'approved', decided_at = CURRENT_TIMESTAMP
         WHERE content_hash = v_hash;
        RETURN jsonb_build_object('decision', 'approve', 'label', v_label, 'faded', v_affected);
    ELSE
        IF array_length(v_ids, 1) > 0 THEN PERFORM touch_memories(v_ids); END IF;
        v_affected := COALESCE(array_length(v_ids, 1), 0);
        UPDATE document_fade_requests SET status = 'kept', decided_at = CURRENT_TIMESTAMP
         WHERE content_hash = v_hash;
        RETURN jsonb_build_object('decision', 'keep', 'label', v_label, 'kept', v_affected);
    END IF;
END;
$$;

-- run_retention_gc now also runs the agent-acquired source pass.
CREATE OR REPLACE FUNCTION run_retention_gc()
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_grace    FLOAT := COALESCE(get_config_float('retention.prune_grace_days'), 14);
    v_capacity FLOAT := COALESCE(get_config_float('retention.capacity'), 0);
    v_pruned INT := 0;
    v_expired INT := 0;
    v_mass FLOAT;
    v_target UUID;
    rec RECORD;
BEGIN
    IF NOT COALESCE(get_config_bool('retention.enabled'), false) THEN
        RETURN jsonb_build_object('skipped', true);
    END IF;

    -- (0) conscious review left undecided past its window -> default LET GO (consolidate)
    FOR rec IN
        SELECT id, memory_ids FROM memory_review_queue
        WHERE status = 'pending' AND expires_at <= CURRENT_TIMESTAMP
    LOOP
        BEGIN
            PERFORM consolidate_memory_group(rec.memory_ids);
        EXCEPTION WHEN OTHERS THEN
            RAISE WARNING 'review expiry consolidate failed: %', SQLERRM;
        END;
        UPDATE memory_review_queue SET status = 'expired', decided_at = CURRENT_TIMESTAMP WHERE id = rec.id;
        v_expired := v_expired + 1;
    END LOOP;

    -- (a) archived originals past grace (the undo window) -> truly delete
    FOR rec IN
        SELECT id FROM memories
        WHERE status = 'archived' AND superseded_by IS NOT NULL
          AND age_in_days(COALESCE((metadata->'consolidation'->>'archived_at')::timestamptz, updated_at)) >= v_grace
          AND NOT is_memory_protected(id)
    LOOP
        IF delete_memory_fully(rec.id) THEN v_pruned := v_pruned + 1; END IF;
    END LOOP;

    -- (b) capacity pressure -> prune the weakest live episodic memories (last resort)
    IF v_capacity > 0 THEN
        LOOP
            SELECT COALESCE(sum(calculate_strength(importance, decay_rate, created_at, last_reinforced)), 0)
              INTO v_mass FROM memories WHERE status = 'active' AND type = 'episodic';
            EXIT WHEN v_mass <= v_capacity;
            SELECT id INTO v_target FROM memories
            WHERE status = 'active' AND type = 'episodic' AND NOT is_memory_protected(id)
            ORDER BY calculate_strength(importance, decay_rate, created_at, last_reinforced) ASC, created_at ASC
            LIMIT 1;
            EXIT WHEN v_target IS NULL;
            EXIT WHEN NOT delete_memory_fully(v_target);
            v_pruned := v_pruned + 1;
        END LOOP;
    END IF;

    -- (c) agent-acquired sources: the ownership-based tier the agent may
    -- retire on its own (user-provided sources only fade via approval).
    RETURN jsonb_build_object(
        'pruned', v_pruned,
        'reviews_expired', v_expired,
        'agent_sources', run_agent_source_retention()
    );
END;
$$;
