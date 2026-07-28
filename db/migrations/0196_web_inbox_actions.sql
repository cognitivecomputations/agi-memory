-- 0196: Web inbox per-message actions (#107 follow-up).
--
-- The inbox becomes an actionable surface: acknowledge / delete / reply /
-- grant / deny per message. Adds the missing DB pieces:
--   1. delete_web_inbox(ids)   -- explicit dismissal, distinct from read.
--   2. get_web_inbox           -- also emits `payload`, so the UI can
--      correlate a message with its underlying request (request_id,
--      content_hash, connector source ids live in payload.delivery).
--   3. file_resource_request   -- embeds request_id in the delivery doc.
--   4. document fade asks      -- embed content_hash the same way.
-- Function bodies below are verbatim copies of the updated baselines
-- (db/76, db/74, db/47).

CREATE OR REPLACE FUNCTION delete_web_inbox(p_ids UUID[])
RETURNS INT AS $$
DECLARE
    removed INT;
BEGIN
    DELETE FROM web_inbox
    WHERE id = ANY(COALESCE(p_ids, ARRAY[]::uuid[]));
    GET DIAGNOSTICS removed = ROW_COUNT;
    RETURN removed;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_web_inbox(p_limit INT DEFAULT 30)
RETURNS JSONB AS $$
    SELECT jsonb_build_object(
        'unread', (SELECT COUNT(*) FROM web_inbox WHERE read_at IS NULL),
        'messages', COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
                'id', m.id,
                'kind', m.kind,
                'intent', m.intent,
                'message', m.message,
                'payload', m.payload,
                'delivered_at', m.delivered_at,
                'read_at', m.read_at
            ) ORDER BY m.delivered_at DESC)
            FROM (
                SELECT * FROM web_inbox
                ORDER BY delivered_at DESC
                LIMIT GREATEST(COALESCE(p_limit, 30), 1)
            ) m
        ), '[]'::jsonb)
    );
$$ LANGUAGE sql STABLE;

CREATE OR REPLACE FUNCTION mark_web_inbox_read(p_ids UUID[])
RETURNS INT AS $$
DECLARE
    updated INT;
BEGIN
    UPDATE web_inbox
    SET read_at = CURRENT_TIMESTAMP
    WHERE id = ANY(COALESCE(p_ids, ARRAY[]::uuid[])) AND read_at IS NULL;
    GET DIAGNOSTICS updated = ROW_COUNT;
    RETURN updated;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION file_resource_request(
    p_kind TEXT,
    p_rationale TEXT,
    p_target_key TEXT DEFAULT NULL,
    p_requested_value JSONB DEFAULT NULL,
    p_duration TEXT DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    new_id UUID;
    summary TEXT;
BEGIN
    IF p_kind IS NULL OR p_kind NOT IN ('energy_boost', 'config_change', 'backup', 'other') THEN
        RAISE EXCEPTION 'kind must be energy_boost, config_change, backup, or other (got %)', p_kind;
    END IF;
    IF NULLIF(btrim(COALESCE(p_rationale, '')), '') IS NULL THEN
        RAISE EXCEPTION 'a rationale is required: say what you need and why';
    END IF;
    IF p_kind = 'config_change' AND NULLIF(btrim(COALESCE(p_target_key, '')), '') IS NULL THEN
        RAISE EXCEPTION 'config_change requests require target_key';
    END IF;

    INSERT INTO resource_requests (kind, target_key, requested_value, rationale, duration)
    VALUES (p_kind, NULLIF(btrim(COALESCE(p_target_key, '')), ''), p_requested_value,
            btrim(p_rationale), NULLIF(btrim(COALESCE(p_duration, '')), ''))
    RETURNING id INTO new_id;

    summary := format('Resource request [%s] %s%s: %s',
        left(new_id::text, 8), p_kind,
        CASE WHEN p_target_key IS NOT NULL THEN ' (' || p_target_key || ')' ELSE '' END,
        btrim(p_rationale));
    BEGIN
        -- delivery doc carries the request id in machine-readable form so the
        -- dashboard inbox can offer grant/deny without parsing the prose.
        PERFORM queue_outbox_message(
            summary || E'\nDecide with: hexis requests grant/deny ' || left(new_id::text, 8),
            'resource_request', 'resource_request',
            jsonb_build_object('mode', 'web_inbox', 'request_id', new_id));
    EXCEPTION WHEN OTHERS THEN
        RAISE WARNING 'resource request % filed but outbox notification failed: %', new_id, SQLERRM;
    END;

    RETURN jsonb_build_object(
        'request_id', new_id,
        'status', 'pending',
        'note', 'The operator decides; the decision will appear in your context.'
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION request_stale_document_fades()
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_batch INT := GREATEST(0, COALESCE(get_config_int('retention.doc_request_batch'), 2));
    v_sent INT := 0;
    rec RECORD;
BEGIN
    IF NOT COALESCE(get_config_bool('retention.enabled'), false) THEN
        RETURN jsonb_build_object('skipped', true);
    END IF;
    FOR rec IN SELECT content_hash, label, memory_count
               FROM find_stale_ingested_documents() LIMIT GREATEST(v_batch, 1)
    LOOP
        EXIT WHEN v_sent >= v_batch;
        INSERT INTO document_fade_requests (content_hash, label, memory_count)
        VALUES (rec.content_hash, rec.label, rec.memory_count)
        ON CONFLICT (content_hash) DO NOTHING;
        IF FOUND THEN
            PERFORM queue_outbox_message(
                'I read "' || COALESCE(rec.label, 'a document') || '" a while back and haven''t drawn on it since. '
                || 'Want me to let it fade, or keep it? Just tell me.',
                'document_fade', 'retention',
                jsonb_build_object('mode', 'web_inbox', 'content_hash', rec.content_hash));
            v_sent := v_sent + 1;
        END IF;
    END LOOP;
    RETURN jsonb_build_object('requested', v_sent);
END;
$$;

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
                    'document_fade', 'retention',
                    jsonb_build_object('mode', 'web_inbox', 'content_hash', rec.content_hash));
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
