-- User-facing memory pressure, fade decisions, and compression reports.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

INSERT INTO config_defaults (key, value, description) VALUES
    ('retention.irreversible_pruning_enabled', 'false'::jsonb,
     'Explicit opt-in for hard deletion after the undo window; false keeps archived originals recoverable'),
    ('retention.review_digest_limit', '5'::jsonb,
     'Maximum pending fade proposals in one user-facing digest'),
    ('retention.compression_reports_enabled', 'true'::jsonb,
     'Publish a bounded factual report after memory gists are compressed'),
    ('retention.compression_report_interval_seconds', '86400'::jsonb,
     'Minimum interval between batched compression reports'),
    ('retention.compression_report_limit', '10'::jsonb,
     'Maximum completed compressions in one report')
ON CONFLICT (key) DO NOTHING;

UPDATE config
SET description = 'Master switch for reversible rest-cycle memory consolidation (kill switch)'
WHERE key = 'retention.enabled';
UPDATE config
SET description = 'Reminder horizon for a pending conscious review; expiry never decides or deletes'
WHERE key = 'retention.review_expiry_days';

ALTER TABLE memory_review_queue
    ADD COLUMN IF NOT EXISTS proposed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS outbox_message_id UUID REFERENCES outbox_messages(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS decision_channel TEXT,
    ADD COLUMN IF NOT EXISTS decision_actor TEXT;

-- The legacy heartbeat action executor predates the user-facing fade review.
-- Keep its non-destructive KEEP veto, but make any transition that would archive
-- source memories prove it came through the explicit decision surface. This is a
-- database invariant, not a prompt convention.
CREATE OR REPLACE FUNCTION require_explicit_memory_fade_decision()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.status = 'pending'
       AND NEW.status = 'released'
       AND NEW.decision IN ('release', 'journal')
       AND COALESCE(
           current_setting('hexis.retention_explicit_decision', TRUE), ''
       ) <> OLD.id::text THEN
        RAISE EXCEPTION USING
            ERRCODE = '42501',
            MESSAGE = 'This load-bearing memory fade requires an explicit user decision. Nothing changed.',
            HINT = 'Use the Forgetting page or reply to the pending fade proposal.';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_require_explicit_memory_fade_decision
    ON memory_review_queue;
CREATE TRIGGER trg_require_explicit_memory_fade_decision
BEFORE UPDATE OF status, decision ON memory_review_queue
FOR EACH ROW EXECUTE FUNCTION require_explicit_memory_fade_decision();

CREATE TABLE IF NOT EXISTS memory_compression_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    gist_memory_id UUID NOT NULL UNIQUE REFERENCES memories(id) ON DELETE CASCADE,
    source_memory_ids UUID[] NOT NULL DEFAULT '{}'::uuid[],
    source_count INT NOT NULL DEFAULT 0,
    fidelity FLOAT NOT NULL CHECK (fidelity >= 0 AND fidelity <= 1),
    summary_preview TEXT NOT NULL,
    compressed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    outbox_message_id UUID REFERENCES outbox_messages(id) ON DELETE SET NULL,
    reported_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_memory_compression_reports_pending
    ON memory_compression_reports (compressed_at, id) WHERE reported_at IS NULL;

CREATE OR REPLACE FUNCTION capture_memory_compression_report()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_sources UUID[];
BEGIN
    IF NEW.metadata #>> '{consolidation,role}' = 'merged'
       AND COALESCE((NEW.metadata #>> '{consolidation,summarized}')::boolean, FALSE)
       AND NOT COALESCE((OLD.metadata #>> '{consolidation,summarized}')::boolean, FALSE) THEN
        SELECT COALESCE(array_agg(value::uuid), '{}'::uuid[])
        INTO v_sources
        FROM jsonb_array_elements_text(
            COALESCE(NEW.metadata #> '{consolidation,source_ids}', '[]'::jsonb)
        );
        INSERT INTO memory_compression_reports (
            gist_memory_id, source_memory_ids, source_count, fidelity,
            summary_preview, metadata
        ) VALUES (
            NEW.id,
            v_sources,
            COALESCE(cardinality(v_sources), 0),
            NEW.fidelity,
            left(NEW.content, 600),
            jsonb_build_object(
                'importance', NEW.importance,
                'source_attribution', NEW.source_attribution
            )
        ) ON CONFLICT (gist_memory_id) DO NOTHING;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_capture_memory_compression_report ON memories;
CREATE TRIGGER trg_capture_memory_compression_report
AFTER UPDATE OF content, fidelity, metadata ON memories
FOR EACH ROW EXECUTE FUNCTION capture_memory_compression_report();

CREATE OR REPLACE FUNCTION memory_fade_review_code(p_id UUID)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE
STRICT
AS $$
    SELECT upper(left(replace(p_id::text, '-', ''), 8));
$$;

CREATE OR REPLACE FUNCTION retention_observe_packet(p_limit INT DEFAULT 5)
RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    WITH status AS (
        SELECT retention_status() AS value
    ), low_fidelity AS (
        SELECT m.id, m.content, m.type, m.fidelity, m.updated_at
        FROM memories m
        WHERE m.status = 'active' AND m.fidelity < 0.75
        ORDER BY m.fidelity, m.updated_at DESC, m.id
        LIMIT LEAST(GREATEST(COALESCE(p_limit, 5), 1), 20)
    ), recent_compression AS (
        SELECT r.*
        FROM memory_compression_reports r
        ORDER BY r.compressed_at DESC, r.id
        LIMIT LEAST(GREATEST(COALESCE(p_limit, 5), 1), 20)
    )
    SELECT jsonb_build_object(
        'enabled', COALESCE((status.value->>'enabled')::boolean, FALSE),
        'irreversible_pruning_enabled', COALESCE(
            (status.value->>'irreversible_pruning_enabled')::boolean, FALSE
        ),
        'pressure', jsonb_build_object(
            'episodic_mass', status.value #> '{episodic,mass}',
            'capacity', status.value #> '{episodic,capacity}',
            'capacity_ratio', CASE
                WHEN COALESCE((status.value #>> '{episodic,capacity}')::float, 0) > 0
                THEN round((
                    (status.value #>> '{episodic,mass}')::numeric
                    / NULLIF((status.value #>> '{episodic,capacity}')::numeric, 0)
                ), 3)
                ELSE NULL
            END,
            'candidate_groups', status.value #> '{consolidation,candidate_groups}',
            'archived_recoverable', status.value #> '{episodic,archived}',
            'summarization_pending', status.value #> '{consolidation,summarize_pending}'
        ),
        'low_fidelity_count', (
            SELECT count(*) FROM memories WHERE status='active' AND fidelity < 0.75
        ),
        'low_fidelity', COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
                'memory_id', l.id,
                'content', l.content,
                'type', l.type,
                'fidelity', l.fidelity,
                'updated_at', l.updated_at
            ) ORDER BY l.fidelity, l.updated_at DESC, l.id)
            FROM low_fidelity l
        ), '[]'::jsonb),
        'recent_compressions', COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
                'report_id', c.id,
                'gist_memory_id', c.gist_memory_id,
                'source_memory_ids', to_jsonb(c.source_memory_ids),
                'source_count', c.source_count,
                'fidelity', c.fidelity,
                'summary_preview', c.summary_preview,
                'compressed_at', c.compressed_at,
                'reported_at', c.reported_at
            ) ORDER BY c.compressed_at DESC, c.id)
            FROM recent_compression c
        ), '[]'::jsonb)
    )
    FROM status;
$$;

-- Keep the existing Observe packet key stable while adding real pressure and
-- fidelity state beside the conscious decision queue.
CREATE OR REPLACE FUNCTION get_memories_at_threshold_context(p_limit INT DEFAULT 5)
RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    SELECT retention_observe_packet(p_limit) || jsonb_build_object(
        'budget_remaining', retention_budget_remaining(),
        'reviews', COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
                'review_id', q.id,
                'code', memory_fade_review_code(q.id),
                'preview', q.preview,
                'reason', q.reason,
                'memory_ids', to_jsonb(q.memory_ids),
                'expires_at', q.expires_at,
                'proposed_at', q.proposed_at
            ) ORDER BY q.created_at, q.id)
            FROM (
                SELECT * FROM memory_review_queue
                WHERE status = 'pending'
                ORDER BY created_at, id
                LIMIT LEAST(GREATEST(COALESCE(p_limit, 5), 1), 20)
            ) q
        ), '[]'::jsonb)
    );
$$;

CREATE OR REPLACE FUNCTION list_memory_fade_reviews(
    p_status TEXT DEFAULT 'pending',
    p_limit INT DEFAULT 50
)
RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', q.id,
        'code', memory_fade_review_code(q.id),
        'status', q.status,
        'decision', q.decision,
        'reason', q.reason,
        'preview', q.preview,
        'created_at', q.created_at,
        'expires_at', q.expires_at,
        'proposed_at', q.proposed_at,
        'decided_at', q.decided_at,
        'budget_remaining', retention_budget_remaining(),
        'memories', COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
                'id', m.id,
                'content', m.content,
                'status', m.status,
                'importance', m.importance,
                'strength', calculate_strength(
                    m.importance, m.decay_rate, m.created_at, m.last_reinforced
                ),
                'fidelity', m.fidelity,
                'load_bearing', is_memory_protected(m.id),
                'source_attribution', m.source_attribution,
                'created_at', m.created_at
            ) ORDER BY m.created_at, m.id)
            FROM memories m WHERE m.id = ANY(q.memory_ids)
        ), '[]'::jsonb)
    ) ORDER BY q.created_at DESC, q.id), '[]'::jsonb)
    FROM (
        SELECT * FROM memory_review_queue
        WHERE p_status = 'all' OR status = p_status
        ORDER BY created_at DESC, id
        LIMIT LEAST(GREATEST(COALESCE(p_limit, 50), 1), 200)
    ) q;
$$;

CREATE OR REPLACE FUNCTION decide_memory_fade_review(
    p_review_id UUID,
    p_decision TEXT,
    p_journal_content TEXT DEFAULT NULL,
    p_decision_channel TEXT DEFAULT NULL,
    p_decision_actor TEXT DEFAULT NULL
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_review memory_review_queue%ROWTYPE;
    v_decision TEXT := lower(btrim(COALESCE(p_decision, '')));
    v_actor TEXT := COALESCE(NULLIF(btrim(COALESCE(p_decision_actor, '')), ''), 'user');
    v_gist UUID;
    v_count INT;
    v_fidelity FLOAT;
BEGIN
    IF v_decision NOT IN ('keep', 'release', 'journal') THEN
        RETURN jsonb_build_object('ok', FALSE, 'error', 'invalid_decision');
    END IF;
    SELECT * INTO v_review
    FROM memory_review_queue
    WHERE id = p_review_id
    FOR UPDATE;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('ok', FALSE, 'error', 'review_not_found');
    END IF;
    IF v_review.status <> 'pending' THEN
        RETURN jsonb_build_object(
            'ok', TRUE, 'already_decided', TRUE,
            'review_id', v_review.id, 'status', v_review.status,
            'decision', v_review.decision
        );
    END IF;
    SELECT count(*)::int INTO v_count
    FROM memories WHERE id = ANY(v_review.memory_ids) AND status = 'active';

    IF v_decision = 'keep' THEN
        IF NOT spend_retention_budget() THEN
            RETURN jsonb_build_object(
                'ok', FALSE,
                'error', 'no_retention_budget',
                'message', 'No finite keep-budget remains in this chapter. Nothing changed; journal the memory or let it compress.'
            );
        END IF;
        UPDATE memories
        SET metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object(
            'protected', TRUE,
            'retention_decision', jsonb_build_object(
                'review_id', v_review.id, 'actor', v_actor, 'decision', 'keep'
            )
        )
        WHERE id = ANY(v_review.memory_ids) AND status = 'active';
        PERFORM touch_memories(v_review.memory_ids);
    ELSE
        -- Transaction-local, exact-review proof consumed by the trigger above.
        -- The value cannot authorize a second review and disappears at commit.
        PERFORM set_config(
            'hexis.retention_explicit_decision', v_review.id::text, TRUE
        );
        IF v_decision = 'journal' THEN
            PERFORM write_journal_entry(
                p_content := COALESCE(
                    NULLIF(btrim(COALESCE(p_journal_content, '')), ''),
                    v_review.preview,
                    'A memory I chose to keep in words before letting it compress.'
                ),
                p_title := 'Before this memory compressed',
                p_metadata := jsonb_build_object(
                    'source', 'memory_review',
                    'review_id', v_review.id,
                    'actor', v_actor
                )
            );
        END IF;
        v_gist := consolidate_memory_group(v_review.memory_ids);
        IF v_gist IS NULL THEN
            RETURN jsonb_build_object(
                'ok', FALSE,
                'error', 'nothing_eligible_to_compress',
                'message', 'The source memories changed or are now protected. Nothing was compressed; refresh the review.'
            );
        END IF;
        SELECT fidelity INTO v_fidelity FROM memories WHERE id = v_gist;
    END IF;

    UPDATE memory_review_queue
    SET status = CASE WHEN v_decision = 'keep' THEN 'kept' ELSE 'released' END,
        decision = v_decision,
        decision_channel = NULLIF(btrim(COALESCE(p_decision_channel, '')), ''),
        decision_actor = v_actor,
        decided_at = CURRENT_TIMESTAMP
    WHERE id = v_review.id;

    RETURN jsonb_strip_nulls(jsonb_build_object(
        'ok', TRUE,
        'review_id', v_review.id,
        'decision', v_decision,
        'status', CASE WHEN v_decision = 'keep' THEN 'kept' ELSE 'released' END,
        'budget_remaining', retention_budget_remaining(),
        'compression', CASE WHEN v_gist IS NULL THEN NULL ELSE jsonb_build_object(
            'source_count', v_count,
            'gist_memory_id', v_gist,
            'fidelity_before_summary', v_fidelity,
            'summarization', 'queued',
            'originals_recoverable', NOT COALESCE(
                get_config_bool('retention.irreversible_pruning_enabled'), FALSE
            )
        ) END,
        'next_step', CASE
            WHEN v_decision = 'keep' THEN 'The memories are protected and reinforced.'
            ELSE 'The full source memories are archived recoverably; a summarization report will show the resulting fidelity.'
        END
    ));
END;
$$;

CREATE OR REPLACE FUNCTION publish_memory_fade_review_digest()
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_limit INT := LEAST(GREATEST(COALESCE(
        get_config_int('retention.review_digest_limit'), 5
    ), 1), 20);
    v_review RECORD;
    v_ids UUID[] := '{}'::uuid[];
    v_message TEXT;
    v_outbox UUID;
BEGIN
    IF NOT COALESCE(get_config_bool('retention.enabled'), FALSE) THEN
        RETURN jsonb_build_object('skipped', TRUE, 'reason', 'retention_disabled');
    END IF;
    v_message := 'I found memories near the compression threshold that may be load-bearing. I have not compressed them; you choose what happens.';
    FOR v_review IN
        SELECT * FROM memory_review_queue
        WHERE status = 'pending' AND proposed_at IS NULL
        ORDER BY created_at, id
        LIMIT v_limit
        FOR UPDATE SKIP LOCKED
    LOOP
        v_ids := array_append(v_ids, v_review.id);
        v_message := v_message || format(
            E'\n\n%s — %s\n%s\nReply `keep %s`, `release %s`, or `journal %s: what should remain in writing`.',
            memory_fade_review_code(v_review.id),
            COALESCE(v_review.reason, 'near the retention threshold'),
            left(COALESCE(v_review.preview, 'No preview available.'), 600),
            memory_fade_review_code(v_review.id),
            memory_fade_review_code(v_review.id),
            memory_fade_review_code(v_review.id)
        );
    END LOOP;
    IF cardinality(v_ids) = 0 THEN
        RETURN jsonb_build_object('skipped', TRUE, 'reason', 'no_unpublished_reviews');
    END IF;
    v_outbox := queue_outbox_message(
        v_message,
        'memory_fade_review',
        'retention',
        jsonb_build_object(
            'mode', 'web_inbox',
            'requires_response', TRUE,
            'review_ids', to_jsonb(v_ids),
            'review_url', '/forgetting'
        )
    );
    UPDATE memory_review_queue
    SET proposed_at = CURRENT_TIMESTAMP, outbox_message_id = v_outbox
    WHERE id = ANY(v_ids);
    RETURN jsonb_build_object(
        'skipped', FALSE,
        'count', cardinality(v_ids),
        'review_ids', to_jsonb(v_ids),
        'outbox_message_id', v_outbox
    );
END;
$$;

CREATE OR REPLACE FUNCTION publish_retention_compression_report_if_due(
    p_force BOOLEAN DEFAULT FALSE
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_state JSONB;
    v_last TIMESTAMPTZ;
    v_interval INT := LEAST(GREATEST(COALESCE(
        get_config_int('retention.compression_report_interval_seconds'), 86400
    ), 60), 604800);
    v_limit INT := LEAST(GREATEST(COALESCE(
        get_config_int('retention.compression_report_limit'), 10
    ), 1), 50);
    v_report RECORD;
    v_ids UUID[] := '{}'::uuid[];
    v_message TEXT;
    v_outbox UUID;
    v_sources INT := 0;
BEGIN
    IF NOT COALESCE(get_config_bool('retention.compression_reports_enabled'), TRUE) THEN
        RETURN jsonb_build_object('skipped', TRUE, 'reason', 'disabled');
    END IF;
    INSERT INTO state (key, value) VALUES ('retention_compression_report_state', '{}'::jsonb)
    ON CONFLICT (key) DO NOTHING;
    SELECT value INTO v_state FROM state
    WHERE key = 'retention_compression_report_state' FOR UPDATE;
    v_last := NULLIF(v_state->>'last_reported_at', '')::timestamptz;
    IF NOT COALESCE(p_force, FALSE)
       AND v_last IS NOT NULL
       AND CURRENT_TIMESTAMP < v_last + make_interval(secs => v_interval) THEN
        RETURN jsonb_build_object('skipped', TRUE, 'reason', 'not_due');
    END IF;
    v_message := 'Here is what I compressed. The fidelity numbers are the actual stored values, not estimates.';
    FOR v_report IN
        SELECT * FROM memory_compression_reports
        WHERE reported_at IS NULL
        ORDER BY compressed_at, id
        LIMIT v_limit
        FOR UPDATE SKIP LOCKED
    LOOP
        v_ids := array_append(v_ids, v_report.id);
        v_sources := v_sources + v_report.source_count;
        v_message := v_message || format(
            E'\n\n%s source memories → one gist at %s%% fidelity\n%s',
            v_report.source_count,
            round((v_report.fidelity * 100)::numeric),
            v_report.summary_preview
        );
    END LOOP;
    IF cardinality(v_ids) = 0 THEN
        RETURN jsonb_build_object('skipped', TRUE, 'reason', 'no_unreported_compressions');
    END IF;
    v_outbox := queue_outbox_message(
        v_message,
        'memory_compression_report',
        'retention',
        jsonb_build_object('mode', 'web_inbox', 'review_url', '/forgetting')
    );
    UPDATE memory_compression_reports
    SET reported_at = CURRENT_TIMESTAMP, outbox_message_id = v_outbox
    WHERE id = ANY(v_ids);
    PERFORM set_state(
        'retention_compression_report_state',
        v_state || jsonb_build_object(
            'last_reported_at', CURRENT_TIMESTAMP,
            'last_count', cardinality(v_ids),
            'last_source_count', v_sources,
            'outbox_message_id', v_outbox
        )
    );
    RETURN jsonb_build_object(
        'skipped', FALSE,
        'compression_count', cardinality(v_ids),
        'source_count', v_sources,
        'outbox_message_id', v_outbox
    );
END;
$$;

CREATE OR REPLACE FUNCTION try_resolve_memory_fade_review_from_inbound(
    p_channel TEXT,
    p_actor TEXT,
    p_text TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_match TEXT[];
    v_decision TEXT;
    v_code TEXT;
    v_journal TEXT;
    v_review UUID;
    v_result JSONB;
BEGIN
    v_match := regexp_match(
        btrim(COALESCE(p_text, '')),
        '^(keep|release)[[:space:]:#-]+([0-9A-Fa-f]{8})[[:space:]]*$'
    );
    IF v_match IS NULL THEN
        v_match := regexp_match(
            btrim(COALESCE(p_text, '')),
            '^journal[[:space:]:#-]+([0-9A-Fa-f]{8})[[:space:]]*:[[:space:]]*(.+)$'
        );
        IF v_match IS NULL THEN
            RETURN jsonb_build_object('recognized', FALSE, 'matched', FALSE);
        END IF;
        v_decision := 'journal';
        v_code := upper(v_match[1]);
        v_journal := btrim(v_match[2]);
    ELSE
        v_decision := lower(v_match[1]);
        v_code := upper(v_match[2]);
    END IF;
    SELECT id INTO v_review
    FROM memory_review_queue
    WHERE status='pending' AND memory_fade_review_code(id)=v_code;
    IF v_review IS NULL THEN
        RETURN jsonb_build_object(
            'recognized', TRUE,
            'matched', FALSE,
            'message', 'That memory-fade code is not pending. Open Forgetting for the current proposals.'
        );
    END IF;
    v_result := decide_memory_fade_review(
        v_review, v_decision, v_journal, p_channel, p_actor
    );
    RETURN v_result || jsonb_build_object(
        'recognized', TRUE,
        'matched', COALESCE((v_result->>'ok')::boolean, FALSE),
        'message', COALESCE(
            v_result->>'message',
            initcap(v_decision) || ' recorded. ' || COALESCE(v_result->>'next_step', '')
        )
    );
END;
$$;

COMMENT ON FUNCTION retention_observe_packet(INT) IS
    'Truthful memory pressure, fidelity, and recent compression state for the heartbeat Observe packet and UI.';
COMMENT ON FUNCTION decide_memory_fade_review(UUID, TEXT, TEXT, TEXT, TEXT) IS
    'Apply an explicit keep, release, or journal decision to a load-bearing fade proposal; no timer decides it.';
