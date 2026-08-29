-- Contradictions are durable review events, never silent graph trivia.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

INSERT INTO config_defaults (key, value, description) VALUES
    ('contradictions.enabled', 'true'::jsonb,
     'Queue new semantic/worldview memories for contradiction detection.'),
    ('contradictions.detection_interval_seconds', '86400'::jsonb,
     'Minimum interval between batched model contradiction checks.'),
    ('contradictions.detection_batch_size', '20'::jsonb,
     'Maximum newly embedded memories checked in one contradiction-detection batch.'),
    ('contradictions.candidates_per_memory', '8'::jsonb,
     'Maximum same-topic memories presented beside one new memory for contradiction adjudication.'),
    ('contradictions.candidate_similarity', '0.55'::jsonb,
     'Minimum embedding similarity for contradiction candidates; lexical matches remain eligible.'),
    ('contradictions.minimum_confidence', '0.78'::jsonb,
     'Minimum model confidence required to file a contradiction review case.'),
    ('contradictions.digest_interval_seconds', '86400'::jsonb,
     'Minimum interval between batched user-facing contradiction review digests.'),
    ('contradictions.digest_limit', '10'::jsonb,
     'Maximum pending contradiction cases included in one review digest.')
ON CONFLICT (key) DO NOTHING;

CREATE TABLE IF NOT EXISTS contradiction_detection_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    attempts INT NOT NULL DEFAULT 0,
    max_attempts INT NOT NULL DEFAULT 3,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    claimed_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    result JSONB,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (memory_id, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_contradiction_detection_pending
    ON contradiction_detection_queue (next_attempt_at, created_at)
    WHERE status IN ('pending', 'processing');

CREATE TABLE IF NOT EXISTS contradiction_cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_a UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    memory_b UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    new_memory_id UUID REFERENCES memories(id) ON DELETE SET NULL,
    tension TEXT NOT NULL,
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    detected_by TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'resolved', 'tension')),
    outcome TEXT CHECK (outcome IN ('new_right', 'old_right', 'tension')),
    winner_memory_id UUID REFERENCES memories(id) ON DELETE SET NULL,
    loser_memory_id UUID REFERENCES memories(id) ON DELETE SET NULL,
    resolution_memory_id UUID REFERENCES memories(id) ON DELETE SET NULL,
    supersession_id UUID REFERENCES memory_supersessions(id) ON DELETE SET NULL,
    resolution_note TEXT,
    decision_channel TEXT,
    decision_actor TEXT,
    proposed_at TIMESTAMPTZ,
    outbox_message_id UUID REFERENCES outbox_messages(id) ON DELETE SET NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (memory_a <> memory_b)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_contradiction_cases_pending_pair
    ON contradiction_cases (LEAST(memory_a, memory_b), GREATEST(memory_a, memory_b))
    WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_contradiction_cases_ledger
    ON contradiction_cases (status, detected_at DESC);

CREATE OR REPLACE FUNCTION contradiction_case_code(p_id UUID)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE
STRICT
AS $$
    SELECT upper(left(replace(p_id::text, '-', ''), 8));
$$;

CREATE OR REPLACE FUNCTION enqueue_contradiction_detection(p_memory_id UUID)
RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    v_memory memories%ROWTYPE;
    v_id UUID;
BEGIN
    IF NOT COALESCE(get_config_bool('contradictions.enabled'), TRUE) THEN
        RETURN NULL;
    END IF;
    SELECT * INTO v_memory
    FROM memories
    WHERE id = p_memory_id
      AND type IN ('semantic', 'worldview')
      AND status = 'active'
      AND (valid_until IS NULL OR valid_until > CURRENT_TIMESTAMP);
    IF NOT FOUND THEN
        RETURN NULL;
    END IF;
    INSERT INTO contradiction_detection_queue (memory_id, content_hash)
    VALUES (v_memory.id, encode(digest(v_memory.content, 'sha256'), 'hex'))
    ON CONFLICT (memory_id, content_hash) DO UPDATE
    SET status = CASE
            WHEN contradiction_detection_queue.status = 'failed'
              OR (contradiction_detection_queue.status = 'completed'
                  AND v_memory.embedding_status = 'embedded')
            THEN 'pending'
            ELSE contradiction_detection_queue.status
        END,
        next_attempt_at = CASE
            WHEN contradiction_detection_queue.status = 'failed'
              OR (contradiction_detection_queue.status = 'completed'
                  AND v_memory.embedding_status = 'embedded')
            THEN CURRENT_TIMESTAMP
            ELSE contradiction_detection_queue.next_attempt_at
        END,
        completed_at = CASE
            WHEN contradiction_detection_queue.status IN ('failed', 'completed')
             AND v_memory.embedding_status = 'embedded'
            THEN NULL
            ELSE contradiction_detection_queue.completed_at
        END,
        updated_at = CURRENT_TIMESTAMP
    RETURNING id INTO v_id;
    RETURN v_id;
END;
$$;

CREATE OR REPLACE FUNCTION queue_memory_contradiction_detection()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.type IN ('semantic', 'worldview')
       AND NEW.status = 'active'
       AND (TG_OP = 'INSERT'
            OR OLD.content IS DISTINCT FROM NEW.content
            OR OLD.embedding_status IS DISTINCT FROM NEW.embedding_status
            OR OLD.status IS DISTINCT FROM NEW.status) THEN
        PERFORM enqueue_contradiction_detection(NEW.id);
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_queue_memory_contradiction_detection ON memories;
CREATE TRIGGER trg_queue_memory_contradiction_detection
AFTER INSERT OR UPDATE OF content, embedding_status, status ON memories
FOR EACH ROW EXECUTE FUNCTION queue_memory_contradiction_detection();

CREATE OR REPLACE FUNCTION contradiction_candidate_memories(
    p_memory_id UUID,
    p_limit INT DEFAULT NULL
)
RETURNS TABLE (
    memory_id UUID,
    content TEXT,
    memory_type memory_type,
    trust_level FLOAT,
    source_attribution JSONB,
    similarity FLOAT,
    lexical_rank FLOAT
)
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_limit INT := LEAST(GREATEST(COALESCE(
        p_limit, get_config_int('contradictions.candidates_per_memory'), 8
    ), 1), 30);
    v_threshold FLOAT := COALESCE(get_config_float('contradictions.candidate_similarity'), 0.55);
BEGIN
    RETURN QUERY
    WITH subject AS (
        SELECT m.id, m.content, m.embedding, m.embedding_status,
               websearch_to_tsquery('english', m.content) AS query
        FROM memories m
        WHERE m.id = p_memory_id
    ), scored AS (
        SELECT m.id, m.content, m.type, m.trust_level, m.source_attribution,
               CASE
                   WHEN s.embedding_status = 'embedded'
                    AND m.embedding_status = 'embedded'
                    AND s.embedding IS NOT NULL AND m.embedding IS NOT NULL
                   THEN (1.0 - (m.embedding <=> s.embedding))::float
                   ELSE NULL::float
               END AS semantic_similarity,
               ts_rank_cd(to_tsvector('english', m.content), s.query, 32)::float AS word_rank
        FROM subject s
        JOIN memories m ON m.id <> s.id
        WHERE m.type IN ('semantic', 'worldview')
          AND m.status = 'active'
          AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
          AND NOT EXISTS (
              SELECT 1 FROM contradiction_cases c
              WHERE LEAST(c.memory_a, c.memory_b) = LEAST(s.id, m.id)
                AND GREATEST(c.memory_a, c.memory_b) = GREATEST(s.id, m.id)
                AND c.status IN ('pending', 'tension')
          )
    )
    SELECT s.id, s.content, s.type, s.trust_level, s.source_attribution,
           s.semantic_similarity, s.word_rank
    FROM scored s
    WHERE COALESCE(s.semantic_similarity >= v_threshold, FALSE)
       OR s.word_rank > 0
    ORDER BY GREATEST(COALESCE(s.semantic_similarity, 0), s.word_rank) DESC,
             s.trust_level DESC,
             s.id
    LIMIT v_limit;
END;
$$;

CREATE OR REPLACE FUNCTION claim_contradiction_detection_batch(
    p_limit INT DEFAULT NULL,
    p_force BOOLEAN DEFAULT FALSE
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_limit INT := LEAST(GREATEST(COALESCE(
        p_limit, get_config_int('contradictions.detection_batch_size'), 20
    ), 1), 100);
    v_interval INT := LEAST(GREATEST(COALESCE(
        get_config_int('contradictions.detection_interval_seconds'), 86400
    ), 10), 604800);
    v_state JSONB;
    v_last TIMESTAMPTZ;
    v_result JSONB;
BEGIN
    IF NOT COALESCE(get_config_bool('contradictions.enabled'), TRUE) THEN
        RETURN jsonb_build_object('skipped', TRUE, 'reason', 'disabled', 'items', '[]'::jsonb);
    END IF;
    INSERT INTO state (key, value)
    VALUES ('contradiction_detection_state', '{}'::jsonb)
    ON CONFLICT (key) DO NOTHING;
    SELECT value INTO v_state
    FROM state
    WHERE key = 'contradiction_detection_state'
    FOR UPDATE;
    v_last := NULLIF(v_state->>'last_batch_started_at', '')::timestamptz;
    IF NOT COALESCE(p_force, FALSE)
       AND v_last IS NOT NULL
       AND CURRENT_TIMESTAMP < v_last + make_interval(secs => v_interval) THEN
        RETURN jsonb_build_object('skipped', TRUE, 'reason', 'not_due', 'items', '[]'::jsonb);
    END IF;

    WITH picked AS (
        SELECT q.id
        FROM contradiction_detection_queue q
        JOIN memories m ON m.id = q.memory_id
        WHERE (
                (q.status = 'pending' AND q.next_attempt_at <= CURRENT_TIMESTAMP)
                OR (
                    q.status = 'processing'
                    AND q.claimed_at < CURRENT_TIMESTAMP - INTERVAL '15 minutes'
                )
              )
          AND m.status = 'active'
          AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
        ORDER BY q.created_at, q.id
        FOR UPDATE OF q SKIP LOCKED
        LIMIT v_limit
    ), claimed AS (
        UPDATE contradiction_detection_queue q
        SET status = 'processing',
            attempts = attempts + 1,
            claimed_at = CURRENT_TIMESTAMP,
            error = NULL,
            updated_at = CURRENT_TIMESTAMP
        FROM picked p
        WHERE q.id = p.id
        RETURNING q.id, q.memory_id, q.attempts
    )
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'queue_id', c.id,
        'memory', jsonb_build_object(
            'memory_id', m.id,
            'content', m.content,
            'type', m.type,
            'created_at', m.created_at,
            'trust_level', m.trust_level,
            'source_attribution', m.source_attribution
        ),
        'candidates', COALESCE((
            SELECT jsonb_agg(to_jsonb(candidate))
            FROM contradiction_candidate_memories(m.id) candidate
        ), '[]'::jsonb),
        'attempt', c.attempts
    ) ORDER BY m.created_at, c.id), '[]'::jsonb)
    INTO v_result
    FROM claimed c
    JOIN memories m ON m.id = c.memory_id;

    PERFORM set_state(
        'contradiction_detection_state',
        v_state || jsonb_build_object(
            'last_batch_started_at', CURRENT_TIMESTAMP,
            'last_batch_size', jsonb_array_length(v_result)
        )
    );
    RETURN jsonb_build_object(
        'skipped', jsonb_array_length(v_result) = 0,
        'reason', CASE WHEN jsonb_array_length(v_result) = 0 THEN 'empty_queue' ELSE NULL END,
        'minimum_confidence', COALESCE(get_config_float('contradictions.minimum_confidence'), 0.78),
        'items', v_result
    );
END;
$$;

CREATE OR REPLACE FUNCTION finish_contradiction_detection_batch(
    p_queue_ids UUID[],
    p_result JSONB DEFAULT '{}'::jsonb,
    p_error TEXT DEFAULT NULL
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_completed INT := 0;
    v_retried INT := 0;
    v_failed INT := 0;
BEGIN
    IF p_queue_ids IS NULL OR array_length(p_queue_ids, 1) IS NULL THEN
        RETURN jsonb_build_object('completed', 0, 'retried', 0, 'failed', 0);
    END IF;
    IF NULLIF(btrim(COALESCE(p_error, '')), '') IS NULL THEN
        UPDATE contradiction_detection_queue
        SET status = 'completed', completed_at = CURRENT_TIMESTAMP,
            result = COALESCE(p_result, '{}'::jsonb), error = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ANY(p_queue_ids) AND status = 'processing';
        GET DIAGNOSTICS v_completed = ROW_COUNT;
    ELSE
        UPDATE contradiction_detection_queue
        SET status = CASE WHEN attempts >= max_attempts THEN 'failed' ELSE 'pending' END,
            next_attempt_at = CURRENT_TIMESTAMP + make_interval(secs => LEAST(3600, 30 * (2 ^ GREATEST(attempts - 1, 0))::int)),
            claimed_at = NULL,
            error = left(p_error, 1000),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ANY(p_queue_ids) AND status = 'processing';
        GET DIAGNOSTICS v_retried = ROW_COUNT;
        SELECT count(*) INTO v_failed
        FROM contradiction_detection_queue
        WHERE id = ANY(p_queue_ids) AND status = 'failed';
        v_retried := v_retried - v_failed;
    END IF;
    RETURN jsonb_build_object('completed', v_completed, 'retried', v_retried, 'failed', v_failed);
END;
$$;

CREATE OR REPLACE FUNCTION file_contradiction_case(
    p_memory_a UUID,
    p_memory_b UUID,
    p_new_memory_id UUID,
    p_tension TEXT,
    p_confidence FLOAT,
    p_detected_by TEXT DEFAULT 'model',
    p_metadata JSONB DEFAULT '{}'::jsonb
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_case contradiction_cases%ROWTYPE;
    v_confidence FLOAT := LEAST(1.0, GREATEST(0.0, COALESCE(p_confidence, 0.0)));
    v_threshold FLOAT := COALESCE(get_config_float('contradictions.minimum_confidence'), 0.78);
BEGIN
    IF p_memory_a IS NULL OR p_memory_b IS NULL OR p_memory_a = p_memory_b THEN
        RETURN jsonb_build_object('created', FALSE, 'reason', 'invalid_pair');
    END IF;
    IF v_confidence < v_threshold THEN
        RETURN jsonb_build_object('created', FALSE, 'reason', 'below_threshold', 'confidence', v_confidence);
    END IF;
    IF NULLIF(btrim(COALESCE(p_tension, '')), '') IS NULL THEN
        RETURN jsonb_build_object('created', FALSE, 'reason', 'missing_tension');
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM memories
        WHERE id IN (p_memory_a, p_memory_b)
          AND type IN ('semantic', 'worldview')
          AND status = 'active'
          AND (valid_until IS NULL OR valid_until > CURRENT_TIMESTAMP)
        HAVING count(*) = 2
    ) THEN
        RETURN jsonb_build_object('created', FALSE, 'reason', 'memory_not_found');
    END IF;

    INSERT INTO contradiction_cases (
        memory_a, memory_b, new_memory_id, tension, confidence, detected_by, metadata
    ) VALUES (
        p_memory_a, p_memory_b,
        CASE WHEN p_new_memory_id IN (p_memory_a, p_memory_b) THEN p_new_memory_id ELSE NULL END,
        btrim(p_tension), v_confidence,
        COALESCE(NULLIF(btrim(p_detected_by), ''), 'model'),
        COALESCE(p_metadata, '{}'::jsonb)
    )
    ON CONFLICT (LEAST(memory_a, memory_b), GREATEST(memory_a, memory_b))
        WHERE status = 'pending'
    DO UPDATE SET
        tension = EXCLUDED.tension,
        confidence = GREATEST(contradiction_cases.confidence, EXCLUDED.confidence),
        new_memory_id = COALESCE(EXCLUDED.new_memory_id, contradiction_cases.new_memory_id),
        metadata = contradiction_cases.metadata || EXCLUDED.metadata,
        updated_at = CURRENT_TIMESTAMP
    RETURNING * INTO v_case;

    BEGIN
        PERFORM create_memory_relationship(
            v_case.memory_a,
            v_case.memory_b,
            'CONTRADICTS',
            jsonb_build_object(
                'confidence', v_case.confidence,
                'source', v_case.detected_by,
                'case_id', v_case.id,
                'tension', v_case.tension
            )
        );
    EXCEPTION WHEN OTHERS THEN
        -- The relational case is authoritative; AGE availability is advisory.
        NULL;
    END;
    UPDATE drives SET current_level = LEAST(1.0, current_level + 0.15)
    WHERE name = 'coherence';
    RETURN jsonb_build_object(
        'created', TRUE,
        'case_id', v_case.id,
        'code', contradiction_case_code(v_case.id),
        'status', v_case.status,
        'confidence', v_case.confidence
    );
END;
$$;

-- The existing subconscious observer records contradiction observations as a
-- strategic memory. Convert those observations into the same durable review
-- ledger so every detector has one propose-and-decide path.
CREATE OR REPLACE FUNCTION file_strategic_contradiction_observation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_evidence JSONB := COALESCE(NEW.metadata->'supporting_evidence', '{}'::jsonb);
    v_memory_a UUID := _db_brain_try_uuid(v_evidence->>'memory_a');
    v_memory_b UUID := _db_brain_try_uuid(v_evidence->>'memory_b');
BEGIN
    IF NEW.type = 'strategic'
       AND v_evidence->>'kind' = 'contradiction'
       AND v_memory_a IS NOT NULL
       AND v_memory_b IS NOT NULL THEN
        PERFORM file_contradiction_case(
            v_memory_a,
            v_memory_b,
            NULL,
            COALESCE(NULLIF(v_evidence->>'tension', ''), NEW.content),
            COALESCE(
                NULLIF(v_evidence->>'confidence', '')::float,
                NULLIF(NEW.metadata->>'confidence_score', '')::float,
                0.0
            ),
            'subconscious',
            jsonb_build_object('observation_memory_id', NEW.id)
        );
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_file_strategic_contradiction_observation ON memories;
CREATE TRIGGER trg_file_strategic_contradiction_observation
AFTER INSERT ON memories
FOR EACH ROW EXECUTE FUNCTION file_strategic_contradiction_observation();

CREATE OR REPLACE FUNCTION decide_contradiction(
    p_case_id UUID,
    p_outcome TEXT,
    p_note TEXT DEFAULT NULL,
    p_decision_channel TEXT DEFAULT NULL,
    p_decision_actor TEXT DEFAULT NULL
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_case contradiction_cases%ROWTYPE;
    v_outcome TEXT := lower(btrim(COALESCE(p_outcome, '')));
    v_new UUID;
    v_old UUID;
    v_winner UUID;
    v_loser UUID;
    v_resolution UUID;
    v_supersession UUID;
    v_actor TEXT := COALESCE(NULLIF(btrim(p_decision_actor), ''), 'user');
    v_note TEXT;
BEGIN
    IF v_outcome NOT IN ('new_right', 'old_right', 'tension') THEN
        RETURN jsonb_build_object('ok', FALSE, 'error', 'invalid_outcome');
    END IF;
    SELECT * INTO v_case
    FROM contradiction_cases
    WHERE id = p_case_id
    FOR UPDATE;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('ok', FALSE, 'error', 'case_not_found');
    END IF;
    IF v_case.status <> 'pending' THEN
        RETURN jsonb_build_object(
            'ok', TRUE, 'already_decided', TRUE, 'case_id', v_case.id,
            'status', v_case.status, 'outcome', v_case.outcome
        );
    END IF;

    v_new := COALESCE(v_case.new_memory_id, (
        SELECT id FROM memories
        WHERE id IN (v_case.memory_a, v_case.memory_b)
        ORDER BY created_at DESC, id DESC LIMIT 1
    ));
    v_old := CASE WHEN v_new = v_case.memory_a THEN v_case.memory_b ELSE v_case.memory_a END;
    IF v_outcome = 'new_right' THEN
        v_winner := v_new;
        v_loser := v_old;
    ELSIF v_outcome = 'old_right' THEN
        v_winner := v_old;
        v_loser := v_new;
    END IF;
    v_note := COALESCE(
        NULLIF(btrim(p_note), ''),
        CASE v_outcome
            WHEN 'new_right' THEN 'The newer memory is correct.'
            WHEN 'old_right' THEN 'The older memory remains correct.'
            ELSE 'Both memories remain valid in different contexts.'
        END
    );

    IF v_outcome <> 'tension' THEN
        v_supersession := record_supersession(
            v_loser,
            v_winner,
            'Contradiction decision ' || contradiction_case_code(v_case.id) || ': ' || v_note,
            v_actor,
            'active',
            CURRENT_TIMESTAMP,
            NULL,
            TRUE,
            jsonb_build_object('contradiction_case_id', v_case.id, 'outcome', v_outcome)
        );
    END IF;

    v_resolution := create_strategic_memory(
        p_content := v_note,
        p_pattern_description := CASE
            WHEN v_outcome = 'tension' THEN 'Contradiction accepted as contextual tension'
            ELSE 'Contradiction resolved by explicit user decision'
        END,
        p_confidence_score := 1.0,
        p_supporting_evidence := jsonb_build_object(
            'contradiction_case_id', v_case.id,
            'memory_a', v_case.memory_a,
            'memory_b', v_case.memory_b,
            'outcome', v_outcome,
            'winner_memory_id', v_winner,
            'loser_memory_id', v_loser
        ),
        p_importance := 0.7,
        p_source_attribution := jsonb_build_object(
            'kind', 'user_testimony',
            'ref', 'contradiction-decision:' || v_case.id::text,
            'label', 'Explicit contradiction decision',
            'trust', 1.0
        ),
        p_trust_level := 1.0
    );

    IF v_outcome <> 'tension' THEN
        BEGIN
            EXECUTE format(
                'SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
                    MATCH (a:MemoryNode {memory_id: %L})-[r:CONTRADICTS]-(b:MemoryNode {memory_id: %L})
                    DELETE r RETURN a
                $q$) as (result ag_catalog.agtype)',
                v_case.memory_a, v_case.memory_b
            );
        EXCEPTION WHEN OTHERS THEN NULL;
        END;
        PERFORM delete_memory_edge('memory', v_case.memory_a::text, 'CONTRADICTS', 'memory', v_case.memory_b::text);
        PERFORM delete_memory_edge('memory', v_case.memory_b::text, 'CONTRADICTS', 'memory', v_case.memory_a::text);
    END IF;

    UPDATE contradiction_cases
    SET status = CASE WHEN v_outcome = 'tension' THEN 'tension' ELSE 'resolved' END,
        outcome = v_outcome,
        winner_memory_id = v_winner,
        loser_memory_id = v_loser,
        resolution_memory_id = v_resolution,
        supersession_id = v_supersession,
        resolution_note = v_note,
        decision_channel = NULLIF(btrim(COALESCE(p_decision_channel, '')), ''),
        decision_actor = v_actor,
        resolved_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = v_case.id;

    RETURN jsonb_strip_nulls(jsonb_build_object(
        'ok', TRUE,
        'case_id', v_case.id,
        'status', CASE WHEN v_outcome = 'tension' THEN 'tension' ELSE 'resolved' END,
        'outcome', v_outcome,
        'winner_memory_id', v_winner,
        'loser_memory_id', v_loser,
        'resolution_memory_id', v_resolution,
        'supersession_id', v_supersession
    ));
END;
$$;

CREATE OR REPLACE FUNCTION list_contradiction_cases(
    p_status TEXT DEFAULT 'all',
    p_limit INT DEFAULT 100
)
RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    SELECT COALESCE(jsonb_agg(jsonb_strip_nulls(jsonb_build_object(
        'id', c.id,
        'code', contradiction_case_code(c.id),
        'status', c.status,
        'outcome', c.outcome,
        'tension', c.tension,
        'confidence', c.confidence,
        'detected_by', c.detected_by,
        'new_memory_id', c.new_memory_id,
        'memory_a', jsonb_build_object(
            'id', a.id, 'content', a.content, 'type', a.type,
            'trust_level', a.trust_level, 'source_attribution', a.source_attribution,
            'created_at', a.created_at, 'valid_from', a.valid_from,
            'valid_until', a.valid_until, 'superseded_by', a.superseded_by
        ),
        'memory_b', jsonb_build_object(
            'id', b.id, 'content', b.content, 'type', b.type,
            'trust_level', b.trust_level, 'source_attribution', b.source_attribution,
            'created_at', b.created_at, 'valid_from', b.valid_from,
            'valid_until', b.valid_until, 'superseded_by', b.superseded_by
        ),
        'winner_memory_id', c.winner_memory_id,
        'loser_memory_id', c.loser_memory_id,
        'resolution_memory_id', c.resolution_memory_id,
        'supersession_id', c.supersession_id,
        'resolution_note', c.resolution_note,
        'proposed_at', c.proposed_at,
        'detected_at', c.detected_at,
        'resolved_at', c.resolved_at
    )) ORDER BY c.detected_at DESC, c.id), '[]'::jsonb)
    FROM (
        SELECT * FROM contradiction_cases
        WHERE p_status = 'all' OR status = p_status
        ORDER BY detected_at DESC, id
        LIMIT LEAST(GREATEST(COALESCE(p_limit, 100), 1), 500)
    ) c
    JOIN memories a ON a.id = c.memory_a
    JOIN memories b ON b.id = c.memory_b;
$$;

CREATE OR REPLACE FUNCTION publish_contradiction_digest_if_due(p_force BOOLEAN DEFAULT FALSE)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_state JSONB;
    v_last TIMESTAMPTZ;
    v_interval INT := LEAST(GREATEST(COALESCE(
        get_config_int('contradictions.digest_interval_seconds'), 86400
    ), 60), 604800);
    v_limit INT := LEAST(GREATEST(COALESCE(get_config_int('contradictions.digest_limit'), 10), 1), 30);
    v_cases JSONB;
    v_case JSONB;
    v_message TEXT;
    v_outbox UUID;
    v_ids UUID[] := ARRAY[]::UUID[];
BEGIN
    INSERT INTO state (key, value) VALUES ('contradiction_digest_state', '{}'::jsonb)
    ON CONFLICT (key) DO NOTHING;
    SELECT value INTO v_state FROM state WHERE key = 'contradiction_digest_state' FOR UPDATE;
    v_last := NULLIF(v_state->>'last_digest_at', '')::timestamptz;
    IF NOT COALESCE(p_force, FALSE)
       AND v_last IS NOT NULL
       AND CURRENT_TIMESTAMP < v_last + make_interval(secs => v_interval) THEN
        RETURN jsonb_build_object('skipped', TRUE, 'reason', 'not_due');
    END IF;
    SELECT list_contradiction_cases('pending', v_limit) INTO v_cases;
    IF jsonb_array_length(v_cases) = 0 THEN
        PERFORM set_state(
            'contradiction_digest_state',
            v_state || jsonb_build_object('last_digest_at', CURRENT_TIMESTAMP, 'last_count', 0)
        );
        RETURN jsonb_build_object('skipped', TRUE, 'reason', 'no_pending_cases');
    END IF;

    v_message := format('I found %s memory contradiction%s to review. I have not changed either memory.',
        jsonb_array_length(v_cases), CASE WHEN jsonb_array_length(v_cases) = 1 THEN '' ELSE 's' END);
    FOR v_case IN SELECT value FROM jsonb_array_elements(v_cases)
    LOOP
        v_ids := array_append(v_ids, (v_case->>'id')::uuid);
        v_message := v_message || format(
            E'\n\n%s — %s\nNew: %s\nOld: %s\nReply `1 %s` for new, `2 %s` for old, or `3 %s` to keep both as context-dependent.',
            v_case->>'code', v_case->>'tension',
            left(CASE WHEN v_case->>'new_memory_id' = v_case #>> '{memory_a,id}'
                      THEN v_case #>> '{memory_a,content}' ELSE v_case #>> '{memory_b,content}' END, 300),
            left(CASE WHEN v_case->>'new_memory_id' = v_case #>> '{memory_a,id}'
                      THEN v_case #>> '{memory_b,content}' ELSE v_case #>> '{memory_a,content}' END, 300),
            v_case->>'code', v_case->>'code', v_case->>'code'
        );
    END LOOP;
    v_outbox := queue_outbox_message(
        v_message,
        'contradiction_review',
        'contradiction_detection',
        jsonb_build_object('requires_response', TRUE, 'contradiction_case_ids', to_jsonb(v_ids))
    );
    UPDATE contradiction_cases
    SET proposed_at = COALESCE(proposed_at, CURRENT_TIMESTAMP),
        outbox_message_id = v_outbox,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = ANY(v_ids);
    PERFORM set_state(
        'contradiction_digest_state',
        v_state || jsonb_build_object(
            'last_digest_at', CURRENT_TIMESTAMP,
            'last_count', array_length(v_ids, 1),
            'outbox_message_id', v_outbox
        )
    );
    RETURN jsonb_build_object(
        'skipped', FALSE, 'count', array_length(v_ids, 1),
        'case_ids', to_jsonb(v_ids), 'outbox_message_id', v_outbox
    );
END;
$$;

CREATE OR REPLACE FUNCTION try_resolve_contradiction_from_inbound(
    p_channel TEXT,
    p_actor TEXT,
    p_text TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_match TEXT[];
    v_choice TEXT;
    v_code TEXT;
    v_case_id UUID;
    v_outcome TEXT;
    v_result JSONB;
BEGIN
    v_match := regexp_match(
        lower(btrim(COALESCE(p_text, ''))),
        '^(1|2|3|new|old|both|tension)(?:[[:space:]]+contradiction)?[[:space:]:#-]+([0-9a-f]{8})$'
    );
    IF v_match IS NULL THEN
        RETURN jsonb_build_object('recognized', FALSE, 'matched', FALSE);
    END IF;
    v_choice := v_match[1];
    v_code := upper(v_match[2]);
    SELECT id INTO v_case_id
    FROM contradiction_cases
    WHERE status = 'pending' AND contradiction_case_code(id) = v_code;
    IF v_case_id IS NULL THEN
        RETURN jsonb_build_object(
            'recognized', TRUE, 'matched', FALSE,
            'message', 'That contradiction code is not pending. Open the contradiction ledger for the current cases.'
        );
    END IF;
    v_outcome := CASE
        WHEN v_choice IN ('1', 'new') THEN 'new_right'
        WHEN v_choice IN ('2', 'old') THEN 'old_right'
        ELSE 'tension'
    END;
    v_result := decide_contradiction(v_case_id, v_outcome, NULL, p_channel, p_actor);
    RETURN v_result || jsonb_build_object(
        'recognized', TRUE,
        'matched', COALESCE((v_result->>'ok')::boolean, FALSE),
        'message', CASE v_outcome
            WHEN 'new_right' THEN 'Recorded: the newer memory is right. The older memory remains in history with its validity window closed.'
            WHEN 'old_right' THEN 'Recorded: the older memory is right. The newer memory remains in history with its validity window closed.'
            ELSE 'Recorded: both memories remain valid as a context-dependent tension.'
        END
    );
END;
$$;

-- Cases are the durable review source of truth. The AGE scan remains only as a
-- compatibility fallback for explicit graph relationships created by callers.
CREATE OR REPLACE FUNCTION find_contradictions(p_memory_id UUID DEFAULT NULL)
RETURNS TABLE (
    memory_a UUID,
    memory_b UUID,
    content_a TEXT,
    content_b TEXT
)
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_filter TEXT;
BEGIN
    -- The review ledger is authoritative for every detected case.
    RETURN QUERY
    SELECT c.memory_a, c.memory_b, a.content, b.content
    FROM contradiction_cases c
    JOIN memories a ON a.id = c.memory_a
    JOIN memories b ON b.id = c.memory_b
    WHERE c.status = 'pending'
      AND (p_memory_id IS NULL OR p_memory_id IN (c.memory_a, c.memory_b))
    ORDER BY c.confidence DESC, c.detected_at, c.id;

    -- Preserve the long-standing public API for callers that explicitly create
    -- a CONTRADICTS graph edge through connect_memories(). Legacy graph-only
    -- pairs remain visible, but never duplicate a pending ledger case.
    v_filter := CASE
        WHEN p_memory_id IS NULL THEN ''
        ELSE format(
            'WHERE a.memory_id = %L OR b.memory_id = %L',
            p_memory_id,
            p_memory_id
        )
    END;
    BEGIN
        RETURN QUERY EXECUTE format($query$
            WITH graph_pairs AS (
                SELECT DISTINCT
                    LEAST(
                        replace(a_id::text, '"', '')::uuid,
                        replace(b_id::text, '"', '')::uuid
                    ) AS a_uuid,
                    GREATEST(
                        replace(a_id::text, '"', '')::uuid,
                        replace(b_id::text, '"', '')::uuid
                    ) AS b_uuid
                FROM ag_catalog.cypher('memory_graph', $cypher$
                    MATCH (a:MemoryNode)-[:CONTRADICTS]-(b:MemoryNode)
                    %s
                    RETURN a.memory_id, b.memory_id
                $cypher$) AS (a_id ag_catalog.agtype, b_id ag_catalog.agtype)
            )
            SELECT p.a_uuid, p.b_uuid, a.content, b.content
            FROM graph_pairs p
            JOIN memories a ON a.id = p.a_uuid
            JOIN memories b ON b.id = p.b_uuid
            WHERE NOT EXISTS (
                SELECT 1
                FROM contradiction_cases c
                WHERE c.status = 'pending'
                  AND LEAST(c.memory_a, c.memory_b) = p.a_uuid
                  AND GREATEST(c.memory_a, c.memory_b) = p.b_uuid
            )
        $query$, v_filter);
    EXCEPTION WHEN OTHERS THEN
        RAISE DEBUG 'Legacy contradiction graph lookup unavailable: %', SQLERRM;
    END;
END;
$$;

-- A heartbeat may notice and surface a contradiction, but it may not choose
-- which human-authored belief to invalidate. Preserve every other legacy
-- action unchanged behind a narrow wrapper.
DO $rename$
BEGIN
    IF to_regprocedure('execute_heartbeat_action_legacy_contradictions(uuid,text,jsonb)') IS NULL THEN
        ALTER FUNCTION execute_heartbeat_action(UUID, TEXT, JSONB)
            RENAME TO execute_heartbeat_action_legacy_contradictions;
    END IF;
END;
$rename$;

CREATE OR REPLACE FUNCTION execute_heartbeat_action(
    p_heartbeat_id UUID,
    p_action TEXT,
    p_params JSONB DEFAULT '{}'
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_case_id UUID := _db_brain_try_uuid(p_params->>'case_id');
    v_memory_a UUID := _db_brain_try_uuid(p_params->>'memory_a');
    v_memory_b UUID := _db_brain_try_uuid(p_params->>'memory_b');
    v_filed JSONB;
BEGIN
    IF p_action NOT IN ('resolve_contradiction', 'accept_tension') THEN
        RETURN execute_heartbeat_action_legacy_contradictions(p_heartbeat_id, p_action, p_params);
    END IF;
    IF v_case_id IS NULL AND v_memory_a IS NOT NULL AND v_memory_b IS NOT NULL THEN
        SELECT id INTO v_case_id
        FROM contradiction_cases
        WHERE status = 'pending'
          AND LEAST(memory_a, memory_b) = LEAST(v_memory_a, v_memory_b)
          AND GREATEST(memory_a, memory_b) = GREATEST(v_memory_a, v_memory_b)
        LIMIT 1;
    END IF;
    IF v_case_id IS NULL AND v_memory_a IS NOT NULL AND v_memory_b IS NOT NULL THEN
        v_filed := file_contradiction_case(
            v_memory_a, v_memory_b, v_memory_b,
            COALESCE(NULLIF(p_params->>'resolution', ''), NULLIF(p_params->>'note', ''), 'Contradictory memories need a user decision.'),
            COALESCE(NULLIF(p_params->>'confidence', '')::float, 0.8),
            'heartbeat',
            jsonb_build_object('heartbeat_id', p_heartbeat_id)
        );
        v_case_id := _db_brain_try_uuid(v_filed->>'case_id');
    END IF;
    IF v_case_id IS NULL THEN
        RETURN jsonb_build_object(
            'success', FALSE,
            'error', 'No pending contradiction case was found. Detection must file a case before review.',
            'action', p_action,
            'cost', 0,
            'energy_remaining', get_current_energy()
        );
    END IF;
    UPDATE contradiction_cases
    SET proposed_at = COALESCE(proposed_at, CURRENT_TIMESTAMP),
        metadata = metadata || jsonb_build_object('noticed_by_heartbeat_id', p_heartbeat_id),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = v_case_id AND status = 'pending';
    RETURN jsonb_build_object(
        'success', TRUE,
        'action', p_action,
        'cost', 0,
        'energy_remaining', get_current_energy(),
        'result', jsonb_build_object(
            'case_id', v_case_id,
            'decision_required', TRUE,
            'changed_memories', FALSE,
            'next_step', 'Include this case in the next daily contradiction review; only the user chooses new, old, or both.'
        ),
        'external_calls', '[]'::jsonb,
        'outbox_messages', '[]'::jsonb
    );
END;
$$;

-- One bounded bootstrap batch lets an existing mind exercise the detector
-- immediately; each queued memory is compared with the whole active corpus.
INSERT INTO contradiction_detection_queue (memory_id, content_hash)
SELECT m.id, encode(digest(m.content, 'sha256'), 'hex')
FROM memories m
WHERE m.type IN ('semantic', 'worldview')
  AND m.status = 'active'
  AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
ORDER BY m.updated_at DESC, m.id
LIMIT 20
ON CONFLICT (memory_id, content_hash) DO NOTHING;
