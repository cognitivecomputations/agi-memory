-- Weekly, user-controlled learning reviews over durable memory changes.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

INSERT INTO config_defaults (key, value, description) VALUES
    ('learning.review.enabled', 'true'::jsonb,
     'Publish one reviewable learning diff when the opted-in weekly improvement pass finds enough grounded change'),
    ('learning.review.max_items', '20'::jsonb,
     'Maximum memory and skill changes included in one weekly learning review')
ON CONFLICT (key) DO NOTHING;

CREATE TABLE IF NOT EXISTS learning_reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'completed')),
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    summary TEXT NOT NULL,
    outbox_message_id UUID REFERENCES outbox_messages(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    CHECK (period_start < period_end)
);

CREATE TABLE IF NOT EXISTS learning_review_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    review_id UUID NOT NULL REFERENCES learning_reviews(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN (
        'semantic_belief', 'new_procedure', 'revised_strategy', 'proposed_skill'
    )),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'approved', 'corrected', 'forgotten')),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source_memory_id UUID REFERENCES memories(id) ON DELETE SET NULL,
    source_skill_proposal_id UUID REFERENCES skill_improvement_proposals(id) ON DELETE SET NULL,
    correction TEXT,
    correction_memory_id UUID REFERENCES memories(id) ON DELETE SET NULL,
    contradiction_case_id UUID REFERENCES contradiction_cases(id) ON DELETE SET NULL,
    decision_channel TEXT,
    decision_actor TEXT,
    decided_at TIMESTAMPTZ,
    evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (
        (kind = 'proposed_skill' AND source_skill_proposal_id IS NOT NULL AND source_memory_id IS NULL)
        OR
        (kind <> 'proposed_skill' AND source_memory_id IS NOT NULL AND source_skill_proposal_id IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_learning_reviews_status_created
    ON learning_reviews (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_learning_review_items_pending
    ON learning_review_items (review_id, created_at, id) WHERE status = 'pending';
CREATE UNIQUE INDEX IF NOT EXISTS idx_learning_review_item_memory_once
    ON learning_review_items (source_memory_id) WHERE source_memory_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_learning_review_item_skill_once
    ON learning_review_items (source_skill_proposal_id)
    WHERE source_skill_proposal_id IS NOT NULL;

CREATE OR REPLACE FUNCTION learning_review_item_code(p_id UUID)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE
STRICT
AS $$
    SELECT upper(left(replace(p_id::text, '-', ''), 8));
$$;

CREATE OR REPLACE FUNCTION create_learning_review(
    p_period_start TIMESTAMPTZ,
    p_period_end TIMESTAMPTZ,
    p_summary TEXT,
    p_memory_ids UUID[] DEFAULT '{}'::uuid[],
    p_skill_proposal_ids UUID[] DEFAULT '{}'::uuid[],
    p_metadata JSONB DEFAULT '{}'::jsonb
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_review UUID;
    v_memory memories%ROWTYPE;
    v_skill skill_improvement_proposals%ROWTYPE;
    v_id UUID;
    v_kind TEXT;
    v_item_count INT := 0;
    v_message TEXT;
    v_item RECORD;
    v_outbox UUID;
    v_max INT := LEAST(GREATEST(COALESCE(get_config_int('learning.review.max_items'), 20), 1), 50);
BEGIN
    IF NOT COALESCE(get_config_bool('learning.review.enabled'), TRUE) THEN
        RETURN jsonb_build_object('created', FALSE, 'reason', 'disabled');
    END IF;
    IF p_period_start IS NULL OR p_period_end IS NULL OR p_period_start >= p_period_end THEN
        RAISE EXCEPTION 'learning review requires a valid period';
    END IF;
    IF NULLIF(btrim(COALESCE(p_summary, '')), '') IS NULL THEN
        RAISE EXCEPTION 'learning review summary is required';
    END IF;

    INSERT INTO learning_reviews (period_start, period_end, summary, metadata)
    VALUES (p_period_start, p_period_end, btrim(p_summary), COALESCE(p_metadata, '{}'::jsonb))
    RETURNING id INTO v_review;

    FOREACH v_id IN ARRAY COALESCE(p_memory_ids, '{}'::uuid[])
    LOOP
        EXIT WHEN v_item_count >= v_max;
        SELECT * INTO v_memory
        FROM memories
        WHERE id = v_id
          AND type IN ('semantic', 'worldview', 'procedural', 'strategic')
          AND status = 'active';
        CONTINUE WHEN NOT FOUND;
        v_kind := CASE
            WHEN v_memory.type IN ('semantic', 'worldview') THEN 'semantic_belief'
            WHEN v_memory.type = 'procedural' THEN 'new_procedure'
            ELSE 'revised_strategy'
        END;
        INSERT INTO learning_review_items (
            review_id, kind, title, content, source_memory_id, evidence
        ) VALUES (
            v_review,
            v_kind,
            CASE v_kind
                WHEN 'semantic_belief' THEN 'Belief learned'
                WHEN 'new_procedure' THEN 'Procedure learned'
                ELSE 'Strategy revised'
            END,
            v_memory.content,
            v_memory.id,
            jsonb_strip_nulls(jsonb_build_object(
                'memory_type', v_memory.type,
                'created_at', v_memory.created_at,
                'updated_at', v_memory.updated_at,
                'importance', v_memory.importance,
                'trust_level', v_memory.trust_level,
                'source_attribution', v_memory.source_attribution,
                'confidence', COALESCE(
                    NULLIF(v_memory.metadata->>'confidence', '')::float,
                    NULLIF(v_memory.metadata->>'confidence_score', '')::float
                )
            ))
        ) ON CONFLICT DO NOTHING;
        IF FOUND THEN v_item_count := v_item_count + 1; END IF;
    END LOOP;

    FOREACH v_id IN ARRAY COALESCE(p_skill_proposal_ids, '{}'::uuid[])
    LOOP
        EXIT WHEN v_item_count >= v_max;
        SELECT * INTO v_skill
        FROM skill_improvement_proposals
        WHERE id = v_id AND status = 'pending';
        CONTINUE WHEN NOT FOUND;
        INSERT INTO learning_review_items (
            review_id, kind, title, content, source_skill_proposal_id, evidence
        ) VALUES (
            v_review,
            'proposed_skill',
            'Skill proposed: ' || v_skill.name,
            v_skill.description,
            v_skill.id,
            jsonb_build_object(
                'name', v_skill.name,
                'mode', v_skill.mode,
                'confidence', v_skill.confidence,
                'rationale', v_skill.rationale,
                'source_memory_ids', to_jsonb(v_skill.source_memory_ids),
                'source_unit_ids', to_jsonb(v_skill.source_unit_ids)
            )
        ) ON CONFLICT DO NOTHING;
        IF FOUND THEN v_item_count := v_item_count + 1; END IF;
    END LOOP;

    IF v_item_count = 0 THEN
        DELETE FROM learning_reviews WHERE id = v_review;
        RETURN jsonb_build_object('created', FALSE, 'reason', 'no_new_items');
    END IF;

    v_message := format(
        'Here is what I learned about you and our work this week (%s change%s). I have not silently accepted, corrected, forgotten, or applied any item.\n\n%s',
        v_item_count,
        CASE WHEN v_item_count = 1 THEN '' ELSE 's' END,
        btrim(p_summary)
    );
    FOR v_item IN
        SELECT id, kind, title, content
        FROM learning_review_items
        WHERE review_id = v_review
        ORDER BY created_at, id
    LOOP
        v_message := v_message || format(
            E'\n\n%s [%s] %s\n%s\nReply `approve %s`, `correct %s: …`, or `forget %s`.',
            learning_review_item_code(v_item.id),
            replace(v_item.kind, '_', ' '),
            v_item.title,
            left(v_item.content, 500),
            learning_review_item_code(v_item.id),
            learning_review_item_code(v_item.id),
            learning_review_item_code(v_item.id)
        );
    END LOOP;
    v_message := v_message || E'\n\nYou can also review the complete diff in Learning review.';
    v_outbox := queue_outbox_message(
        v_message,
        'learning_review',
        'weekly_learning_review',
        jsonb_build_object(
            'mode', 'web_inbox',
            'requires_response', TRUE,
            'learning_review_id', v_review,
            'review_url', '/learning-review'
        )
    );
    UPDATE learning_reviews SET outbox_message_id = v_outbox WHERE id = v_review;
    RETURN jsonb_build_object(
        'created', TRUE,
        'review_id', v_review,
        'item_count', v_item_count,
        'outbox_message_id', v_outbox
    );
END;
$$;

CREATE OR REPLACE FUNCTION list_learning_reviews(
    p_status TEXT DEFAULT 'pending',
    p_limit INT DEFAULT 20
)
RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', r.id,
        'status', r.status,
        'summary', r.summary,
        'period_start', r.period_start,
        'period_end', r.period_end,
        'created_at', r.created_at,
        'completed_at', r.completed_at,
        'items', COALESCE((
            SELECT jsonb_agg(jsonb_strip_nulls(jsonb_build_object(
                'id', i.id,
                'code', learning_review_item_code(i.id),
                'kind', i.kind,
                'status', i.status,
                'title', i.title,
                'content', i.content,
                'source_memory_id', i.source_memory_id,
                'source_skill_proposal_id', i.source_skill_proposal_id,
                'skill_proposal_status', sp.status,
                'skill_last_error', sp.last_error,
                'correction', i.correction,
                'correction_memory_id', i.correction_memory_id,
                'contradiction_case_id', i.contradiction_case_id,
                'evidence', i.evidence,
                'metadata', i.metadata,
                'decided_at', i.decided_at
            )) ORDER BY i.created_at, i.id)
            FROM learning_review_items i
            LEFT JOIN skill_improvement_proposals sp ON sp.id = i.source_skill_proposal_id
            WHERE i.review_id = r.id
        ), '[]'::jsonb)
    ) ORDER BY r.created_at DESC, r.id), '[]'::jsonb)
    FROM (
        SELECT * FROM learning_reviews
        WHERE p_status = 'all' OR status = p_status
        ORDER BY created_at DESC, id
        LIMIT LEAST(GREATEST(COALESCE(p_limit, 20), 1), 100)
    ) r;
$$;

CREATE OR REPLACE FUNCTION decide_learning_review_item(
    p_item_id UUID,
    p_action TEXT,
    p_correction TEXT DEFAULT NULL,
    p_decision_channel TEXT DEFAULT NULL,
    p_decision_actor TEXT DEFAULT NULL,
    p_confirm_load_bearing BOOLEAN DEFAULT FALSE
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_item learning_review_items%ROWTYPE;
    v_memory memories%ROWTYPE;
    v_action TEXT := lower(btrim(COALESCE(p_action, '')));
    v_correction TEXT := NULLIF(btrim(COALESCE(p_correction, '')), '');
    v_actor TEXT := COALESCE(NULLIF(btrim(COALESCE(p_decision_actor, '')), ''), 'user');
    v_new UUID;
    v_filed JSONB;
    v_decision JSONB;
    v_case UUID;
    v_supersession UUID;
    v_next_step TEXT;
BEGIN
    IF v_action NOT IN ('approve', 'correct', 'forget') THEN
        RETURN jsonb_build_object('ok', FALSE, 'error', 'invalid_action');
    END IF;
    IF v_action = 'correct' AND v_correction IS NULL THEN
        RETURN jsonb_build_object('ok', FALSE, 'error', 'correction_required');
    END IF;
    SELECT * INTO v_item
    FROM learning_review_items
    WHERE id = p_item_id
    FOR UPDATE;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('ok', FALSE, 'error', 'item_not_found');
    END IF;
    IF v_item.status <> 'pending' THEN
        RETURN jsonb_build_object(
            'ok', TRUE,
            'already_decided', TRUE,
            'item_id', v_item.id,
            'status', v_item.status
        );
    END IF;

    IF v_item.kind = 'proposed_skill' THEN
        IF v_action = 'approve' THEN
            v_next_step := 'The skill was explicitly approved and is queued for application by the maintenance worker.';
        ELSE
            PERFORM transition_skill_improvement_proposal(
                v_item.source_skill_proposal_id,
                'reject'
            );
            v_next_step := CASE
                WHEN v_action = 'correct'
                    THEN 'The original skill proposal was rejected. The correction remains in the review record as evidence for a future revised proposal.'
                ELSE 'The skill proposal was rejected and retained in review history.'
            END;
        END IF;
    ELSE
        SELECT * INTO v_memory FROM memories WHERE id = v_item.source_memory_id FOR UPDATE;
        IF NOT FOUND THEN
            RETURN jsonb_build_object('ok', FALSE, 'error', 'source_memory_not_found');
        END IF;
        IF v_action = 'forget'
           AND is_memory_protected(v_memory.id)
           AND NOT COALESCE(p_confirm_load_bearing, FALSE) THEN
            RETURN jsonb_build_object(
                'ok', FALSE,
                'error', 'load_bearing_confirmation_required',
                'confirmation_required', TRUE,
                'item_id', v_item.id,
                'message', 'This memory is protected or load-bearing. Review its evidence and confirm forgetting explicitly; nothing changed.'
            );
        END IF;
        IF v_action = 'correct' THEN
            PERFORM record_memory_correction(
                v_memory.id,
                v_correction,
                'learning_review',
                jsonb_build_object(
                    'kind', 'user_testimony',
                    'ref', 'learning-review:' || v_item.id::text,
                    'label', 'Weekly learning review correction',
                    'trust', 1.0
                ),
                TRUE
            );
            v_new := create_memory(
                v_memory.type,
                v_correction,
                v_memory.importance,
                jsonb_build_object(
                    'kind', 'user_testimony',
                    'ref', 'learning-review:' || v_item.id::text,
                    'label', 'Explicit learning review correction',
                    'observed_at', CURRENT_TIMESTAMP,
                    'trust', 1.0
                ),
                1.0,
                jsonb_build_object(
                    'learning_review_correction', jsonb_build_object(
                        'item_id', v_item.id,
                        'corrects_memory_id', v_memory.id,
                        'actor', v_actor
                    ),
                    'confidence', 1.0
                )
            );
            IF v_memory.type IN ('semantic', 'worldview') THEN
                v_filed := file_contradiction_case(
                    v_memory.id,
                    v_new,
                    v_new,
                    'The explicit weekly review correction replaces the learned belief.',
                    1.0,
                    'user_learning_review',
                    jsonb_build_object('learning_review_item_id', v_item.id)
                );
                v_case := NULLIF(v_filed->>'case_id', '')::uuid;
                IF v_case IS NULL THEN
                    RAISE EXCEPTION 'could not file correction in contradiction ledger: %', v_filed;
                END IF;
                v_decision := decide_contradiction(
                    v_case,
                    'new_right',
                    'Explicit correction from weekly learning review: ' || v_correction,
                    p_decision_channel,
                    v_actor
                );
                IF NOT COALESCE((v_decision->>'ok')::boolean, FALSE) THEN
                    RAISE EXCEPTION 'could not resolve correction contradiction: %', v_decision;
                END IF;
            ELSE
                v_supersession := record_supersession(
                    v_memory.id,
                    v_new,
                    'Explicit correction from weekly learning review: ' || v_correction,
                    v_actor,
                    'active',
                    CURRENT_TIMESTAMP,
                    NULL,
                    TRUE,
                    jsonb_build_object('learning_review_item_id', v_item.id)
                );
                UPDATE memories SET status = 'archived' WHERE id = v_memory.id;
            END IF;
            v_next_step := 'The correction is active. The prior version remains queryable in memory history.';
        ELSIF v_action = 'forget' THEN
            UPDATE memories
            SET status = 'archived',
                valid_until = COALESCE(valid_until, CURRENT_TIMESTAMP),
                metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object(
                    'forgotten_by_learning_review', jsonb_build_object(
                        'item_id', v_item.id,
                        'actor', v_actor,
                        'forgotten_at', CURRENT_TIMESTAMP
                    )
                ),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = v_memory.id;
            v_next_step := 'The learning left active recall but remains visible in point-in-time history.';
        ELSE
            v_next_step := 'The learning remains active and its evidence is unchanged.';
        END IF;
    END IF;

    UPDATE learning_review_items
    SET status = CASE v_action
            WHEN 'approve' THEN 'approved'
            WHEN 'correct' THEN 'corrected'
            ELSE 'forgotten'
        END,
        correction = CASE WHEN v_action = 'correct' THEN v_correction ELSE correction END,
        correction_memory_id = COALESCE(v_new, correction_memory_id),
        contradiction_case_id = COALESCE(v_case, contradiction_case_id),
        decision_channel = NULLIF(btrim(COALESCE(p_decision_channel, '')), ''),
        decision_actor = v_actor,
        decided_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP,
        metadata = metadata || jsonb_strip_nulls(jsonb_build_object(
            'supersession_id', v_supersession,
            'next_step', v_next_step
        ))
    WHERE id = v_item.id;

    UPDATE learning_reviews r
    SET status = 'completed', completed_at = CURRENT_TIMESTAMP
    WHERE r.id = v_item.review_id
      AND NOT EXISTS (
          SELECT 1 FROM learning_review_items i
          WHERE i.review_id = r.id AND i.status = 'pending'
      );

    RETURN jsonb_strip_nulls(jsonb_build_object(
        'ok', TRUE,
        'item_id', v_item.id,
        'review_id', v_item.review_id,
        'action', v_action,
        'status', CASE v_action
            WHEN 'approve' THEN 'approved'
            WHEN 'correct' THEN 'corrected'
            ELSE 'forgotten'
        END,
        'skill_proposal_id', v_item.source_skill_proposal_id,
        'correction_memory_id', v_new,
        'contradiction_case_id', v_case,
        'next_step', v_next_step
    ));
END;
$$;

CREATE OR REPLACE FUNCTION try_resolve_learning_review_from_inbound(
    p_channel TEXT,
    p_actor TEXT,
    p_text TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_match TEXT[];
    v_action TEXT;
    v_code TEXT;
    v_correction TEXT;
    v_item UUID;
    v_result JSONB;
BEGIN
    v_match := regexp_match(
        btrim(COALESCE(p_text, '')),
        '^(approve|forget)[[:space:]:#-]+([0-9A-Fa-f]{8})[[:space:]]*$'
    );
    IF v_match IS NULL THEN
        v_match := regexp_match(
            btrim(COALESCE(p_text, '')),
            '^correct[[:space:]:#-]+([0-9A-Fa-f]{8})[[:space:]]*:[[:space:]]*(.+)$'
        );
        IF v_match IS NULL THEN
            RETURN jsonb_build_object('recognized', FALSE, 'matched', FALSE);
        END IF;
        v_action := 'correct';
        v_code := upper(v_match[1]);
        v_correction := btrim(v_match[2]);
    ELSE
        v_action := lower(v_match[1]);
        v_code := upper(v_match[2]);
    END IF;

    SELECT id INTO v_item
    FROM learning_review_items
    WHERE status = 'pending' AND learning_review_item_code(id) = v_code;
    IF v_item IS NULL THEN
        RETURN jsonb_build_object(
            'recognized', TRUE,
            'matched', FALSE,
            'message', 'That learning-review code is not pending. Open Learning review for the current items.'
        );
    END IF;
    v_result := decide_learning_review_item(
        v_item, v_action, v_correction, p_channel, p_actor, FALSE
    );
    RETURN v_result || jsonb_build_object(
        'recognized', TRUE,
        'matched', COALESCE((v_result->>'ok')::boolean, FALSE),
        'message', COALESCE(
            v_result->>'message',
            CASE v_action
                WHEN 'approve' THEN 'Approved. ' || COALESCE(v_result->>'next_step', '')
                WHEN 'correct' THEN 'Corrected. ' || COALESCE(v_result->>'next_step', '')
                ELSE 'Forgotten from active recall. ' || COALESCE(v_result->>'next_step', '')
            END
        )
    );
END;
$$;

CREATE OR REPLACE FUNCTION claim_approved_learning_skill_application()
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_item learning_review_items%ROWTYPE;
BEGIN
    SELECT i.* INTO v_item
    FROM learning_review_items i
    JOIN skill_improvement_proposals p ON p.id = i.source_skill_proposal_id
    WHERE i.kind = 'proposed_skill'
      AND i.status = 'approved'
      AND p.status = 'pending'
      AND COALESCE(i.metadata->>'application_status', '') <> 'applied'
      AND (
          NULLIF(i.metadata->>'application_claimed_at', '') IS NULL
          OR (i.metadata->>'application_claimed_at')::timestamptz
                < CURRENT_TIMESTAMP - INTERVAL '30 minutes'
      )
      AND (
          NULLIF(i.metadata->>'application_next_attempt_at', '') IS NULL
          OR (i.metadata->>'application_next_attempt_at')::timestamptz
                <= CURRENT_TIMESTAMP
      )
    ORDER BY i.decided_at, i.id
    FOR UPDATE OF i SKIP LOCKED
    LIMIT 1;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('claimed', FALSE, 'reason', 'none_pending');
    END IF;
    UPDATE learning_review_items
    SET metadata = metadata || jsonb_build_object(
            'application_status', 'applying',
            'application_claimed_at', CURRENT_TIMESTAMP
        ),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = v_item.id;
    RETURN jsonb_build_object(
        'claimed', TRUE,
        'item_id', v_item.id,
        'proposal_id', v_item.source_skill_proposal_id
    );
END;
$$;

CREATE OR REPLACE FUNCTION finish_learning_skill_application(
    p_item_id UUID,
    p_status TEXT,
    p_error TEXT DEFAULT NULL
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_status TEXT := lower(btrim(COALESCE(p_status, '')));
    v_item learning_review_items%ROWTYPE;
BEGIN
    IF v_status NOT IN ('applied', 'failed') THEN
        RAISE EXCEPTION 'skill application status must be applied or failed';
    END IF;
    UPDATE learning_review_items
    SET metadata = metadata
            || jsonb_build_object(
                'application_status', v_status,
                'application_finished_at', CURRENT_TIMESTAMP,
                'application_claimed_at', NULL,
                'application_error', CASE
                    WHEN v_status = 'failed' THEN left(COALESCE(p_error, ''), 1000)
                    ELSE NULL
                END,
                'application_next_attempt_at', CASE
                    WHEN v_status = 'failed' THEN CURRENT_TIMESTAMP + INTERVAL '1 hour'
                    ELSE NULL
                END
            ),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_item_id
      AND kind = 'proposed_skill'
      AND status = 'approved'
    RETURNING * INTO v_item;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('ok', FALSE, 'error', 'approved_skill_item_not_found');
    END IF;
    RETURN jsonb_build_object(
        'ok', TRUE,
        'item_id', v_item.id,
        'application_status', v_status
    );
END;
$$;

COMMENT ON TABLE learning_reviews IS
    'Weekly, outbox-delivered learning diffs that remain inert until the user responds.';
COMMENT ON FUNCTION decide_learning_review_item(UUID, TEXT, TEXT, TEXT, TEXT, BOOLEAN) IS
    'Approve, correct, or forget one grounded learning item; semantic corrections resolve through the contradiction ledger and protected forgetting requires explicit reconfirmation.';
