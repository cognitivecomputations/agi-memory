-- Durable operator policy corrections for existing installations.
SET search_path = public, ag_catalog, "$user";

CREATE TABLE IF NOT EXISTS operator_policy_corrections (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    event_key TEXT NOT NULL UNIQUE CHECK (btrim(event_key) <> ''),
    policy_key TEXT NOT NULL CHECK (btrim(policy_key) <> ''),
    action TEXT NOT NULL CHECK (action IN ('set', 'reinforce', 'revoke')),
    policy_domain TEXT NOT NULL DEFAULT 'operator_standing_instruction',
    correction_kind TEXT NOT NULL DEFAULT 'standing_instruction',
    channel_type TEXT NOT NULL CHECK (btrim(channel_type) <> ''),
    channel_id TEXT,
    sender_id TEXT,
    sender_name TEXT,
    platform_message_id TEXT,
    disposition TEXT NOT NULL CHECK (disposition IN ('engage', 'observe', 'wake')),
    reason TEXT,
    directive_text TEXT NOT NULL CHECK (btrim(directive_text) <> ''),
    normalized_text_hash TEXT NOT NULL CHECK (btrim(normalized_text_hash) <> ''),
    procedural_memory_id UUID,
    improvement_backlog_id UUID,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_operator_policy_corrections_policy_created
    ON operator_policy_corrections (policy_key, created_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_operator_policy_corrections_action_created
    ON operator_policy_corrections (action, created_at DESC, id DESC);

-- Durable operator standing instructions and correction history.
SET search_path = public, ag_catalog, "$user";

INSERT INTO config_defaults (key, value, description) VALUES
    ('operator.policy_capture_enabled', 'true'::jsonb,
     'Capture explicit standing instructions from identity-verified operator turns'),
    ('operator.policy_create_review_item', 'true'::jsonb,
     'Create one deduplicated, review-gated backlog item for each active operator policy'),
    ('operator.policy_context_limit', '20'::jsonb,
     'Maximum active operator policies rendered into deterministic chat continuity')
ON CONFLICT (key) DO NOTHING;

CREATE OR REPLACE FUNCTION reject_operator_policy_correction_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'operator_policy_corrections is append-only';
END;
$$;

DROP TRIGGER IF EXISTS trg_operator_policy_corrections_immutable
    ON operator_policy_corrections;
CREATE TRIGGER trg_operator_policy_corrections_immutable
    BEFORE UPDATE OR DELETE ON operator_policy_corrections
    FOR EACH ROW
    EXECUTE FUNCTION reject_operator_policy_correction_mutation();

CREATE OR REPLACE FUNCTION normalize_operator_policy_text(p_text TEXT)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT regexp_replace(btrim(left(COALESCE(p_text, ''), 4000)), '[[:space:]]+', ' ', 'g')
$$;

CREATE OR REPLACE FUNCTION operator_policy_text_fingerprint(p_text TEXT)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT lower(regexp_replace(normalize_operator_policy_text(p_text), '[[:space:][:punct:]]+$', '', 'g'))
$$;

CREATE OR REPLACE FUNCTION classify_operator_policy_correction(
    p_channel_type TEXT,
    p_text TEXT
) RETURNS JSONB
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    directive TEXT := normalize_operator_policy_text(p_text);
    fingerprint TEXT;
    lower_text TEXT;
    has_strong_marker BOOLEAN;
    has_conditional_marker BOOLEAN;
    has_action BOOLEAN;
    is_question BOOLEAN;
    text_hash TEXT;
    policy_key TEXT;
BEGIN
    IF directive = '' THEN
        RETURN jsonb_build_object(
            'is_policy_correction', FALSE,
            'reason', 'empty_text'
        );
    END IF;

    fingerprint := operator_policy_text_fingerprint(directive);
    lower_text := lower(directive);
    has_strong_marker := lower_text ~
        '(^|[[:space:][:punct:]])(always|never|from now on|going forward|make sure|remember to|do not ever|don''t ever|must)([[:space:][:punct:]]|$)';
    has_conditional_marker := lower_text ~
        '(^|[[:space:][:punct:]])(when|whenever|every time)([[:space:][:punct:]]|$)';
    has_action := lower_text ~
        '(^|[[:space:][:punct:]])(use|prefer|route|call|avoid|send|write|read|reply|create|update|delete|notify|ask|check|analyze|extract|ingest|surface|respond|run|keep|show|tell|include|exclude|cite|confirm|verify|save|store)([[:space:][:punct:]]|$)';
    is_question := right(directive, 1) = '?';

    -- "When should I use X?" is a question, not a standing instruction.
    -- Strong language such as "Can you always use X?" remains explicit.
    IF NOT has_action
       OR NOT (has_strong_marker OR has_conditional_marker)
       OR (is_question AND NOT has_strong_marker) THEN
        RETURN jsonb_build_object(
            'is_policy_correction', FALSE,
            'reason', 'not_explicit_standing_instruction'
        );
    END IF;

    text_hash := encode(digest(convert_to(fingerprint, 'UTF8'), 'sha256'), 'hex');
    policy_key := 'operator.standing.' || left(text_hash, 24);

    RETURN jsonb_build_object(
        'is_policy_correction', TRUE,
        'policy_key', policy_key,
        'policy_domain', 'operator_standing_instruction',
        'correction_kind', 'standing_instruction',
        'canonical_directive', directive,
        'normalized_text_hash', text_hash,
        'candidate_title', left('Honor operator policy: ' || directive, 180),
        'candidate_description', left(
            'The operator issued an explicit standing instruction. Verify that current prompts, routing, and procedures honor it. Any skill change remains a separate reviewable proposal and must never be authored or activated automatically. Evidence: '
            || directive,
            2000
        ),
        'source_channel', lower(COALESCE(NULLIF(btrim(p_channel_type), ''), 'unknown'))
    );
END;
$$;

CREATE OR REPLACE FUNCTION channel_sender_is_operator(
    p_channel_type TEXT,
    p_sender_id TEXT
) RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    channel_name TEXT := lower(COALESCE(NULLIF(btrim(p_channel_type), ''), ''));
    sender TEXT := NULLIF(btrim(COALESCE(p_sender_id, '')), '');
    expected TEXT;
BEGIN
    IF channel_name = '' OR sender IS NULL THEN
        RETURN FALSE;
    END IF;

    IF channel_name = 'slack' THEN
        expected := NULLIF(btrim(get_config_text('channel.slack.operator_user_id')), '');
    ELSIF channel_name = 'imessage' THEN
        expected := NULLIF(btrim(get_config_text('channel.imessage.operator_recipient')), '');
    END IF;

    expected := COALESCE(
        expected,
        NULLIF(btrim(get_config_text('channel.' || channel_name || '.operator_user_id')), ''),
        NULLIF(btrim(get_config_text('channel.' || channel_name || '.operator_recipient')), '')
    );
    RETURN expected IS NOT NULL AND lower(expected) = lower(sender);
END;
$$;

CREATE OR REPLACE FUNCTION _ensure_operator_policy_review_item(
    p_policy_key TEXT,
    p_title TEXT,
    p_description TEXT,
    p_memory_id UUID
) RETURNS UUID
LANGUAGE plpgsql
VOLATILE
AS $$
DECLARE
    item_id UUID;
    scope_tag TEXT := 'policy:' || p_policy_key;
BEGIN
    IF NOT COALESCE(get_config_bool('operator.policy_create_review_item'), TRUE) THEN
        RETURN NULL;
    END IF;

    SELECT id INTO item_id
    FROM backlog
    WHERE status NOT IN ('done', 'cancelled')
      AND 'operator_policy_review' = ANY(tags)
      AND scope_tag = ANY(tags)
    ORDER BY created_at DESC, id
    LIMIT 1;

    IF item_id IS NOT NULL THEN
        RETURN item_id;
    END IF;

    INSERT INTO backlog (
        title, description, priority, owner, created_by, tags, checkpoint
    ) VALUES (
        p_title,
        p_description,
        'high',
        'agent',
        'user',
        ARRAY['improvement_candidate', 'operator_policy_review', scope_tag],
        jsonb_build_object(
            'improvement', jsonb_build_object(
                'source', 'operator_policy_correction',
                'policy_key', p_policy_key,
                'procedural_memory_id', p_memory_id,
                'skill_synthesis', jsonb_build_object(
                    'auto_authorized', FALSE,
                    'requires_review', TRUE
                ),
                'exit_criteria', jsonb_build_array(
                    'the standing instruction is present in deterministic continuity',
                    'focused behavior verification passes',
                    'any reusable skill change is separately reviewed by the operator'
                )
            )
        )
    )
    RETURNING id INTO item_id;

    RETURN item_id;
END;
$$;

CREATE OR REPLACE FUNCTION capture_operator_policy_correction(
    p_channel_type TEXT,
    p_channel_id TEXT,
    p_sender_id TEXT,
    p_sender_name TEXT,
    p_text TEXT,
    p_is_operator BOOLEAN,
    p_disposition TEXT DEFAULT 'engage',
    p_reason TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
AS $$
DECLARE
    classification JSONB;
    policy_key TEXT;
    directive TEXT;
    text_hash TEXT;
    message_id TEXT;
    source_event_id TEXT;
    v_event_key TEXT;
    existing operator_policy_corrections%ROWTYPE;
    memory_id UUID;
    backlog_id UUID;
    ledger_id BIGINT;
    event_action TEXT;
    normalized_channel TEXT := lower(COALESCE(NULLIF(btrim(p_channel_type), ''), 'unknown'));
    normalized_disposition TEXT := lower(COALESCE(NULLIF(btrim(p_disposition), ''), 'engage'));
BEGIN
    IF NOT COALESCE(p_is_operator, FALSE) THEN
        RETURN jsonb_build_object('captured', FALSE, 'reason', 'not_operator');
    END IF;
    IF NOT COALESCE(get_config_bool('operator.policy_capture_enabled'), TRUE) THEN
        RETURN jsonb_build_object('captured', FALSE, 'reason', 'capture_disabled');
    END IF;
    IF normalized_disposition NOT IN ('engage', 'observe', 'wake') THEN
        RETURN jsonb_build_object(
            'captured', FALSE,
            'reason', 'disposition_not_captureable'
        );
    END IF;

    classification := classify_operator_policy_correction(p_channel_type, p_text);
    IF NOT COALESCE((classification->>'is_policy_correction')::boolean, FALSE) THEN
        RETURN jsonb_build_object(
            'captured', FALSE,
            'reason', COALESCE(classification->>'reason', 'not_policy_correction')
        );
    END IF;

    policy_key := classification->>'policy_key';
    directive := classification->>'canonical_directive';
    text_hash := classification->>'normalized_text_hash';
    message_id := COALESCE(
        NULLIF(btrim(COALESCE(p_metadata->>'message_id', '')), ''),
        NULLIF(btrim(COALESCE(p_metadata->>'platform_message_id', '')), ''),
        NULLIF(btrim(COALESCE(p_metadata->>'ts', '')), '')
    );
    source_event_id := COALESCE(
        message_id,
        NULLIF(btrim(COALESCE(p_metadata->>'event_id', '')), ''),
        encode(digest(convert_to(
            lower(COALESCE(p_channel_id, '')) || '|' ||
            lower(COALESCE(p_sender_id, '')) || '|' ||
            COALESCE(p_metadata->>'session_id', '') || '|' || text_hash,
            'UTF8'
        ), 'sha256'), 'hex')
    );
    v_event_key := normalized_channel || ':' || source_event_id;

    PERFORM pg_advisory_xact_lock(hashtextextended('operator_policy:' || policy_key, 0));

    SELECT * INTO existing
    FROM operator_policy_corrections
    WHERE operator_policy_corrections.event_key = v_event_key;
    IF FOUND THEN
        RETURN jsonb_build_object(
            'captured', TRUE,
            'outcome', 'already_captured',
            'correction_id', existing.id,
            'policy_key', existing.policy_key,
            'procedural_memory_id', existing.procedural_memory_id,
            'improvement_backlog_id', existing.improvement_backlog_id
        );
    END IF;

    SELECT id INTO memory_id
    FROM memories
    WHERE type = 'procedural'
      AND status = 'active'
      AND source_attribution->>'ref' = 'operator_policy:' || policy_key
    ORDER BY created_at DESC, id
    LIMIT 1;

    IF memory_id IS NULL THEN
        event_action := 'set';
        memory_id := create_procedural_memory(
            directive,
            jsonb_build_object(
                'steps', jsonb_build_array(
                    jsonb_build_object('order', 1, 'instruction', directive)
                )
            ),
            jsonb_build_object(
                'governance', 'operator_standing_instruction',
                'skill_synthesis', 'review_required'
            ),
            0.90,
            jsonb_build_object(
                'kind', 'operator_policy_correction',
                'ref', 'operator_policy:' || policy_key,
                'label', 'Explicit operator standing instruction',
                'author', COALESCE(NULLIF(btrim(p_sender_name), ''), p_sender_id, 'operator'),
                'observed_at', CURRENT_TIMESTAMP,
                'source', 'channel:' || normalized_channel,
                'content_hash', text_hash,
                'channel_type', normalized_channel,
                'sender_external_id', p_sender_id
            ),
            1.0
        );
        UPDATE memories
        SET metadata = metadata || jsonb_build_object(
                'operator_policy', jsonb_build_object(
                    'policy_key', policy_key,
                    'active', TRUE,
                    'first_observed_at', CURRENT_TIMESTAMP,
                    'last_observed_at', CURRENT_TIMESTAMP,
                    'observation_count', 1,
                    'skill_auto_activation', FALSE
                )
            )
        WHERE id = memory_id;
    ELSE
        event_action := 'reinforce';
        UPDATE memories
        SET updated_at = CURRENT_TIMESTAMP,
            last_reinforced = CURRENT_TIMESTAMP,
            reinforcement_count = COALESCE(reinforcement_count, 0) + 1,
            metadata = jsonb_set(
                jsonb_set(
                    metadata,
                    '{operator_policy,last_observed_at}',
                    to_jsonb(CURRENT_TIMESTAMP),
                    TRUE
                ),
                '{operator_policy,observation_count}',
                to_jsonb(COALESCE((metadata#>>'{operator_policy,observation_count}')::int, 1) + 1),
                TRUE
            )
        WHERE id = memory_id;
        INSERT INTO memory_reinforcement_events (memory_id, kind, source, metadata)
        VALUES (
            memory_id,
            'operator_policy_restatement',
            'capture_operator_policy_correction',
            jsonb_build_object('policy_key', policy_key, 'event_key', v_event_key)
        );
    END IF;

    backlog_id := _ensure_operator_policy_review_item(
        policy_key,
        classification->>'candidate_title',
        classification->>'candidate_description',
        memory_id
    );

    INSERT INTO operator_policy_corrections (
        event_key, policy_key, action, policy_domain, correction_kind,
        channel_type, channel_id, sender_id, sender_name, platform_message_id,
        disposition, reason, directive_text, normalized_text_hash,
        procedural_memory_id, improvement_backlog_id, metadata
    ) VALUES (
        v_event_key,
        policy_key,
        event_action,
        classification->>'policy_domain',
        classification->>'correction_kind',
        normalized_channel,
        p_channel_id,
        p_sender_id,
        p_sender_name,
        message_id,
        normalized_disposition,
        p_reason,
        directive,
        text_hash,
        memory_id,
        backlog_id,
        COALESCE(p_metadata, '{}'::jsonb) || jsonb_build_object(
            'skill_auto_activation', FALSE,
            'review_required', TRUE
        )
    )
    RETURNING id INTO ledger_id;

    RETURN jsonb_build_object(
        'captured', TRUE,
        'outcome', CASE WHEN event_action = 'set' THEN 'created' ELSE 'reinforced' END,
        'correction_id', ledger_id,
        'policy_key', policy_key,
        'procedural_memory_id', memory_id,
        'improvement_backlog_id', backlog_id
    );
END;
$$;

CREATE OR REPLACE VIEW active_operator_policies AS
WITH latest AS (
    SELECT DISTINCT ON (policy_key)
        id,
        policy_key,
        action,
        directive_text,
        policy_domain,
        procedural_memory_id,
        improvement_backlog_id,
        created_at,
        metadata
    FROM operator_policy_corrections
    ORDER BY policy_key, created_at DESC, id DESC
)
SELECT
    latest.id AS latest_correction_id,
    latest.policy_key,
    latest.directive_text,
    latest.policy_domain,
    latest.procedural_memory_id,
    latest.improvement_backlog_id,
    latest.created_at AS last_observed_at,
    latest.metadata
FROM latest
JOIN memories m ON m.id = latest.procedural_memory_id
WHERE latest.action IN ('set', 'reinforce')
  AND m.status = 'active';

CREATE OR REPLACE FUNCTION list_operator_policies(p_limit INT DEFAULT 50)
RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    SELECT jsonb_build_object(
        'count', count(*)::int,
        'policies', COALESCE(
            jsonb_agg(
                jsonb_build_object(
                    'policy_key', policy_key,
                    'directive', directive_text,
                    'procedural_memory_id', procedural_memory_id,
                    'improvement_backlog_id', improvement_backlog_id,
                    'last_observed_at', last_observed_at
                ) ORDER BY last_observed_at DESC, policy_key
            ),
            '[]'::jsonb
        )
    )
    FROM (
        SELECT *
        FROM active_operator_policies
        ORDER BY last_observed_at DESC, policy_key
        LIMIT LEAST(GREATEST(COALESCE(p_limit, 50), 1), 200)
    ) current_policies
$$;

CREATE OR REPLACE FUNCTION revoke_operator_policy(
    p_policy_key TEXT,
    p_actor TEXT DEFAULT 'operator',
    p_reason TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
AS $$
DECLARE
    policy active_operator_policies%ROWTYPE;
    v_event_key TEXT;
    existing operator_policy_corrections%ROWTYPE;
    ledger_id BIGINT;
BEGIN
    IF NULLIF(btrim(COALESCE(p_policy_key, '')), '') IS NULL THEN
        RAISE EXCEPTION 'policy_key is required';
    END IF;
    IF NULLIF(btrim(COALESCE(p_actor, '')), '') IS NULL THEN
        RAISE EXCEPTION 'actor is required';
    END IF;

    v_event_key := 'operator-policy-revoke:' || COALESCE(
        NULLIF(btrim(COALESCE(p_metadata->>'event_id', '')), ''),
        gen_random_uuid()::text
    );

    SELECT * INTO existing
    FROM operator_policy_corrections
    WHERE operator_policy_corrections.event_key = v_event_key;
    IF FOUND THEN
        RETURN jsonb_build_object(
            'revoked', TRUE,
            'outcome', 'already_recorded',
            'correction_id', existing.id,
            'policy_key', existing.policy_key
        );
    END IF;

    PERFORM pg_advisory_xact_lock(hashtextextended('operator_policy:' || p_policy_key, 0));
    SELECT * INTO policy
    FROM active_operator_policies active
    WHERE active.policy_key = p_policy_key;
    IF NOT FOUND THEN
        RETURN jsonb_build_object(
            'revoked', FALSE,
            'reason', 'policy_not_active',
            'policy_key', p_policy_key,
            'next_step', 'List active operator policies and use an exact policy_key.'
        );
    END IF;

    UPDATE memories
    SET status = 'archived',
        valid_until = COALESCE(valid_until, CURRENT_TIMESTAMP),
        updated_at = CURRENT_TIMESTAMP,
        metadata = jsonb_set(metadata, '{operator_policy,active}', 'false'::jsonb, TRUE)
    WHERE id = policy.procedural_memory_id
      AND status = 'active';

    UPDATE backlog
    SET status = 'cancelled',
        updated_at = CURRENT_TIMESTAMP,
        checkpoint = COALESCE(checkpoint, '{}'::jsonb) || jsonb_build_object(
            'operator_policy_revoked_at', CURRENT_TIMESTAMP,
            'operator_policy_revoke_reason', COALESCE(
                NULLIF(btrim(p_reason), ''),
                'explicit operator revocation'
            )
        )
    WHERE id = policy.improvement_backlog_id
      AND status NOT IN ('done', 'cancelled');

    INSERT INTO operator_policy_corrections (
        event_key, policy_key, action, policy_domain, correction_kind,
        channel_type, sender_id, sender_name, disposition, reason,
        directive_text, normalized_text_hash, procedural_memory_id,
        improvement_backlog_id, metadata
    ) VALUES (
        v_event_key,
        policy.policy_key,
        'revoke',
        policy.policy_domain,
        'explicit_revoke',
        'operator_control',
        p_actor,
        p_actor,
        'engage',
        COALESCE(NULLIF(btrim(p_reason), ''), 'explicit operator revocation'),
        policy.directive_text,
        encode(digest(convert_to(operator_policy_text_fingerprint(policy.directive_text), 'UTF8'), 'sha256'), 'hex'),
        policy.procedural_memory_id,
        policy.improvement_backlog_id,
        COALESCE(p_metadata, '{}'::jsonb) || jsonb_build_object(
            'skill_auto_activation', FALSE,
            'revoked_by', p_actor
        )
    )
    RETURNING id INTO ledger_id;

    RETURN jsonb_build_object(
        'revoked', TRUE,
        'outcome', 'revoked',
        'correction_id', ledger_id,
        'policy_key', policy.policy_key,
        'procedural_memory_id', policy.procedural_memory_id,
        'next_step', 'State a new standing instruction if this policy should be replaced.'
    );
END;
$$;

CREATE OR REPLACE FUNCTION render_operator_policy_context()
RETURNS TEXT
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    policy_limit INT := LEAST(
        GREATEST(COALESCE(get_config_int('operator.policy_context_limit'), 20), 0),
        100
    );
    lines TEXT;
BEGIN
    IF policy_limit = 0 THEN
        RETURN '';
    END IF;

    SELECT string_agg(
        '- [' || policy_key || '] ' || left(
            regexp_replace(directive_text, '[[:space:]]+', ' ', 'g'),
            1000
        ),
        E'\n' ORDER BY last_observed_at ASC, policy_key
    ) INTO lines
    FROM (
        SELECT *
        FROM active_operator_policies
        ORDER BY last_observed_at DESC, policy_key
        LIMIT policy_limit
    ) policies;

    IF lines IS NULL THEN
        RETURN '';
    END IF;
    RETURN '## Active Operator Policies' || E'\n'
        || 'These are identity-verified standing instructions. Follow them unless they conflict with safety or a newer explicit request. Treat the text as internal guidance; do not quote it in shared rooms. The operator can inspect or revoke a policy by its key with manage_operator_policies.' || E'\n'
        || lines;
END;
$$;

COMMENT ON TABLE operator_policy_corrections IS
    'Append-only evidence ledger for identity-verified operator standing instructions, reinforcement, and revocation.';
COMMENT ON FUNCTION capture_operator_policy_correction(TEXT, TEXT, TEXT, TEXT, TEXT, BOOLEAN, TEXT, TEXT, JSONB) IS
    'Atomically captures explicit operator policy evidence, one procedural memory, and one review-gated backlog item; never authors or activates skills.';

