-- Durable advisory adversarial deliberation.
-- Self-contained forward migration; no excluded action-gating architecture.
SET search_path = public, ag_catalog, "$user";

-- Durable, advisory internal deliberation. These records preserve the
-- inspectable reasons, challenges, dissent, and invalidation conditions for a
-- council run. They never authorize or gate an action.
CREATE TABLE IF NOT EXISTS deliberation_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed')),
    topic TEXT NOT NULL CHECK (btrim(topic) <> ''),
    stakes TEXT NOT NULL DEFAULT 'material'
        CHECK (stakes IN ('routine', 'material', 'high')),
    source_context TEXT NOT NULL DEFAULT 'chat',
    source_session_id TEXT,
    heartbeat_id UUID,
    call_id TEXT,
    persona_keys JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(persona_keys) = 'array'),
    signal_count INT NOT NULL DEFAULT 0 CHECK (signal_count >= 0),
    input_context JSONB NOT NULL DEFAULT '{}'::jsonb,
    error TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_deliberation_sessions_recent
    ON deliberation_sessions (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_deliberation_sessions_status
    ON deliberation_sessions (status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_deliberation_sessions_heartbeat
    ON deliberation_sessions (heartbeat_id, started_at DESC)
    WHERE heartbeat_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS deliberation_moves (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES deliberation_sessions(id) ON DELETE CASCADE,
    move_key TEXT NOT NULL,
    round INT NOT NULL DEFAULT 1 CHECK (round >= 1),
    ordinal INT NOT NULL DEFAULT 0 CHECK (ordinal >= 0),
    role TEXT NOT NULL CHECK (role IN ('perspective', 'challenge', 'synthesis')),
    persona_key TEXT,
    content TEXT NOT NULL CHECK (btrim(content) <> ''),
    target_move_id UUID REFERENCES deliberation_moves(id) ON DELETE SET NULL,
    evidence_memory_ids UUID[] NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (session_id, move_key)
);

CREATE INDEX IF NOT EXISTS idx_deliberation_moves_session
    ON deliberation_moves (session_id, round, ordinal, created_at);

CREATE TABLE IF NOT EXISTS deliberation_verdicts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL UNIQUE
        REFERENCES deliberation_sessions(id) ON DELETE CASCADE,
    recommendation TEXT NOT NULL CHECK (btrim(recommendation) <> ''),
    report TEXT NOT NULL CHECK (btrim(report) <> ''),
    agreements JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(agreements) = 'array'),
    disagreements JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(disagreements) = 'array'),
    risks JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(risks) = 'array'),
    missing_evidence JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(missing_evidence) = 'array'),
    dissent JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(dissent) = 'array'),
    invalidation_conditions JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(invalidation_conditions) = 'array'),
    evidence_memory_ids UUID[] NOT NULL DEFAULT '{}',
    summary_memory_id UUID,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Keep the forward delta safe if an earlier development baseline created the
-- table before this migration was recorded.
ALTER TABLE deliberation_verdicts
    ADD COLUMN IF NOT EXISTS missing_evidence JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(missing_evidence) = 'array');

CREATE INDEX IF NOT EXISTS idx_deliberation_verdicts_recent
    ON deliberation_verdicts (created_at DESC);

-- Durable, advisory adversarial deliberation.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

INSERT INTO config_defaults (key, value, description) VALUES
    ('deliberation.max_personas', '5'::jsonb,
     'Maximum council perspectives in one deliberation'),
    ('deliberation.signal_limit', '10'::jsonb,
     'Default maximum compact evidence signals supplied to deliberation'),
    ('deliberation.max_topic_chars', '2000'::jsonb,
     'Maximum topic length accepted by a deliberation'),
    ('deliberation.max_context_chars', '8000'::jsonb,
     'Maximum additional context length accepted by a deliberation'),
    ('deliberation.perspective_max_tokens', '700'::jsonb,
     'Maximum output tokens for each council perspective'),
    ('deliberation.challenge_max_tokens', '900'::jsonb,
     'Maximum output tokens for the adversarial challenge pass'),
    ('deliberation.synthesis_max_tokens', '900'::jsonb,
     'Maximum output tokens for the deliberation synthesis pass'),
    ('deliberation.create_summary_memory', 'true'::jsonb,
     'Create one concise episodic memory for a grounded completed deliberation')
ON CONFLICT (key) DO NOTHING;

CREATE OR REPLACE FUNCTION get_deliberation_config()
RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    SELECT jsonb_build_object(
        'max_personas', GREATEST(COALESCE(get_config_int('deliberation.max_personas'), 5), 1),
        'signal_limit', GREATEST(COALESCE(get_config_int('deliberation.signal_limit'), 10), 0),
        'max_topic_chars', GREATEST(COALESCE(get_config_int('deliberation.max_topic_chars'), 2000), 1),
        'max_context_chars', GREATEST(COALESCE(get_config_int('deliberation.max_context_chars'), 8000), 0),
        'perspective_max_tokens', GREATEST(COALESCE(get_config_int('deliberation.perspective_max_tokens'), 700), 128),
        'challenge_max_tokens', GREATEST(COALESCE(get_config_int('deliberation.challenge_max_tokens'), 900), 128),
        'synthesis_max_tokens', GREATEST(COALESCE(get_config_int('deliberation.synthesis_max_tokens'), 900), 128),
        'create_summary_memory', COALESCE(get_config_bool('deliberation.create_summary_memory'), TRUE)
    )
$$;

CREATE OR REPLACE FUNCTION begin_deliberation(
    p_topic TEXT,
    p_stakes TEXT DEFAULT 'material',
    p_source_context TEXT DEFAULT 'chat',
    p_source_session_id TEXT DEFAULT NULL,
    p_heartbeat_id UUID DEFAULT NULL,
    p_call_id TEXT DEFAULT NULL,
    p_persona_keys JSONB DEFAULT '[]'::jsonb,
    p_signal_count INT DEFAULT 0,
    p_input_context JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    config JSONB := get_deliberation_config();
    topic TEXT := btrim(COALESCE(p_topic, ''));
    stakes TEXT := lower(btrim(COALESCE(p_stakes, 'material')));
    personas JSONB := COALESCE(p_persona_keys, '[]'::jsonb);
    created deliberation_sessions%ROWTYPE;
BEGIN
    IF topic = '' THEN
        RAISE EXCEPTION 'deliberation topic must not be blank';
    END IF;
    IF length(topic) > (config->>'max_topic_chars')::int THEN
        RAISE EXCEPTION 'deliberation topic exceeds % characters',
            (config->>'max_topic_chars')::int;
    END IF;
    IF stakes NOT IN ('routine', 'material', 'high') THEN
        RAISE EXCEPTION 'deliberation stakes must be routine, material, or high';
    END IF;
    IF jsonb_typeof(personas) <> 'array' OR jsonb_array_length(personas) = 0 THEN
        RAISE EXCEPTION 'deliberation requires at least one council persona';
    END IF;
    IF jsonb_array_length(personas) > (config->>'max_personas')::int THEN
        RAISE EXCEPTION 'deliberation exceeds the live maximum of % personas',
            (config->>'max_personas')::int;
    END IF;
    IF EXISTS (
        SELECT 1
        FROM jsonb_array_elements(personas) AS item(value)
        WHERE jsonb_typeof(item.value) <> 'string'
           OR btrim(item.value #>> '{}') = ''
    ) THEN
        RAISE EXCEPTION 'deliberation persona keys must be non-blank strings';
    END IF;
    IF (
        SELECT COUNT(DISTINCT item.value #>> '{}')
        FROM jsonb_array_elements(personas) AS item(value)
    ) <> jsonb_array_length(personas) THEN
        RAISE EXCEPTION 'deliberation persona keys must be unique';
    END IF;
    IF GREATEST(COALESCE(p_signal_count, 0), 0)
       > (config->>'signal_limit')::int THEN
        RAISE EXCEPTION 'deliberation exceeds the live maximum of % evidence signals',
            (config->>'signal_limit')::int;
    END IF;
    IF jsonb_typeof(COALESCE(p_input_context, '{}'::jsonb)) = 'object'
       AND p_input_context ? 'signals'
       AND jsonb_typeof(p_input_context->'signals') <> 'array' THEN
        RAISE EXCEPTION 'deliberation input signals must be an array';
    END IF;
    IF jsonb_typeof(COALESCE(p_input_context, '{}'::jsonb)) = 'object'
       AND jsonb_typeof(p_input_context->'signals') = 'array'
       AND jsonb_array_length(p_input_context->'signals')
           > (config->>'signal_limit')::int THEN
        RAISE EXCEPTION 'deliberation exceeds the live maximum of % evidence signals',
            (config->>'signal_limit')::int;
    END IF;
    IF jsonb_typeof(COALESCE(p_input_context, '{}'::jsonb)) = 'object'
       AND length(COALESCE(p_input_context->>'additional_context', ''))
           > (config->>'max_context_chars')::int THEN
        RAISE EXCEPTION 'deliberation context exceeds % characters',
            (config->>'max_context_chars')::int;
    END IF;

    INSERT INTO deliberation_sessions (
        topic, stakes, source_context, source_session_id, heartbeat_id,
        call_id, persona_keys, signal_count, input_context
    ) VALUES (
        topic,
        stakes,
        COALESCE(NULLIF(btrim(p_source_context), ''), 'chat'),
        NULLIF(btrim(COALESCE(p_source_session_id, '')), ''),
        p_heartbeat_id,
        NULLIF(btrim(COALESCE(p_call_id, '')), ''),
        personas,
        GREATEST(COALESCE(p_signal_count, 0), 0),
        CASE WHEN jsonb_typeof(COALESCE(p_input_context, '{}'::jsonb)) = 'object'
             THEN COALESCE(p_input_context, '{}'::jsonb)
             ELSE '{}'::jsonb END
    )
    RETURNING * INTO created;

    RETURN to_jsonb(created);
END;
$$;

CREATE OR REPLACE FUNCTION record_deliberation_move(
    p_session_id UUID,
    p_move_key TEXT,
    p_role TEXT,
    p_content TEXT,
    p_round INT DEFAULT 1,
    p_ordinal INT DEFAULT 0,
    p_persona_key TEXT DEFAULT NULL,
    p_target_move_id UUID DEFAULT NULL,
    p_evidence_memory_ids UUID[] DEFAULT '{}',
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    move_id UUID;
    session_status TEXT;
    role TEXT := lower(btrim(COALESCE(p_role, '')));
BEGIN
    SELECT status INTO session_status
    FROM deliberation_sessions
    WHERE id = p_session_id
    FOR UPDATE;
    IF session_status IS NULL THEN
        RAISE EXCEPTION 'deliberation session % was not found', p_session_id;
    END IF;
    IF session_status <> 'running' THEN
        RAISE EXCEPTION 'deliberation session % is %, not running',
            p_session_id, session_status;
    END IF;
    IF role NOT IN ('perspective', 'challenge', 'synthesis') THEN
        RAISE EXCEPTION 'unsupported deliberation move role: %', role;
    END IF;
    IF NULLIF(btrim(COALESCE(p_move_key, '')), '') IS NULL THEN
        RAISE EXCEPTION 'deliberation move key must not be blank';
    END IF;
    IF NULLIF(btrim(COALESCE(p_content, '')), '') IS NULL THEN
        RAISE EXCEPTION 'deliberation move content must not be blank';
    END IF;

    INSERT INTO deliberation_moves (
        session_id, move_key, round, ordinal, role, persona_key, content,
        target_move_id, evidence_memory_ids, metadata
    ) VALUES (
        p_session_id,
        btrim(p_move_key),
        GREATEST(COALESCE(p_round, 1), 1),
        GREATEST(COALESCE(p_ordinal, 0), 0),
        role,
        NULLIF(btrim(COALESCE(p_persona_key, '')), ''),
        btrim(p_content),
        p_target_move_id,
        COALESCE(p_evidence_memory_ids, '{}'),
        CASE WHEN jsonb_typeof(COALESCE(p_metadata, '{}'::jsonb)) = 'object'
             THEN COALESCE(p_metadata, '{}'::jsonb)
             ELSE '{}'::jsonb END
    )
    ON CONFLICT (session_id, move_key) DO NOTHING
    RETURNING id INTO move_id;

    IF move_id IS NULL THEN
        SELECT id INTO move_id
        FROM deliberation_moves
        WHERE session_id = p_session_id AND move_key = btrim(p_move_key);
    END IF;
    RETURN move_id;
END;
$$;

CREATE OR REPLACE FUNCTION complete_deliberation(
    p_session_id UUID,
    p_recommendation TEXT,
    p_report TEXT,
    p_agreements JSONB DEFAULT '[]'::jsonb,
    p_disagreements JSONB DEFAULT '[]'::jsonb,
    p_risks JSONB DEFAULT '[]'::jsonb,
    p_missing_evidence JSONB DEFAULT '[]'::jsonb,
    p_dissent JSONB DEFAULT '[]'::jsonb,
    p_invalidation_conditions JSONB DEFAULT '[]'::jsonb,
    p_evidence_memory_ids UUID[] DEFAULT '{}',
    p_create_summary_memory BOOLEAN DEFAULT TRUE,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    session_row deliberation_sessions%ROWTYPE;
    verdict_row deliberation_verdicts%ROWTYPE;
    recommendation TEXT := btrim(COALESCE(p_recommendation, ''));
    report TEXT := btrim(COALESCE(p_report, ''));
    agreements JSONB;
    disagreements JSONB;
    risks JSONB;
    missing_evidence JSONB;
    dissent JSONB;
    invalidation_conditions JSONB;
    evidence_ids UUID[];
    summary_id UUID;
BEGIN
    SELECT * INTO session_row
    FROM deliberation_sessions
    WHERE id = p_session_id
    FOR UPDATE;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('applied', FALSE, 'reason', 'unknown_session');
    END IF;
    IF session_row.status <> 'running' THEN
        RETURN COALESCE(
            (SELECT to_jsonb(v) FROM deliberation_verdicts v
             WHERE v.session_id = p_session_id),
            '{}'::jsonb
        ) || jsonb_build_object(
            'applied', FALSE,
            'reason', 'already_' || session_row.status
        );
    END IF;
    IF recommendation = '' OR report = '' THEN
        RAISE EXCEPTION 'deliberation completion requires a recommendation and report';
    END IF;

    agreements := CASE WHEN jsonb_typeof(COALESCE(p_agreements, '[]'::jsonb)) = 'array'
                       THEN COALESCE(p_agreements, '[]'::jsonb) ELSE '[]'::jsonb END;
    disagreements := CASE WHEN jsonb_typeof(COALESCE(p_disagreements, '[]'::jsonb)) = 'array'
                          THEN COALESCE(p_disagreements, '[]'::jsonb) ELSE '[]'::jsonb END;
    risks := CASE WHEN jsonb_typeof(COALESCE(p_risks, '[]'::jsonb)) = 'array'
                  THEN COALESCE(p_risks, '[]'::jsonb) ELSE '[]'::jsonb END;
    missing_evidence := CASE
        WHEN jsonb_typeof(COALESCE(p_missing_evidence, '[]'::jsonb)) = 'array'
        THEN COALESCE(p_missing_evidence, '[]'::jsonb)
        ELSE '[]'::jsonb END;
    dissent := CASE WHEN jsonb_typeof(COALESCE(p_dissent, '[]'::jsonb)) = 'array'
                    THEN COALESCE(p_dissent, '[]'::jsonb) ELSE '[]'::jsonb END;
    invalidation_conditions := CASE
        WHEN jsonb_typeof(COALESCE(p_invalidation_conditions, '[]'::jsonb)) = 'array'
        THEN COALESCE(p_invalidation_conditions, '[]'::jsonb)
        ELSE '[]'::jsonb END;

    SELECT COALESCE(array_agg(m.id ORDER BY m.id), '{}')
    INTO evidence_ids
    FROM memories m
    WHERE m.id = ANY(COALESCE(p_evidence_memory_ids, '{}'))
      AND m.status = 'active';

    PERFORM record_deliberation_move(
        p_session_id,
        'synthesis',
        'synthesis',
        report,
        3,
        0,
        'moderator',
        NULL,
        evidence_ids,
        jsonb_build_object('recommendation', recommendation)
    );

    IF COALESCE(p_create_summary_memory, TRUE) THEN
        summary_id := create_episodic_memory(
            p_content := format(
                'Council deliberation on "%s": %s',
                left(session_row.topic, 300),
                left(report, 1500)
            ),
            p_action_taken := jsonb_build_object(
                'action', 'adversarial_deliberation',
                'deliberation_session_id', p_session_id
            ),
            p_context := jsonb_build_object(
                'topic', session_row.topic,
                'stakes', session_row.stakes,
                'source_context', session_row.source_context,
                'heartbeat_id', session_row.heartbeat_id
            ),
            p_result := jsonb_build_object(
                'recommendation', recommendation,
                'missing_evidence', missing_evidence,
                'dissent', dissent,
                'invalidation_conditions', invalidation_conditions
            ),
            p_emotional_valence := 0.0,
            p_event_time := CURRENT_TIMESTAMP,
            p_importance := CASE session_row.stakes
                WHEN 'high' THEN 0.8
                WHEN 'material' THEN 0.65
                ELSE 0.5
            END,
            p_source_attribution := jsonb_build_object(
                'kind', 'internal',
                'ref', 'deliberation:' || p_session_id::text,
                'label', 'Internal council deliberation',
                'observed_at', CURRENT_TIMESTAMP
            ),
            p_trust_level := 0.8
        );
    END IF;

    INSERT INTO deliberation_verdicts (
        session_id, recommendation, report, agreements, disagreements,
        risks, missing_evidence, dissent, invalidation_conditions, evidence_memory_ids,
        summary_memory_id, metadata
    ) VALUES (
        p_session_id, recommendation, report, agreements, disagreements,
        risks, missing_evidence, dissent, invalidation_conditions, evidence_ids,
        summary_id,
        CASE WHEN jsonb_typeof(COALESCE(p_metadata, '{}'::jsonb)) = 'object'
             THEN COALESCE(p_metadata, '{}'::jsonb)
             ELSE '{}'::jsonb END
    )
    RETURNING * INTO verdict_row;

    UPDATE deliberation_sessions
    SET status = 'completed',
        completed_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_session_id;

    RETURN to_jsonb(verdict_row)
        || jsonb_build_object(
            'applied', TRUE,
            'memory_id', summary_id,
            'deliberation_id', p_session_id
        );
END;
$$;

CREATE OR REPLACE FUNCTION fail_deliberation(
    p_session_id UUID,
    p_error TEXT
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    changed INT;
BEGIN
    UPDATE deliberation_sessions
    SET status = 'failed',
        error = left(COALESCE(NULLIF(btrim(p_error), ''), 'Deliberation failed'), 2000),
        completed_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_session_id AND status = 'running';
    GET DIAGNOSTICS changed = ROW_COUNT;
    RETURN jsonb_build_object(
        'applied', changed > 0,
        'deliberation_id', p_session_id,
        'status', CASE WHEN changed > 0 THEN 'failed' ELSE 'unchanged' END
    );
END;
$$;

CREATE OR REPLACE FUNCTION list_deliberations(
    p_limit INT DEFAULT 20,
    p_status TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    WITH selected AS (
        SELECT
            s.id,
            s.status,
            s.topic,
            s.stakes,
            s.source_context,
            s.source_session_id,
            s.heartbeat_id,
            s.persona_keys,
            s.signal_count,
            s.started_at,
            s.completed_at,
            s.error,
            v.recommendation,
            CASE
                WHEN jsonb_typeof(v.metadata->'degraded') = 'boolean'
                THEN (v.metadata->>'degraded')::boolean
                ELSE FALSE
            END AS degraded,
            v.summary_memory_id
        FROM deliberation_sessions s
        LEFT JOIN deliberation_verdicts v ON v.session_id = s.id
        WHERE p_status IS NULL OR s.status = lower(btrim(p_status))
        ORDER BY s.started_at DESC
        LIMIT LEAST(GREATEST(COALESCE(p_limit, 20), 1), 100)
    )
    SELECT jsonb_build_object(
        'count', COUNT(*),
        'items', COALESCE(jsonb_agg(to_jsonb(selected) ORDER BY started_at DESC), '[]'::jsonb)
    )
    FROM selected
$$;

CREATE OR REPLACE FUNCTION inspect_deliberation(p_session_id UUID)
RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    SELECT CASE WHEN s.id IS NULL THEN
        jsonb_build_object('found', FALSE, 'deliberation_id', p_session_id)
    ELSE
        jsonb_build_object(
            'found', TRUE,
            'session', to_jsonb(s),
            'moves', COALESCE(
                (SELECT jsonb_agg(to_jsonb(m) ORDER BY m.round, m.ordinal, m.created_at)
                 FROM deliberation_moves m WHERE m.session_id = s.id),
                '[]'::jsonb
            ),
            'verdict',
                (SELECT to_jsonb(v) FROM deliberation_verdicts v WHERE v.session_id = s.id)
        )
    END
    FROM (SELECT 1) anchor
    LEFT JOIN deliberation_sessions s ON s.id = p_session_id
$$;

SET check_function_bodies = on;
