-- Inline-appraisal pushdown (plans/db_pushdown.md 3.6).
-- The subconscious appraisal's pre-LLM context gathering (five round-trips)
-- becomes one call, and the post-LLM normalization (confidence thresholds,
-- clamps, memory-id allow-listing) becomes SQL with config-owned knobs.
-- The LLM call itself, and the clipping of already-in-process memories,
-- stay in Python.
SET search_path = public, ag_catalog, "$user";

INSERT INTO config_defaults (key, value, description) VALUES
    ('subconscious.min_signal_confidence', '0.6'::jsonb,
     'Appraisal signals below this confidence are dropped during normalization'),
    ('subconscious.response_max_chars', '500'::jsonb,
     'Gut-reaction text cap applied during appraisal normalization')
ON CONFLICT (key) DO NOTHING;

INSERT INTO config_defaults (key, value, description) VALUES
    ('subconscious.appraisal_memory_limit', '10'::jsonb,
     'How many hydrated memories ride into the inline appraisal payload'),
    ('subconscious.appraisal_memory_chars', '1200'::jsonb,
     'Per-memory content clip inside the appraisal payload'),
    ('subconscious.appraisal_context_chars', '4000'::jsonb,
     'Total memory-content budget inside the appraisal payload'),
    ('subconscious.appraisal_total_chars', '7000'::jsonb,
     'Whole-payload serialization budget for the appraisal call'),
    ('subconscious.appraisal_skim_memory_limit', '4'::jsonb,
     'Memory count for low-salience skim appraisal'),
    ('subconscious.appraisal_skim_context_chars', '1800'::jsonb,
     'Context budget for low-salience skim appraisal'),
    ('subconscious.appraisal_skim_total_chars', '3200'::jsonb,
     'Whole-payload budget for low-salience skim appraisal'),
    ('subconscious.appraisal_deep_memory_limit', '16'::jsonb,
     'Memory count for high-salience deep appraisal'),
    ('subconscious.appraisal_deep_context_chars', '7000'::jsonb,
     'Context budget for high-salience deep appraisal'),
    ('subconscious.appraisal_deep_total_chars', '11000'::jsonb,
     'Whole-payload budget for high-salience deep appraisal'),
    ('subconscious.appraisal_deep_max_tokens', '2400'::jsonb,
     'Maximum JSON-mode output tokens for high-salience appraisal'),
    ('subconscious.appraisal_normal_max_tokens', '1800'::jsonb,
     'Maximum JSON-mode output tokens for normal appraisal'),
    ('subconscious.appraisal_skim_max_tokens', '900'::jsonb,
     'Maximum JSON-mode output tokens for skim appraisal')
ON CONFLICT (key) DO NOTHING;

CREATE OR REPLACE FUNCTION appraisal_depth_for_stimulus(
    p_message TEXT,
    p_context JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB AS $$
DECLARE
    msg TEXT := lower(COALESCE(p_message, ''));
    salience FLOAT := 0.0;
    depth TEXT := 'normal';
    limits JSONB;
BEGIN
    salience := salience
        + CASE WHEN length(msg) >= 1200 THEN 0.20 WHEN length(msg) >= 400 THEN 0.10 ELSE 0.0 END
        + CASE WHEN msg LIKE '%?%' THEN 0.05 ELSE 0.0 END
        + CASE WHEN msg ~ '(urgent|emergency|asap|deadline|critical|important|blocked|stuck)' THEN 0.30 ELSE 0.0 END
        + CASE WHEN msg ~ '(hurt|danger|unsafe|crash|hospital|lawyer|legal|doctor|medical|suicide|self-harm)' THEN 0.45 ELSE 0.0 END
        + CASE WHEN msg ~ '(remember|forget|consent|permission|private|password|secret|delete|wipe|replace)' THEN 0.20 ELSE 0.0 END
        + CASE WHEN COALESCE((p_context->>'attachment_count')::int, 0) > 0 THEN 0.15 ELSE 0.0 END;

    salience := LEAST(1.0, GREATEST(0.0, salience));
    IF salience >= 0.65 THEN
        depth := 'deep';
    ELSIF salience < 0.20 THEN
        depth := 'skim';
    END IF;

    limits := CASE depth
        WHEN 'deep' THEN jsonb_build_object(
            'memory_limit', COALESCE(get_config_int('subconscious.appraisal_deep_memory_limit'), 16),
            'memory_chars', COALESCE(get_config_int('subconscious.appraisal_memory_chars'), 1200),
            'context_chars', COALESCE(get_config_int('subconscious.appraisal_deep_context_chars'), 7000),
            'total_chars', COALESCE(get_config_int('subconscious.appraisal_deep_total_chars'), 11000),
            'max_tokens', COALESCE(get_config_int('subconscious.appraisal_deep_max_tokens'), 2400))
        WHEN 'skim' THEN jsonb_build_object(
            'memory_limit', COALESCE(get_config_int('subconscious.appraisal_skim_memory_limit'), 4),
            'memory_chars', LEAST(COALESCE(get_config_int('subconscious.appraisal_memory_chars'), 1200), 700),
            'context_chars', COALESCE(get_config_int('subconscious.appraisal_skim_context_chars'), 1800),
            'total_chars', COALESCE(get_config_int('subconscious.appraisal_skim_total_chars'), 3200),
            'max_tokens', COALESCE(get_config_int('subconscious.appraisal_skim_max_tokens'), 900))
        ELSE jsonb_build_object(
            'memory_limit', COALESCE(get_config_int('subconscious.appraisal_memory_limit'), 10),
            'memory_chars', COALESCE(get_config_int('subconscious.appraisal_memory_chars'), 1200),
            'context_chars', COALESCE(get_config_int('subconscious.appraisal_context_chars'), 4000),
            'total_chars', COALESCE(get_config_int('subconscious.appraisal_total_chars'), 7000),
            'max_tokens', COALESCE(get_config_int('subconscious.appraisal_normal_max_tokens'), 1800))
    END;

    RETURN jsonb_build_object(
        'depth', depth,
        'salience', salience,
        'limits', limits
    );
EXCEPTION WHEN OTHERS THEN
    RETURN jsonb_build_object(
        'depth', 'normal',
        'salience', 0.5,
        'limits', jsonb_build_object(
            'memory_limit', COALESCE(get_config_int('subconscious.appraisal_memory_limit'), 10),
            'memory_chars', COALESCE(get_config_int('subconscious.appraisal_memory_chars'), 1200),
            'context_chars', COALESCE(get_config_int('subconscious.appraisal_context_chars'), 4000),
            'total_chars', COALESCE(get_config_int('subconscious.appraisal_total_chars'), 7000),
            'max_tokens', COALESCE(get_config_int('subconscious.appraisal_normal_max_tokens'), 1800))
    );
END;
$$ LANGUAGE plpgsql STABLE;

-- One round-trip for everything the appraisal payload needs from the DB.
CREATE OR REPLACE FUNCTION get_appraisal_db_context()
RETURNS JSONB AS $$
DECLARE
    turn_ctx JSONB;
BEGIN
    turn_ctx := gather_turn_context();
    RETURN jsonb_strip_nulls(jsonb_build_object(
        'identity', COALESCE((
            SELECT jsonb_agg(x) FROM (
                SELECT x FROM jsonb_array_elements(COALESCE(turn_ctx->'identity', '[]'::jsonb)) x LIMIT 5
            ) t), '[]'::jsonb),
        'worldview', COALESCE((
            SELECT jsonb_agg(x) FROM (
                SELECT x FROM jsonb_array_elements(COALESCE(turn_ctx->'worldview', '[]'::jsonb)) x LIMIT 5
            ) t), '[]'::jsonb),
        'emotional_state', NULLIF(get_current_affective_state(), '{}'::jsonb),
        'goals', NULLIF(CASE WHEN jsonb_typeof(turn_ctx->'goals') = 'object'
                             THEN turn_ctx->'goals' ELSE '{}'::jsonb END, '{}'::jsonb),
        'relationships', NULLIF(get_relationships_context(8), '[]'::jsonb),
        'dopamine_state', NULLIF(get_dopamine_state(), '{}'::jsonb),
        'emotion_families', COALESCE(get_config('emotion.families'), '{}'::jsonb),
        'limits', jsonb_build_object(
            'memory_limit', COALESCE(get_config_int('subconscious.appraisal_memory_limit'), 10),
            'memory_chars', COALESCE(get_config_int('subconscious.appraisal_memory_chars'), 1200),
            'context_chars', COALESCE(get_config_int('subconscious.appraisal_context_chars'), 4000),
            'total_chars', COALESCE(get_config_int('subconscious.appraisal_total_chars'), 7000))
    ));
END;
$$ LANGUAGE plpgsql;

-- Passthrough lists keep only object items (parity with the Python parser).
CREATE OR REPLACE FUNCTION _appraisal_dict_items(p_val JSONB)
RETURNS JSONB AS $$
    SELECT COALESCE(jsonb_agg(x), '[]'::jsonb)
    FROM jsonb_array_elements(CASE WHEN jsonb_typeof(p_val) = 'array' THEN p_val ELSE '[]'::jsonb END) x
    WHERE jsonb_typeof(x) = 'object';
$$ LANGUAGE sql IMMUTABLE;

-- Post-LLM normalization: everything _parse_subconscious_output did, with
-- the thresholds config-owned. Returns the cleaned doc; the Python side maps
-- it 1:1 into SubconsciousOutput with no logic of its own.
CREATE OR REPLACE FUNCTION normalize_inline_appraisal(
    p_doc JSONB,
    p_allowed_memory_ids TEXT[] DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    doc JSONB := COALESCE(p_doc, '{}'::jsonb);
    min_conf FLOAT := COALESCE(get_config_float('subconscious.min_signal_confidence'), 0.6);
    resp_cap INT := COALESCE(get_config_int('subconscious.response_max_chars'), 500);
    salient JSONB;
    ignored JSONB;
    expansions JSONB;
    instincts JSONB;
    emo JSONB := '{}'::jsonb;
    emo_raw JSONB := doc->'emotional_state';
    emo_conf FLOAT;
    valence FLOAT;
    arousal FLOAT;
    intensity FLOAT;
    emotion TEXT;
    emotion_family TEXT;
    response TEXT;
BEGIN
    -- Memory references: confidence-filtered, clamped, allow-listed, and
    -- required to carry a reason.
    SELECT COALESCE(jsonb_agg(item), '[]'::jsonb) INTO salient FROM (
        SELECT (x || jsonb_build_object('confidence', LEAST(1.0, (x->>'confidence')::float))) AS item
        FROM jsonb_array_elements(CASE WHEN jsonb_typeof(doc->'salient_memories') = 'array'
                                       THEN doc->'salient_memories' ELSE '[]'::jsonb END) x
        WHERE jsonb_typeof(x) = 'object'
          AND (x->>'confidence') ~ '^-?[0-9.]+$'
          AND (x->>'confidence')::float >= min_conf
          AND NULLIF(trim(COALESCE(x->>'reason', '')), '') IS NOT NULL
          AND (p_allowed_memory_ids IS NULL OR COALESCE(x->>'memory_id', '') = ANY(p_allowed_memory_ids))
    ) s;

    SELECT COALESCE(jsonb_agg(item), '[]'::jsonb) INTO ignored FROM (
        SELECT (x || jsonb_build_object('confidence', LEAST(1.0, (x->>'confidence')::float))) AS item
        FROM jsonb_array_elements(CASE WHEN jsonb_typeof(doc->'ignored_memories') = 'array'
                                       THEN doc->'ignored_memories' ELSE '[]'::jsonb END) x
        WHERE jsonb_typeof(x) = 'object'
          AND (x->>'confidence') ~ '^-?[0-9.]+$'
          AND (x->>'confidence')::float >= min_conf
          AND NULLIF(trim(COALESCE(x->>'reason', '')), '') IS NOT NULL
          AND (p_allowed_memory_ids IS NULL OR COALESCE(x->>'memory_id', '') = ANY(p_allowed_memory_ids))
    ) s;

    SELECT COALESCE(jsonb_agg(item), '[]'::jsonb) INTO expansions FROM (
        SELECT (x || jsonb_build_object('confidence', LEAST(1.0, (x->>'confidence')::float))) AS item
        FROM jsonb_array_elements(CASE WHEN jsonb_typeof(doc->'memory_expansions') = 'array'
                                       THEN doc->'memory_expansions' ELSE '[]'::jsonb END) x
        WHERE jsonb_typeof(x) = 'object'
          AND (x->>'confidence') ~ '^-?[0-9.]+$'
          AND (x->>'confidence')::float >= min_conf
          AND NULLIF(trim(COALESCE(x->>'query', '')), '') IS NOT NULL
          AND NULLIF(trim(COALESCE(x->>'reason', '')), '') IS NOT NULL
    ) s;

    SELECT COALESCE(jsonb_agg(item), '[]'::jsonb) INTO instincts FROM (
        SELECT (x || jsonb_build_object(
                   'confidence', LEAST(1.0, (x->>'confidence')::float),
                   'intensity', LEAST(1.0, GREATEST(0.0, (x->>'intensity')::float)))) AS item
        FROM jsonb_array_elements(CASE WHEN jsonb_typeof(doc->'instincts') = 'array'
                                       THEN doc->'instincts' ELSE '[]'::jsonb END) x
        WHERE jsonb_typeof(x) = 'object'
          AND (x->>'confidence') ~ '^-?[0-9.]+$'
          AND (x->>'confidence')::float >= min_conf
          AND (x->>'intensity') ~ '^-?[0-9.]+$'
          AND NULLIF(trim(COALESCE(x->>'impulse', '')), '') IS NOT NULL
          AND NULLIF(trim(COALESCE(x->>'reason', '')), '') IS NOT NULL
    ) s;

    IF jsonb_typeof(emo_raw) = 'object' THEN
        emo_conf := CASE WHEN (emo_raw->>'confidence') ~ '^-?[0-9.]+$'
                         THEN (emo_raw->>'confidence')::float ELSE 0.0 END;
        IF emo_conf >= min_conf THEN
            emotion := NULLIF(trim(COALESCE(emo_raw->>'primary_emotion', '')), '');
            emotion_family := normalize_emotion_family(emo_raw->>'family');
            valence := CASE WHEN (emo_raw->>'valence') ~ '^-?[0-9.]+$'
                            THEN LEAST(1.0, GREATEST(-1.0, (emo_raw->>'valence')::float)) END;
            arousal := CASE WHEN (emo_raw->>'arousal') ~ '^-?[0-9.]+$'
                            THEN LEAST(1.0, GREATEST(0.0, (emo_raw->>'arousal')::float)) END;
            intensity := CASE WHEN (emo_raw->>'intensity') ~ '^-?[0-9.]+$'
                              THEN LEAST(1.0, GREATEST(0.0, (emo_raw->>'intensity')::float)) END;
            IF emotion IS NOT NULL AND valence IS NOT NULL
               AND arousal IS NOT NULL AND intensity IS NOT NULL THEN
                emo := jsonb_build_object(
                    'primary_emotion', emotion,
                    'valence', valence,
                    'arousal', arousal,
                    'intensity', intensity,
                    'confidence', LEAST(1.0, emo_conf)
                ) || CASE WHEN emotion_family IS NULL THEN '{}'::jsonb
                          ELSE jsonb_build_object('family', emotion_family) END;
            END IF;
        END IF;
    END IF;

    response := left(trim(COALESCE(doc->>'subconscious_response', '')), resp_cap);
    IF salient = '[]'::jsonb AND expansions = '[]'::jsonb
       AND instincts = '[]'::jsonb AND emo = '{}'::jsonb THEN
        response := '';
    END IF;

    RETURN jsonb_build_object(
        'salient_memories', salient,
        'ignored_memories', ignored,
        'memory_expansions', expansions,
        'instincts', instincts,
        'emotional_state', emo,
        'subconscious_response', response,
        'narrative_observations', _appraisal_dict_items(doc->'narrative_observations'),
        'relationship_observations', _appraisal_dict_items(doc->'relationship_observations'),
        'contradiction_observations', _appraisal_dict_items(doc->'contradiction_observations'),
        'emotional_observations', _appraisal_dict_items(
            CASE WHEN jsonb_typeof(doc->'emotional_observations') = 'array'
                 THEN doc->'emotional_observations' ELSE doc->'emotional_patterns' END),
        'consolidation_observations', _appraisal_dict_items(
            CASE WHEN jsonb_typeof(doc->'consolidation_observations') = 'array'
                 THEN doc->'consolidation_observations' ELSE doc->'consolidation_suggestions' END)
    );
END;
$$ LANGUAGE plpgsql STABLE;
