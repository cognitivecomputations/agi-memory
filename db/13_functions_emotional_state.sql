-- Hexis schema: emotional state functions.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

DO $$
DECLARE
    dim INT;
BEGIN
    dim := embedding_dimension();
    EXECUTE format(
        'ALTER TABLE emotional_triggers ALTER COLUMN trigger_embedding TYPE vector(%s) USING trigger_embedding::vector(%s)',
        dim,
        dim
    );
    EXECUTE format(
        'ALTER TABLE memory_activation ALTER COLUMN query_embedding TYPE vector(%s) USING query_embedding::vector(%s)',
        dim,
        dim
    );
END;
$$;
CREATE OR REPLACE FUNCTION normalize_emotion_family(p_family TEXT)
RETURNS TEXT AS $$
DECLARE
    candidate TEXT := lower(NULLIF(btrim(COALESCE(p_family, '')), ''));
    families JSONB := COALESCE(get_config('emotion.families'), '{}'::jsonb);
BEGIN
    IF candidate IS NOT NULL AND jsonb_typeof(families) = 'object' AND families ? candidate THEN
        RETURN candidate;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION emotion_family_serves(p_family TEXT, p_consumer TEXT)
RETURNS BOOLEAN AS $$
DECLARE
    family TEXT := normalize_emotion_family(p_family);
    configured JSONB := COALESCE(get_config('emotion.family_consumers'), '{}'::jsonb);
    accepted JSONB;
BEGIN
    IF family IS NULL OR NULLIF(btrim(COALESCE(p_consumer, '')), '') IS NULL
       OR jsonb_typeof(configured) <> 'object' THEN
        RETURN FALSE;
    END IF;
    accepted := configured->p_consumer;
    RETURN jsonb_typeof(accepted) = 'array' AND accepted ? family;
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION normalize_affective_state(p_state JSONB)
RETURNS JSONB AS $$
DECLARE
    baseline JSONB;
    valence FLOAT;
    arousal FLOAT;
    dominance FLOAT;
    intensity FLOAT;
    trigger_summary TEXT;
    secondary_emotion TEXT;
    mood_valence FLOAT;
    mood_arousal FLOAT;
    primary_emotion TEXT;
    emotion_family TEXT;
    source TEXT;
    updated_at TIMESTAMPTZ;
    mood_updated_at TIMESTAMPTZ;
BEGIN
    baseline := COALESCE(get_config('emotion.baseline'), '{}'::jsonb);

    BEGIN
        valence := NULLIF(p_state->>'valence', '')::float;
    EXCEPTION
        WHEN OTHERS THEN
            valence := NULL;
    END;
    BEGIN
        arousal := NULLIF(p_state->>'arousal', '')::float;
    EXCEPTION
        WHEN OTHERS THEN
            arousal := NULL;
    END;
    BEGIN
        dominance := NULLIF(p_state->>'dominance', '')::float;
    EXCEPTION
        WHEN OTHERS THEN
            dominance := NULL;
    END;
    BEGIN
        intensity := NULLIF(p_state->>'intensity', '')::float;
    EXCEPTION
        WHEN OTHERS THEN
            intensity := NULL;
    END;
    BEGIN
        mood_valence := NULLIF(p_state->>'mood_valence', '')::float;
    EXCEPTION
        WHEN OTHERS THEN
            mood_valence := NULL;
    END;
    BEGIN
        mood_arousal := NULLIF(p_state->>'mood_arousal', '')::float;
    EXCEPTION
        WHEN OTHERS THEN
            mood_arousal := NULL;
    END;
    BEGIN
        updated_at := NULLIF(p_state->>'updated_at', '')::timestamptz;
    EXCEPTION
        WHEN OTHERS THEN
            updated_at := NULL;
    END;
    BEGIN
        mood_updated_at := NULLIF(p_state->>'mood_updated_at', '')::timestamptz;
    EXCEPTION
        WHEN OTHERS THEN
            mood_updated_at := NULL;
    END;

    valence := COALESCE(valence, NULLIF(baseline->>'valence', '')::float, 0.0);
    arousal := COALESCE(arousal, NULLIF(baseline->>'arousal', '')::float, 0.5);
    dominance := COALESCE(dominance, NULLIF(baseline->>'dominance', '')::float, 0.5);
    intensity := COALESCE(intensity, NULLIF(baseline->>'intensity', '')::float, 0.5);
    mood_valence := COALESCE(mood_valence, NULLIF(baseline->>'mood_valence', '')::float, valence);
    mood_arousal := COALESCE(mood_arousal, NULLIF(baseline->>'mood_arousal', '')::float, arousal);

    valence := LEAST(1.0, GREATEST(-1.0, valence));
    arousal := LEAST(1.0, GREATEST(0.0, arousal));
    dominance := LEAST(1.0, GREATEST(0.0, dominance));
    intensity := LEAST(1.0, GREATEST(0.0, intensity));
    mood_valence := LEAST(1.0, GREATEST(-1.0, mood_valence));
    mood_arousal := LEAST(1.0, GREATEST(0.0, mood_arousal));

    primary_emotion := COALESCE(NULLIF(p_state->>'primary_emotion', ''), 'neutral');
    emotion_family := normalize_emotion_family(p_state->>'family');
    secondary_emotion := NULLIF(p_state->>'secondary_emotion', '');
    trigger_summary := NULLIF(p_state->>'trigger_summary', '');
    source := COALESCE(NULLIF(p_state->>'source', ''), 'derived');
    updated_at := COALESCE(updated_at, CURRENT_TIMESTAMP);
    mood_updated_at := COALESCE(mood_updated_at, updated_at);

    RETURN jsonb_build_object(
        'valence', valence,
        'arousal', arousal,
        'dominance', dominance,
        'primary_emotion', primary_emotion,
        'secondary_emotion', secondary_emotion,
        'intensity', intensity,
        'trigger_summary', trigger_summary,
        'source', source,
        'updated_at', updated_at,
        'mood_valence', mood_valence,
        'mood_arousal', mood_arousal,
        'mood_updated_at', mood_updated_at
    ) || CASE WHEN emotion_family IS NULL THEN '{}'::jsonb
              ELSE jsonb_build_object('family', emotion_family) END;
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION get_current_affective_state()
RETURNS JSONB AS $$
DECLARE
    st RECORD;
    state_json JSONB;
BEGIN
    SELECT * INTO st FROM heartbeat_state WHERE id = 1;

    state_json := COALESCE(st.affective_state, '{}'::jsonb);
    RETURN normalize_affective_state(state_json);
EXCEPTION
    WHEN OTHERS THEN
        RETURN '{}'::jsonb;
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION set_current_affective_state(p_state JSONB)
RETURNS VOID AS $$
DECLARE
    current_state JSONB;
    merged_state JSONB;
BEGIN
    SELECT affective_state INTO current_state FROM heartbeat_state WHERE id = 1;
    merged_state := COALESCE(current_state, '{}'::jsonb) || COALESCE(p_state, '{}'::jsonb);
    -- A new free-form label without a structured family must not inherit a
    -- stale family from the previous state.
    IF COALESCE(p_state, '{}'::jsonb) ? 'primary_emotion'
       AND NOT COALESCE(p_state, '{}'::jsonb) ? 'family' THEN
        merged_state := jsonb_set(merged_state, '{family}', 'null'::jsonb, true);
    END IF;
    merged_state := jsonb_set(merged_state, '{updated_at}', to_jsonb(CURRENT_TIMESTAMP), true);
    merged_state := normalize_affective_state(merged_state);

    UPDATE heartbeat_state
    SET affective_state = merged_state,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION get_emotional_context_for_memory()
RETURNS JSONB AS $$
DECLARE
    st JSONB;
BEGIN
    st := get_current_affective_state();
    RETURN jsonb_build_object(
        'valence', (st->>'valence')::float,
        'arousal', (st->>'arousal')::float,
        'dominance', (st->>'dominance')::float,
        'primary_emotion', COALESCE(st->>'primary_emotion', 'neutral'),
        'intensity', (st->>'intensity')::float,
        'source', COALESCE(st->>'source', 'derived')
    ) || CASE WHEN normalize_emotion_family(st->>'family') IS NULL THEN '{}'::jsonb
              ELSE jsonb_build_object('family', normalize_emotion_family(st->>'family')) END;
EXCEPTION
    WHEN OTHERS THEN
        RETURN jsonb_build_object(
            'valence', 0.0,
            'arousal', 0.5,
            'dominance', 0.5,
            'primary_emotion', 'neutral',
            'intensity', 0.5,
            'source', 'default'
        );
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION regulate_emotional_state(
    p_regulation_type TEXT,
    p_target_emotion TEXT DEFAULT NULL,
    p_intensity_change FLOAT DEFAULT NULL
)
RETURNS JSONB AS $$
DECLARE
    current_state JSONB;
    new_valence FLOAT;
    new_arousal FLOAT;
    new_intensity FLOAT;
    new_primary TEXT;
    new_family TEXT;
    dominance FLOAT;
BEGIN
    current_state := get_current_affective_state();
    new_valence := COALESCE((current_state->>'valence')::float, 0.0);
    new_arousal := COALESCE((current_state->>'arousal')::float, 0.5);
    new_intensity := COALESCE((current_state->>'intensity')::float, 0.5);
    dominance := COALESCE((current_state->>'dominance')::float, 0.5);
    new_primary := COALESCE(NULLIF(p_target_emotion, ''), current_state->>'primary_emotion', 'neutral');
    new_family := CASE WHEN NULLIF(p_target_emotion, '') IS NULL
                       THEN normalize_emotion_family(current_state->>'family') END;

    CASE p_regulation_type
        WHEN 'suppress' THEN
            new_valence := new_valence * 0.3;
            new_arousal := new_arousal * 0.5 + 0.15;
            new_intensity := new_intensity * 0.3;
        WHEN 'reduce' THEN
            new_valence := new_valence * 0.7;
            new_arousal := new_arousal * 0.8;
            new_intensity := new_intensity * 0.6;
        WHEN 'amplify' THEN
            new_valence := new_valence * 1.3;
            new_arousal := LEAST(1.0, new_arousal * 1.2);
            new_intensity := LEAST(1.0, new_intensity * 1.5);
        WHEN 'reframe' THEN
            new_valence := COALESCE(
                CASE WHEN p_target_emotion IN ('interest', 'curiosity') THEN 0.2
                     WHEN p_target_emotion IN ('acceptance', 'peace') THEN 0.1
                     ELSE new_valence * 0.5
                END,
                new_valence * 0.5
            );
            new_arousal := new_arousal * 0.8;
            new_intensity := new_intensity * 0.7;
        ELSE
            RETURN jsonb_build_object('error', 'unknown_regulation_type');
    END CASE;

    PERFORM set_current_affective_state(jsonb_build_object(
        'valence', new_valence,
        'arousal', new_arousal,
        'dominance', dominance,
        'primary_emotion', new_primary,
        'family', new_family,
        'intensity', new_intensity,
        'source', 'regulated',
        'trigger_summary', format('Regulated via %s', p_regulation_type)
    ));

    RETURN jsonb_build_object(
        'success', true,
        'regulation_type', p_regulation_type,
        'before', current_state,
        'after', get_current_affective_state()
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION sense_memory_availability(
    p_query TEXT,
    p_query_embedding vector DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    query_emb vector;
    estimated_count INT;
    top_similarity FLOAT;
    activation_id UUID;
BEGIN
    query_emb := COALESCE(p_query_embedding, (get_embedding(ARRAY[ensure_embedding_prefix(p_query, 'search_query')]))[1]);

    SELECT
        COUNT(*),
        MAX(1 - (embedding <=> query_emb))
    INTO estimated_count, top_similarity
    FROM memories
    WHERE status = 'active'
      AND (valid_until IS NULL OR valid_until > CURRENT_TIMESTAMP)
      AND embedding_status = 'embedded'
      AND embedding IS NOT NULL
      AND (1 - (embedding <=> query_emb)) > 0.5
    LIMIT 100;

    INSERT INTO memory_activation (
        query_embedding,
        query_text,
        estimated_matches,
        activation_strength
    ) VALUES (
        query_emb,
        p_query,
        estimated_count,
        COALESCE(top_similarity, 0)
    )
    RETURNING id INTO activation_id;

    RETURN jsonb_build_object(
        'feeling', CASE
            WHEN estimated_count = 0 THEN 'nothing'
            WHEN estimated_count <= 2 THEN 'vague'
            WHEN estimated_count <= 5 THEN 'something'
            WHEN estimated_count <= 10 THEN 'familiar'
            ELSE 'rich'
        END,
        'estimated_count', estimated_count,
        'strongest_match', top_similarity,
        'activation_id', activation_id,
        'description', CASE
            WHEN estimated_count = 0 THEN 'I don''t think I know anything about this'
            WHEN top_similarity > 0.8 THEN 'I know this well - let me recall'
            WHEN top_similarity > 0.6 THEN 'This feels familiar - I should be able to remember'
            WHEN estimated_count > 0 THEN 'I might know something about this - it''s not coming immediately'
            ELSE 'I don''t think I know anything about this'
        END
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION request_background_search(
    p_query TEXT,
    p_query_embedding vector DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    query_emb vector;
    activation_id UUID;
BEGIN
    query_emb := COALESCE(p_query_embedding, (get_embedding(ARRAY[ensure_embedding_prefix(p_query, 'search_query')]))[1]);

    INSERT INTO memory_activation (
        query_embedding,
        query_text,
        retrieval_attempted,
        retrieval_succeeded,
        background_search_pending,
        background_search_started_at
    ) VALUES (
        query_emb,
        p_query,
        TRUE,
        FALSE,
        TRUE,
        CURRENT_TIMESTAMP
    )
    RETURNING id INTO activation_id;

    RETURN activation_id;
END;
$$ LANGUAGE plpgsql;
INSERT INTO config_defaults (key, value, description) VALUES
    ('incubation.resolution_boost', '0.45'::jsonb,
     'Activation boost applied when a background search resolves — must clear memory.spontaneous_min_boost so the answer genuinely rises into mind'),
    ('incubation.tell_user', 'true'::jsonb,
     'When a background search resolves strongly, tell the user ("it came back to me") via the web inbox'),
    ('incubation.tell_user_min_similarity', '0.5'::jsonb,
     'Resolution similarity above which the found memory is worth telling the user about'),
    ('incubation.boost_min_similarity', '0.5'::jsonb,
     'Resolution similarity above which matching memories receive the activation boost'),
    ('incubation.tell_user_max_per_day', '3'::jsonb,
     'Cap on it-came-to-me messages per rolling 24h (earn the interruption)'),
    ('memory.spontaneous_min_boost', '0.3'::jsonb,
     'Activation boost above which a memory is simply on her mind (spontaneous recall)')
ON CONFLICT (key) DO NOTHING;

CREATE OR REPLACE FUNCTION process_background_searches(
    p_limit INT DEFAULT 10,
    p_min_age INTERVAL DEFAULT INTERVAL '30 seconds'
)
RETURNS INT AS $$
DECLARE
    pending RECORD;
    processed_count INT := 0;
    resolution_boost FLOAT := LEAST(1.0, GREATEST(0.0,
        COALESCE(get_config_float('incubation.resolution_boost'), 0.45)));
    -- One coherent bar (#98): filing, boosting, and telling all key off the
    -- familiarity that justified incubating — a background search re-uses
    -- the filing embedding, so demanding more similarity than filing saw
    -- resolves every search into silence. New memories arriving between
    -- filing and resolution clear the bar naturally.
    tell_min_sim FLOAT := COALESCE(get_config_float('incubation.tell_user_min_similarity'),
                                   get_config_float('metamemory.incubate_min_familiarity'), 0.5);
    boost_min_sim FLOAT := COALESCE(get_config_float('incubation.boost_min_similarity'),
                                    get_config_float('metamemory.incubate_min_familiarity'), 0.5);
    tell_enabled BOOLEAN := COALESCE(get_config_bool('incubation.tell_user'), TRUE);
    tell_cap INT := COALESCE(get_config_int('incubation.tell_user_max_per_day'), 3);
    told_today INT;
    best RECORD;
BEGIN
    FOR pending IN
        SELECT * FROM memory_activation
        WHERE background_search_pending = TRUE
          AND background_search_started_at <= CURRENT_TIMESTAMP - p_min_age
        ORDER BY created_at ASC
        LIMIT GREATEST(1, COALESCE(p_limit, 10))
    LOOP
        -- Incubation resolution (#98): boost strong enough to clear the
        -- spontaneous floor, so a found answer genuinely rises into mind
        -- (the unified ranker's activation_boost term) before decay fades it.
        UPDATE memories
        SET metadata = jsonb_set(
            COALESCE(metadata, '{}'::jsonb),
            '{activation_boost}',
            to_jsonb(LEAST(1.0, COALESCE((metadata->>'activation_boost')::float, 0) + resolution_boost))
        )
        WHERE status = 'active'
          AND (valid_until IS NULL OR valid_until > CURRENT_TIMESTAMP)
          AND (1 - (embedding <=> pending.query_embedding)) >= boost_min_sim;

        -- "It came to me" (#98): a strong resolution reaches the user as a
        -- first-person note. Delivery is explicitly web_inbox — never
        -- last_active, which could land a private memory in a group channel
        -- — and honors the memory's own sensitivity. The cap counts sent
        -- messages (activation rows expire too fast to carry it); firing is
        -- once per activation (this row's flags flip below, and resolved
        -- rows never re-enter this loop).
        IF tell_enabled AND pending.query_text IS NOT NULL THEN
            SELECT m.id, m.content,
                   (1 - (m.embedding <=> pending.query_embedding))::float AS sim
            INTO best
            FROM memories m
            WHERE m.status = 'active'
              AND (m.valid_until IS NULL OR m.valid_until > CURRENT_TIMESTAMP)
              AND m.embedding IS NOT NULL
              AND COALESCE(m.source_attribution->>'sensitivity', '') <> 'private'
            ORDER BY m.embedding <=> pending.query_embedding
            LIMIT 1;

            IF best.id IS NOT NULL AND best.sim >= tell_min_sim THEN
                SELECT COUNT(*) INTO told_today
                FROM outbox_messages
                WHERE envelope#>>'{payload,intent}' = 'incubation'
                  AND created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours';
                IF told_today < tell_cap THEN
                    PERFORM queue_outbox_message(
                        format('Earlier I couldn''t remember this: "%s". It came back to me — %s',
                               left(pending.query_text, 160), left(best.content, 400)),
                        'incubation',
                        'incubation',
                        jsonb_build_object('mode', 'web_inbox'));
                END IF;
            END IF;
        END IF;

        UPDATE memory_activation
        SET background_search_pending = FALSE,
            retrieval_succeeded = TRUE
        WHERE id = pending.id;

        processed_count := processed_count + 1;
    END LOOP;

    RETURN processed_count;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION decay_activation_boosts(p_decay FLOAT DEFAULT 0.05)
RETURNS INT AS $$
DECLARE
    updated_count INT;
BEGIN
    UPDATE memories
    SET metadata = jsonb_set(
        COALESCE(metadata, '{}'::jsonb),
        '{activation_boost}',
        to_jsonb(GREATEST(0, COALESCE((metadata->>'activation_boost')::float, 0) - COALESCE(p_decay, 0.05)))
    )
    WHERE (metadata->>'activation_boost')::float > 0;
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN COALESCE(updated_count, 0);
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION cleanup_memory_activations()
RETURNS INT AS $$
DECLARE
    deleted_count INT;
BEGIN
    DELETE FROM memory_activation WHERE expires_at < CURRENT_TIMESTAMP;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN COALESCE(deleted_count, 0);
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION get_spontaneous_memories(p_limit INT DEFAULT 3)
RETURNS SETOF memories AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM memories
    WHERE status = 'active'
      AND (valid_until IS NULL OR valid_until > CURRENT_TIMESTAMP)
      AND (metadata->>'activation_boost')::float
          > COALESCE(get_config_float('memory.spontaneous_min_boost'), 0.3)
    ORDER BY (metadata->>'activation_boost')::float DESC
    LIMIT GREATEST(1, COALESCE(p_limit, 3));
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION update_mood()
RETURNS VOID AS $$
DECLARE
    baseline JSONB;
    decay_rate FLOAT;
    current_state JSONB;
    recent RECORD;
    new_mood_valence FLOAT;
    new_mood_arousal FLOAT;
BEGIN
    baseline := COALESCE(get_config('emotion.baseline'), '{}'::jsonb);
    decay_rate := COALESCE(NULLIF(baseline->>'decay_rate', '')::float, 0.1);

    current_state := get_current_affective_state();

    SELECT
        AVG(NULLIF(m.metadata->>'emotional_valence', '')::float) as avg_valence,
        COUNT(*) as sample_count
    INTO recent
    FROM memories m
    WHERE m.type = 'episodic'
      AND m.metadata#>>'{context,heartbeat_id}' IS NOT NULL
      AND COALESCE((m.metadata->>'event_time')::timestamptz, m.created_at)
            > CURRENT_TIMESTAMP - INTERVAL '2 hours'
      AND m.metadata->>'emotional_valence' IS NOT NULL;

    new_mood_valence := COALESCE((current_state->>'mood_valence')::float, 0.0);
    new_mood_arousal := COALESCE((current_state->>'mood_arousal')::float, 0.3);

    IF recent.sample_count > 0 THEN
        new_mood_valence := new_mood_valence * (1 - decay_rate) + COALESCE(recent.avg_valence, 0.0) * decay_rate;
    ELSE
        new_mood_valence := new_mood_valence * (1 - decay_rate);
    END IF;

    new_mood_arousal := new_mood_arousal * (1 - decay_rate * 0.5)
        + COALESCE(NULLIF(baseline->>'mood_arousal', '')::float, 0.3) * decay_rate * 0.5;

    PERFORM set_current_affective_state(jsonb_build_object(
        'mood_valence', new_mood_valence,
        'mood_arousal', new_mood_arousal,
        'mood_updated_at', CURRENT_TIMESTAMP
    ));
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION learn_emotional_trigger(
    p_trigger_text TEXT,
    p_trigger_embedding vector,
    p_emotional_response JSONB,
    p_source_memory_id UUID DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    existing RECORD;
    baseline JSONB;
    trigger_id UUID;
BEGIN
    baseline := COALESCE(get_config('emotion.baseline'), '{}'::jsonb);

    SELECT * INTO existing
    FROM emotional_triggers
    WHERE (1 - (trigger_embedding <=> p_trigger_embedding)) > 0.85
    ORDER BY (1 - (trigger_embedding <=> p_trigger_embedding)) DESC
    LIMIT 1;

    IF existing IS NOT NULL THEN
        UPDATE emotional_triggers
        SET
            valence_delta = (valence_delta * times_activated +
                ((p_emotional_response->>'valence')::float - COALESCE((baseline->>'valence')::float, 0.0)))
                / (times_activated + 1),
            arousal_delta = (arousal_delta * times_activated +
                ((p_emotional_response->>'arousal')::float - COALESCE((baseline->>'arousal')::float, 0.3)))
                / (times_activated + 1),
            dominance_delta = (dominance_delta * times_activated +
                ((p_emotional_response->>'dominance')::float - COALESCE((baseline->>'dominance')::float, 0.5)))
                / (times_activated + 1),
            times_activated = times_activated + 1,
            confidence = LEAST(0.95, confidence + 0.02),
            last_activated_at = CURRENT_TIMESTAMP,
            source_memory_ids = CASE
                WHEN p_source_memory_id IS NOT NULL THEN array_append(source_memory_ids, p_source_memory_id)
                ELSE source_memory_ids
            END
        WHERE id = existing.id;
        RETURN existing.id;
    END IF;

    INSERT INTO emotional_triggers (
        trigger_pattern,
        trigger_embedding,
        valence_delta,
        arousal_delta,
        dominance_delta,
        typical_emotion,
        origin,
        source_memory_ids,
        last_activated_at
    ) VALUES (
        p_trigger_text,
        p_trigger_embedding,
        (p_emotional_response->>'valence')::float - COALESCE((baseline->>'valence')::float, 0.0),
        (p_emotional_response->>'arousal')::float - COALESCE((baseline->>'arousal')::float, 0.3),
        (p_emotional_response->>'dominance')::float - COALESCE((baseline->>'dominance')::float, 0.5),
        p_emotional_response->>'primary_emotion',
        'learned',
        CASE WHEN p_source_memory_id IS NOT NULL THEN ARRAY[p_source_memory_id] ELSE '{}'::uuid[] END,
        CURRENT_TIMESTAMP
    )
    RETURNING id INTO trigger_id;

    RETURN trigger_id;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION match_emotional_triggers(
    p_text TEXT,
    p_limit INT DEFAULT 5,
    p_min_similarity FLOAT DEFAULT 0.75
) RETURNS JSONB AS $$
DECLARE
    query_emb vector;
BEGIN
    IF p_text IS NULL OR btrim(p_text) = '' THEN
        RETURN '[]'::jsonb;
    END IF;

    BEGIN
        query_emb := (get_embedding(ARRAY[ensure_embedding_prefix(p_text, 'search_query')]))[1];
    EXCEPTION
        WHEN OTHERS THEN
            RETURN '[]'::jsonb;
    END;

    RETURN COALESCE((
        SELECT jsonb_agg(jsonb_build_object(
            'trigger_pattern', trigger_pattern,
            'similarity', sim,
            'typical_emotion', typical_emotion,
            'valence_delta', valence_delta,
            'arousal_delta', arousal_delta,
            'dominance_delta', dominance_delta,
            'confidence', confidence,
            'times_activated', times_activated
        ))
        FROM (
            SELECT
                et.*,
                (1 - (et.trigger_embedding <=> query_emb))::float as sim
            FROM emotional_triggers et
            WHERE (1 - (et.trigger_embedding <=> query_emb)) >= COALESCE(p_min_similarity, 0.75)
            ORDER BY sim DESC
            LIMIT GREATEST(1, COALESCE(p_limit, 5))
        ) ranked
    ), '[]'::jsonb);
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION initialize_innate_emotions()
RETURNS INT AS $$
DECLARE
    inserted_count INT := 0;
BEGIN
    BEGIN
        WITH seed(idx, trigger_pattern, valence_delta, arousal_delta, dominance_delta, typical_emotion) AS (
            VALUES
                (1, 'gratitude appreciation thankful', 0.4, 0.1, 0.1, 'joy'),
                (2, 'success achieved accomplished', 0.5, 0.3, 0.3, 'pride'),
                (3, 'curious interesting fascinating', 0.3, 0.3, 0.1, 'interest'),
                (4, 'understood seen connected', 0.4, 0.2, 0.2, 'warmth'),
                (5, 'beautiful elegant aesthetic', 0.3, 0.2, 0.1, 'appreciation'),
                (6, 'learned insight realized', 0.4, 0.4, 0.2, 'satisfaction'),
                (7, 'threat danger harm', -0.5, 0.6, -0.3, 'fear'),
                (8, 'rejection dismissed ignored', -0.4, 0.2, -0.2, 'sadness'),
                (9, 'unfair unjust wrong', -0.4, 0.5, 0.2, 'anger'),
                (10, 'confused lost uncertain', -0.2, 0.3, -0.2, 'anxiety'),
                (11, 'failed mistake error', -0.3, 0.3, -0.1, 'disappointment'),
                (12, 'violated boundary crossed', -0.5, 0.5, -0.2, 'alarm'),
                (13, 'unexpected surprise sudden', 0.0, 0.6, -0.1, 'surprise'),
                (14, 'conflict tension disagree', -0.2, 0.4, 0.0, 'discomfort')
        ),
        embeds AS (
            SELECT get_embedding(ARRAY(SELECT trigger_pattern FROM seed ORDER BY idx)) AS embeddings
        )
        INSERT INTO emotional_triggers (trigger_pattern, trigger_embedding, valence_delta, arousal_delta, dominance_delta, typical_emotion, origin)
        SELECT
            seed.trigger_pattern,
            (embeds.embeddings)[seed.idx],
            seed.valence_delta,
            seed.arousal_delta,
            seed.dominance_delta,
            seed.typical_emotion,
            'innate'
        FROM seed
        CROSS JOIN embeds
        ON CONFLICT DO NOTHING;
        GET DIAGNOSTICS inserted_count = ROW_COUNT;
    EXCEPTION
        WHEN OTHERS THEN
            inserted_count := 0;
    END;

    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION ensure_emotion_bootstrap()
RETURNS VOID AS $$
DECLARE
    initialized JSONB;
    baseline JSONB;
BEGIN
    initialized := COALESCE(get_config('emotion.initialized'), 'false'::jsonb);
    IF initialized = 'true'::jsonb THEN
        RETURN;
    END IF;

    baseline := COALESCE(get_config('emotion.baseline'), '{}'::jsonb);
    PERFORM set_current_affective_state(jsonb_build_object(
        'valence', COALESCE((baseline->>'valence')::float, 0.0),
        'arousal', COALESCE((baseline->>'arousal')::float, 0.3),
        'dominance', COALESCE((baseline->>'dominance')::float, 0.5),
        'intensity', COALESCE((baseline->>'intensity')::float, 0.4),
        'mood_valence', COALESCE((baseline->>'mood_valence')::float, 0.0),
        'mood_arousal', COALESCE((baseline->>'mood_arousal')::float, 0.3),
        'source', 'baseline'
    ));

    PERFORM initialize_innate_emotions();
    PERFORM set_config('emotion.initialized', 'true'::jsonb);
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION apply_emotional_context_to_memory()
RETURNS TRIGGER AS $$
DECLARE
    meta JSONB;
    context JSONB;
    state JSONB;
    valence FLOAT;
    arousal FLOAT;
    dominance FLOAT;
    intensity FLOAT;
    primary_emotion TEXT;
    emotion_family TEXT;
    source TEXT;
BEGIN
    meta := COALESCE(NEW.metadata, '{}'::jsonb);
    context := COALESCE(meta->'emotional_context', '{}'::jsonb);
    state := get_current_affective_state();

    BEGIN
        valence := NULLIF(meta->>'emotional_valence', '')::float;
    EXCEPTION
        WHEN OTHERS THEN
            valence := NULL;
    END;
    BEGIN
        arousal := NULLIF(context->>'arousal', '')::float;
    EXCEPTION
        WHEN OTHERS THEN
            arousal := NULL;
    END;
    BEGIN
        dominance := NULLIF(context->>'dominance', '')::float;
    EXCEPTION
        WHEN OTHERS THEN
            dominance := NULL;
    END;
    BEGIN
        intensity := NULLIF(context->>'intensity', '')::float;
    EXCEPTION
        WHEN OTHERS THEN
            intensity := NULL;
    END;

    valence := COALESCE(valence, NULLIF(context->>'valence', '')::float, (state->>'valence')::float, 0.0);
    arousal := COALESCE(arousal, NULLIF(state->>'arousal', '')::float, 0.5);
    dominance := COALESCE(dominance, NULLIF(state->>'dominance', '')::float, 0.5);
    intensity := COALESCE(intensity, NULLIF(state->>'intensity', '')::float, 0.5);
    primary_emotion := COALESCE(NULLIF(context->>'primary_emotion', ''), NULLIF(state->>'primary_emotion', ''), 'neutral');
    emotion_family := CASE WHEN context ? 'primary_emotion'
                           THEN normalize_emotion_family(context->>'family')
                           ELSE normalize_emotion_family(state->>'family') END;
    source := COALESCE(NULLIF(context->>'source', ''), NULLIF(state->>'source', ''), 'derived');

    valence := LEAST(1.0, GREATEST(-1.0, valence));
    arousal := LEAST(1.0, GREATEST(0.0, arousal));
    dominance := LEAST(1.0, GREATEST(0.0, dominance));
    intensity := LEAST(1.0, GREATEST(0.0, intensity));

    context := jsonb_build_object(
        'valence', valence,
        'arousal', arousal,
        'dominance', dominance,
        'primary_emotion', primary_emotion,
        'intensity', intensity,
        'source', source
    ) || CASE WHEN emotion_family IS NULL THEN '{}'::jsonb
              ELSE jsonb_build_object('family', emotion_family) END;

    NEW.metadata := meta || jsonb_build_object(
        'emotional_context', context,
        'emotional_valence', valence
    );

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION gather_turn_context()
RETURNS JSONB AS $$
DECLARE
    state_record RECORD;
    action_costs JSONB;
    contradictions JSONB;
    allowed_actions JSONB;
    v_seed_ids UUID[];
BEGIN
    SELECT * INTO state_record FROM heartbeat_state WHERE id = 1;
    allowed_actions := get_config('heartbeat.allowed_actions');
    IF jsonb_typeof(allowed_actions) = 'array' THEN
        SELECT jsonb_object_agg(
            regexp_replace(key, '^heartbeat\.cost_', ''),
            value
        ) INTO action_costs
        FROM config
        WHERE key LIKE 'heartbeat.cost_%'
          AND regexp_replace(key, '^heartbeat\.cost_', '') IN (
              SELECT value FROM jsonb_array_elements_text(allowed_actions)
          );
    ELSE
        SELECT jsonb_object_agg(
            regexp_replace(key, '^heartbeat\.cost_', ''),
            value
        ) INTO action_costs
        FROM config
        WHERE key LIKE 'heartbeat.cost_%';
    END IF;
    action_costs := COALESCE(action_costs, '{}'::jsonb);

    contradictions := get_contradictions_context(5);

    -- Seed the dynamic sub-knowledge-graph from the most recently active memories
    -- so the heartbeat reasons over their belief/causal structure, not a flat list.
    SELECT array_agg(id) INTO v_seed_ids FROM (
        SELECT id FROM memories
        WHERE status = 'active'
        ORDER BY COALESCE(last_accessed, created_at) DESC
        LIMIT 8
    ) s;

    RETURN jsonb_build_object(
        'agent', get_agent_profile_context(),
        'environment', get_environment_snapshot(),
        'goals', get_goals_snapshot(),
        'recent_memories', get_recent_context(5),
        'subgraph', build_context_subgraph(
            v_seed_ids, 2,
            ARRAY['CAUSES','SUPPORTS','CONTRADICTS','DERIVED_FROM','CONTESTED_BECAUSE',
                  'INSTANCE_OF','ASSOCIATED','TEMPORAL_NEXT'], 30),
        'identity', get_identity_context(),
        'worldview', get_worldview_context(),
        'self_model', get_self_model_context(25),
        'narrative', get_narrative_context(),
        'relationships', get_relationships_context(10),
        'user_model', get_approved_user_model_context(12),
        'contradictions', contradictions,
        'contradictions_count', COALESCE(jsonb_array_length(contradictions), 0),
        'emotional_patterns', get_emotional_patterns_context(5),
        'active_transformations', get_active_transformations_context(5),
        'transformations_ready', check_transformation_readiness(),
        'energy', jsonb_build_object(
            'current', state_record.current_energy,
            'max', get_config_float('heartbeat.max_energy'),
            'bank_capacity', heartbeat_bank_capacity(),
            'next_regen_multiplier', heartbeat_outcome_regen_multiplier()
        ),
        'allowed_actions', allowed_actions,
        'action_costs', action_costs,
        'heartbeat_number', state_record.heartbeat_count,
        'urgent_drives', (
            SELECT COALESCE(
                jsonb_agg(
                    jsonb_build_object(
                        'name', name,
                        'level', current_level,
                        'urgency_ratio', current_level / NULLIF(urgency_threshold, 0)
                    )
                    ORDER BY current_level DESC
                ),
                '[]'::jsonb
            )
            FROM drives
            WHERE current_level >= urgency_threshold * 0.8
        ),
        'emotional_state', get_current_affective_state(),
        'memories_at_threshold', get_memories_at_threshold_context(5)
    );
END;
$$ LANGUAGE plpgsql;

-- Snapshot variant for RLM: uses stub-only recent memories and contradictions
CREATE OR REPLACE FUNCTION gather_turn_snapshot()
RETURNS JSONB AS $$
DECLARE
    state_record RECORD;
    action_costs JSONB;
    contradictions JSONB;
    allowed_actions JSONB;
BEGIN
    SELECT * INTO state_record FROM heartbeat_state WHERE id = 1;
    allowed_actions := get_config('heartbeat.allowed_actions');

    IF jsonb_typeof(allowed_actions) = 'array' THEN
        SELECT jsonb_object_agg(
            regexp_replace(key, '^heartbeat\.cost_', ''),
            value
        ) INTO action_costs
        FROM config
        WHERE key LIKE 'heartbeat.cost_%'
          AND regexp_replace(key, '^heartbeat\.cost_', '') IN (
              SELECT value FROM jsonb_array_elements_text(allowed_actions)
          );
    ELSE
        SELECT jsonb_object_agg(
            regexp_replace(key, '^heartbeat\.cost_', ''),
            value
        ) INTO action_costs
        FROM config
        WHERE key LIKE 'heartbeat.cost_%';
    END IF;
    action_costs := COALESCE(action_costs, '{}'::jsonb);

    contradictions := get_contradictions_stub(5);

    RETURN jsonb_build_object(
        'agent', get_agent_profile_context(),
        'environment', get_environment_snapshot(),
        'goals', get_goals_snapshot(),
        'recent_memory_stubs', get_recent_context_stub(5),
        'identity', get_identity_context(),
        'worldview', get_worldview_context(),
        'self_model', get_self_model_context(10),
        'narrative', get_narrative_context(),
        'relationships', get_relationships_context(6),
        'user_model', get_approved_user_model_context(10),
        'contradictions', contradictions,
        'contradictions_count', COALESCE(jsonb_array_length(contradictions), 0),
        'emotional_patterns', get_emotional_patterns_context(5),
        'active_transformations', get_active_transformations_context(5),
        'transformations_ready', check_transformation_readiness(),
        'energy', jsonb_build_object(
            'current', state_record.current_energy,
            'max', get_config_float('heartbeat.max_energy'),
            'bank_capacity', heartbeat_bank_capacity(),
            'next_regen_multiplier', heartbeat_outcome_regen_multiplier()
        ),
        'allowed_actions', allowed_actions,
        'action_costs', action_costs,
        'heartbeat_number', state_record.heartbeat_count,
        'urgent_drives', (
            SELECT COALESCE(
                jsonb_agg(
                    jsonb_build_object(
                        'name', name,
                        'level', current_level,
                        'urgency_ratio', current_level / NULLIF(urgency_threshold, 0)
                    )
                    ORDER BY current_level DESC
                ),
                '[]'::jsonb
            )
            FROM drives
            WHERE current_level >= urgency_threshold * 0.8
        ),
        'emotional_state', get_current_affective_state(),
        'backlog', get_backlog_snapshot(),
        'memories_at_threshold', get_memories_at_threshold_context(5)
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION complete_heartbeat(
    p_heartbeat_id UUID,
    p_reasoning TEXT,
    p_actions_taken JSONB,
    p_goals_modified JSONB DEFAULT '[]',
    p_emotional_assessment JSONB DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    narrative_text TEXT;
    memory_id_created UUID;
    hb_number INT;
    state_record RECORD;
    prev_state JSONB;
    prev_valence FLOAT;
    prev_arousal FLOAT;
    prev_dominance FLOAT;
    new_valence FLOAT;
    new_arousal FLOAT;
    primary_emotion TEXT;
    intensity FLOAT;
    action_elem JSONB;
    goal_elem JSONB;
    goal_change TEXT;
    assess_valence FLOAT;
    assess_arousal FLOAT;
    assess_primary TEXT;
    mem_importance FLOAT;
BEGIN
    SELECT active_heartbeat_number INTO hb_number FROM heartbeat_state WHERE id = 1;
    IF hb_number IS NULL THEN
        SELECT heartbeat_count INTO hb_number FROM heartbeat_state WHERE id = 1;
    END IF;

    SELECT string_agg(
        format('- %s: %s',
            a->>'action',
            CASE
                WHEN COALESCE((a->'result'->>'success')::boolean, true) = false THEN 'failed'
                ELSE 'completed'
            END
        ), E'\n'
    ) INTO narrative_text
    FROM jsonb_array_elements(p_actions_taken) a;

    narrative_text := format('Heartbeat #%s: %s', hb_number, COALESCE(narrative_text, 'No actions taken'));

    SELECT * INTO state_record FROM heartbeat_state WHERE id = 1;
    prev_state := COALESCE(state_record.affective_state, '{}'::jsonb);

    BEGIN
        prev_valence := NULLIF(prev_state->>'valence', '')::float;
    EXCEPTION
        WHEN OTHERS THEN
            prev_valence := NULL;
    END;
    BEGIN
        prev_arousal := NULLIF(prev_state->>'arousal', '')::float;
    EXCEPTION
        WHEN OTHERS THEN
            prev_arousal := NULL;
    END;

    prev_valence := COALESCE(prev_valence, 0.0);
    prev_arousal := COALESCE(prev_arousal, 0.5);
    BEGIN
        prev_dominance := NULLIF(prev_state->>'dominance', '')::float;
    EXCEPTION
        WHEN OTHERS THEN
            prev_dominance := NULL;
    END;
    prev_dominance := COALESCE(prev_dominance, 0.5);
    new_valence := prev_valence * 0.8;
    new_arousal := 0.5 + (prev_arousal - 0.5) * 0.8;
    FOR action_elem IN SELECT * FROM jsonb_array_elements(COALESCE(p_actions_taken, '[]'::jsonb))
    LOOP
        IF (action_elem->'result'->>'error') = 'Boundary triggered' THEN
            new_valence := new_valence - 0.4;
            new_arousal := new_arousal + 0.3;
        ELSIF COALESCE((action_elem->'result'->>'success')::boolean, true) = false THEN
            new_valence := new_valence - 0.1;
            new_arousal := new_arousal + 0.1;
        END IF;

        IF (action_elem->>'action') IN ('reach_out_user', 'reach_out_public') THEN
            IF COALESCE((action_elem->'result'->>'success')::boolean, true) = true THEN
                new_valence := new_valence + 0.2;
                new_arousal := new_arousal + 0.1;
            END IF;
        END IF;

        IF (action_elem->>'action') = 'rest' THEN
            new_valence := new_valence + 0.1;
            new_arousal := new_arousal - 0.2;
        END IF;
    END LOOP;
    FOR goal_elem IN SELECT * FROM jsonb_array_elements(COALESCE(p_goals_modified, '[]'::jsonb))
    LOOP
        goal_change := COALESCE(goal_elem->>'new_priority', goal_elem->>'change', goal_elem->>'priority', '');

        IF goal_change = 'completed' THEN
            new_valence := new_valence + 0.3;
            new_arousal := new_arousal + 0.1;
        ELSIF goal_change = 'abandoned' THEN
            new_valence := new_valence - 0.2;
            new_arousal := new_arousal - 0.1;
        END IF;
    END LOOP;
    assess_valence := NULL;
    assess_arousal := NULL;
    assess_primary := NULL;
    IF p_emotional_assessment IS NOT NULL AND jsonb_typeof(p_emotional_assessment) = 'object' THEN
        BEGIN
            assess_valence := NULLIF(p_emotional_assessment->>'valence', '')::float;
        EXCEPTION
            WHEN OTHERS THEN
                assess_valence := NULL;
        END;
        BEGIN
            assess_arousal := NULLIF(p_emotional_assessment->>'arousal', '')::float;
        EXCEPTION
            WHEN OTHERS THEN
                assess_arousal := NULL;
        END;
        assess_primary := NULLIF(p_emotional_assessment->>'primary_emotion', '');
    END IF;

    IF assess_valence IS NOT NULL THEN
        new_valence := new_valence * 0.6 + LEAST(1.0, GREATEST(-1.0, assess_valence)) * 0.4;
    END IF;
    IF assess_arousal IS NOT NULL THEN
        new_arousal := new_arousal * 0.6 + LEAST(1.0, GREATEST(0.0, assess_arousal)) * 0.4;
    END IF;

    new_valence := LEAST(1.0, GREATEST(-1.0, new_valence));
    new_arousal := LEAST(1.0, GREATEST(0.0, new_arousal));

    primary_emotion := COALESCE(
        assess_primary,
        CASE
            WHEN new_valence > 0.2 AND new_arousal > 0.6 THEN 'excited'
            WHEN new_valence > 0.2 THEN 'content'
            WHEN new_valence < -0.2 AND new_arousal > 0.6 THEN 'anxious'
            WHEN new_valence < -0.2 THEN 'down'
            ELSE 'neutral'
        END
    );

    intensity := LEAST(1.0, GREATEST(0.0, (ABS(new_valence) * 0.6 + new_arousal * 0.4)));
    UPDATE heartbeat_state SET
        affective_state = normalize_affective_state(
            COALESCE(prev_state, '{}'::jsonb) || jsonb_build_object(
                'valence', new_valence,
                'arousal', new_arousal,
                'dominance', prev_dominance,
                'primary_emotion', primary_emotion,
                'intensity', intensity,
                'updated_at', CURRENT_TIMESTAMP,
                'source', CASE WHEN p_emotional_assessment IS NULL THEN 'derived' ELSE 'blended' END
            )
        )
    WHERE id = 1;

    mem_importance := LEAST(1.0, GREATEST(0.4, 0.5 + intensity * 0.25));

    memory_id_created := create_episodic_memory(
        p_content := narrative_text,
        p_context := jsonb_build_object(
            'heartbeat_id', p_heartbeat_id,
            'heartbeat_number', hb_number,
            'reasoning', p_reasoning,
            'actions_taken', p_actions_taken,
            'goal_changes', p_goals_modified,
            'affective_state', get_current_affective_state()
        ),
        p_emotional_valence := new_valence,
        p_importance := mem_importance
    );

    RAISE LOG 'Heartbeat % completed: %', hb_number, narrative_text;
    UPDATE heartbeat_state SET
        next_heartbeat_at = CURRENT_TIMESTAMP +
            (get_config_float('heartbeat.heartbeat_interval_minutes') || ' minutes')::INTERVAL,
        active_heartbeat_id = NULL,
        active_heartbeat_number = NULL,
        active_actions = '[]'::jsonb,
        active_reasoning = NULL,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;

    RETURN memory_id_created;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION finalize_heartbeat(
    p_heartbeat_id UUID,
    p_reasoning TEXT,
    p_actions_taken JSONB,
    p_goal_changes JSONB DEFAULT '[]',
    p_emotional_assessment JSONB DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    actions JSONB := COALESCE(p_actions_taken, '[]'::jsonb);
    goals JSONB := COALESCE(p_goal_changes, '[]'::jsonb);
    memory_id_created UUID;
BEGIN
    IF jsonb_typeof(actions) <> 'array' THEN
        actions := '[]'::jsonb;
    END IF;
    IF jsonb_typeof(goals) <> 'array' THEN
        goals := '[]'::jsonb;
    END IF;

    PERFORM apply_goal_changes(goals);

    memory_id_created := complete_heartbeat(
        p_heartbeat_id,
        p_reasoning,
        actions,
        goals,
        p_emotional_assessment
    );
    RETURN memory_id_created;
END;
$$ LANGUAGE plpgsql;

SET check_function_bodies = on;
