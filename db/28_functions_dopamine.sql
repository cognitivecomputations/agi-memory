-- ============================================================================
-- DOPAMINE SYSTEM
--
-- Neuromodulatory reinforcement layer.  Dopamine is a gain dial on existing
-- systems — it does not introduce a new memory type or subsystem.
--
-- Tonic dopamine:  slowly-drifting baseline (0.0–1.0, default 0.5) that
--   modulates activation-boost decay rate, mood persistence, and memory
--   encoding importance.
--
-- Phasic dopamine: discrete spikes/dips triggered by reward-prediction error
--   (RPE) during subconscious appraisal.  A spike atomically boosts
--   importance + activation on recent memories, spreads activation through
--   neighborhoods, shifts tonic via EMA, and modulates drives.
--
-- The conscious agent never sees dopamine directly — only its downstream
-- effects (boosted memories, shifted mood, stronger instincts).
-- ============================================================================

-- Each baseline file runs as its own psql session at initdb, so it must pin
-- its own search_path: under an ag_catalog-first cluster default, unqualified
-- CREATEs would land Hexis objects in ag_catalog (the #77 fossil bug).
SET search_path = public, ag_catalog, "$user";

SET check_function_bodies = off;

INSERT INTO config_defaults (key, value, description) VALUES
    ('reward.rpe_spike_threshold', '0.35'::jsonb,
     'Absolute reward-prediction error required to fire a dopamine spike'),
    ('reward.dopamine_spike_salience_threshold', '0.65'::jsonb,
     'Reward salience required to fire a dopamine spike from a generic reward event')
ON CONFLICT (key) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Override normalize_affective_state to preserve dopamine fields
-- ---------------------------------------------------------------------------

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
    -- Dopamine fields
    da_tonic FLOAT;
    da_phasic FLOAT;
    da_spike_at TIMESTAMPTZ;
    da_spike_trigger TEXT;
    da_spike_rpe FLOAT;
BEGIN
    baseline := COALESCE(get_config('emotion.baseline'), '{}'::jsonb);

    BEGIN valence := NULLIF(p_state->>'valence', '')::float;
    EXCEPTION WHEN OTHERS THEN valence := NULL; END;
    BEGIN arousal := NULLIF(p_state->>'arousal', '')::float;
    EXCEPTION WHEN OTHERS THEN arousal := NULL; END;
    BEGIN dominance := NULLIF(p_state->>'dominance', '')::float;
    EXCEPTION WHEN OTHERS THEN dominance := NULL; END;
    BEGIN intensity := NULLIF(p_state->>'intensity', '')::float;
    EXCEPTION WHEN OTHERS THEN intensity := NULL; END;
    BEGIN mood_valence := NULLIF(p_state->>'mood_valence', '')::float;
    EXCEPTION WHEN OTHERS THEN mood_valence := NULL; END;
    BEGIN mood_arousal := NULLIF(p_state->>'mood_arousal', '')::float;
    EXCEPTION WHEN OTHERS THEN mood_arousal := NULL; END;
    BEGIN updated_at := NULLIF(p_state->>'updated_at', '')::timestamptz;
    EXCEPTION WHEN OTHERS THEN updated_at := NULL; END;
    BEGIN mood_updated_at := NULLIF(p_state->>'mood_updated_at', '')::timestamptz;
    EXCEPTION WHEN OTHERS THEN mood_updated_at := NULL; END;

    -- Dopamine extraction (preserve through normalization)
    BEGIN da_tonic := NULLIF(p_state->>'dopamine_tonic', '')::float;
    EXCEPTION WHEN OTHERS THEN da_tonic := NULL; END;
    BEGIN da_phasic := NULLIF(p_state->>'dopamine_phasic', '')::float;
    EXCEPTION WHEN OTHERS THEN da_phasic := NULL; END;
    BEGIN da_spike_at := NULLIF(p_state->>'dopamine_spike_at', '')::timestamptz;
    EXCEPTION WHEN OTHERS THEN da_spike_at := NULL; END;
    BEGIN da_spike_rpe := NULLIF(p_state->>'dopamine_spike_rpe', '')::float;
    EXCEPTION WHEN OTHERS THEN da_spike_rpe := NULL; END;
    da_spike_trigger := NULLIF(p_state->>'dopamine_spike_trigger', '');

    -- Apply defaults and clamp affect fields
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

    -- Dopamine defaults and clamp
    da_tonic := LEAST(1.0, GREATEST(0.0, COALESCE(da_tonic, 0.5)));
    da_phasic := LEAST(1.0, GREATEST(-1.0, COALESCE(da_phasic, 0.0)));

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
        'mood_updated_at', mood_updated_at,
        -- Dopamine fields preserved
        'dopamine_tonic', da_tonic,
        'dopamine_phasic', da_phasic,
        'dopamine_spike_at', da_spike_at,
        'dopamine_spike_rpe', da_spike_rpe,
        'dopamine_spike_trigger', da_spike_trigger
    ) || CASE WHEN emotion_family IS NULL THEN '{}'::jsonb
              ELSE jsonb_build_object('family', emotion_family) END;
END;
$$ LANGUAGE plpgsql STABLE;


-- ---------------------------------------------------------------------------
-- get_dopamine_state()  —  read current tonic + recent phasic from state
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION get_dopamine_state()
RETURNS JSONB AS $$
DECLARE
    state JSONB;
    tonic FLOAT;
    phasic FLOAT;
    spike_at TIMESTAMPTZ;
    spike_age_seconds FLOAT;
BEGIN
    state := get_current_affective_state();

    BEGIN tonic := NULLIF(state->>'dopamine_tonic', '')::float;
    EXCEPTION WHEN OTHERS THEN tonic := NULL; END;
    BEGIN phasic := NULLIF(state->>'dopamine_phasic', '')::float;
    EXCEPTION WHEN OTHERS THEN phasic := NULL; END;
    BEGIN spike_at := NULLIF(state->>'dopamine_spike_at', '')::timestamptz;
    EXCEPTION WHEN OTHERS THEN spike_at := NULL; END;

    tonic := COALESCE(tonic, 0.5);
    phasic := COALESCE(phasic, 0.0);

    -- Compute how long ago the last spike was
    IF spike_at IS NOT NULL THEN
        spike_age_seconds := EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - spike_at));
    ELSE
        spike_age_seconds := NULL;
    END IF;

    RETURN jsonb_build_object(
        'tonic', tonic,
        'phasic', phasic,
        'spike_at', spike_at,
        'spike_age_seconds', spike_age_seconds,
        'spike_rpe', COALESCE(state->>'dopamine_spike_rpe', '0')::float,
        'spike_trigger', state->>'dopamine_spike_trigger',
        -- Derived: effective dopamine level (tonic + decaying phasic)
        'effective', LEAST(1.0, GREATEST(0.0,
            tonic + phasic * GREATEST(0, 1.0 - COALESCE(spike_age_seconds, 9999) / 1800.0)
        ))
    );
END;
$$ LANGUAGE plpgsql STABLE;


-- ---------------------------------------------------------------------------
-- fire_dopamine_spike()  —  the core reinforcement function
--
-- Called by the subconscious when RPE exceeds threshold.
-- Atomically:
--   1. Shifts tonic dopamine via EMA
--   2. Boosts/suppresses activation + importance on recent memories
--   3. Spreads activation through neighborhoods
--   4. Modulates drives (curiosity/connection ↑ on positive, rest ↑ on negative)
--   5. Records spike in affective state
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION fire_dopamine_spike(
    p_rpe FLOAT,
    p_trigger TEXT DEFAULT '',
    p_retroactive_window INTERVAL DEFAULT INTERVAL '30 minutes'
)
RETURNS JSONB AS $$
DECLARE
    state JSONB;
    old_tonic FLOAT;
    new_tonic FLOAT;
    ema_alpha FLOAT := 0.15;  -- how fast tonic tracks phasic events
    boosted_count INT := 0;
    spread_count INT := 0;
    mem RECORD;
    neighbor_id UUID;
    neighbor_ids UUID[];
    current_boost FLOAT;
    boost_delta FLOAT;
    importance_delta FLOAT;
    abs_rpe FLOAT;
BEGIN
    abs_rpe := ABS(p_rpe);

    -- 1. Read current tonic
    state := get_current_affective_state();
    BEGIN old_tonic := NULLIF(state->>'dopamine_tonic', '')::float;
    EXCEPTION WHEN OTHERS THEN old_tonic := NULL; END;
    old_tonic := COALESCE(old_tonic, 0.5);

    -- EMA update: positive RPE pushes tonic up, negative pushes down
    -- Map RPE [-1,1] to target [0,1]: target = 0.5 + rpe * 0.5
    new_tonic := old_tonic * (1.0 - ema_alpha) + (0.5 + p_rpe * 0.5) * ema_alpha;
    new_tonic := LEAST(1.0, GREATEST(0.0, new_tonic));

    -- 2. Retroactive memory modulation
    -- Boost or suppress memories created within the retroactive window
    IF p_rpe > 0 THEN
        -- Positive RPE: enhance recent memories
        boost_delta := p_rpe * 0.4;       -- activation boost up to +0.4
        importance_delta := p_rpe * 0.12;  -- importance boost up to +0.12
    ELSE
        -- Negative RPE: suppress recent memories
        boost_delta := p_rpe * 0.25;       -- activation suppression up to -0.25
        importance_delta := p_rpe * 0.05;  -- slight importance reduction
    END IF;

    FOR mem IN
        SELECT id, metadata, importance
        FROM memories
        WHERE status = 'active'
          AND created_at >= CURRENT_TIMESTAMP - p_retroactive_window
        ORDER BY created_at DESC
        LIMIT 50  -- safety cap
    LOOP
        current_boost := COALESCE((mem.metadata->>'activation_boost')::float, 0);

        UPDATE memories
        SET metadata = jsonb_set(
                jsonb_set(
                    COALESCE(metadata, '{}'::jsonb),
                    '{activation_boost}',
                    to_jsonb(LEAST(1.0, GREATEST(0, current_boost + boost_delta)))
                ),
                '{dopamine_spike_rpe}',
                to_jsonb(p_rpe)
            ),
            importance = LEAST(1.0, GREATEST(0.1, importance + importance_delta))
        WHERE id = mem.id;

        boosted_count := boosted_count + 1;

        -- 3. Spread activation through neighborhoods (positive RPE only)
        IF p_rpe > 0 THEN
            SELECT ARRAY(
                SELECT (kv.value)::uuid
                FROM jsonb_each_text(
                    COALESCE(
                        (SELECT neighbors FROM memory_neighborhoods WHERE memory_id = mem.id),
                        '{}'::jsonb
                    )
                ) AS kv
                LIMIT 5  -- top 5 neighbors
            ) INTO neighbor_ids;

            IF neighbor_ids IS NOT NULL AND array_length(neighbor_ids, 1) > 0 THEN
                UPDATE memories
                SET metadata = jsonb_set(
                    COALESCE(metadata, '{}'::jsonb),
                    '{activation_boost}',
                    to_jsonb(LEAST(1.0, GREATEST(0,
                        COALESCE((metadata->>'activation_boost')::float, 0) + p_rpe * 0.15
                    )))
                )
                WHERE id = ANY(neighbor_ids)
                  AND status = 'active';

                GET DIAGNOSTICS spread_count = ROW_COUNT;
            END IF;
        END IF;
    END LOOP;

    -- 4. Modulate drives
    IF p_rpe > 0 THEN
        -- Positive RPE: satisfy curiosity + connection, reduce rest urgency
        -- Floor at each drive's baseline, matching satisfy_drive(): reward can
        -- calm a drive to rest, never push it below its resting level.
        UPDATE drives SET
            current_level = GREATEST(baseline, current_level - abs_rpe * 0.15),
            last_satisfied = CURRENT_TIMESTAMP
        WHERE name IN ('curiosity', 'connection');

        UPDATE drives SET
            current_level = GREATEST(baseline, current_level - abs_rpe * 0.1)
        WHERE name = 'rest';
    ELSE
        -- Negative RPE: increase rest drive, build coherence need
        UPDATE drives SET
            current_level = LEAST(1.0, current_level + abs_rpe * 0.1)
        WHERE name = 'rest';

        UPDATE drives SET
            current_level = LEAST(1.0, current_level + abs_rpe * 0.08)
        WHERE name = 'coherence';
    END IF;

    -- 5. Record spike in affective state
    PERFORM set_current_affective_state(jsonb_build_object(
        'dopamine_tonic', new_tonic,
        'dopamine_phasic', p_rpe,
        'dopamine_spike_at', CURRENT_TIMESTAMP,
        'dopamine_spike_rpe', p_rpe,
        'dopamine_spike_trigger', LEFT(COALESCE(p_trigger, ''), 500)
    ));

    BEGIN
        PERFORM record_reward_event(
            'dopamine_spike',
            p_rpe,
            abs_rpe,
            'dopamine',
            jsonb_build_object('trigger', LEFT(COALESCE(p_trigger, ''), 500))
        );
    EXCEPTION WHEN undefined_function THEN
        NULL;
    END;

    RETURN jsonb_build_object(
        'fired', true,
        'rpe', p_rpe,
        'tonic_old', old_tonic,
        'tonic_new', new_tonic,
        'memories_boosted', boosted_count,
        'neighbors_spread', spread_count,
        'trigger', LEFT(COALESCE(p_trigger, ''), 200)
    );
END;
$$ LANGUAGE plpgsql;


-- ---------------------------------------------------------------------------
-- dopamine_decay_multiplier()  —  pure function
--
-- High dopamine → slower activation decay (reward memories persist).
-- Returns 0.3 – 1.0.   At tonic 0.5 (neutral) returns ~0.65.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION dopamine_decay_multiplier(p_tonic FLOAT)
RETURNS FLOAT AS $$
BEGIN
    RETURN LEAST(1.0, GREATEST(0.3, 1.0 - COALESCE(p_tonic, 0.5) * 0.7));
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- ---------------------------------------------------------------------------
-- Override decay_activation_boosts()  —  dopamine-modulated decay
--
-- Original: flat -0.05 per call.
-- New: decay is scaled by dopamine_decay_multiplier(tonic).
-- High tonic dopamine → slower decay → reward memories stay salient longer.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION decay_activation_boosts(p_decay FLOAT DEFAULT 0.05)
RETURNS INT AS $$
DECLARE
    updated_count INT;
    da_state JSONB;
    tonic FLOAT;
    effective_decay FLOAT;
BEGIN
    -- Read dopamine tonic to modulate decay rate
    da_state := get_dopamine_state();
    BEGIN tonic := (da_state->>'tonic')::float;
    EXCEPTION WHEN OTHERS THEN tonic := 0.5; END;

    effective_decay := p_decay * dopamine_decay_multiplier(tonic);

    UPDATE memories
    SET metadata = jsonb_set(
        COALESCE(metadata, '{}'::jsonb),
        '{activation_boost}',
        to_jsonb(GREATEST(0, COALESCE((metadata->>'activation_boost')::float, 0) - effective_decay))
    )
    WHERE (metadata->>'activation_boost')::float > 0;

    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;


-- ---------------------------------------------------------------------------
-- Override update_mood()  —  dopamine-modulated mood persistence
--
-- Original: decay_rate from config (default 0.1).
-- New: high tonic dopamine reduces decay_rate (mood persists longer).
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION update_mood()
RETURNS VOID AS $$
DECLARE
    baseline JSONB;
    decay_rate FLOAT;
    current_state JSONB;
    recent RECORD;
    new_mood_valence FLOAT;
    new_mood_arousal FLOAT;
    da_state JSONB;
    tonic FLOAT;
BEGIN
    baseline := COALESCE(get_config('emotion.baseline'), '{}'::jsonb);
    decay_rate := COALESCE(NULLIF(baseline->>'decay_rate', '')::float, 0.1);

    -- Dopamine modulation: high tonic → slower mood decay (persist longer)
    da_state := get_dopamine_state();
    BEGIN tonic := (da_state->>'tonic')::float;
    EXCEPTION WHEN OTHERS THEN tonic := 0.5; END;
    -- At tonic 0.5 (neutral): no change.  At 1.0: decay halved.  At 0.0: decay doubled.
    decay_rate := decay_rate * (1.5 - COALESCE(tonic, 0.5));

    current_state := get_current_affective_state();

    -- Exact query from original update_mood — heartbeat_id is nested under context
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


-- ---------------------------------------------------------------------------
-- Override apply_emotional_context_to_memory()  —  tag dopamine at encoding
--
-- Existing behavior preserved; adds dopamine_at_encoding to metadata.
-- ---------------------------------------------------------------------------

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
    da_tonic FLOAT;
BEGIN
    meta := COALESCE(NEW.metadata, '{}'::jsonb);
    context := COALESCE(meta->'emotional_context', '{}'::jsonb);
    state := get_current_affective_state();

    BEGIN valence := NULLIF(meta->>'emotional_valence', '')::float;
    EXCEPTION WHEN OTHERS THEN valence := NULL; END;
    BEGIN arousal := NULLIF(context->>'arousal', '')::float;
    EXCEPTION WHEN OTHERS THEN arousal := NULL; END;
    BEGIN dominance := NULLIF(context->>'dominance', '')::float;
    EXCEPTION WHEN OTHERS THEN dominance := NULL; END;
    BEGIN intensity := NULLIF(context->>'intensity', '')::float;
    EXCEPTION WHEN OTHERS THEN intensity := NULL; END;

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

    -- Read current dopamine tonic for encoding tag
    BEGIN da_tonic := NULLIF(state->>'dopamine_tonic', '')::float;
    EXCEPTION WHEN OTHERS THEN da_tonic := NULL; END;
    da_tonic := COALESCE(da_tonic, 0.5);

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
        'emotional_valence', valence,
        'dopamine_at_encoding', da_tonic
    );

    -- Dopamine importance boost: memories encoded during high dopamine
    -- get a small importance bump (better initial encoding)
    IF da_tonic > 0.6 THEN
        NEW.importance := LEAST(1.0, NEW.importance + (da_tonic - 0.6) * 0.15);
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


-- ---------------------------------------------------------------------------
-- drift_dopamine_tonic()  —  called during maintenance to drift toward 0.5
--
-- Tonic dopamine naturally regresses toward baseline (0.5) over time.
-- This prevents runaway positive or negative states.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION drift_dopamine_tonic(p_drift_rate FLOAT DEFAULT 0.02)
RETURNS JSONB AS $$
DECLARE
    state JSONB;
    old_tonic FLOAT;
    new_tonic FLOAT;
BEGIN
    state := get_current_affective_state();
    BEGIN old_tonic := NULLIF(state->>'dopamine_tonic', '')::float;
    EXCEPTION WHEN OTHERS THEN old_tonic := NULL; END;
    old_tonic := COALESCE(old_tonic, 0.5);

    -- Drift toward 0.5 (homeostasis)
    new_tonic := old_tonic + (0.5 - old_tonic) * COALESCE(p_drift_rate, 0.02);
    new_tonic := LEAST(1.0, GREATEST(0.0, new_tonic));

    IF ABS(new_tonic - old_tonic) > 0.001 THEN
        PERFORM set_current_affective_state(jsonb_build_object(
            'dopamine_tonic', new_tonic
        ));
    END IF;

    RETURN jsonb_build_object(
        'old_tonic', old_tonic,
        'new_tonic', new_tonic,
        'drifted', ABS(new_tonic - old_tonic) > 0.001
    );
END;
$$ LANGUAGE plpgsql;


-- ---------------------------------------------------------------------------
-- Override run_subconscious_maintenance()  —  add dopamine tonic drift
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION run_subconscious_maintenance(p_params JSONB DEFAULT '{}'::jsonb)
RETURNS JSONB AS $$
DECLARE
    got_lock BOOLEAN;
    min_imp FLOAT;
    min_acc INT;
    neighborhood_batch INT;
    cache_days INT;
    wm_stats JSONB;
    recomputed INT;
    cache_deleted INT;
    bg_processed INT;
    activation_decay INT;
    activation_cleaned INT;
    ready_transformations JSONB;
    dopamine_drift JSONB;
BEGIN
    IF is_agent_terminated() THEN
        RETURN jsonb_build_object('skipped', true, 'reason', 'terminated');
    END IF;
    got_lock := pg_try_advisory_lock(hashtext('hexis_subconscious_maintenance'));
    IF NOT got_lock THEN
        RETURN jsonb_build_object('skipped', true, 'reason', 'locked');
    END IF;
    min_imp := COALESCE(
        NULLIF(p_params->>'working_memory_promote_min_importance', '')::float,
        get_config_float('maintenance.working_memory_promote_min_importance'),
        0.75
    );
    min_acc := COALESCE(
        NULLIF(p_params->>'working_memory_promote_min_accesses', '')::int,
        get_config_int('maintenance.working_memory_promote_min_accesses'),
        3
    );
    neighborhood_batch := COALESCE(
        NULLIF(p_params->>'neighborhood_batch_size', '')::int,
        get_config_int('maintenance.neighborhood_batch_size'),
        10
    );
    cache_days := COALESCE(
        NULLIF(p_params->>'embedding_cache_older_than_days', '')::int,
        get_config_int('maintenance.embedding_cache_older_than_days'),
        7
    );

    wm_stats := cleanup_working_memory(min_imp, min_acc);
    recomputed := batch_recompute_neighborhoods(neighborhood_batch);
    cache_deleted := cleanup_embedding_cache((cache_days || ' days')::interval);
    bg_processed := process_background_searches();
    activation_decay := decay_activation_boosts();  -- now dopamine-modulated
    activation_cleaned := cleanup_memory_activations();
    PERFORM update_mood();                           -- now dopamine-modulated
    ready_transformations := check_transformation_readiness();
    dopamine_drift := drift_dopamine_tonic();        -- new: homeostatic drift

    -- Memory retention (compression-native fade ladder): consolidate aged episodes
    -- into gists, then prune past-grace originals. No-op unless retention.enabled.
    -- Guarded so a failure never breaks the maintenance tick.
    BEGIN
        PERFORM run_memory_rest();
        PERFORM run_retention_gc();
        PERFORM request_stale_document_fades();  -- ask the user before fading their documents
    EXCEPTION WHEN OTHERS THEN
        RAISE WARNING 'memory retention pass failed: %', SQLERRM;
    END;

    UPDATE maintenance_state
    SET last_maintenance_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;

    PERFORM pg_advisory_unlock(hashtext('hexis_subconscious_maintenance'));

    RETURN jsonb_build_object(
        'success', true,
        'working_memory', wm_stats,
        'neighborhoods_recomputed', COALESCE(recomputed, 0),
        'embedding_cache_deleted', COALESCE(cache_deleted, 0),
        'background_searches_processed', COALESCE(bg_processed, 0),
        'activation_boosts_decayed', COALESCE(activation_decay, 0),
        'memory_activations_cleaned', COALESCE(activation_cleaned, 0),
        'transformations_ready', COALESCE(ready_transformations, '[]'::jsonb),
        'dopamine_drift', COALESCE(dopamine_drift, '{}'::jsonb),
        'ran_at', CURRENT_TIMESTAMP
    );
EXCEPTION
    WHEN OTHERS THEN
        PERFORM pg_advisory_unlock(hashtext('hexis_subconscious_maintenance'));
        RAISE;
END;
$$ LANGUAGE plpgsql;


CREATE TABLE IF NOT EXISTS reward_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kind TEXT NOT NULL,
    valence FLOAT NOT NULL DEFAULT 0.0 CHECK (valence >= -1.0 AND valence <= 1.0),
    salience FLOAT NOT NULL DEFAULT 0.5 CHECK (salience >= 0.0 AND salience <= 1.0),
    source TEXT NOT NULL DEFAULT 'unknown',
    expected FLOAT CHECK (expected IS NULL OR (expected >= -1.0 AND expected <= 1.0)),
    actual FLOAT CHECK (actual IS NULL OR (actual >= -1.0 AND actual <= 1.0)),
    rpe FLOAT CHECK (rpe IS NULL OR (rpe >= -1.0 AND rpe <= 1.0)),
    memory_id UUID REFERENCES memories(id) ON DELETE SET NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_reward_events_kind_created
    ON reward_events (kind, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reward_events_rpe_created
    ON reward_events (rpe, created_at DESC)
    WHERE rpe IS NOT NULL;

CREATE OR REPLACE FUNCTION record_reward_event(
    p_kind TEXT,
    p_valence FLOAT,
    p_salience FLOAT DEFAULT 0.5,
    p_source TEXT DEFAULT 'agent',
    p_metadata JSONB DEFAULT '{}'::jsonb,
    p_memory_id UUID DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    row_event reward_events%ROWTYPE;
    val FLOAT := LEAST(1.0, GREATEST(-1.0, COALESCE(p_valence, 0.0)));
    sal FLOAT := LEAST(1.0, GREATEST(0.0, COALESCE(p_salience, ABS(COALESCE(p_valence, 0.0)), 0.5)));
BEGIN
    INSERT INTO reward_events (kind, valence, salience, source, memory_id, metadata)
    VALUES (
        COALESCE(NULLIF(btrim(p_kind), ''), 'reward'),
        val,
        sal,
        COALESCE(NULLIF(btrim(p_source), ''), 'agent'),
        p_memory_id,
        COALESCE(p_metadata, '{}'::jsonb)
    )
    RETURNING * INTO row_event;

    IF row_event.kind <> 'dopamine_spike'
       AND sal >= COALESCE(get_config_float('reward.dopamine_spike_salience_threshold'), 0.65) THEN
        PERFORM fire_dopamine_spike(
            val * sal,
            COALESCE(NULLIF(p_kind, ''), 'reward') || ': ' || COALESCE(p_metadata->>'summary', '')
        );
    END IF;

    RETURN jsonb_build_object(
        'event_id', row_event.id::text,
        'kind', row_event.kind,
        'valence', row_event.valence,
        'salience', row_event.salience,
        'source', row_event.source
    );
END;
$$;

CREATE OR REPLACE FUNCTION record_prediction_error(
    p_expected FLOAT,
    p_actual FLOAT,
    p_kind TEXT DEFAULT 'prediction_error',
    p_source TEXT DEFAULT 'agent',
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    row_event reward_events%ROWTYPE;
    expected_value FLOAT := LEAST(1.0, GREATEST(-1.0, COALESCE(p_expected, 0.0)));
    actual_value FLOAT := LEAST(1.0, GREATEST(-1.0, COALESCE(p_actual, 0.0)));
    rpe_value FLOAT;
BEGIN
    rpe_value := LEAST(1.0, GREATEST(-1.0, actual_value - expected_value));

    INSERT INTO reward_events (
        kind, valence, salience, source, expected, actual, rpe, metadata
    )
    VALUES (
        COALESCE(NULLIF(btrim(p_kind), ''), 'prediction_error'),
        rpe_value,
        ABS(rpe_value),
        COALESCE(NULLIF(btrim(p_source), ''), 'agent'),
        expected_value,
        actual_value,
        rpe_value,
        COALESCE(p_metadata, '{}'::jsonb)
    )
    RETURNING * INTO row_event;

    IF ABS(rpe_value) >= COALESCE(get_config_float('reward.rpe_spike_threshold'), 0.35) THEN
        PERFORM fire_dopamine_spike(
            rpe_value,
            COALESCE(NULLIF(p_kind, ''), 'prediction_error') || ': ' || COALESCE(p_metadata->>'summary', '')
        );
    END IF;

    RETURN jsonb_build_object(
        'event_id', row_event.id::text,
        'expected', expected_value,
        'actual', actual_value,
        'rpe', rpe_value,
        'dopamine_triggered', ABS(rpe_value) >= COALESCE(get_config_float('reward.rpe_spike_threshold'), 0.35)
    );
END;
$$;

CREATE OR REPLACE FUNCTION record_social_reward(
    p_signal TEXT,
    p_valence FLOAT DEFAULT 0.4,
    p_salience FLOAT DEFAULT 0.5,
    p_source TEXT DEFAULT 'conversation',
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE sql
AS $$
    SELECT record_reward_event(
        'social:' || COALESCE(NULLIF(btrim(p_signal), ''), 'interaction'),
        p_valence,
        p_salience,
        p_source,
        COALESCE(p_metadata, '{}'::jsonb)
    );
$$;

CREATE OR REPLACE FUNCTION apply_appraisal_reward_effects(p_signals JSONB)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    signals JSONB := COALESCE(p_signals, '{}'::jsonb);
    emo JSONB := CASE WHEN jsonb_typeof(signals->'emotional_state') = 'object'
                      THEN signals->'emotional_state' ELSE '{}'::jsonb END;
    primary_emotion TEXT := lower(COALESCE(emo->>'primary_emotion', ''));
    emotion_family TEXT := normalize_emotion_family(emo->>'family');
    valence FLOAT := COALESCE(NULLIF(emo->>'valence', '')::float, 0.0);
    intensity FLOAT := COALESCE(NULLIF(emo->>'intensity', '')::float, 0.0);
    confidence FLOAT := COALESCE(NULLIF(emo->>'confidence', '')::float, 0.0);
    recorded JSONB := NULL;
BEGIN
    valence := LEAST(1.0, GREATEST(-1.0, valence));
    intensity := LEAST(1.0, GREATEST(0.0, intensity));
    confidence := LEAST(1.0, GREATEST(0.0, confidence));

    IF valence >= 0.35
       AND intensity >= 0.35
       AND confidence >= COALESCE(get_config_float('subconscious.min_signal_confidence'), 0.6)
       AND emotion_family_serves(emotion_family, 'social_reward') THEN
        recorded := record_social_reward(
            primary_emotion,
            valence,
            intensity,
            'inline_appraisal',
            jsonb_build_object(
                'emotional_state', emo,
                'subconscious_response', left(COALESCE(signals->>'subconscious_response', ''), 300)
            )
        );
    END IF;

    RETURN jsonb_build_object(
        'recorded', recorded IS NOT NULL,
        'event', COALESCE(recorded, '{}'::jsonb),
        'primary_emotion', primary_emotion,
        'family', emotion_family,
        'valence', valence,
        'intensity', intensity,
        'confidence', confidence
    );
EXCEPTION WHEN OTHERS THEN
    RAISE LOG 'apply_appraisal_reward_effects failed: %', SQLERRM;
    RETURN jsonb_build_object('recorded', false, 'error', SQLERRM);
END;
$$;

CREATE OR REPLACE FUNCTION reward_state_summary(
    p_since INTERVAL DEFAULT INTERVAL '24 hours'
) RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    since_ts TIMESTAMPTZ := CURRENT_TIMESTAMP - COALESCE(p_since, INTERVAL '24 hours');
    event_count INT;
    avg_val NUMERIC;
    avg_sal NUMERIC;
    avg_rpe NUMERIC;
    by_kind JSONB;
BEGIN
    SELECT count(*)::int,
           COALESCE(round(avg(valence)::numeric, 4), 0),
           COALESCE(round(avg(salience)::numeric, 4), 0),
           COALESCE(round(avg(rpe)::numeric, 4), 0)
    INTO event_count, avg_val, avg_sal, avg_rpe
    FROM reward_events
    WHERE created_at >= since_ts;

    SELECT COALESCE(jsonb_object_agg(kind, count), '{}'::jsonb)
    INTO by_kind
    FROM (
        SELECT kind, count(*)::int AS count
        FROM reward_events
        WHERE created_at >= since_ts
        GROUP BY kind
    ) grouped;

    RETURN jsonb_build_object(
        'since', CURRENT_TIMESTAMP - COALESCE(p_since, INTERVAL '24 hours'),
        'events', event_count,
        'avg_valence', avg_val,
        'avg_salience', avg_sal,
        'avg_rpe', avg_rpe,
        'by_kind', by_kind,
        'dopamine', get_dopamine_state()
    );
END;
$$;

SET check_function_bodies = on;
