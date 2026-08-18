-- Two baseline-drift fixes surfaced by CI once the schema-guard noise cleared:
-- 1. hmx_apply_reembed_batch declared a variable named embedding_model, which
--    became ambiguous when memories.embedding_model was added -- every HMX
--    re-embed batch failed with "column reference is ambiguous".
-- 2. fire_dopamine_spike floored reward-driven drive satisfaction at 0 instead
--    of the drive's baseline, contradicting satisfy_drive()'s own floor and
--    letting rewards push drives below their resting level.
SET search_path = public, ag_catalog, "$user";

CREATE OR REPLACE FUNCTION hmx_apply_reembed_batch(p_memory_ids UUID[])
RETURNS JSONB AS $$
DECLARE
    valid_ids UUID[];
    contents TEXT[];
    embeddings vector[];
    v_embedding_model TEXT;
    derivative_result JSONB;
    index INT;
    updated_count INT := 0;
    affected INT;
BEGIN
    SELECT array_agg(m.id ORDER BY requested.ordinality),
           array_agg(m.content ORDER BY requested.ordinality)
    INTO valid_ids, contents
    FROM unnest(COALESCE(p_memory_ids, '{}'::uuid[])) WITH ORDINALITY AS requested(id, ordinality)
    JOIN memories m ON m.id = requested.id
    WHERE m.status = 'active'
      AND m.metadata->>'embedding_status' = 'in_progress'
      AND hmx_is_imported_memory(m.metadata);

    IF COALESCE(cardinality(valid_ids), 0) = 0 THEN
        RETURN jsonb_build_object('embedded', 0, 'memory_ids', '[]'::jsonb);
    END IF;

    embeddings := get_embedding(contents);
    IF cardinality(embeddings) <> cardinality(valid_ids) THEN
        RAISE EXCEPTION 'HMX embedding response size mismatch: expected %, got %',
            cardinality(valid_ids), cardinality(embeddings);
    END IF;

    v_embedding_model := COALESCE(
        (SELECT value #>> '{}' FROM config WHERE key = 'embedding.model_id'),
        'unknown'
    );

    FOR index IN 1..cardinality(valid_ids) LOOP
        UPDATE memories m
        SET embedding = embeddings[index],
            metadata = (COALESCE(m.metadata, '{}'::jsonb)
                - 'embedding_error'
                - 'embedding_claimed_at')
                || jsonb_build_object(
                    'embedding_status', 'embedded',
                    'embedding_completed_at', CURRENT_TIMESTAMP,
                    'embedding_model', v_embedding_model
                ),
            updated_at = CURRENT_TIMESTAMP
        WHERE m.id = valid_ids[index]
          AND m.status = 'active'
          AND m.metadata->>'embedding_status' = 'in_progress';
        GET DIAGNOSTICS affected = ROW_COUNT;
        updated_count := updated_count + affected;
    END LOOP;

    derivative_result := hmx_refresh_reembed_derivatives(valid_ids);
    RETURN jsonb_build_object(
        'embedded', updated_count,
        'memory_ids', to_jsonb(valid_ids),
        'derivatives', derivative_result
    );
END;
$$ LANGUAGE plpgsql;

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
