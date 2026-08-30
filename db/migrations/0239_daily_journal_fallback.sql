-- Daily journal fallback (issue #114): journal_entries had zero rows ever,
-- despite the prompt nagging the agent to journal and every write/read/search
-- path working end to end. Journaling stayed 100% opt-in and the agent simply
-- never opted in. This adds one narrow, DB-owned exception: subconscious
-- maintenance may write a single minimal entry for a local day that had real
-- activity (meaningful episodic memories) and nothing journaled deliberately.
-- It fires at most once per local date, never overwrites/duplicates a
-- deliberate entry, and is fully guarded so a failure never breaks maintenance.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

INSERT INTO config_defaults (key, value, description) VALUES
    ('maintenance.daily_journal_fallback_enabled', 'true'::jsonb, 'If a local day had meaningful activity and nothing was journaled deliberately, write one minimal fallback entry (issue #114)'),
    ('maintenance.daily_journal_fallback_hour', '21'::jsonb, 'Local hour (agent.timezone) after which the daily journal fallback may consider stepping in'),
    ('maintenance.daily_journal_fallback_min_episodes', '3'::jsonb, 'Minimum notable (importance >= 0.5) episodic memories in the local day for the fallback to consider it "meaningful activity"')
ON CONFLICT (key) DO NOTHING;

CREATE OR REPLACE FUNCTION maybe_write_daily_journal_fallback()
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_tz TEXT := COALESCE(
        NULLIF(get_config_text('agent.timezone'), ''),
        NULLIF(get_config_text('heartbeat.timezone'), ''),
        'UTC'
    );
    v_local_ts TIMESTAMP;
    v_local_date DATE;
    v_local_hour INT;
    v_state JSONB;
    v_already_journaled BOOLEAN;
    v_min_count INT;
    v_count INT;
    v_samples TEXT[];
    v_content TEXT;
    v_entry_id UUID;
BEGIN
    IF NOT COALESCE(get_config_bool('maintenance.daily_journal_fallback_enabled'), true) THEN
        RETURN jsonb_build_object('skipped', true, 'reason', 'disabled');
    END IF;

    BEGIN
        v_local_ts := CURRENT_TIMESTAMP AT TIME ZONE v_tz;
    EXCEPTION WHEN OTHERS THEN
        v_local_ts := CURRENT_TIMESTAMP AT TIME ZONE 'UTC';
    END;
    v_local_date := v_local_ts::date;
    v_local_hour := EXTRACT(HOUR FROM v_local_ts)::int;

    -- Give the agent the whole local day to journal deliberately before this
    -- fallback ever considers stepping in.
    IF v_local_hour < COALESCE(get_config_int('maintenance.daily_journal_fallback_hour'), 21) THEN
        RETURN jsonb_build_object('skipped', true, 'reason', 'too_early', 'local_hour', v_local_hour);
    END IF;

    v_state := COALESCE(get_state('daily_journal_fallback'), '{}'::jsonb);
    IF v_state->>'last_checked_date' = v_local_date::text THEN
        RETURN jsonb_build_object('skipped', true, 'reason', 'already_checked_today');
    END IF;

    -- Record the attempt regardless of outcome -- a quiet day, or one already
    -- journaled, should not be re-evaluated every maintenance tick until midnight.
    PERFORM set_state('daily_journal_fallback', jsonb_build_object('last_checked_date', v_local_date::text));

    SELECT EXISTS (
        SELECT 1 FROM journal_entries
        WHERE (written_at AT TIME ZONE v_tz)::date = v_local_date
    ) INTO v_already_journaled;
    IF v_already_journaled THEN
        RETURN jsonb_build_object('skipped', true, 'reason', 'already_journaled', 'date', v_local_date);
    END IF;

    v_min_count := GREATEST(COALESCE(get_config_int('maintenance.daily_journal_fallback_min_episodes'), 3), 1);
    SELECT count(*)::int INTO v_count
    FROM memories
    WHERE type = 'episodic'
      AND status = 'active'
      AND importance >= 0.5
      AND (created_at AT TIME ZONE v_tz)::date = v_local_date;

    IF v_count < v_min_count THEN
        RETURN jsonb_build_object('skipped', true, 'reason', 'not_enough_activity',
                                   'episodes', v_count, 'threshold', v_min_count);
    END IF;

    SELECT array_agg(top.snippet) INTO v_samples
    FROM (
        SELECT left(content, 140) AS snippet
        FROM memories
        WHERE type = 'episodic'
          AND status = 'active'
          AND importance >= 0.5
          AND (created_at AT TIME ZONE v_tz)::date = v_local_date
        ORDER BY importance DESC, created_at DESC
        LIMIT 5
    ) top;

    v_content := 'A day worth remembering, though I never stopped to write it myself: '
        || array_to_string(v_samples, ' ~ ');

    v_entry_id := write_journal_entry(
        p_content  := v_content,
        p_title    := 'Auto-noted -- ' || to_char(v_local_date, 'FMMonth DD, YYYY'),
        p_tags     := ARRAY['auto-generated'],
        p_metadata := jsonb_build_object(
            'source', 'subconscious_daily_fallback',
            'episode_count', v_count,
            'date', v_local_date::text
        )
    );

    RETURN jsonb_build_object(
        'journaled', true, 'entry_id', v_entry_id,
        'episode_count', v_count, 'date', v_local_date
    );
END;
$$;

-- Re-publish run_subconscious_maintenance (db/28's version, which is the live
-- one) with the fallback wired in, guarded so a failure never breaks the tick.
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
    journal_fallback JSONB;
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
    activation_decay := decay_activation_boosts();  -- dopamine-modulated
    activation_cleaned := cleanup_memory_activations();
    PERFORM update_mood();                           -- dopamine-modulated
    ready_transformations := check_transformation_readiness();
    dopamine_drift := drift_dopamine_tonic();        -- homeostatic drift

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

    -- Daily journal fallback (issue #114): a day with real activity and no
    -- deliberate entry gets one minimal one. Guarded so a failure never breaks
    -- the maintenance tick.
    BEGIN
        journal_fallback := maybe_write_daily_journal_fallback();
    EXCEPTION WHEN OTHERS THEN
        RAISE WARNING 'daily journal fallback failed: %', SQLERRM;
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
        'daily_journal_fallback', COALESCE(journal_fallback, jsonb_build_object('skipped', true, 'reason', 'error')),
        'ran_at', CURRENT_TIMESTAMP
    );
EXCEPTION
    WHEN OTHERS THEN
        PERFORM pg_advisory_unlock(hashtext('hexis_subconscious_maintenance'));
        RAISE;
END;
$$ LANGUAGE plpgsql;
