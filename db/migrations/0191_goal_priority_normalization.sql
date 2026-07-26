-- Normalize heartbeat goal-change priority aliases at the DB boundary.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

CREATE OR REPLACE FUNCTION normalize_goal_priority(p_priority TEXT)
RETURNS goal_priority AS $$
DECLARE
    normalized TEXT := lower(trim(COALESCE(p_priority, '')));
BEGIN
    IF normalized = '' THEN
        RETURN NULL;
    END IF;

    normalized := replace(normalized, '-', '_');
    normalized := replace(normalized, ' ', '_');

    RETURN CASE normalized
        WHEN 'active' THEN 'active'::goal_priority
        WHEN 'high' THEN 'active'::goal_priority
        WHEN 'urgent' THEN 'active'::goal_priority
        WHEN 'now' THEN 'active'::goal_priority
        WHEN 'queued' THEN 'queued'::goal_priority
        WHEN 'queue' THEN 'queued'::goal_priority
        WHEN 'medium' THEN 'queued'::goal_priority
        WHEN 'normal' THEN 'queued'::goal_priority
        WHEN 'later' THEN 'queued'::goal_priority
        WHEN 'backburner' THEN 'backburner'::goal_priority
        WHEN 'back_burner' THEN 'backburner'::goal_priority
        WHEN 'low' THEN 'backburner'::goal_priority
        WHEN 'defer' THEN 'backburner'::goal_priority
        WHEN 'deferred' THEN 'backburner'::goal_priority
        WHEN 'completed' THEN 'completed'::goal_priority
        WHEN 'complete' THEN 'completed'::goal_priority
        WHEN 'done' THEN 'completed'::goal_priority
        WHEN 'resolved' THEN 'completed'::goal_priority
        WHEN 'abandoned' THEN 'abandoned'::goal_priority
        WHEN 'abandon' THEN 'abandoned'::goal_priority
        WHEN 'cancelled' THEN 'abandoned'::goal_priority
        WHEN 'canceled' THEN 'abandoned'::goal_priority
        WHEN 'dropped' THEN 'abandoned'::goal_priority
        ELSE NULL
    END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION apply_goal_changes(p_changes JSONB)
RETURNS JSONB AS $$
DECLARE
    change JSONB;
    goal_id UUID;
    change_kind goal_priority;
    raw_priority TEXT;
    reason TEXT;
    applied INT := 0;
    skipped INT := 0;
BEGIN
    IF p_changes IS NULL OR jsonb_typeof(p_changes) <> 'array' THEN
        RETURN jsonb_build_object('applied', 0, 'skipped', 0);
    END IF;

    FOR change IN SELECT * FROM jsonb_array_elements(p_changes)
    LOOP
        BEGIN
            goal_id := NULLIF(change->>'goal_id', '')::uuid;
        EXCEPTION
            WHEN OTHERS THEN
                goal_id := NULL;
        END;
        IF goal_id IS NULL THEN
            skipped := skipped + 1;
            CONTINUE;
        END IF;

        raw_priority := COALESCE(
            NULLIF(change->>'change', ''),
            NULLIF(change->>'new_priority', ''),
            NULLIF(change->>'priority', '')
        );
        change_kind := normalize_goal_priority(raw_priority);
        IF change_kind IS NULL THEN
            skipped := skipped + 1;
            RAISE LOG 'Skipping invalid goal priority change for goal %: %', goal_id, raw_priority;
            CONTINUE;
        END IF;

        reason := COALESCE(change->>'reason', '');
        PERFORM change_goal_priority(goal_id, change_kind, reason);
        IF change_kind = 'completed' THEN
            BEGIN
                PERFORM record_reward_event(
                    'goal_completed',
                    0.75,
                    0.7,
                    'goal',
                    jsonb_build_object(
                        'goal_id', goal_id,
                        'reason', NULLIF(reason, ''),
                        'change', change_kind::text
                    ),
                    goal_id
                );
            EXCEPTION WHEN OTHERS THEN
                RAISE LOG 'record_reward_event failed for completed goal %: %', goal_id, SQLERRM;
            END;
        ELSIF change_kind = 'abandoned' THEN
            BEGIN
                PERFORM record_prediction_error(
                    0.2,
                    -0.3,
                    'goal_abandoned',
                    'goal',
                    jsonb_build_object(
                        'goal_id', goal_id,
                        'reason', NULLIF(reason, ''),
                        'change', change_kind::text
                    )
                );
            EXCEPTION WHEN OTHERS THEN
                RAISE LOG 'record_prediction_error failed for abandoned goal %: %', goal_id, SQLERRM;
            END;
        END IF;
        applied := applied + 1;
    END LOOP;

    RETURN jsonb_build_object('applied', applied, 'skipped', skipped);
END;
$$ LANGUAGE plpgsql;

SET check_function_bodies = on;
