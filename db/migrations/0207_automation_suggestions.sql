-- Phase 3: durable, consent-first automation suggestions.
SET search_path = public, ag_catalog, "$user";

-- Consent-first automation proposals. A proposal is inert until the user
-- accepts it; dedup_key is deliberately permanent so "Not for me" latches.
CREATE TABLE IF NOT EXISTS automation_suggestions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source TEXT NOT NULL
        CHECK (source IN ('catalog', 'blueprint', 'usage', 'connector')),
    dedup_key TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    rationale TEXT NOT NULL,
    task_spec JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'accepted', 'dismissed')),
    scheduled_task_id UUID REFERENCES scheduled_tasks(id) ON DELETE SET NULL,
    outbox_message_id UUID,
    decision_channel TEXT,
    decision_actor TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    decided_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_automation_suggestions_status_created
    ON automation_suggestions (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_automation_suggestions_task
    ON automation_suggestions (scheduled_task_id)
    WHERE scheduled_task_id IS NOT NULL;

-- Curated proposals are data, not branches in application code. Preconditions
-- are evaluated against the live connector registry before a suggestion is
-- filed; catalog rows never create schedules themselves.
CREATE TABLE IF NOT EXISTS automation_suggestion_catalog (
    dedup_key TEXT PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'catalog'
        CHECK (source IN ('catalog', 'connector')),
    title TEXT NOT NULL,
    rationale TEXT NOT NULL,
    task_spec JSONB NOT NULL,
    precondition TEXT NOT NULL DEFAULT 'none'
        CHECK (precondition IN ('none', 'gmail_connected', 'calendar_connected')),
    sort_order INTEGER NOT NULL DEFAULT 100,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_automation_suggestion_catalog_enabled
    ON automation_suggestion_catalog (enabled, sort_order, dedup_key);
-- Consent-first automation suggestions: curated, connector, usage, and skill
-- blueprint sources all converge on one durable accept/dismiss lifecycle.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

INSERT INTO config_defaults (key, value, description) VALUES
    ('automation.suggestions.enabled', 'true'::jsonb,
     'Allow Hexis to file inert automation suggestions; schedules are created only after explicit acceptance.'),
    ('automation.suggestions.catalog_enabled', 'true'::jsonb,
     'Offer curated starter routines whose declared live preconditions are satisfied.'),
    ('automation.suggestions.connector_enabled', 'true'::jsonb,
     'Offer a relevant routine when a supported connector becomes connected.'),
    ('automation.suggestions.blueprint_enabled', 'true'::jsonb,
     'Register blueprint blocks found in installed skills as suggestions, never as schedules.'),
    ('automation.suggestions.usage_enabled', 'true'::jsonb,
     'Allow the separately opted-in skill-improvement review to suggest a routine after three matching asks.'),
    ('automation.suggestions.usage_min_confidence', '0.85'::jsonb,
     'Minimum model confidence for a recurring-usage automation suggestion.'),
    ('automation.suggestions.refresh_interval_seconds', '60'::jsonb,
     'Minimum seconds between catalog and installed-skill blueprint refreshes.')
ON CONFLICT (key) DO UPDATE SET
    value = EXCLUDED.value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO automation_suggestion_catalog (
    dedup_key, source, title, rationale, task_spec, precondition, sort_order, metadata
) VALUES
    (
        'catalog:morning-briefing',
        'catalog',
        'Morning briefing',
        'A dependable morning prompt makes it easier to review today''s schedule, priorities, and open loops before the day gets noisy.',
        '{
          "name": "Morning briefing",
          "description": "Prompt me to ask Hexis for a current daily briefing.",
          "schedule": "daily:08:00",
          "action_kind": "queue_user_message",
          "message": "Morning briefing time — open Hexis and ask for today''s briefing to review your schedule, priorities, and open loops.",
          "delivery_mode": "outbox"
        }'::jsonb,
        'none',
        10,
        '{"schedule_template":"morning"}'::jsonb
    ),
    (
        'catalog:evening-wind-down',
        'catalog',
        'Evening wind-down',
        'A short end-of-day pause can catch unfinished commitments and make tomorrow easier to start.',
        '{
          "name": "Evening wind-down",
          "description": "Prompt me to close out the day with Hexis.",
          "schedule": "daily:21:00",
          "action_kind": "queue_user_message",
          "message": "Evening wind-down — open Hexis to capture loose ends, note what mattered today, and choose tomorrow''s first step.",
          "delivery_mode": "outbox"
        }'::jsonb,
        'none',
        20,
        '{"schedule_template":"evening"}'::jsonb
    ),
    (
        'catalog:weekly-review',
        'catalog',
        'Weekly review',
        'A weekly review helps reconcile plans with what actually happened and keeps stale commitments from disappearing silently.',
        '{
          "name": "Weekly review",
          "description": "Prompt me to review the week with Hexis before planning the next one.",
          "schedule": "weekly:sunday:17:00",
          "action_kind": "queue_user_message",
          "message": "Weekly review time — open Hexis to review wins, blockers, open commitments, and the priorities you want for next week.",
          "delivery_mode": "outbox"
        }'::jsonb,
        'none',
        30,
        '{"schedule_template":"weekly_review"}'::jsonb
    ),
    (
        'connector:gmail:important-mail-monitor',
        'connector',
        'Important-mail check',
        'Gmail is connected, so a regular prompt can help you ask Hexis to surface important unread mail before it is buried.',
        '{
          "name": "Important-mail check",
          "description": "Prompt me to have Hexis scan connected Gmail for important unread mail.",
          "schedule": "daily:09:00",
          "action_kind": "queue_user_message",
          "message": "Important-mail check — open Hexis and ask me to scan connected Gmail for important unread messages.",
          "delivery_mode": "outbox"
        }'::jsonb,
        'gmail_connected',
        100,
        '{"connector_id":"gmail","schedule_template":"morning_plus_one"}'::jsonb
    )
ON CONFLICT (dedup_key) DO UPDATE SET
    source = EXCLUDED.source,
    title = EXCLUDED.title,
    rationale = EXCLUDED.rationale,
    task_spec = EXCLUDED.task_spec,
    precondition = EXCLUDED.precondition,
    sort_order = EXCLUDED.sort_order,
    metadata = automation_suggestion_catalog.metadata || EXCLUDED.metadata,
    updated_at = CURRENT_TIMESTAMP;

CREATE OR REPLACE FUNCTION automation_suggestion_code(p_id UUID)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE
STRICT
AS $$
    SELECT upper(left(replace(p_id::text, '-', ''), 8));
$$;

-- Materialize catalog timing from the live active-hours preference when it
-- carries a useful signal. The all-day default is intentionally treated as
-- "no preference" and falls back to the catalog's expert-chosen times.
CREATE OR REPLACE FUNCTION materialize_automation_catalog_task_spec(
    p_task_spec JSONB,
    p_template TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_spec JSONB := COALESCE(p_task_spec, '{}'::jsonb);
    v_hours TEXT := NULLIF(btrim(get_config_text('heartbeat.active_hours')), '');
    v_parts TEXT[];
    v_start TIME;
    v_end TIME;
    v_time TIME;
BEGIN
    IF COALESCE(p_template, '') NOT IN ('morning', 'morning_plus_one', 'evening')
       OR v_hours IS NULL
       OR v_hours = '00:00-23:59' THEN
        RETURN v_spec;
    END IF;

    BEGIN
        v_parts := string_to_array(v_hours, '-');
        IF array_length(v_parts, 1) <> 2 THEN
            RETURN v_spec;
        END IF;
        v_start := btrim(v_parts[1])::time;
        v_end := btrim(v_parts[2])::time;

        IF p_template = 'morning' AND v_start >= '05:00'::time AND v_start <= '11:30'::time THEN
            v_time := v_start;
        ELSIF p_template = 'morning_plus_one' AND v_start >= '05:00'::time AND v_start <= '10:30'::time THEN
            v_time := v_start + INTERVAL '1 hour';
        ELSIF p_template = 'evening' AND v_end >= '18:00'::time THEN
            v_time := v_end - INTERVAL '1 hour';
        ELSE
            RETURN v_spec;
        END IF;
        RETURN jsonb_set(v_spec, '{schedule}', to_jsonb('daily:' || to_char(v_time, 'HH24:MI')), TRUE);
    EXCEPTION WHEN OTHERS THEN
        RETURN v_spec;
    END;
END;
$$;

CREATE OR REPLACE FUNCTION propose_automation(
    p_source TEXT,
    p_dedup_key TEXT,
    p_title TEXT,
    p_rationale TEXT,
    p_task_spec JSONB,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_source TEXT := lower(btrim(COALESCE(p_source, '')));
    v_key TEXT := btrim(COALESCE(p_dedup_key, ''));
    v_spec JSONB;
    v_row automation_suggestions%ROWTYPE;
    v_outbox_id UUID;
    v_code TEXT;
    v_schedule TEXT;
    v_message TEXT;
    v_notification_error TEXT;
BEGIN
    IF NOT COALESCE(get_config_bool('automation.suggestions.enabled'), TRUE) THEN
        RETURN jsonb_build_object('created', FALSE, 'status', 'disabled', 'reason', 'automation_suggestions_disabled');
    END IF;
    IF v_source NOT IN ('catalog', 'blueprint', 'usage', 'connector') THEN
        RAISE EXCEPTION 'automation suggestion source must be catalog, blueprint, usage, or connector';
    END IF;
    IF v_key = '' OR length(v_key) > 300 THEN
        RAISE EXCEPTION 'automation suggestion dedup_key is required and must be at most 300 characters';
    END IF;
    IF NULLIF(btrim(COALESCE(p_title, '')), '') IS NULL
       OR NULLIF(btrim(COALESCE(p_rationale, '')), '') IS NULL THEN
        RAISE EXCEPTION 'automation suggestion title and rationale are required';
    END IF;
    IF jsonb_typeof(COALESCE(p_task_spec, 'null'::jsonb)) <> 'object' THEN
        RAISE EXCEPTION 'automation suggestion task_spec must be a JSON object';
    END IF;

    v_spec := p_task_spec || jsonb_build_object('action', 'create');
    IF NULLIF(btrim(COALESCE(v_spec->>'name', '')), '') IS NULL THEN
        RAISE EXCEPTION 'automation suggestion task_spec.name is required';
    END IF;
    IF COALESCE(NULLIF(v_spec->>'action_kind', ''), 'queue_user_message') = 'queue_user_message'
       AND NULLIF(btrim(COALESCE(v_spec->>'message', '')), '') IS NULL THEN
        RAISE EXCEPTION 'automation suggestion queue_user_message task_spec requires message';
    END IF;
    IF COALESCE(NULLIF(v_spec->>'action_kind', ''), 'queue_user_message') NOT IN ('queue_user_message', 'create_goal') THEN
        RAISE EXCEPTION 'automation suggestion task_spec has unsupported action_kind';
    END IF;
    IF COALESCE(NULLIF(v_spec->>'delivery_mode', ''), 'outbox') NOT IN ('outbox', 'channel', 'webhook', 'silent') THEN
        RAISE EXCEPTION 'automation suggestion task_spec has unsupported delivery_mode';
    END IF;
    -- Validate syntax now; acceptance repeats validation and creates atomically.
    PERFORM parse_schedule_input(v_spec);

    INSERT INTO automation_suggestions (
        source, dedup_key, title, rationale, task_spec, metadata
    ) VALUES (
        v_source, v_key, btrim(p_title), btrim(p_rationale), v_spec,
        COALESCE(p_metadata, '{}'::jsonb)
    )
    ON CONFLICT (dedup_key) DO NOTHING
    RETURNING * INTO v_row;

    IF v_row.id IS NULL THEN
        SELECT * INTO v_row
        FROM automation_suggestions
        WHERE dedup_key = v_key;
        RETURN jsonb_build_object(
            'created', FALSE,
            'suggestion_id', v_row.id,
            'status', v_row.status,
            'dedup_key', v_row.dedup_key,
            'scheduled_task_id', v_row.scheduled_task_id,
            'reason', CASE WHEN v_row.status = 'dismissed' THEN 'dismissal_latched' ELSE 'duplicate' END
        );
    END IF;

    v_code := automation_suggestion_code(v_row.id);
    v_schedule := COALESCE(NULLIF(v_spec->>'schedule', ''), NULLIF(v_spec->>'schedule_kind', ''), 'the proposed schedule');
    v_message := format(
        E'I have an automation suggestion: %s\n\n%s\n\nSchedule: %s. Nothing has been scheduled yet.\n\nReply:\n1 %s — Accept\n2 %s — Not for me',
        v_row.title, v_row.rationale, v_schedule, v_code, v_code
    );
    BEGIN
        v_outbox_id := queue_outbox_message(
            v_message,
            'automation_suggestion',
            'automation_suggestion',
            jsonb_build_object(
                'suggestion_id', v_row.id::text,
                'suggestion_code', v_code,
                'requires_response', TRUE
            )
        );
    EXCEPTION WHEN OTHERS THEN
        v_notification_error := SQLERRM;
    END;

    UPDATE automation_suggestions
    SET outbox_message_id = v_outbox_id,
        metadata = metadata || CASE
            WHEN v_notification_error IS NULL THEN jsonb_build_object('notification', 'queued')
            ELSE jsonb_build_object('notification', 'failed', 'notification_error', left(v_notification_error, 500))
        END,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = v_row.id
    RETURNING * INTO v_row;

    RETURN jsonb_build_object(
        'created', TRUE,
        'suggestion_id', v_row.id,
        'status', v_row.status,
        'dedup_key', v_row.dedup_key,
        'outbox_message_id', v_row.outbox_message_id,
        'notification_queued', v_outbox_id IS NOT NULL,
        'notification_error', v_notification_error
    );
END;
$$;

CREATE OR REPLACE FUNCTION accept_automation(
    p_id UUID,
    p_decision_channel TEXT DEFAULT NULL,
    p_decision_actor TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_row automation_suggestions%ROWTYPE;
    v_spec JSONB;
    v_timezone TEXT;
    v_created JSONB;
    v_task_id UUID;
BEGIN
    SELECT * INTO v_row
    FROM automation_suggestions
    WHERE id = p_id
    FOR UPDATE;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('ok', FALSE, 'error', 'suggestion_not_found');
    END IF;
    IF v_row.status = 'accepted' THEN
        RETURN jsonb_build_object(
            'ok', TRUE, 'status', 'accepted', 'already_decided', TRUE,
            'suggestion_id', v_row.id, 'scheduled_task_id', v_row.scheduled_task_id,
            'title', v_row.title
        );
    END IF;
    IF v_row.status = 'dismissed' THEN
        RETURN jsonb_build_object(
            'ok', FALSE, 'status', 'dismissed', 'error', 'dismissal_is_final',
            'suggestion_id', v_row.id, 'title', v_row.title
        );
    END IF;

    v_spec := v_row.task_spec || jsonb_build_object('action', 'create');
    IF NOT v_spec ? 'timezone' OR NULLIF(btrim(v_spec->>'timezone'), '') IS NULL THEN
        v_timezone := COALESCE(NULLIF(get_config_text('agent.timezone'), ''), 'UTC');
        v_spec := jsonb_set(v_spec, '{timezone}', to_jsonb(v_timezone), TRUE);
    END IF;
    v_created := manage_schedule_tool(v_spec);
    IF NOT COALESCE((v_created->>'success')::boolean, FALSE) THEN
        RETURN jsonb_build_object(
            'ok', FALSE,
            'status', 'pending',
            'error', COALESCE(v_created->>'error', 'scheduled_task_creation_failed'),
            'error_type', COALESCE(v_created->>'error_type', 'execution_failed'),
            'suggestion_id', v_row.id,
            'next_step', 'Review the proposed schedule, correct the configuration error, and accept again.'
        );
    END IF;
    v_task_id := NULLIF(v_created #>> '{output,task_id}', '')::uuid;
    IF v_task_id IS NULL THEN
        RAISE EXCEPTION 'manage_schedule created no task id for automation suggestion %', p_id;
    END IF;

    UPDATE automation_suggestions
    SET status = 'accepted',
        scheduled_task_id = v_task_id,
        decision_channel = NULLIF(btrim(COALESCE(p_decision_channel, '')), ''),
        decision_actor = NULLIF(btrim(COALESCE(p_decision_actor, '')), ''),
        decided_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_id;

    RETURN jsonb_build_object(
        'ok', TRUE, 'status', 'accepted', 'suggestion_id', p_id,
        'scheduled_task_id', v_task_id, 'title', v_row.title,
        'schedule', v_spec->>'schedule', 'timezone', v_spec->>'timezone'
    );
END;
$$;

CREATE OR REPLACE FUNCTION dismiss_automation(
    p_id UUID,
    p_decision_channel TEXT DEFAULT NULL,
    p_decision_actor TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_row automation_suggestions%ROWTYPE;
BEGIN
    SELECT * INTO v_row
    FROM automation_suggestions
    WHERE id = p_id
    FOR UPDATE;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('ok', FALSE, 'error', 'suggestion_not_found');
    END IF;
    IF v_row.status = 'accepted' THEN
        RETURN jsonb_build_object(
            'ok', FALSE, 'status', 'accepted', 'error', 'already_accepted',
            'suggestion_id', v_row.id, 'scheduled_task_id', v_row.scheduled_task_id,
            'next_step', 'Cancel or pause the scheduled task if you no longer want it.'
        );
    END IF;
    IF v_row.status = 'dismissed' THEN
        RETURN jsonb_build_object(
            'ok', TRUE, 'status', 'dismissed', 'already_decided', TRUE,
            'suggestion_id', v_row.id, 'title', v_row.title
        );
    END IF;

    UPDATE automation_suggestions
    SET status = 'dismissed',
        decision_channel = NULLIF(btrim(COALESCE(p_decision_channel, '')), ''),
        decision_actor = NULLIF(btrim(COALESCE(p_decision_actor, '')), ''),
        decided_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_id;

    RETURN jsonb_build_object(
        'ok', TRUE, 'status', 'dismissed', 'suggestion_id', p_id,
        'title', v_row.title, 'dedup_key', v_row.dedup_key,
        'dismissal_latched', TRUE
    );
END;
$$;

CREATE OR REPLACE FUNCTION list_automation_suggestions(
    p_status TEXT DEFAULT NULL,
    p_limit INTEGER DEFAULT 20
) RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', s.id,
        'source', s.source,
        'dedup_key', s.dedup_key,
        'title', s.title,
        'rationale', s.rationale,
        'task_spec', s.task_spec,
        'status', s.status,
        'scheduled_task_id', s.scheduled_task_id,
        'decision_channel', s.decision_channel,
        'created_at', s.created_at,
        'decided_at', s.decided_at
    ) ORDER BY s.created_at DESC, s.id), '[]'::jsonb)
    FROM (
        SELECT *
        FROM automation_suggestions
        WHERE (p_status IS NULL AND status = 'pending')
           OR status = p_status
           OR p_status = 'all'
        ORDER BY created_at DESC, id
        LIMIT LEAST(GREATEST(COALESCE(p_limit, 20), 1), 200)
    ) s;
$$;

CREATE OR REPLACE FUNCTION automation_catalog_precondition_met(p_precondition TEXT)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
AS $$
    SELECT CASE COALESCE(p_precondition, 'none')
        WHEN 'none' THEN TRUE
        WHEN 'gmail_connected' THEN EXISTS (
            SELECT 1 FROM integration_connections
            WHERE connector_id = 'gmail' AND status = 'connected'
              AND (
                  capabilities = '[]'::jsonb
                  OR capabilities ? 'read'
                  OR capabilities ? 'search'
              )
        )
        WHEN 'calendar_connected' THEN EXISTS (
            SELECT 1 FROM integration_connections
            WHERE connector_id IN ('calendar', 'google_calendar') AND status = 'connected'
        )
        ELSE FALSE
    END;
$$;

CREATE OR REPLACE FUNCTION refresh_automation_suggestion_catalog()
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_entry automation_suggestion_catalog%ROWTYPE;
    v_spec JSONB;
    v_result JSONB;
    v_created INTEGER := 0;
    v_existing INTEGER := 0;
    v_ineligible INTEGER := 0;
BEGIN
    IF NOT COALESCE(get_config_bool('automation.suggestions.enabled'), TRUE)
       OR NOT is_agent_configured()
       OR NOT is_init_complete()
       OR is_agent_terminated() THEN
        RETURN jsonb_build_object('skipped', TRUE, 'reason', 'disabled_or_agent_not_ready');
    END IF;

    FOR v_entry IN
        SELECT * FROM automation_suggestion_catalog
        WHERE enabled
        ORDER BY sort_order, dedup_key
    LOOP
        IF v_entry.source = 'catalog'
           AND NOT COALESCE(get_config_bool('automation.suggestions.catalog_enabled'), TRUE) THEN
            v_ineligible := v_ineligible + 1;
            CONTINUE;
        END IF;
        IF v_entry.source = 'connector'
           AND NOT COALESCE(get_config_bool('automation.suggestions.connector_enabled'), TRUE) THEN
            v_ineligible := v_ineligible + 1;
            CONTINUE;
        END IF;
        IF NOT automation_catalog_precondition_met(v_entry.precondition) THEN
            v_ineligible := v_ineligible + 1;
            CONTINUE;
        END IF;
        v_spec := materialize_automation_catalog_task_spec(
            v_entry.task_spec,
            v_entry.metadata->>'schedule_template'
        );
        v_result := propose_automation(
            v_entry.source,
            v_entry.dedup_key,
            v_entry.title,
            v_entry.rationale,
            v_spec,
            jsonb_build_object(
                'catalog_precondition', v_entry.precondition,
                'catalog_metadata', v_entry.metadata
            )
        );
        IF COALESCE((v_result->>'created')::boolean, FALSE) THEN
            v_created := v_created + 1;
        ELSE
            v_existing := v_existing + 1;
        END IF;
    END LOOP;
    RETURN jsonb_build_object(
        'created', v_created,
        'existing', v_existing,
        'ineligible', v_ineligible
    );
END;
$$;

CREATE OR REPLACE FUNCTION claim_automation_suggestion_refresh()
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    v_state JSONB;
    v_last TIMESTAMPTZ;
    v_interval INTEGER := LEAST(GREATEST(COALESCE(
        get_config_int('automation.suggestions.refresh_interval_seconds'), 60
    ), 5), 86400);
BEGIN
    IF NOT COALESCE(get_config_bool('automation.suggestions.enabled'), TRUE)
       OR NOT is_agent_configured()
       OR NOT is_init_complete()
       OR is_agent_terminated() THEN
        RETURN FALSE;
    END IF;
    INSERT INTO state (key, value)
    VALUES ('automation_suggestions_state', '{}'::jsonb)
    ON CONFLICT (key) DO NOTHING;
    SELECT value INTO v_state
    FROM state
    WHERE key = 'automation_suggestions_state'
    FOR UPDATE;
    v_last := NULLIF(v_state->>'last_refresh_started_at', '')::timestamptz;
    IF v_last IS NOT NULL
       AND CURRENT_TIMESTAMP < v_last + make_interval(secs => v_interval) THEN
        RETURN FALSE;
    END IF;
    PERFORM set_state(
        'automation_suggestions_state',
        v_state || jsonb_build_object('last_refresh_started_at', CURRENT_TIMESTAMP)
    );
    RETURN TRUE;
END;
$$;

CREATE OR REPLACE FUNCTION mark_automation_suggestion_refresh(
    p_result JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_state JSONB;
BEGIN
    v_state := COALESCE(get_state('automation_suggestions_state'), '{}'::jsonb)
        || jsonb_build_object(
            'last_refresh_completed_at', CURRENT_TIMESTAMP,
            'last_result', COALESCE(p_result, '{}'::jsonb)
        );
    PERFORM set_state('automation_suggestions_state', v_state);
    RETURN v_state;
END;
$$;

CREATE OR REPLACE FUNCTION try_resolve_automation_suggestion_from_inbound(
    p_channel TEXT,
    p_actor TEXT,
    p_text TEXT
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_text TEXT := lower(btrim(COALESCE(p_text, '')));
    v_match TEXT[];
    v_choice TEXT;
    v_code TEXT;
    v_count INTEGER;
    v_id UUID;
    v_result JSONB;
BEGIN
    v_match := regexp_match(
        v_text,
        '^(1|2|accept|dismiss|not[[:space:]]+for[[:space:]]+me)(?:[[:space:]]+automation)?(?:[[:space:]:#-]+([0-9a-f]{8}))?$'
    );
    IF v_match IS NULL THEN
        RETURN jsonb_build_object('recognized', FALSE, 'matched', FALSE);
    END IF;
    v_choice := v_match[1];
    v_code := v_match[2];

    IF v_code IS NULL THEN
        SELECT count(*), (array_agg(id ORDER BY created_at, id))[1]
        INTO v_count, v_id
        FROM automation_suggestions
        WHERE status = 'pending';
        IF v_count <> 1 THEN
            RETURN jsonb_build_object(
                'recognized', TRUE, 'matched', FALSE,
                'reason', CASE WHEN v_count = 0 THEN 'no_pending_suggestion' ELSE 'ambiguous_without_code' END,
                'message', CASE WHEN v_count = 0
                    THEN 'There is no pending automation suggestion to decide.'
                    ELSE 'More than one automation suggestion is pending. Reply `1 CODE` to accept or `2 CODE` for Not for me, using the code in its message.'
                END
            );
        END IF;
    ELSE
        SELECT count(*), (array_agg(id ORDER BY created_at, id))[1]
        INTO v_count, v_id
        FROM automation_suggestions
        WHERE status = 'pending'
          AND left(replace(id::text, '-', ''), 8) = v_code;
        IF v_count <> 1 THEN
            RETURN jsonb_build_object(
                'recognized', TRUE, 'matched', FALSE,
                'reason', 'suggestion_code_not_found',
                'message', 'That automation code is not pending. Use the eight-character code from the suggestion message.'
            );
        END IF;
    END IF;

    IF v_choice IN ('1', 'accept') THEN
        v_result := accept_automation(v_id, lower(btrim(COALESCE(p_channel, ''))), p_actor);
    ELSE
        v_result := dismiss_automation(v_id, lower(btrim(COALESCE(p_channel, ''))), p_actor);
    END IF;
    RETURN jsonb_build_object(
        'recognized', TRUE,
        'matched', COALESCE((v_result->>'ok')::boolean, FALSE),
        'status', v_result->>'status',
        'suggestion_id', v_id,
        'result', v_result,
        'message', CASE
            WHEN NOT COALESCE((v_result->>'ok')::boolean, FALSE) THEN
                format('I could not record that automation decision: %s. %s',
                    COALESCE(v_result->>'error', 'unknown error'), COALESCE(v_result->>'next_step', 'Open the Hexis inbox and try again.'))
            WHEN v_result->>'status' = 'accepted' THEN
                format('Accepted “%s”. The scheduled task is active%s.',
                    COALESCE(v_result->>'title', 'automation'),
                    CASE WHEN v_result->>'schedule' IS NULL THEN '' ELSE ' on ' || (v_result->>'schedule') END)
            ELSE
                format('Marked “%s” Not for me. I will not suggest that routine again.',
                    COALESCE(v_result->>'title', 'automation'))
        END
    );
END;
$$;

CREATE OR REPLACE FUNCTION automation_suggestion_on_connector_connected()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.status = 'connected'
       AND (TG_OP = 'INSERT' OR OLD.status IS DISTINCT FROM 'connected')
       AND COALESCE(get_config_bool('automation.suggestions.enabled'), TRUE)
       AND COALESCE(get_config_bool('automation.suggestions.connector_enabled'), TRUE) THEN
        -- The refresh is precondition-driven and deduplicated, so it safely
        -- handles the supported connector that just transitioned as well as
        -- any catalog entry that became eligible at the same instant.
        PERFORM refresh_automation_suggestion_catalog();
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_automation_suggestion_connector_connected
    ON integration_connections;
CREATE TRIGGER trg_automation_suggestion_connector_connected
AFTER INSERT OR UPDATE OF status ON integration_connections
FOR EACH ROW
EXECUTE FUNCTION automation_suggestion_on_connector_connected();

-- Existing configured agents see the starter catalog after migration. Fresh
-- databases remain quiet until initialization completes; the worker refresh
-- picks the catalog up on its next tick.
SELECT refresh_automation_suggestion_catalog();
