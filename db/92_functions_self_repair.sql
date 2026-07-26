-- DB-owned self-repair substrate: classify failures, preserve evidence, and
-- draft bounded repair plans without granting autonomous source writes.
SET search_path = public, ag_catalog, "$user";

CREATE OR REPLACE FUNCTION normalize_defect_error(p_error TEXT)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT left(
        regexp_replace(
            regexp_replace(
                regexp_replace(lower(COALESCE(p_error, '')),
                    '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
                    '<uuid>', 'gi'),
                '[0-9]+', '<n>', 'g'),
            '\s+', ' ', 'g'),
        500)
$$;

CREATE OR REPLACE FUNCTION classify_defect_event(
    p_component TEXT,
    p_error TEXT,
    p_context JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    component TEXT := COALESCE(NULLIF(btrim(p_component), ''), 'unknown');
    error_text TEXT := lower(COALESCE(p_error, ''));
    category TEXT := 'execution_failure';
    severity TEXT := 'medium';
    title TEXT;
    summary TEXT;
BEGIN
    IF error_text LIKE '%unknown tool:%'
       OR error_text LIKE '%unknown action:%'
       OR error_text LIKE '%validation errors:%'
       OR error_text LIKE '%missing required field:%'
       OR error_text LIKE '%not allowed in % context%' THEN
        category := 'tool_contract';
        severity := 'medium';
        title := 'Tool/action contract failure: ' || component;
        summary := 'A tool, heartbeat action, or argument schema did not match the executor contract.';
    ELSIF error_text LIKE '%embedding service%'
       OR error_text LIKE '%connection refused%'
       OR error_text LIKE '%failed to connect%'
       OR error_text LIKE '%not reachable%' THEN
        category := 'dependency_unavailable';
        severity := 'high';
        title := 'Dependency unavailable: ' || component;
        summary := 'A required local service or dependency was unavailable when the agent tried to use it.';
    ELSIF error_text LIKE '%not configured%'
       OR error_text LIKE '%missing api key%'
       OR error_text LIKE '%missing config%'
       OR error_text LIKE '%credentials%' THEN
        category := 'configuration';
        severity := 'low';
        title := 'Configuration needed: ' || component;
        summary := 'The operation needs user/provider configuration rather than code repair.';
    ELSIF error_text LIKE '%timed out%'
       OR error_text LIKE '%timeout%' THEN
        category := 'timeout';
        severity := 'medium';
        title := 'Timeout: ' || component;
        summary := 'The operation exceeded its execution window and needs retry/backoff or workload reduction.';
    ELSIF error_text LIKE '%network error%'
       OR error_text LIKE '%http error%'
       OR error_text LIKE '%rate limit%' THEN
        category := 'network_or_provider';
        severity := 'medium';
        title := 'Provider/network failure: ' || component;
        summary := 'The operation failed outside the local code path and may need retry or provider-specific handling.';
    ELSE
        title := 'Execution failure: ' || component;
        summary := 'The agent observed a failed operation that needs inspection before repair.';
    END IF;

    RETURN jsonb_build_object(
        'category', category,
        'severity', severity,
        'title', title,
        'summary', summary
    );
END;
$$;

CREATE OR REPLACE FUNCTION record_defect_event(
    p_source TEXT,
    p_component TEXT,
    p_error TEXT,
    p_context JSONB DEFAULT '{}'::jsonb
) RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    ctx JSONB := COALESCE(p_context, '{}'::jsonb);
    classification JSONB;
    normalized TEXT;
    fingerprint_value TEXT;
    evidence_item JSONB;
    defect_id UUID;
    heartbeat_uuid UUID;
    tool_name TEXT;
BEGIN
    IF NULLIF(btrim(COALESCE(p_error, '')), '') IS NULL THEN
        RAISE EXCEPTION 'defect error is required';
    END IF;

    classification := classify_defect_event(p_component, p_error, ctx);
    normalized := normalize_defect_error(p_error);
    fingerprint_value := md5(
        lower(COALESCE(NULLIF(btrim(p_source), ''), 'unknown'))
        || '|'
        || COALESCE(classification->>'category', 'execution_failure')
        || '|'
        || COALESCE(NULLIF(btrim(p_component), ''), 'unknown')
        || '|'
        || normalized
    );

    BEGIN
        heartbeat_uuid := NULLIF(ctx->>'heartbeat_id', '')::uuid;
    EXCEPTION WHEN OTHERS THEN
        heartbeat_uuid := NULL;
    END;
    tool_name := COALESCE(NULLIF(ctx->>'tool_name', ''), NULLIF(ctx->>'action', ''), NULLIF(btrim(p_component), ''));

    evidence_item := jsonb_build_object(
        'at', CURRENT_TIMESTAMP,
        'source', COALESCE(NULLIF(btrim(p_source), ''), 'unknown'),
        'component', COALESCE(NULLIF(btrim(p_component), ''), 'unknown'),
        'error', p_error,
        'context', ctx
    );

    INSERT INTO defect_reports (
        fingerprint,
        status,
        severity,
        category,
        source,
        component,
        title,
        summary,
        last_error,
        heartbeat_ids,
        tool_names,
        evidence
    )
    VALUES (
        fingerprint_value,
        'open',
        COALESCE(classification->>'severity', 'medium'),
        COALESCE(classification->>'category', 'execution_failure'),
        COALESCE(NULLIF(btrim(p_source), ''), 'unknown'),
        NULLIF(btrim(p_component), ''),
        COALESCE(classification->>'title', 'Execution failure'),
        COALESCE(classification->>'summary', 'The agent observed a failed operation.'),
        p_error,
        CASE WHEN heartbeat_uuid IS NULL THEN ARRAY[]::uuid[] ELSE ARRAY[heartbeat_uuid] END,
        CASE WHEN tool_name IS NULL THEN ARRAY[]::text[] ELSE ARRAY[tool_name] END,
        jsonb_build_array(evidence_item)
    )
    ON CONFLICT (fingerprint) DO UPDATE SET
        status = CASE
            WHEN defect_reports.status IN ('resolved', 'ignored') THEN 'open'
            ELSE defect_reports.status
        END,
        severity = EXCLUDED.severity,
        category = EXCLUDED.category,
        title = EXCLUDED.title,
        summary = EXCLUDED.summary,
        last_error = EXCLUDED.last_error,
        last_seen_at = CURRENT_TIMESTAMP,
        occurrence_count = defect_reports.occurrence_count + 1,
        heartbeat_ids = CASE
            WHEN heartbeat_uuid IS NULL OR heartbeat_uuid = ANY(defect_reports.heartbeat_ids)
            THEN defect_reports.heartbeat_ids
            ELSE array_append(defect_reports.heartbeat_ids, heartbeat_uuid)
        END,
        tool_names = CASE
            WHEN tool_name IS NULL OR tool_name = ANY(defect_reports.tool_names)
            THEN defect_reports.tool_names
            ELSE array_append(defect_reports.tool_names, tool_name)
        END,
        evidence = (
            SELECT COALESCE(jsonb_agg(value ORDER BY ord), '[]'::jsonb)
            FROM (
                SELECT value, ord
                FROM jsonb_array_elements(defect_reports.evidence || jsonb_build_array(evidence_item))
                     WITH ORDINALITY AS e(value, ord)
                ORDER BY ord DESC
                LIMIT 20
            ) recent
        ),
        updated_at = CURRENT_TIMESTAMP
    RETURNING id INTO defect_id;

    RETURN defect_id;
END;
$$;

CREATE OR REPLACE FUNCTION record_heartbeat_action_defects(
    p_heartbeat_id UUID,
    p_actions JSONB,
    p_reasoning TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    action_record JSONB;
    action_name TEXT;
    success_text TEXT;
    failed BOOLEAN;
    error_text TEXT;
    ids UUID[] := ARRAY[]::UUID[];
    defect_id UUID;
BEGIN
    IF p_actions IS NULL OR jsonb_typeof(p_actions) <> 'array' THEN
        RETURN '[]'::jsonb;
    END IF;

    FOR action_record IN SELECT value FROM jsonb_array_elements(p_actions)
    LOOP
        success_text := lower(COALESCE(action_record#>>'{result,success}', ''));
        failed := CASE
            WHEN success_text = 'true' THEN FALSE
            WHEN success_text = 'false' THEN TRUE
            ELSE action_record ? 'error'
                 OR NULLIF(action_record#>>'{result,error}', '') IS NOT NULL
                 OR NULLIF(action_record#>>'{result,output_preview}', '') IS NOT NULL
        END;
        IF NOT failed THEN
            CONTINUE;
        END IF;

        action_name := COALESCE(NULLIF(action_record->>'action', ''), NULLIF(action_record->>'tool_name', ''), 'unknown_action');
        error_text := COALESCE(
            NULLIF(action_record#>>'{result,error}', ''),
            NULLIF(action_record#>>'{result,output_preview}', ''),
            NULLIF(action_record->>'error', ''),
            'action failed without an error message'
        );
        defect_id := record_defect_event(
            'heartbeat',
            action_name,
            error_text,
            jsonb_build_object(
                'heartbeat_id', p_heartbeat_id,
                'action', action_name,
                'action_record', action_record,
                'reasoning_excerpt', left(COALESCE(p_reasoning, ''), 1000)
            )
        );
        ids := array_append(ids, defect_id);
    END LOOP;

    RETURN COALESCE(to_jsonb(ids), '[]'::jsonb);
END;
$$;

CREATE OR REPLACE FUNCTION list_defect_reports(
    p_status TEXT DEFAULT 'open',
    p_limit INT DEFAULT 10
) RETURNS JSONB
LANGUAGE sql
STABLE
AS $$
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', id,
        'status', status,
        'severity', severity,
        'category', category,
        'source', source,
        'component', component,
        'title', title,
        'summary', summary,
        'last_error', last_error,
        'occurrence_count', occurrence_count,
        'first_seen_at', first_seen_at,
        'last_seen_at', last_seen_at,
        'heartbeat_ids', heartbeat_ids,
        'tool_names', tool_names,
        'diagnosis', diagnosis,
        'proposed_repair', proposed_repair,
        'verification', verification
    ) ORDER BY last_seen_at DESC), '[]'::jsonb)
    FROM (
        SELECT *
        FROM defect_reports
        WHERE COALESCE(p_status, 'open') = 'all'
           OR status = COALESCE(p_status, 'open')
        ORDER BY
            CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
            last_seen_at DESC
        LIMIT LEAST(GREATEST(COALESCE(p_limit, 10), 1), 50)
    ) d
$$;

CREATE OR REPLACE FUNCTION diagnose_defect_report(
    p_defect_id UUID
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    d defect_reports%ROWTYPE;
    likely_files JSONB;
    diagnosis_doc JSONB;
    repair_doc JSONB;
BEGIN
    SELECT * INTO d FROM defect_reports WHERE id = p_defect_id;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('success', false, 'error', 'defect not found');
    END IF;

    likely_files := CASE
        WHEN d.category = 'tool_contract' AND COALESCE(d.component, '') = 'get_strategies' THEN
            '["core/tools/memory.py","db/38_functions_db_native_tools.sql","services/prompts/rlm_heartbeat_system.md"]'::jsonb
        WHEN d.category = 'tool_contract' AND COALESCE(d.last_error, '') ILIKE '%unknown action%' THEN
            '["db/17_functions_subconscious_observations.sql","services/prompts/rlm_heartbeat_system.md","services/heartbeat_runner.py"]'::jsonb
        WHEN d.category = 'tool_contract' AND COALESCE(d.last_error, '') ILIKE '%unknown tool%' THEN
            '["core/tools/registry.py","core/tools/self_inspection.py","services/prompts/rlm_heartbeat_system.md"]'::jsonb
        WHEN d.category = 'dependency_unavailable' THEN
            '["apps/hexis_cli.py","services/worker_service.py","core/config.py"]'::jsonb
        WHEN d.category = 'configuration' THEN
            '["hexis-ui/app","core/tools/config.py","docs"]'::jsonb
        WHEN d.category = 'timeout' THEN
            '["services/worker_service.py","core/tools/registry.py","services/heartbeat_runner.py"]'::jsonb
        ELSE
            '["db","core","services"]'::jsonb
    END;

    diagnosis_doc := jsonb_build_object(
        'category', d.category,
        'severity', d.severity,
        'hypothesis', CASE
            WHEN d.category = 'tool_contract' THEN
                'The model, prompt, registry schema, or DB action executor disagrees about the valid tool/action contract.'
            WHEN d.category = 'dependency_unavailable' THEN
                'A required local service was unavailable or unreachable when the operation ran.'
            WHEN d.category = 'configuration' THEN
                'The user must authorize or configure a provider; this is not primarily a code defect.'
            WHEN d.category = 'timeout' THEN
                'The workload or provider call exceeded the current timeout and needs smaller units, retry/backoff, or better progress handling.'
            ELSE
                'The failure needs source and log inspection before a safe repair can be selected.'
        END,
        'evidence_count', jsonb_array_length(COALESCE(d.evidence, '[]'::jsonb)),
        'latest_error', d.last_error,
        'likely_files', likely_files
    );

    repair_doc := jsonb_build_object(
        'mode', 'proposal_only',
        'safe_to_apply_autonomously', false,
        'why_not_auto_apply', 'Heartbeat source edits remain approval-gated; self-repair may inspect and draft, not silently modify code.',
        'next_steps', jsonb_build_array(
            'Inspect the likely files and live schema for the recorded component/error.',
            'Reproduce or cite the failing path from the preserved evidence.',
            'Prepare the smallest source/schema/prompt patch that addresses the contract mismatch.',
            'Run the focused regression that would have caught the defect.',
            'Ask the user to approve applying the patch, or apply only in an explicitly granted dev mode.'
        ),
        'suggested_tests', CASE
            WHEN d.category = 'tool_contract' THEN
                '["focused tool/heartbeat regression for the failing component","git diff --check"]'::jsonb
            WHEN d.category = 'dependency_unavailable' THEN
                '["doctor/health check for the dependency","worker startup smoke test"]'::jsonb
            ELSE
                '["focused regression for the failing path","git diff --check"]'::jsonb
        END
    );

    UPDATE defect_reports
    SET status = CASE WHEN status = 'open' THEN 'repair_proposed' ELSE status END,
        diagnosis = diagnosis_doc,
        proposed_repair = repair_doc,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_defect_id;

    RETURN jsonb_build_object(
        'success', true,
        'defect_id', p_defect_id,
        'diagnosis', diagnosis_doc,
        'proposed_repair', repair_doc
    );
END;
$$;

CREATE OR REPLACE FUNCTION mark_defect_report_resolved(
    p_defect_id UUID,
    p_resolution TEXT,
    p_verification JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE defect_reports
    SET status = 'resolved',
        resolution = NULLIF(btrim(COALESCE(p_resolution, '')), ''),
        verification = COALESCE(p_verification, '{}'::jsonb),
        resolved_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_defect_id;

    IF NOT FOUND THEN
        RETURN jsonb_build_object('success', false, 'error', 'defect not found');
    END IF;
    RETURN jsonb_build_object('success', true, 'defect_id', p_defect_id, 'status', 'resolved');
END;
$$;

CREATE OR REPLACE FUNCTION render_defect_reports_context(
    p_limit INT DEFAULT 5
) RETURNS TEXT
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    lines TEXT;
BEGIN
    WITH defects AS (
        SELECT *
        FROM defect_reports
        WHERE status IN ('open', 'diagnosed', 'repair_proposed')
        ORDER BY
            CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
            last_seen_at DESC
        LIMIT LEAST(GREATEST(COALESCE(p_limit, 5), 1), 20)
    )
    SELECT string_agg(
        '- [' || severity || '] ' || title
        || ' (' || occurrence_count || ' occurrence' || CASE WHEN occurrence_count = 1 THEN '' ELSE 's' END || '; status=' || status || ')' || E'\n'
        || '  summary: ' || left(regexp_replace(COALESCE(summary, ''), '[[:space:]]+', ' ', 'g'), 360) || E'\n'
        || '  latest error: ' || left(regexp_replace(COALESCE(last_error, ''), '[[:space:]]+', ' ', 'g'), 360)
        || CASE WHEN jsonb_typeof(diagnosis) = 'object' AND diagnosis <> '{}'::jsonb
                THEN E'\n  diagnosis: ' || left(regexp_replace(COALESCE(diagnosis->>'hypothesis', ''), '[[:space:]]+', ' ', 'g'), 360)
                ELSE '' END,
        E'\n' ORDER BY
            CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
            last_seen_at DESC
    )
    INTO lines
    FROM defects;

    RETURN lines;
END;
$$;
