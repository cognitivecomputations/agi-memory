-- #111: classify_defect_event's network/provider bucket had no pattern for
-- DNS/name-resolution failures -- the exact class that hit a real provider
-- outage in the wild ("[Errno -2] Name or service not known"), which fell
-- through to the generic execution_failure bucket instead. Re-applied here
-- (rather than only in the baseline db/92_functions_self_repair.sql)
-- because migration 0190_self_repair_defect_reports.sql already shipped an
-- earlier CREATE OR REPLACE of this function; migrations apply after the
-- baseline, so without this the old definition would win on every install.
SET search_path = public, ag_catalog, "$user";

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
       OR error_text LIKE '%rate limit%'
       -- DNS/name-resolution failures (#111): the exact class that hit a
       -- provider outage in the wild ("[Errno -2] Name or service not
       -- known") and previously fell through to the generic bucket below.
       OR error_text LIKE '%name or service not known%'
       OR error_text LIKE '%nodename nor servname%'
       OR error_text LIKE '%temporary failure in name resolution%'
       OR error_text LIKE '%getaddrinfo failed%'
       OR error_text LIKE '%name resolution%' THEN
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
