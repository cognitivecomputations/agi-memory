-- sync_tool_definitions() only ever upserts (#115): a renamed or removed
-- tool keeps advertising to the chat/heartbeat path forever, since nothing
-- prunes rows that stop appearing in every process's sync payload.
-- Deliberately NOT auto-deleted: every worker/API/channel/MCP process syncs
-- its own in-memory registry independently, so a name absent from one
-- process's payload does not mean it is gone everywhere -- an aggressive
-- per-call diff-delete would risk erasing a tool another still-running
-- process legitimately serves (Experience Bar: no destructive action on a
-- timer or by default). This adds the read-only advisory `hexis doctor`
-- uses instead: names untouched by any sync for longer than the configured
-- threshold, surfaced for an operator to verify and remove manually.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

INSERT INTO config_defaults (key, value, description) VALUES
    ('tools.definition_stale_days', '14'::jsonb, 'A tool_definitions row untouched by any process sync for this long is flagged by hexis doctor as a likely renamed/removed tool (#115); never auto-deleted')
ON CONFLICT (key) DO NOTHING;

CREATE OR REPLACE FUNCTION stale_tool_definitions()
RETURNS JSONB
LANGUAGE sql STABLE
AS $$
    SELECT COALESCE(jsonb_agg(
        jsonb_build_object(
            'name', name,
            'updated_at', updated_at,
            'age_days', round(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - updated_at)) / 86400.0, 1)
        ) ORDER BY updated_at
    ), '[]'::jsonb)
    FROM tool_definitions
    WHERE updated_at < CURRENT_TIMESTAMP - (
        GREATEST(COALESCE(get_config_int('tools.definition_stale_days'), 14), 1) || ' days'
    )::interval;
$$;
