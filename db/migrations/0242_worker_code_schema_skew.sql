-- Code/schema version skew detection (#113): migrations auto-apply on
-- worker/API startup, but nothing previously checked or surfaced that a
-- long-running worker's *code* predates migrations someone else (a fresher
-- checkout's `hexis migrate`, or an already-restarted sibling worker) has
-- since applied to the shared database. Each worker now records the highest
-- migration version bundled in its own db/migrations/ at registration
-- (metadata.bundled_latest_migration); this adds the read-only comparison
-- `hexis status`/`hexis doctor` use to WARN when a running worker is behind.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

CREATE OR REPLACE FUNCTION worker_code_schema_skew_report()
RETURNS JSONB
LANGUAGE sql STABLE
AS $$
    WITH latest AS (
        SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1
    ),
    skewed AS (
        SELECT
            w.id AS worker_id,
            w.mode,
            w.instance_name,
            w.build_id,
            NULLIF(w.metadata->>'bundled_latest_migration', '') AS bundled_latest_migration,
            (SELECT version FROM latest) AS db_latest_migration
        FROM worker_runtime_status w
        WHERE NOT w.is_stale
          AND w.status IN ('starting', 'running', 'stopping')
          AND NULLIF(w.metadata->>'bundled_latest_migration', '') IS NOT NULL
          AND (SELECT version FROM latest) IS NOT NULL
          AND NULLIF(w.metadata->>'bundled_latest_migration', '') < (SELECT version FROM latest)
    )
    SELECT jsonb_build_object(
        'db_latest_migration', (SELECT version FROM latest),
        'skewed_workers', COALESCE(jsonb_agg(to_jsonb(skewed)), '[]'::jsonb)
    )
    FROM skewed;
$$;
