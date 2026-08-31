-- Three heartbeat actions were offered, priced, and unimplemented.
--
-- `fast_ingest`, `hybrid_ingest`, and `slow_ingest` appear in
-- heartbeat.allowed_actions with configured energy costs, but
-- execute_heartbeat_action has no branch for them: choosing one returns
-- {"success": false, "error": "Unknown action: fast_ingest"}. It fails loudly
-- and charges nothing, which is right — but the beat's entire decision call is
-- spent picking something impossible.
--
-- They are also redundant. The tools of the same name exist and are bound to
-- the `knowledge-ingest` skill, which loads in heartbeat context, so the agent
-- can already ingest by calling a tool. Removing the duplicate action is the
-- honest direction: one way to do a thing, and it works.
--
-- The four cognitive actions once suspected dead — debate_internally,
-- inquire_deep, study, meditate — are all implemented, via multi-literal WHEN
-- branches shared with contemplate and inquire_shallow. Verified by executing
-- each one.
SET search_path = public, ag_catalog, "$user";

UPDATE config
   SET value = (
        SELECT jsonb_agg(action ORDER BY ord)
        FROM jsonb_array_elements_text(value) WITH ORDINALITY AS t(action, ord)
        WHERE action NOT IN ('fast_ingest', 'hybrid_ingest', 'slow_ingest')
   )
 WHERE key = 'heartbeat.allowed_actions'
   AND value @> '["fast_ingest"]'::jsonb;

UPDATE config_defaults
   SET value = (
        SELECT jsonb_agg(action ORDER BY ord)
        FROM jsonb_array_elements_text(value) WITH ORDINALITY AS t(action, ord)
        WHERE action NOT IN ('fast_ingest', 'hybrid_ingest', 'slow_ingest')
   )
 WHERE key = 'heartbeat.allowed_actions'
   AND value @> '["fast_ingest"]'::jsonb;
