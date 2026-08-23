-- Select on the *shape* of the similarity distribution, not an absolute cutoff.
--
-- Measured over the eleven-request probe: genuine matches score z = 2.1–3.8
-- against the run's own mean, while "hello", "tell me a joke", and a plain
-- recall question sit at z = 1.4–1.9. Raw cosine does not separate them —
-- signal spans 0.46–0.73 and noise spans 0.40–0.54, overlapping badly, because
-- absolute similarity from this model is compressed and query-dependent.
--
-- A peaked distribution is what "this request is about something in particular"
-- looks like; a flat one means "nothing in particular", which is the right
-- answer for a greeting. z is scale-free, so this is a property rather than
-- another hand-tuned constant.
SET search_path = public, ag_catalog, "$user";

INSERT INTO config_defaults (key, value, description) VALUES
    ('skills.semantic_z_threshold', '2.0'::jsonb,
     'Standard deviations above the run mean a skill must score to auto-activate')
ON CONFLICT (key) DO NOTHING;

-- 0.45 was a first guess before the distribution was measured; the observed
-- noise floor sits at 0.40. This is a backstop against a peaked-but-weak run,
-- not the primary gate.
UPDATE config_defaults
   SET value = '0.40'::jsonb,
       description = 'Absolute cosine floor; backstop under the z-score gate'
 WHERE key = 'skills.semantic_threshold';
