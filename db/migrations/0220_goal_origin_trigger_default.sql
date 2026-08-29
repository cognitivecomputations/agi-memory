-- A raw goal insert has no authenticated turn context. Never infer authority
-- from metadata.source; only the typed column or the initialization lifecycle
-- marker may establish a non-derived origin.
SET search_path = public, ag_catalog, "$user";

CREATE OR REPLACE FUNCTION normalize_memory_goal_origin()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.type = 'goal'::memory_type THEN
        IF NEW.metadata->>'origin' = 'initialization' THEN
            NEW.goal_origin := 'user_request'::goal_source;
        ELSIF NEW.goal_origin IS NULL THEN
            NEW.goal_origin := 'derived'::goal_source;
        END IF;
    ELSE
        NEW.goal_origin := NULL;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
