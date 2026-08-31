-- Durable memory-revision lineage and temporal validity.
SET search_path = public, ag_catalog, "$user";

CREATE OR REPLACE VIEW memory_supersessions_active AS
SELECT s.*
FROM memory_supersessions s
WHERE s.status = 'active';

CREATE OR REPLACE VIEW memory_effective_replacement AS
SELECT
    s.superseded_memory_id AS memory_id,
    s.replacement_memory_id,
    s.reason,
    s.actor,
    s.superseded_at,
    replacement.status AS replacement_status
FROM memory_supersessions s
LEFT JOIN memories replacement ON replacement.id = s.replacement_memory_id
WHERE s.status = 'active';

-- Every memory participates in valid-time queries from birth. Imported rows
-- that deliberately carry an older valid_from retain it; legacy NULLs become
-- their transaction creation time.
CREATE OR REPLACE FUNCTION normalize_memory_temporal_validity()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.valid_from := COALESCE(NEW.valid_from, NEW.created_at, CURRENT_TIMESTAMP);
    IF NEW.valid_until IS NOT NULL AND NEW.valid_until < NEW.valid_from THEN
        RAISE EXCEPTION
            'memory valid_until (%) cannot precede valid_from (%)',
            NEW.valid_until,
            NEW.valid_from;
    END IF;
    IF NEW.superseded_by IS NOT NULL AND NEW.superseded_by = NEW.id THEN
        RAISE EXCEPTION 'memory % cannot supersede itself', NEW.id;
    END IF;
    RETURN NEW;
END;
$$;

-- Record one explicit revision event. An active event atomically closes the
-- old memory's validity window and updates its fast current-state pointer.
-- The old row is never deleted, so point-in-time recall remains possible.
CREATE OR REPLACE FUNCTION record_supersession(
    p_superseded_memory_id UUID,
    p_replacement_memory_id UUID DEFAULT NULL,
    p_reason TEXT DEFAULT NULL,
    p_actor TEXT DEFAULT NULL,
    p_status TEXT DEFAULT 'active',
    p_superseded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    p_resolved_at TIMESTAMPTZ DEFAULT NULL,
    p_replacement_planned BOOLEAN DEFAULT FALSE,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    old_memory memories%ROWTYPE;
    active_supersession memory_supersessions%ROWTYPE;
    new_id UUID;
    normalized_reason TEXT := NULLIF(btrim(COALESCE(p_reason, '')), '');
    normalized_actor TEXT := NULLIF(btrim(COALESCE(p_actor, '')), '');
    normalized_status TEXT := lower(NULLIF(btrim(COALESCE(p_status, '')), ''));
    effective_at TIMESTAMPTZ := COALESCE(p_superseded_at, CURRENT_TIMESTAMP);
BEGIN
    IF p_superseded_memory_id IS NULL THEN
        RAISE EXCEPTION 'record_supersession requires superseded_memory_id';
    END IF;
    IF normalized_reason IS NULL THEN
        RAISE EXCEPTION 'record_supersession requires a non-empty reason';
    END IF;
    IF normalized_actor IS NULL THEN
        RAISE EXCEPTION 'record_supersession requires a non-empty actor';
    END IF;
    IF normalized_status NOT IN ('active', 'reverted', 'pending') THEN
        RAISE EXCEPTION 'invalid memory supersession status: %', p_status;
    END IF;
    IF p_replacement_memory_id = p_superseded_memory_id THEN
        RAISE EXCEPTION 'memory % cannot supersede itself', p_superseded_memory_id;
    END IF;
    IF normalized_status = 'active' AND p_resolved_at IS NOT NULL THEN
        RAISE EXCEPTION 'an active supersession cannot already be resolved';
    END IF;

    SELECT *
    INTO old_memory
    FROM memories
    WHERE id = p_superseded_memory_id
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'superseded memory % does not exist', p_superseded_memory_id;
    END IF;
    IF effective_at < COALESCE(old_memory.valid_from, old_memory.created_at, effective_at) THEN
        RAISE EXCEPTION
            'supersession time (%) cannot precede memory % valid_from (%)',
            effective_at,
            p_superseded_memory_id,
            old_memory.valid_from;
    END IF;
    IF p_replacement_memory_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM memories WHERE id = p_replacement_memory_id
    ) THEN
        RAISE EXCEPTION 'replacement memory % does not exist', p_replacement_memory_id;
    END IF;

    IF normalized_status = 'active' THEN
        SELECT *
        INTO active_supersession
        FROM memory_supersessions
        WHERE superseded_memory_id = p_superseded_memory_id
          AND status = 'active'
        ORDER BY superseded_at DESC
        LIMIT 1;
        IF FOUND THEN
            IF active_supersession.replacement_memory_id IS NOT DISTINCT FROM p_replacement_memory_id
               AND active_supersession.reason = normalized_reason
               AND active_supersession.actor = normalized_actor THEN
                RETURN active_supersession.id;
            END IF;
            RAISE EXCEPTION
                'memory % already has active supersession %; revert it explicitly before recording another',
                p_superseded_memory_id,
                active_supersession.id;
        END IF;
    END IF;

    INSERT INTO memory_supersessions (
        superseded_memory_id,
        replacement_memory_id,
        reason,
        actor,
        status,
        superseded_at,
        resolved_at,
        replacement_planned,
        metadata
    ) VALUES (
        p_superseded_memory_id,
        p_replacement_memory_id,
        normalized_reason,
        normalized_actor,
        normalized_status,
        effective_at,
        CASE
            WHEN normalized_status = 'active' THEN NULL
            ELSE COALESCE(p_resolved_at, effective_at)
        END,
        COALESCE(p_replacement_planned, FALSE)
            OR p_replacement_memory_id IS NOT NULL,
        COALESCE(p_metadata, '{}'::jsonb)
    )
    RETURNING id INTO new_id;

    IF normalized_status = 'active' THEN
        UPDATE memories
        SET valid_until = effective_at,
            superseded_by = p_replacement_memory_id,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = p_superseded_memory_id;

        IF p_replacement_memory_id IS NOT NULL THEN
            UPDATE memories
            SET valid_from = LEAST(COALESCE(valid_from, effective_at), effective_at),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = p_replacement_memory_id;
        END IF;
    END IF;

    RETURN new_id;
END;
$$;

CREATE OR REPLACE FUNCTION revert_supersession(
    p_supersession_id UUID,
    p_reason TEXT,
    p_actor TEXT
) RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    supersession memory_supersessions%ROWTYPE;
    normalized_reason TEXT := NULLIF(btrim(COALESCE(p_reason, '')), '');
    normalized_actor TEXT := NULLIF(btrim(COALESCE(p_actor, '')), '');
BEGIN
    IF normalized_reason IS NULL OR normalized_actor IS NULL THEN
        RAISE EXCEPTION 'revert_supersession requires non-empty reason and actor';
    END IF;

    SELECT *
    INTO supersession
    FROM memory_supersessions
    WHERE id = p_supersession_id
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'memory supersession % does not exist', p_supersession_id;
    END IF;
    IF supersession.status <> 'active' THEN
        RETURN FALSE;
    END IF;

    UPDATE memory_supersessions
    SET status = 'reverted',
        resolved_at = GREATEST(CURRENT_TIMESTAMP, superseded_at),
        metadata = metadata || jsonb_build_object(
            'reverted_by', normalized_actor,
            'revert_reason', normalized_reason
        )
    WHERE id = p_supersession_id;

    UPDATE memories
    SET superseded_by = CASE
            WHEN superseded_by IS NOT DISTINCT FROM supersession.replacement_memory_id
                THEN NULL
            ELSE superseded_by
        END,
        valid_until = CASE
            WHEN valid_until IS NOT DISTINCT FROM supersession.superseded_at
                THEN NULL
            ELSE valid_until
        END,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = supersession.superseded_memory_id;

    RETURN TRUE;
END;
$$;

-- Compatibility guard for direct writes that predate record_supersession().
-- Rich callers should use the function so their reason and actor are exact;
-- this trigger guarantees the lineage cannot silently disappear meanwhile.
CREATE OR REPLACE FUNCTION sync_memory_supersession_from_memory()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    derived_reason TEXT;
    derived_actor TEXT;
    effective_at TIMESTAMPTZ;
BEGIN
    IF TG_OP = 'UPDATE' AND OLD.superseded_by IS NOT DISTINCT FROM NEW.superseded_by THEN
        RETURN NEW;
    END IF;
    IF NEW.superseded_by IS NULL THEN
        IF TG_OP = 'UPDATE' AND OLD.superseded_by IS NOT NULL THEN
            UPDATE memory_supersessions
            SET replacement_memory_id = NULL,
                replacement_planned = FALSE,
                metadata = metadata || jsonb_build_object(
                    'replacement_pointer_cleared_at', CURRENT_TIMESTAMP
                )
            WHERE superseded_memory_id = NEW.id
              AND status = 'active'
              AND replacement_memory_id = OLD.superseded_by;
        END IF;
        RETURN NEW;
    END IF;
    IF EXISTS (
        SELECT 1
        FROM memory_supersessions
        WHERE superseded_memory_id = NEW.id
          AND status = 'active'
          AND replacement_memory_id = NEW.superseded_by
    ) THEN
        RETURN NEW;
    END IF;

    IF COALESCE(NEW.metadata, '{}'::jsonb) ? 'superseded_by_user_model_claim_id' THEN
        derived_reason := 'user model claim revised';
        derived_actor := 'connector_cognition';
    ELSIF COALESCE(NEW.metadata->'consolidation', '{}'::jsonb) ? 'archived_at' THEN
        derived_reason := 'retention consolidation';
        derived_actor := 'retention';
    ELSE
        derived_reason := 'legacy superseded_by write';
        derived_actor := 'database_sync';
    END IF;
    effective_at := COALESCE(NEW.valid_until, CURRENT_TIMESTAMP);

    PERFORM record_supersession(
        NEW.id,
        NEW.superseded_by,
        derived_reason,
        derived_actor,
        'active',
        effective_at,
        NULL,
        TRUE,
        jsonb_strip_nulls(jsonb_build_object(
            'source', 'memories.superseded_by trigger',
            'user_model_claim_id', NEW.metadata->>'superseded_by_user_model_claim_id'
        ))
    );
    RETURN NEW;
END;
$$;
