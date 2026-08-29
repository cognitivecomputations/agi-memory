-- Promote memory revision lineage into a durable side-table and make the
-- existing valid_from/valid_until columns authoritative at write time.
SET search_path = public, ag_catalog, "$user";

CREATE TABLE IF NOT EXISTS memory_supersessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    superseded_memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    replacement_memory_id UUID REFERENCES memories(id) ON DELETE SET NULL,
    reason TEXT NOT NULL CHECK (btrim(reason) <> ''),
    actor TEXT NOT NULL CHECK (btrim(actor) <> ''),
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'reverted', 'pending')),
    superseded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMPTZ,
    replacement_planned BOOLEAN NOT NULL DEFAULT FALSE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT memory_supersessions_no_self CHECK (
        replacement_memory_id IS NULL
        OR replacement_memory_id <> superseded_memory_id
    ),
    CONSTRAINT memory_supersessions_resolution_order CHECK (
        resolved_at IS NULL OR resolved_at >= superseded_at
    ),
    CONSTRAINT memory_supersessions_active_unresolved CHECK (
        status <> 'active' OR resolved_at IS NULL
    )
);

CREATE INDEX IF NOT EXISTS idx_memory_supersessions_superseded
    ON memory_supersessions (superseded_memory_id, superseded_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_supersessions_replacement
    ON memory_supersessions (replacement_memory_id, superseded_at DESC)
    WHERE replacement_memory_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_supersessions_active_per_memory
    ON memory_supersessions (superseded_memory_id)
    WHERE status = 'active';

ALTER TABLE memories
    ALTER COLUMN valid_from SET DEFAULT CURRENT_TIMESTAMP;
UPDATE memories
SET valid_from = COALESCE(valid_from, created_at, CURRENT_TIMESTAMP)
WHERE valid_from IS NULL;
ALTER TABLE memories
    ALTER COLUMN valid_from SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'memories'::regclass
          AND conname = 'memories_valid_time_order'
    ) THEN
        ALTER TABLE memories ADD CONSTRAINT memories_valid_time_order
            CHECK (valid_until IS NULL OR valid_until >= valid_from);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'memories'::regclass
          AND conname = 'memories_no_self_supersession'
    ) THEN
        ALTER TABLE memories ADD CONSTRAINT memories_no_self_supersession
            CHECK (superseded_by IS NULL OR superseded_by <> id);
    END IF;
END;
$$;

-- Recover legacy metadata pointers only when they name an existing memory.
UPDATE memories old_memory
SET superseded_by = replacement.id,
    valid_until = COALESCE(old_memory.valid_until, old_memory.updated_at, CURRENT_TIMESTAMP)
FROM memories replacement
WHERE old_memory.superseded_by IS NULL
  AND replacement.id::text = COALESCE(
      CASE
          WHEN old_memory.metadata->>'superseded_by' ~ '^[0-9a-fA-F-]{36}$'
              THEN old_memory.metadata->>'superseded_by'
      END,
      CASE
          WHEN old_memory.metadata->>'replacement_memory_id' ~ '^[0-9a-fA-F-]{36}$'
              THEN old_memory.metadata->>'replacement_memory_id'
      END
  );

-- The scalar pointer is the current-state authority during cutover. Preserve
-- every existing relation in the history table before installing sync triggers.
INSERT INTO memory_supersessions (
    superseded_memory_id,
    replacement_memory_id,
    reason,
    actor,
    status,
    superseded_at,
    replacement_planned,
    metadata
)
SELECT
    old_memory.id,
    old_memory.superseded_by,
    CASE
        WHEN old_memory.metadata ? 'superseded_by_user_model_claim_id'
            THEN 'user model claim revised'
        WHEN COALESCE(old_memory.metadata->'consolidation', '{}'::jsonb) ? 'archived_at'
            THEN 'retention consolidation'
        ELSE 'legacy supersession backfill'
    END,
    CASE
        WHEN old_memory.metadata ? 'superseded_by_user_model_claim_id'
            THEN 'connector_cognition'
        WHEN COALESCE(old_memory.metadata->'consolidation', '{}'::jsonb) ? 'archived_at'
            THEN 'retention'
        ELSE 'migration_0213'
    END,
    'active',
    COALESCE(old_memory.valid_until, old_memory.updated_at, old_memory.created_at, CURRENT_TIMESTAMP),
    TRUE,
    jsonb_build_object('source', 'memories.superseded_by backfill')
FROM memories old_memory
WHERE old_memory.superseded_by IS NOT NULL
ON CONFLICT (superseded_memory_id) WHERE status = 'active' DO NOTHING;

UPDATE memories old_memory
SET valid_until = supersession.superseded_at
FROM memory_supersessions supersession
WHERE supersession.superseded_memory_id = old_memory.id
  AND supersession.status = 'active'
  AND old_memory.valid_until IS NULL;

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

    SELECT * INTO old_memory
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
        SELECT * INTO active_supersession
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
        superseded_memory_id, replacement_memory_id, reason, actor, status,
        superseded_at, resolved_at, replacement_planned, metadata
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

    SELECT * INTO supersession
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
        SELECT 1 FROM memory_supersessions
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

DROP TRIGGER IF EXISTS trg_memory_temporal_validity_insert ON memories;
CREATE TRIGGER trg_memory_temporal_validity_insert
    BEFORE INSERT ON memories
    FOR EACH ROW
    EXECUTE FUNCTION normalize_memory_temporal_validity();
DROP TRIGGER IF EXISTS trg_memory_temporal_validity_update ON memories;
CREATE TRIGGER trg_memory_temporal_validity_update
    BEFORE UPDATE OF valid_from, valid_until, superseded_by ON memories
    FOR EACH ROW
    EXECUTE FUNCTION normalize_memory_temporal_validity();
DROP TRIGGER IF EXISTS trg_memory_supersession_insert ON memories;
CREATE TRIGGER trg_memory_supersession_insert
    AFTER INSERT ON memories
    FOR EACH ROW
    WHEN (NEW.superseded_by IS NOT NULL)
    EXECUTE FUNCTION sync_memory_supersession_from_memory();
DROP TRIGGER IF EXISTS trg_memory_supersession_update ON memories;
CREATE TRIGGER trg_memory_supersession_update
    AFTER UPDATE OF superseded_by ON memories
    FOR EACH ROW
    WHEN (OLD.superseded_by IS DISTINCT FROM NEW.superseded_by)
    EXECUTE FUNCTION sync_memory_supersession_from_memory();

-- Retention is an existing supersession writer. Route it through the explicit
-- event function and stop duplicating lineage inside memories.metadata.
CREATE OR REPLACE FUNCTION consolidate_memory_group(p_ids UUID[])
RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    v_ids UUID[];
    v_gist_id UUID;
    v_full_content TEXT;
    v_importance FLOAT;
    v_valence FLOAT;
    v_private BOOLEAN;
    v_orig UUID;
BEGIN
    SELECT array_agg(id ORDER BY created_at),
           string_agg(content, E'\n\n---\n\n' ORDER BY created_at),
           max(importance),
           avg((metadata->>'emotional_valence')::float),
           bool_or(source_attribution->>'sensitivity' = 'private')
      INTO v_ids, v_full_content, v_importance, v_valence, v_private
      FROM memories
      WHERE id = ANY(p_ids) AND status = 'active' AND type = 'episodic'
        AND NOT is_memory_protected(id);

    IF v_ids IS NULL OR array_length(v_ids, 1) < 2 THEN
        RETURN NULL;
    END IF;

    v_gist_id := create_memory_with_embedding(
        'episodic', v_full_content,
        (get_embedding(ARRAY[left(v_full_content, 8000)]))[1],
        LEAST(1.0, COALESCE(v_importance, 0.5)),
        jsonb_build_object('kind', 'consolidation', 'source', 'rest')
            || CASE WHEN v_private
                    THEN jsonb_build_object('sensitivity', 'private')
                    ELSE '{}'::jsonb END,
        NULL,
        jsonb_build_object('consolidation', jsonb_build_object(
            'role', 'merged', 'source_ids', to_jsonb(v_ids), 'summarized', false))
    );
    IF v_valence IS NOT NULL THEN
        UPDATE memories SET metadata = metadata || jsonb_build_object('emotional_valence', v_valence)
        WHERE id = v_gist_id;
    END IF;

    PERFORM merge_memory_edges(v_gist_id, v_ids);

    FOREACH v_orig IN ARRAY v_ids LOOP
        BEGIN
            PERFORM create_memory_relationship(v_gist_id, v_orig, 'DERIVED_FROM', '{}'::jsonb);
        EXCEPTION WHEN OTHERS THEN NULL;
        END;
        PERFORM record_supersession(
            v_orig,
            v_gist_id,
            'retention consolidation',
            'retention',
            'active',
            CURRENT_TIMESTAMP,
            NULL,
            TRUE,
            jsonb_build_object('source', 'consolidate_memory_group')
        );
    END LOOP;

    UPDATE memories SET
        status = 'archived',
        metadata = jsonb_set(metadata, '{consolidation}',
                     COALESCE(metadata->'consolidation', '{}'::jsonb)
                       || jsonb_build_object('archived_at', clock_timestamp()::text))
    WHERE id = ANY(v_ids);

    INSERT INTO memory_summarization_queue (memory_id) VALUES (v_gist_id)
    ON CONFLICT (memory_id) DO NOTHING;

    RETURN v_gist_id;
END;
$$;
