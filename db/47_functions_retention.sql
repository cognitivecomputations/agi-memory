-- Memory retention: the compression-native fade ladder
-- (docs/memory_retention_design.md §5, §6, §8, §9). During subconscious REST,
-- aged/low-strength/unprotected episodic memories are grouped, merged into one
-- summarized GIST (their lesson distilled upward first), the originals archived,
-- and — after a grace window — TRULY deleted, reclaiming representational mass
-- against a capacity dial.
--
-- Reversible consolidation ships on; retention.enabled remains the kill switch.
-- Irreversible deletion is separately opt-in and off by default.
-- Reuses phase-1 strength (calculate_strength) and the memory_edges helpers.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

-- ============================================================================
-- Part A: protection — what must never fade
-- ============================================================================
-- A memory is protected (exempt from consolidation AND pruning) if it is an
-- identity/goal type, highly important, emotionally charged (intensity or
-- valence), or explicitly pinned. Conservative by design: far worse to erase a
-- first kiss than to keep a mundane memory a cycle too long.
CREATE OR REPLACE FUNCTION is_memory_protected(p_memory_id UUID)
RETURNS BOOLEAN
LANGUAGE sql STABLE
AS $$
    SELECT EXISTS (
        SELECT 1 FROM memories m
        WHERE m.id = p_memory_id
          AND (
                m.type::text IN ('worldview', 'goal')
             OR COALESCE(m.importance, 0) >= COALESCE(get_config_float('retention.protect_importance'), 0.85)
             OR COALESCE((m.metadata->'emotional_context'->>'intensity')::float, 0)
                    >= COALESCE(get_config_float('retention.protect_intensity'), 0.75)
             OR abs(COALESCE((m.metadata->>'emotional_valence')::float, 0))
                    >= COALESCE(get_config_float('retention.protect_valence_abs'), 0.7)
             OR COALESCE((m.metadata->>'protected')::boolean, false)
             -- Ingested documents are the USER's data: never auto-fade them.
             -- They leave only with explicit user approval (resolve_document_fade).
             OR m.source_attribution->>'content_hash' IS NOT NULL
          )
    );
$$;

-- ============================================================================
-- Summarization queue (filled by consolidate_memory_group, drained by the worker)
-- ============================================================================
CREATE TABLE IF NOT EXISTS memory_summarization_queue (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id       UUID NOT NULL UNIQUE REFERENCES memories(id) ON DELETE CASCADE,
    status          TEXT NOT NULL DEFAULT 'pending',   -- pending|in_progress|done|failed
    attempts        INT  NOT NULL DEFAULT 0,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    enqueued_at     TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at    TIMESTAMPTZ,
    last_error      TEXT
);
CREATE INDEX IF NOT EXISTS idx_memory_summarization_queue_claim
    ON memory_summarization_queue (status, next_attempt_at);

-- ============================================================================
-- Part B: candidate selection + merge into a gist (reversible)
-- ============================================================================

-- Intelligent edge merge (§5d): carry the group's EXTERNAL relationships onto the
-- gist -- collapse duplicates by (direction, neighbor, rel_type), reinforce weight
-- by corroboration (w = 1 - Π(1-wᵢ)), drop intra-group edges. The originals' own
-- edges vanish when the originals are hard-deleted (GC); until then the phase-1
-- live-memory filter already hides edges to archived endpoints.
CREATE OR REPLACE FUNCTION merge_memory_edges(p_gist_id UUID, p_source_ids UUID[])
RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    src TEXT[] := ARRAY(SELECT unnest(p_source_ids)::text);
    rec RECORD;
BEGIN
    -- outgoing: source -> external neighbor
    FOR rec IN
        SELECT e.rel_type, e.dst_type, e.dst_id,
               (1 - exp(sum(ln(GREATEST(1e-9, 1 - LEAST(1.0, COALESCE(e.weight, 1.0))))))) AS w
        FROM memory_edges e
        WHERE e.src_type = 'memory' AND e.src_id = ANY(src)
          AND NOT (e.dst_type = 'memory' AND e.dst_id = ANY(src))
        GROUP BY e.rel_type, e.dst_type, e.dst_id
    LOOP
        PERFORM upsert_memory_edge('memory', p_gist_id::text, rec.rel_type, rec.dst_type, rec.dst_id,
                                   LEAST(1.0, rec.w), NULL, 'consolidation',
                                   jsonb_build_object('merged_from', to_jsonb(p_source_ids)));
    END LOOP;
    -- incoming: external neighbor -> source
    FOR rec IN
        SELECT e.rel_type, e.src_type, e.src_id,
               (1 - exp(sum(ln(GREATEST(1e-9, 1 - LEAST(1.0, COALESCE(e.weight, 1.0))))))) AS w
        FROM memory_edges e
        WHERE e.dst_type = 'memory' AND e.dst_id = ANY(src)
          AND NOT (e.src_type = 'memory' AND e.src_id = ANY(src))
        GROUP BY e.rel_type, e.src_type, e.src_id
    LOOP
        PERFORM upsert_memory_edge(rec.src_type, rec.src_id, rec.rel_type, 'memory', p_gist_id::text,
                                   LEAST(1.0, rec.w), NULL, 'consolidation',
                                   jsonb_build_object('merged_from', to_jsonb(p_source_ids)));
    END LOOP;
END;
$$;

-- Episode-grouped candidate groups: aged, idle, low-strength, unprotected active
-- episodic memories -- only groups with >= retention.min_group_size members.
CREATE OR REPLACE FUNCTION find_consolidation_candidates()
RETURNS TABLE (episode_id TEXT, memory_ids UUID[])
LANGUAGE sql STABLE
AS $$
    SELECT e.dst_id AS episode_id, array_agg(m.id ORDER BY m.created_at)
    FROM memory_edges e
    JOIN memories m ON m.id = _safe_uuid(e.src_id)
    WHERE e.rel_type = 'IN_EPISODE' AND e.dst_type = 'episode' AND e.src_type = 'memory'
      AND m.type = 'episodic'
      AND m.status = 'active'
      AND age_in_days(m.created_at) >= COALESCE(get_config_float('retention.min_age_days'), 30)
      AND (m.last_reinforced IS NULL OR age_in_days(m.last_reinforced) >= COALESCE(get_config_float('retention.min_idle_days'), 21))
      AND calculate_strength(m.importance, m.decay_rate, m.created_at, m.last_reinforced)
            < COALESCE(get_config_float('retention.consolidate_max_strength'), 0.4)
      AND NOT is_memory_protected(m.id)
    GROUP BY e.dst_id
    HAVING count(*) >= COALESCE(get_config_int('retention.min_group_size'), 3);
$$;

-- Merge a group of episodic memories into one GIST holding their full concatenated
-- content, migrate the graph onto it, archive the originals (reversible until GC),
-- and enqueue the gist for summarization+distillation. Returns the gist id (or
-- NULL if nothing eligible). Idempotent-ish: only touches active, unprotected rows.
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
        -- Sensitivity propagates from source to derivation (#92): one
        -- private memory in the group marks the merged gist private.
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
        EXCEPTION WHEN OTHERS THEN NULL;  -- provenance edge is best-effort (source_ids in metadata is canonical)
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

-- ============================================================================
-- Part C: summarization queue worker interface (LLM compaction + distill-upward)
-- ============================================================================

-- Claim a batch of gists to summarize (concurrency-safe). Returns (memory_id, content).
CREATE OR REPLACE FUNCTION claim_memory_summarization_batch(p_limit INT DEFAULT 8)
RETURNS TABLE (memory_id UUID, content TEXT)
LANGUAGE sql
AS $$
    WITH claimed AS (
        SELECT id
        FROM memory_summarization_queue
        WHERE status = 'pending' AND next_attempt_at <= CURRENT_TIMESTAMP
        ORDER BY enqueued_at
        FOR UPDATE SKIP LOCKED
        LIMIT GREATEST(COALESCE(p_limit, 8), 1)
    ),
    upd AS (
        UPDATE memory_summarization_queue q
           SET status = 'in_progress', attempts = attempts + 1
        WHERE q.id IN (SELECT id FROM claimed)
        RETURNING q.memory_id
    )
    SELECT u.memory_id, m.content
    FROM upd u JOIN memories m ON m.id = u.memory_id;
$$;

-- Apply the worker's result: compact the gist to the summary (drop fidelity, keep
-- the pre-summary text), and distill each durable lesson UPWARD into the schema
-- (semantic/strategic, deduped vs existing schema, DERIVED_FROM the gist).
CREATE OR REPLACE FUNCTION apply_memory_summary(
    p_id UUID,
    p_summary TEXT,
    p_lessons JSONB DEFAULT '[]'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_drop FLOAT := COALESCE(get_config_float('retention.fidelity_drop'), 0.7);
    lesson JSONB;
    v_lesson_id UUID;
    v_lesson_emb vector;
    v_dup UUID;
    v_created INT := 0;
    -- Sensitivity propagates from source to derivation (#92): lessons
    -- distilled from a private memory are themselves private.
    v_lesson_attr JSONB;
BEGIN
    IF COALESCE(btrim(p_summary), '') = '' THEN
        RAISE EXCEPTION 'summary must not be empty';
    END IF;

    SELECT jsonb_build_object('kind', 'distillation', 'from', p_id::text)
           || CASE WHEN m.source_attribution->>'sensitivity' = 'private'
                   THEN jsonb_build_object('sensitivity', 'private')
                   ELSE '{}'::jsonb END
      INTO v_lesson_attr
      FROM memories m WHERE m.id = p_id;
    v_lesson_attr := COALESCE(
        v_lesson_attr, jsonb_build_object('kind', 'distillation', 'from', p_id::text));

    UPDATE memories m SET
        content = p_summary,
        embedding = (get_embedding(ARRAY[p_summary]))[1],
        fidelity = GREATEST(0.0, LEAST(1.0, m.fidelity * v_drop)),
        metadata = jsonb_set(
                     jsonb_set(m.metadata,
                               '{consolidation,full_content}',
                               to_jsonb(COALESCE(m.metadata->'consolidation'->>'full_content', m.content)), true),
                     '{consolidation,summarized}', 'true'::jsonb, true),
        updated_at = CURRENT_TIMESTAMP
    WHERE m.id = p_id;

    FOR lesson IN SELECT * FROM jsonb_array_elements(COALESCE(p_lessons, '[]'::jsonb))
    LOOP
        CONTINUE WHEN COALESCE(btrim(lesson->>'content'), '') = '';
        v_lesson_emb := (get_embedding(ARRAY[lesson->>'content']))[1];
        -- schema dedup: skip lessons already known (>= 0.92 cosine to an active fact/pattern)
        SELECT id INTO v_dup FROM memories
        WHERE status = 'active' AND type IN ('semantic', 'strategic')
          AND (1 - (embedding <=> v_lesson_emb)) >= 0.92
        ORDER BY embedding <=> v_lesson_emb
        LIMIT 1;
        IF v_dup IS NOT NULL THEN CONTINUE; END IF;

        IF COALESCE(lesson->>'kind', 'semantic') = 'strategic' THEN
            v_lesson_id := create_strategic_memory(
                p_content := lesson->>'content',
                p_pattern_description := COALESCE(lesson->>'pattern', 'consolidated lesson'),
                p_confidence_score := 0.7,
                p_importance := 0.6,
                p_source_attribution := v_lesson_attr);
        ELSE
            v_lesson_id := create_semantic_memory(
                p_content := lesson->>'content',
                p_confidence := 0.7,
                p_importance := 0.55,
                p_source_attribution := v_lesson_attr);
        END IF;

        BEGIN
            PERFORM create_memory_relationship(v_lesson_id, p_id, 'DERIVED_FROM', '{}'::jsonb);
        EXCEPTION WHEN OTHERS THEN NULL;
        END;
        v_created := v_created + 1;
    END LOOP;

    UPDATE memory_summarization_queue
       SET status = 'done', completed_at = CURRENT_TIMESTAMP
     WHERE memory_id = p_id;

    RETURN jsonb_build_object('memory_id', p_id, 'lessons_created', v_created);
END;
$$;

-- Retry/backoff on worker failure (mirrors fail_recmem_consolidation_task).
CREATE OR REPLACE FUNCTION fail_memory_summarization(p_id UUID, p_error TEXT)
RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    v_attempts INT;
    v_max INT := COALESCE(get_config_int('memory.recmem_task_max_attempts'), 3);
    v_base INT := COALESCE(get_config_int('memory.recmem_task_backoff_base_s'), 30);
BEGIN
    SELECT attempts INTO v_attempts FROM memory_summarization_queue WHERE memory_id = p_id;
    UPDATE memory_summarization_queue SET
        status = CASE WHEN COALESCE(v_attempts, 1) >= v_max THEN 'failed' ELSE 'pending' END,
        next_attempt_at = CURRENT_TIMESTAMP + ((v_base * power(2, COALESCE(v_attempts, 1)))::text || ' seconds')::interval,
        last_error = left(COALESCE(p_error, ''), 1000)
    WHERE memory_id = p_id;
END;
$$;

-- ============================================================================
-- Part D: true deletion (cross-store) + the GC
-- ============================================================================

-- Per-node AGE cleanup (only a full-graph wipe existed before). Best-effort.
CREATE OR REPLACE FUNCTION remove_memory_node(p_id UUID)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    EXECUTE format(
        'SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
            MATCH (m:MemoryNode {memory_id: %L}) DETACH DELETE m
        $q$) as (result ag_catalog.agtype)', p_id);
EXCEPTION WHEN OTHERS THEN
    NULL;  -- graph cleanup is best-effort; memory_edges is the primary substrate
END;
$$;

-- Hard delete a memory and everything that references it, across all stores.
-- Ordered so the NO-ACTION FKs (reconsolidation_tasks) can't block the delete.
CREATE OR REPLACE FUNCTION delete_memory_fully(p_id UUID)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
BEGIN
    -- 1. NO-ACTION FKs that would otherwise raise on the final delete
    DELETE FROM reconsolidation_tasks WHERE belief_id = p_id OR summary_memory_id = p_id;
    -- 2. relational edge substrate (TEXT ids, no FK) -- both directions
    DELETE FROM memory_edges
     WHERE (src_type = 'memory' AND src_id = p_id::text)
        OR (dst_type = 'memory' AND dst_id = p_id::text);
    -- 3. activation cache (no FK)
    DELETE FROM activation_cache WHERE memory_id = p_id;
    -- 4. AGE graph node
    PERFORM remove_memory_node(p_id);
    -- 5. the row (CASCADE cleans memory_source_units + memory_neighborhoods; SET-NULL FKs auto-null)
    DELETE FROM memories WHERE id = p_id;
    RETURN TRUE;
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING 'delete_memory_fully(%) failed: %', p_id, SQLERRM;
    RETURN FALSE;
END;
$$;

-- Reclaim: hard-delete archived originals past the grace/undo window, then (if a
-- capacity dial is set and episodic mass is still over it) prune the weakest live
-- episodic memories until under target. Protected memories are never touched.
CREATE OR REPLACE FUNCTION run_retention_gc()
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_grace    FLOAT := COALESCE(get_config_float('retention.prune_grace_days'), 14);
    v_capacity FLOAT := COALESCE(get_config_float('retention.capacity'), 0);
    v_pruned INT := 0;
    v_expired INT := 0;
    v_mass FLOAT;
    v_target UUID;
    rec RECORD;
    v_irreversible BOOLEAN := COALESCE(
        get_config_bool('retention.irreversible_pruning_enabled'), FALSE
    );
BEGIN
    IF NOT COALESCE(get_config_bool('retention.enabled'), false) THEN
        RETURN jsonb_build_object('skipped', true);
    END IF;

    -- A timer never chooses forgetting. Expired review horizons remain pending
    -- until the user or conscious agent makes an explicit decision.
    SELECT count(*)::int INTO v_expired
    FROM memory_review_queue
    WHERE status = 'pending' AND expires_at <= CURRENT_TIMESTAMP;

    -- (a) archived originals past grace (the undo window) -> truly delete
    IF v_irreversible THEN
        FOR rec IN
            SELECT id FROM memories
            WHERE status = 'archived' AND superseded_by IS NOT NULL
              AND age_in_days(COALESCE((metadata->'consolidation'->>'archived_at')::timestamptz, updated_at)) >= v_grace
              AND NOT is_memory_protected(id)
        LOOP
            IF delete_memory_fully(rec.id) THEN v_pruned := v_pruned + 1; END IF;
        END LOOP;
    END IF;

    -- (b) capacity pressure -> prune the weakest live episodic memories (last resort)
    IF v_irreversible AND v_capacity > 0 THEN
        LOOP
            SELECT COALESCE(sum(calculate_strength(importance, decay_rate, created_at, last_reinforced)), 0)
              INTO v_mass FROM memories WHERE status = 'active' AND type = 'episodic';
            EXIT WHEN v_mass <= v_capacity;
            SELECT id INTO v_target FROM memories
            WHERE status = 'active' AND type = 'episodic' AND NOT is_memory_protected(id)
            ORDER BY calculate_strength(importance, decay_rate, created_at, last_reinforced) ASC, created_at ASC
            LIMIT 1;
            EXIT WHEN v_target IS NULL;
            EXIT WHEN NOT delete_memory_fully(v_target);
            v_pruned := v_pruned + 1;
        END LOOP;
    END IF;

    -- (c) agent-acquired sources: the ownership-based tier the agent may
    -- retire on its own (user-provided sources only fade via approval).
    RETURN jsonb_build_object(
        'pruned', v_pruned,
        'irreversible_pruning_enabled', v_irreversible,
        'reviews_awaiting_decision', v_expired,
        'agent_sources', run_agent_source_retention()
    );
END;
$$;

-- ============================================================================
-- Part E: the rest orchestrator (DB half; the LLM summarization runs in the worker)
-- ============================================================================
-- Group aged/low-strength/unprotected episodic memories and merge each group into
-- a gist (archiving the originals, enqueuing summarization). No LLM here -- the
-- summarization worker drains the queue. No-op unless retention.enabled.
CREATE OR REPLACE FUNCTION run_memory_rest()
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_batch INT := COALESCE(get_config_int('retention.rest_batch_size'), 8);
    v_escalate_max INT := GREATEST(0, COALESCE(get_config_int('retention.escalate_batch'), 3));
    v_expiry_days INT := GREATEST(1, COALESCE(get_config_int('retention.review_expiry_days'), 7));
    v_consolidated INT := 0;
    v_escalated INT := 0;
    v_gist UUID;
    v_preview TEXT;
    rec RECORD;
BEGIN
    IF NOT COALESCE(get_config_bool('retention.enabled'), false) THEN
        RETURN jsonb_build_object('skipped', true);
    END IF;

    -- Refill the KEEP budget if we've entered a new life chapter.
    PERFORM reset_retention_budget_if_new_chapter();

    FOR rec IN SELECT episode_id, memory_ids FROM find_consolidation_candidates() LIMIT GREATEST(v_batch, 1)
    LOOP
        BEGIN
            -- A group already awaiting a conscious decision is left alone (neither
            -- re-escalated nor consolidated behind the conscious mind's back).
            IF EXISTS (SELECT 1 FROM memory_review_queue q
                       WHERE q.status = 'pending' AND q.memory_ids && rec.memory_ids) THEN
                CONTINUE;
            END IF;
            -- Borderline groups escalate to the conscious mind -- but only up to
            -- escalate_batch per pass, and only when there's budget to actually act
            -- (out of points => the subconscious just proceeds). Everything else
            -- consolidates silently, exactly as before.
            IF is_consolidation_borderline(rec.memory_ids)
               AND v_escalated < v_escalate_max
               AND retention_budget_remaining() > 0
            THEN
                SELECT string_agg(left(content, 120), ' / ' ORDER BY created_at)
                  INTO v_preview FROM memories WHERE id = ANY(rec.memory_ids) AND status = 'active';
                INSERT INTO memory_review_queue (episode_id, memory_ids, reason, preview, expires_at)
                VALUES (rec.episode_id, rec.memory_ids, 'near_protection_threshold', v_preview,
                        CURRENT_TIMESTAMP + (v_expiry_days || ' days')::interval);
                v_escalated := v_escalated + 1;
            ELSE
                v_gist := consolidate_memory_group(rec.memory_ids);
                IF v_gist IS NOT NULL THEN v_consolidated := v_consolidated + 1; END IF;
            END IF;
        EXCEPTION WHEN OTHERS THEN
            RAISE WARNING 'memory rest group failed: %', SQLERRM;
        END;
    END LOOP;
    RETURN jsonb_build_object('consolidated', v_consolidated, 'escalated', v_escalated);
END;
$$;

-- ============================================================================
-- Part F: subconscious triage -> conscious veto (docs/memory_retention_design.md §5)
-- Most forgetting stays pre-conscious; BORDERLINE consolidations escalate to the
-- conscious heartbeat, which may spend a point from a finite, per-life-chapter
-- budget to KEEP a memory, JOURNAL it, or LET IT GO (the default).
-- ============================================================================

CREATE TABLE IF NOT EXISTS memory_review_queue (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id  TEXT,
    memory_ids  UUID[] NOT NULL,
    reason      TEXT,                                  -- why the subconscious was unsure
    preview     TEXT,                                  -- short content for the conscious mind
    status      TEXT NOT NULL DEFAULT 'pending',       -- pending|kept|released|expired
    decision    TEXT CHECK (decision IN ('keep', 'release', 'journal')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP + INTERVAL '7 days',
    decided_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_memory_review_queue_pending
    ON memory_review_queue (status, expires_at);

-- Current life chapter, read RELATIONALLY (no AGE dependency in the rest cycle).
CREATE OR REPLACE FUNCTION current_chapter_relational()
RETURNS TEXT
LANGUAGE sql STABLE
AS $$
    SELECT COALESCE(
        (SELECT properties->>'name' FROM memory_edges
          WHERE src_type = 'self' AND dst_type = 'life_chapter' AND dst_id = 'current' LIMIT 1),
        'unknown');
$$;

-- Finite KEEP budget in the state KV as {chapter, remaining, total}. Refills when
-- the life chapter changes -- a fresh chapter is a fresh allowance to hold onto.
CREATE OR REPLACE FUNCTION reset_retention_budget_if_new_chapter()
RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    v_total   INT  := GREATEST(0, COALESCE(get_config_int('retention.veto_budget_per_chapter'), 5));
    v_chapter TEXT := current_chapter_relational();
    v_state   JSONB := get_state('retention_veto_budget');
BEGIN
    IF v_state IS NULL OR COALESCE(v_state->>'chapter', '') IS DISTINCT FROM v_chapter THEN
        PERFORM set_state('retention_veto_budget', jsonb_build_object(
            'chapter', v_chapter, 'remaining', v_total, 'total', v_total));
    END IF;
END;
$$;

CREATE OR REPLACE FUNCTION retention_budget_remaining()
RETURNS INT
LANGUAGE sql STABLE
AS $$
    SELECT COALESCE((get_state('retention_veto_budget')->>'remaining')::int, 0);
$$;

-- Spend one point if any remain; returns true on success.
CREATE OR REPLACE FUNCTION spend_retention_budget()
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    v_state JSONB := get_state('retention_veto_budget');
    v_remaining INT := COALESCE((v_state->>'remaining')::int, 0);
BEGIN
    IF v_remaining <= 0 THEN
        RETURN false;
    END IF;
    PERFORM set_state('retention_veto_budget',
        COALESCE(v_state, '{}'::jsonb) || jsonb_build_object('remaining', v_remaining - 1));
    RETURN true;
END;
$$;

-- A candidate group is BORDERLINE (worth the conscious mind's attention) when any
-- member sits just below a protection threshold -- close enough that the cheap
-- subconscious shouldn't decide alone. Not-yet-protected, but nearly precious.
CREATE OR REPLACE FUNCTION is_consolidation_borderline(p_ids UUID[])
RETURNS BOOLEAN
LANGUAGE sql STABLE
AS $$
    SELECT EXISTS (
        SELECT 1 FROM memories m
        WHERE m.id = ANY(p_ids) AND m.status = 'active'
          AND NOT is_memory_protected(m.id)
          AND (
                m.importance >= COALESCE(get_config_float('retention.protect_importance'), 0.85)
                                - COALESCE(get_config_float('retention.borderline_margin'), 0.15)
             OR current_emotional_intensity((m.metadata->'emotional_context'->>'intensity')::float,
                    (m.metadata->>'emotional_valence')::float, m.created_at, m.last_reinforced)
                    >= COALESCE(get_config_float('retention.protect_intensity'), 0.75)
                       - COALESCE(get_config_float('retention.borderline_margin'), 0.15)
             OR abs(COALESCE((m.metadata->>'emotional_valence')::float, 0))
                    >= COALESCE(get_config_float('retention.protect_valence_abs'), 0.7)
                       - COALESCE(get_config_float('retention.borderline_margin'), 0.15)
             -- Relationally significant: this memory is cited as evidence for a
             -- relationship. Memories about the people we care about deserve a
             -- conscious look before they fade.
             OR EXISTS (SELECT 1 FROM memory_edges e
                        WHERE e.dst_type = 'concept'
                          AND e.properties->>'kind' = 'relationship'
                          AND e.properties->>'evidence_memory_id' = m.id::text)
             -- Poor schema fit: nothing in the semantic/strategic schema is close,
             -- so this is novel knowledge not yet captured elsewhere -- worth a look
             -- rather than a silent merge. Opt-in (0 disables); nearest-neighbour
             -- via the HNSW index.
             OR (COALESCE(get_config_float('retention.borderline_schema_fit'), 0) > 0
                 AND m.embedding IS NOT NULL
                 AND COALESCE((SELECT 1 - (s.embedding <=> m.embedding)
                               FROM memories s
                               WHERE s.status = 'active' AND s.type IN ('semantic', 'strategic') AND s.id <> m.id
                               ORDER BY s.embedding <=> m.embedding
                               LIMIT 1), 0)
                     < get_config_float('retention.borderline_schema_fit'))
          )
    );
$$;

-- The conscious-review slice for the heartbeat context: memories at the threshold of
-- fading + how many KEEP points remain. Empty reviews => the section renders nothing.
CREATE OR REPLACE FUNCTION get_memories_at_threshold_context(p_limit INT DEFAULT 5)
RETURNS JSONB
LANGUAGE sql STABLE
AS $$
    SELECT jsonb_build_object(
        'budget_remaining', retention_budget_remaining(),
        'reviews', COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
                'review_id', q.id,
                'preview', q.preview,
                'reason', q.reason,
                'memory_ids', to_jsonb(q.memory_ids),
                'expires_at', q.expires_at
            ) ORDER BY q.created_at)
            FROM (SELECT * FROM memory_review_queue
                  WHERE status = 'pending' ORDER BY created_at LIMIT GREATEST(p_limit, 1)) q
        ), '[]'::jsonb)
    );
$$;

-- ============================================================================
-- Part G: ingested-document approval (the USER's data; user keeps control)
-- Ingested memories are auto-fade-immune (see is_memory_protected). Hexis instead
-- ASKS the user via the outbox before letting a stale document go; the user
-- approves/keeps via a chat tool. A document's memories share source_attribution
-- ->>'content_hash'; label = ->>'label'; ingest time = ->>'observed_at'.
-- ============================================================================

CREATE TABLE IF NOT EXISTS document_fade_requests (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_hash  TEXT NOT NULL UNIQUE,
    label         TEXT,
    memory_count  INT  NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'pending',   -- pending|approved|kept
    requested_at  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    decided_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_document_fade_requests_status
    ON document_fade_requests (status);

-- Ingested documents old enough AND untouched long enough to seem stale, with no
-- outstanding request. Grouped by content_hash (a document's shared identity).
CREATE OR REPLACE FUNCTION find_stale_ingested_documents()
RETURNS TABLE (content_hash TEXT, label TEXT, memory_count INT)
LANGUAGE sql STABLE
AS $$
    SELECT g.content_hash, g.label, g.memory_count
    FROM (
        SELECT m.source_attribution->>'content_hash' AS content_hash,
               max(m.source_attribution->>'label') AS label,
               count(*)::int AS memory_count
        FROM memories m
        WHERE m.status = 'active'
          AND m.source_attribution->>'content_hash' IS NOT NULL
          -- Protected memories (e.g. seeded origin documents) never fade.
          AND COALESCE((m.metadata->>'protected')::boolean, FALSE) = FALSE
        GROUP BY m.source_attribution->>'content_hash'
        HAVING max(age_in_days(COALESCE((m.source_attribution->>'observed_at')::timestamptz, m.created_at)))
                 >= COALESCE(get_config_float('retention.doc_stale_days'), 180)
           AND min(age_in_days(GREATEST(m.last_reinforced, m.last_accessed, m.created_at)))
                 >= COALESCE(get_config_float('retention.doc_idle_days'), 90)
    ) g
    WHERE NOT EXISTS (
        SELECT 1 FROM document_fade_requests r
        WHERE r.content_hash = g.content_hash AND r.status = 'pending');
$$;

-- Seek the user's approval for up to doc_request_batch stale documents (records a
-- pending request + queues one outbox message each). No-op unless retention.enabled.
CREATE OR REPLACE FUNCTION request_stale_document_fades()
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_batch INT := GREATEST(0, COALESCE(get_config_int('retention.doc_request_batch'), 2));
    v_sent INT := 0;
    rec RECORD;
BEGIN
    IF NOT COALESCE(get_config_bool('retention.enabled'), false) THEN
        RETURN jsonb_build_object('skipped', true);
    END IF;
    FOR rec IN SELECT content_hash, label, memory_count
               FROM find_stale_ingested_documents() LIMIT GREATEST(v_batch, 1)
    LOOP
        EXIT WHEN v_sent >= v_batch;
        INSERT INTO document_fade_requests (content_hash, label, memory_count)
        VALUES (rec.content_hash, rec.label, rec.memory_count)
        ON CONFLICT (content_hash) DO NOTHING;
        IF FOUND THEN
            PERFORM queue_outbox_message(
                'I read "' || COALESCE(rec.label, 'a document') || '" a while back and haven''t drawn on it since. '
                || 'Want me to let it fade, or keep it? Just tell me.',
                'document_fade', 'retention',
                jsonb_build_object('mode', 'web_inbox', 'content_hash', rec.content_hash));
            v_sent := v_sent + 1;
        END IF;
    END LOOP;
    RETURN jsonb_build_object('requested', v_sent);
END;
$$;

-- The user's verdict on a stale-document approval. Matches a pending request by
-- exact content_hash or (fuzzy) label -- so the LLM can pass the doc name the user
-- used. 'approve' truly deletes every memory of the document; anything else KEEPS
-- them (the safe default -- never delete without an explicit approve) and lifts them.
CREATE OR REPLACE FUNCTION resolve_document_fade(p_ref TEXT, p_decision TEXT)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_hash  TEXT;
    v_label TEXT;
    v_ids   UUID[];
    v_affected INT := 0;
    v_id UUID;
BEGIN
    IF p_ref IS NULL OR btrim(p_ref) = '' THEN
        RETURN jsonb_build_object('error', 'no document reference given');
    END IF;
    SELECT content_hash, label INTO v_hash, v_label
    FROM document_fade_requests
    WHERE status = 'pending'
      AND (content_hash = p_ref OR label ILIKE p_ref OR label ILIKE '%' || p_ref || '%')
    ORDER BY (content_hash = p_ref) DESC, requested_at
    LIMIT 1;
    IF v_hash IS NULL THEN
        RETURN jsonb_build_object('error', 'no pending document approval matches', 'ref', p_ref);
    END IF;

    v_ids := ARRAY(SELECT id FROM memories
                   WHERE status = 'active' AND source_attribution->>'content_hash' = v_hash);

    IF lower(COALESCE(p_decision, '')) = 'approve' THEN
        FOREACH v_id IN ARRAY v_ids LOOP
            IF delete_memory_fully(v_id) THEN v_affected := v_affected + 1; END IF;
        END LOOP;
        -- Approval cascades to the filing cabinet: the source document is
        -- archived (tombstone kept, never silently deleted), its durable
        -- chunks are removed, and preserved artifact bytes are released.
        -- Desk copies are swept by recmem_gc's archived-source pass.
        DELETE FROM source_document_chunks c
        USING source_documents d
        WHERE c.source_document_id = d.id
          AND d.content_hash = v_hash;
        UPDATE source_artifacts a
        SET bytes = NULL,
            status = 'archived',
            metadata = a.metadata || jsonb_build_object(
                'retention', jsonb_build_object(
                    'bytes_released_at', CURRENT_TIMESTAMP,
                    'reason', 'document_fade_approved')),
            updated_at = CURRENT_TIMESTAMP
        FROM source_documents d
        WHERE a.source_document_id = d.id
          AND d.content_hash = v_hash
          AND a.status <> 'redacted';
        UPDATE source_documents
        SET status = 'archived',
            metadata = metadata || jsonb_build_object(
                'retention', jsonb_build_object(
                    'archived_at', CURRENT_TIMESTAMP,
                    'reason', 'document_fade_approved')),
            updated_at = CURRENT_TIMESTAMP
        WHERE content_hash = v_hash AND status = 'active';
        UPDATE document_fade_requests SET status = 'approved', decided_at = CURRENT_TIMESTAMP
         WHERE content_hash = v_hash;
        RETURN jsonb_build_object('decision', 'approve', 'label', v_label, 'faded', v_affected);
    ELSE
        IF array_length(v_ids, 1) > 0 THEN PERFORM touch_memories(v_ids); END IF;
        v_affected := COALESCE(array_length(v_ids, 1), 0);
        UPDATE document_fade_requests SET status = 'kept', decided_at = CURRENT_TIMESTAMP
         WHERE content_hash = v_hash;
        RETURN jsonb_build_object('decision', 'keep', 'label', v_label, 'kept', v_affected);
    END IF;
END;
$$;

-- ============================================================================
-- Agent-acquired source retention (ownership-based two-tier policy).
--
-- User-provided sources (CLI, uploads, chat drops, connectors) are the
-- USER's data: they never auto-fade — only the fade-request → approval flow
-- above touches them. Sources the AGENT chose to keep on its own
-- (source_attribution->>'acquisition' = 'agent': url_ingest/git_ingest from
-- a heartbeat) are the agent's working set, so the daily subconscious pass
-- may archive them autonomously once truly idle — reversibly (doc archived,
-- chunks kept, artifact bytes kept). Escalates to a user fade request
-- instead when the source is heavily referenced by memories.
-- ============================================================================

INSERT INTO config_defaults (key, value, description) VALUES
    ('retention.agent_source_idle_days', '60'::jsonb,
     'Archive agent-acquired sources untouched (chunks, desk, memories) for this many days'),
    ('retention.agent_source_escalate_memories', '5'::jsonb,
     'Agent-acquired sources cited by at least this many active memories escalate to a user fade request instead of auto-archiving'),
    ('retention.agent_source_batch', '5'::jsonb,
     'Agent-acquired sources processed per daily retention pass')
ON CONFLICT (key) DO NOTHING;

CREATE OR REPLACE FUNCTION run_agent_source_retention()
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_idle_days FLOAT := GREATEST(COALESCE(get_config_float('retention.agent_source_idle_days'), 60), 1);
    v_escalate INT := GREATEST(COALESCE(get_config_int('retention.agent_source_escalate_memories'), 5), 1);
    v_batch INT := GREATEST(COALESCE(get_config_int('retention.agent_source_batch'), 5), 1);
    v_archived INT := 0;
    v_escalated INT := 0;
    rec RECORD;
    v_memory_count INT;
BEGIN
    IF NOT COALESCE(get_config_bool('retention.enabled'), false) THEN
        RETURN jsonb_build_object('skipped', true);
    END IF;

    FOR rec IN
        SELECT d.id, d.content_hash, d.title
        FROM source_documents d
        WHERE d.status = 'active'
          AND d.source_attribution->>'acquisition' = 'agent'
          AND age_in_days(d.last_ingested_at) >= v_idle_days
          -- no recently-touched chunks
          AND NOT EXISTS (
              SELECT 1 FROM source_document_chunks c
              WHERE c.source_document_id = d.id
                AND c.last_accessed IS NOT NULL
                AND age_in_days(c.last_accessed) < v_idle_days
          )
          -- nothing of it still on the active desk
          AND NOT EXISTS (
              SELECT 1 FROM subconscious_units u
              WHERE u.status = 'active'
                AND u.metadata #>> '{recmem,kind}' = 'source_document_desk'
                AND u.metadata #>> '{recmem,document_id}' = d.id::text
          )
          -- no recently-reinforced memories citing it
          AND NOT EXISTS (
              SELECT 1 FROM memories m
              WHERE m.status = 'active'
                AND m.source_attribution->>'content_hash' = d.content_hash
                AND age_in_days(GREATEST(m.last_reinforced, m.last_accessed, m.created_at)) < v_idle_days
          )
          AND NOT EXISTS (
              SELECT 1 FROM document_fade_requests r
              WHERE r.content_hash = d.content_hash AND r.status = 'pending'
          )
        ORDER BY d.last_ingested_at
        LIMIT v_batch
    LOOP
        SELECT count(*) INTO v_memory_count
        FROM memories m
        WHERE m.status = 'active'
          AND m.source_attribution->>'content_hash' = rec.content_hash;

        IF v_memory_count >= v_escalate THEN
            -- Heavily referenced: this rose to user-attention level.
            INSERT INTO document_fade_requests (content_hash, label, memory_count)
            VALUES (rec.content_hash, rec.title, v_memory_count)
            ON CONFLICT (content_hash) DO NOTHING;
            IF FOUND THEN
                PERFORM queue_outbox_message(
                    'A while back I fetched "' || COALESCE(rec.title, 'a web source')
                    || '" on my own and built ' || v_memory_count || ' memories from it, '
                    || 'but I haven''t drawn on it lately. Want me to let it fade, or keep it?',
                    'document_fade', 'retention',
                    jsonb_build_object('mode', 'web_inbox', 'content_hash', rec.content_hash));
                v_escalated := v_escalated + 1;
            END IF;
        ELSE
            -- Low-stakes: archive reversibly (chunks and artifact bytes kept;
            -- re-ingesting or un-archiving restores full retrieval).
            UPDATE source_documents
            SET status = 'archived',
                metadata = metadata || jsonb_build_object(
                    'retention', jsonb_build_object(
                        'archived_at', CURRENT_TIMESTAMP,
                        'reason', 'agent_source_idle')),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = rec.id AND status = 'active';
            v_archived := v_archived + 1;
        END IF;
    END LOOP;

    RETURN jsonb_build_object('archived', v_archived, 'escalated', v_escalated);
END;
$$;

-- Pending stale-document approval asks, so the agent can enumerate/confirm.
CREATE OR REPLACE FUNCTION list_document_fade_requests()
RETURNS JSONB
LANGUAGE sql STABLE
AS $$
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'content_hash', content_hash, 'label', label,
        'memory_count', memory_count, 'requested_at', requested_at
    ) ORDER BY requested_at), '[]'::jsonb)
    FROM document_fade_requests WHERE status = 'pending';
$$;

-- Tool dispatch (mirrors execute_journal_tool: {success, output, display_output}).
CREATE OR REPLACE FUNCTION execute_document_tool(p_tool TEXT, p_params JSONB DEFAULT '{}'::jsonb)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_out JSONB;
BEGIN
    IF p_tool = 'list_document_fade_requests' THEN
        v_out := list_document_fade_requests();
        RETURN jsonb_build_object('success', true, 'output', v_out,
            'display_output', 'Documents awaiting your approval to fade: ' || jsonb_array_length(v_out));
    ELSIF p_tool = 'resolve_document_fade' THEN
        v_out := resolve_document_fade(p_params->>'document', p_params->>'decision');
        IF v_out ? 'error' THEN
            RETURN jsonb_build_object('success', false, 'error', v_out->>'error');
        END IF;
        RETURN jsonb_build_object('success', true, 'output', v_out,
            'display_output', initcap(COALESCE(v_out->>'decision', 'kept')) || ' "' || COALESCE(v_out->>'label', 'document') || '"');
    ELSE
        RETURN jsonb_build_object('success', false, 'error', 'unknown document tool: ' || COALESCE(p_tool, '<null>'));
    END IF;
END;
$$;

-- ============================================================================
-- Observability: a single snapshot of everything the retention system holds and
-- would do -- so an operator can SEE it before (and after) flipping retention on.
-- Candidate counts are computed even while disabled (a preview of what would fade).
-- ============================================================================
CREATE OR REPLACE FUNCTION retention_status()
RETURNS JSONB
LANGUAGE sql STABLE
AS $$
    SELECT jsonb_build_object(
        'enabled', COALESCE(get_config_bool('retention.enabled'), false),
        'irreversible_pruning_enabled', COALESCE(
            get_config_bool('retention.irreversible_pruning_enabled'), false
        ),
        'episodic', jsonb_build_object(
            'active', (SELECT count(*) FROM memories WHERE status = 'active' AND type = 'episodic'),
            'mass', (SELECT round(COALESCE(sum(calculate_strength(importance, decay_rate, created_at, last_reinforced)), 0)::numeric, 2)
                     FROM memories WHERE status = 'active' AND type = 'episodic'),
            'capacity', COALESCE(get_config_float('retention.capacity'), 0),
            'archived', (SELECT count(*) FROM memories WHERE status = 'archived')),
        'consolidation', jsonb_build_object(
            'candidate_groups', (SELECT count(*) FROM find_consolidation_candidates()),
            'gists', (SELECT count(*) FROM memories WHERE status = 'active' AND metadata->'consolidation'->>'role' = 'merged'),
            'summarize_pending', (SELECT count(*) FROM memory_summarization_queue WHERE status = 'pending')),
        'conscious_review', jsonb_build_object(
            'pending', (SELECT count(*) FROM memory_review_queue WHERE status = 'pending'),
            'veto_budget', get_state('retention_veto_budget')),
        'documents', jsonb_build_object(
            'protected', (SELECT count(DISTINCT source_attribution->>'content_hash')
                          FROM memories WHERE status = 'active' AND source_attribution->>'content_hash' IS NOT NULL),
            'approvals_pending', (SELECT count(*) FROM document_fade_requests WHERE status = 'pending'),
            'approval_labels', (SELECT COALESCE(jsonb_agg(label ORDER BY requested_at), '[]'::jsonb)
                                FROM document_fade_requests WHERE status = 'pending'))
    );
$$;

-- Simulate ONE rest cycle and return the diff -- WITHOUT changing anything. The
-- whole cycle (temporarily enabling retention, consolidating, pruning, queuing
-- document asks) runs inside a subtransaction that is always rolled back before
-- returning; the plpgsql result variable survives the rollback, so the caller
-- gets a truthful preview using the REAL functions and a clean database after.
-- Safe to call whether retention is on or off, and even inside another txn.
CREATE OR REPLACE FUNCTION retention_dry_run()
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_before JSONB := retention_status();
    v_rest JSONB;
    v_gc JSONB;
    v_docs JSONB;
    v_after JSONB;
    v_diff JSONB;
BEGIN
    BEGIN
        UPDATE config SET value = 'true'::jsonb WHERE key = 'retention.enabled';
        v_rest := run_memory_rest();
        v_gc   := run_retention_gc();
        v_docs := request_stale_document_fades();
        v_after := retention_status();
        v_diff := jsonb_build_object(
            'dry_run', true,
            'rest', v_rest, 'gc', v_gc, 'documents', v_docs,
            'before', v_before, 'after', v_after);
        RAISE EXCEPTION 'DRY_RUN_ROLLBACK' USING ERRCODE = 'P0001';
    EXCEPTION WHEN OTHERS THEN
        IF v_diff IS NULL THEN
            RAISE;  -- a genuine error before we captured the diff -> surface it
        END IF;
    END;
    RETURN v_diff;
END;
$$;
