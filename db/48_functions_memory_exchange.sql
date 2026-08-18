-- ============================================================================
-- HMX (Hexis Memory Exchange) v1.7 — export functions (plans/hmx.md, Slice 1)
--
-- Each function returns the raw section data as JSONB with LOCAL UUIDs and no
-- embeddings. The Python layer (core/memory_exchange.py) applies export-id ref
-- scoping, content hashes, provenance enrichment, and section digests — so the
-- one canonical hashing implementation lives in core/digest.py.
--
-- Mirrored as db/migrations/0004_hmx_export_functions.sql for existing DBs.
-- ============================================================================
-- Each baseline file runs as its own psql session at initdb, so it must pin
-- its own search_path: under an ag_catalog-first cluster default, unqualified
-- CREATEs would land Hexis objects in ag_catalog (the #77 fossil bug).
SET search_path = public, ag_catalog, "$user";

SET check_function_bodies = off;

-- Memories section: the non-protected memory types. Worldview and goal
-- memories are exported through their dedicated (protected) sections instead,
-- so one row never rides in two sections.
CREATE OR REPLACE FUNCTION hmx_export_memories(
    p_types TEXT[] DEFAULT NULL,
    p_since TIMESTAMPTZ DEFAULT NULL,
    p_until TIMESTAMPTZ DEFAULT NULL,
    -- Sensitivity egress control (#92): private-marked memories travel only
    -- when the caller opts in — port/duplicate (her whole brain moving house)
    -- pass TRUE; telepathy/analysis default FALSE.
    p_include_sensitive BOOLEAN DEFAULT FALSE
) RETURNS JSONB AS $$
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', m.id,
        'type', m.type,
        'status', m.status,
        'content', m.content,
        'importance', m.importance,
        'trust_level', m.trust_level,
        'decay_rate', m.decay_rate,
        'created_at', m.created_at,
        'updated_at', m.updated_at,
        'valid_from', m.valid_from,
        'valid_until', m.valid_until,
        'access_count', m.access_count,
        'last_accessed', m.last_accessed,
        'superseded_by', m.superseded_by,
        'source_attribution', m.source_attribution,
        'metadata', m.metadata
    ) ORDER BY m.created_at, m.id), '[]'::jsonb)
    FROM memories m
    WHERE m.status IN ('active', 'archived')
      AND m.type::text = ANY(COALESCE(p_types, ARRAY['episodic','semantic','procedural','strategic']))
      AND m.type::text NOT IN ('worldview', 'goal')
      AND (p_include_sensitive
           OR COALESCE(m.source_attribution->>'sensitivity', '') <> 'private')
      AND (p_since IS NULL OR m.created_at >= p_since)
      AND (p_until IS NULL OR m.created_at <= p_until);
$$ LANGUAGE sql STABLE;

-- Episodes with membership resolved from the flat edge substrate.
CREATE OR REPLACE FUNCTION hmx_export_episodes() RETURNS JSONB AS $$
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', e.id,
        'started_at', e.started_at,
        'ended_at', e.ended_at,
        'summary', e.summary,
        'metadata', e.metadata,
        'memory_ids', COALESCE((
            SELECT jsonb_agg(DISTINCT me.src_id)
            FROM memory_edges me
            WHERE me.rel_type = 'IN_EPISODE'
              AND me.dst_type = 'episode'
              AND me.dst_id = e.id::text
        ), '[]'::jsonb)
    ) ORDER BY e.started_at, e.id), '[]'::jsonb)
    FROM episodes e;
$$ LANGUAGE sql STABLE;

-- Graph edges from the primary flat substrate, plus canonical SUPERSEDES
-- edges derived from the legacy memories.superseded_by column.
CREATE OR REPLACE FUNCTION hmx_export_relationships() RETURNS JSONB AS $$
    SELECT COALESCE(jsonb_agg(edge), '[]'::jsonb)
    FROM (
        SELECT jsonb_build_object(
            'source_id', me.src_id,
            'source_type', me.src_type,
            'target_id', me.dst_id,
            'target_type', me.dst_type,
            'edge_type', me.rel_type,
            'properties', COALESCE(me.properties, '{}'::jsonb) || jsonb_build_object(
                'weight', me.weight,
                'kind', me.kind,
                'source', me.source,
                'created_at', me.created_at
            )
        ) AS edge
        FROM memory_edges me
        UNION ALL
        SELECT jsonb_build_object(
            'source_id', m.id::text,
            'source_type', 'memory',
            'target_id', m.superseded_by::text,
            'target_type', 'memory',
            'edge_type', 'SUPERSEDES',
            'properties', jsonb_build_object('created_at', m.updated_at, 'reason', 'superseded_by_column')
        )
        FROM memories m
        WHERE m.superseded_by IS NOT NULL
    ) edges;
$$ LANGUAGE sql STABLE;

-- Worldview (protected): beliefs/boundaries/values with graph evidence.
CREATE OR REPLACE FUNCTION hmx_export_worldview() RETURNS JSONB AS $$
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', m.id,
        'category', COALESCE(m.metadata->>'category', 'belief'),
        'content', m.content,
        'confidence', COALESCE((m.metadata->>'confidence')::float, 0.8),
        'stability', COALESCE((m.metadata->>'stability')::float, 0.7),
        'importance', m.importance,
        'status', m.status,
        'supporting_ids', COALESCE((
            SELECT jsonb_agg(DISTINCT me.src_id)
            FROM memory_edges me
            WHERE me.rel_type IN ('SUPPORTS', 'EVIDENCE_FOR')
              AND me.dst_type = 'memory' AND me.dst_id = m.id::text
        ), '[]'::jsonb),
        'contesting_ids', COALESCE((
            SELECT jsonb_agg(DISTINCT me.src_id)
            FROM memory_edges me
            WHERE me.rel_type IN ('CONTRADICTS', 'CONTESTED_BECAUSE')
              AND me.dst_type = 'memory' AND me.dst_id = m.id::text
        ), '[]'::jsonb),
        'metadata', m.metadata
    ) ORDER BY m.created_at, m.id), '[]'::jsonb)
    FROM memories m
    WHERE m.type = 'worldview' AND m.status IN ('active', 'archived');
$$ LANGUAGE sql STABLE;

-- Goals (protected): goal memories carry their structure in metadata.
CREATE OR REPLACE FUNCTION hmx_export_goals() RETURNS JSONB AS $$
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', m.id,
        'title', COALESCE(m.metadata->>'title', m.content),
        'description', m.metadata->>'description',
        'priority', COALESCE(m.metadata->>'priority', 'queued'),
        'source', COALESCE(m.metadata->>'source', 'curiosity'),
        'due_at', m.metadata->'due_at',
        'progress', COALESCE(m.metadata->'progress', '[]'::jsonb),
        'blocked_by', m.metadata->'blocked_by',
        'parent_goal_id', m.metadata->>'parent_goal_id',
        'status', m.status,
        'metadata', m.metadata
    ) ORDER BY m.created_at, m.id), '[]'::jsonb)
    FROM memories m
    WHERE m.type = 'goal' AND m.status IN ('active', 'archived');
$$ LANGUAGE sql STABLE;

-- Drives (protected): live motivational dynamics.
CREATE OR REPLACE FUNCTION hmx_export_drives() RETURNS JSONB AS $$
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'name', d.name,
        'description', d.description,
        'current_level', d.current_level,
        'baseline', d.baseline,
        'accumulation_rate', d.accumulation_rate,
        'decay_rate', d.decay_rate,
        'satisfaction_cooldown', d.satisfaction_cooldown::text,
        'last_satisfied', d.last_satisfied,
        'urgency_threshold', d.urgency_threshold,
        'metadata', d.metadata
    ) ORDER BY d.name), '[]'::jsonb)
    FROM drives d;
$$ LANGUAGE sql STABLE;

-- Emotional triggers (protected). Embedding omitted by design.
CREATE OR REPLACE FUNCTION hmx_export_emotional_triggers() RETURNS JSONB AS $$
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', t.id,
        'trigger_pattern', t.trigger_pattern,
        'valence_delta', t.valence_delta,
        'arousal_delta', t.arousal_delta,
        'dominance_delta', t.dominance_delta,
        'typical_emotion', t.typical_emotion,
        'confidence', t.confidence,
        'times_activated', t.times_activated,
        'origin', t.origin,
        'source_memory_ids', COALESCE(to_jsonb(t.source_memory_ids), '[]'::jsonb),
        'metadata', t.metadata
    ) ORDER BY t.trigger_pattern, t.id), '[]'::jsonb)
    FROM emotional_triggers t;
$$ LANGUAGE sql STABLE;

-- Clusters without centroid embeddings (recomputed on import).
CREATE OR REPLACE FUNCTION hmx_export_clusters() RETURNS JSONB AS $$
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', c.id,
        'cluster_type', c.cluster_type,
        'name', c.name,
        'member_ids', COALESCE((
            SELECT jsonb_agg(gcm.memory_id)
            FROM get_cluster_members_graph(c.id) gcm
        ), '[]'::jsonb)
    ) ORDER BY c.name, c.id), '[]'::jsonb)
    FROM clusters c;
$$ LANGUAGE sql STABLE;

-- Narrative scaffolding (protected): AGE vertex properties, exported
-- explicitly per subsection.
CREATE OR REPLACE FUNCTION hmx_export_narrative() RETURNS JSONB AS $$
DECLARE
    chapters JSONB;
    turning_points JSONB;
    threads JSONB;
    conflicts JSONB;
BEGIN
    SELECT COALESCE(jsonb_agg(
        (props::text::jsonb - 'hmx_payload')
        || COALESCE(NULLIF(props::text::jsonb->>'hmx_payload', '')::jsonb, '{}'::jsonb)
        || jsonb_build_object('id', replace(node_id::text, '"', ''))
    ), '[]'::jsonb) INTO chapters
    FROM ag_catalog.cypher('memory_graph', $q$
        MATCH (n:LifeChapterNode) RETURN id(n), properties(n)
    $q$) AS (node_id ag_catalog.agtype, props ag_catalog.agtype);

    SELECT COALESCE(jsonb_agg(
        (props::text::jsonb - 'hmx_payload')
        || COALESCE(NULLIF(props::text::jsonb->>'hmx_payload', '')::jsonb, '{}'::jsonb)
        || jsonb_build_object('id', replace(node_id::text, '"', ''))
    ), '[]'::jsonb) INTO turning_points
    FROM ag_catalog.cypher('memory_graph', $q$
        MATCH (n:TurningPointNode) RETURN id(n), properties(n)
    $q$) AS (node_id ag_catalog.agtype, props ag_catalog.agtype);

    SELECT COALESCE(jsonb_agg(
        (props::text::jsonb - 'hmx_payload')
        || COALESCE(NULLIF(props::text::jsonb->>'hmx_payload', '')::jsonb, '{}'::jsonb)
        || jsonb_build_object('id', replace(node_id::text, '"', ''))
    ), '[]'::jsonb) INTO threads
    FROM ag_catalog.cypher('memory_graph', $q$
        MATCH (n:NarrativeThreadNode) RETURN id(n), properties(n)
    $q$) AS (node_id ag_catalog.agtype, props ag_catalog.agtype);

    SELECT COALESCE(jsonb_agg(
        (props::text::jsonb - 'hmx_payload')
        || COALESCE(NULLIF(props::text::jsonb->>'hmx_payload', '')::jsonb, '{}'::jsonb)
        || jsonb_build_object('id', replace(node_id::text, '"', ''))
    ), '[]'::jsonb) INTO conflicts
    FROM ag_catalog.cypher('memory_graph', $q$
        MATCH (n:ValueConflictNode) RETURN id(n), properties(n)
    $q$) AS (node_id ag_catalog.agtype, props ag_catalog.agtype);

    RETURN jsonb_build_object(
        'life_chapters', chapters,
        'turning_points', turning_points,
        'narrative_threads', threads,
        'value_conflicts', conflicts
    );
END;
$$ LANGUAGE plpgsql STABLE;

-- Identity (protected): the self-model facets plus the initialized profile.
CREATE OR REPLACE FUNCTION hmx_export_identity() RETURNS JSONB AS $$
DECLARE
    profile JSONB;
    facets JSONB;
BEGIN
    SELECT value->'agent' INTO profile FROM config WHERE key = 'agent.init_profile';
    facets := get_self_model_context(200);
    RETURN jsonb_build_array(jsonb_build_object(
        'key', 'core_identity',
        'content', COALESCE(profile->>'description', ''),
        'profile', COALESCE(profile, '{}'::jsonb),
        'facets', COALESCE(facets, '[]'::jsonb),
        'metadata', jsonb_build_object(
            'provenance', COALESCE(
                (SELECT value FROM config WHERE key = 'agent.hmx_identity_provenance'),
                jsonb_build_object('acquisition_mode', 'experienced')
            )
        )
    ));
END;
$$ LANGUAGE plpgsql STABLE;

-- In-flight work (port/duplicate): memories-in-becoming.
CREATE OR REPLACE FUNCTION hmx_export_in_flight_work() RETURNS JSONB AS $$
    SELECT jsonb_build_object(
        'consolidation_tasks', COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
                'id', t.id,
                'task_type', t.task_type,
                'status', t.status,
                'created_at', t.created_at,
                'updated_at', t.updated_at,
                'input_ids', COALESCE(to_jsonb(t.source_unit_ids), '[]'::jsonb),
                'target_memory_id', t.target_memory_id,
                'attempt_count', t.attempts,
                'properties', COALESCE(t.task_payload, '{}'::jsonb)
            ) ORDER BY t.created_at, t.id)
            FROM recmem_consolidation_tasks t
            WHERE t.status IN ('pending', 'in_progress', 'failed')
        ), '[]'::jsonb),
        'reconsolidation_tasks', COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
                'id', r.id,
                'status', r.status,
                'memory_ids', CASE WHEN r.belief_id IS NULL THEN '[]'::jsonb
                                   ELSE jsonb_build_array(r.belief_id) END,
                'reason', r.transformation_type,
                'created_at', r.created_at,
                'properties', jsonb_build_object(
                    'old_content', r.old_content,
                    'new_content', r.new_content
                )
            ) ORDER BY r.created_at, r.id)
            FROM reconsolidation_tasks r
            WHERE r.status IN ('pending', 'in_progress', 'failed')
        ), '[]'::jsonb)
    );
$$ LANGUAGE sql STABLE;

-- Baseline placeholder. db/53_hmx_protected_replacement.sql replaces this
-- function after creating the immutable protected-replacement audit tables.
CREATE OR REPLACE FUNCTION hmx_export_audit_records() RETURNS JSONB AS $$
    SELECT jsonb_build_object(
        'protected_replacement_audit', '[]'::jsonb,
        'protected_section_verified_audit', '[]'::jsonb,
        'protected_replacement_reversion_audit', '[]'::jsonb,
        'transformation_history', '[]'::jsonb
    );
$$ LANGUAGE sql STABLE;

-- Raw conversation units are explicitly opt-in because they can contain the
-- most sensitive verbatim user content. Embeddings are never exported.
CREATE OR REPLACE FUNCTION hmx_export_raw_units(
    -- Sensitivity egress control (#92) — same opt-in as hmx_export_memories.
    p_include_sensitive BOOLEAN DEFAULT FALSE
) RETURNS JSONB AS $$
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'id', u.id,
        'user_text', u.user_text,
        'assistant_text', u.assistant_text,
        'turn_at', u.turn_at,
        'importance', u.importance,
        'route_status', u.route_status,
        'source_identity', u.source_identity,
        'idempotency_key', u.idempotency_key,
        'derived_memory_ids', COALESCE((
            SELECT jsonb_agg(msu.memory_id ORDER BY msu.memory_id)
            FROM memory_source_units msu
            WHERE msu.subconscious_unit_id = u.id
        ), '[]'::jsonb)
    ) ORDER BY u.turn_at, u.id), '[]'::jsonb)
    FROM subconscious_units u
    WHERE u.status <> 'redacted'
      AND (p_include_sensitive
           OR COALESCE(u.source_attribution->>'sensitivity', '') <> 'private');
$$ LANGUAGE sql STABLE;

-- Configuration is also opt-in. Verification material and credentials stay
-- local; filtering uses the same patterns declared in the privacy envelope.
CREATE OR REPLACE FUNCTION hmx_export_config() RETURNS JSONB AS $$
    SELECT COALESCE(jsonb_object_agg(c.key, c.value ORDER BY c.key), '{}'::jsonb)
    FROM config c
    WHERE lower(c.key) !~ '(key|secret|token|password|signature|credential|auth|trust|anchor|certificate)';
$$ LANGUAGE sql STABLE;

-- -------------------------------------------------------------------------
-- Slice 2: additive import primitives
-- -------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION hexis_instance_is_empty() RETURNS JSONB AS $$
DECLARE
    blockers JSONB := '[]'::jsonb;
    details JSONB;
    row_count BIGINT;
    graph_count BIGINT := 0;
BEGIN
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'kind', 'protected_memory',
        'id', m.id,
        'type', m.type,
        'acquisition_mode', COALESCE(m.metadata#>>'{provenance,acquisition_mode}', 'missing')
    )), '[]'::jsonb)
    INTO details
    FROM memories m
    WHERE m.type IN ('worldview', 'goal')
      AND COALESCE(m.metadata#>>'{provenance,acquisition_mode}', 'missing') <> 'bootstrap';
    blockers := blockers || details;

    SELECT count(*) INTO row_count
    FROM emotional_triggers
    WHERE COALESCE(metadata#>>'{provenance,acquisition_mode}', 'missing') <> 'bootstrap';
    IF row_count > 0 THEN
        blockers := blockers || jsonb_build_array(jsonb_build_object(
            'kind', 'emotional_triggers', 'count', row_count,
            'reason', 'emotional trigger provenance is not bootstrap'
        ));
    END IF;

    SELECT count(*) INTO row_count
    FROM drives
    WHERE COALESCE(metadata#>>'{provenance,acquisition_mode}', 'missing') <> 'bootstrap';
    IF row_count > 0 THEN
        blockers := blockers || jsonb_build_array(jsonb_build_object(
            'kind', 'experienced_drive_state', 'count', row_count
        ));
    END IF;

    BEGIN
        SELECT replace(n::text, '"', '')::bigint INTO graph_count
        FROM ag_catalog.cypher('memory_graph', $q$
            MATCH (n)
            WHERE n:SelfNode OR n:LifeChapterNode OR n:TurningPointNode
               OR n:NarrativeThreadNode OR n:ValueConflictNode
            RETURN count(n)
        $q$) AS (n ag_catalog.agtype);
    EXCEPTION WHEN OTHERS THEN
        graph_count := 0;
    END;
    IF graph_count > 0 THEN
        blockers := blockers || jsonb_build_array(jsonb_build_object(
            'kind', 'identity_or_narrative_graph', 'count', graph_count
        ));
    END IF;

    IF to_regclass('public.protected_replacement_audit') IS NOT NULL THEN
        EXECUTE $query$
            SELECT COALESCE(jsonb_agg(jsonb_build_object(
                'kind', 'protected_audit',
                'event_type', event_type,
                'foreign_diagnostic', is_foreign_diagnostic,
                'count', count
            ) ORDER BY event_type, is_foreign_diagnostic), '[]'::jsonb)
            FROM (
                SELECT event_type, is_foreign_diagnostic, count(*) AS count
                FROM protected_replacement_audit
                GROUP BY event_type, is_foreign_diagnostic
            ) audit_counts
        $query$ INTO details;
        blockers := blockers || details;
    END IF;

    IF to_regclass('public.hmx_consent') IS NOT NULL THEN
        EXECUTE 'SELECT count(*) FROM hmx_consent' INTO row_count;
        IF row_count > 0 THEN
            blockers := blockers || jsonb_build_array(jsonb_build_object(
                'kind', 'protected_consent', 'count', row_count
            ));
        END IF;
    END IF;

    RETURN jsonb_build_object(
        'is_empty', jsonb_array_length(blockers) = 0,
        'state', CASE WHEN jsonb_array_length(blockers) = 0 THEN 'empty' ELSE 'active' END,
        'blockers', blockers
    );
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION hmx_import_memories(p_records JSONB) RETURNS JSONB AS $$
DECLARE
    item JSONB;
    local_id UUID;
    source_ref TEXT;
    metadata JSONB;
    ref_map JSONB := '{}'::jsonb;
    duplicate_refs JSONB := '[]'::jsonb;
    errors JSONB := '[]'::jsonb;
    inserted_count INT := 0;
BEGIN
    FOR item IN SELECT value FROM jsonb_array_elements(COALESCE(p_records, '[]'::jsonb))
    LOOP
        BEGIN
            source_ref := item->>'ref';
            IF NULLIF(source_ref, '') IS NULL OR NULLIF(item->>'content', '') IS NULL THEN
                RAISE EXCEPTION 'memory ref and content are required';
            END IF;

            SELECT m.id INTO local_id
            FROM memories m
            WHERE regexp_replace(lower(btrim(m.content)), '\s+', ' ', 'g') =
                  item->>'_transient_normalized_content'
            ORDER BY m.created_at, m.id
            LIMIT 1;

            IF local_id IS NOT NULL THEN
                ref_map := ref_map || jsonb_build_object(source_ref, local_id::text);
                duplicate_refs := duplicate_refs || jsonb_build_array(source_ref);
                CONTINUE;
            END IF;

            metadata := COALESCE(item->'metadata', '{}'::jsonb)
                || jsonb_build_object(
                    'embedding_status', 'pending_import',
                    'hmx', COALESCE(item->'metadata'->'hmx', '{}'::jsonb)
                        || jsonb_build_object('content_hash_v1', item->>'content_hash_v1')
                );
            metadata := jsonb_set(
                metadata,
                '{provenance}',
                COALESCE(item->'provenance', '{}'::jsonb),
                true
            );

            INSERT INTO memories (
                type, status, content, embedding, importance, trust_level,
                decay_rate, created_at, updated_at, valid_from, valid_until,
                access_count, last_accessed, source_attribution, metadata
            ) VALUES (
                (item->>'type')::memory_type,
                CASE WHEN item->>'status' IN ('active','archived','invalidated','staged')
                     THEN (item->>'status')::memory_status ELSE 'active'::memory_status END,
                item->>'content',
                array_fill(0.0::float, ARRAY[embedding_dimension()])::vector,
                COALESCE((item->>'importance')::float, 0.5),
                LEAST(1.0, GREATEST(0.0, COALESCE((item->>'trust_level')::float, 0.5))),
                COALESCE((item->>'decay_rate')::float, 0.01),
                COALESCE((item->>'created_at')::timestamptz, CURRENT_TIMESTAMP),
                COALESCE((item->>'updated_at')::timestamptz,
                         (item->>'created_at')::timestamptz, CURRENT_TIMESTAMP),
                NULLIF(item->>'valid_from', '')::timestamptz,
                NULLIF(item->>'valid_until', '')::timestamptz,
                GREATEST(COALESCE((item->>'access_count')::int, 0), 0),
                NULLIF(item->>'last_accessed', '')::timestamptz,
                COALESCE(item->'source_attribution', '{}'::jsonb),
                metadata
            ) RETURNING id INTO local_id;

            PERFORM sync_memory_node(local_id);
            INSERT INTO memory_neighborhoods (memory_id, is_stale)
            VALUES (local_id, TRUE)
            ON CONFLICT (memory_id) DO UPDATE SET is_stale = TRUE;

            ref_map := ref_map || jsonb_build_object(source_ref, local_id::text);
            inserted_count := inserted_count + 1;
        EXCEPTION WHEN OTHERS THEN
            errors := errors || jsonb_build_array(jsonb_build_object(
                'ref', source_ref, 'error', SQLERRM
            ));
        END;
    END LOOP;

    RETURN jsonb_build_object(
        'ref_map', ref_map,
        'inserted', inserted_count,
        'duplicate_refs', duplicate_refs,
        'errors', errors
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION hmx_import_episodes(
    p_records JSONB,
    p_ref_map JSONB
) RETURNS JSONB AS $$
DECLARE
    item JSONB;
    memory_ref JSONB;
    episode_id UUID;
    memory_id UUID;
    source_ref TEXT;
    ref_map JSONB := COALESCE(p_ref_map, '{}'::jsonb);
    warnings JSONB := '[]'::jsonb;
    inserted_count INT := 0;
    sequence_order INT;
BEGIN
    FOR item IN SELECT value FROM jsonb_array_elements(COALESCE(p_records, '[]'::jsonb))
    LOOP
        source_ref := item->>'ref';
        INSERT INTO episodes (started_at, ended_at, summary, metadata)
        VALUES (
            COALESCE((item->>'started_at')::timestamptz, CURRENT_TIMESTAMP),
            NULLIF(item->>'ended_at', '')::timestamptz,
            item->>'summary',
            COALESCE(item->'metadata', '{}'::jsonb)
        ) RETURNING id INTO episode_id;
        PERFORM sync_episode_node(episode_id);
        ref_map := ref_map || jsonb_build_object(source_ref, episode_id::text);
        inserted_count := inserted_count + 1;
        sequence_order := 0;

        FOR memory_ref IN SELECT value FROM jsonb_array_elements(COALESCE(item->'memory_refs', '[]'::jsonb))
        LOOP
            sequence_order := sequence_order + 1;
            BEGIN
                memory_id := (ref_map->>trim(both '"' from memory_ref::text))::uuid;
            EXCEPTION WHEN OTHERS THEN
                memory_id := NULL;
            END;
            IF memory_id IS NULL THEN
                warnings := warnings || jsonb_build_array(jsonb_build_object(
                    'code', 'orphaned_reference',
                    'section', 'episodes',
                    'ref', trim(both '"' from memory_ref::text)
                ));
            ELSE
                PERFORM link_memory_to_episode_graph(memory_id, episode_id, sequence_order);
            END IF;
        END LOOP;
    END LOOP;

    RETURN jsonb_build_object(
        'ref_map', ref_map, 'inserted', inserted_count, 'warnings', warnings
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION hmx_remap_goal_references(
    p_records JSONB,
    p_ref_map JSONB
) RETURNS JSONB AS $$
DECLARE
    item JSONB;
    blocked_ref JSONB;
    local_id UUID;
    parent_id UUID;
    blocked_ids JSONB;
    warnings JSONB := '[]'::jsonb;
BEGIN
    FOR item IN SELECT value FROM jsonb_array_elements(COALESCE(p_records, '[]'::jsonb))
    LOOP
        BEGIN
            local_id := (p_ref_map->>(item->>'ref'))::uuid;
        EXCEPTION WHEN OTHERS THEN
            local_id := NULL;
        END;
        IF local_id IS NULL THEN
            CONTINUE;
        END IF;

        BEGIN
            parent_id := (p_ref_map->>(item->>'parent_ref'))::uuid;
        EXCEPTION WHEN OTHERS THEN
            parent_id := NULL;
        END;
        IF item->>'parent_ref' IS NOT NULL AND parent_id IS NULL THEN
            warnings := warnings || jsonb_build_array(jsonb_build_object(
                'code', 'orphaned_reference', 'section', 'goals',
                'ref', item->>'parent_ref'
            ));
        END IF;

        blocked_ids := '[]'::jsonb;
        FOR blocked_ref IN SELECT value FROM jsonb_array_elements(COALESCE(item->'blocked_by', '[]'::jsonb))
        LOOP
            IF p_ref_map ? trim(both '"' from blocked_ref::text) THEN
                blocked_ids := blocked_ids || jsonb_build_array(
                    p_ref_map->>trim(both '"' from blocked_ref::text)
                );
            ELSE
                warnings := warnings || jsonb_build_array(jsonb_build_object(
                    'code', 'orphaned_reference', 'section', 'goals',
                    'ref', trim(both '"' from blocked_ref::text)
                ));
            END IF;
        END LOOP;

        UPDATE memories
        SET metadata = metadata
            || jsonb_build_object('parent_goal_id', parent_id, 'blocked_by', blocked_ids)
            - 'parent_ref'
        WHERE id = local_id AND type = 'goal';
    END LOOP;
    RETURN jsonb_build_object('warnings', warnings);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION hmx_import_clusters(
    p_records JSONB,
    p_ref_map JSONB
) RETURNS JSONB AS $$
DECLARE
    item JSONB;
    member_ref JSONB;
    cluster_id UUID;
    memory_id UUID;
    source_ref TEXT;
    ref_map JSONB := COALESCE(p_ref_map, '{}'::jsonb);
    warnings JSONB := '[]'::jsonb;
    inserted_count INT := 0;
BEGIN
    FOR item IN SELECT value FROM jsonb_array_elements(COALESCE(p_records, '[]'::jsonb))
    LOOP
        source_ref := item->>'ref';
        INSERT INTO clusters (cluster_type, name, centroid_embedding)
        VALUES (
            COALESCE(NULLIF(item->>'cluster_type', ''), 'mixed')::cluster_type,
            COALESCE(NULLIF(item->>'name', ''), 'Imported cluster'),
            NULL
        ) RETURNING id INTO cluster_id;
        PERFORM sync_cluster_node(cluster_id);
        ref_map := ref_map || jsonb_build_object(source_ref, cluster_id::text);
        inserted_count := inserted_count + 1;

        FOR member_ref IN SELECT value FROM jsonb_array_elements(COALESCE(item->'member_refs', '[]'::jsonb))
        LOOP
            BEGIN
                memory_id := (ref_map->>trim(both '"' from member_ref::text))::uuid;
            EXCEPTION WHEN OTHERS THEN
                memory_id := NULL;
            END;
            IF memory_id IS NULL THEN
                warnings := warnings || jsonb_build_array(jsonb_build_object(
                    'code', 'orphaned_reference',
                    'section', 'clusters',
                    'ref', trim(both '"' from member_ref::text)
                ));
            ELSE
                PERFORM link_memory_to_cluster_graph(memory_id, cluster_id, 1.0);
            END IF;
        END LOOP;
    END LOOP;

    RETURN jsonb_build_object(
        'ref_map', ref_map, 'inserted', inserted_count, 'warnings', warnings
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION hmx_import_relationships(
    p_records JSONB,
    p_ref_map JSONB
) RETURNS JSONB AS $$
DECLARE
    item JSONB;
    source_id TEXT;
    target_id TEXT;
    source_type TEXT;
    target_type TEXT;
    edge_type TEXT;
    props JSONB;
    inserted_count INT := 0;
    duplicate_count INT := 0;
    affected_count INT;
    warnings JSONB := '[]'::jsonb;
BEGIN
    FOR item IN SELECT value FROM jsonb_array_elements(COALESCE(p_records, '[]'::jsonb))
    LOOP
        source_id := p_ref_map->>(item->>'source_ref');
        target_id := p_ref_map->>(item->>'target_ref');
        IF source_id IS NULL OR target_id IS NULL THEN
            warnings := warnings || jsonb_build_array(jsonb_build_object(
                'code', 'orphaned_reference',
                'section', 'relationships',
                'source_ref', item->>'source_ref',
                'target_ref', item->>'target_ref'
            ));
            CONTINUE;
        END IF;

        props := COALESCE(item->'properties', '{}'::jsonb);
        source_type := COALESCE(NULLIF(props->>'source_type', ''), 'memory');
        target_type := COALESCE(NULLIF(props->>'target_type', ''), 'memory');
        edge_type := item->>'edge_type';

        INSERT INTO memory_edges (
            src_type, src_id, rel_type, dst_type, dst_id,
            weight, kind, source, properties
        ) VALUES (
            source_type, source_id, edge_type, target_type, target_id,
            COALESCE((props->>'weight')::float, (props->>'strength')::float, 1.0),
            props->>'kind', props->>'source', props
        ) ON CONFLICT (src_type, src_id, rel_type, dst_type, dst_id) DO NOTHING;
        GET DIAGNOSTICS affected_count = ROW_COUNT;
        IF affected_count = 1 THEN
            inserted_count := inserted_count + 1;
        ELSE
            duplicate_count := duplicate_count + 1;
        END IF;

        IF NOT EXISTS (
            SELECT 1 FROM pg_enum e JOIN pg_type t ON t.oid = e.enumtypid
            WHERE t.typname = 'graph_edge_type' AND e.enumlabel = edge_type
        ) THEN
            warnings := warnings || jsonb_build_array(jsonb_build_object(
                'code', 'unknown_edge_type', 'edge_type', edge_type
            ));
        END IF;
    END LOOP;

    RETURN jsonb_build_object(
        'inserted', inserted_count,
        'duplicates', duplicate_count,
        'warnings', warnings
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION hmx_mark_drive_experienced() RETURNS TRIGGER AS $$
BEGIN
    IF OLD.metadata#>>'{provenance,acquisition_mode}' = 'bootstrap'
       AND NEW.metadata = OLD.metadata
       AND (NEW.current_level IS DISTINCT FROM OLD.current_level
            OR NEW.baseline IS DISTINCT FROM OLD.baseline
            OR NEW.last_satisfied IS DISTINCT FROM OLD.last_satisfied) THEN
        NEW.metadata := jsonb_set(
            jsonb_set(NEW.metadata, '{provenance,acquisition_mode}', '"experienced"'::jsonb, true),
            '{replaceable_during_bootstrap}', 'false'::jsonb, true
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION hmx_default_emotional_trigger_provenance() RETURNS TRIGGER AS $$
BEGIN
    IF NEW.metadata->'provenance' IS NULL THEN
        NEW.metadata := jsonb_set(
            COALESCE(NEW.metadata, '{}'::jsonb),
            '{provenance}',
            jsonb_build_object('acquisition_mode', 'experienced'),
            true
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION hmx_import_drives(p_records JSONB) RETURNS JSONB AS $$
DECLARE
    item JSONB;
    inserted_count INT := 0;
BEGIN
    FOR item IN SELECT value FROM jsonb_array_elements(COALESCE(p_records, '[]'::jsonb))
    LOOP
        INSERT INTO drives (
            name, description, current_level, baseline, accumulation_rate,
            decay_rate, satisfaction_cooldown, last_satisfied,
            urgency_threshold, metadata
        ) VALUES (
            item->>'name', item->>'description',
            COALESCE((item->>'current_level')::float, 0.5),
            COALESCE((item->>'baseline')::float, 0.5),
            COALESCE((item->>'accumulation_rate')::float, 0.01),
            COALESCE((item->>'decay_rate')::float, 0.05),
            COALESCE(NULLIF(item->>'satisfaction_cooldown', '')::interval, '1 hour'::interval),
            NULLIF(item->>'last_satisfied', '')::timestamptz,
            COALESCE((item->>'urgency_threshold')::float, 0.8),
            jsonb_set(
                COALESCE(item->'metadata', '{}'::jsonb),
                '{provenance}', COALESCE(item->'provenance', '{}'::jsonb), true
            )
        ) ON CONFLICT (name) DO UPDATE SET
            description = EXCLUDED.description,
            current_level = EXCLUDED.current_level,
            baseline = EXCLUDED.baseline,
            accumulation_rate = EXCLUDED.accumulation_rate,
            decay_rate = EXCLUDED.decay_rate,
            satisfaction_cooldown = EXCLUDED.satisfaction_cooldown,
            last_satisfied = EXCLUDED.last_satisfied,
            urgency_threshold = EXCLUDED.urgency_threshold,
            metadata = EXCLUDED.metadata;
        inserted_count := inserted_count + 1;
    END LOOP;
    RETURN jsonb_build_object('imported', inserted_count, 'warnings', '[]'::jsonb);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION hmx_import_emotional_triggers(
    p_records JSONB,
    p_ref_map JSONB
) RETURNS JSONB AS $$
DECLARE
    item JSONB;
    source_ref JSONB;
    source_ids UUID[];
    mapped_id UUID;
    inserted_count INT := 0;
    duplicate_count INT := 0;
    affected_count INT;
    warnings JSONB := '[]'::jsonb;
BEGIN
    FOR item IN SELECT value FROM jsonb_array_elements(COALESCE(p_records, '[]'::jsonb))
    LOOP
        source_ids := '{}'::uuid[];
        FOR source_ref IN SELECT value FROM jsonb_array_elements(COALESCE(item->'source_memory_refs', '[]'::jsonb))
        LOOP
            BEGIN
                mapped_id := (p_ref_map->>trim(both '"' from source_ref::text))::uuid;
            EXCEPTION WHEN OTHERS THEN
                mapped_id := NULL;
            END;
            IF mapped_id IS NULL THEN
                warnings := warnings || jsonb_build_array(jsonb_build_object(
                    'code', 'orphaned_reference', 'section', 'emotional_triggers',
                    'ref', trim(both '"' from source_ref::text)
                ));
            ELSE
                source_ids := array_append(source_ids, mapped_id);
            END IF;
        END LOOP;

        IF EXISTS (
            SELECT 1 FROM emotional_triggers t
            WHERE regexp_replace(lower(btrim(t.trigger_pattern)), '\s+', ' ', 'g') =
                  item->>'_transient_normalized_content'
        ) THEN
            duplicate_count := duplicate_count + 1;
            CONTINUE;
        END IF;

        INSERT INTO emotional_triggers (
            trigger_pattern, trigger_embedding, valence_delta, arousal_delta,
            dominance_delta, typical_emotion, times_activated, confidence,
            origin, source_memory_ids, metadata
        ) VALUES (
            item->>'trigger_pattern',
            array_fill(0.0::float, ARRAY[embedding_dimension()])::vector,
            COALESCE((item->>'valence_delta')::float, 0.0),
            COALESCE((item->>'arousal_delta')::float, 0.0),
            COALESCE((item->>'dominance_delta')::float, 0.0),
            item->>'typical_emotion',
            GREATEST(COALESCE((item->>'times_activated')::int, 0), 0),
            LEAST(1.0, GREATEST(0.0, COALESCE((item->>'confidence')::float, 0.5))),
            COALESCE(NULLIF(item->>'origin', ''), 'imported'),
            source_ids,
            jsonb_set(
                COALESCE(item->'metadata', '{}'::jsonb),
                '{provenance}', COALESCE(item->'provenance', '{}'::jsonb), true
            )
        );
        GET DIAGNOSTICS affected_count = ROW_COUNT;
        inserted_count := inserted_count + affected_count;
    END LOOP;
    RETURN jsonb_build_object(
        'inserted', inserted_count, 'duplicates', duplicate_count, 'warnings', warnings
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION hmx_import_identity(
    p_records JSONB,
    p_ref_map JSONB
) RETURNS JSONB AS $$
DECLARE
    item JSONB;
    facet JSONB;
    facet_kind TEXT;
    evidence_id UUID;
    imported_count INT := 0;
BEGIN
    FOR item IN SELECT value FROM jsonb_array_elements(COALESCE(p_records, '[]'::jsonb))
    LOOP
        PERFORM set_config(
            'agent.init_profile',
            jsonb_build_object(
                'agent', COALESCE(item->'profile', jsonb_build_object('description', item->>'content'))
            )
        );
        PERFORM set_config(
            'agent.hmx_identity_provenance', COALESCE(item->'provenance', '{}'::jsonb)
        );
        PERFORM ensure_self_node();
        FOR facet IN SELECT value FROM jsonb_array_elements(COALESCE(item->'facets', '[]'::jsonb))
        LOOP
            facet_kind := COALESCE(
                NULLIF(facet->>'kind', ''),
                NULLIF(facet->>'type', ''),
                'identity'
            );
            BEGIN
                evidence_id := (p_ref_map->>(facet->>'evidence_memory_ref'))::uuid;
            EXCEPTION WHEN OTHERS THEN
                evidence_id := NULL;
            END;
            IF facet_kind = 'life_chapter_current' THEN
                IF NOT hmx_link_current_chapter_identity(
                    COALESCE((facet->>'strength')::float, 0.8)
                ) THEN
                    PERFORM upsert_self_concept_edge(
                        facet_kind,
                        facet->>'concept',
                        COALESCE((facet->>'strength')::float, 0.8),
                        evidence_id
                    );
                END IF;
            ELSE
                PERFORM upsert_self_concept_edge(
                    facet_kind,
                    facet->>'concept',
                    COALESCE((facet->>'strength')::float, 0.8),
                    evidence_id
                );
            END IF;
        END LOOP;
        imported_count := imported_count + 1;
    END LOOP;
    RETURN jsonb_build_object('imported', imported_count, 'warnings', '[]'::jsonb);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION hmx_import_narrative(
    p_data JSONB,
    p_ref_map JSONB
) RETURNS JSONB AS $$
DECLARE
    group_name TEXT;
    label_name TEXT;
    item JSONB;
    local_id UUID;
    ref_field TEXT;
    source_ref JSONB;
    mapped_refs JSONB;
    payload JSONB;
    ref_map JSONB := COALESCE(p_ref_map, '{}'::jsonb);
    imported_count INT := 0;
    warnings JSONB := '[]'::jsonb;
BEGIN
    -- Allocate every narrative ID first so cross-node references can resolve
    -- regardless of subsection or record order.
    FOREACH group_name IN ARRAY ARRAY[
        'life_chapters', 'turning_points', 'narrative_threads', 'value_conflicts'
    ] LOOP
        FOR item IN SELECT value FROM jsonb_array_elements(COALESCE(p_data->group_name, '[]'::jsonb))
        LOOP
            ref_map := ref_map || jsonb_build_object(item->>'ref', gen_random_uuid()::text);
        END LOOP;
    END LOOP;

    FOREACH group_name IN ARRAY ARRAY[
        'life_chapters', 'turning_points', 'narrative_threads', 'value_conflicts'
    ] LOOP
        label_name := CASE group_name
            WHEN 'life_chapters' THEN 'LifeChapterNode'
            WHEN 'turning_points' THEN 'TurningPointNode'
            WHEN 'narrative_threads' THEN 'NarrativeThreadNode'
            WHEN 'value_conflicts' THEN 'ValueConflictNode'
        END;
        FOR item IN SELECT value FROM jsonb_array_elements(COALESCE(p_data->group_name, '[]'::jsonb))
        LOOP
            local_id := (ref_map->>(item->>'ref'))::uuid;
            payload := item - 'ref';
            FOREACH ref_field IN ARRAY ARRAY[
                'memory_refs', 'chapter_refs', 'supporting_refs', 'contesting_refs'
            ] LOOP
                IF payload ? ref_field THEN
                    mapped_refs := '[]'::jsonb;
                    FOR source_ref IN SELECT value FROM jsonb_array_elements(COALESCE(payload->ref_field, '[]'::jsonb))
                    LOOP
                        IF ref_map ? trim(both '"' from source_ref::text) THEN
                            mapped_refs := mapped_refs || jsonb_build_array(
                                ref_map->>trim(both '"' from source_ref::text)
                            );
                        ELSE
                            warnings := warnings || jsonb_build_array(jsonb_build_object(
                                'code', 'orphaned_reference', 'section', 'narrative',
                                'ref', trim(both '"' from source_ref::text)
                            ));
                        END IF;
                    END LOOP;
                    payload := jsonb_set(payload, ARRAY[ref_field], mapped_refs, true);
                END IF;
            END LOOP;
            EXECUTE format(
                'SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
                    CREATE (n:%s {hmx_id: %L, hmx_payload: %L}) RETURN n
                $q$) AS (n ag_catalog.agtype)',
                label_name, local_id::text, payload::text
            );
            imported_count := imported_count + 1;
        END LOOP;
    END LOOP;
    RETURN jsonb_build_object(
        'ref_map', ref_map, 'imported', imported_count, 'warnings', warnings
    );
END;
$$ LANGUAGE plpgsql;

SET check_function_bodies = on;
