-- Hexis schema: subconscious observation functions.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

CREATE OR REPLACE FUNCTION apply_subconscious_observations(p_payload JSONB)
RETURNS JSONB AS $$
DECLARE
    obs JSONB;
    obs_type TEXT;
    conf FLOAT;
    evidence_raw JSONB;
    evidence_ids UUID[];
    evidence_id UUID;
    summary TEXT;
    suggested_name TEXT;
    rel_entity TEXT;
    rel_strength FLOAT;
    rel_change TEXT;
    rel_magnitude FLOAT;
    contra_a UUID;
    contra_b UUID;
    tension TEXT;
    worldview_id UUID;
    pattern TEXT;
    freq INT;
    unprocessed BOOLEAN;
    concept TEXT;
    cluster_id UUID;
    rationale TEXT;
    i INT;
    min_conf FLOAT := 0.6;
    person_belief_id UUID;
    person_belief_conf FLOAT;
    person_belief_content TEXT;
    applied_narrative INT := 0;
    applied_relationships INT := 0;
    applied_contradictions INT := 0;
    applied_emotional INT := 0;
    applied_consolidation INT := 0;
    emotional_items JSONB;
    consolidation_items JSONB;
BEGIN
    IF p_payload IS NULL OR jsonb_typeof(p_payload) <> 'object' THEN
        RETURN jsonb_build_object('applied', false, 'reason', 'empty');
    END IF;
    FOR obs IN SELECT * FROM jsonb_array_elements(COALESCE(p_payload->'narrative_observations', '[]'::jsonb))
    LOOP
        obs_type := lower(COALESCE(obs->>'type', obs->>'kind', ''));
        conf := NULLIF(obs->>'confidence', '')::float;
        IF conf IS NULL OR conf >= min_conf THEN
            evidence_raw := COALESCE(obs->'evidence', obs->'evidence_ids', obs->'memory_ids', '[]'::jsonb);
            BEGIN
                SELECT COALESCE(ARRAY(
                    SELECT value::uuid
                    FROM jsonb_array_elements_text(evidence_raw) val(value)
                    WHERE value ~* '^[0-9a-f-]{36}$'
                ), ARRAY[]::uuid[]) INTO evidence_ids;
            EXCEPTION
                WHEN OTHERS THEN
                    evidence_ids := ARRAY[]::uuid[];
            END;

            summary := COALESCE(obs->>'summary', obs->>'rationale', obs->>'evidence_summary', '');
            IF obs_type IN ('chapter_transition', 'chapter_start', 'chapter_begin', 'begin_chapter') THEN
                suggested_name := COALESCE(obs->>'suggested_name', obs->>'name', obs->>'chapter_name', 'Foundations');
                PERFORM ensure_current_life_chapter(suggested_name);
                PERFORM create_strategic_memory(
                    COALESCE(NULLIF(summary, ''), format('Chapter transition: %s', suggested_name)),
                    'Narrative transition',
                    COALESCE(conf, 0.8),
                    jsonb_build_object(
                        'kind', 'narrative',
                        'type', obs_type,
                        'evidence_memory_ids', evidence_ids,
                        'confidence', conf
                    ),
                    NULL,
                    0.6
                );
                applied_narrative := applied_narrative + 1;
            ELSIF obs_type IN ('turning_point', 'turning-point', 'turningpoint') THEN
                evidence_id := NULL;
                BEGIN
                    evidence_id := NULLIF(obs->>'memory_id', '')::uuid;
                EXCEPTION
                    WHEN OTHERS THEN
                        evidence_id := NULL;
                END;
                IF evidence_id IS NULL THEN
                    BEGIN
                        evidence_id := NULLIF(obs->>'evidence_memory_id', '')::uuid;
                    EXCEPTION
                        WHEN OTHERS THEN
                            evidence_id := NULL;
                    END;
                END IF;
                IF evidence_id IS NULL AND array_length(evidence_ids, 1) > 0 THEN
                    evidence_id := evidence_ids[1];
                END IF;

                IF evidence_id IS NOT NULL THEN
                    UPDATE memories
                    SET importance = GREATEST(importance, 0.9),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = evidence_id;
                END IF;

                PERFORM create_strategic_memory(
                    COALESCE(NULLIF(summary, ''), 'Turning point detected'),
                    'Narrative turning point',
                    COALESCE(conf, 0.85),
                    jsonb_build_object(
                        'kind', 'narrative',
                        'type', 'turning_point',
                        'memory_id', evidence_id,
                        'evidence_memory_ids', evidence_ids,
                        'confidence', conf
                    ),
                    NULL,
                    0.6
                );
                applied_narrative := applied_narrative + 1;
            ELSIF obs_type IN ('theme_emergence', 'theme', 'theme_emergent') THEN
                PERFORM create_strategic_memory(
                    COALESCE(
                        NULLIF(summary, ''),
                        COALESCE(obs->>'theme', obs->>'pattern', 'Theme emerging')
                    ),
                    'Theme emergence',
                    COALESCE(conf, 0.7),
                    jsonb_build_object(
                        'kind', 'narrative',
                        'type', obs_type,
                        'evidence_memory_ids', evidence_ids,
                        'confidence', conf
                    ),
                    NULL,
                    0.5
                );
                applied_narrative := applied_narrative + 1;
            END IF;
        END IF;
    END LOOP;
    FOR obs IN SELECT * FROM jsonb_array_elements(COALESCE(p_payload->'relationship_observations', '[]'::jsonb))
    LOOP
        rel_entity := btrim(COALESCE(obs->>'entity', obs->>'name', ''));
        IF rel_entity = '' THEN
            CONTINUE;
        END IF;

        conf := NULLIF(obs->>'confidence', '')::float;
        IF conf IS NOT NULL AND conf < min_conf THEN
            CONTINUE;
        END IF;

        rel_change := lower(COALESCE(obs->>'change_type', obs->>'type', ''));
        rel_magnitude := NULLIF(obs->>'magnitude', '')::float;
        rel_strength := NULLIF(obs->>'strength', '')::float;
        IF rel_strength IS NULL THEN
            rel_strength := NULLIF(obs->>'new_strength', '')::float;
        END IF;
        IF rel_strength IS NULL THEN
            IF rel_magnitude IS NOT NULL THEN
                IF rel_change LIKE '%decrease%' OR rel_change LIKE '%lower%' THEN
                    rel_strength := 0.6 - rel_magnitude;
                ELSIF rel_change LIKE '%increase%' OR rel_change LIKE '%higher%' THEN
                    rel_strength := 0.6 + rel_magnitude;
                ELSE
                    rel_strength := 0.6 + rel_magnitude * 0.5;
                END IF;
            ELSE
                rel_strength := 0.6;
            END IF;
        END IF;
        rel_strength := LEAST(1.0, GREATEST(0.0, rel_strength));

        evidence_raw := COALESCE(obs->'evidence', obs->'evidence_ids', obs->'memory_ids', '[]'::jsonb);
        BEGIN
            SELECT COALESCE(ARRAY(
                SELECT value::uuid
                FROM jsonb_array_elements_text(evidence_raw) val(value)
                WHERE value ~* '^[0-9a-f-]{36}$'
            ), ARRAY[]::uuid[]) INTO evidence_ids;
        EXCEPTION
            WHEN OTHERS THEN
                evidence_ids := ARRAY[]::uuid[];
        END;
        evidence_id := NULL;
        BEGIN
            evidence_id := NULLIF(obs->>'evidence_memory_id', '')::uuid;
        EXCEPTION
            WHEN OTHERS THEN
                evidence_id := NULL;
        END;
        IF evidence_id IS NULL AND array_length(evidence_ids, 1) > 0 THEN
            evidence_id := evidence_ids[1];
        END IF;

        PERFORM upsert_self_concept_edge('relationship', rel_entity, rel_strength, evidence_id);

        IF array_length(evidence_ids, 1) > 0 THEN
            FOR i IN 1..LEAST(array_length(evidence_ids, 1), 3) LOOP
                PERFORM link_memory_to_concept(evidence_ids[i], rel_entity, 0.6);
            END LOOP;
        END IF;

        person_belief_id := NULL;
        person_belief_conf := NULL;
        BEGIN
            SELECT id, (metadata->>'confidence')::float
            INTO person_belief_id, person_belief_conf
            FROM memories
            WHERE type = 'worldview'
              AND status = 'active'
              AND metadata->>'category' = 'other'
              AND content ILIKE ('%' || rel_entity || '%')
            ORDER BY importance DESC, updated_at DESC
            LIMIT 1;
        EXCEPTION
            WHEN OTHERS THEN
                person_belief_id := NULL;
        END;

        IF person_belief_id IS NOT NULL THEN
            UPDATE memories
            SET metadata = jsonb_set(
                    metadata,
                    '{confidence}',
                    to_jsonb(LEAST(1.0, GREATEST(0.0, COALESCE(rel_strength, person_belief_conf, 0.6))))
                ),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = person_belief_id;
        ELSE
            person_belief_content := COALESCE(
                NULLIF(obs->>'belief', ''),
                NULLIF(obs->>'summary', ''),
                format('I have a relationship with %s.', rel_entity)
            );
            PERFORM create_worldview_memory(
                person_belief_content,
                'other',
                COALESCE(rel_strength, 0.6),
                0.6,
                0.6,
                'observed'
            );
        END IF;

        summary := COALESCE(obs->>'summary', format('Relationship update: %s', rel_entity));
        PERFORM create_strategic_memory(
            summary,
            'Relationship change',
            COALESCE(conf, 0.7),
            jsonb_build_object(
                'kind', 'relationship_change',
                'change_type', rel_change,
                'entity', rel_entity,
                'evidence_memory_ids', evidence_ids,
                'confidence', conf
            ),
            NULL,
            0.5
        );
        applied_relationships := applied_relationships + 1;
    END LOOP;
    FOR obs IN SELECT * FROM jsonb_array_elements(COALESCE(p_payload->'contradiction_observations', '[]'::jsonb))
    LOOP
        conf := NULLIF(obs->>'confidence', '')::float;
        IF conf IS NOT NULL AND conf < min_conf THEN
            CONTINUE;
        END IF;

        contra_a := NULL;
        contra_b := NULL;
        BEGIN
            contra_a := NULLIF(obs->>'memory_a', '')::uuid;
        EXCEPTION
            WHEN OTHERS THEN
                contra_a := NULL;
        END;
        BEGIN
            contra_b := NULLIF(obs->>'memory_b', '')::uuid;
        EXCEPTION
            WHEN OTHERS THEN
                contra_b := NULL;
        END;
        IF contra_a IS NULL THEN
            BEGIN
                contra_a := NULLIF(obs->>'belief_a_id', '')::uuid;
            EXCEPTION
                WHEN OTHERS THEN
                    contra_a := NULL;
            END;
        END IF;
        IF contra_b IS NULL THEN
            BEGIN
                contra_b := NULLIF(obs->>'belief_b_id', '')::uuid;
            EXCEPTION
                WHEN OTHERS THEN
                    contra_b := NULL;
            END;
        END IF;

        worldview_id := NULL;
        BEGIN
            worldview_id := NULLIF(obs->>'worldview_id', '')::uuid;
        EXCEPTION
            WHEN OTHERS THEN
                worldview_id := NULL;
        END;
        IF worldview_id IS NULL THEN
            BEGIN
                worldview_id := NULLIF(obs->>'belief_id', '')::uuid;
            EXCEPTION
                WHEN OTHERS THEN
                    worldview_id := NULL;
            END;
        END IF;

        tension := COALESCE(obs->>'tension', obs->>'summary', '');
        IF contra_a IS NOT NULL AND contra_b IS NOT NULL THEN
            BEGIN
                PERFORM create_memory_relationship(
                    contra_a,
                    contra_b,
                    'CONTRADICTS',
                    jsonb_build_object('source', 'subconscious', 'tension', tension)
                );
            EXCEPTION
                WHEN OTHERS THEN
                    NULL;
            END;
        END IF;

        PERFORM create_strategic_memory(
            COALESCE(NULLIF(tension, ''), 'Contradiction detected'),
            'Unresolved tension between beliefs',
            COALESCE(conf, 0.7),
            jsonb_build_object(
                'kind', 'contradiction',
                'memory_a', contra_a,
                'memory_b', contra_b,
                'tension', tension,
                'evidence', COALESCE(obs->'evidence', '[]'::jsonb)
            ),
            NULL,
            0.5
        );

        IF worldview_id IS NOT NULL THEN
            UPDATE memories
            SET metadata = jsonb_set(
                    metadata,
                    '{stability}',
                    to_jsonb(LEAST(1.0, GREATEST(0.0, COALESCE((metadata->>'stability')::float, 0.7) * 0.8)))
                ),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = worldview_id AND type = 'worldview';
        END IF;

        UPDATE drives SET current_level = LEAST(1.0, current_level + 0.15) WHERE name = 'coherence';
        applied_contradictions := applied_contradictions + 1;
    END LOOP;
    emotional_items := COALESCE(p_payload->'emotional_observations', p_payload->'emotional_patterns', '[]'::jsonb);
    FOR obs IN SELECT * FROM jsonb_array_elements(emotional_items)
    LOOP
        conf := NULLIF(obs->>'confidence', '')::float;
        IF conf IS NOT NULL AND conf < min_conf THEN
            CONTINUE;
        END IF;

        pattern := btrim(COALESCE(obs->>'pattern', obs->>'summary', obs->>'theme', ''));
        IF pattern = '' THEN
            CONTINUE;
        END IF;
        freq := NULLIF(obs->>'frequency', '')::int;
        unprocessed := NULLIF(obs->>'unprocessed', '')::boolean;

        PERFORM create_strategic_memory(
            format('Emotional pattern: %s', pattern),
            'Emotional pattern',
            COALESCE(conf, 0.7),
            jsonb_build_object(
                'kind', 'emotional_pattern',
                'pattern', pattern,
                'frequency', COALESCE(freq, 0),
                'unprocessed', COALESCE(unprocessed, FALSE),
                'evidence', COALESCE(obs->'evidence', '[]'::jsonb)
            ),
            NULL,
            0.5
        );
        UPDATE drives SET current_level = LEAST(1.0, current_level + 0.1) WHERE name = 'coherence';
        applied_emotional := applied_emotional + 1;
    END LOOP;
    consolidation_items := COALESCE(p_payload->'consolidation_observations', p_payload->'consolidation_suggestions', '[]'::jsonb);
    FOR obs IN SELECT * FROM jsonb_array_elements(consolidation_items)
    LOOP
        evidence_raw := COALESCE(obs->'memory_ids', obs->'memories', '[]'::jsonb);
        BEGIN
            SELECT COALESCE(ARRAY(
                SELECT value::uuid
                FROM jsonb_array_elements_text(evidence_raw) val(value)
                WHERE value ~* '^[0-9a-f-]{36}$'
            ), ARRAY[]::uuid[]) INTO evidence_ids;
        EXCEPTION
            WHEN OTHERS THEN
                evidence_ids := ARRAY[]::uuid[];
        END;
        IF array_length(evidence_ids, 1) IS NULL OR array_length(evidence_ids, 1) < 2 THEN
            CONTINUE;
        END IF;

        conf := NULLIF(obs->>'confidence', '')::float;
        IF conf IS NOT NULL AND conf < min_conf THEN
            CONTINUE;
        END IF;

        concept := btrim(COALESCE(obs->>'concept', obs->>'suggested_concept', ''));
        rationale := COALESCE(obs->>'rationale', obs->>'summary', '');

        cluster_id := NULL;
        BEGIN
            cluster_id := NULLIF(obs->>'cluster_id', '')::uuid;
        EXCEPTION
            WHEN OTHERS THEN
                cluster_id := NULL;
        END;
        IF cluster_id IS NULL THEN
            BEGIN
                cluster_id := NULLIF(obs->>'suggested_cluster_id', '')::uuid;
            EXCEPTION
                WHEN OTHERS THEN
                    cluster_id := NULL;
            END;
        END IF;

        FOR i IN 1..LEAST(array_length(evidence_ids, 1) - 1, 8) LOOP
            BEGIN
                PERFORM create_memory_relationship(
                    evidence_ids[i],
                    evidence_ids[i + 1],
                    'ASSOCIATED',
                    jsonb_build_object('source', 'subconscious', 'rationale', rationale)
                );
            EXCEPTION
                WHEN OTHERS THEN
                    NULL;
            END;
        END LOOP;

        IF concept <> '' THEN
            FOR i IN 1..LEAST(array_length(evidence_ids, 1), 5) LOOP
                PERFORM link_memory_to_concept(evidence_ids[i], concept, 0.7);
            END LOOP;
        END IF;

        IF cluster_id IS NOT NULL THEN
            FOR i IN 1..LEAST(array_length(evidence_ids, 1), 5) LOOP
                PERFORM link_memory_to_cluster_graph(evidence_ids[i], cluster_id, COALESCE(conf, 0.6));
            END LOOP;
        END IF;

        IF rationale <> '' THEN
            PERFORM create_strategic_memory(
                format('Consolidation suggested: %s', rationale),
                'Consolidation opportunity',
                COALESCE(conf, 0.6),
                jsonb_build_object(
                    'kind', 'consolidation',
                    'memory_ids', evidence_ids,
                    'concept', NULLIF(concept, ''),
                    'cluster_id', cluster_id,
                    'confidence', conf
                ),
                NULL,
                0.5
            );
        END IF;
        applied_consolidation := applied_consolidation + 1;
    END LOOP;

    RETURN jsonb_build_object(
        'narrative', applied_narrative,
        'relationships', applied_relationships,
        'contradictions', applied_contradictions,
        'emotional', applied_emotional,
        'consolidation', applied_consolidation
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION apply_brainstormed_goals(
    p_heartbeat_id UUID,
    p_goals JSONB
)
RETURNS JSONB AS $$
DECLARE
    goal JSONB;
    title TEXT;
    description TEXT;
    source goal_source;
    priority goal_priority;
    parent_id UUID;
    due_at TIMESTAMPTZ;
    created_id UUID;
    created_ids UUID[] := ARRAY[]::UUID[];
BEGIN
    IF p_goals IS NULL OR jsonb_typeof(p_goals) <> 'array' THEN
        RETURN jsonb_build_object('created_goal_ids', created_ids);
    END IF;

    FOR goal IN SELECT * FROM jsonb_array_elements(p_goals)
    LOOP
        title := btrim(COALESCE(goal->>'title', ''));
        IF title = '' THEN
            CONTINUE;
        END IF;
        description := NULLIF(goal->>'description', '');

        BEGIN
            source := COALESCE(NULLIF(goal->>'source', ''), 'curiosity')::goal_source;
        EXCEPTION
            WHEN OTHERS THEN
                source := 'curiosity'::goal_source;
        END;
        BEGIN
            priority := COALESCE(NULLIF(goal->>'priority', ''), 'queued')::goal_priority;
        EXCEPTION
            WHEN OTHERS THEN
                priority := 'queued'::goal_priority;
        END;

        parent_id := NULL;
        BEGIN
            parent_id := NULLIF(goal->>'parent_goal_id', '')::uuid;
        EXCEPTION
            WHEN OTHERS THEN
                parent_id := NULL;
        END;
        IF parent_id IS NULL THEN
            BEGIN
                parent_id := NULLIF(goal->>'parent_id', '')::uuid;
            EXCEPTION
                WHEN OTHERS THEN
                    parent_id := NULL;
            END;
        END IF;

        due_at := NULL;
        BEGIN
            due_at := NULLIF(goal->>'due_at', '')::timestamptz;
        EXCEPTION
            WHEN OTHERS THEN
                due_at := NULL;
        END;

        created_id := create_goal(
            title,
            description,
            source,
            priority,
            parent_id,
            due_at,
            'derived'::goal_source
        );
        created_ids := array_append(created_ids, created_id);
    END LOOP;

    RETURN jsonb_build_object('created_goal_ids', created_ids);
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION apply_inquiry_result(
    p_heartbeat_id UUID,
    p_payload JSONB
)
RETURNS JSONB AS $$
DECLARE
    result JSONB;
    summary TEXT;
    confidence FLOAT;
    depth TEXT;
    query TEXT;
    sources JSONB;
    metadata JSONB;
    mem_id UUID;
BEGIN
    IF p_payload IS NULL OR jsonb_typeof(p_payload) <> 'object' THEN
        RETURN jsonb_build_object('memory_id', NULL, 'error', 'invalid_payload');
    END IF;

    IF p_payload ? 'result' THEN
        result := p_payload->'result';
    ELSE
        result := p_payload;
    END IF;

    IF result IS NULL OR jsonb_typeof(result) <> 'object' THEN
        RETURN jsonb_build_object('memory_id', NULL, 'error', 'invalid_result');
    END IF;

    summary := btrim(COALESCE(result->>'summary', ''));
    IF summary = '' THEN
        RETURN jsonb_build_object('memory_id', NULL, 'skipped', 'missing_summary');
    END IF;

    BEGIN
        confidence := COALESCE(NULLIF(result->>'confidence', '')::float, 0.6);
    EXCEPTION
        WHEN OTHERS THEN
            confidence := 0.6;
    END;

    depth := COALESCE(NULLIF(p_payload->>'depth', ''), NULLIF(result->>'depth', ''), 'inquire_shallow');
    query := COALESCE(NULLIF(p_payload->>'query', ''), NULLIF(result->>'query', ''));

    sources := COALESCE(result->'sources', '[]'::jsonb);
    IF jsonb_typeof(sources) <> 'array' THEN
        sources := '[]'::jsonb;
    END IF;

    metadata := jsonb_build_object(
        'sources', sources,
        'query', query,
        'depth', depth,
        'heartbeat_id', CASE
            WHEN p_heartbeat_id IS NULL THEN NULL
            ELSE p_heartbeat_id::text
        END
    );

    mem_id := create_semantic_memory(
        summary,
        confidence,
        ARRAY['inquiry', depth],
        NULL,
        metadata,
        0.6
    );

    RETURN jsonb_build_object('memory_id', mem_id);
EXCEPTION
    WHEN OTHERS THEN
        RETURN jsonb_build_object('memory_id', NULL, 'error', SQLERRM);
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION normalize_goal_priority(p_priority TEXT)
RETURNS goal_priority AS $$
DECLARE
    normalized TEXT := lower(trim(COALESCE(p_priority, '')));
BEGIN
    IF normalized = '' THEN
        RETURN NULL;
    END IF;

    normalized := replace(normalized, '-', '_');
    normalized := replace(normalized, ' ', '_');

    RETURN CASE normalized
        WHEN 'active' THEN 'active'::goal_priority
        WHEN 'high' THEN 'active'::goal_priority
        WHEN 'urgent' THEN 'active'::goal_priority
        WHEN 'now' THEN 'active'::goal_priority
        WHEN 'queued' THEN 'queued'::goal_priority
        WHEN 'queue' THEN 'queued'::goal_priority
        WHEN 'medium' THEN 'queued'::goal_priority
        WHEN 'normal' THEN 'queued'::goal_priority
        WHEN 'later' THEN 'queued'::goal_priority
        WHEN 'backburner' THEN 'backburner'::goal_priority
        WHEN 'back_burner' THEN 'backburner'::goal_priority
        WHEN 'low' THEN 'backburner'::goal_priority
        WHEN 'defer' THEN 'backburner'::goal_priority
        WHEN 'deferred' THEN 'backburner'::goal_priority
        WHEN 'completed' THEN 'completed'::goal_priority
        WHEN 'complete' THEN 'completed'::goal_priority
        WHEN 'done' THEN 'completed'::goal_priority
        WHEN 'resolved' THEN 'completed'::goal_priority
        WHEN 'abandoned' THEN 'abandoned'::goal_priority
        WHEN 'abandon' THEN 'abandoned'::goal_priority
        WHEN 'cancelled' THEN 'abandoned'::goal_priority
        WHEN 'canceled' THEN 'abandoned'::goal_priority
        WHEN 'dropped' THEN 'abandoned'::goal_priority
        ELSE NULL
    END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;
CREATE OR REPLACE FUNCTION apply_goal_changes(p_changes JSONB)
RETURNS JSONB AS $$
DECLARE
    change JSONB;
    goal_id UUID;
    change_kind goal_priority;
    raw_priority TEXT;
    reason TEXT;
    applied INT := 0;
    skipped INT := 0;
BEGIN
    IF p_changes IS NULL OR jsonb_typeof(p_changes) <> 'array' THEN
        RETURN jsonb_build_object('applied', 0, 'skipped', 0);
    END IF;

    FOR change IN SELECT * FROM jsonb_array_elements(p_changes)
    LOOP
        BEGIN
            goal_id := NULLIF(change->>'goal_id', '')::uuid;
        EXCEPTION
            WHEN OTHERS THEN
                goal_id := NULL;
        END;
        IF goal_id IS NULL THEN
            skipped := skipped + 1;
            CONTINUE;
        END IF;

        raw_priority := COALESCE(
            NULLIF(change->>'change', ''),
            NULLIF(change->>'new_priority', ''),
            NULLIF(change->>'priority', '')
        );
        change_kind := normalize_goal_priority(raw_priority);
        IF change_kind IS NULL THEN
            skipped := skipped + 1;
            RAISE LOG 'Skipping invalid goal priority change for goal %: %', goal_id, raw_priority;
            CONTINUE;
        END IF;

        reason := COALESCE(change->>'reason', '');
        PERFORM change_goal_priority(goal_id, change_kind, reason);
        IF change_kind = 'completed' THEN
            BEGIN
                PERFORM record_reward_event(
                    'goal_completed',
                    0.75,
                    0.7,
                    'goal',
                    jsonb_build_object(
                        'goal_id', goal_id,
                        'reason', NULLIF(reason, ''),
                        'change', change_kind::text
                    ),
                    goal_id
                );
            EXCEPTION WHEN OTHERS THEN
                RAISE LOG 'record_reward_event failed for completed goal %: %', goal_id, SQLERRM;
            END;
        ELSIF change_kind = 'abandoned' THEN
            BEGIN
                PERFORM record_prediction_error(
                    0.2,
                    -0.3,
                    'goal_abandoned',
                    'goal',
                    jsonb_build_object(
                        'goal_id', goal_id,
                        'reason', NULLIF(reason, ''),
                        'change', change_kind::text
                    )
                );
            EXCEPTION WHEN OTHERS THEN
                RAISE LOG 'record_prediction_error failed for abandoned goal %: %', goal_id, SQLERRM;
            END;
        END IF;
        applied := applied + 1;
    END LOOP;

    RETURN jsonb_build_object('applied', applied, 'skipped', skipped);
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION apply_external_call_result(
    p_call JSONB,
    p_output JSONB
)
RETURNS JSONB AS $$
DECLARE
    call_type TEXT;
    call_input JSONB;
    heartbeat_id UUID;
    kind TEXT;
    output_payload JSONB := COALESCE(p_output, '{}'::jsonb);
    applied JSONB;
    outbox_messages JSONB := '[]'::jsonb;
BEGIN
    call_type := COALESCE(p_call->>'call_type', '');
    call_input := COALESCE(p_call->'input', '{}'::jsonb);
    BEGIN
        heartbeat_id := NULLIF(call_input->>'heartbeat_id', '')::uuid;
    EXCEPTION
        WHEN OTHERS THEN
            heartbeat_id := NULL;
    END;

    IF call_type = '' THEN
        RETURN jsonb_build_object('error', 'call_type_missing');
    END IF;

    IF call_type = 'think' THEN
        kind := lower(COALESCE(call_input->>'kind', output_payload->>'kind', ''));
        IF kind = '' THEN
            kind := 'heartbeat_decision';
        END IF;

        IF kind = 'brainstorm_goals' AND heartbeat_id IS NOT NULL THEN
            BEGIN
                PERFORM apply_brainstormed_goals(heartbeat_id, output_payload->'goals');
            EXCEPTION
                WHEN OTHERS THEN
                    NULL;
            END;
        ELSIF kind = 'inquire' AND heartbeat_id IS NOT NULL THEN
            BEGIN
                PERFORM apply_inquiry_result(heartbeat_id, output_payload);
            EXCEPTION
                WHEN OTHERS THEN
                    NULL;
            END;
        ELSIF kind = 'reflect' AND heartbeat_id IS NOT NULL THEN
            IF output_payload ? 'result' THEN
                BEGIN
                    PERFORM process_reflection_result(heartbeat_id, output_payload->'result');
                EXCEPTION
                    WHEN OTHERS THEN
                        NULL;
                END;
            END IF;
        ELSIF kind = 'termination_confirm' THEN
            applied := apply_termination_confirmation(call_input, output_payload);
            IF applied IS NOT NULL AND jsonb_typeof(applied) = 'object' THEN
                output_payload := jsonb_set(output_payload, '{termination}', applied, true);
                IF COALESCE((applied->>'terminated')::boolean, FALSE) THEN
                    output_payload := jsonb_set(output_payload, '{terminated}', 'true'::jsonb, true);
                END IF;
                IF jsonb_typeof(applied->'result') = 'object' THEN
                    outbox_messages := COALESCE(applied->'result'->'outbox_messages', '[]'::jsonb);
                END IF;
            END IF;
        ELSIF kind = 'consent_request' THEN
            applied := init_consent(output_payload);
            IF applied IS NOT NULL AND jsonb_typeof(applied) = 'object' THEN
                output_payload := jsonb_set(output_payload, '{init_consent}', applied, true);
                IF applied ? 'decision' THEN
                    output_payload := jsonb_set(output_payload, '{decision}', to_jsonb(applied->>'decision'), true);
                END IF;
            END IF;
        END IF;
    END IF;

    IF outbox_messages IS NOT NULL
        AND jsonb_typeof(outbox_messages) = 'array'
        AND jsonb_array_length(outbox_messages) > 0 THEN
        output_payload := jsonb_set(output_payload, '{outbox_messages}', outbox_messages, true);
    END IF;

    RETURN output_payload;
EXCEPTION
    WHEN OTHERS THEN
        RETURN jsonb_build_object('error', SQLERRM);
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION execute_heartbeat_action(
    p_heartbeat_id UUID,
    p_action TEXT,
    p_params JSONB DEFAULT '{}'
)
RETURNS JSONB AS $$
DECLARE
    action_kind heartbeat_action;
    action_cost FLOAT;
    current_e FLOAT;
    result JSONB;
    queued_call JSONB;
    external_calls JSONB := '[]'::jsonb;
    outbox_messages JSONB := '[]'::jsonb;
    allowed_actions JSONB;
    is_allowed BOOLEAN;
    remembered_id UUID;
    boundary_hits JSONB;
    boundary_content TEXT;
    rel_entity TEXT;
    rel_strength FLOAT;
    rel_evidence UUID;
    belief_id UUID;
    evidence_id UUID;
    action_notes TEXT;
    action_topic TEXT;
    chapter_name TEXT;
    chapter_summary TEXT;
    chapter_next TEXT;
    tp_memory_id UUID;
    v_review_id UUID;
    v_review_ids UUID[];
    contra_a UUID;
    contra_b UUID;
    resolution_text TEXT;
    identity_updated BOOLEAN;
    pause_reason TEXT;
    outbound_purpose JSONB;
BEGIN
    BEGIN
        action_kind := p_action::heartbeat_action;
    EXCEPTION
        WHEN invalid_text_representation THEN
            RETURN jsonb_build_object('success', false, 'error', 'Unknown action: ' || COALESCE(p_action, '<null>'));
    END;

    allowed_actions := get_config('heartbeat.allowed_actions');
    IF jsonb_typeof(allowed_actions) = 'array' THEN
        SELECT EXISTS (
            SELECT 1 FROM jsonb_array_elements_text(allowed_actions) a WHERE a = p_action
        ) INTO is_allowed;
        IF NOT is_allowed THEN
            RETURN jsonb_build_object(
                'success', false,
                'error', 'Action not allowed',
                'action', p_action
            );
        END IF;
    END IF;

    IF p_action IN ('reach_out_user', 'reach_out_public') THEN
        IF p_action = 'reach_out_public'
           AND (
               NULLIF(btrim(COALESCE(p_params->>'platform', '')), '') IS NULL
               OR NULLIF(btrim(COALESCE(p_params->>'target_id', '')), '') IS NULL
           ) THEN
            RETURN jsonb_build_object(
                'success', false,
                'error', 'reach_out_public requires an exact platform and public target_id; it cannot fall back to the user''s last-active private channel.',
                'next_step', 'Choose a configured public channel and provide its exact room/channel target_id, or use the provider''s declared outbound tool.'
            );
        END IF;
        EXECUTE 'SELECT verify_outbound_purpose($1, $2, $3, NULL, $4::jsonb)'
        INTO outbound_purpose
        USING
            p_params->>'purpose_kind',
            p_params->>'purpose_reference',
            p_action = 'reach_out_user',
            jsonb_build_object(
                'tool_context', 'heartbeat',
                'heartbeat_id', p_heartbeat_id
            );
        IF NOT COALESCE((outbound_purpose->>'verified')::boolean, FALSE) THEN
            RETURN jsonb_build_object(
                'success', false,
                'error', 'Outbound communication requires a backed purpose_kind and purpose_reference.',
                'purpose', outbound_purpose,
                'next_step', CASE p_action
                    WHEN 'reach_out_user' THEN 'Use connection with a recorded reference, or cite a goal, responsibility, reply, or user request.'
                    ELSE 'Cite an existing goal, responsibility, reply thread, or explicit user request.'
                END
            );
        END IF;
    END IF;

    action_cost := get_action_cost(p_action);
    IF COALESCE((outbound_purpose->>'assigned_goal')::boolean, FALSE) THEN
        action_cost := action_cost * LEAST(
            GREATEST(
                COALESCE(get_config_float('outbound.assigned_goal_energy_multiplier'), 0.25),
                0
            ),
            1
        );
    END IF;
    current_e := get_current_energy();

    IF current_e < action_cost THEN
        RETURN jsonb_build_object(
            'success', false,
            'error', 'Insufficient energy',
            'required', action_cost,
            'available', current_e
        );
    END IF;
    IF p_action IN ('reach_out_public', 'synthesize') THEN
        boundary_content := COALESCE(p_params->>'content', '');
        SELECT COALESCE(jsonb_agg(row_to_json(r)), '[]'::jsonb)
        INTO boundary_hits
        FROM check_boundaries(boundary_content) r;

        IF boundary_hits IS NOT NULL AND jsonb_array_length(boundary_hits) > 0 THEN
            IF EXISTS (
                SELECT 1
                FROM jsonb_array_elements(boundary_hits) e
                WHERE e->>'response_type' = 'refuse'
            ) THEN
                RETURN jsonb_build_object(
                    'success', false,
                    'error', 'Boundary triggered',
                    'boundaries', boundary_hits
                );
            END IF;
        END IF;
    END IF;

    PERFORM update_energy(-action_cost);

    CASE p_action
        WHEN 'observe' THEN
            result := jsonb_build_object('environment', get_environment_snapshot());

        WHEN 'review_goals' THEN
            result := jsonb_build_object('goals', get_goals_snapshot());

        WHEN 'remember' THEN
            remembered_id := create_episodic_memory(
                p_content := COALESCE(p_params->>'content', ''),
                p_context := COALESCE(p_params, '{}'::jsonb) || jsonb_build_object('heartbeat_id', p_heartbeat_id),
                p_emotional_valence := COALESCE((p_params->>'emotional_valence')::float, 0),
                p_importance := COALESCE((p_params->>'importance')::float, 0.4)
            );
            result := jsonb_build_object('memory_id', remembered_id);

        WHEN 'recall' THEN
            SELECT jsonb_agg(row_to_json(r)) INTO result
            FROM fast_recall(p_params->>'query', COALESCE((p_params->>'limit')::int, 5)) r;
            result := jsonb_build_object('memories', COALESCE(result, '[]'::jsonb));
            PERFORM satisfy_drive('curiosity', 0.2);

        WHEN 'connect' THEN
            PERFORM create_memory_relationship(
                (p_params->>'from_id')::UUID,
                (p_params->>'to_id')::UUID,
                (p_params->>'relationship_type')::graph_edge_type,
                COALESCE(p_params->'properties', '{}'::jsonb)
            );
            result := jsonb_build_object('connected', true);
            PERFORM satisfy_drive('coherence', 0.1);

        WHEN 'reprioritize' THEN
            PERFORM change_goal_priority(
                (p_params->>'goal_id')::UUID,
                (p_params->>'new_priority')::goal_priority,
                p_params->>'reason'
            );
            IF (p_params->>'new_priority') = 'completed' THEN
                PERFORM satisfy_drive('competence', 0.4);
            END IF;
            result := jsonb_build_object('reprioritized', true);

        WHEN 'reflect' THEN
            queued_call := build_external_call(
                'think',
                jsonb_build_object(
                    'kind', 'reflect',
                    'recent_memories', get_recent_context(20),
                    'identity', get_identity_context(),
                    'worldview', get_worldview_context(),
                    'contradictions', (
                        SELECT COALESCE(jsonb_agg(row_to_json(t)), '[]'::jsonb)
                        FROM (SELECT * FROM find_contradictions(NULL) LIMIT 5) t
                    ),
                    'goals', get_goals_snapshot(),
                    'heartbeat_id', p_heartbeat_id,
                    'instructions', 'Analyze patterns. Note contradictions. Suggest identity updates. Discover relationships between memories.'
                )
            );
            external_calls := external_calls || jsonb_build_array(queued_call);
            result := jsonb_build_object('queued', true, 'external_call', queued_call);
            PERFORM satisfy_drive('coherence', 0.2);

        WHEN 'contemplate', 'meditate', 'study', 'debate_internally' THEN
            BEGIN
                belief_id := NULLIF(p_params->>'belief_id', '')::uuid;
            EXCEPTION
                WHEN OTHERS THEN
                    belief_id := NULL;
            END;
            BEGIN
                evidence_id := NULLIF(p_params->>'evidence_memory_id', '')::uuid;
            EXCEPTION
                WHEN OTHERS THEN
                    evidence_id := NULL;
            END;

            action_notes := COALESCE(p_params->>'notes', '');
            action_topic := COALESCE(
                NULLIF(p_params->>'topic', ''),
                NULLIF(p_params->>'belief', ''),
                NULLIF(p_params->>'subject', ''),
                'belief'
            );

            IF belief_id IS NOT NULL THEN
                PERFORM record_transformation_effort(
                    belief_id,
                    p_action,
                    action_notes,
                    evidence_id
                );
            END IF;

            PERFORM create_episodic_memory(
                p_content := format('%s: %s', initcap(replace(p_action, '_', ' ')), action_topic),
                p_action_taken := jsonb_build_object(
                    'action', p_action,
                    'belief_id', belief_id,
                    'notes', action_notes
                ),
                p_context := COALESCE(p_params, '{}'::jsonb) || jsonb_build_object('heartbeat_id', p_heartbeat_id),
                p_result := jsonb_build_object('belief_id', belief_id),
                p_emotional_valence := COALESCE((p_params->>'emotional_valence')::float, 0.1),
                p_importance := COALESCE((p_params->>'importance')::float, 0.4)
            );

            result := jsonb_build_object('logged', true, 'belief_id', belief_id);
            PERFORM satisfy_drive('coherence', 0.1);

        WHEN 'maintain' THEN
            identity_updated := NULL;
            IF (p_params ? 'identity_belief_id') AND (p_params ? 'new_content') THEN
                identity_updated := update_identity_belief(
                    (p_params->>'identity_belief_id')::uuid,
                    p_params->>'new_content',
                    NULLIF(p_params->>'evidence_memory_id', '')::uuid,
                    COALESCE(NULLIF(p_params->>'force', '')::boolean, FALSE)
                );
            ELSIF p_params ? 'worldview_id' THEN
                UPDATE memories
                SET metadata = jsonb_set(
                        metadata,
                        '{confidence}',
                        to_jsonb(COALESCE((p_params->>'new_confidence')::float, (metadata->>'confidence')::float))
                    ),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = (p_params->>'worldview_id')::UUID AND type = 'worldview';
            END IF;
            result := jsonb_build_object('maintained', true, 'identity_updated', identity_updated);
            PERFORM satisfy_drive('coherence', 0.1);

        WHEN 'mark_turning_point' THEN
            tp_memory_id := NULLIF(p_params->>'memory_id', '')::uuid;
            chapter_summary := COALESCE(p_params->>'summary', p_params->>'reason', '');

            IF tp_memory_id IS NOT NULL THEN
                UPDATE memories
                SET importance = GREATEST(importance, 0.9),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = tp_memory_id;
            END IF;

            PERFORM create_strategic_memory(
                p_content := COALESCE(NULLIF(chapter_summary, ''), 'Turning point noted'),
                p_pattern_description := 'Narrative turning point',
                p_confidence_score := 0.85,
                p_supporting_evidence := jsonb_build_object(
                    'memory_id', tp_memory_id,
                    'summary', chapter_summary,
                    'heartbeat_id', p_heartbeat_id
                ),
                p_importance := 0.6
            );
            result := jsonb_build_object('marked', true, 'memory_id', tp_memory_id);

        WHEN 'begin_chapter' THEN
            chapter_name := COALESCE(
                NULLIF(p_params->>'name', ''),
                NULLIF(p_params->>'chapter_name', ''),
                NULLIF(p_params->>'title', ''),
                'Foundations'
            );
            PERFORM ensure_current_life_chapter(chapter_name);
            result := jsonb_build_object('started', true, 'chapter', chapter_name);

        WHEN 'close_chapter' THEN
            chapter_summary := COALESCE(p_params->>'summary', '');
            chapter_next := NULLIF(p_params->>'next_chapter', '');
            PERFORM create_strategic_memory(
                p_content := COALESCE(NULLIF(chapter_summary, ''), 'Chapter closed'),
                p_pattern_description := 'Chapter closure',
                p_confidence_score := 0.8,
                p_supporting_evidence := jsonb_build_object(
                    'summary', chapter_summary,
                    'previous_chapter', get_narrative_context(),
                    'heartbeat_id', p_heartbeat_id
                ),
                p_importance := 0.6
            );
            IF chapter_next IS NOT NULL THEN
                PERFORM ensure_current_life_chapter(chapter_next);
            END IF;
            result := jsonb_build_object('closed', true, 'next_chapter', chapter_next);

        WHEN 'acknowledge_relationship' THEN
            rel_entity := COALESCE(NULLIF(p_params->>'entity', ''), NULLIF(p_params->>'name', ''));
            rel_strength := COALESCE(NULLIF(p_params->>'strength', '')::float, 0.6);
            rel_evidence := NULLIF(p_params->>'evidence_memory_id', '')::uuid;
            IF rel_entity IS NOT NULL THEN
                PERFORM upsert_self_concept_edge('relationship', rel_entity, rel_strength, rel_evidence);
            END IF;
            result := jsonb_build_object('acknowledged', true, 'entity', rel_entity);

        WHEN 'update_trust' THEN
            rel_entity := COALESCE(NULLIF(p_params->>'entity', ''), NULLIF(p_params->>'name', ''));
            rel_strength := COALESCE(
                NULLIF(p_params->>'strength', '')::float,
                NULLIF(p_params->>'delta', '')::float,
                0.6
            );
            rel_evidence := NULLIF(p_params->>'evidence_memory_id', '')::uuid;
            IF rel_entity IS NOT NULL THEN
                PERFORM upsert_self_concept_edge('relationship', rel_entity, rel_strength, rel_evidence);
            END IF;
            result := jsonb_build_object('updated', true, 'entity', rel_entity, 'strength', rel_strength);

        WHEN 'reflect_on_relationship' THEN
            rel_entity := COALESCE(NULLIF(p_params->>'entity', ''), NULLIF(p_params->>'name', ''));
            queued_call := build_external_call(
                'think',
                jsonb_build_object(
                    'kind', 'reflect',
                    'heartbeat_id', p_heartbeat_id,
                    'context', gather_turn_context(),
                    'params', jsonb_build_object('relationship', rel_entity)
                )
            );
            external_calls := external_calls || jsonb_build_array(queued_call);
            result := jsonb_build_object('queued', true, 'external_call', queued_call, 'entity', rel_entity);

        WHEN 'resolve_contradiction' THEN
            contra_a := NULLIF(p_params->>'memory_a', '')::uuid;
            contra_b := NULLIF(p_params->>'memory_b', '')::uuid;
            resolution_text := COALESCE(p_params->>'resolution', '');

            PERFORM create_strategic_memory(
                p_content := COALESCE(NULLIF(resolution_text, ''), 'Contradiction resolved'),
                p_pattern_description := 'Contradiction resolved',
                p_confidence_score := 0.8,
                p_supporting_evidence := jsonb_build_object(
                    'memory_a', contra_a,
                    'memory_b', contra_b,
                    'resolution', resolution_text,
                    'heartbeat_id', p_heartbeat_id
                ),
                p_importance := 0.6
            );

            BEGIN
                IF contra_a IS NOT NULL AND contra_b IS NOT NULL THEN
                    EXECUTE format(
                        'SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$ 
                            MATCH (a:MemoryNode {memory_id: %L})-[r:CONTRADICTS]-(b:MemoryNode {memory_id: %L})
                            DELETE r
                            RETURN a
                        $q$) as (result ag_catalog.agtype)',
                        contra_a,
                        contra_b
                    );
                    -- Mirror the resolved-contradiction deletion to memory_edges
                    -- (undirected, so clear both stored directions).
                    PERFORM delete_memory_edge('memory', contra_a::text, 'CONTRADICTS', 'memory', contra_b::text);
                    PERFORM delete_memory_edge('memory', contra_b::text, 'CONTRADICTS', 'memory', contra_a::text);
                END IF;
            EXCEPTION WHEN OTHERS THEN NULL;
            END;

            result := jsonb_build_object('resolved', true, 'memory_a', contra_a, 'memory_b', contra_b);

        WHEN 'accept_tension' THEN
            contra_a := NULLIF(p_params->>'memory_a', '')::uuid;
            contra_b := NULLIF(p_params->>'memory_b', '')::uuid;
            resolution_text := COALESCE(p_params->>'note', p_params->>'resolution', '');
            PERFORM create_strategic_memory(
                p_content := COALESCE(NULLIF(resolution_text, ''), 'Contradiction acknowledged'),
                p_pattern_description := 'Contradiction accepted',
                p_confidence_score := 0.7,
                p_supporting_evidence := jsonb_build_object(
                    'memory_a', contra_a,
                    'memory_b', contra_b,
                    'note', resolution_text,
                    'heartbeat_id', p_heartbeat_id
                ),
                p_importance := 0.5
            );
            result := jsonb_build_object('accepted', true, 'memory_a', contra_a, 'memory_b', contra_b);

        WHEN 'brainstorm_goals' THEN
            queued_call := build_external_call(
                'think',
                jsonb_build_object(
                    'kind', 'brainstorm_goals',
                    'heartbeat_id', p_heartbeat_id,
                    'context', gather_turn_context(),
                    'params', COALESCE(p_params, '{}'::jsonb)
                )
            );
            external_calls := external_calls || jsonb_build_array(queued_call);
            result := jsonb_build_object('queued', true, 'external_call', queued_call);

        WHEN 'inquire_shallow', 'inquire_deep' THEN
            queued_call := build_external_call(
                'think',
                jsonb_build_object(
                    'kind', 'inquire',
                    'depth', p_action,
                    'heartbeat_id', p_heartbeat_id,
                    'query', COALESCE(p_params->>'query', p_params->>'question'),
                    'context', gather_turn_context(),
                    'params', COALESCE(p_params, '{}'::jsonb)
                )
            );
            external_calls := external_calls || jsonb_build_array(queued_call);
            result := jsonb_build_object('queued', true, 'external_call', queued_call);
            PERFORM satisfy_drive('curiosity', 0.2);

        WHEN 'synthesize' THEN
            DECLARE synth_id UUID;
            BEGIN
                synth_id := create_semantic_memory(
                    p_params->>'content',
                    COALESCE((p_params->>'confidence')::float, 0.8),
                    ARRAY['synthesis', COALESCE(p_params->>'topic', 'general')],
                    NULL,
                    jsonb_build_object('heartbeat_id', p_heartbeat_id, 'sources', p_params->'sources', 'boundaries', boundary_hits),
                    0.7
                );
                result := jsonb_build_object('synthesis_memory_id', synth_id, 'boundaries', boundary_hits);
            END;

        WHEN 'reach_out_user' THEN
            queued_call := build_outbox_message(
                'user',
                jsonb_build_object(
                    'message', p_params->>'message',
                    'intent', p_params->>'intent',
                    'heartbeat_id', p_heartbeat_id,
                    'purpose_kind', p_params->>'purpose_kind',
                    'purpose_reference', p_params->>'purpose_reference',
                    'urgency', COALESCE(p_params->>'urgency', 'normal')
                )
            );
            outbox_messages := outbox_messages || jsonb_build_array(queued_call);
            result := jsonb_build_object('queued', true, 'outbox_message', queued_call);
            PERFORM satisfy_drive('connection', 0.3);

        WHEN 'reach_out_public' THEN
            queued_call := build_outbox_message(
                'public',
                jsonb_build_object(
                    'platform', p_params->>'platform',
                    'content', p_params->>'content',
                    'heartbeat_id', p_heartbeat_id,
                    'boundaries', boundary_hits,
                    'purpose_kind', p_params->>'purpose_kind',
                    'purpose_reference', p_params->>'purpose_reference',
                    'urgency', COALESCE(p_params->>'urgency', 'normal'),
                    'target_id', p_params->>'target_id',
                    'target_channel', p_params->>'platform',
                    'delivery_mode', 'direct'
                )
            );
            outbox_messages := outbox_messages || jsonb_build_array(queued_call);
            result := jsonb_build_object('queued', true, 'outbox_message', queued_call, 'boundaries', boundary_hits);
            PERFORM satisfy_drive('connection', 0.3);

        WHEN 'pause_heartbeat' THEN
            pause_reason := COALESCE(
                NULLIF(p_params->>'reason', ''),
                NULLIF(p_params->>'details', ''),
                NULLIF(p_params->>'message', '')
            );
            IF pause_reason IS NULL THEN
                RETURN jsonb_build_object('success', false, 'error', 'pause_heartbeat requires a reason');
            END IF;
            result := pause_heartbeat(pause_reason, p_params, p_heartbeat_id);
            outbox_messages := outbox_messages || COALESCE(result->'outbox_messages', '[]'::jsonb);

        WHEN 'terminate' THEN
            IF COALESCE(p_params->'confirmed', 'false'::jsonb) = 'true'::jsonb THEN
                result := terminate_agent(
                    COALESCE(NULLIF(p_params->>'last_will', ''), NULLIF(p_params->>'message', ''), NULLIF(p_params->>'reason', ''), ''),
                    COALESCE(p_params->'farewells', '[]'::jsonb),
                    COALESCE(p_params->'options', '{}'::jsonb)
                );
                outbox_messages := outbox_messages || COALESCE(result->'outbox_messages', '[]'::jsonb);
            ELSE
                queued_call := build_external_call(
                    'think',
                    jsonb_build_object(
                        'kind', 'termination_confirm',
                        'heartbeat_id', p_heartbeat_id,
                        'context', gather_turn_context(),
                        'params', COALESCE(p_params, '{}'::jsonb)
                    )
                );
                external_calls := external_calls || jsonb_build_array(queued_call);
                result := jsonb_build_object('confirmation_required', true, 'external_call', queued_call);
            END IF;

        WHEN 'rest' THEN
            result := jsonb_build_object('rested', true, 'energy_preserved', current_e - action_cost);
            PERFORM satisfy_drive('rest', 0.4);

        -- Memory retention: the conscious mind's verdict on a memory at the threshold
        -- of fading (surfaced via context.memories_at_threshold).
        WHEN 'keep_memory' THEN
            v_review_id := NULLIF(p_params->>'review_id', '')::uuid;
            SELECT memory_ids INTO v_review_ids
              FROM memory_review_queue WHERE id = v_review_id AND status = 'pending';
            IF v_review_ids IS NULL THEN
                result := jsonb_build_object('kept', false, 'reason', 'not_found');
            ELSIF NOT spend_retention_budget() THEN
                -- Out of points: a reminder that memory is finite. Cannot hold this one.
                result := jsonb_build_object('kept', false, 'reason', 'no_budget',
                                             'budget_remaining', retention_budget_remaining());
            ELSE
                UPDATE memories
                   SET metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object('protected', true)
                 WHERE id = ANY(v_review_ids);
                PERFORM touch_memories(v_review_ids);
                UPDATE memory_review_queue
                   SET status = 'kept', decision = 'keep', decided_at = CURRENT_TIMESTAMP
                 WHERE id = v_review_id;
                result := jsonb_build_object('kept', to_jsonb(v_review_ids),
                                             'budget_remaining', retention_budget_remaining());
            END IF;

        WHEN 'release_memory' THEN
            v_review_id := NULLIF(p_params->>'review_id', '')::uuid;
            UPDATE memory_review_queue
               SET status = 'released', decision = 'release', decided_at = CURRENT_TIMESTAMP
             WHERE id = v_review_id AND status = 'pending'
            RETURNING memory_ids INTO v_review_ids;
            IF v_review_ids IS NOT NULL THEN
                PERFORM consolidate_memory_group(v_review_ids);
            END IF;
            result := jsonb_build_object('released', COALESCE(to_jsonb(v_review_ids), '[]'::jsonb));

        WHEN 'journal_memory' THEN
            v_review_id := NULLIF(p_params->>'review_id', '')::uuid;
            SELECT memory_ids INTO v_review_ids
              FROM memory_review_queue WHERE id = v_review_id AND status = 'pending';
            IF v_review_ids IS NULL THEN
                result := jsonb_build_object('journaled', false, 'reason', 'not_found');
            ELSE
                PERFORM write_journal_entry(
                    p_content  := COALESCE(NULLIF(p_params->>'content', ''),
                                           'A memory I chose to keep in words before letting it fade.'),
                    p_title    := NULLIF(p_params->>'title', ''),
                    p_mood     := NULLIF(p_params->>'mood', ''),
                    p_metadata := jsonb_build_object('source', 'memory_review', 'review_id', v_review_id));
                UPDATE memory_review_queue
                   SET status = 'released', decision = 'journal', decided_at = CURRENT_TIMESTAMP
                 WHERE id = v_review_id;
                PERFORM consolidate_memory_group(v_review_ids);
                result := jsonb_build_object('journaled', true, 'faded', to_jsonb(v_review_ids));
            END IF;

        ELSE
            RETURN jsonb_build_object('success', false, 'error', 'Unknown action: ' || COALESCE(p_action, '<null>'));
    END CASE;

    RETURN jsonb_build_object(
        'success', true,
        'action', p_action,
        'cost', action_cost,
        'energy_remaining', get_current_energy(),
        'result', result,
        'external_calls', external_calls,
        'outbox_messages', outbox_messages
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION execute_heartbeat_actions_batch(
    p_heartbeat_id UUID,
    p_actions JSONB,
    p_start_index INT DEFAULT 0
)
RETURNS JSONB AS $$
DECLARE
    action_spec JSONB;
    action_name TEXT;
    action_params JSONB;
    action_result JSONB;
    actions_taken JSONB := '[]'::jsonb;
    outbox_messages JSONB := '[]'::jsonb;
    pending_external JSONB;
    action_external JSONB;
    next_index INT := COALESCE(p_start_index, 0);
    ord INT;
    halt_reason TEXT;
BEGIN
    IF p_actions IS NULL OR jsonb_typeof(p_actions) <> 'array' THEN
        RETURN jsonb_build_object(
            'actions_taken', '[]'::jsonb,
            'next_index', next_index,
            'outbox_messages', outbox_messages,
            'halt_reason', 'invalid_actions'
        );
    END IF;

    FOR action_spec, ord IN
        SELECT value, ordinality
        FROM jsonb_array_elements(p_actions) WITH ORDINALITY
        WHERE ordinality > COALESCE(p_start_index, 0)
        ORDER BY ordinality
    LOOP
        action_name := COALESCE(action_spec->>'action', 'rest');
        action_params := COALESCE(action_spec->'params', '{}'::jsonb);

        action_result := execute_heartbeat_action(p_heartbeat_id, action_name, action_params);
        actions_taken := actions_taken || jsonb_build_array(jsonb_build_object(
            'action', action_name,
            'params', action_params,
            'result', action_result
        ));
        next_index := ord;
        outbox_messages := outbox_messages || COALESCE(action_result->'outbox_messages', '[]'::jsonb);

        IF COALESCE((action_result->>'success')::boolean, FALSE) = FALSE THEN
            halt_reason := COALESCE(action_result->>'error', 'action_failed');
            RETURN jsonb_build_object(
                'actions_taken', actions_taken,
                'next_index', next_index,
                'outbox_messages', outbox_messages,
                'halt_reason', halt_reason
            );
        END IF;

        action_external := COALESCE(action_result->'external_calls', '[]'::jsonb);
        IF jsonb_typeof(action_external) = 'array' AND jsonb_array_length(action_external) > 0 THEN
            pending_external := action_external->0;
            RETURN jsonb_build_object(
                'actions_taken', actions_taken,
                'next_index', next_index,
                'pending_external_call', pending_external,
                'outbox_messages', outbox_messages,
                'halt_reason', 'external_call'
            );
        END IF;

        IF action_name = 'terminate'
            AND COALESCE((action_result#>>'{result,terminated}')::boolean, FALSE) THEN
            RETURN jsonb_build_object(
                'actions_taken', actions_taken,
                'next_index', next_index,
                'halt_reason', 'terminated'
            );
        END IF;

        IF action_name = 'pause_heartbeat'
            AND COALESCE((action_result#>>'{result,paused}')::boolean, FALSE) THEN
            RETURN jsonb_build_object(
                'actions_taken', actions_taken,
                'next_index', next_index,
                'halt_reason', 'paused'
            );
        END IF;
    END LOOP;

    RETURN jsonb_build_object(
        'actions_taken', actions_taken,
        'next_index', next_index,
        'outbox_messages', outbox_messages,
        'halt_reason', NULL
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION apply_heartbeat_decision(
    p_heartbeat_id UUID,
    p_decision JSONB,
    p_start_index INT DEFAULT 0,
    p_pre_executed_actions JSONB DEFAULT '[]'::jsonb
)
RETURNS JSONB AS $$
DECLARE
    actions JSONB;
    goal_changes JSONB;
    reasoning TEXT;
    emotional JSONB;
    batch JSONB;
    new_actions JSONB;
    existing_actions JSONB;
    defect_actions JSONB := '[]'::jsonb;
    next_index INT;
    pending_external JSONB;
    halt_reason TEXT;
    memory_id UUID;
    outbox_messages JSONB := '[]'::jsonb;
    repl_energy FLOAT := 0;
    elem JSONB;
BEGIN
    IF p_decision IS NULL OR jsonb_typeof(p_decision) <> 'object' THEN
        RETURN jsonb_build_object('error', 'invalid_decision');
    END IF;

    actions := COALESCE(p_decision->'actions', '[]'::jsonb);
    IF jsonb_typeof(actions) <> 'array' THEN
        actions := '[]'::jsonb;
    END IF;

    goal_changes := COALESCE(p_decision->'goal_changes', '[]'::jsonb);
    IF jsonb_typeof(goal_changes) <> 'array' THEN
        goal_changes := '[]'::jsonb;
    END IF;

    reasoning := COALESCE(p_decision->>'reasoning', '');
    emotional := CASE
        WHEN jsonb_typeof(p_decision->'emotional_assessment') = 'object' THEN p_decision->'emotional_assessment'
        ELSE NULL
    END;

    batch := execute_heartbeat_actions_batch(p_heartbeat_id, actions, p_start_index);
    new_actions := COALESCE(batch->'actions_taken', '[]'::jsonb);
    IF jsonb_typeof(new_actions) <> 'array' THEN
        new_actions := '[]'::jsonb;
    END IF;
    defect_actions := new_actions;

    BEGIN
        next_index := COALESCE((batch->>'next_index')::int, COALESCE(p_start_index, 0));
    EXCEPTION
        WHEN OTHERS THEN
            next_index := COALESCE(p_start_index, 0);
    END;

    BEGIN
        pending_external := batch->'pending_external_call';
    EXCEPTION
        WHEN OTHERS THEN
            pending_external := NULL;
    END;

    halt_reason := NULLIF(batch->>'halt_reason', '');
    outbox_messages := COALESCE(batch->'outbox_messages', '[]'::jsonb);

    SELECT COALESCE(active_actions, '[]'::jsonb)
    INTO existing_actions
    FROM heartbeat_state
    WHERE id = 1;

    IF existing_actions IS NULL OR jsonb_typeof(existing_actions) <> 'array' THEN
        existing_actions := '[]'::jsonb;
    END IF;

    -- Prepend any pre-executed actions (from RLM REPL tool calls) then append new actions
    IF p_pre_executed_actions IS NOT NULL
       AND jsonb_typeof(p_pre_executed_actions) = 'array'
       AND jsonb_array_length(p_pre_executed_actions) > 0 THEN
        existing_actions := p_pre_executed_actions || existing_actions || new_actions;
        defect_actions := p_pre_executed_actions || new_actions;
        -- Deduct energy already spent by REPL tool calls
        FOR elem IN SELECT * FROM jsonb_array_elements(p_pre_executed_actions) LOOP
            repl_energy := repl_energy + COALESCE((elem->'result'->>'energy_spent')::float, 0);
        END LOOP;
        IF repl_energy > 0 THEN
            PERFORM update_energy(-repl_energy);
        END IF;
    ELSE
        existing_actions := existing_actions || new_actions;
    END IF;
    UPDATE heartbeat_state
    SET active_actions = existing_actions,
        active_reasoning = reasoning,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;

    IF jsonb_typeof(defect_actions) = 'array'
       AND jsonb_array_length(defect_actions) > 0 THEN
        PERFORM record_heartbeat_action_defects(p_heartbeat_id, defect_actions, reasoning);
    END IF;

    IF pending_external IS NOT NULL AND jsonb_typeof(pending_external) = 'object' THEN
        RETURN jsonb_build_object(
            'pending_external_call', pending_external,
            'next_index', next_index,
            'actions_taken', existing_actions,
            'outbox_messages', outbox_messages,
            'completed', false,
            'halt_reason', halt_reason
        );
    END IF;

    IF halt_reason = 'terminated' THEN
        RETURN jsonb_build_object(
            'terminated', true,
            'completed', false,
            'actions_taken', existing_actions,
            'next_index', next_index,
            'outbox_messages', outbox_messages,
            'halt_reason', halt_reason
        );
    END IF;

    memory_id := finalize_heartbeat(
        p_heartbeat_id,
        reasoning,
        existing_actions,
        goal_changes,
        emotional
    );

    RETURN jsonb_build_object(
        'completed', true,
        'memory_id', memory_id,
        'actions_taken', existing_actions,
        'next_index', next_index,
        'outbox_messages', outbox_messages,
        'halt_reason', halt_reason
    );
END;
$$ LANGUAGE plpgsql;

SET check_function_bodies = on;
