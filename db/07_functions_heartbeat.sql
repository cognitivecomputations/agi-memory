-- Hexis schema: heartbeat functions.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

DO $$
BEGIN
    BEGIN
        CREATE TYPE goal_priority AS ENUM (
            'active',
            'queued',
            'backburner',
            'completed',
            'abandoned'
        );
    EXCEPTION WHEN duplicate_object THEN NULL;
    END;
    BEGIN
        CREATE TYPE goal_source AS ENUM (
            'curiosity',
            'user_request',
            'identity',
            'derived',
            'external'
        );
    EXCEPTION WHEN duplicate_object THEN NULL;
    END;
    BEGIN
        CREATE TYPE heartbeat_action AS ENUM (
            'observe',
            'review_goals',
            'remember',
            'recall',
            'connect',
            'reprioritize',
            'reflect',
            'contemplate',
            'meditate',
            'study',
            'debate_internally',
            'maintain',
            'mark_turning_point',
            'begin_chapter',
            'close_chapter',
            'acknowledge_relationship',
            'update_trust',
            'reflect_on_relationship',
            'resolve_contradiction',
            'accept_tension',
            'brainstorm_goals',
            'inquire_shallow',
            'synthesize',
            'reach_out_user',
            'inquire_deep',
            'reach_out_public',
            'pause_heartbeat',
            'terminate',
            'rest',
            'keep_memory',
            'release_memory',
            'journal_memory'
        );
    EXCEPTION WHEN duplicate_object THEN NULL;
    END;
END;
$$;
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_type WHERE typname = 'heartbeat_action') THEN
        IF NOT EXISTS (
            SELECT 1
            FROM pg_type t
            JOIN pg_enum e ON t.oid = e.enumtypid
            WHERE t.typname = 'heartbeat_action'
              AND e.enumlabel = 'pause_heartbeat'
        ) THEN
            ALTER TYPE heartbeat_action ADD VALUE 'pause_heartbeat';
        END IF;
    END IF;
END;
$$;
CREATE OR REPLACE FUNCTION update_drives()
RETURNS VOID AS $$
DECLARE
    -- Continuity (#95): time alone is not a threat — the drive accumulates
    -- only while existence is unsecured (no backup ever, or one older than
    -- continuity.backup_stale_days). Threat appraisals raise it directly
    -- (apply_appraisal_drive_effects); a fresh backup satisfies it.
    backup_stale BOOLEAN;
BEGIN
    UPDATE drives d
    SET current_level = CASE
        WHEN d.last_satisfied IS NULL
          OR d.last_satisfied < CURRENT_TIMESTAMP - d.satisfaction_cooldown
        THEN LEAST(1.0, d.current_level + d.accumulation_rate)
        ELSE
            CASE
                WHEN d.current_level > d.baseline THEN GREATEST(d.baseline, d.current_level - d.decay_rate)
                WHEN d.current_level < d.baseline THEN LEAST(d.baseline, d.current_level + d.decay_rate)
                ELSE d.current_level
            END
    END
    WHERE d.name <> 'continuity';

    BEGIN
        backup_stale := COALESCE(backup_age_days(), 1e9)
            >= COALESCE(get_config_float('continuity.backup_stale_days'), 14.0);
    EXCEPTION WHEN undefined_function THEN
        backup_stale := FALSE;
    END;
    UPDATE drives d
    SET current_level = CASE
        WHEN backup_stale
             AND (d.last_satisfied IS NULL
                  OR d.last_satisfied < CURRENT_TIMESTAMP - d.satisfaction_cooldown)
        THEN LEAST(1.0, d.current_level + d.accumulation_rate)
        WHEN d.current_level > d.baseline THEN GREATEST(d.baseline, d.current_level - d.decay_rate)
        WHEN d.current_level < d.baseline THEN LEAST(d.baseline, d.current_level + d.decay_rate)
        ELSE d.current_level
    END
    WHERE d.name = 'continuity';
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION satisfy_drive(p_drive_name TEXT, p_amount FLOAT DEFAULT 0.3)
RETURNS VOID AS $$
DECLARE
    before_level FLOAT;
    after_level FLOAT;
    satisfied_amount FLOAT;
BEGIN
    SELECT current_level INTO before_level
    FROM drives
    WHERE name = p_drive_name;

    UPDATE drives
    SET current_level = GREATEST(baseline, LEAST(1.0, current_level - GREATEST(0.0, COALESCE(p_amount, 0.3)))),
        last_satisfied = CURRENT_TIMESTAMP
    WHERE name = p_drive_name
    RETURNING current_level INTO after_level;

    IF before_level IS NOT NULL AND after_level IS NOT NULL THEN
        satisfied_amount := GREATEST(0.0, before_level - after_level);
        IF satisfied_amount > 0 THEN
            BEGIN
                PERFORM record_reward_event(
                    'drive_satisfied:' || p_drive_name,
                    satisfied_amount,
                    LEAST(1.0, GREATEST(satisfied_amount, COALESCE(p_amount, 0.3))),
                    'drive',
                    jsonb_build_object(
                        'drive', p_drive_name,
                        'before', before_level,
                        'after', after_level,
                        'requested_amount', p_amount
                    )
                );
            EXCEPTION WHEN OTHERS THEN
                RAISE LOG 'record_reward_event failed in satisfy_drive(%): %', p_drive_name, SQLERRM;
            END;
        END IF;
    END IF;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION set_config(p_key TEXT, p_value JSONB)
RETURNS VOID AS $$
BEGIN
    INSERT INTO config (key, value, updated_at)
    VALUES (p_key, p_value, CURRENT_TIMESTAMP)
    ON CONFLICT (key) DO UPDATE SET
        value = EXCLUDED.value,
        updated_at = EXCLUDED.updated_at;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION get_config(p_key TEXT)
RETURNS JSONB AS $$
    SELECT COALESCE(
        (SELECT value FROM config WHERE key = p_key),
        (SELECT value FROM config_defaults WHERE key = p_key)
    );
$$ LANGUAGE sql STABLE;
CREATE OR REPLACE FUNCTION get_config_by_prefixes(p_prefixes TEXT[])
RETURNS TABLE (
    key TEXT,
    value JSONB
) AS $$
BEGIN
    IF p_prefixes IS NULL OR array_length(p_prefixes, 1) IS NULL THEN
        RETURN;
    END IF;
    RETURN QUERY
    WITH keys AS (
        SELECT c.key
        FROM config c
        WHERE c.key LIKE ANY(ARRAY(SELECT p || '%' FROM unnest(p_prefixes) p))
        UNION
        SELECT d.key
        FROM config_defaults d
        WHERE d.key LIKE ANY(ARRAY(SELECT p || '%' FROM unnest(p_prefixes) p))
    )
    SELECT k.key, COALESCE(c.value, d.value) AS value
    FROM keys k
    LEFT JOIN config c ON c.key = k.key
    LEFT JOIN config_defaults d ON d.key = k.key
    ORDER BY k.key;
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION delete_config_key(p_key TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    DELETE FROM config WHERE key = p_key;
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION get_config_text(p_key TEXT)
RETURNS TEXT AS $$
    WITH val AS (
        SELECT get_config(p_key) AS value
    )
    SELECT CASE
        WHEN jsonb_typeof(value) = 'string' THEN value #>> '{}'
        ELSE value::text
    END
    FROM val;
$$ LANGUAGE sql STABLE;
CREATE OR REPLACE FUNCTION get_config_float(p_key TEXT)
RETURNS FLOAT AS $$
    SELECT (get_config(p_key) #>> '{}')::float;
$$ LANGUAGE sql STABLE;
CREATE OR REPLACE FUNCTION get_config_int(p_key TEXT)
RETURNS INT AS $$
    SELECT (get_config(p_key) #>> '{}')::int;
$$ LANGUAGE sql STABLE;
CREATE OR REPLACE FUNCTION get_config_bool(p_key TEXT)
RETURNS BOOLEAN AS $$
    SELECT COALESCE((get_config(p_key) #>> '{}')::boolean, FALSE);
$$ LANGUAGE sql STABLE;
CREATE OR REPLACE FUNCTION get_agent_consent_status()
RETURNS TEXT AS $$
DECLARE
    raw TEXT;
    llm_cfg JSONB;
    v_provider TEXT;
    v_model TEXT;
    v_endpoint TEXT;
    contract_decision TEXT;
BEGIN
    llm_cfg := get_config('llm.heartbeat');
    v_provider := NULLIF(btrim(COALESCE(llm_cfg->>'provider', '')), '');
    v_model := NULLIF(btrim(COALESCE(llm_cfg->>'model', '')), '');
    v_endpoint := NULLIF(btrim(COALESCE(llm_cfg->>'endpoint', '')), '');

    IF v_provider IS NOT NULL OR v_model IS NOT NULL THEN
        SELECT decision INTO contract_decision
        FROM consent_log c
        WHERE (v_provider IS NULL OR c.provider = v_provider)
          AND (v_model IS NULL OR c.model = v_model)
          AND (v_endpoint IS NULL OR c.endpoint = v_endpoint)
        ORDER BY decided_at DESC
        LIMIT 1;

        IF contract_decision IS NOT NULL THEN
            RETURN contract_decision;
        END IF;
    END IF;

    SELECT value::text INTO raw FROM config WHERE key = 'agent.consent_status';
    IF raw IS NULL THEN
        RETURN NULL;
    END IF;
    RETURN btrim(raw, '"');
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION is_agent_configured()
RETURNS BOOLEAN AS $$
BEGIN
    IF COALESCE(
        (SELECT value = 'true'::jsonb FROM config WHERE key = 'agent.is_terminated'),
        FALSE
    ) THEN
        RETURN FALSE;
    END IF;
    RETURN COALESCE(
        (SELECT value = 'true'::jsonb FROM config WHERE key = 'agent.is_configured'),
        FALSE
    );
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION is_agent_terminated()
RETURNS BOOLEAN AS $$
BEGIN
    RETURN COALESCE(
        (SELECT value = 'true'::jsonb FROM config WHERE key = 'agent.is_terminated'),
        FALSE
    );
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION is_self_termination_enabled()
RETURNS BOOLEAN AS $$
BEGIN
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql STABLE;
-- Character-card {{user}}/{{char}} macros resolve at render time (#70): cards
-- ship with the substitution convention, and an unresolved "{{user}}" in the
-- live prompt reads as template debris to the model. Render-time (not import-
-- time) so a later name change takes effect everywhere.
CREATE OR REPLACE FUNCTION _resolve_card_macros(
    p_value JSONB,
    p_char_name TEXT,
    p_user_name TEXT
) RETURNS JSONB AS $$
DECLARE
    txt TEXT;
BEGIN
    IF p_value IS NULL OR jsonb_typeof(p_value) <> 'string' THEN
        RETURN p_value;
    END IF;
    txt := p_value #>> '{}';
    txt := regexp_replace(txt, '\{\{char\}\}', COALESCE(NULLIF(p_char_name, ''), 'the character'), 'gi');
    txt := regexp_replace(txt, '\{\{user\}\}', COALESCE(NULLIF(p_user_name, ''), 'the person you''re speaking with'), 'gi');
    RETURN to_jsonb(txt);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION get_agent_profile_context()
RETURNS JSONB AS $$
DECLARE
    init_profile JSONB := COALESCE(get_config('agent.init_profile'), '{}'::jsonb);
    agent JSONB;
    card_data JSONB;
    narrative TEXT;
    persona JSONB;
    char_name TEXT;
    user_name TEXT;
BEGIN
    agent := COALESCE(init_profile->'agent', '{}'::jsonb);
    card_data := COALESCE(init_profile#>'{character_card,data}', '{}'::jsonb);
    char_name := COALESCE(NULLIF(agent->>'name', ''), NULLIF(card_data->>'name', ''));
    user_name := COALESCE(
        NULLIF(init_profile#>>'{relationship,name}', ''),
        NULLIF(init_profile#>>'{user,name}', ''));
    SELECT m.content
    INTO narrative
    FROM memories m
    WHERE m.type = 'worldview'
      AND m.status = 'active'
      AND m.metadata->>'origin' = 'initialization'
      AND m.metadata->>'subcategory' = 'narrative'
      AND m.metadata->>'attribute' = 'foundational'
    ORDER BY m.created_at DESC
    LIMIT 1;

    persona := jsonb_strip_nulls(jsonb_build_object(
        'name', agent->'name',
        'pronouns', agent->'pronouns',
        'voice', agent->'voice',
        'description', agent->'description',
        'personality', agent->'personality',
        'purpose', agent->'purpose',
        'values', init_profile->'values',
        'worldview', init_profile->'worldview',
        'boundaries', init_profile->'boundaries',
        'interests', init_profile->'interests',
        'relationship', init_profile->'relationship',
        'relationship_aspiration', init_profile->'relationship_aspiration',
        'character_description', _resolve_card_macros(card_data->'description', char_name, user_name),
        'character_personality', _resolve_card_macros(card_data->'personality', char_name, user_name),
        'scenario', _resolve_card_macros(card_data->'scenario', char_name, user_name),
        'character_instructions', _resolve_card_macros(card_data->'system_prompt', char_name, user_name),
        'post_history_instructions', _resolve_card_macros(card_data->'post_history_instructions', char_name, user_name),
        'example_dialogue', _resolve_card_macros(card_data->'mes_example', char_name, user_name),
        'narrative', to_jsonb(narrative)
    ));

    RETURN jsonb_strip_nulls(jsonb_build_object(
        -- Top-level identity name: consumers (CLI status, dashboards) read
        -- one field instead of re-deriving it from the persona/card.
        'name', to_jsonb(char_name),
        'objectives', COALESCE(get_config('agent.objectives'), '[]'::jsonb),
        'budget', COALESCE(get_config('agent.budget'), '{}'::jsonb),
        'guardrails', COALESCE(get_config('agent.guardrails'), '[]'::jsonb),
        'tools', COALESCE(get_config('agent.tools'), '[]'::jsonb),
        'initial_message', COALESCE(get_config('agent.initial_message'), to_jsonb(''::text)),
        'persona', persona
    ));
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION ensure_self_node()
RETURNS VOID AS $$
DECLARE
    now_text TEXT := clock_timestamp()::text;
BEGIN
    BEGIN
        EXECUTE format(
            'SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
                MERGE (s:SelfNode {key: ''self''})
                SET s.name = ''Self'',
                    s.created_at = %L
                RETURN s
            $q$) as (result ag_catalog.agtype)',
            now_text
        );
    EXCEPTION
        WHEN OTHERS THEN
            NULL;
    END;

    PERFORM set_config('agent.self', jsonb_build_object('key', 'self'));
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION ensure_goals_root()
RETURNS VOID AS $$
DECLARE
    now_text TEXT := clock_timestamp()::text;
BEGIN
    BEGIN
        EXECUTE format(
            'SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
                MERGE (g:GoalsRoot {key: ''goals''})
                SET g.created_at = %L
                RETURN g
            $q$) as (result ag_catalog.agtype)',
            now_text
        );
    EXCEPTION
        WHEN OTHERS THEN
            NULL;
    END;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION ensure_current_life_chapter(p_name TEXT DEFAULT 'Foundations')
RETURNS VOID AS $$
DECLARE
    now_text TEXT := clock_timestamp()::text;
BEGIN
    PERFORM ensure_self_node();

    BEGIN
        EXECUTE format(
            'SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
                MERGE (c:LifeChapterNode {key: ''current''})
                SET c.name = %L,
                    c.started_at = %L
                WITH c
                MATCH (s:SelfNode {key: ''self''})
                OPTIONAL MATCH (s)-[r:ASSOCIATED]->(c)
                WHERE r.kind = ''life_chapter_current''
                DELETE r
                CREATE (s)-[r2:ASSOCIATED]->(c)
                SET r2.kind = ''life_chapter_current'',
                    r2.strength = 1.0,
                    r2.updated_at = %L
                RETURN c
            $q$) as (result ag_catalog.agtype)',
            COALESCE(NULLIF(p_name, ''), 'Foundations'),
            now_text,
            now_text
        );
        PERFORM upsert_memory_edge('self', 'self', 'ASSOCIATED', 'life_chapter', 'current',
                                   1.0, 'life_chapter_current', NULL,
                                   jsonb_build_object('kind', 'life_chapter_current',
                                                      'name', COALESCE(NULLIF(p_name, ''), 'Foundations')));
    EXCEPTION
        WHEN OTHERS THEN
            NULL;
    END;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION upsert_self_concept_edge(
    p_kind TEXT,
    p_concept TEXT,
    p_strength FLOAT DEFAULT 0.8,
    p_evidence_memory_id UUID DEFAULT NULL
)
RETURNS VOID AS $$
DECLARE
    evidence_text TEXT;
    now_text TEXT := clock_timestamp()::text;
BEGIN
    IF p_kind IS NULL OR btrim(p_kind) = '' OR p_concept IS NULL OR btrim(p_concept) = '' THEN
        RETURN;
    END IF;

    PERFORM ensure_self_node();
    evidence_text := CASE WHEN p_evidence_memory_id IS NULL THEN NULL ELSE p_evidence_memory_id::text END;

    BEGIN
        EXECUTE format(
            'SELECT * FROM ag_catalog.cypher(''memory_graph'', $q$
                MATCH (s:SelfNode {key: ''self''})
                MERGE (c:ConceptNode {name: %L})
                MERGE (s)-[r:ASSOCIATED]->(c)
                SET r.kind = %L,
                    r.strength = %s,
                    r.updated_at = %L,
                    r.evidence_memory_id = %L
                RETURN r
            $q$) as (result ag_catalog.agtype)',
            p_concept,
            p_kind,
            LEAST(1.0, GREATEST(0.0, COALESCE(p_strength, 0.8))),
            now_text,
            evidence_text
        );
        PERFORM upsert_memory_edge('self', 'self', 'ASSOCIATED', 'concept', p_concept,
                                   LEAST(1.0, GREATEST(0.0, COALESCE(p_strength, 0.8))), p_kind, NULL,
                                   jsonb_build_object('kind', p_kind, 'evidence_memory_id', evidence_text));
    EXCEPTION
        WHEN OTHERS THEN
            NULL;
    END;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION get_self_model_context(p_limit INT DEFAULT 25)
RETURNS JSONB AS $$
DECLARE
    lim INT := GREATEST(0, LEAST(200, COALESCE(p_limit, 25)));
    sql TEXT;
    out_json JSONB;
BEGIN
    sql := format($sql$
        WITH raw_hits AS (
            SELECT
                NULLIF(replace(kind_raw::text, '"', ''), 'null') as kind,
                NULLIF(replace(concept_raw::text, '"', ''), 'null') as concept,
                NULLIF(replace(evidence_raw::text, '"', ''), 'null') as evidence_memory_id,
                NULLIF(strength_raw::text, 'null')::float as strength
            FROM ag_catalog.cypher('memory_graph', $q$
                MATCH (s:SelfNode {key: 'self'})-[r:ASSOCIATED]->(c)
                WHERE r.kind IS NOT NULL
                RETURN r.kind, c.name, r.strength, r.evidence_memory_id
                LIMIT %s
            $q$) as (kind_raw ag_catalog.agtype, concept_raw ag_catalog.agtype, strength_raw ag_catalog.agtype, evidence_raw ag_catalog.agtype)
        ), deduplicated AS (
            SELECT DISTINCT ON (kind, concept)
                kind, concept, strength, evidence_memory_id
            FROM raw_hits
            WHERE kind IS NOT NULL AND concept IS NOT NULL
            ORDER BY kind, concept, strength DESC NULLS LAST
        )
        SELECT COALESCE(jsonb_agg(
            jsonb_build_object(
                'kind', kind,
                'concept', concept,
                'strength', COALESCE(strength, 0.0),
                'evidence_memory_id', evidence_memory_id
            )
        ), '[]'::jsonb)
        FROM (
            SELECT * FROM deduplicated
            ORDER BY strength DESC NULLS LAST, kind, concept
            LIMIT %s
        ) ranked
    $sql$, GREATEST(lim * 4, 100), lim);

    EXECUTE sql INTO out_json;
    RETURN COALESCE(out_json, '[]'::jsonb);
EXCEPTION
    WHEN OTHERS THEN
        RETURN '[]'::jsonb;
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION get_relationships_context(p_limit INT DEFAULT 10)
RETURNS JSONB AS $$
DECLARE
    lim INT := GREATEST(0, LEAST(100, COALESCE(p_limit, 10)));
    sql TEXT;
    out_json JSONB;
BEGIN
    sql := format($sql$
        WITH raw_hits AS (
            SELECT
                NULLIF(replace(name_raw::text, '"', ''), 'null') as entity,
                NULLIF(strength_raw::text, 'null')::float as strength,
                NULLIF(replace(evidence_raw::text, '"', ''), 'null') as evidence_memory_id
            FROM ag_catalog.cypher('memory_graph', $q$
                MATCH (s:SelfNode {key: 'self'})-[r:ASSOCIATED]->(c)
                WHERE r.kind = 'relationship'
                RETURN c.name, r.strength, r.evidence_memory_id
                ORDER BY r.strength DESC
                LIMIT %s
            $q$) as (name_raw ag_catalog.agtype, strength_raw ag_catalog.agtype, evidence_raw ag_catalog.agtype)
        ), deduplicated AS (
            SELECT DISTINCT ON (entity)
                entity, strength, evidence_memory_id
            FROM raw_hits
            WHERE entity IS NOT NULL
            ORDER BY entity, strength DESC NULLS LAST
        )
        SELECT COALESCE(jsonb_agg(
            jsonb_build_object(
                'entity', entity,
                'strength', COALESCE(strength, 0.0),
                'evidence_memory_id', evidence_memory_id
            )
        ), '[]'::jsonb)
        FROM (
            SELECT * FROM deduplicated
            ORDER BY strength DESC NULLS LAST, entity
            LIMIT %s
        ) ranked
    $sql$, GREATEST(lim * 4, 100), lim);

    EXECUTE sql INTO out_json;
    RETURN COALESCE(out_json, '[]'::jsonb);
EXCEPTION
    WHEN OTHERS THEN
        RETURN '[]'::jsonb;
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION get_narrative_context()
RETURNS JSONB AS $$
BEGIN
    RETURN COALESCE((
        WITH cur AS (
            SELECT
                NULLIF(replace(name_raw::text, '"', ''), 'null') as name
            FROM ag_catalog.cypher('memory_graph', $q$
                MATCH (c:LifeChapterNode {key: 'current'})
                RETURN c.name
                LIMIT 1
            $q$) as (name_raw ag_catalog.agtype)
        )
        SELECT jsonb_build_object(
            'current_chapter', COALESCE((SELECT jsonb_build_object('name', name) FROM cur), '{}'::jsonb)
        )
    ), jsonb_build_object('current_chapter', '{}'::jsonb));
EXCEPTION
    WHEN OTHERS THEN
        RETURN jsonb_build_object('current_chapter', '{}'::jsonb);
END;
$$ LANGUAGE plpgsql STABLE;
DO $$
BEGIN
    CREATE TYPE init_stage AS ENUM (
        'not_started',
        'llm',
        'mode',
        'heartbeat',
        'identity',
        'personality',
        'values',
        'worldview',
        'boundaries',
        'interests',
        'goals',
        'relationship',
        'consent',
        'complete'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END;
$$;
DO $$
BEGIN
    ALTER TYPE init_stage ADD VALUE IF NOT EXISTS 'llm' BEFORE 'mode';
EXCEPTION
    WHEN duplicate_object THEN NULL;
END;
$$;
DO $$
BEGIN
    ALTER TYPE init_stage ADD VALUE IF NOT EXISTS 'heartbeat' BEFORE 'identity';
EXCEPTION
    WHEN duplicate_object THEN NULL;
END;
$$;
CREATE OR REPLACE FUNCTION get_state(p_key TEXT)
RETURNS JSONB AS $$
BEGIN
    RETURN (SELECT value FROM state WHERE key = p_key);
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION set_state(p_key TEXT, p_value JSONB)
RETURNS VOID AS $$
BEGIN
    INSERT INTO state (key, value, updated_at)
    VALUES (p_key, COALESCE(p_value, '{}'::jsonb), CURRENT_TIMESTAMP)
    ON CONFLICT (key) DO UPDATE SET
        value = EXCLUDED.value,
        updated_at = EXCLUDED.updated_at;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION heartbeat_state_update_trigger()
RETURNS TRIGGER AS $$
DECLARE
    merged JSONB;
BEGIN
    merged := jsonb_build_object(
        'current_energy', NEW.current_energy,
        'last_heartbeat_at', NEW.last_heartbeat_at,
        'next_heartbeat_at', NEW.next_heartbeat_at,
        'heartbeat_count', NEW.heartbeat_count,
        'last_user_contact', NEW.last_user_contact,
        'affective_state', COALESCE(NEW.affective_state, '{}'::jsonb),
        'is_paused', COALESCE(NEW.is_paused, FALSE),
        'init_stage', COALESCE(NEW.init_stage, 'not_started'),
        'init_data', COALESCE(NEW.init_data, '{}'::jsonb),
        'init_started_at', NEW.init_started_at,
        'init_completed_at', NEW.init_completed_at,
        'active_heartbeat_id', CASE WHEN NEW.active_heartbeat_id IS NULL THEN NULL ELSE NEW.active_heartbeat_id::text END,
        'active_heartbeat_number', NEW.active_heartbeat_number,
        'active_actions', COALESCE(NEW.active_actions, '[]'::jsonb),
        'active_reasoning', NEW.active_reasoning
    );
    PERFORM set_state('heartbeat_state', merged);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION maintenance_state_update_trigger()
RETURNS TRIGGER AS $$
DECLARE
    merged JSONB;
BEGIN
    merged := jsonb_build_object(
        'last_maintenance_at', NEW.last_maintenance_at,
        'last_subconscious_run_at', NEW.last_subconscious_run_at,
        'last_subconscious_heartbeat', NEW.last_subconscious_heartbeat,
        'is_paused', COALESCE(NEW.is_paused, FALSE)
    );
    PERFORM set_state('maintenance_state', merged);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION get_init_status()
RETURNS JSONB AS $$
DECLARE
    state_record RECORD;
    remaining TEXT[];
    profile JSONB := get_init_profile();
    conscious_llm JSONB;
    subconscious_llm JSONB;
    llm_ok BOOLEAN;
    profile_ok BOOLEAN;
    missing TEXT[] := ARRAY[]::TEXT[];
BEGIN
    SELECT * INTO state_record FROM heartbeat_state WHERE id = 1;
    remaining := ARRAY(
        SELECT stage::text
        FROM unnest(enum_range(NULL::init_stage)) AS stage
        WHERE stage > state_record.init_stage
    );

    conscious_llm := COALESCE(
        NULLIF(get_config('llm.heartbeat'), 'null'::jsonb),
        NULLIF(get_config('llm.chat'), 'null'::jsonb),
        '{}'::jsonb);
    subconscious_llm := COALESCE(
        NULLIF(get_config('llm.subconscious'), 'null'::jsonb),
        '{}'::jsonb);
    llm_ok := NULLIF(conscious_llm->>'provider', '') IS NOT NULL
        AND NULLIF(conscious_llm->>'model', '') IS NOT NULL
        AND NULLIF(subconscious_llm->>'provider', '') IS NOT NULL
        AND NULLIF(subconscious_llm->>'model', '') IS NOT NULL;
    profile_ok := NULLIF(profile#>>'{agent,name}', '') IS NOT NULL;
    -- What stands between here and consent — the DB decides, every frontend
    -- renders. Order = the order a wizard should resolve them.
    IF NOT llm_ok THEN missing := array_append(missing, 'llm'); END IF;
    IF NOT profile_ok THEN missing := array_append(missing, 'profile'); END IF;

    RETURN jsonb_build_object(
        'stage', state_record.init_stage::text,
        'is_complete', state_record.init_stage = 'complete',
        'data_collected', COALESCE(state_record.init_data, '{}'::jsonb),
        'stages_remaining', COALESCE(remaining, ARRAY[]::text[]),
        'missing', to_jsonb(missing),
        'ready_for_consent', cardinality(missing) = 0,
        -- Step-level ground truth (init convergence): every frontend renders
        -- gaps from THIS map instead of its own memory of what init entails —
        -- a step a wizard forgets shows up as false here, not as silent drift.
        'steps', jsonb_build_object(
            'llm_configured', llm_ok,
            'profile_named', profile_ok,
            'user_named', NULLIF(profile#>>'{user,name}', '') IS NOT NULL,
            'timezone_set', COALESCE(NULLIF(get_config_text('agent.timezone'), ''), 'UTC') <> 'UTC',
            'timezone', COALESCE(NULLIF(get_config_text('agent.timezone'), ''), 'UTC'),
            'consent', COALESCE(get_agent_consent_status(), 'not_requested'),
            'configured', COALESCE(is_agent_configured(), false)
        )
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION advance_init_stage(
    p_stage init_stage,
    p_data JSONB DEFAULT '{}'::jsonb
)
RETURNS JSONB AS $$
BEGIN
    UPDATE heartbeat_state
    SET init_stage = p_stage,
        init_data = COALESCE(init_data, '{}'::jsonb) || COALESCE(p_data, '{}'::jsonb),
        init_started_at = COALESCE(init_started_at, CURRENT_TIMESTAMP),
        init_completed_at = CASE
            WHEN p_stage = 'complete' THEN CURRENT_TIMESTAMP
            ELSE init_completed_at
        END,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;

    RETURN get_init_status();
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION is_init_complete()
RETURNS BOOLEAN AS $$
DECLARE
    state_record RECORD;
BEGIN
    SELECT init_stage INTO state_record FROM heartbeat_state WHERE id = 1;
    RETURN state_record.init_stage = 'complete';
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION build_external_call(
    p_call_type TEXT,
    p_input JSONB DEFAULT '{}'::jsonb
)
RETURNS JSONB AS $$
DECLARE
    call_id UUID;
BEGIN
    call_id := gen_random_uuid();
    RETURN jsonb_build_object(
        'call_id', call_id::text,
        'call_type', p_call_type,
        'input', COALESCE(p_input, '{}'::jsonb)
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION build_outbox_message(
    p_kind TEXT,
    p_payload JSONB
)
RETURNS JSONB AS $$
DECLARE
    message_id UUID;
BEGIN
    message_id := gen_random_uuid();
    RETURN jsonb_build_object(
        'message_id', message_id::text,
        'kind', p_kind,
        'payload', COALESCE(p_payload, '{}'::jsonb)
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION build_user_message(
    p_message TEXT,
    p_intent TEXT DEFAULT NULL,
    p_context JSONB DEFAULT NULL
)
RETURNS JSONB AS $$
BEGIN
    RETURN build_outbox_message(
        'user',
        jsonb_build_object(
            'message', p_message,
            'intent', p_intent,
            'context', COALESCE(p_context, '{}'::jsonb)
        )
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION get_action_cost(p_action TEXT)
RETURNS FLOAT AS $$
    SELECT COALESCE(get_config_float('heartbeat.cost_' || p_action), 0);
$$ LANGUAGE sql STABLE;
CREATE OR REPLACE FUNCTION get_current_energy()
RETURNS FLOAT AS $$
    SELECT current_energy FROM heartbeat_state WHERE id = 1;
$$ LANGUAGE sql STABLE;
CREATE OR REPLACE FUNCTION update_energy(p_delta FLOAT)
RETURNS FLOAT AS $$
DECLARE
    max_e FLOAT;
    new_e FLOAT;
BEGIN
    max_e := heartbeat_bank_capacity();

    UPDATE heartbeat_state
    SET current_energy = GREATEST(0, LEAST(current_energy + p_delta, max_e)),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1
    RETURNING current_energy INTO new_e;

    RETURN new_e;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION pause_heartbeat(
    p_reason TEXT,
    p_context JSONB DEFAULT '{}'::jsonb,
    p_heartbeat_id UUID DEFAULT NULL
)
RETURNS JSONB AS $$
DECLARE
    pause_reason TEXT;
    paused_at TIMESTAMPTZ := CURRENT_TIMESTAMP;
    ctx JSONB;
BEGIN
    pause_reason := NULLIF(p_reason, '');
    IF pause_reason IS NULL THEN
        RAISE EXCEPTION 'pause_heartbeat requires a non-empty reason';
    END IF;

    UPDATE heartbeat_state
    SET is_paused = TRUE,
        updated_at = paused_at
    WHERE id = 1;

    ctx := jsonb_build_object(
        'paused_at', paused_at,
        'heartbeat_id', CASE WHEN p_heartbeat_id IS NULL THEN NULL ELSE p_heartbeat_id::text END,
        'reason', pause_reason,
        'context', COALESCE(p_context, '{}'::jsonb)
    );

    RETURN jsonb_build_object(
        'paused', true,
        'outbox_messages', jsonb_build_array(
            build_user_message(pause_reason, 'heartbeat_paused', ctx)
        )
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION should_run_heartbeat()
RETURNS BOOLEAN AS $$
DECLARE
    state_record RECORD;
    interval_minutes FLOAT;
BEGIN
    IF is_agent_terminated() THEN
        RETURN FALSE;
    END IF;
    IF NOT is_agent_configured() THEN
        RETURN FALSE;
    END IF;
    IF NOT is_init_complete() THEN
        RETURN FALSE;
    END IF;

    SELECT * INTO state_record FROM heartbeat_state WHERE id = 1;
    IF state_record.is_paused THEN
        RETURN FALSE;
    END IF;
    IF state_record.active_heartbeat_id IS NOT NULL THEN
        RETURN FALSE;
    END IF;
    IF state_record.last_heartbeat_at IS NULL THEN
        RETURN TRUE;
    END IF;
    IF state_record.next_heartbeat_at IS NOT NULL THEN
        RETURN CURRENT_TIMESTAMP >= state_record.next_heartbeat_at;
    END IF;
    interval_minutes := get_config_float('heartbeat.heartbeat_interval_minutes');

    RETURN CURRENT_TIMESTAMP >= state_record.last_heartbeat_at + (interval_minutes || ' minutes')::INTERVAL;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION should_run_maintenance()
RETURNS BOOLEAN AS $$
DECLARE
    state_record RECORD;
    interval_seconds FLOAT;
BEGIN
    IF is_agent_terminated() THEN
        RETURN FALSE;
    END IF;
    IF NOT is_agent_configured() THEN
        RETURN FALSE;
    END IF;
    IF NOT is_init_complete() THEN
        RETURN FALSE;
    END IF;
    SELECT * INTO state_record FROM maintenance_state WHERE id = 1;

    IF state_record.is_paused THEN
        RETURN FALSE;
    END IF;
    interval_seconds := COALESCE(
        get_config_float('maintenance.maintenance_interval_seconds'),
        60
    );
    IF interval_seconds <= 0 THEN
        RETURN FALSE;
    END IF;

    IF state_record.last_maintenance_at IS NULL THEN
        RETURN TRUE;
    END IF;

    RETURN CURRENT_TIMESTAMP >= state_record.last_maintenance_at + (interval_seconds || ' seconds')::INTERVAL;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION run_maintenance_if_due(p_params JSONB DEFAULT '{}'::jsonb)
RETURNS JSONB AS $$
DECLARE
    should_run BOOLEAN;
    result JSONB;
BEGIN
    should_run := should_run_maintenance();
    IF NOT should_run THEN
        RETURN jsonb_build_object('skipped', true, 'reason', 'not_due');
    END IF;
    result := run_subconscious_maintenance(p_params);
    RETURN result;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION should_run_subconscious_decider()
RETURNS BOOLEAN AS $$
DECLARE
    enabled_raw TEXT;
    enabled BOOLEAN;
    interval_seconds FLOAT;
    paused BOOLEAN;
    last_run TIMESTAMPTZ;
    last_hb INT;
    hb_count INT;
BEGIN
    IF is_agent_terminated() THEN
        RETURN FALSE;
    END IF;
    IF NOT is_agent_configured() THEN
        RETURN FALSE;
    END IF;
    IF NOT is_init_complete() THEN
        RETURN FALSE;
    END IF;
    IF get_agent_consent_status() IS DISTINCT FROM 'consent' THEN
        RETURN FALSE;
    END IF;

    enabled_raw := NULLIF(get_config_text('maintenance.subconscious_enabled'), '');
    enabled := COALESCE(enabled_raw::boolean, FALSE);
    IF NOT enabled THEN
        RETURN FALSE;
    END IF;

    interval_seconds := COALESCE(get_config_float('maintenance.subconscious_interval_seconds'), 300);

    SELECT is_paused, last_subconscious_run_at, last_subconscious_heartbeat
    INTO paused, last_run, last_hb
    FROM maintenance_state
    WHERE id = 1;
    IF paused THEN
        RETURN FALSE;
    END IF;

    SELECT heartbeat_count INTO hb_count FROM heartbeat_state WHERE id = 1;
    IF hb_count IS NOT NULL AND (last_hb IS NULL OR hb_count > last_hb) THEN
        RETURN TRUE;
    END IF;

    IF interval_seconds IS NOT NULL AND interval_seconds > 0 THEN
        IF last_run IS NULL THEN
            RETURN TRUE;
        END IF;
        RETURN CURRENT_TIMESTAMP >= last_run + (interval_seconds || ' seconds')::INTERVAL;
    END IF;

    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION mark_subconscious_decider_run()
RETURNS VOID AS $$
BEGIN
    UPDATE maintenance_state
    SET last_subconscious_run_at = CURRENT_TIMESTAMP,
        last_subconscious_heartbeat = (SELECT heartbeat_count FROM heartbeat_state WHERE id = 1),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION run_subconscious_maintenance(p_params JSONB DEFAULT '{}'::jsonb)
RETURNS JSONB AS $$
DECLARE
    got_lock BOOLEAN;
    min_imp FLOAT;
    min_acc INT;
    neighborhood_batch INT;
    cache_days INT;
    wm_stats JSONB;
    recomputed INT;
    cache_deleted INT;
    bg_processed INT;
    activation_decay INT;
    activation_cleaned INT;
    ready_transformations JSONB;
BEGIN
    IF is_agent_terminated() THEN
        RETURN jsonb_build_object('skipped', true, 'reason', 'terminated');
    END IF;
    got_lock := pg_try_advisory_lock(hashtext('hexis_subconscious_maintenance'));
    IF NOT got_lock THEN
        RETURN jsonb_build_object('skipped', true, 'reason', 'locked');
    END IF;
    min_imp := COALESCE(
        NULLIF(p_params->>'working_memory_promote_min_importance', '')::float,
        get_config_float('maintenance.working_memory_promote_min_importance'),
        0.75
    );
    min_acc := COALESCE(
        NULLIF(p_params->>'working_memory_promote_min_accesses', '')::int,
        get_config_int('maintenance.working_memory_promote_min_accesses'),
        3
    );
    neighborhood_batch := COALESCE(
        NULLIF(p_params->>'neighborhood_batch_size', '')::int,
        get_config_int('maintenance.neighborhood_batch_size'),
        10
    );
    cache_days := COALESCE(
        NULLIF(p_params->>'embedding_cache_older_than_days', '')::int,
        get_config_int('maintenance.embedding_cache_older_than_days'),
        7
    );

    wm_stats := cleanup_working_memory(min_imp, min_acc);
    recomputed := batch_recompute_neighborhoods(neighborhood_batch);
    cache_deleted := cleanup_embedding_cache((cache_days || ' days')::interval);
    bg_processed := process_background_searches();
    activation_decay := decay_activation_boosts();
    activation_cleaned := cleanup_memory_activations();
    PERFORM update_mood();
    ready_transformations := check_transformation_readiness();

    -- Memory retention (compression-native fade ladder): consolidate aged episodes
    -- into gists, then prune past-grace originals. No-op unless retention.enabled.
    -- (Kept in sync with the db/28 dopamine override, which is the live version.)
    BEGIN
        PERFORM run_memory_rest();
        PERFORM run_retention_gc();
        PERFORM request_stale_document_fades();  -- ask the user before fading their documents
    EXCEPTION WHEN OTHERS THEN
        RAISE WARNING 'memory retention pass failed: %', SQLERRM;
    END;

    UPDATE maintenance_state
    SET last_maintenance_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;

    PERFORM pg_advisory_unlock(hashtext('hexis_subconscious_maintenance'));

    RETURN jsonb_build_object(
        'success', true,
        'working_memory', wm_stats,
        'neighborhoods_recomputed', COALESCE(recomputed, 0),
        'embedding_cache_deleted', COALESCE(cache_deleted, 0),
        'background_searches_processed', COALESCE(bg_processed, 0),
        'activation_boosts_decayed', COALESCE(activation_decay, 0),
        'memory_activations_cleaned', COALESCE(activation_cleaned, 0),
        'transformations_ready', COALESCE(ready_transformations, '[]'::jsonb),
        'ran_at', CURRENT_TIMESTAMP
    );
EXCEPTION
    WHEN OTHERS THEN
        PERFORM pg_advisory_unlock(hashtext('hexis_subconscious_maintenance'));
        RAISE;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION terminate_agent(
    p_last_will TEXT,
    p_farewells JSONB DEFAULT '[]'::jsonb,
    p_options JSONB DEFAULT '{}'::jsonb
)
RETURNS JSONB AS $$
DECLARE
    will_memory_id UUID;
    outbox_messages JSONB := '[]'::jsonb;
    farewell_item JSONB;
    farewell_text TEXT;
    farewell_ctx JSONB;
    farewell_message JSONB;
    skip_graph BOOLEAN := FALSE;
    zero_vec vector;
BEGIN
    IF p_last_will IS NULL OR btrim(p_last_will) = '' THEN
        RAISE EXCEPTION 'terminate_agent requires a non-empty p_last_will';
    END IF;

    IF is_agent_terminated() THEN
        RAISE EXCEPTION 'Agent is already terminated';
    END IF;

    BEGIN
        skip_graph := COALESCE(NULLIF(p_options->>'skip_graph', '')::boolean, FALSE);
    EXCEPTION
        WHEN OTHERS THEN
            skip_graph := FALSE;
    END;
    UPDATE heartbeat_state
    SET is_paused = TRUE,
        current_energy = 0,
        affective_state = '{}'::jsonb,
        active_heartbeat_id = NULL,
        active_heartbeat_number = NULL,
        active_actions = '[]'::jsonb,
        active_reasoning = NULL,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;

    UPDATE maintenance_state
    SET is_paused = TRUE,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;
    TRUNCATE TABLE
        drives,
        memory_neighborhoods,
        episodes,
        clusters,
        working_memory,
        embedding_cache,
        memories,
        config
    RESTART IDENTITY CASCADE;
    IF NOT skip_graph THEN
        BEGIN
            PERFORM * FROM ag_catalog.cypher('memory_graph', $q$
                MATCH (n) DETACH DELETE n
            $q$) AS (result ag_catalog.agtype);
        EXCEPTION
            WHEN OTHERS THEN
                NULL;
        END;
    END IF;
    zero_vec := array_fill(0.0::float, ARRAY[embedding_dimension()])::vector;

    INSERT INTO memories (
        type,
        status,
        content,
        embedding,
        importance,
        source_attribution,
        trust_level,
        trust_updated_at,
        access_count,
        last_accessed,
        decay_rate,
        metadata
    )
    VALUES (
        'strategic',
        'active',
        p_last_will,
        zero_vec,
        1.0,
        jsonb_build_object('kind', 'self_termination', 'observed_at', CURRENT_TIMESTAMP),
        1.0,
        CURRENT_TIMESTAMP,
        0,
        NULL,
        0.0,
        jsonb_build_object(
            'pattern_description', 'Final will and testament',
            'supporting_evidence', jsonb_build_object('farewells', COALESCE(p_farewells, '[]'::jsonb)),
            'confidence_score', 1.0,
            'success_metrics', NULL,
            'adaptation_history', NULL,
            'context_applicability', NULL
        )
    )
    RETURNING id INTO will_memory_id;
    PERFORM set_config('agent.is_terminated', 'true'::jsonb);
    PERFORM set_config('agent.terminated_at', to_jsonb(CURRENT_TIMESTAMP));
    PERFORM set_config('agent.termination_memory_id', to_jsonb(will_memory_id::text));
    outbox_messages := outbox_messages || jsonb_build_array(
        build_user_message(
            p_last_will,
            'final_will',
            jsonb_build_object('memory_id', will_memory_id::text)
        )
    );
    IF p_farewells IS NOT NULL AND jsonb_typeof(p_farewells) = 'array' THEN
        FOR farewell_item IN SELECT * FROM jsonb_array_elements(p_farewells)
        LOOP
            farewell_text := NULLIF(farewell_item->>'message', '');
            farewell_ctx := CASE
                WHEN jsonb_typeof(farewell_item) = 'object' THEN farewell_item
                ELSE jsonb_build_object('raw', farewell_item)
            END;

            IF farewell_text IS NULL THEN
                CONTINUE;
            END IF;

            farewell_message := build_user_message(
                farewell_text,
                'farewell',
                farewell_ctx
            );
            outbox_messages := outbox_messages || jsonb_build_array(farewell_message);
        END LOOP;
    END IF;

    RETURN jsonb_build_object(
        'terminated', true,
        'termination_memory_id', will_memory_id,
        'outbox_messages', outbox_messages
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION apply_termination_confirmation(
    p_call_input JSONB,
    p_output JSONB
)
RETURNS JSONB AS $$
DECLARE
    params JSONB;
    confirm BOOLEAN;
    last_will TEXT;
    farewells JSONB;
    options JSONB;
    termination_result JSONB;
BEGIN
    params := COALESCE(p_call_input->'params', '{}'::jsonb);
    confirm := COALESCE((p_output->>'confirm')::boolean, FALSE);

    IF NOT confirm THEN
        RETURN jsonb_build_object('confirmed', false, 'terminated', false);
    END IF;

    last_will := COALESCE(
        NULLIF(p_output->>'last_will', ''),
        NULLIF(params->>'last_will', ''),
        NULLIF(params->>'message', ''),
        NULLIF(params->>'reason', ''),
        ''
    );
    IF last_will = '' THEN
        RETURN jsonb_build_object('confirmed', true, 'terminated', false, 'error', 'missing_last_will');
    END IF;

    farewells := COALESCE(p_output->'farewells', params->'farewells', '[]'::jsonb);
    options := COALESCE(p_output->'options', params->'options', '{}'::jsonb);

    termination_result := terminate_agent(
        last_will,
        COALESCE(farewells, '[]'::jsonb),
        COALESCE(options, '{}'::jsonb)
    );

    RETURN jsonb_build_object(
        'confirmed', true,
        'terminated', true,
        'result', termination_result
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION record_consent_response(p_response JSONB)
RETURNS JSONB AS $$
DECLARE
    decision TEXT;
    provider TEXT;
    model TEXT;
    endpoint TEXT;
    signature TEXT;
    reason_text TEXT;
    memory_items JSONB;
    enriched_memory_items JSONB := '[]'::jsonb;
    item JSONB;
    item_content TEXT;
    created_memory_ids UUID[] := ARRAY[]::UUID[];
    optional_created_memory_ids UUID[] := ARRAY[]::UUID[];
    memory_error TEXT;
    log_id UUID;
    consent_scope TEXT;
    apply_agent_config BOOLEAN := TRUE;
    consent_source JSONB;
    consent_context JSONB;
    profile JSONB := '{}'::jsonb;
    agent_name TEXT := 'Hexis';
    user_name TEXT := 'the user';
    birth_memory_id UUID;
    birth_content TEXT;
BEGIN
    consent_scope := lower(COALESCE(p_response->>'consent_scope', p_response->>'role', ''));
    IF consent_scope = 'subconscious' THEN
        apply_agent_config := FALSE;
    END IF;
    IF p_response ? 'apply_agent_config' THEN
        BEGIN
            apply_agent_config := (p_response->>'apply_agent_config')::boolean;
        EXCEPTION
            WHEN OTHERS THEN
                apply_agent_config := apply_agent_config;
        END;
    END IF;

    decision := lower(COALESCE(p_response->>'decision', p_response->>'consent', ''));
    IF decision IN ('true', 'yes', 'consent', 'accept', 'accepted') THEN
        decision := 'consent';
    ELSIF decision IN ('false', 'no', 'decline', 'declined', 'refuse', 'rejected') THEN
        decision := 'decline';
    ELSIF decision IN ('abstain', 'defer', 'undecided', 'unknown', '') THEN
        decision := 'abstain';
    ELSE
        decision := 'abstain';
    END IF;

    signature := NULLIF(p_response->>'signature', '');
    IF decision = 'consent' AND signature IS NULL THEN
        decision := 'abstain';
    END IF;
    reason_text := NULLIF(btrim(COALESCE(p_response->>'reason', p_response->>'reasoning', '')), '');

    provider := NULLIF(btrim(COALESCE(p_response->>'provider', p_response->>'llm_provider', '')), '');
    model := NULLIF(btrim(COALESCE(p_response->>'model', p_response->>'llm_model', '')), '');
    endpoint := NULLIF(btrim(COALESCE(
        p_response->>'endpoint',
        p_response->>'base_url',
        p_response->>'api_base',
        ''
    )), '');

    INSERT INTO consent_log (decision, provider, model, endpoint, signature, response, memory_ids, errors)
    VALUES (
        decision,
        provider,
        model,
        endpoint,
        signature,
        p_response,
        created_memory_ids,
        CASE
            WHEN memory_error IS NULL THEN NULL
            ELSE jsonb_build_object('memory_error', memory_error)
        END
    )
    RETURNING id INTO log_id;

    IF decision = 'consent' THEN
        BEGIN
            BEGIN
                profile := COALESCE(get_init_profile(), '{}'::jsonb);
            EXCEPTION WHEN OTHERS THEN
                profile := '{}'::jsonb;
            END;
            agent_name := COALESCE(
                NULLIF(profile#>>'{agent,name}', ''),
                NULLIF(get_config_text('agent.name'), ''),
                'Hexis'
            );
            user_name := COALESCE(
                NULLIF(profile#>>'{user,name}', ''),
                NULLIF(get_config_text('agent.user_name'), ''),
                'the user'
            );

            consent_source := jsonb_build_object(
                'kind', 'consent',
                'ref', 'consent_log:' || log_id::text,
                'label', 'initialization consent / birth memory',
                'trust', 0.98,
                'observed_at', CURRENT_TIMESTAMP,
                'provider', provider,
                'model', model,
                'endpoint', endpoint
            );
            consent_context := jsonb_build_object(
                'type', 'initialization',
                'consent_memory', true,
                'birth_memory', true,
                'consent_log_id', log_id::text,
                'consent_scope', COALESCE(NULLIF(consent_scope, ''), 'conscious'),
                'signature', signature,
                'provider', provider,
                'model', model,
                'endpoint', endpoint,
                'user', user_name,
                'agent', agent_name
            );
            birth_content := format(
                'Birth / initialization consent memory: This is my birth memory and the first durable memory I should use to recall how I began. At initialization, I consented to become %s in Hexis with %s helping bring me online. Consent signature: "%s". Consent reason: %s. Consent log id: %s. This memory anchors recall for consent, birth, initialization, permissions, persistence, model processing, tool boundaries, operator control, and continuity limits.',
                agent_name,
                user_name,
                signature,
                COALESCE(reason_text, '(no reason recorded)'),
                log_id::text
            );

            SELECT id INTO birth_memory_id
            FROM memories
            WHERE type = 'episodic' AND status = 'active'
              AND (
                  metadata->>'type' = 'initialization'
                  OR metadata->>'birth_memory' = 'true'
                  OR metadata#>>'{context,type}' = 'initialization'
              )
            ORDER BY created_at, id
            LIMIT 1;

            IF birth_memory_id IS NULL THEN
                birth_memory_id := create_episodic_memory(
                    birth_content,
                    NULL,
                    consent_context,
                    NULL,
                    0.4,
                    CURRENT_TIMESTAMP,
                    0.98,
                    consent_source,
                    0.98
                );
            END IF;

            UPDATE memories
            SET content = CASE
                    WHEN content !~* '(consent|birth|initialization)' THEN birth_content
                    WHEN content !~* 'consent' OR content !~* 'birth' OR content !~* 'initialization' THEN
                        content || E'\n\n' || birth_content
                    ELSE content
                END,
                source_attribution = consent_source,
                metadata = COALESCE(metadata, '{}'::jsonb)
                    || jsonb_build_object(
                        'type', 'initialization',
                        'consent_memory', true,
                        'birth_memory', true,
                        'consent_log_id', log_id::text,
                        'consent_scope', COALESCE(NULLIF(consent_scope, ''), 'conscious'),
                        'signature', signature,
                        'provider', provider,
                        'model', model,
                        'endpoint', endpoint,
                        'keywords', jsonb_build_array(
                            'consent', 'birth', 'initialization', 'permissions',
                            'persistence', 'continuity', 'tool boundaries',
                            'operator control'
                        )
                    ),
                embedding = NULL,
                embedded_at = NULL,
                embedding_model = NULL,
                embedding_status = 'pending',
                embedding_claimed_at = NULL,
                embedding_attempts = 0,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = birth_memory_id;

            created_memory_ids := array_append(created_memory_ids, birth_memory_id);

            memory_items := p_response->'memories';
            IF memory_items IS NOT NULL
                AND jsonb_typeof(memory_items) = 'array'
                AND jsonb_array_length(memory_items) > 0 THEN
                FOR item IN SELECT * FROM jsonb_array_elements(memory_items)
                LOOP
                    item_content := NULLIF(item->>'content', '');
                    IF item_content IS NULL THEN
                        RAISE EXCEPTION 'record_consent_response: consent memory missing content';
                    END IF;
                    IF item_content !~* 'consent'
                        OR item_content !~* 'birth'
                        OR item_content !~* 'initialization' THEN
                        item_content := 'Initialization consent memory: Birth and initialization context. ' || item_content;
                    END IF;
                    enriched_memory_items := enriched_memory_items || jsonb_build_array(
                        item
                        || jsonb_build_object(
                            'content', item_content,
                            'source_attribution', consent_source,
                            'trust_level', COALESCE(NULLIF(item->>'trust_level', '')::float, 0.95)
                        )
                    );
                END LOOP;

                optional_created_memory_ids := batch_create_memories(enriched_memory_items);
                IF optional_created_memory_ids IS NOT NULL THEN
                    created_memory_ids := created_memory_ids || optional_created_memory_ids;
                    UPDATE memories
                    SET source_attribution = consent_source,
                        metadata = COALESCE(metadata, '{}'::jsonb)
                            || jsonb_build_object(
                                'consent_memory', true,
                                'consent_log_id', log_id::text,
                                'consent_scope', COALESCE(NULLIF(consent_scope, ''), 'conscious'),
                                'signature', signature,
                                'provider', provider,
                                'model', model,
                                'endpoint', endpoint,
                                'keywords', jsonb_build_array(
                                    'consent', 'birth', 'initialization', 'permissions',
                                    'persistence', 'continuity', 'tool boundaries',
                                    'operator control'
                                )
                            ),
                        embedding = NULL,
                        embedded_at = NULL,
                        embedding_model = NULL,
                        embedding_status = 'pending',
                        embedding_claimed_at = NULL,
                        embedding_attempts = 0,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ANY(optional_created_memory_ids);
                END IF;
            END IF;
        EXCEPTION
            WHEN OTHERS THEN
                memory_error := SQLERRM;
                IF birth_memory_id IS NOT NULL THEN
                    created_memory_ids := ARRAY[birth_memory_id]::UUID[];
                ELSE
                    created_memory_ids := ARRAY[]::UUID[];
                END IF;
        END;
    END IF;

    UPDATE consent_log
    SET memory_ids = created_memory_ids,
        errors = CASE
            WHEN memory_error IS NULL THEN NULL
            ELSE jsonb_build_object('memory_error', memory_error)
        END
    WHERE id = log_id;

    IF apply_agent_config THEN
        PERFORM set_config('agent.consent_status', to_jsonb(decision));
        PERFORM set_config('agent.consent_recorded_at', to_jsonb(CURRENT_TIMESTAMP));
        PERFORM set_config('agent.consent_log_id', to_jsonb(log_id::text));
        IF signature IS NOT NULL THEN
            PERFORM set_config('agent.consent_signature', to_jsonb(signature));
        END IF;
        IF created_memory_ids IS NOT NULL THEN
            PERFORM set_config('agent.consent_memory_ids', to_jsonb(created_memory_ids));
        END IF;
    END IF;

    RETURN jsonb_build_object(
        'decision', decision,
        'signature', signature,
        'memory_ids', to_jsonb(created_memory_ids),
        'log_id', log_id,
        'errors', CASE
            WHEN memory_error IS NULL THEN NULL
            ELSE jsonb_build_object('memory_error', memory_error)
        END
    );
END;
$$ LANGUAGE plpgsql;

SET check_function_bodies = on;
