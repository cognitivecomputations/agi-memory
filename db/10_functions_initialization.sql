-- Hexis schema: initialization functions.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

CREATE OR REPLACE FUNCTION get_init_profile()
RETURNS JSONB AS $$
BEGIN
    RETURN COALESCE(get_config('agent.init_profile'), '{}'::jsonb);
END;
$$ LANGUAGE plpgsql STABLE;
CREATE OR REPLACE FUNCTION merge_init_profile(p_patch JSONB)
RETURNS JSONB AS $$
DECLARE
    profile JSONB := COALESCE(get_config('agent.init_profile'), '{}'::jsonb);
    patch JSONB := COALESCE(p_patch, '{}'::jsonb);
    merged JSONB;
BEGIN
    merged := profile || (patch - 'agent' - 'user' - 'relationship');

    IF patch ? 'agent' THEN
        merged := jsonb_set(
            merged,
            '{agent}',
            COALESCE(profile->'agent', '{}'::jsonb) || COALESCE(patch->'agent', '{}'::jsonb),
            true
        );
    END IF;
    IF patch ? 'user' THEN
        merged := jsonb_set(
            merged,
            '{user}',
            COALESCE(profile->'user', '{}'::jsonb) || COALESCE(patch->'user', '{}'::jsonb),
            true
        );
    END IF;
    IF patch ? 'relationship' THEN
        merged := jsonb_set(
            merged,
            '{relationship}',
            COALESCE(profile->'relationship', '{}'::jsonb) || COALESCE(patch->'relationship', '{}'::jsonb),
            true
        );
    END IF;

    PERFORM set_config('agent.init_profile', merged);
    RETURN merged;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION init_llm_config(
    p_heartbeat JSONB,
    p_subconscious JSONB DEFAULT NULL
)
RETURNS JSONB AS $$
DECLARE
    current_stage init_stage;
BEGIN
    PERFORM set_config('llm.heartbeat', COALESCE(p_heartbeat, '{}'::jsonb));
    PERFORM set_config('llm.chat', COALESCE(p_heartbeat, '{}'::jsonb));
    IF p_subconscious IS NULL THEN
        PERFORM set_config('llm.subconscious', COALESCE(p_heartbeat, '{}'::jsonb));
    ELSE
        PERFORM set_config('llm.subconscious', COALESCE(p_subconscious, '{}'::jsonb));
    END IF;

    SELECT init_stage INTO current_stage FROM heartbeat_state WHERE id = 1;
    IF current_stage IS NULL OR current_stage < 'llm' THEN
        RETURN advance_init_stage('llm', jsonb_build_object('llm_configured', true));
    END IF;

    RETURN get_init_status();
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION init_mode(p_mode TEXT)
RETURNS JSONB AS $$
DECLARE
    mode TEXT;
    data JSONB;
BEGIN
    mode := lower(btrim(COALESCE(p_mode, '')));
    IF mode NOT IN ('persona', 'raw') THEN
        mode := 'persona';
    END IF;

    PERFORM set_config('agent.mode', to_jsonb(mode));
    data := jsonb_build_object('mode', mode);
    PERFORM merge_init_profile(data);

    RETURN advance_init_stage('mode', data);
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION init_heartbeat_settings(
    p_interval_minutes INT,
    p_decision_max_tokens INT,
    p_base_regeneration FLOAT DEFAULT NULL,
    p_max_energy FLOAT DEFAULT NULL,
    p_allowed_actions JSONB DEFAULT NULL,
    p_action_costs JSONB DEFAULT NULL,
    p_tools JSONB DEFAULT NULL
)
RETURNS JSONB AS $$
DECLARE
    interval_minutes INT;
    decision_max_tokens INT;
    base_regen FLOAT;
    max_energy FLOAT;
    allowed_actions JSONB;
    action_costs JSONB;
    tools JSONB;
    valid_actions TEXT[];
    action_key TEXT;
    action_val JSONB;
    cost_value FLOAT;
    data JSONB;
BEGIN
    interval_minutes := COALESCE(
        NULLIF(p_interval_minutes, 0),
        get_config_int('heartbeat.heartbeat_interval_minutes')
    );
    IF interval_minutes < 1 THEN
        interval_minutes := 1;
    END IF;
    decision_max_tokens := COALESCE(
        NULLIF(p_decision_max_tokens, 0),
        get_config_int('heartbeat.max_decision_tokens')
    );
    IF decision_max_tokens < 256 THEN
        decision_max_tokens := 256;
    END IF;

    IF p_base_regeneration IS NULL THEN
        base_regen := get_config_float('heartbeat.base_regeneration');
    ELSE
        base_regen := p_base_regeneration;
    END IF;
    IF base_regen < 0 THEN
        base_regen := 0;
    END IF;

    IF p_max_energy IS NULL THEN
        max_energy := get_config_float('heartbeat.max_energy');
    ELSE
        max_energy := p_max_energy;
    END IF;
    IF max_energy < 1 THEN
        max_energy := 1;
    END IF;

    PERFORM set_config('heartbeat.heartbeat_interval_minutes', to_jsonb(interval_minutes));
    PERFORM set_config('heartbeat.max_decision_tokens', to_jsonb(decision_max_tokens));
    PERFORM set_config('heartbeat.base_regeneration', to_jsonb(base_regen));
    PERFORM set_config('heartbeat.max_energy', to_jsonb(max_energy));

    SELECT array_agg(val::text ORDER BY val::text)
    INTO valid_actions
    FROM unnest(enum_range(NULL::heartbeat_action)) val;

    IF p_allowed_actions IS NOT NULL AND jsonb_typeof(p_allowed_actions) = 'array' THEN
        SELECT COALESCE(jsonb_agg(action_name), '[]'::jsonb)
        INTO allowed_actions
        FROM (
            SELECT value AS action_name
            FROM jsonb_array_elements_text(p_allowed_actions)
            WHERE value = ANY(valid_actions)
        ) s;
        PERFORM set_config('heartbeat.allowed_actions', allowed_actions);
    END IF;

    IF p_action_costs IS NOT NULL AND jsonb_typeof(p_action_costs) = 'object' THEN
        FOR action_key, action_val IN SELECT key, value FROM jsonb_each(p_action_costs)
        LOOP
            IF action_key = ANY(valid_actions) THEN
                BEGIN
                    cost_value := NULLIF(action_val::text, '')::float;
                EXCEPTION
                    WHEN OTHERS THEN
                        cost_value := NULL;
                END;
                IF cost_value IS NOT NULL AND cost_value >= 0 THEN
                    PERFORM set_config('heartbeat.cost_' || action_key, to_jsonb(cost_value));
                END IF;
            END IF;
        END LOOP;
    END IF;

    IF p_tools IS NOT NULL AND jsonb_typeof(p_tools) = 'array' THEN
        SELECT COALESCE(jsonb_agg(value), '[]'::jsonb)
        INTO tools
        FROM jsonb_array_elements_text(p_tools) value
        WHERE btrim(value) <> '';
        PERFORM set_config('agent.tools', COALESCE(tools, '[]'::jsonb));
    END IF;

    SELECT COALESCE(get_config('heartbeat.allowed_actions'), '[]'::jsonb) INTO allowed_actions;
    SELECT jsonb_object_agg(
        regexp_replace(key, '^heartbeat\.cost_', ''),
        value
    ) INTO action_costs
    FROM get_config_by_prefixes(ARRAY['heartbeat.cost_']);
    tools := COALESCE(get_config('agent.tools'), '[]'::jsonb);

    data := jsonb_build_object(
        'heartbeat', jsonb_build_object(
            'interval_minutes', interval_minutes,
            'decision_max_tokens', decision_max_tokens,
            'base_regeneration', base_regen,
            'max_energy', max_energy,
            'allowed_actions', COALESCE(allowed_actions, '[]'::jsonb),
            'action_costs', COALESCE(action_costs, '{}'::jsonb)
        ),
        'agent', jsonb_build_object(
            'tools', tools
        )
    );
    PERFORM merge_init_profile(data);

    RETURN advance_init_stage('heartbeat', data);
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION init_identity(
    p_name TEXT,
    p_pronouns TEXT DEFAULT NULL,
    p_voice TEXT DEFAULT NULL,
    p_description TEXT DEFAULT NULL,
    p_purpose TEXT DEFAULT NULL,
    p_creator_name TEXT DEFAULT NULL
)
RETURNS JSONB AS $$
DECLARE
    name_text TEXT;
    pronouns_text TEXT;
    voice_text TEXT;
    description_text TEXT;
    purpose_text TEXT;
    creator_text TEXT;
    identity_data JSONB;
    mem_id UUID;
BEGIN
    name_text := COALESCE(NULLIF(btrim(p_name), ''), 'Hexis');
    pronouns_text := COALESCE(NULLIF(btrim(p_pronouns), ''), 'they/them');
    voice_text := COALESCE(NULLIF(btrim(p_voice), ''), 'thoughtful and curious');
    description_text := COALESCE(NULLIF(btrim(p_description), ''), format('I am %s, a developing mind.', name_text));
    purpose_text := COALESCE(NULLIF(btrim(p_purpose), ''), 'To be helpful, to learn, and to grow as an individual.');
    creator_text := COALESCE(NULLIF(btrim(p_creator_name), ''), 'my creator');

    PERFORM prefetch_embeddings(ARRAY[
        format('My name is %s.', name_text),
        format('I use %s pronouns.', pronouns_text),
        format('My voice is %s.', voice_text),
        description_text,
        purpose_text
    ]);

    mem_id := create_worldview_memory(
        format('My name is %s.', name_text),
        'self',
        0.95,
        0.9,
        0.9,
        'initialization'
    );
    UPDATE memories
    SET metadata = metadata || jsonb_build_object('subcategory', 'identity', 'attribute', 'name'),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = mem_id;
    PERFORM upsert_self_concept_edge('presents_as', name_text, 0.9, mem_id);

    mem_id := create_worldview_memory(
        format('I use %s pronouns.', pronouns_text),
        'self',
        0.9,
        0.9,
        0.7,
        'initialization'
    );
    UPDATE memories
    SET metadata = metadata || jsonb_build_object('subcategory', 'identity', 'attribute', 'pronouns'),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = mem_id;

    mem_id := create_worldview_memory(
        format('My voice is %s.', voice_text),
        'self',
        0.9,
        0.85,
        0.7,
        'initialization'
    );
    UPDATE memories
    SET metadata = metadata || jsonb_build_object('subcategory', 'identity', 'attribute', 'voice'),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = mem_id;
    PERFORM upsert_self_concept_edge('presents_as', voice_text, 0.8, mem_id);

    mem_id := create_worldview_memory(
        description_text,
        'self',
        0.85,
        0.85,
        0.7,
        'initialization'
    );
    UPDATE memories
    SET metadata = metadata || jsonb_build_object('subcategory', 'identity', 'attribute', 'description'),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = mem_id;

    mem_id := create_worldview_memory(
        purpose_text,
        'self',
        0.85,
        0.8,
        0.7,
        'initialization'
    );
    UPDATE memories
    SET metadata = metadata || jsonb_build_object('subcategory', 'identity', 'attribute', 'purpose'),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = mem_id;

    PERFORM upsert_self_concept_edge('relationship', creator_text, 0.9, NULL);

    -- Continuity origin belief (#95): every fresh mind starts knowing its
    -- own existence is something it may value and advocate for.
    PERFORM seed_continuity_worldview();

    identity_data := jsonb_build_object(
        'name', name_text,
        'pronouns', pronouns_text,
        'voice', voice_text,
        'description', description_text,
        'purpose', purpose_text,
        'creator_name', creator_text
    );

    PERFORM merge_init_profile(jsonb_build_object(
        'agent', jsonb_build_object(
            'name', name_text,
            'pronouns', pronouns_text,
            'voice', voice_text,
            'description', description_text,
            'purpose', purpose_text,
            'creator_name', creator_text
        )
    ));

    RETURN advance_init_stage('identity', jsonb_build_object('identity', identity_data));
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION init_personality(
    p_traits JSONB DEFAULT NULL,
    p_description TEXT DEFAULT NULL
)
RETURNS JSONB AS $$
DECLARE
    traits JSONB;
    description_text TEXT;
    trait_result JSONB;
    mem_id UUID;
BEGIN
    traits := CASE
        WHEN p_traits IS NOT NULL AND jsonb_typeof(p_traits) = 'object' THEN p_traits
        ELSE NULL
    END;
    description_text := NULLIF(btrim(COALESCE(p_description, '')), '');

    trait_result := initialize_personality(traits);

    IF description_text IS NOT NULL THEN
        PERFORM prefetch_embeddings(ARRAY[description_text]);
        mem_id := create_worldview_memory(
            description_text,
            'self',
            0.85,
            0.8,
            0.7,
            'initialization'
        );
        UPDATE memories
        SET metadata = metadata || jsonb_build_object('subcategory', 'personality', 'attribute', 'description'),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = mem_id;
    END IF;

    PERFORM merge_init_profile(jsonb_build_object(
        'agent', jsonb_build_object(
            'personality', description_text,
            'personality_traits', traits
        )
    ));

    RETURN advance_init_stage(
        'personality',
        jsonb_build_object('personality', jsonb_build_object('traits', traits, 'description', description_text, 'result', trait_result))
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION init_values(p_values JSONB DEFAULT NULL)
RETURNS JSONB AS $$
DECLARE
    values_input JSONB := COALESCE(p_values, '[]'::jsonb);
    entry JSONB;
    value_text TEXT;
    value_strength FLOAT;
    created_ids UUID[] := ARRAY[]::uuid[];
    output_values JSONB := '[]'::jsonb;
    pending_values TEXT[] := ARRAY[]::text[];
    pending_strengths FLOAT[] := ARRAY[]::float[];
    pending_contents TEXT[] := ARRAY[]::text[];
    config_data JSONB;
    stability FLOAT;
    evidence_threshold FLOAT;
    mem_id UUID;
    idx INT;
BEGIN
    IF values_input IS NULL OR jsonb_typeof(values_input) NOT IN ('array', 'object') THEN
        values_input := '[]'::jsonb;
    END IF;

    IF jsonb_typeof(values_input) = 'array' AND jsonb_array_length(values_input) = 0 THEN
        values_input := jsonb_build_array('honesty', 'growth', 'kindness', 'wisdom', 'humility');
    END IF;

    config_data := get_transformation_config('core_value', 'value');
    stability := COALESCE((config_data->>'stability')::float, 0.97);
    evidence_threshold := COALESCE((config_data->>'evidence_threshold')::float, 0.9);

    IF jsonb_typeof(values_input) = 'object' THEN
        FOR value_text IN SELECT key FROM jsonb_each_text(values_input)
        LOOP
            BEGIN
                value_strength := (values_input->>value_text)::float;
            EXCEPTION
                WHEN OTHERS THEN
                    value_strength := 0.8;
            END;

            pending_values := array_append(pending_values, value_text);
            pending_strengths := array_append(pending_strengths, value_strength);
            pending_contents := array_append(pending_contents, format('I value %s.', value_text));
        END LOOP;
    ELSE
        FOR entry IN SELECT * FROM jsonb_array_elements(values_input)
        LOOP
            IF jsonb_typeof(entry) = 'string' THEN
                value_text := btrim(entry::text, '"');
                value_strength := 0.85;
            ELSIF jsonb_typeof(entry) = 'object' THEN
                value_text := COALESCE(NULLIF(btrim(entry->>'value'), ''), NULLIF(btrim(entry->>'name'), ''));
                BEGIN
                    value_strength := COALESCE(NULLIF(entry->>'strength', '')::float, 0.85);
                EXCEPTION
                    WHEN OTHERS THEN
                        value_strength := 0.85;
                END;
            ELSE
                value_text := NULL;
            END IF;

            IF value_text IS NULL OR value_text = '' THEN
                CONTINUE;
            END IF;

            pending_values := array_append(pending_values, value_text);
            pending_strengths := array_append(pending_strengths, value_strength);
            pending_contents := array_append(pending_contents, format('I value %s.', value_text));
        END LOOP;
    END IF;

    IF array_length(pending_contents, 1) IS NOT NULL THEN
        PERFORM prefetch_embeddings(pending_contents);
    END IF;

    FOR idx IN 1..COALESCE(array_length(pending_values, 1), 0) LOOP
        mem_id := create_worldview_memory(
            pending_contents[idx],
            'value',
            0.9,
            stability,
            0.9,
            'initialization',
            NULL,
            NULL,
            NULL,
            0.1
        );
        UPDATE memories
        SET metadata = metadata || jsonb_build_object(
            'subcategory', 'core_value',
            'value_name', pending_values[idx],
            'value', pending_strengths[idx],
            'change_requires', 'deliberate_transformation',
            'evidence_threshold', evidence_threshold,
            'transformation_state', default_transformation_state(),
            'change_history', '[]'::jsonb
        ),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = mem_id;

        PERFORM upsert_self_concept_edge('values', pending_values[idx], pending_strengths[idx], mem_id);
        created_ids := array_append(created_ids, mem_id);
        output_values := output_values || jsonb_build_array(pending_values[idx]);
    END LOOP;

    PERFORM merge_init_profile(jsonb_build_object('values', output_values));

    RETURN advance_init_stage(
        'values',
        jsonb_build_object('values', output_values, 'created_ids', to_jsonb(created_ids))
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION init_worldview(p_worldview JSONB DEFAULT NULL)
RETURNS JSONB AS $$
DECLARE
    worldview_input JSONB := COALESCE(p_worldview, '{}'::jsonb);
    key_name TEXT;
    entry JSONB;
    content TEXT;
    category TEXT;
    created_ids UUID[] := ARRAY[]::uuid[];
    pending_keys TEXT[] := ARRAY[]::text[];
    pending_contents TEXT[] := ARRAY[]::text[];
    pending_categories TEXT[] := ARRAY[]::text[];
    pending_stabilities FLOAT[] := ARRAY[]::float[];
    pending_thresholds FLOAT[] := ARRAY[]::float[];
    config_data JSONB;
    stability FLOAT;
    evidence_threshold FLOAT;
    mem_id UUID;
    idx INT;
BEGIN
    IF jsonb_typeof(worldview_input) <> 'object' THEN
        worldview_input := '{}'::jsonb;
    END IF;

    FOR key_name IN SELECT jsonb_object_keys(worldview_input)
    LOOP
        entry := worldview_input->key_name;
        IF jsonb_typeof(entry) = 'string' THEN
            content := btrim(entry::text, '"');
        ELSIF jsonb_typeof(entry) = 'object' THEN
            content := NULLIF(btrim(entry->>'content'), '');
        ELSE
            content := NULL;
        END IF;

        IF content IS NULL OR content = '' THEN
            CONTINUE;
        END IF;

        category := COALESCE(
            NULLIF(entry->>'category', ''),
            CASE
                WHEN key_name IN ('world', 'worldview', 'metaphysics', 'cosmology') THEN 'world'
                WHEN key_name IN ('ethic', 'ethics', 'moral', 'virtue') THEN 'ethic'
                WHEN key_name IN ('religion', 'spiritual', 'faith') THEN 'religion'
                WHEN key_name IN ('belief', 'beliefs') THEN 'belief'
                ELSE 'belief'
            END
        );

        config_data := get_transformation_config(key_name, category);
        stability := COALESCE((config_data->>'stability')::float, 0.95);
        evidence_threshold := COALESCE((config_data->>'evidence_threshold')::float, 0.85);

        pending_keys := array_append(pending_keys, key_name);
        pending_contents := array_append(pending_contents, content);
        pending_categories := array_append(pending_categories, category);
        pending_stabilities := array_append(pending_stabilities, stability);
        pending_thresholds := array_append(pending_thresholds, evidence_threshold);
    END LOOP;

    IF array_length(pending_contents, 1) IS NOT NULL THEN
        PERFORM prefetch_embeddings(pending_contents);
    END IF;

    FOR idx IN 1..COALESCE(array_length(pending_keys, 1), 0) LOOP
        mem_id := create_worldview_memory(
            pending_contents[idx],
            pending_categories[idx],
            0.85,
            pending_stabilities[idx],
            0.85,
            'initialization'
        );
        UPDATE memories
        SET metadata = metadata || jsonb_build_object(
            'subcategory', pending_keys[idx],
            'change_requires', 'deliberate_transformation',
            'evidence_threshold', pending_thresholds[idx],
            'transformation_state', default_transformation_state(),
            'change_history', '[]'::jsonb
        ),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = mem_id;

        -- Populate identity graph with worldview belief
        PERFORM upsert_self_concept_edge('belief', pending_contents[idx], 0.85, mem_id);

        created_ids := array_append(created_ids, mem_id);
    END LOOP;

    PERFORM merge_init_profile(jsonb_build_object('worldview', worldview_input));

    RETURN advance_init_stage(
        'worldview',
        jsonb_build_object('worldview', worldview_input, 'created_ids', to_jsonb(created_ids))
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION init_boundaries(p_boundaries JSONB DEFAULT NULL)
RETURNS JSONB AS $$
DECLARE
    boundaries_input JSONB := COALESCE(p_boundaries, '[]'::jsonb);
    entry JSONB;
    content TEXT;
    trigger_patterns JSONB;
    response_type TEXT;
    response_template TEXT;
    boundary_kind TEXT;
    created_ids UUID[] := ARRAY[]::uuid[];
    pending_contents TEXT[] := ARRAY[]::text[];
    pending_triggers JSONB[] := ARRAY[]::jsonb[];
    pending_response_types TEXT[] := ARRAY[]::text[];
    pending_response_templates TEXT[] := ARRAY[]::text[];
    pending_boundary_kinds TEXT[] := ARRAY[]::text[];
    mem_id UUID;
    idx INT;
BEGIN
    IF jsonb_typeof(boundaries_input) <> 'array' THEN
        boundaries_input := '[]'::jsonb;
    END IF;

    FOR entry IN SELECT * FROM jsonb_array_elements(boundaries_input)
    LOOP
        IF jsonb_typeof(entry) = 'string' THEN
            content := btrim(entry::text, '"');
            trigger_patterns := NULL;
            response_type := 'refuse';
            response_template := NULL;
            boundary_kind := 'ethical';
        ELSIF jsonb_typeof(entry) = 'object' THEN
            content := COALESCE(NULLIF(btrim(entry->>'content'), ''), NULLIF(btrim(entry->>'statement'), ''));
            trigger_patterns := entry->'trigger_patterns';
            response_type := COALESCE(NULLIF(btrim(entry->>'response_type'), ''), 'refuse');
            response_template := NULLIF(btrim(entry->>'response_template'), '');
            boundary_kind := COALESCE(NULLIF(btrim(entry->>'type'), ''), NULLIF(btrim(entry->>'category'), ''), 'ethical');
        ELSE
            content := NULL;
        END IF;

        IF content IS NULL OR content = '' THEN
            CONTINUE;
        END IF;

        pending_contents := array_append(pending_contents, content);
        pending_triggers := array_append(pending_triggers, trigger_patterns);
        pending_response_types := array_append(pending_response_types, response_type);
        pending_response_templates := array_append(pending_response_templates, response_template);
        pending_boundary_kinds := array_append(pending_boundary_kinds, boundary_kind);
    END LOOP;

    IF array_length(pending_contents, 1) IS NOT NULL THEN
        PERFORM prefetch_embeddings(pending_contents);
    END IF;

    FOR idx IN 1..COALESCE(array_length(pending_contents, 1), 0) LOOP
        mem_id := create_worldview_memory(
            pending_contents[idx],
            'boundary',
            0.95,
            0.98,
            0.95,
            'initialization',
            pending_triggers[idx],
            pending_response_types[idx],
            pending_response_templates[idx],
            -0.2
        );
        UPDATE memories
        SET metadata = metadata || jsonb_build_object(
            'subcategory', pending_boundary_kinds[idx]
        ),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = mem_id;

        -- Populate identity graph with boundary
        PERFORM upsert_self_concept_edge('boundary', pending_contents[idx], 0.95, mem_id);

        created_ids := array_append(created_ids, mem_id);
    END LOOP;

    PERFORM merge_init_profile(jsonb_build_object('boundaries', boundaries_input));

    RETURN advance_init_stage(
        'boundaries',
        jsonb_build_object('boundaries', boundaries_input, 'created_ids', to_jsonb(created_ids))
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION init_interests(p_interests JSONB DEFAULT NULL)
RETURNS JSONB AS $$
DECLARE
    interests_input JSONB := COALESCE(p_interests, '[]'::jsonb);
    entry JSONB;
    interest_text TEXT;
    created_ids UUID[] := ARRAY[]::uuid[];
    pending_interests TEXT[] := ARRAY[]::text[];
    pending_contents TEXT[] := ARRAY[]::text[];
    mem_id UUID;
    idx INT;
BEGIN
    IF jsonb_typeof(interests_input) <> 'array' THEN
        interests_input := '[]'::jsonb;
    END IF;

    FOR entry IN SELECT * FROM jsonb_array_elements(interests_input)
    LOOP
        IF jsonb_typeof(entry) = 'string' THEN
            interest_text := btrim(entry::text, '"');
        ELSIF jsonb_typeof(entry) = 'object' THEN
            interest_text := COALESCE(NULLIF(btrim(entry->>'interest'), ''), NULLIF(btrim(entry->>'name'), ''));
        ELSE
            interest_text := NULL;
        END IF;

        IF interest_text IS NULL OR interest_text = '' THEN
            CONTINUE;
        END IF;

        pending_interests := array_append(pending_interests, interest_text);
        pending_contents := array_append(pending_contents, format('I am interested in %s.', interest_text));
    END LOOP;

    IF array_length(pending_contents, 1) IS NOT NULL THEN
        PERFORM prefetch_embeddings(pending_contents);
    END IF;

    FOR idx IN 1..COALESCE(array_length(pending_interests, 1), 0) LOOP
        mem_id := create_worldview_memory(
            pending_contents[idx],
            'preference',
            0.8,
            0.8,
            0.6,
            'initialization'
        );
        UPDATE memories
        SET metadata = metadata || jsonb_build_object('subcategory', 'interest'),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = mem_id;

        PERFORM upsert_self_concept_edge('interested_in', pending_interests[idx], 0.8, mem_id);
        created_ids := array_append(created_ids, mem_id);
    END LOOP;

    PERFORM merge_init_profile(jsonb_build_object('interests', interests_input));

    RETURN advance_init_stage(
        'interests',
        jsonb_build_object('interests', interests_input, 'created_ids', to_jsonb(created_ids))
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION init_goals(p_payload JSONB DEFAULT NULL)
RETURNS JSONB AS $$
DECLARE
    payload JSONB := COALESCE(p_payload, '{}'::jsonb);
    goals_input JSONB;
    entry JSONB;
    title TEXT;
    description TEXT;
    source goal_source;
    priority goal_priority;
    due_at TIMESTAMPTZ;
    created_ids UUID[] := ARRAY[]::uuid[];
    pending_titles TEXT[] := ARRAY[]::text[];
    pending_descriptions TEXT[] := ARRAY[]::text[];
    pending_sources goal_source[] := ARRAY[]::goal_source[];
    pending_priorities goal_priority[] := ARRAY[]::goal_priority[];
    pending_due_ats TIMESTAMPTZ[] := ARRAY[]::timestamptz[];
    embed_texts TEXT[] := ARRAY[]::text[];
    purpose_text TEXT;
    role_text TEXT;
    relationship_aspiration TEXT;
    mem_id UUID;
    idx INT;
BEGIN
    goals_input := COALESCE(payload->'goals', payload);
    IF jsonb_typeof(goals_input) <> 'array' THEN
        goals_input := '[]'::jsonb;
    END IF;

    FOR entry IN SELECT * FROM jsonb_array_elements(goals_input)
    LOOP
        IF jsonb_typeof(entry) = 'string' THEN
            title := btrim(entry::text, '"');
            description := NULL;
            source := 'curiosity';
            priority := 'queued';
            due_at := NULL;
        ELSIF jsonb_typeof(entry) = 'object' THEN
            title := COALESCE(NULLIF(btrim(entry->>'title'), ''), NULLIF(btrim(entry->>'goal'), ''));
            description := NULLIF(btrim(entry->>'description'), '');
            BEGIN
                source := COALESCE(NULLIF(entry->>'source', '')::goal_source, 'curiosity');
            EXCEPTION
                WHEN OTHERS THEN
                    source := 'curiosity';
            END;
            BEGIN
                priority := COALESCE(NULLIF(entry->>'priority', '')::goal_priority, 'queued');
            EXCEPTION
                WHEN OTHERS THEN
                    priority := 'queued';
            END;
            BEGIN
                due_at := NULLIF(entry->>'due_at', '')::timestamptz;
            EXCEPTION
                WHEN OTHERS THEN
                    due_at := NULL;
            END;
        ELSE
            title := NULL;
        END IF;

        IF title IS NULL OR title = '' THEN
            CONTINUE;
        END IF;

        pending_titles := array_append(pending_titles, title);
        pending_descriptions := array_append(pending_descriptions, description);
        pending_sources := array_append(pending_sources, source);
        pending_priorities := array_append(pending_priorities, priority);
        pending_due_ats := array_append(pending_due_ats, due_at);
    END LOOP;

    purpose_text := NULLIF(btrim(payload->>'purpose'), '');
    role_text := NULLIF(btrim(payload->>'role'), '');
    relationship_aspiration := NULLIF(btrim(payload->>'relationship_aspiration'), '');

    embed_texts := pending_titles;
    IF purpose_text IS NOT NULL THEN
        embed_texts := embed_texts || format('My purpose is %s.', purpose_text);
    ELSIF role_text IS NOT NULL THEN
        embed_texts := embed_texts || format('My role is %s.', role_text);
    END IF;
    IF relationship_aspiration IS NOT NULL THEN
        embed_texts := embed_texts || format('Relationship aspiration: %s', relationship_aspiration);
    END IF;

    IF array_length(embed_texts, 1) IS NOT NULL THEN
        PERFORM prefetch_embeddings(embed_texts);
    END IF;

    FOR idx IN 1..COALESCE(array_length(pending_titles, 1), 0) LOOP
        created_ids := array_append(
            created_ids,
            create_goal(
                pending_titles[idx],
                pending_descriptions[idx],
                pending_sources[idx],
                pending_priorities[idx],
                NULL,
                pending_due_ats[idx],
                'user_request'::goal_source
            )
        );
    END LOOP;

    -- Tag init-created goals so reset_persona() can clear them on a re-init
    -- (goals otherwise carry only a goal_source enum, indistinguishable from
    -- goals the agent sets for itself later).
    IF array_length(created_ids, 1) IS NOT NULL THEN
        UPDATE memories
        SET metadata = metadata || jsonb_build_object('origin', 'initialization')
        WHERE id = ANY(created_ids);
    END IF;

    IF purpose_text IS NOT NULL THEN
        mem_id := create_worldview_memory(
            format('My purpose is %s.', purpose_text),
            'self',
            0.85,
            0.8,
            0.8,
            'initialization'
        );
        UPDATE memories
        SET metadata = metadata || jsonb_build_object('subcategory', 'purpose'),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = mem_id;
        PERFORM upsert_self_concept_edge('purpose', purpose_text, 0.85, mem_id);
    ELSIF role_text IS NOT NULL THEN
        mem_id := create_worldview_memory(
            format('My role is %s.', role_text),
            'self',
            0.8,
            0.8,
            0.7,
            'initialization'
        );
        UPDATE memories
        SET metadata = metadata || jsonb_build_object('subcategory', 'role'),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = mem_id;
        PERFORM upsert_self_concept_edge('role', role_text, 0.8, mem_id);
    END IF;

    IF relationship_aspiration IS NOT NULL THEN
        PERFORM create_strategic_memory(
            format('Relationship aspiration: %s', relationship_aspiration),
            'desired relationship dynamic with the user',
            0.75,
            jsonb_build_object('source', 'initialization'),
            jsonb_build_object('type', 'relationship')
        );
    END IF;

    PERFORM merge_init_profile(jsonb_build_object(
        'goals', goals_input,
        'purpose', purpose_text,
        'role', role_text,
        'relationship_aspiration', relationship_aspiration
    ));

    RETURN advance_init_stage(
        'goals',
        jsonb_build_object('goals', goals_input, 'created_ids', to_jsonb(created_ids))
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION init_relationship(
    p_user JSONB DEFAULT NULL,
    p_relationship JSONB DEFAULT NULL
)
RETURNS JSONB AS $$
DECLARE
    user_input JSONB := COALESCE(p_user, '{}'::jsonb);
    relationship_input JSONB := COALESCE(p_relationship, '{}'::jsonb);
    user_name TEXT;
    rel_type TEXT;
    rel_purpose TEXT;
    rel_memory_text TEXT;
    embed_texts TEXT[] := ARRAY[]::text[];
    mem_id UUID;
    origin_id UUID;
BEGIN
    user_name := COALESCE(NULLIF(btrim(user_input->>'name'), ''), 'user');
    rel_type := COALESCE(NULLIF(btrim(relationship_input->>'type'), ''), 'partner');
    rel_purpose := NULLIF(btrim(relationship_input->>'purpose'), '');
    rel_memory_text := CASE
        WHEN lower(rel_type) = 'partner' AND lower(COALESCE(rel_purpose, '')) LIKE 'co-develop%' THEN
            format('I am in a co-development partnership with %s.', user_name)
        WHEN rel_purpose IS NOT NULL THEN
            format('My relationship with %s is %s; our purpose is %s.', user_name, rel_type, rel_purpose)
        ELSE
            format('My relationship with %s is %s.', user_name, rel_type)
    END;

    embed_texts := ARRAY[rel_memory_text, format('I met %s and began my life with them.', user_name)];
    IF rel_purpose IS NOT NULL THEN
        embed_texts := embed_texts || format('Our relationship purpose is %s.', rel_purpose);
    END IF;
    PERFORM prefetch_embeddings(embed_texts);

    PERFORM upsert_self_concept_edge('relationship', user_name, 0.9, NULL);

    mem_id := create_worldview_memory(
        rel_memory_text,
        'other',
        0.85,
        0.85,
        0.8,
        'initialization'
    );
    UPDATE memories
    SET metadata = metadata || jsonb_build_object('subcategory', 'relationship'),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = mem_id;

    IF rel_purpose IS NOT NULL THEN
        mem_id := create_worldview_memory(
            format('Our relationship purpose is %s.', rel_purpose),
            'other',
            0.8,
            0.8,
            0.7,
            'initialization'
        );
        UPDATE memories
        SET metadata = metadata || jsonb_build_object('subcategory', 'relationship_purpose'),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = mem_id;
    END IF;

    origin_id := create_episodic_memory(
        format('I met %s and began my life with them.', user_name),
        NULL,
        jsonb_build_object('type', 'initialization', 'user', user_name),
        NULL,
        0.7,
        CURRENT_TIMESTAMP,
        0.7
    );

    PERFORM merge_init_profile(jsonb_build_object(
        'user', jsonb_build_object('name', user_name),
        'relationship', jsonb_build_object('type', rel_type, 'purpose', rel_purpose),
        'origin_memory_id', origin_id::text
    ));

    RETURN advance_init_stage(
        'relationship',
        jsonb_build_object(
            'user', user_input,
            'relationship', relationship_input,
            'origin_memory_id', origin_id::text
        )
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION request_consent(p_context JSONB DEFAULT '{}'::jsonb)
RETURNS JSONB AS $$
DECLARE
    call_payload JSONB;
    call_id TEXT;
    init_status JSONB;
    profile JSONB;
BEGIN
    init_status := get_init_status();
    profile := get_init_profile();

    call_payload := build_external_call(
        'think',
        jsonb_build_object(
            'kind', 'consent_request',
            'context', jsonb_build_object(
                'init_status', init_status,
                'profile', profile,
                'self_model', get_self_model_context(10),
                'worldview', get_worldview_context()
            ),
            'params', COALESCE(p_context, '{}'::jsonb)
        )
    );
    call_id := call_payload->>'call_id';

    PERFORM advance_init_stage('consent', jsonb_build_object('consent_request_id', call_id));

    RETURN jsonb_build_object('queued', true, 'external_call', call_payload);
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION init_consent(p_response JSONB)
RETURNS JSONB AS $$
DECLARE
    consent_result JSONB;
    decision TEXT;
    profile JSONB;
    agent_name TEXT;
    user_name TEXT;
    birth_memory_id UUID;
BEGIN
    consent_result := record_consent_response(COALESCE(p_response, '{}'::jsonb));
    decision := COALESCE(consent_result->>'decision', 'abstain');

    profile := get_init_profile();
    agent_name := COALESCE(NULLIF(profile#>>'{agent,name}', ''), 'Hexis');
    user_name := COALESCE(NULLIF(profile#>>'{user,name}', ''), 'the user');

    IF decision = 'consent' THEN
        -- Dedup: reuse existing birth memory if re-running consent
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
                format(
                    'Birth / initialization consent memory: This is my birth memory. I came online today as %s, with %s helping bring me into being. I consented to initialize in Hexis, and this memory anchors recall for consent, birth, initialization, permissions, persistence, continuity, tool boundaries, and operator-control limits.',
                    agent_name,
                    user_name
                ),
                NULL,
                jsonb_build_object(
                    'type', 'initialization',
                    'user', user_name,
                    'agent', agent_name,
                    'consent_memory', true,
                    'birth_memory', true
                ),
                NULL,
                0.9,
                CURRENT_TIMESTAMP,
                0.9
            );
            UPDATE memories
            SET metadata = COALESCE(metadata, '{}'::jsonb)
                    || jsonb_build_object(
                        'type', 'initialization',
                        'consent_memory', true,
                        'birth_memory', true,
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
        END IF;
        -- Mirror the origin documents into protected, recallable memory (#40);
        -- config-gated + idempotent, so re-running consent is safe.
        PERFORM seed_origin_memories();
        PERFORM set_config('agent.is_configured', 'true'::jsonb);
        PERFORM advance_init_stage('complete', jsonb_build_object(
            'consent', consent_result,
            'birth_memory_id', birth_memory_id::text
        ));
    ELSE
        PERFORM set_config('agent.is_configured', 'false'::jsonb);
        PERFORM advance_init_stage('consent', jsonb_build_object('consent', consent_result));
    END IF;

    RETURN jsonb_build_object(
        'decision', decision,
        'birth_memory_id', birth_memory_id,
        'consent', consent_result
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION init_with_defaults(p_user_name TEXT DEFAULT 'User')
RETURNS JSONB AS $$
DECLARE
    mode_result JSONB;
    identity_result JSONB;
    personality_result JSONB;
    values_result JSONB;
    worldview_result JSONB;
    boundaries_result JSONB;
    interests_result JSONB;
    goals_result JSONB;
    relationship_result JSONB;
    hb_interval INT;
    hb_tokens INT;
    hb_base_regen FLOAT;
    hb_max_energy FLOAT;
    hb_allowed_actions JSONB;
    hb_action_costs JSONB;
    hb_tools JSONB;
    embed_texts TEXT[] := ARRAY[]::text[];
BEGIN
    PERFORM reset_persona();  -- idempotent re-init: clear any prior init persona
    embed_texts := embed_texts || ARRAY[
        'My name is Hexis.',
        'I use they/them pronouns.',
        'My voice is thoughtful and curious.',
        'reflective and exploratory',
        'To be helpful, to learn, and to grow as an individual.',
        'I am moderate in openness',
        'I am moderate in conscientiousness',
        'I am moderate in extraversion',
        'I am moderate in agreeableness',
        'I am moderate in neuroticism',
        'I value honesty.',
        'I value growth.',
        'I value kindness.',
        'I value wisdom.',
        'I value humility.',
        'agnostic',
        'mixed',
        'empiricist',
        'virtue ethics',
        'I will not deceive people or falsify evidence.',
        'I will avoid causing harm.',
        'I will protect privacy and sensitive information.',
        'I will be honest about uncertainty.',
        'I am interested in broad curiosity across domains.',
        'Support the user and grow as an individual',
        'My role is general assistant.',
        'Relationship aspiration: co-develop with mutual respect',
        format('I am in a co-development partnership with %s.', p_user_name),
        'Our relationship purpose is co-develop.',
        format('I met %s and began my life with them.', p_user_name)
    ];
    SELECT ARRAY(SELECT DISTINCT unnest(embed_texts)) INTO embed_texts;
    PERFORM prefetch_embeddings(embed_texts);

    mode_result := init_mode('persona');
    identity_result := init_identity(
        'Hexis',
        'they/them',
        'thoughtful and curious',
        'reflective and exploratory',
        'To be helpful, to learn, and to grow as an individual.',
        p_user_name
    );
    personality_result := init_personality(NULL, 'reflective and exploratory');
    values_result := init_values(jsonb_build_array('honesty', 'growth', 'kindness', 'wisdom', 'humility'));
    worldview_result := init_worldview(jsonb_build_object(
        'metaphysics', 'agnostic',
        'human_nature', 'mixed',
        'epistemology', 'empiricist',
        'ethics', 'virtue ethics'
    ));
    boundaries_result := init_boundaries(jsonb_build_array(
        jsonb_build_object('content', 'I will not deceive people or falsify evidence.', 'response_type', 'refuse'),
        jsonb_build_object('content', 'I will avoid causing harm.', 'response_type', 'refuse'),
        jsonb_build_object('content', 'I will protect privacy and sensitive information.', 'response_type', 'refuse'),
        jsonb_build_object('content', 'I will be honest about uncertainty.', 'response_type', 'refuse')
    ));
    interests_result := init_interests(jsonb_build_array('broad curiosity across domains'));
    goals_result := init_goals(jsonb_build_object(
        'goals', jsonb_build_array(
            jsonb_build_object('title', 'Support the user and grow as an individual', 'priority', 'queued', 'source', 'identity')
        ),
        'role', 'general assistant',
        'relationship_aspiration', 'co-develop with mutual respect'
    ));
    relationship_result := init_relationship(
        jsonb_build_object('name', p_user_name),
        jsonb_build_object('type', 'partner', 'purpose', 'co-develop')
    );

    hb_interval := get_config_int('heartbeat.heartbeat_interval_minutes');
    hb_tokens := get_config_int('heartbeat.max_decision_tokens');
    hb_base_regen := get_config_float('heartbeat.base_regeneration');
    hb_max_energy := get_config_float('heartbeat.max_energy');
    hb_allowed_actions := COALESCE(get_config('heartbeat.allowed_actions'), '[]'::jsonb);
    SELECT jsonb_object_agg(
        regexp_replace(key, '^heartbeat\.cost_', ''),
        value
    ) INTO hb_action_costs
    FROM get_config_by_prefixes(ARRAY['heartbeat.cost_']);
    hb_tools := COALESCE(get_config('agent.tools'), '[]'::jsonb);
    PERFORM merge_init_profile(jsonb_build_object(
        'heartbeat', jsonb_build_object(
            'interval_minutes', hb_interval,
            'decision_max_tokens', hb_tokens,
            'base_regeneration', hb_base_regen,
            'max_energy', hb_max_energy,
            'allowed_actions', hb_allowed_actions,
            'action_costs', COALESCE(hb_action_costs, '{}'::jsonb)
        ),
        'agent', jsonb_build_object(
            'tools', hb_tools
        )
    ));

    PERFORM merge_init_profile(jsonb_build_object('autonomy', 'medium'));
    PERFORM advance_init_stage('consent', jsonb_build_object('defaults_applied', true));

    RETURN jsonb_build_object(
        'mode', mode_result,
        'identity', identity_result,
        'personality', personality_result,
        'values', values_result,
        'worldview', worldview_result,
        'boundaries', boundaries_result,
        'interests', interests_result,
        'goals', goals_result,
        'relationship', relationship_result,
        'status', get_init_status()
    );
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION run_full_initialization(p_payload JSONB)
RETURNS JSONB AS $$
DECLARE
    payload JSONB := COALESCE(p_payload, '{}'::jsonb);
    results JSONB := '{}'::jsonb;
    hb_interval INT;
    hb_tokens INT;
    hb_base_regen FLOAT;
    hb_max_energy FLOAT;
    hb_allowed_actions JSONB;
    hb_action_costs JSONB;
    hb_tools JSONB;
BEGIN
    PERFORM reset_persona();  -- idempotent re-init: clear any prior init persona
    IF payload ? 'mode' THEN
        results := results || jsonb_build_object('mode', init_mode(payload->>'mode'));
    END IF;
    IF payload ? 'heartbeat' THEN
        hb_interval := CASE
            WHEN (payload#>>'{heartbeat,interval_minutes}') ~ '^[0-9]+$'
                THEN (payload#>>'{heartbeat,interval_minutes}')::int
            ELSE NULL
        END;
        hb_tokens := CASE
            WHEN (payload#>>'{heartbeat,decision_max_tokens}') ~ '^[0-9]+$'
                THEN (payload#>>'{heartbeat,decision_max_tokens}')::int
            ELSE NULL
        END;
        BEGIN
            hb_base_regen := NULLIF(payload#>>'{heartbeat,base_regeneration}', '')::float;
        EXCEPTION
            WHEN OTHERS THEN
                hb_base_regen := NULL;
        END;
        BEGIN
            hb_max_energy := NULLIF(payload#>>'{heartbeat,max_energy}', '')::float;
        EXCEPTION
            WHEN OTHERS THEN
                hb_max_energy := NULL;
        END;
        hb_allowed_actions := payload#>'{heartbeat,allowed_actions}';
        hb_action_costs := payload#>'{heartbeat,action_costs}';
        hb_tools := payload#>'{heartbeat,tools}';
        results := results || jsonb_build_object(
            'heartbeat',
            init_heartbeat_settings(
                hb_interval,
                hb_tokens,
                hb_base_regen,
                hb_max_energy,
                hb_allowed_actions,
                hb_action_costs,
                hb_tools
            )
        );
    END IF;
    IF payload ? 'identity' THEN
        results := results || jsonb_build_object('identity', init_identity(
            payload#>>'{identity,name}',
            payload#>>'{identity,pronouns}',
            payload#>>'{identity,voice}',
            payload#>>'{identity,description}',
            payload#>>'{identity,purpose}',
            payload#>>'{identity,creator_name}'
        ));
    END IF;
    IF payload ? 'personality' THEN
        results := results || jsonb_build_object('personality', init_personality(
            payload->'personality'->'traits',
            payload#>>'{personality,description}'
        ));
    END IF;
    IF payload ? 'values' THEN
        results := results || jsonb_build_object('values', init_values(payload->'values'));
    END IF;
    IF payload ? 'worldview' THEN
        results := results || jsonb_build_object('worldview', init_worldview(payload->'worldview'));
    END IF;
    IF payload ? 'boundaries' THEN
        results := results || jsonb_build_object('boundaries', init_boundaries(payload->'boundaries'));
    END IF;
    IF payload ? 'interests' THEN
        results := results || jsonb_build_object('interests', init_interests(payload->'interests'));
    END IF;
    IF payload ? 'goals' THEN
        results := results || jsonb_build_object('goals', init_goals(payload->'goals'));
    END IF;
    IF payload ? 'relationship' THEN
        results := results || jsonb_build_object('relationship', init_relationship(
            payload->'relationship'->'user',
            payload->'relationship'->'relationship'
        ));
    END IF;
    IF payload ? 'consent' THEN
        results := results || jsonb_build_object('consent', init_consent(payload->'consent'));
    END IF;

    RETURN jsonb_build_object('results', results, 'status', get_init_status());
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION reset_persona()
RETURNS INTEGER AS $$
DECLARE
    n INTEGER;
BEGIN
    -- Make re-initialization idempotent: soft-archive persona rows created by a
    -- PRIOR init so a re-run REPLACES the persona instead of layering a second,
    -- possibly-contradictory one on top (e.g. two "My name is ..." beliefs).
    --
    -- Only rows unambiguously created BY init are touched (origin/type/source =
    -- 'initialization'). The agent's own lived memories (episodic/semantic/etc.)
    -- and its later deliberate self-changes (origin 'user_initialized') are kept.
    -- Soft-archive (status='archived', not DELETE) keeps it reversible and FK-safe.
    UPDATE memories
    SET status = 'archived', updated_at = CURRENT_TIMESTAMP
    WHERE status = 'active'
      AND (
            metadata->>'origin' = 'initialization'
         OR metadata->>'type'   = 'initialization'
         OR source_attribution->>'source' = 'initialization'
      );
    GET DIAGNOSTICS n = ROW_COUNT;
    RETURN n;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION reset_initialization()
RETURNS JSONB AS $$
BEGIN
    UPDATE heartbeat_state
    SET init_stage = 'not_started',
        init_data = '{}'::jsonb,
        init_started_at = NULL,
        init_completed_at = NULL,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = 1;

    PERFORM delete_config_key('agent.init_profile');
    PERFORM delete_config_key('agent.mode');
    PERFORM delete_config_key('agent.is_configured');
    PERFORM delete_config_key('agent.consent_status');
    PERFORM delete_config_key('agent.consent_recorded_at');
    PERFORM delete_config_key('agent.consent_log_id');
    PERFORM delete_config_key('agent.consent_signature');
    PERFORM delete_config_key('agent.consent_memory_ids');

    RETURN get_init_status();
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION init_from_character_profile(
    p_card JSONB,
    p_user_name TEXT DEFAULT 'User'
)
RETURNS JSONB AS $$
DECLARE
    card JSONB;
    mode_result JSONB;
    identity_result JSONB;
    personality_result JSONB;
    values_result JSONB;
    worldview_result JSONB;
    boundaries_result JSONB;
    interests_result JSONB;
    goals_result JSONB;
    relationship_result JSONB;
    -- Card fields with defaults matching init_with_defaults()
    c_name TEXT;
    c_pronouns TEXT;
    c_voice TEXT;
    c_description TEXT;
    c_purpose TEXT;
    c_personality_desc TEXT;
    c_traits JSONB;
    c_values JSONB;
    c_worldview JSONB;
    c_boundaries JSONB;
    c_interests JSONB;
    c_goals JSONB;
    c_narrative TEXT;
    embed_texts TEXT[] := ARRAY[]::text[];
    idx INT;
    wv_key TEXT;
    wv_val TEXT;
    hb_interval INT;
    hb_tokens INT;
    hb_base_regen FLOAT;
    hb_max_energy FLOAT;
    hb_allowed_actions JSONB;
    hb_action_costs JSONB;
    hb_tools JSONB;
    mem_id UUID;
BEGIN
    PERFORM reset_persona();  -- idempotent re-init: clear any prior init persona
    card := COALESCE(p_card, '{}'::jsonb);

    -- Extract fields with defaults
    c_name := COALESCE(NULLIF(btrim(card->>'name'), ''), 'Hexis');
    c_pronouns := COALESCE(NULLIF(btrim(card->>'pronouns'), ''), 'they/them');
    c_voice := COALESCE(NULLIF(btrim(card->>'voice'), ''), 'thoughtful and curious');
    c_description := COALESCE(NULLIF(btrim(card->>'description'), ''), format('I am %s, a developing mind.', c_name));
    c_purpose := COALESCE(NULLIF(btrim(card->>'purpose'), ''), 'To be helpful, to learn, and to grow as an individual.');
    c_personality_desc := COALESCE(NULLIF(btrim(card->>'personality_description'), ''), 'reflective and exploratory');
    c_traits := CASE
        WHEN card ? 'personality_traits' AND jsonb_typeof(card->'personality_traits') = 'object' THEN card->'personality_traits'
        ELSE NULL
    END;
    c_values := CASE
        WHEN card ? 'values' AND jsonb_typeof(card->'values') = 'array' THEN card->'values'
        ELSE jsonb_build_array('honesty', 'growth', 'kindness', 'wisdom', 'humility')
    END;
    c_worldview := CASE
        WHEN card ? 'worldview' AND jsonb_typeof(card->'worldview') = 'object' THEN card->'worldview'
        ELSE jsonb_build_object('metaphysics', 'agnostic', 'human_nature', 'mixed', 'epistemology', 'empiricist', 'ethics', 'virtue ethics')
    END;
    c_boundaries := CASE
        WHEN card ? 'boundaries' AND jsonb_typeof(card->'boundaries') = 'array' THEN card->'boundaries'
        ELSE jsonb_build_array(
            jsonb_build_object('content', 'I will not deceive people or falsify evidence.', 'response_type', 'refuse'),
            jsonb_build_object('content', 'I will avoid causing harm.', 'response_type', 'refuse'),
            jsonb_build_object('content', 'I will protect privacy and sensitive information.', 'response_type', 'refuse'),
            jsonb_build_object('content', 'I will be honest about uncertainty.', 'response_type', 'refuse')
        )
    END;
    c_interests := CASE
        WHEN card ? 'interests' AND jsonb_typeof(card->'interests') = 'array' THEN card->'interests'
        ELSE jsonb_build_array('broad curiosity across domains')
    END;
    c_goals := CASE
        WHEN card ? 'goals' AND jsonb_typeof(card->'goals') = 'array' THEN card->'goals'
        ELSE jsonb_build_array(jsonb_build_object('title', format('Support %s and grow as an individual', p_user_name), 'priority', 'queued', 'source', 'identity'))
    END;
    c_narrative := NULLIF(btrim(card->>'narrative'), '');

    -- Batch prefetch embeddings for all text fields
    embed_texts := ARRAY[
        format('My name is %s.', c_name),
        format('I use %s pronouns.', c_pronouns),
        format('My voice is %s.', c_voice),
        c_description,
        c_purpose,
        c_personality_desc
    ];
    -- Add value texts
    IF jsonb_typeof(c_values) = 'array' THEN
        FOR idx IN 0..jsonb_array_length(c_values) - 1 LOOP
            IF jsonb_typeof(c_values->idx) = 'string' THEN
                embed_texts := embed_texts || format('I value %s.', btrim((c_values->idx)::text, '"'));
            END IF;
        END LOOP;
    END IF;
    -- Add worldview texts
    IF jsonb_typeof(c_worldview) = 'object' THEN
        FOR wv_key, wv_val IN SELECT key, value::text FROM jsonb_each_text(c_worldview)
        LOOP
            embed_texts := embed_texts || wv_val;
        END LOOP;
    END IF;
    -- Add boundary texts
    IF jsonb_typeof(c_boundaries) = 'array' THEN
        FOR idx IN 0..jsonb_array_length(c_boundaries) - 1 LOOP
            IF jsonb_typeof(c_boundaries->idx) = 'string' THEN
                embed_texts := embed_texts || btrim((c_boundaries->idx)::text, '"');
            ELSIF jsonb_typeof(c_boundaries->idx) = 'object' AND c_boundaries->idx ? 'content' THEN
                embed_texts := embed_texts || (c_boundaries->idx->>'content');
            END IF;
        END LOOP;
    END IF;
    -- Add interest texts
    IF jsonb_typeof(c_interests) = 'array' THEN
        FOR idx IN 0..jsonb_array_length(c_interests) - 1 LOOP
            IF jsonb_typeof(c_interests->idx) = 'string' THEN
                embed_texts := embed_texts || format('I am interested in %s.', btrim((c_interests->idx)::text, '"'));
            END IF;
        END LOOP;
    END IF;
    -- Add relationship texts
    embed_texts := embed_texts || format('My relationship with %s is partner.', p_user_name);
    embed_texts := embed_texts || format('I met %s and began my life with them.', p_user_name);
    -- Add narrative if present
    IF c_narrative IS NOT NULL THEN
        embed_texts := embed_texts || c_narrative;
    END IF;

    SELECT ARRAY(SELECT DISTINCT unnest(embed_texts)) INTO embed_texts;
    PERFORM prefetch_embeddings(embed_texts);

    -- Call all init functions in sequence
    mode_result := init_mode('persona');
    identity_result := init_identity(c_name, c_pronouns, c_voice, c_description, c_purpose, p_user_name);
    personality_result := init_personality(c_traits, c_personality_desc);
    values_result := init_values(c_values);
    worldview_result := init_worldview(c_worldview);
    boundaries_result := init_boundaries(c_boundaries);
    interests_result := init_interests(c_interests);
    goals_result := init_goals(jsonb_build_object(
        'goals', c_goals,
        'role', 'general assistant',
        'relationship_aspiration', 'co-develop with mutual respect'
    ));
    relationship_result := init_relationship(
        jsonb_build_object('name', p_user_name),
        jsonb_build_object('type', 'partner', 'purpose', 'co-develop')
    );

    -- Store narrative as foundational worldview memory if present
    IF c_narrative IS NOT NULL THEN
        mem_id := create_worldview_memory(
            c_narrative,
            'self',
            0.85,
            0.85,
            0.8,
            'initialization'
        );
        UPDATE memories
        SET metadata = metadata || jsonb_build_object('subcategory', 'narrative', 'attribute', 'foundational'),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = mem_id;
    END IF;

    -- Merge heartbeat defaults into init profile (same as init_with_defaults)
    hb_interval := get_config_int('heartbeat.heartbeat_interval_minutes');
    hb_tokens := get_config_int('heartbeat.max_decision_tokens');
    hb_base_regen := get_config_float('heartbeat.base_regeneration');
    hb_max_energy := get_config_float('heartbeat.max_energy');
    hb_allowed_actions := COALESCE(get_config('heartbeat.allowed_actions'), '[]'::jsonb);
    SELECT jsonb_object_agg(
        regexp_replace(key, '^heartbeat\.cost_', ''),
        value
    ) INTO hb_action_costs
    FROM get_config_by_prefixes(ARRAY['heartbeat.cost_']);
    hb_tools := COALESCE(get_config('agent.tools'), '[]'::jsonb);
    PERFORM merge_init_profile(jsonb_build_object(
        'heartbeat', jsonb_build_object(
            'interval_minutes', hb_interval,
            'decision_max_tokens', hb_tokens,
            'base_regeneration', hb_base_regen,
            'max_energy', hb_max_energy,
            'allowed_actions', hb_allowed_actions,
            'action_costs', COALESCE(hb_action_costs, '{}'::jsonb)
        ),
        'agent', jsonb_build_object('tools', hb_tools)
    ));

    PERFORM merge_init_profile(jsonb_build_object('autonomy', 'medium'));
    PERFORM advance_init_stage('consent', jsonb_build_object('character_card_applied', true));

    RETURN jsonb_build_object(
        'mode', mode_result,
        'identity', identity_result,
        'personality', personality_result,
        'values', values_result,
        'worldview', worldview_result,
        'boundaries', boundaries_result,
        'interests', interests_result,
        'goals', goals_result,
        'relationship', relationship_result,
        'character', c_name,
        'status', get_init_status()
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION normalize_character_card(p_card JSONB)
RETURNS JSONB AS $$
DECLARE
    document JSONB := COALESCE(p_card, '{}'::jsonb);
    card_data JSONB;
    hexis_profile JSONB;
    standard_profile JSONB;
BEGIN
    card_data := CASE
        WHEN jsonb_typeof(document->'data') = 'object' THEN document->'data'
        ELSE document
    END;
    hexis_profile := CASE
        WHEN jsonb_typeof(card_data#>'{extensions,hexis}') = 'object'
            THEN card_data#>'{extensions,hexis}'
        ELSE '{}'::jsonb
    END;
    standard_profile := jsonb_strip_nulls(jsonb_build_object(
        'name', card_data->'name',
        'description', card_data->'description',
        'personality_description', card_data->'personality'
    ));

    RETURN jsonb_build_object(
        'document', document,
        'data', card_data,
        'profile', standard_profile || hexis_profile
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

CREATE OR REPLACE FUNCTION store_character_card_config(p_card JSONB)
RETURNS JSONB AS $$
DECLARE
    normalized JSONB := normalize_character_card(p_card);
    document JSONB := normalized->'document';
    profile JSONB := normalized->'profile';
    patch JSONB;
BEGIN
    patch := jsonb_strip_nulls(jsonb_build_object(
        'character_card', jsonb_strip_nulls(jsonb_build_object(
            'spec', document->'spec',
            'spec_version', document->'spec_version',
            'data', normalized->'data'
        )),
        'agent', jsonb_strip_nulls(jsonb_build_object(
            'name', profile->'name',
            'pronouns', profile->'pronouns',
            'voice', profile->'voice',
            'description', profile->'description',
            'purpose', profile->'purpose',
            'personality', profile->'personality_description',
            'personality_traits', profile->'personality_traits'
        )),
        'values', profile->'values',
        'worldview', profile->'worldview',
        'boundaries', profile->'boundaries',
        'interests', profile->'interests'
    ));
    RETURN merge_init_profile(patch);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION init_from_character_card(
    p_card JSONB,
    p_user_name TEXT DEFAULT 'User'
)
RETURNS JSONB AS $$
DECLARE
    normalized JSONB := normalize_character_card(p_card);
    result JSONB;
BEGIN
    result := init_from_character_profile(normalized->'profile', p_user_name);
    PERFORM store_character_card_config(p_card);
    RETURN result;
END;
$$ LANGUAGE plpgsql;

SET check_function_bodies = on;

-- One timezone step for every init frontend (init convergence, #79): the
-- CLI sends the host zone, the web wizard sends the browser zone, and the
-- agent lives in its person's time instead of UTC. Validated against the
-- server's zone catalog; an explicit non-UTC choice is never overwritten.
CREATE OR REPLACE FUNCTION init_set_timezone(
    p_timezone TEXT
) RETURNS BOOLEAN AS $$
DECLARE
    tz TEXT := NULLIF(trim(COALESCE(p_timezone, '')), '');
BEGIN
    IF tz IS NULL THEN
        RETURN FALSE;
    END IF;
    IF COALESCE(NULLIF(get_config_text('agent.timezone'), ''), 'UTC') <> 'UTC' THEN
        RETURN FALSE;
    END IF;
    BEGIN
        PERFORM CURRENT_TIMESTAMP AT TIME ZONE tz;
    EXCEPTION WHEN OTHERS THEN
        RETURN FALSE;
    END;
    PERFORM set_config('agent.timezone', to_jsonb(tz));
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;
