-- DB-native tool execution helpers for tools that do not need Python side effects.
SET search_path = public, ag_catalog, "$user";

-- Memory-count budgets (WS6): counts protect context and cost; relevance is
-- governed by min_score, never by a hardcoded N.
INSERT INTO config_defaults (key, value, description) VALUES
    ('memory.recall_default_limit', '5'::jsonb,
     'Default memory count for recall when the caller does not specify one'),
    ('memory.recall_max_limit', '50'::jsonb,
     'Ceiling on recall count — a context/cost budget, not a knowledge limit'),
    ('memory.recall_min_score', '0.35'::jsonb,
     'Default relevance floor for conscious recall: below it, memory honestly fails (and metamemory reports the felt state) instead of returning nearest neighbors to anything'),
    ('memory.hydrate_memory_limit', '10'::jsonb,
     'Default memory count for RAG hydration'),
    ('memory.context_section_limits', '{"recent": 20, "self": 25, "relationship": 15, "contradiction": 5, "emotional_pattern": 5, "trigger": 5}'::jsonb,
     'Per-section caps for subconscious/hydration context assembly')
ON CONFLICT (key) DO NOTHING;

CREATE OR REPLACE FUNCTION tool_success(
    p_output JSONB,
    p_display_output TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT jsonb_build_object('success', true, 'output', COALESCE(p_output, '{}'::jsonb), 'display_output', p_display_output);
$$;

CREATE OR REPLACE FUNCTION tool_error(
    p_error TEXT,
    p_error_type TEXT DEFAULT 'execution_failed'
) RETURNS JSONB
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT jsonb_build_object('success', false, 'error', COALESCE(p_error, 'Tool failed'), 'error_type', COALESCE(p_error_type, 'execution_failed'));
$$;

CREATE OR REPLACE FUNCTION execute_goals_tool(
    p_args JSONB,
    p_goal_origin goal_source
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    action TEXT := COALESCE(p_args->>'action', '');
    title TEXT;
    goal_id UUID;
    priority TEXT;
    source_value TEXT;
    snapshot JSONB;
    rows_json JSONB;
BEGIN
    IF action NOT IN ('create', 'update_priority', 'add_progress', 'list') THEN
        RETURN tool_error(format('Invalid action %L', action), 'invalid_params');
    END IF;
    IF action = 'create' THEN
        title := NULLIF(btrim(COALESCE(p_args->>'title', '')), '');
        IF title IS NULL THEN
            RETURN tool_error('Title is required for create', 'invalid_params');
        END IF;
        priority := COALESCE(NULLIF(p_args->>'priority', ''), 'queued');
        IF priority NOT IN ('active', 'queued', 'backburner', 'completed', 'abandoned') THEN
            priority := 'queued';
        END IF;
        source_value := COALESCE(NULLIF(p_args->>'source', ''), 'curiosity');
        IF source_value NOT IN ('curiosity', 'user_request', 'identity', 'derived', 'external') THEN
            source_value := 'curiosity';
        END IF;
        goal_id := create_goal(
            title,
            p_args->>'description',
            source_value::goal_source,
            priority::goal_priority,
            NULL,
            NULL,
            COALESCE(p_goal_origin, 'derived'::goal_source)
        );
        RETURN tool_success(
            jsonb_build_object(
                'goal_id', goal_id::text,
                'title', title,
                'priority', priority,
                'origin', COALESCE(p_goal_origin, 'derived'::goal_source)::text
            ),
            format('Created goal: %s (%s)', title, priority)
        );
    ELSIF action = 'update_priority' THEN
        priority := COALESCE(p_args->>'priority', '');
        IF NULLIF(p_args->>'goal_id', '') IS NULL THEN
            RETURN tool_error('goal_id is required for update_priority', 'invalid_params');
        END IF;
        IF priority NOT IN ('active', 'queued', 'backburner', 'completed', 'abandoned') THEN
            RETURN tool_error(format('Invalid priority %L', priority), 'invalid_params');
        END IF;
        BEGIN
            goal_id := (p_args->>'goal_id')::uuid;
        EXCEPTION WHEN invalid_text_representation THEN
            RETURN tool_error(format('Invalid goal_id: %s', p_args->>'goal_id'), 'invalid_params');
        END;
        PERFORM change_goal_priority(goal_id, priority::goal_priority, COALESCE(p_args->>'reason', ''));
        RETURN tool_success(jsonb_build_object('goal_id', goal_id::text, 'new_priority', priority, 'reason', COALESCE(p_args->>'reason', '')), format('Updated goal %s... to %s', left(goal_id::text, 8), priority));
    ELSIF action = 'add_progress' THEN
        IF NULLIF(p_args->>'goal_id', '') IS NULL THEN
            RETURN tool_error('goal_id is required for add_progress', 'invalid_params');
        END IF;
        IF NULLIF(btrim(COALESCE(p_args->>'note', '')), '') IS NULL THEN
            RETURN tool_error('note is required for add_progress', 'invalid_params');
        END IF;
        BEGIN
            goal_id := (p_args->>'goal_id')::uuid;
        EXCEPTION WHEN invalid_text_representation THEN
            RETURN tool_error(format('Invalid goal_id: %s', p_args->>'goal_id'), 'invalid_params');
        END;
        PERFORM add_goal_progress(goal_id, p_args->>'note');
        RETURN tool_success(jsonb_build_object('goal_id', goal_id::text, 'note', p_args->>'note'), format('Added progress to goal %s...', left(goal_id::text, 8)));
    ELSE
        priority := NULLIF(p_args->>'priority', '');
        IF priority IS NOT NULL AND priority IN ('active', 'queued', 'backburner', 'completed', 'abandoned') THEN
            SELECT COALESCE(jsonb_agg(to_jsonb(g)), '[]'::jsonb) INTO rows_json
            FROM get_goals_by_priority(priority::goal_priority) g;
            RETURN tool_success(jsonb_build_object('goals', rows_json, 'count', jsonb_array_length(rows_json)));
        END IF;
        snapshot := get_goals_snapshot();
        RETURN tool_success(COALESCE(snapshot, '{}'::jsonb));
    END IF;
EXCEPTION WHEN OTHERS THEN
    RETURN tool_error(SQLERRM);
END;
$$;

-- Direct database compatibility path. Without trusted execution context the
-- conservative origin is derived, regardless of model-authored source text.
CREATE OR REPLACE FUNCTION execute_goals_tool(p_args JSONB)
RETURNS JSONB
LANGUAGE sql
AS $$
    SELECT execute_goals_tool(
        p_args,
        'derived'::goal_source
    );
$$;

CREATE OR REPLACE FUNCTION record_backlog_user_change(
    p_action TEXT,
    p_title TEXT,
    p_item_id UUID
) RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO memories (type, content, embedding, importance, trust_level, metadata)
    VALUES (
        'episodic',
        format('User %s backlog item: %s', p_action, p_title),
        array_fill(0.1, ARRAY[embedding_dimension()])::vector,
        0.6,
        1.0,
        jsonb_build_object('backlog_item_id', p_item_id::text, 'action', p_action, 'source', 'user_backlog_change')
    );
EXCEPTION WHEN OTHERS THEN
    NULL;
END;
$$;

CREATE OR REPLACE FUNCTION execute_backlog_tool(
    p_args JSONB,
    p_context JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    action TEXT := COALESCE(p_args->>'action', '');
    item_id UUID;
    row_data backlog%ROWTYPE;
    rows_json JSONB;
    fields JSONB := '{}'::jsonb;
    is_user BOOLEAN := COALESCE(p_context->>'tool_context', 'chat') IN ('chat', 'mcp');
BEGIN
    IF action NOT IN ('create', 'update', 'delete', 'list', 'get', 'set_status', 'set_checkpoint') THEN
        RETURN tool_error(format('Invalid action %L', action), 'invalid_params');
    END IF;
    IF action = 'create' THEN
        IF NULLIF(btrim(COALESCE(p_args->>'title', '')), '') IS NULL THEN
            RETURN tool_error('Title is required for create', 'invalid_params');
        END IF;
        SELECT * INTO row_data
        FROM create_backlog_item(
            p_args->>'title',
            COALESCE(p_args->>'description', ''),
            CASE WHEN p_args->>'priority' IN ('urgent', 'high', 'normal', 'low') THEN p_args->>'priority' ELSE 'normal' END,
            CASE WHEN p_args->>'owner' IN ('agent', 'user', 'shared') THEN p_args->>'owner' ELSE 'agent' END,
            CASE WHEN is_user THEN 'user' ELSE 'agent' END,
            COALESCE(ARRAY(SELECT jsonb_array_elements_text(COALESCE(p_args->'tags', '[]'::jsonb))), ARRAY[]::TEXT[]),
            NULLIF(p_args->>'parent_id', '')::uuid
        );
        IF is_user THEN
            PERFORM record_backlog_user_change('created', row_data.title, row_data.id);
        END IF;
        RETURN tool_success(jsonb_build_object('item_id', row_data.id::text, 'title', row_data.title, 'priority', row_data.priority, 'owner', row_data.owner, 'created_by', row_data.created_by), format('Created backlog item: %s (%s)', row_data.title, row_data.priority));
    ELSIF action = 'list' THEN
        SELECT COALESCE(jsonb_agg(to_jsonb(b)), '[]'::jsonb)
        INTO rows_json
        FROM list_backlog(
            CASE WHEN p_args->>'status_filter' IN ('todo', 'in_progress', 'done', 'blocked', 'cancelled') THEN p_args->>'status_filter' ELSE NULL END,
            CASE WHEN p_args->>'priority_filter' IN ('urgent', 'high', 'normal', 'low') THEN p_args->>'priority_filter' ELSE NULL END,
            CASE WHEN p_args->>'owner_filter' IN ('agent', 'user', 'shared') THEN p_args->>'owner_filter' ELSE NULL END
        ) b;
        RETURN tool_success(jsonb_build_object('items', rows_json, 'count', jsonb_array_length(rows_json)));
    END IF;

    IF NULLIF(p_args->>'item_id', '') IS NULL THEN
        RETURN tool_error(format('item_id is required for %s', action), 'invalid_params');
    END IF;
    BEGIN
        item_id := (p_args->>'item_id')::uuid;
    EXCEPTION WHEN invalid_text_representation THEN
        RETURN tool_error(format('Invalid item_id: %s', p_args->>'item_id'), 'invalid_params');
    END;

    IF action = 'get' THEN
        SELECT * INTO row_data FROM get_backlog_item(item_id);
        IF row_data.id IS NULL THEN
            RETURN tool_error(format('Backlog item %s not found', item_id), 'execution_failed');
        END IF;
        RETURN tool_success(to_jsonb(row_data));
    ELSIF action = 'delete' THEN
        SELECT * INTO row_data FROM get_backlog_item(item_id);
        IF NOT delete_backlog_item(item_id) THEN
            RETURN tool_error(format('Backlog item %s not found', item_id), 'execution_failed');
        END IF;
        IF is_user AND row_data.title IS NOT NULL THEN
            PERFORM record_backlog_user_change('deleted', row_data.title, item_id);
        END IF;
        RETURN tool_success(jsonb_build_object('item_id', item_id::text, 'deleted', true), format('Deleted backlog item %s...', left(item_id::text, 8)));
    ELSIF action = 'set_status' THEN
        IF p_args->>'status' NOT IN ('todo', 'in_progress', 'done', 'blocked', 'cancelled') THEN
            RETURN tool_error(format('Invalid status %L', p_args->>'status'), 'invalid_params');
        END IF;
        fields := jsonb_build_object('status', p_args->>'status');
    ELSIF action = 'set_checkpoint' THEN
        IF NOT p_args ? 'checkpoint' THEN
            RETURN tool_error('checkpoint data is required for set_checkpoint', 'invalid_params');
        END IF;
        fields := jsonb_build_object('checkpoint', p_args->'checkpoint');
    ELSE
        FOR fields IN SELECT jsonb_object_agg(key, value)
        FROM jsonb_each(p_args)
        WHERE key IN ('title', 'description', 'priority', 'owner', 'status', 'tags')
        LOOP
            fields := COALESCE(fields, '{}'::jsonb);
        END LOOP;
        IF fields = '{}'::jsonb THEN
            RETURN tool_error('No fields to update. Provide at least one of: title, description, priority, owner, status, tags.', 'invalid_params');
        END IF;
    END IF;

    SELECT * INTO row_data FROM update_backlog_item(item_id, fields);
    IF row_data.id IS NULL THEN
        RETURN tool_error(format('Backlog item %s not found', item_id), 'execution_failed');
    END IF;
    IF is_user THEN
        PERFORM record_backlog_user_change(CASE WHEN action = 'set_status' THEN format('changed status to %L on', p_args->>'status') ELSE 'updated' END, row_data.title, item_id);
    END IF;
    RETURN tool_success(
        CASE action
            WHEN 'set_status' THEN jsonb_build_object('item_id', row_data.id::text, 'title', row_data.title, 'new_status', row_data.status)
            WHEN 'set_checkpoint' THEN jsonb_build_object('item_id', row_data.id::text, 'title', row_data.title, 'checkpoint_saved', true)
            ELSE jsonb_build_object('item_id', row_data.id::text, 'title', row_data.title, 'status', row_data.status, 'priority', row_data.priority, 'owner', row_data.owner)
        END,
        CASE action
            WHEN 'set_status' THEN format('Set %s to %s', row_data.title, row_data.status)
            WHEN 'set_checkpoint' THEN format('Saved checkpoint for %s', row_data.title)
            ELSE format('Updated backlog item %s...', left(item_id::text, 8))
        END
    );
EXCEPTION WHEN OTHERS THEN
    RETURN tool_error(SQLERRM);
END;
$$;

CREATE OR REPLACE FUNCTION contact_row_json(c contacts)
RETURNS JSONB
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT jsonb_build_object(
        'id', c.id,
        'name', c.name,
        'email', c.email,
        'company', c.company,
        'role', c.role,
        'phone', c.phone,
        'notes', c.notes,
        'tags', to_jsonb(c.tags),
        'source', c.source,
        'first_seen', c.first_seen,
        'last_touch', c.last_touch
    );
$$;

CREATE OR REPLACE FUNCTION execute_contact_tool(
    p_tool_name TEXT,
    p_args JSONB
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    row_data contacts%ROWTYPE;
    rows_json JSONB;
    contact_id BIGINT;
    keep_id BIGINT;
    remove_id BIGINT;
    query TEXT;
    limit_value INT;
    updated BOOLEAN;
BEGIN
    limit_value := LEAST(GREATEST(COALESCE(NULLIF(p_args->>'limit', '')::int, 20), 1), 100);
    IF p_tool_name = 'search_contacts' THEN
        query := NULLIF(btrim(COALESCE(p_args->>'query', '')), '');
        IF query IS NULL THEN
            SELECT COALESCE(jsonb_agg(contact_row_json(c)), '[]'::jsonb) INTO rows_json FROM recent_contacts(limit_value) c;
        ELSE
            SELECT COALESCE(jsonb_agg(contact_row_json(c)), '[]'::jsonb) INTO rows_json FROM search_contacts(query, limit_value) c;
        END IF;
        RETURN tool_success(jsonb_build_object('count', jsonb_array_length(rows_json), 'contacts', rows_json), format('Found %s contact(s)', jsonb_array_length(rows_json)));
    ELSIF p_tool_name = 'get_contact' THEN
        IF p_args ? 'id' THEN
            SELECT * INTO row_data FROM contacts WHERE id = (p_args->>'id')::bigint;
        ELSIF NULLIF(p_args->>'email', '') IS NOT NULL THEN
            SELECT * INTO row_data FROM get_contact_by_email(p_args->>'email');
        ELSE
            RETURN tool_error('Provide either id or email.', 'invalid_params');
        END IF;
        IF row_data.id IS NULL THEN
            RETURN tool_success('{"found": false}'::jsonb);
        END IF;
        RETURN tool_success(jsonb_build_object('found', true, 'contact', contact_row_json(row_data)));
    ELSIF p_tool_name = 'create_contact' THEN
        IF NULLIF(btrim(COALESCE(p_args->>'name', '')), '') IS NULL THEN
            RETURN tool_error('Name is required.', 'invalid_params');
        END IF;
        contact_id := create_contact(
            p_args->>'name',
            p_args->>'email',
            p_args->>'company',
            p_args->>'role',
            p_args->>'phone',
            p_args->>'notes',
            COALESCE(ARRAY(SELECT jsonb_array_elements_text(COALESCE(p_args->'tags', '[]'::jsonb))), ARRAY[]::TEXT[]),
            COALESCE(NULLIF(p_args->>'source', ''), 'manual')
        );
        RETURN tool_success(jsonb_build_object('id', contact_id, 'name', p_args->>'name'), format('Created contact #%s: %s', contact_id, p_args->>'name'));
    ELSIF p_tool_name = 'update_contact' THEN
        contact_id := NULLIF(p_args->>'id', '')::bigint;
        IF contact_id IS NULL THEN
            RETURN tool_error('Contact ID is required.', 'invalid_params');
        END IF;
        updated := update_contact(
            contact_id,
            p_args->>'name',
            p_args->>'email',
            p_args->>'company',
            p_args->>'role',
            p_args->>'phone',
            p_args->>'notes',
            CASE WHEN p_args ? 'tags' THEN ARRAY(SELECT jsonb_array_elements_text(COALESCE(p_args->'tags', '[]'::jsonb))) ELSE NULL END
        );
        IF NOT updated THEN
            RETURN tool_error(format('Contact #%s not found.', contact_id), 'execution_failed');
        END IF;
        PERFORM touch_contact(contact_id);
        RETURN tool_success(jsonb_build_object('id', contact_id, 'updated', true), format('Updated contact #%s', contact_id));
    ELSIF p_tool_name = 'merge_contacts' THEN
        keep_id := NULLIF(p_args->>'keep_id', '')::bigint;
        remove_id := NULLIF(p_args->>'remove_id', '')::bigint;
        IF keep_id IS NULL OR remove_id IS NULL THEN
            RETURN tool_error('Both keep_id and remove_id are required.', 'invalid_params');
        END IF;
        IF keep_id = remove_id THEN
            RETURN tool_error('Cannot merge a contact with itself.', 'invalid_params');
        END IF;
        IF NOT merge_contacts(keep_id, remove_id) THEN
            RETURN tool_error(format('Contact #%s not found.', remove_id), 'execution_failed');
        END IF;
        RETURN tool_success(jsonb_build_object('keep_id', keep_id, 'removed_id', remove_id, 'merged', true), format('Merged contact #%s into #%s', remove_id, keep_id));
    END IF;
    RETURN tool_error(format('Unsupported contact tool: %s', p_tool_name), 'invalid_params');
EXCEPTION WHEN OTHERS THEN
    RETURN tool_error(SQLERRM);
END;
$$;

INSERT INTO config_defaults (key, value, description) VALUES
    ('memory.remember_duplicate_similarity', '0.9'::jsonb,
     'Trigram similarity that makes a recent remember write equivalent within one chat session'),
    ('memory.remember_duplicate_window_minutes', '120'::jsonb,
     'How long remember reuses an equivalent tool-created memory within one chat session')
ON CONFLICT (key) DO NOTHING;

CREATE OR REPLACE FUNCTION execute_remember_tool(
    p_args JSONB
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    remember_content TEXT := NULLIF(btrim(COALESCE(p_args->>'content', '')), '');
    memory_type_value TEXT := COALESCE(NULLIF(p_args->>'type', ''), 'episodic');
    importance_value FLOAT;
    memory_id UUID;
    existing_id UUID;
    existing_content TEXT;
    session_uuid UUID := _db_brain_try_uuid(p_args#>>'{_execution_context,session_id}');
    call_id TEXT := NULLIF(p_args#>>'{_execution_context,call_id}', '');
    source_references JSONB := NULL;
    source_attribution JSONB := NULL;
    source_trust FLOAT := NULL;
    derived_conversation_source BOOLEAN := FALSE;
    canonical_content TEXT;
    duplicate_similarity FLOAT := LEAST(1.0, GREATEST(0.0, COALESCE(
        get_config_float('memory.remember_duplicate_similarity'), 0.9)));
    duplicate_window_minutes INT := GREATEST(1, COALESCE(
        get_config_int('memory.remember_duplicate_window_minutes'), 120));
    tool_write JSONB;
    result JSONB;
BEGIN
    IF remember_content IS NULL THEN
        RETURN tool_error('content is required', 'invalid_params');
    END IF;
    IF memory_type_value NOT IN ('episodic', 'semantic', 'procedural', 'strategic') THEN
        RETURN tool_error(format('Invalid memory type: %s', memory_type_value), 'invalid_params');
    END IF;
    importance_value := LEAST(1.0, GREATEST(0.0, COALESCE(
        NULLIF(p_args->>'importance', '')::float, 0.5)));
    canonical_content := lower(btrim(regexp_replace(remember_content, '[^[:alnum:]]+', ' ', 'g')));
    tool_write := jsonb_strip_nulls(jsonb_build_object(
        'source', 'remember_tool',
        'session_id', CASE WHEN session_uuid IS NULL THEN NULL ELSE session_uuid::text END,
        'call_id', call_id,
        'recorded_at', CURRENT_TIMESTAMP
    ));

    IF jsonb_typeof(p_args->'sources') = 'array'
       AND jsonb_array_length(p_args->'sources') > 0 THEN
        source_references := p_args->'sources';
        source_attribution := source_references->0;
    ELSIF session_uuid IS NOT NULL THEN
        source_trust := COALESCE(get_config_float('memory.conversation_turn_trust'), 0.8);
        source_attribution := jsonb_build_object(
            'kind', 'conversation',
            'ref', 'chat_session:' || session_uuid::text,
            'label', 'current chat session',
            'observed_at', CURRENT_TIMESTAMP,
            'trust', source_trust
        );
        source_references := jsonb_build_array(source_attribution);
        derived_conversation_source := TRUE;
    END IF;

    -- Serialize equivalent writes for one session so parallel/retried calls
    -- cannot both observe absence and create duplicates.
    IF session_uuid IS NOT NULL THEN
        PERFORM pg_advisory_xact_lock(hashtext(session_uuid::text), hashtext(canonical_content));
        SELECT m.id, m.content
        INTO existing_id, existing_content
        FROM memories m
        WHERE m.type = memory_type_value::memory_type
          AND m.status = 'active'
          AND m.created_at >= CURRENT_TIMESTAMP - make_interval(mins => duplicate_window_minutes)
          AND m.metadata#>>'{tool_write,session_id}' = session_uuid::text
          AND (
              lower(btrim(regexp_replace(m.content, '[^[:alnum:]]+', ' ', 'g'))) = canonical_content
              OR similarity(
                    lower(btrim(regexp_replace(m.content, '[^[:alnum:]]+', ' ', 'g'))),
                    canonical_content
                 ) >= duplicate_similarity
          )
        ORDER BY
            (lower(btrim(regexp_replace(m.content, '[^[:alnum:]]+', ' ', 'g'))) = canonical_content) DESC,
            similarity(lower(m.content), lower(remember_content)) DESC,
            m.created_at DESC
        LIMIT 1
        FOR UPDATE;
    END IF;

    IF existing_id IS NOT NULL THEN
        IF memory_type_value = 'semantic'
           AND jsonb_typeof(source_references) = 'array' THEN
            PERFORM add_semantic_source_reference(existing_id, source.value)
            FROM jsonb_array_elements(source_references) source(value);
            PERFORM sync_memory_trust(existing_id);
        END IF;
        IF jsonb_typeof(COALESCE(p_args->'concepts', '[]'::jsonb)) = 'array' THEN
            PERFORM link_memory_to_concept(existing_id, value)
            FROM jsonb_array_elements_text(p_args->'concepts') concept(value);
        END IF;
        UPDATE memories
        SET importance = GREATEST(importance, importance_value),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = existing_id;
        SELECT jsonb_strip_nulls(jsonb_build_object(
            'memory_id', m.id::text,
            'type', m.type::text,
            'content', left(m.content, 100),
            'confidence', NULLIF(m.metadata->>'confidence', '')::float,
            'trust_level', m.trust_level,
            'reused', TRUE
        ))
        INTO result
        FROM memories m
        WHERE m.id = existing_id;
        RETURN tool_success(
            result,
            format('Already stored; reused %s memory: %s...', memory_type_value, left(existing_content, 50))
        );
    END IF;

    IF memory_type_value = 'semantic' THEN
        memory_id := create_semantic_memory(
            remember_content,
            LEAST(1.0, GREATEST(0.0, COALESCE(NULLIF(p_args->>'confidence', '')::float, 0.5))),
            NULL,
            NULL,
            source_references,
            importance_value,
            source_attribution
        );
    ELSE
        memory_id := create_memory(
            memory_type_value::memory_type,
            remember_content,
            importance_value,
            source_attribution,
            CASE WHEN derived_conversation_source THEN source_trust ELSE NULL END,
            jsonb_build_object('tool_write', tool_write)
        );
    END IF;
    UPDATE memories
    SET metadata = jsonb_set(COALESCE(metadata, '{}'::jsonb), '{tool_write}', tool_write, TRUE)
    WHERE id = memory_id;
    IF jsonb_typeof(COALESCE(p_args->'concepts', '[]'::jsonb)) = 'array' THEN
        PERFORM link_memory_to_concept(memory_id, value)
        FROM jsonb_array_elements_text(p_args->'concepts') concept(value);
    END IF;
    SELECT jsonb_strip_nulls(jsonb_build_object(
        'memory_id', m.id::text,
        'type', m.type::text,
        'content', left(m.content, 100),
        'confidence', NULLIF(m.metadata->>'confidence', '')::float,
        'trust_level', m.trust_level,
        'reused', FALSE
    ))
    INTO result
    FROM memories m
    WHERE m.id = memory_id;
    RETURN tool_success(
        result,
        format('Stored %s memory: %s...', memory_type_value, left(remember_content, 50))
    );
EXCEPTION WHEN OTHERS THEN
    RETURN tool_error(SQLERRM);
END;
$$;

-- The wrapper below owns remember idempotency while this function retains the
-- established dispatch for every other memory tool.
CREATE OR REPLACE FUNCTION _execute_memory_tool_dispatch(
    p_tool_name TEXT,
    p_args JSONB
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    content TEXT;
    memory_type_value TEXT;
    importance_value FLOAT;
    memory_id UUID;
    query TEXT;
    limit_value INT;
    rows_json JSONB;
    type_filter memory_type[];
    has_filters BOOLEAN;
    use_hybrid BOOLEAN;
    target_id UUID;
    stance_value TEXT;
    revision JSONB;
    display TEXT;
    min_score_value FLOAT := 0.0;
    exclude_sensitive BOOLEAN := FALSE;
    sense_json JSONB;
    partials_json JSONB;
    metamemory_json JSONB;
    incubated BOOLEAN := FALSE;
    after_ts TIMESTAMPTZ;
    before_ts TIMESTAMPTZ;
    history_sources TEXT[];
    history_browse BOOLEAN;
    oldest_ts TIMESTAMPTZ;
    type_filter_uuids UUID[];
BEGIN
    IF p_tool_name = 'remember' THEN
        content := NULLIF(btrim(COALESCE(p_args->>'content', '')), '');
        IF content IS NULL THEN
            RETURN tool_error('content is required', 'invalid_params');
        END IF;
        memory_type_value := COALESCE(NULLIF(p_args->>'type', ''), 'episodic');
        IF memory_type_value NOT IN ('episodic', 'semantic', 'procedural', 'strategic') THEN
            RETURN tool_error(format('Invalid memory type: %s', memory_type_value), 'invalid_params');
        END IF;
        importance_value := LEAST(1.0, GREATEST(0.0, COALESCE(NULLIF(p_args->>'importance', '')::float, 0.5)));
        -- Semantic memories carry confidence + full source provenance (#33);
        -- other types accept the first source as their attribution.
        IF memory_type_value = 'semantic' THEN
            memory_id := create_semantic_memory(
                content,
                LEAST(1.0, GREATEST(0.0, COALESCE(NULLIF(p_args->>'confidence', '')::float, 0.5))),
                NULL,
                NULL,
                CASE WHEN jsonb_typeof(p_args->'sources') = 'array' THEN p_args->'sources' ELSE NULL END,
                importance_value
            );
        ELSE
            memory_id := create_memory(
                memory_type_value::memory_type,
                content,
                importance_value,
                CASE WHEN jsonb_typeof(p_args->'sources') = 'array' THEN p_args->'sources'->0 ELSE NULL END
            );
        END IF;
        IF jsonb_typeof(COALESCE(p_args->'concepts', '[]'::jsonb)) = 'array' THEN
            PERFORM link_memory_to_concept(memory_id, value)
            FROM jsonb_array_elements_text(p_args->'concepts') c(value);
        END IF;
        RETURN tool_success(jsonb_strip_nulls(jsonb_build_object(
            'memory_id', memory_id::text,
            'type', memory_type_value,
            'content', left(content, 100),
            'confidence', (SELECT NULLIF(m.metadata->>'confidence', '')::float FROM memories m WHERE m.id = memory_id),
            'trust_level', (SELECT m.trust_level FROM memories m WHERE m.id = memory_id)
        )), format('Stored %s memory: %s...', memory_type_value, left(content, 50)));
    ELSIF p_tool_name = 'add_evidence' THEN
        target_id := _db_brain_try_uuid(p_args->>'memory_id');
        IF target_id IS NULL THEN
            RETURN tool_error('memory_id must be a valid uuid', 'invalid_params');
        END IF;
        stance_value := lower(COALESCE(p_args->>'stance', ''));
        IF stance_value NOT IN ('supports', 'contradicts') THEN
            RETURN tool_error('stance must be supports or contradicts', 'invalid_params');
        END IF;
        IF jsonb_typeof(p_args->'source') <> 'object'
           OR COALESCE(NULLIF(p_args->'source'->>'ref', ''), NULLIF(p_args->'source'->>'label', '')) IS NULL THEN
            RETURN tool_error('source must be an object with at least a ref or label', 'invalid_params');
        END IF;
        revision := add_memory_evidence(target_id, stance_value, p_args->'source', NULLIF(p_args->>'note', ''), NULL, 'add_evidence');
        IF revision->>'reason' = 'not_found' THEN
            RETURN tool_error(format('memory not found: %s', target_id), 'invalid_params');
        ELSIF revision->>'reason' = 'not_semantic' THEN
            RETURN tool_error('add_evidence targets semantic memories; this memory is another type. Episodic records are the immutable audit trail — recall with memory_types=[''semantic''] to find the revisable belief that was built on this episode, and attach the evidence there.', 'invalid_params');
        END IF;
        display := CASE
            WHEN COALESCE((revision->>'applied')::boolean, FALSE) THEN
                format('Belief confidence %s -> %s (%s; independent source)',
                       round((revision->>'prior')::numeric, 2),
                       round((revision->>'posterior')::numeric, 2),
                       stance_value)
            WHEN revision->>'reason' = 'duplicate_source' THEN
                'No change: this source is already part of the belief''s evidence'
            WHEN revision->>'reason' = 'protected' THEN
                'Recorded as a contradiction flag: this belief is protected and is questioned, not rewritten'
            ELSE
                format('No confidence change (%s); evidence recorded', revision->>'reason')
        END;
        RETURN tool_success(revision, display);
    ELSIF p_tool_name = 'sense_memory_availability' THEN
        query := NULLIF(btrim(COALESCE(p_args->>'query', '')), '');
        IF query IS NULL THEN
            RETURN tool_error('query is required', 'invalid_params');
        END IF;
        SELECT to_jsonb(s) INTO rows_json FROM sense_memory_availability(query) s;
        RETURN tool_success(COALESCE(rows_json, '{"feeling": "nothing", "estimated_count": 0, "strongest_match": 0.0}'::jsonb), format('Memory availability: %s', COALESCE(rows_json->>'feeling', 'nothing')));
    ELSIF p_tool_name = 'recall' THEN
        query := NULLIF(p_args->>'query', '');
        -- Count is a context/cost budget, not a knowledge limit (#42/WS6):
        -- default and ceiling are config-driven; min_score cuts the tail by
        -- relevance instead of position.
        limit_value := LEAST(
            GREATEST(COALESCE(
                NULLIF(p_args->>'limit', '')::int,
                get_config_int('memory.recall_default_limit'),
                5
            ), 1),
            COALESCE(get_config_int('memory.recall_max_limit'), 50)
        );
        min_score_value := GREATEST(0.0, COALESCE(
            NULLIF(p_args->>'min_score', '')::float,
            get_config_float('memory.recall_min_score'),
            0.0));
        -- Sensitivity enforcement (#92/#96 stopgap): group-context turns set
        -- exclude_sensitive; private memories stay out of shared rooms
        -- through the tool path exactly as they do through hydrate.
        exclude_sensitive := COALESCE(NULLIF(p_args->>'exclude_sensitive', '')::boolean, FALSE);
        IF jsonb_typeof(p_args->'memory_types') = 'array' AND jsonb_array_length(p_args->'memory_types') > 0 THEN
            SELECT ARRAY(SELECT value::memory_type FROM jsonb_array_elements_text(p_args->'memory_types') t(value)) INTO type_filter;
        END IF;
        has_filters := type_filter IS NOT NULL
            OR NULLIF(p_args->>'source_path', '') IS NOT NULL
            OR NULLIF(p_args->>'source_kind', '') IS NOT NULL
            OR NULLIF(p_args->>'created_after', '') IS NOT NULL
            OR NULLIF(p_args->>'created_before', '') IS NOT NULL
            OR NULLIF(p_args->>'concept', '') IS NOT NULL;
        IF query IS NULL AND NOT has_filters THEN
            RETURN tool_error('Provide at least a query or one filter (memory_types, source_path, source_kind, created_after, created_before, concept).', 'invalid_params');
        END IF;
        -- Plain-query recalls use the hybrid retriever (vector + lexical);
        -- any filter or importance floor routes to the structured query.
        use_hybrid := query IS NOT NULL AND NOT has_filters
            AND COALESCE(NULLIF(p_args->>'min_importance', '')::float, 0.0) <= 0.0;
        IF use_hybrid THEN
            SELECT COALESCE(jsonb_agg(jsonb_strip_nulls(jsonb_build_object(
                'memory_id', r.memory_id::text,
                'content', r.content,
                'type', r.memory_type::text,
                'score', COALESCE(r.score, 0.0),
                'importance', COALESCE(r.importance, 0.0),
                'retrieval_source', NULLIF(r.source, ''),
                'trust', COALESCE(r.trust_level, 0.0),
                'confidence', (SELECT NULLIF(m.metadata->>'confidence', '')::float FROM memories m WHERE m.id = r.memory_id),
                'source_kind', NULLIF(r.source_attribution->>'kind', ''),
                'source_label', NULLIF(r.source_attribution->>'label', ''),
                'source_path', NULLIF(r.source_attribution->>'path', ''),
                'source_ref', NULLIF(r.source_attribution->>'ref', '')
            ))), '[]'::jsonb)
            INTO rows_json
            FROM recall_hybrid(query, limit_value) r
            WHERE COALESCE(r.score, 0.0) >= min_score_value
              AND (NOT exclude_sensitive
                   OR COALESCE(r.source_attribution->>'sensitivity', '') <> 'private');
        ELSE
            SELECT COALESCE(jsonb_agg(jsonb_strip_nulls(jsonb_build_object(
                'memory_id', r.memory_id::text,
                'content', r.content,
                'type', r.memory_type::text,
                'score', COALESCE(r.score, 0.0),
                'importance', COALESCE(r.importance, 0.0),
                'trust', COALESCE(r.trust_level, 0.0),
                'confidence', (SELECT NULLIF(m.metadata->>'confidence', '')::float FROM memories m WHERE m.id = r.memory_id),
                'source_kind', NULLIF(r.source_attribution->>'kind', ''),
                'source_label', NULLIF(r.source_attribution->>'label', ''),
                'source_path', NULLIF(r.source_attribution->>'path', ''),
                'source_ref', NULLIF(r.source_attribution->>'ref', '')
            ))), '[]'::jsonb)
            INTO rows_json
            FROM recall_memories_structured(
                query,
                limit_value,
                type_filter,
                COALESCE(NULLIF(p_args->>'min_importance', '')::float, 0.0),
                -- Empty strings are absent filters, not filters that match
                -- nothing: models routinely fill optional params with "".
                NULLIF(p_args->>'source_path', ''),
                NULLIF(p_args->>'source_kind', ''),
                NULLIF(p_args->>'created_after', '')::timestamptz,
                NULLIF(p_args->>'created_before', '')::timestamptz,
                NULLIF(p_args->>'concept', ''),
                NULL
            ) r
            WHERE COALESCE(r.score, 0.0) >= min_score_value
              AND (NOT exclude_sensitive
                   OR COALESCE(r.source_attribution->>'sensitivity', '') <> 'private');
        END IF;
        PERFORM touch_memories(ARRAY(SELECT (value->>'memory_id')::uuid FROM jsonb_array_elements(rows_json) value));

        -- Metamemory (#96, the ice-cream test): a thin or empty recall is
        -- itself information. Report the felt state — familiar-but-blocked
        -- (tip of the tongue) vs unfamiliar (perhaps never known) — and let
        -- a blocked-but-familiar query incubate in the background: the
        -- subconscious keeps searching, and a resolution surfaces later as
        -- spontaneous recall.
        IF query IS NOT NULL AND jsonb_array_length(rows_json) < LEAST(3, limit_value) THEN
            SELECT to_jsonb(s) INTO sense_json FROM sense_memory_availability(query) s;
            SELECT COALESCE(jsonb_agg(jsonb_build_object(
                       'topic', fp.cluster_name,
                       'closeness', round(fp.cluster_similarity::numeric, 3))), '[]'::jsonb)
            INTO partials_json
            FROM find_partial_activations(query) fp;
            IF jsonb_array_length(rows_json) = 0
               AND COALESCE((sense_json->>'strongest_match')::float, 0.0)
                   >= COALESCE(get_config_float('metamemory.incubate_min_familiarity'), 0.55) THEN
                PERFORM request_background_search(query);
                incubated := TRUE;
            END IF;
            metamemory_json := jsonb_build_object(
                'feeling', COALESCE(sense_json->>'feeling', 'nothing'),
                'familiarity', COALESCE((sense_json->>'strongest_match')::float, 0.0),
                'description', sense_json->>'description',
                'tip_of_tongue', partials_json,
                'incubating', incubated);
            RETURN tool_success(
                jsonb_build_object('memories', rows_json,
                                   'count', jsonb_array_length(rows_json),
                                   'query', COALESCE(query, '(filters only)'),
                                   'metamemory', metamemory_json),
                CASE
                    WHEN incubated THEN
                        format('Nothing surfaced for %L yet, but it feels familiar — I''ll let it simmer; it may come to me later.', query)
                    WHEN jsonb_array_length(rows_json) = 0
                         AND COALESCE(sense_json->>'feeling', 'nothing') IN ('nothing', 'vague') THEN
                        format('Nothing for %L — and it doesn''t feel like something I ever knew.', query)
                    ELSE
                        format('Found %s memories for %L', jsonb_array_length(rows_json), query)
                END);
        END IF;

        RETURN tool_success(jsonb_build_object('memories', rows_json, 'count', jsonb_array_length(rows_json), 'query', COALESCE(query, '(filters only)')), format('Found %s memories for %L', jsonb_array_length(rows_json), COALESCE(query, '(filters only)')));
    ELSIF p_tool_name = 'belief_history' THEN
        target_id := _db_brain_try_uuid(p_args->>'memory_id');
        IF target_id IS NULL THEN
            RETURN tool_error('memory_id must be a valid uuid', 'invalid_params');
        END IF;
        revision := get_belief_history(target_id, COALESCE(NULLIF(p_args->>'limit', '')::int, 20));
        IF revision->>'error' = 'not_found' THEN
            RETURN tool_error(format('memory not found: %s', target_id), 'invalid_params');
        END IF;
        display := format('Belief at confidence %s after %s revision(s); %s evidence link(s)',
            COALESCE(revision#>>'{memory,confidence}', 'n/a'),
            jsonb_array_length(COALESCE(revision->'revisions', '[]'::jsonb)),
            jsonb_array_length(COALESCE(revision->'evidence', '[]'::jsonb)));
        RETURN tool_success(revision, display);
    ELSIF p_tool_name = 'open_memory' THEN
        -- Graded recall's drill-down (#76): the verbatim experience behind a
        -- gist — source units time-ordered, pre-summary full text, members a
        -- retention gist superseded.
        target_id := _db_brain_try_uuid(p_args->>'memory_id');
        IF target_id IS NULL THEN
            RETURN tool_error('memory_id must be a valid uuid', 'invalid_params');
        END IF;
        revision := get_memory_story(target_id, COALESCE(NULLIF(p_args->>'max_units', '')::int, 40));
        IF revision->>'error' = 'not_found' THEN
            RETURN tool_error(format('memory not found: %s', target_id), 'invalid_params');
        END IF;
        display := format('Opened memory: %s source unit(s)%s%s',
            jsonb_array_length(COALESCE(revision->'source_units', '[]'::jsonb)),
            CASE WHEN revision ? 'full_content' THEN ', pre-gist full text preserved' ELSE '' END,
            CASE WHEN revision ? 'superseded_members'
                 THEN format(', %s gisted member(s)', jsonb_array_length(revision->'superseded_members'))
                 ELSE '' END);
        RETURN tool_success(revision, display);
    ELSIF p_tool_name = 'search_history' THEN
        -- Cross-session lexical/timeline search: validation, browse-vs-keyword
        -- limit policy, and the loud-truncation paging hint all live here.
        query := trim(COALESCE(p_args->>'query', ''));
        BEGIN
            after_ts := NULLIF(p_args->>'created_after', '')::timestamptz;
            before_ts := NULLIF(p_args->>'created_before', '')::timestamptz;
        EXCEPTION WHEN OTHERS THEN
            RETURN tool_error('created_after/created_before must be ISO-8601 timestamps', 'invalid_params');
        END;
        IF after_ts IS NOT NULL AND before_ts IS NOT NULL AND after_ts >= before_ts THEN
            RETURN tool_error('created_after must be earlier than created_before', 'invalid_params');
        END IF;
        history_browse := NULLIF(trim(BOTH '* ' FROM query), '') IS NULL;
        IF history_browse AND after_ts IS NULL AND before_ts IS NULL THEN
            RETURN tool_error('Provide query keywords, or a created_after/created_before window to browse a time range chronologically', 'invalid_params');
        END IF;
        IF p_args ? 'sources' THEN
            IF jsonb_typeof(p_args->'sources') <> 'array' OR jsonb_array_length(p_args->'sources') = 0 THEN
                RETURN tool_error('history search requires at least one source', 'invalid_params');
            END IF;
            SELECT array_agg(DISTINCT value) INTO history_sources
            FROM jsonb_array_elements_text(p_args->'sources') t(value);
            IF EXISTS (SELECT 1 FROM unnest(history_sources) s(v) WHERE v NOT IN ('turn', 'memory', 'desk')) THEN
                RETURN tool_error(
                    'history search sources must be ''turn'', ''memory'', and/or ''desk''; invalid: '
                    || (SELECT string_agg(v, ', ' ORDER BY v) FROM unnest(history_sources) s(v) WHERE v NOT IN ('turn', 'memory', 'desk')),
                    'invalid_params');
            END IF;
        ELSE
            history_sources := ARRAY['turn', 'memory'];
        END IF;
        -- Browse mode reads preview-grain rows, so it affords the higher
        -- config-owned ceiling (#76); keyword search stays at 50.
        limit_value := LEAST(
            GREATEST(COALESCE(NULLIF(p_args->>'limit', '')::int, 20), 1),
            CASE WHEN history_browse
                 THEN GREATEST(COALESCE(get_config_int('memory.history_browse_max'), 200), 1)
                 ELSE 50 END);
        WITH hits AS (
            SELECT h.*, ROW_NUMBER() OVER () AS ord
            FROM search_cross_session_history(
                query, limit_value, history_sources, after_ts, before_ts,
                _db_brain_try_uuid(p_args->>'exclude_session_id'),
                COALESCE(NULLIF(p_args->>'exclude_sensitive', '')::boolean, FALSE)) h
        )
        SELECT COALESCE(jsonb_agg(jsonb_build_object(
                   'source_kind', h.source_kind,
                   'item_id', h.item_id::text,
                   'session_id', h.session_id::text,
                   'content', h.content,
                   'user_text', h.user_text,
                   'assistant_text', h.assistant_text,
                   'memory_type', h.memory_type,
                   'occurred_at', h.occurred_at,
                   'rank', h.rank,
                   'source_unit_ids', COALESCE((SELECT jsonb_agg(u::text) FROM unnest(h.source_unit_ids) u), '[]'::jsonb),
                   'source_attribution', h.source_attribution,
                   'metadata', h.metadata
               ) ORDER BY h.ord), '[]'::jsonb),
               min(h.occurred_at)
        INTO rows_json, oldest_ts
        FROM hits h;
        revision := jsonb_build_object(
            'query', query,
            'results', rows_json,
            'count', jsonb_array_length(rows_json),
            'limit', limit_value,
            -- Loud truncation (#76): a full page means the window holds
            -- more — silence here once read as "the morning was blank."
            'truncated', jsonb_array_length(rows_json) >= limit_value,
            'excluded_session_id', _db_brain_try_uuid(p_args->>'exclude_session_id')::text);
        IF jsonb_array_length(rows_json) > 0 AND jsonb_array_length(rows_json) >= limit_value THEN
            revision := revision || jsonb_build_object('note',
                'window truncated — older entries exist; page with created_before='
                || (to_jsonb(oldest_ts) #>> '{}'));
        END IF;
        RETURN tool_success(revision,
            format('Found %s history result(s)', jsonb_array_length(rows_json))
            || CASE WHEN jsonb_array_length(rows_json) >= limit_value
                    THEN ' (page full — more exist in this window)' ELSE '' END);
    ELSIF p_tool_name = 'explore_concept' THEN
        query := NULLIF(btrim(COALESCE(p_args->>'concept', '')), '');
        IF query IS NULL THEN
            RETURN tool_error('concept is required', 'invalid_params');
        END IF;
        limit_value := LEAST(GREATEST(COALESCE(NULLIF(p_args->>'limit', '')::int, 5), 1), 20);
        SELECT COALESCE(jsonb_agg(jsonb_build_object(
                   'memory_id', f.memory_id::text,
                   'content', f.memory_content,
                   'type', f.memory_type::text,
                   'importance', f.memory_importance,
                   'concept_strength', f.link_strength)), '[]'::jsonb)
        INTO rows_json
        FROM find_memories_by_concept(query, limit_value) f;
        revision := jsonb_build_object(
            'concept', query,
            'memories', rows_json,
            'related_concepts', '[]'::jsonb,
            'count', jsonb_array_length(rows_json));
        IF COALESCE(NULLIF(p_args->>'include_related', '')::boolean, TRUE)
           AND jsonb_array_length(rows_json) > 0 THEN
            revision := jsonb_set(revision, '{related_concepts}', COALESCE((
                SELECT jsonb_agg(jsonb_build_object('name', r.name, 'shared_memories', r.shared_memories))
                FROM find_related_concepts_for_memories(
                    ARRAY(SELECT (value->>'memory_id')::uuid FROM jsonb_array_elements(rows_json) value),
                    query, 10) r), '[]'::jsonb));
        END IF;
        RETURN tool_success(revision,
            format('Found %s memories for concept ''%s''', jsonb_array_length(rows_json), query));
    ELSIF p_tool_name = 'explore_subgraph' THEN
        IF jsonb_typeof(p_args->'seeds') = 'array' AND jsonb_array_length(p_args->'seeds') > 0 THEN
            BEGIN
                SELECT array_agg(value::uuid) INTO type_filter_uuids
                FROM jsonb_array_elements_text(p_args->'seeds') t(value);
            EXCEPTION WHEN OTHERS THEN
                RETURN tool_error('seeds must be memory uuids', 'invalid_params');
            END;
        ELSIF NULLIF(btrim(COALESCE(p_args->>'query', '')), '') IS NOT NULL THEN
            SELECT array_agg(f.memory_id) INTO type_filter_uuids
            FROM fast_recall(p_args->>'query', 10) f;
        ELSE
            RETURN tool_error('Provide ''query'' or ''seeds''.', 'invalid_params');
        END IF;
        IF type_filter_uuids IS NULL OR cardinality(type_filter_uuids) = 0 THEN
            RETURN tool_success(
                '{"nodes": [], "edges": [], "rendered": null}'::jsonb,
                'No seed memories found.');
        END IF;
        revision := build_context_subgraph(
            type_filter_uuids,
            LEAST(GREATEST(COALESCE(NULLIF(p_args->>'depth', '')::int, 2), 1), 4),
            CASE WHEN jsonb_typeof(p_args->'rel_types') = 'array'
                 THEN ARRAY(SELECT jsonb_array_elements_text(p_args->'rel_types')) END,
            LEAST(GREATEST(COALESCE(NULLIF(p_args->>'budget', '')::int, 30), 1), 100));
        display := render_subgraph(revision);
        RETURN tool_success(jsonb_build_object(
                'nodes', COALESCE(revision->'nodes', '[]'::jsonb),
                'edges', COALESCE(revision->'edges', '[]'::jsonb),
                'rendered', display),
            COALESCE(display, format('No typed connections among %s seed memory(ies).',
                                     cardinality(type_filter_uuids))));
    ELSIF p_tool_name IN ('get_procedures', 'get_strategies') THEN
        -- fast_recall filtered to one memory type. (The former Python path
        -- filtered on a column fast_recall does not return, so these tools
        -- errored on every call — fixed here.)
        query := NULLIF(btrim(COALESCE(
            p_args->>'task', p_args->>'situation', p_args->>'query', '')), '');
        IF query IS NULL THEN
            RETURN tool_error(
                CASE WHEN p_tool_name = 'get_procedures'
                     THEN 'task is required' ELSE 'situation is required' END,
                'invalid_params');
        END IF;
        limit_value := LEAST(GREATEST(COALESCE(NULLIF(p_args->>'limit', '')::int, 3), 1), 10);
        memory_type_value := CASE WHEN p_tool_name = 'get_procedures'
                                  THEN 'procedural' ELSE 'strategic' END;
        SELECT COALESCE(jsonb_agg(item), '[]'::jsonb) INTO rows_json FROM (
            SELECT jsonb_build_object(
                'memory_id', f.memory_id::text,
                'content', f.content,
                'similarity', f.score) AS item
            FROM fast_recall(query, limit_value * 2) f
            WHERE f.memory_type::text = memory_type_value
            LIMIT limit_value
        ) s;
        IF p_tool_name = 'get_procedures' THEN
            RETURN tool_success(
                jsonb_build_object('procedures', rows_json,
                                   'count', jsonb_array_length(rows_json), 'task', query),
                format('Found %s procedures for ''%s''', jsonb_array_length(rows_json), query));
        END IF;
        RETURN tool_success(
            jsonb_build_object('strategies', rows_json,
                               'count', jsonb_array_length(rows_json), 'situation', query),
            format('Found %s strategies for ''%s''', jsonb_array_length(rows_json), query));
    END IF;
    RETURN tool_error(format('Unsupported memory tool: %s', p_tool_name), 'invalid_params');
EXCEPTION WHEN OTHERS THEN
    RETURN tool_error(SQLERRM);
END;
$$;

CREATE OR REPLACE FUNCTION execute_memory_tool(
    p_tool_name TEXT,
    p_args JSONB
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
BEGIN
    IF p_tool_name = 'remember' THEN
        RETURN execute_remember_tool(p_args);
    END IF;
    RETURN _execute_memory_tool_dispatch(p_tool_name, p_args);
END;
$$;
