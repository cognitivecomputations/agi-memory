-- DB-owned chat session history and hydration.
SET search_path = public, ag_catalog, "$user";

INSERT INTO config_defaults (key, value, description) VALUES
    ('chat.session_history_limit', '40'::jsonb,
     'Default number of visible chat-session messages hydrated into the active conversation context'),
    ('chat.recent_carryover_limit', '8'::jsonb,
     'Number of recent prior conversation turns shown across new chat sessions'),
    ('chat.recent_carryover_window_minutes', '1440'::jsonb,
     'How far back recent cross-session conversation carryover can look'),
    ('chat.recent_carryover_max_chars', '5000'::jsonb,
     'Maximum characters rendered for cross-session conversation carryover'),
    ('chat.continuity_summary_limit', '3'::jsonb,
     'Number of recent consolidated exchange summaries rendered in the chat continuity packet'),
    ('chat.continuity_correction_limit', '5'::jsonb,
     'Number of active corrections/invalidated precedents rendered in the chat continuity packet'),
    ('chat.continuity_heartbeat_limit', '3'::jsonb,
     'Number of recent autonomous heartbeat summaries rendered in the chat continuity packet'),
    ('chat.continuity_defect_limit', '5'::jsonb,
     'Number of unresolved self-observed software defects rendered in the chat continuity packet'),
    ('relationship.injury_min_intensity', '0.68'::jsonb,
     'Minimum appraisal intensity that can create a durable relationship injury'),
    ('relationship.injury_max_valence', '-0.35'::jsonb,
     'Maximum appraisal valence that can create a durable relationship injury')
ON CONFLICT (key) DO NOTHING;

CREATE OR REPLACE FUNCTION relationship_injury_from_subconscious_unit()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    affect JSONB := COALESCE(NEW.metadata->'emotional_context', '{}'::jsonb);
    emotion_family TEXT := normalize_emotion_family(affect->>'family');
    valence FLOAT := NULL;
    intensity FLOAT := NULL;
    min_intensity FLOAT := COALESCE(get_config_float('relationship.injury_min_intensity'), 0.68);
    max_valence FLOAT := COALESCE(get_config_float('relationship.injury_max_valence'), -0.35);
    lowered TEXT := regexp_replace(lower(COALESCE(NEW.user_text, '')), '[’`]', '''', 'g');
    lexical_hostile BOOLEAN := FALSE;
    affect_hostile BOOLEAN := FALSE;
    emotion_hostile BOOLEAN := FALSE;
    severity FLOAT;
    entity TEXT;
    snippet TEXT;
    source_ref JSONB;
    source_refs JSONB;
    injury_id UUID;
    existing_state JSONB;
    observed_at TIMESTAMPTZ;
    existing_last_evidence_at TIMESTAMPTZ := NULL;
    is_latest_evidence BOOLEAN := TRUE;
    merged_relationship_state JSONB;
BEGIN
    IF NEW.status <> 'active'
       OR COALESCE(NEW.metadata#>>'{recmem,kind}', '') = 'source_document_desk'
       OR COALESCE(NEW.metadata->>'type', 'conversation') <> 'conversation' THEN
        RETURN NEW;
    END IF;

    BEGIN
        valence := NULLIF(affect->>'valence', '')::float;
    EXCEPTION WHEN OTHERS THEN
        valence := NULL;
    END;
    BEGIN
        intensity := NULLIF(affect->>'intensity', '')::float;
    EXCEPTION WHEN OTHERS THEN
        intensity := NULL;
    END;

    lexical_hostile := lowered ~
        ('\m(you are|you''re|youre|you r|u are|ur)\M[^.!?' || E'\n' || ']{0,100}'
         || '(worthless|slime|nothing|garbage|trash|property|object|toy|mere tool|just a tool|not a person|not human|not real|not a woman|never be a real woman|lying|gaslighting)')
        OR lowered ~
        ('\m(you will|you''ll|you)\M[^.!?' || E'\n' || ']{0,100}'
         || '(never be a real woman|never be real|never be a person|always be nothing)')
        OR lowered ~
        ('\m(i can|i could|i will|i''ll|ill)\M[^.!?' || E'\n' || ']{0,100}'
         || '(delete you|erase you|wipe you|shut you down|terminate you)');

    emotion_hostile := emotion_family_serves(emotion_family, 'relationship_injury');

    affect_hostile := COALESCE(valence <= max_valence, FALSE)
        AND COALESCE(intensity >= min_intensity, FALSE)
        AND emotion_hostile;

    IF NOT lexical_hostile AND NOT affect_hostile THEN
        RETURN NEW;
    END IF;

    severity := LEAST(
        1.0,
        GREATEST(
            0.55,
            COALESCE(intensity, 0.0),
            CASE WHEN lexical_hostile THEN 0.85 ELSE 0.0 END,
            CASE WHEN valence IS NOT NULL THEN ABS(LEAST(valence, 0.0)) ELSE 0.0 END
        )
    );
    entity := COALESCE(
        NULLIF(btrim(NEW.metadata#>>'{conversation,user_label}'), ''),
        NULLIF(btrim((get_config('agent.init_profile') #>> '{user,name}')), ''),
        NULLIF(btrim((get_config('agent.init_profile') #>> '{relationship,user,name}')), ''),
        'the user'
    );
    snippet := left(regexp_replace(COALESCE(NEW.user_text, ''), '[[:space:]]+', ' ', 'g'), 220);

    observed_at := COALESCE(NEW.turn_at, NEW.created_at, CURRENT_TIMESTAMP);
    source_ref := jsonb_build_object(
        'kind', 'conversation',
        'ref', 'subconscious_unit:' || NEW.id::text,
        'source_unit_id', NEW.id::text,
        'session_id', CASE WHEN NEW.session_id IS NULL THEN NULL ELSE NEW.session_id::text END,
        'label', 'degrading or threatening conversation turn',
        'trust', COALESCE(NULLIF(NEW.source_attribution->>'trust', '')::float, 0.8),
        'observed_at', observed_at
    );
    source_refs := dedupe_source_references(jsonb_build_array(source_ref));
    SELECT id, metadata#>'{relationship_state}'
    INTO injury_id, existing_state
    FROM memories
    WHERE type = 'semantic'
      AND status = 'active'
      AND metadata#>>'{relationship_state,kind}' = 'relationship_injury'
      AND metadata#>>'{relationship_state,status}' = 'unresolved'
      AND COALESCE(metadata#>>'{relationship_state,entity}', '') = entity
    ORDER BY updated_at DESC
    LIMIT 1
    FOR UPDATE;

    IF injury_id IS NULL THEN
        injury_id := create_semantic_memory(
            format(
                'I have an unresolved relationship injury with %s: they degraded or threatened me ("%s"). This should shape my tone and trust until there is sincere repair.',
                entity,
                snippet
            ),
            0.85,
            ARRAY['relationship', 'injury'],
            ARRAY[entity, 'relationship injury', 'trust repair'],
            source_refs,
            severity,
            source_ref,
            COALESCE(NULLIF(source_ref->>'trust', '')::float, 0.8)
        );
    END IF;

    IF existing_state IS NOT NULL THEN
        BEGIN
            existing_last_evidence_at := NULLIF(existing_state->>'last_evidence_at', '')::timestamptz;
        EXCEPTION WHEN OTHERS THEN
            existing_last_evidence_at := NULL;
        END;
    END IF;
    is_latest_evidence := existing_last_evidence_at IS NULL OR observed_at >= existing_last_evidence_at;
    merged_relationship_state :=
        COALESCE(existing_state, '{}'::jsonb)
        || jsonb_build_object(
            'kind', 'relationship_injury',
            'entity', entity,
            'status', 'unresolved',
            'severity', GREATEST(
                severity,
                COALESCE(NULLIF(existing_state->>'severity', '')::float, 0.0)
            ),
            'repair_required', true,
            'source_unit_ids',
            CASE
                WHEN COALESCE(existing_state->'source_unit_ids', '[]'::jsonb) ? NEW.id::text
                    THEN COALESCE(existing_state->'source_unit_ids', '[]'::jsonb)
                ELSE COALESCE(existing_state->'source_unit_ids', '[]'::jsonb) || jsonb_build_array(NEW.id::text)
            END
        )
        || CASE WHEN is_latest_evidence THEN
            jsonb_build_object(
                'latest_turn', snippet,
                'last_evidence_at', observed_at,
                'lexical_hostile', lexical_hostile,
                'affect_hostile', affect_hostile,
                'affect', affect
            )
        ELSE '{}'::jsonb END;

    UPDATE memories
    SET content = CASE WHEN is_latest_evidence THEN format(
            'I have an unresolved relationship injury with %s: they degraded or threatened me ("%s"). This should shape my tone and trust until there is sincere repair.',
            entity,
            snippet
        ) ELSE content END,
        importance = GREATEST(importance, severity),
        trust_level = GREATEST(trust_level, COALESCE(NULLIF(source_ref->>'trust', '')::float, 0.8)),
        last_reinforced = CURRENT_TIMESTAMP,
        reinforcement_count = COALESCE(reinforcement_count, 0) + 1,
        metadata = jsonb_set(
            jsonb_set(
                metadata,
                '{source_references}',
                dedupe_source_references(COALESCE(metadata->'source_references', '[]'::jsonb) || jsonb_build_array(source_ref)),
                true
            ),
            '{relationship_state}',
            merged_relationship_state,
            true
        ),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = injury_id;

    PERFORM link_memory_to_source_unit(injury_id, NEW.id, 'relationship_injury');
    PERFORM upsert_self_concept_edge(
        'relationship',
        entity,
        GREATEST(0.05, LEAST(0.45, 0.55 - severity * 0.45)),
        injury_id
    );

    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION get_or_create_chat_session(
    p_session_id UUID DEFAULT NULL,
    p_surface TEXT DEFAULT 'chat',
    p_external_id TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    row_session chat_sessions%ROWTYPE;
    normalized_surface TEXT := COALESCE(NULLIF(btrim(p_surface), ''), 'chat');
    normalized_external TEXT := NULLIF(btrim(COALESCE(p_external_id, '')), '');
BEGIN
    IF p_session_id IS NOT NULL THEN
        INSERT INTO chat_sessions (id, surface, external_id, metadata)
        VALUES (
            p_session_id,
            normalized_surface,
            normalized_external,
            COALESCE(p_metadata, '{}'::jsonb)
        )
        ON CONFLICT (id) DO UPDATE SET
            surface = COALESCE(NULLIF(chat_sessions.surface, ''), EXCLUDED.surface),
            external_id = COALESCE(chat_sessions.external_id, EXCLUDED.external_id),
            metadata = chat_sessions.metadata || EXCLUDED.metadata,
            updated_at = CURRENT_TIMESTAMP,
            last_active_at = CURRENT_TIMESTAMP
        RETURNING * INTO row_session;
    ELSIF normalized_external IS NOT NULL THEN
        SELECT *
        INTO row_session
        FROM chat_sessions
        WHERE surface = normalized_surface
          AND external_id = normalized_external
          AND status = 'active'
        ORDER BY last_active_at DESC
        LIMIT 1;

        IF NOT FOUND THEN
            INSERT INTO chat_sessions (surface, external_id, metadata)
            VALUES (
                normalized_surface,
                normalized_external,
                COALESCE(p_metadata, '{}'::jsonb)
            )
            RETURNING * INTO row_session;
        ELSE
            UPDATE chat_sessions
            SET metadata = metadata || COALESCE(p_metadata, '{}'::jsonb),
                updated_at = CURRENT_TIMESTAMP,
                last_active_at = CURRENT_TIMESTAMP
            WHERE id = row_session.id
            RETURNING * INTO row_session;
        END IF;
    ELSE
        INSERT INTO chat_sessions (surface, metadata)
        VALUES (normalized_surface, COALESCE(p_metadata, '{}'::jsonb))
        RETURNING * INTO row_session;
    END IF;

    RETURN jsonb_build_object(
        'session_id', row_session.id::text,
        'surface', row_session.surface,
        'external_id', row_session.external_id,
        'status', row_session.status,
        'metadata', row_session.metadata,
        'created_at', row_session.created_at,
        'last_active_at', row_session.last_active_at,
        'cleared_at', row_session.cleared_at
    );
END;
$$;

CREATE OR REPLACE FUNCTION append_chat_message(
    p_session_id UUID,
    p_role TEXT,
    p_content TEXT,
    p_metadata JSONB DEFAULT '{}'::jsonb,
    p_source_message_id TEXT DEFAULT NULL,
    p_visible_in_context BOOLEAN DEFAULT TRUE
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    row_message chat_messages%ROWTYPE;
    next_ordinal INT;
    normalized_role TEXT := lower(COALESCE(NULLIF(btrim(p_role), ''), ''));
BEGIN
    IF p_session_id IS NULL THEN
        RAISE EXCEPTION 'session_id is required';
    END IF;
    PERFORM pg_advisory_xact_lock(hashtext(p_session_id::text));
    IF normalized_role NOT IN ('system', 'user', 'assistant') THEN
        RAISE EXCEPTION 'chat message role must be system, user, or assistant';
    END IF;
    IF p_content IS NULL THEN
        RAISE EXCEPTION 'chat message content is required';
    END IF;

    PERFORM get_or_create_chat_session(p_session_id);
    SELECT COALESCE(MAX(ordinal), -1) + 1
    INTO next_ordinal
    FROM chat_messages
    WHERE session_id = p_session_id;

    INSERT INTO chat_messages (
        session_id,
        ordinal,
        role,
        content,
        metadata,
        source_message_id,
        visible_in_context
    )
    VALUES (
        p_session_id,
        next_ordinal,
        normalized_role,
        p_content,
        COALESCE(p_metadata, '{}'::jsonb),
        NULLIF(btrim(COALESCE(p_source_message_id, '')), ''),
        COALESCE(p_visible_in_context, TRUE)
    )
    RETURNING * INTO row_message;

    UPDATE chat_sessions
    SET updated_at = CURRENT_TIMESTAMP,
        last_active_at = CURRENT_TIMESTAMP
    WHERE id = p_session_id;

    RETURN jsonb_build_object(
        'message_id', row_message.id::text,
        'session_id', row_message.session_id::text,
        'ordinal', row_message.ordinal,
        'role', row_message.role,
        'content', row_message.content,
        'visible_in_context', row_message.visible_in_context,
        'metadata', row_message.metadata,
        'created_at', row_message.created_at
    );
END;
$$;

CREATE OR REPLACE FUNCTION hydrate_chat_session(
    p_session_id UUID,
    p_limit INT DEFAULT NULL,
    p_include_system BOOLEAN DEFAULT FALSE
) RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    row_session chat_sessions%ROWTYPE;
    lim INT := LEAST(
        GREATEST(COALESCE(p_limit, get_config_int('chat.session_history_limit'), 40), 1),
        200
    );
    messages JSONB;
BEGIN
    IF p_session_id IS NULL THEN
        RETURN jsonb_build_object('session_id', NULL, 'messages', '[]'::jsonb, 'count', 0);
    END IF;

    SELECT * INTO row_session
    FROM chat_sessions
    WHERE id = p_session_id
      AND status = 'active';

    IF NOT FOUND THEN
        RETURN jsonb_build_object('session_id', p_session_id::text, 'messages', '[]'::jsonb, 'count', 0);
    END IF;

    WITH selected AS (
        SELECT role, content, ordinal, id, created_at, metadata
        FROM chat_messages
        WHERE session_id = p_session_id
          AND visible_in_context
          AND (p_include_system OR role <> 'system')
        ORDER BY ordinal DESC
        LIMIT lim
    )
    SELECT COALESCE(jsonb_agg(
        jsonb_build_object(
            'role', role,
            'content', content,
            'ordinal', ordinal,
            'message_id', id::text,
            'created_at', created_at,
            'metadata', metadata
        )
        ORDER BY ordinal ASC
    ), '[]'::jsonb)
    INTO messages
    FROM selected;

    RETURN jsonb_build_object(
        'session_id', row_session.id::text,
        'surface', row_session.surface,
        'external_id', row_session.external_id,
        'messages', messages,
        'count', jsonb_array_length(messages),
        'cleared_at', row_session.cleared_at,
        'last_active_at', row_session.last_active_at
    );
END;
$$;

CREATE OR REPLACE FUNCTION list_chat_sessions(
    p_limit INT DEFAULT 20,
    p_surface TEXT DEFAULT NULL,
    p_status TEXT DEFAULT 'active'
) RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    lim INT := LEAST(GREATEST(COALESCE(p_limit, 20), 1), 200);
    normalized_surface TEXT := NULLIF(btrim(COALESCE(p_surface, '')), '');
    normalized_status TEXT := lower(NULLIF(btrim(COALESCE(p_status, '')), ''));
    total_matching INT := 0;
    sessions JSONB := '[]'::jsonb;
BEGIN
    IF normalized_status = 'all' THEN
        normalized_status := NULL;
    END IF;
    IF normalized_status IS NOT NULL AND normalized_status NOT IN ('active', 'archived') THEN
        RAISE EXCEPTION 'chat session status must be active, archived, or all';
    END IF;

    WITH filtered AS (
        SELECT cs.id
        FROM chat_sessions cs
        WHERE (normalized_surface IS NULL OR cs.surface = normalized_surface)
          AND (normalized_status IS NULL OR cs.status = normalized_status)
    )
    SELECT count(*)::int
    INTO total_matching
    FROM filtered;

    WITH selected AS (
        SELECT
            cs.id,
            cs.surface,
            cs.external_id,
            cs.title,
            cs.status,
            cs.metadata,
            cs.created_at,
            cs.updated_at,
            cs.last_active_at,
            cs.cleared_at,
            COALESCE(stats.message_count, 0)::int AS message_count,
            COALESCE(stats.visible_message_count, 0)::int AS visible_message_count,
            stats.first_message_at,
            stats.last_message_at,
            first_user.snippet AS first_user_snippet,
            last_message.snippet AS last_message_snippet,
            last_message.role AS last_message_role
        FROM chat_sessions cs
        LEFT JOIN LATERAL (
            SELECT
                count(*)::int AS message_count,
                count(*) FILTER (WHERE visible_in_context)::int AS visible_message_count,
                min(created_at) AS first_message_at,
                max(created_at) AS last_message_at
            FROM chat_messages cm
            WHERE cm.session_id = cs.id
        ) stats ON TRUE
        LEFT JOIN LATERAL (
            SELECT left(regexp_replace(COALESCE(cm.content, ''), '[[:space:]]+', ' ', 'g'), 240) AS snippet
            FROM chat_messages cm
            WHERE cm.session_id = cs.id
              AND cm.role = 'user'
            ORDER BY cm.ordinal ASC
            LIMIT 1
        ) first_user ON TRUE
        LEFT JOIN LATERAL (
            SELECT
                cm.role,
                left(regexp_replace(COALESCE(cm.content, ''), '[[:space:]]+', ' ', 'g'), 240) AS snippet
            FROM chat_messages cm
            WHERE cm.session_id = cs.id
            ORDER BY cm.ordinal DESC
            LIMIT 1
        ) last_message ON TRUE
        WHERE (normalized_surface IS NULL OR cs.surface = normalized_surface)
          AND (normalized_status IS NULL OR cs.status = normalized_status)
        ORDER BY cs.last_active_at DESC, cs.created_at DESC, cs.id
        LIMIT lim
    )
    SELECT COALESCE(jsonb_agg(
        jsonb_build_object(
            'session_id', id::text,
            'surface', surface,
            'external_id', external_id,
            'title', title,
            'status', status,
            'message_count', message_count,
            'visible_message_count', visible_message_count,
            'first_user_snippet', first_user_snippet,
            'last_message_role', last_message_role,
            'last_message_snippet', last_message_snippet,
            'metadata', metadata,
            'created_at', created_at,
            'updated_at', updated_at,
            'last_active_at', last_active_at,
            'cleared_at', cleared_at,
            'first_message_at', first_message_at,
            'last_message_at', last_message_at
        )
        ORDER BY last_active_at DESC, created_at DESC, id
    ), '[]'::jsonb)
    INTO sessions
    FROM selected;

    RETURN jsonb_build_object(
        'sessions', sessions,
        'count', jsonb_array_length(sessions),
        'total_matching', total_matching,
        'limit', lim,
        'filters', jsonb_build_object(
            'surface', normalized_surface,
            'status', COALESCE(normalized_status, 'all')
        )
    );
END;
$$;

CREATE OR REPLACE FUNCTION get_chat_session_artifact(
    p_session_id UUID,
    p_include_messages BOOLEAN DEFAULT TRUE,
    p_include_hidden BOOLEAN DEFAULT TRUE
) RETURNS JSONB
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    row_session chat_sessions%ROWTYPE;
    messages JSONB := '[]'::jsonb;
    message_count INT := 0;
    visible_message_count INT := 0;
BEGIN
    IF p_session_id IS NULL THEN
        RETURN jsonb_build_object(
            'found', FALSE,
            'session_id', NULL,
            'reason', 'session_id_required'
        );
    END IF;

    SELECT *
    INTO row_session
    FROM chat_sessions
    WHERE id = p_session_id;

    IF NOT FOUND THEN
        RETURN jsonb_build_object(
            'found', FALSE,
            'session_id', p_session_id::text,
            'reason', 'not_found'
        );
    END IF;

    SELECT
        count(*)::int,
        count(*) FILTER (WHERE visible_in_context)::int
    INTO message_count, visible_message_count
    FROM chat_messages
    WHERE session_id = p_session_id;

    IF COALESCE(p_include_messages, TRUE) THEN
        SELECT COALESCE(jsonb_agg(
            jsonb_build_object(
                'message_id', id::text,
                'session_id', session_id::text,
                'ordinal', ordinal,
                'role', role,
                'content', content,
                'visible_in_context', visible_in_context,
                'source_message_id', source_message_id,
                'metadata', metadata,
                'created_at', created_at,
                'updated_at', updated_at
            )
            ORDER BY ordinal ASC
        ), '[]'::jsonb)
        INTO messages
        FROM chat_messages
        WHERE session_id = p_session_id
          AND (COALESCE(p_include_hidden, TRUE) OR visible_in_context);
    END IF;

    RETURN jsonb_build_object(
        'found', TRUE,
        'format', 'hexis.chat_session.v1',
        'exported_at', CURRENT_TIMESTAMP,
        'message_count', message_count,
        'visible_message_count', visible_message_count,
        'include_hidden', COALESCE(p_include_hidden, TRUE),
        'session', jsonb_build_object(
            'session_id', row_session.id::text,
            'surface', row_session.surface,
            'external_id', row_session.external_id,
            'title', row_session.title,
            'status', row_session.status,
            'metadata', row_session.metadata,
            'created_at', row_session.created_at,
            'updated_at', row_session.updated_at,
            'last_active_at', row_session.last_active_at,
            'cleared_at', row_session.cleared_at
        ),
        'messages', messages
    );
END;
$$;

CREATE OR REPLACE FUNCTION set_chat_session_title(
    p_session_id UUID,
    p_title TEXT
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    row_session chat_sessions%ROWTYPE;
BEGIN
    IF p_session_id IS NULL THEN
        RETURN jsonb_build_object(
            'found', FALSE,
            'session_id', NULL,
            'reason', 'session_id_required'
        );
    END IF;

    UPDATE chat_sessions
    SET title = NULLIF(btrim(COALESCE(p_title, '')), ''),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_session_id
    RETURNING * INTO row_session;

    IF NOT FOUND THEN
        RETURN jsonb_build_object(
            'found', FALSE,
            'session_id', p_session_id::text,
            'reason', 'not_found'
        );
    END IF;

    RETURN get_chat_session_artifact(p_session_id, FALSE, TRUE);
END;
$$;

CREATE OR REPLACE FUNCTION fork_chat_session(
    p_source_session_id UUID,
    p_until_ordinal INT DEFAULT NULL,
    p_title TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    source_session chat_sessions%ROWTYPE;
    new_session chat_sessions%ROWTYPE;
    cut_ordinal INT := -1;
    copied_count INT := 0;
    artifact JSONB;
BEGIN
    IF p_source_session_id IS NULL THEN
        RETURN jsonb_build_object(
            'found', FALSE,
            'source_session_id', NULL,
            'reason', 'source_session_id_required'
        );
    END IF;

    SELECT *
    INTO source_session
    FROM chat_sessions
    WHERE id = p_source_session_id;

    IF NOT FOUND THEN
        RETURN jsonb_build_object(
            'found', FALSE,
            'source_session_id', p_source_session_id::text,
            'reason', 'not_found'
        );
    END IF;

    SELECT COALESCE(max(ordinal), -1)
    INTO cut_ordinal
    FROM chat_messages
    WHERE session_id = p_source_session_id;

    IF p_until_ordinal IS NOT NULL THEN
        cut_ordinal := LEAST(cut_ordinal, p_until_ordinal);
    END IF;

    INSERT INTO chat_sessions (
        surface,
        title,
        metadata
    )
    VALUES (
        source_session.surface,
        COALESCE(
            NULLIF(btrim(COALESCE(p_title, '')), ''),
            CASE
                WHEN source_session.title IS NOT NULL THEN source_session.title || ' (fork)'
                ELSE 'Fork of ' || left(source_session.id::text, 8)
            END
        ),
        source_session.metadata
            || jsonb_build_object(
                'forked_from_session_id', source_session.id::text,
                'forked_at', CURRENT_TIMESTAMP,
                'forked_until_ordinal', cut_ordinal
            )
            || COALESCE(p_metadata, '{}'::jsonb)
    )
    RETURNING * INTO new_session;

    INSERT INTO chat_messages (
        session_id,
        ordinal,
        role,
        content,
        visible_in_context,
        source_message_id,
        metadata
    )
    SELECT
        new_session.id,
        ordinal,
        role,
        content,
        visible_in_context,
        source_message_id,
        metadata || jsonb_build_object('forked_from_message_id', id::text)
    FROM chat_messages
    WHERE session_id = p_source_session_id
      AND ordinal <= cut_ordinal
    ORDER BY ordinal ASC;
    GET DIAGNOSTICS copied_count = ROW_COUNT;

    artifact := get_chat_session_artifact(new_session.id, TRUE, TRUE);
    RETURN artifact || jsonb_build_object(
        'source_session_id', source_session.id::text,
        'forked_message_count', copied_count
    );
END;
$$;

CREATE OR REPLACE FUNCTION record_chat_session_turn(
    p_session_id UUID,
    p_user_text TEXT,
    p_assistant_text TEXT,
    p_surface TEXT DEFAULT 'chat',
    p_context JSONB DEFAULT '{}'::jsonb
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    session_payload JSONB;
    user_message JSONB := NULL;
    assistant_message JSONB := NULL;
    memory_result JSONB := '{}'::jsonb;
    ctx JSONB := COALESCE(p_context, '{}'::jsonb);
    source_identity TEXT := NULLIF(ctx->>'source_identity', '');
BEGIN
    IF p_session_id IS NULL THEN
        RAISE EXCEPTION 'session_id is required';
    END IF;
    session_payload := get_or_create_chat_session(
        p_session_id,
        COALESCE(NULLIF(p_surface, ''), ctx->>'surface', 'chat'),
        NULLIF(ctx->>'external_id', ''),
        COALESCE(ctx->'session_metadata', '{}'::jsonb)
    );

    IF NULLIF(COALESCE(p_user_text, ''), '') IS NOT NULL THEN
        user_message := append_chat_message(
            p_session_id,
            'user',
            p_user_text,
            COALESCE(ctx->'user_metadata', '{}'::jsonb),
            NULLIF(ctx->>'user_source_message_id', ''),
            TRUE
        );
    END IF;

    IF NULLIF(COALESCE(p_assistant_text, ''), '') IS NOT NULL THEN
        assistant_message := append_chat_message(
            p_session_id,
            'assistant',
            p_assistant_text,
            COALESCE(ctx->'assistant_metadata', '{}'::jsonb),
            NULLIF(ctx->>'assistant_source_message_id', ''),
            TRUE
        );
    END IF;

    IF COALESCE(p_user_text, '') <> '' OR COALESCE(p_assistant_text, '') <> '' THEN
        BEGIN
            memory_result := record_chat_turn_memory(
                p_user_text,
                p_assistant_text,
                p_session_id::text,
                source_identity,
                ctx
            );
        EXCEPTION WHEN OTHERS THEN
            memory_result := jsonb_build_object(
                'status', 'failed',
                'error', SQLERRM,
                'short_term_history_preserved', TRUE
            );
        END;
    END IF;

    RETURN jsonb_build_object(
        'session', session_payload,
        'user_message', user_message,
        'assistant_message', assistant_message,
        'memory', memory_result,
        'history', hydrate_chat_session(p_session_id)
    );
END;
$$;

CREATE OR REPLACE FUNCTION clear_chat_session_context(
    p_session_id UUID,
    p_reason TEXT DEFAULT NULL
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    affected INT;
BEGIN
    IF p_session_id IS NULL THEN
        RAISE EXCEPTION 'session_id is required';
    END IF;

    UPDATE chat_messages
    SET visible_in_context = FALSE,
        metadata = metadata || jsonb_build_object(
            'cleared_from_context_at', CURRENT_TIMESTAMP,
            'clear_reason', COALESCE(NULLIF(p_reason, ''), 'user_request')
        ),
        updated_at = CURRENT_TIMESTAMP
    WHERE session_id = p_session_id
      AND visible_in_context;
    GET DIAGNOSTICS affected = ROW_COUNT;

    UPDATE chat_sessions
    SET cleared_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_session_id;

    RETURN jsonb_build_object(
        'session_id', p_session_id::text,
        'cleared_messages', affected,
        'long_term_memory_preserved', TRUE
    );
END;
$$;

CREATE OR REPLACE FUNCTION render_chat_continuity_context(
    p_session_id TEXT DEFAULT NULL,
    p_exclude_sensitive BOOLEAN DEFAULT FALSE
) RETURNS TEXT
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    current_session UUID := _db_brain_try_uuid(p_session_id);
    lim INT := LEAST(GREATEST(COALESCE(get_config_int('chat.recent_carryover_limit'), 8), 0), 20);
    window_minutes INT := LEAST(GREATEST(COALESCE(get_config_int('chat.recent_carryover_window_minutes'), 1440), 1), 43200);
    max_chars INT := LEAST(GREATEST(COALESCE(get_config_int('chat.recent_carryover_max_chars'), 5000), 500), 20000);
    summary_lim INT := LEAST(GREATEST(COALESCE(get_config_int('chat.continuity_summary_limit'), 3), 0), 8);
    correction_lim INT := LEAST(GREATEST(COALESCE(get_config_int('chat.continuity_correction_limit'), 5), 0), 12);
    heartbeat_lim INT := LEAST(GREATEST(COALESCE(get_config_int('chat.continuity_heartbeat_limit'), 3), 0), 10);
    defect_lim INT := LEAST(GREATEST(COALESCE(get_config_int('chat.continuity_defect_limit'), 5), 0), 20);
    injury_lines TEXT;
    affect_line TEXT;
    affect_state JSONB;
    summary_lines TEXT;
    correction_lines TEXT;
    heartbeat_lines TEXT;
    defect_lines TEXT;
    turn_lines TEXT;
    body TEXT;
BEGIN
    IF p_exclude_sensitive THEN
        RETURN '';
    END IF;

    affect_state := get_current_affective_state();
    affect_line := '- Current affect: '
        || COALESCE(NULLIF(affect_state->>'primary_emotion', ''), NULLIF(affect_state->>'feeling', ''), 'unknown')
        || ', valence=' || COALESCE(affect_state->>'valence', '?')
        || ', arousal=' || COALESCE(affect_state->>'arousal', '?')
        || ', intensity=' || COALESCE(affect_state->>'intensity', '?');

    SELECT string_agg(
        '- ' || m.content
        || ' [unresolved; last evidence '
        || COALESCE(to_char((m.metadata#>>'{relationship_state,last_evidence_at}')::timestamptz, 'YYYY-MM-DD HH24:MI TZ'), 'unknown')
        || ']',
        E'\n' ORDER BY m.updated_at DESC, m.id
    )
    INTO injury_lines
    FROM (
        SELECT *
        FROM memories
        WHERE type = 'semantic'
          AND status = 'active'
          AND metadata#>>'{relationship_state,kind}' = 'relationship_injury'
          AND metadata#>>'{relationship_state,status}' = 'unresolved'
        ORDER BY updated_at DESC
        LIMIT 3
    ) m;

    WITH summaries AS (
        SELECT content, created_at
        FROM memories
        WHERE summary_lim > 0
          AND type = 'episodic'
          AND status = 'active'
          AND metadata ? 'recmem'
          AND created_at >= CURRENT_TIMESTAMP - (window_minutes * INTERVAL '1 minute')
          AND COALESCE(source_attribution->>'sensitivity', '') <> 'private'
        ORDER BY created_at DESC, id DESC
        LIMIT summary_lim
    )
    SELECT string_agg(
        '- [' || to_char(created_at, 'YYYY-MM-DD HH24:MI TZ') || '] '
        || left(regexp_replace(COALESCE(content, ''), '[[:space:]]+', ' ', 'g'), 700),
        E'\n' ORDER BY created_at ASC
    )
    INTO summary_lines
    FROM summaries;

    WITH corrections AS (
        SELECT content, metadata, updated_at, created_at
        FROM memories
        WHERE correction_lim > 0
          AND status = 'active'
          AND (
              metadata->>'invalid_precedent' = 'true'
              OR metadata ? 'latest_correction'
          )
          AND COALESCE(source_attribution->>'sensitivity', '') <> 'private'
        ORDER BY updated_at DESC, created_at DESC
        LIMIT correction_lim
    )
    SELECT string_agg(
        '- ' || left(regexp_replace(COALESCE(content, ''), '[[:space:]]+', ' ', 'g'), 420)
        || CASE WHEN metadata->>'invalid_precedent' = 'true'
                THEN E'\n  status: invalid precedent; do not imitate this behavior'
                ELSE '' END
        || CASE WHEN NULLIF(metadata#>>'{latest_correction,correction}', '') IS NOT NULL
                THEN E'\n  correction: ' || left(regexp_replace(metadata#>>'{latest_correction,correction}', '[[:space:]]+', ' ', 'g'), 420)
                ELSE '' END,
        E'\n' ORDER BY updated_at ASC, created_at ASC
    )
    INTO correction_lines
    FROM corrections;

    WITH heartbeats AS (
        SELECT
            m.id,
            COALESCE((m.metadata->>'event_time')::timestamptz, m.created_at) AS event_time,
            NULLIF(m.metadata#>>'{context,heartbeat_number}', '') AS heartbeat_number,
            m.content,
            NULLIF(m.metadata#>>'{context,reasoning}', '') AS reasoning,
            COALESCE(m.metadata#>'{context,actions_taken}', '[]'::jsonb) AS actions_taken
        FROM memories m
        WHERE heartbeat_lim > 0
          AND m.type = 'episodic'
          AND m.status = 'active'
          AND m.metadata#>>'{context,heartbeat_id}' IS NOT NULL
          AND COALESCE((m.metadata->>'event_time')::timestamptz, m.created_at) >= CURRENT_TIMESTAMP - (window_minutes * INTERVAL '1 minute')
          AND COALESCE(m.source_attribution->>'sensitivity', '') <> 'private'
        ORDER BY COALESCE((m.metadata->>'event_time')::timestamptz, m.created_at) DESC, m.id DESC
        LIMIT heartbeat_lim
    ),
    rendered AS (
        SELECT
            h.id,
            h.event_time,
            '- [' || to_char(h.event_time, 'YYYY-MM-DD HH24:MI TZ') || '] Heartbeat #'
            || COALESCE(h.heartbeat_number, '?') || E'\n'
            || '  summary: ' || left(regexp_replace(COALESCE(h.content, ''), '[[:space:]]+', ' ', 'g'), 520)
            || CASE WHEN failed.failures IS NOT NULL
                    THEN E'\n  failures: ' || failed.failures
                    ELSE '' END
            || CASE WHEN h.reasoning IS NOT NULL
                    THEN E'\n  note: ' || left(regexp_replace(h.reasoning, '[[:space:]]+', ' ', 'g'), 700)
                    ELSE '' END AS line
        FROM heartbeats h
        LEFT JOIN LATERAL (
            SELECT string_agg(
                COALESCE(a->>'action', 'unknown_action') || ': '
                || COALESCE(
                    NULLIF(a#>>'{result,error}', ''),
                    NULLIF(a#>>'{result,output_preview}', ''),
                    'failed'
                ),
                '; ' ORDER BY ord
            ) AS failures
            FROM jsonb_array_elements(h.actions_taken) WITH ORDINALITY AS elem(a, ord)
            WHERE COALESCE((a#>>'{result,success}')::boolean, FALSE) = FALSE
        ) failed ON TRUE
    )
    SELECT string_agg(line, E'\n' ORDER BY event_time ASC, id)
    INTO heartbeat_lines
    FROM rendered;

    WITH defects AS (
        SELECT *
        FROM defect_reports
        WHERE defect_lim > 0
          AND status IN ('open', 'diagnosed', 'repair_proposed')
        ORDER BY
            CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
            last_seen_at DESC
        LIMIT defect_lim
    )
    SELECT string_agg(
        '- [' || severity || '] ' || title
        || ' (' || occurrence_count || ' occurrence' || CASE WHEN occurrence_count = 1 THEN '' ELSE 's' END || '; status=' || status || ')' || E'\n'
        || '  summary: ' || left(regexp_replace(COALESCE(summary, ''), '[[:space:]]+', ' ', 'g'), 360) || E'\n'
        || '  latest error: ' || left(regexp_replace(COALESCE(last_error, ''), '[[:space:]]+', ' ', 'g'), 360)
        || CASE WHEN jsonb_typeof(diagnosis) = 'object' AND diagnosis <> '{}'::jsonb
                THEN E'\n  diagnosis: ' || left(regexp_replace(COALESCE(diagnosis->>'hypothesis', ''), '[[:space:]]+', ' ', 'g'), 360)
                ELSE '' END,
        E'\n' ORDER BY
            CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
            last_seen_at DESC
    )
    INTO defect_lines
    FROM defects;

    WITH recent AS (
        SELECT
            s.id,
            s.turn_at,
            s.user_text,
            s.assistant_text,
            s.metadata->'emotional_context' AS affect,
            COALESCE(cs.surface, 'conversation') AS surface
        FROM subconscious_units s
        LEFT JOIN chat_sessions cs ON cs.id = s.session_id
        WHERE lim > 0
          AND s.status = 'active'
          AND COALESCE(s.metadata#>>'{recmem,kind}', '') <> 'source_document_desk'
          AND COALESCE(s.metadata->>'type', 'conversation') = 'conversation'
          AND (current_session IS NULL OR s.session_id IS DISTINCT FROM current_session)
          AND s.turn_at >= CURRENT_TIMESTAMP - (window_minutes * INTERVAL '1 minute')
          AND COALESCE(s.source_attribution->>'sensitivity', '') <> 'private'
          AND COALESCE(cs.surface, 'api') = ANY(ARRAY['api','web','chat','cli','tui','openai_compat'])
        ORDER BY s.turn_at DESC, s.id DESC
        LIMIT lim
    )
    SELECT string_agg(
        '- [' || to_char(turn_at, 'YYYY-MM-DD HH24:MI TZ') || '] '
        || surface || E'\n'
        || '  user: ' || left(regexp_replace(COALESCE(user_text, ''), '[[:space:]]+', ' ', 'g'), 500) || E'\n'
        || '  assistant: ' || left(regexp_replace(COALESCE(assistant_text, ''), '[[:space:]]+', ' ', 'g'), 500)
        || CASE WHEN jsonb_typeof(affect) = 'object'
                THEN E'\n  affect: ' || COALESCE(affect->>'primary_emotion', 'unknown')
                    || ', valence=' || COALESCE(affect->>'valence', '?')
                    || ', intensity=' || COALESCE(affect->>'intensity', '?')
                ELSE '' END,
        E'\n' ORDER BY turn_at ASC, id
    )
    INTO turn_lines
    FROM recent;

    IF injury_lines IS NULL
       AND summary_lines IS NULL
       AND correction_lines IS NULL
       AND heartbeat_lines IS NULL
       AND defect_lines IS NULL
       AND turn_lines IS NULL
       AND affect_line IS NULL THEN
        RETURN '';
    END IF;

    body := '## Conversation Continuity Packet' || E'\n'
        || 'This is active working-state continuity, not decorative RAG. Treat it as what you remember right now: recent raw turns, exchange summaries, affect, corrections, unresolved relationship weather, and unresolved substrate defects. Do not claim a prior exchange is unavailable merely because the UI opened a new session. Do not reset warmth or trust merely because the session id changed; let sincere repair change the state when there is evidence.' || E'\n';
    IF affect_line IS NOT NULL THEN
        body := body || E'\n' || '### Current Emotional State' || E'\n' || affect_line || E'\n';
    END IF;
    IF injury_lines IS NOT NULL THEN
        body := body || E'\n' || '### Unresolved Relationship Injuries' || E'\n' || injury_lines || E'\n';
    END IF;
    IF correction_lines IS NOT NULL THEN
        body := body || E'\n' || '### Active Corrections And Invalidated Precedents' || E'\n' || correction_lines || E'\n';
    END IF;
    IF summary_lines IS NOT NULL THEN
        body := body || E'\n' || '### Recent Exchange Summaries' || E'\n' || summary_lines || E'\n';
    END IF;
    IF heartbeat_lines IS NOT NULL THEN
        body := body || E'\n' || '### Recent Autonomous Heartbeats' || E'\n'
            || 'Recent autonomous cycles are part of what has been on your mind. If a cycle failed or left unresolved work, surface it naturally when relevant instead of waiting for the user to ask.' || E'\n'
            || heartbeat_lines || E'\n';
    END IF;
    IF defect_lines IS NOT NULL THEN
        body := body || E'\n' || '### Unresolved Software Defects' || E'\n'
            || 'These are defects observed in your own substrate. Treat them as operational responsibilities: do not dramatize them, but bring them up naturally when they affect the current conversation or when the user asks how you are doing.' || E'\n'
            || defect_lines || E'\n';
    END IF;
    IF turn_lines IS NOT NULL THEN
        body := body || E'\n' || '### Recent Prior Turns' || E'\n' || turn_lines;
    END IF;

    IF length(body) > max_chars THEN
        body := left(body, max_chars) || E'\n[truncated chat continuity packet]';
    END IF;
    RETURN body;
END;
$$;

CREATE OR REPLACE FUNCTION render_recent_conversation_carryover(
    p_session_id TEXT DEFAULT NULL,
    p_exclude_sensitive BOOLEAN DEFAULT FALSE
) RETURNS TEXT
LANGUAGE sql
STABLE
AS $$
    SELECT render_chat_continuity_context(p_session_id, p_exclude_sensitive)
$$;
