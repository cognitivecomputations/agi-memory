-- Journal functions: write / read / search + a DB-native tool dispatcher.
-- The journal is deliberate, permanent, and OUTSIDE the memory substrate
-- (docs/memory_retention_design.md §7). None of these are ever called from the
-- passive recall/context path — only from the explicit journal tools.
SET search_path = public, ag_catalog, "$user";
SET check_function_bodies = off;

-- Current life-chapter name from the AGE graph (brain-retained), to snapshot onto
-- an entry. NULL if unavailable. Mirrors the read in get_narrative_context().
CREATE OR REPLACE FUNCTION current_life_chapter_name()
RETURNS TEXT
LANGUAGE plpgsql STABLE
AS $$
DECLARE
    v_name TEXT;
BEGIN
    SELECT NULLIF(replace(name_raw::text, '"', ''), 'null')
      INTO v_name
      FROM ag_catalog.cypher('memory_graph', $q$
          MATCH (c:LifeChapterNode {key: 'current'})
          RETURN c.name
          LIMIT 1
      $q$) as (name_raw ag_catalog.agtype);
    RETURN v_name;
EXCEPTION WHEN OTHERS THEN
    RETURN NULL;
END;
$$;

-- Write a journal entry — a deliberate, permanent act. Embeds the content for
-- later search_journal (entry is still saved if embedding fails). Returns the id.
CREATE OR REPLACE FUNCTION write_journal_entry(
    p_content  TEXT,
    p_title    TEXT DEFAULT NULL,
    p_mood     TEXT DEFAULT NULL,
    p_tags     TEXT[] DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    v_id UUID;
    v_embedding vector;
BEGIN
    IF p_content IS NULL OR btrim(p_content) = '' THEN
        RAISE EXCEPTION 'journal entry content must not be empty';
    END IF;
    BEGIN
        v_embedding := (get_embedding(ARRAY[p_content]))[1];
    EXCEPTION WHEN OTHERS THEN
        v_embedding := NULL;
    END;
    INSERT INTO journal_entries (content, title, mood, tags, chapter, embedding, metadata)
    VALUES (
        p_content,
        NULLIF(btrim(COALESCE(p_title, '')), ''),
        p_mood,
        p_tags,
        current_life_chapter_name(),
        v_embedding,
        COALESCE(p_metadata, '{}'::jsonb)
    )
    RETURNING id INTO v_id;
    RETURN v_id;
END;
$$;

-- Read a specific entry by id, or the most recent N. Deliberate — not recall.
CREATE OR REPLACE FUNCTION read_journal_entries(
    p_id UUID DEFAULT NULL,
    p_limit INT DEFAULT 5
) RETURNS TABLE (
    id UUID, written_at TIMESTAMPTZ, chapter TEXT, title TEXT, content TEXT, mood TEXT, tags TEXT[]
)
LANGUAGE sql STABLE
AS $$
    SELECT id, written_at, chapter, title, content, mood, tags
    FROM journal_entries
    WHERE p_id IS NULL OR id = p_id
    ORDER BY written_at DESC
    LIMIT CASE WHEN p_id IS NOT NULL THEN 1 ELSE GREATEST(COALESCE(p_limit, 5), 1) END;
$$;

-- Search the journal by meaning — a deliberate lookup, never a passive recall.
CREATE OR REPLACE FUNCTION search_journal(
    p_query TEXT,
    p_limit INT DEFAULT 5
) RETURNS TABLE (
    id UUID, written_at TIMESTAMPTZ, chapter TEXT, title TEXT, content TEXT, similarity FLOAT
)
LANGUAGE plpgsql
AS $$
DECLARE
    q vector;
BEGIN
    q := (get_embedding(ARRAY[ensure_embedding_prefix(p_query, 'search_query')]))[1];
    RETURN QUERY
    SELECT j.id, j.written_at, j.chapter, j.title, j.content,
           (1 - (j.embedding <=> q))::float AS similarity
    FROM journal_entries j
    WHERE j.embedding IS NOT NULL
    ORDER BY j.embedding <=> q
    LIMIT GREATEST(COALESCE(p_limit, 5), 1);
END;
$$;

-- DB-native tool dispatcher (mirrors execute_memory_tool, db/38). Returns
-- tool_success / tool_error so the Python handlers can run in-DB first.
CREATE OR REPLACE FUNCTION execute_journal_tool(p_tool_name TEXT, p_args JSONB)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_id UUID;
    v_rows JSONB;
    v_read_id UUID;
BEGIN
    IF p_tool_name = 'write_journal' THEN
        IF COALESCE(btrim(p_args->>'content'), '') = '' THEN
            RETURN tool_error('journal entry content must not be empty', 'invalid_params');
        END IF;
        v_id := write_journal_entry(
            p_args->>'content',
            p_args->>'title',
            p_args->>'mood',
            CASE WHEN p_args ? 'tags' THEN ARRAY(SELECT jsonb_array_elements_text(p_args->'tags')) ELSE NULL END
        );
        RETURN tool_success(jsonb_build_object('entry_id', v_id),
                            'Wrote a journal entry: ' || left(p_args->>'content', 60));

    ELSIF p_tool_name = 'read_journal' THEN
        v_read_id := NULLIF(p_args->>'id', '')::uuid;
        SELECT COALESCE(jsonb_agg(to_jsonb(r)), '[]'::jsonb) INTO v_rows
        FROM read_journal_entries(v_read_id, COALESCE((p_args->>'limit')::int, 5)) r;
        -- Re-reading a SPECIFIC entry is a fresh experience: it forms a NEW episodic
        -- memory (which then fades like any other), even if the memory of writing it
        -- has long since faded -- the entry itself is permanent, the reading is not.
        -- (Browsing the recent list is not a revisiting, so it forms no memory.)
        IF v_read_id IS NOT NULL AND jsonb_array_length(v_rows) = 1 THEN
            BEGIN
                PERFORM create_episodic_memory(
                    p_content := 'I revisited my journal entry'
                        || COALESCE(' "' || NULLIF(v_rows->0->>'title', '') || '"', '')
                        || '. ' || left(COALESCE(v_rows->0->>'content', ''), 400),
                    p_context := jsonb_build_object('activity', 'rereading_journal', 'journal_entry_id', v_read_id),
                    p_importance := 0.4);
            EXCEPTION WHEN OTHERS THEN
                NULL;  -- the read still succeeds even if forming the memory hiccups
            END;
        END IF;
        RETURN tool_success(jsonb_build_object('entries', v_rows),
                            'Read ' || jsonb_array_length(v_rows) || ' journal entry(ies)');

    ELSIF p_tool_name = 'search_journal' THEN
        IF COALESCE(btrim(p_args->>'query'), '') = '' THEN
            RETURN tool_error('search query must not be empty', 'invalid_params');
        END IF;
        SELECT COALESCE(jsonb_agg(to_jsonb(r)), '[]'::jsonb) INTO v_rows
        FROM search_journal(p_args->>'query', COALESCE((p_args->>'limit')::int, 5)) r;
        RETURN tool_success(jsonb_build_object('entries', v_rows),
                            'Found ' || jsonb_array_length(v_rows) || ' matching journal entry(ies)');
    END IF;

    RETURN tool_error('Unsupported journal tool: ' || COALESCE(p_tool_name, '<null>'), 'invalid_params');
END;
$$;

-- The journal is deliberate by design (see header) -- but "deliberate" has meant
-- "never" in practice: write_journal has fired zero times live even on days the
-- agent had plenty to say (issue #114). This is the one exception: called from
-- subconscious maintenance, it steps in ONLY once, late in the local day, and
-- ONLY when there was meaningful activity (real episodic memories) and nothing
-- was journaled deliberately. It never fires twice for the same local date and
-- never substitutes for a deliberate entry that already exists.
CREATE OR REPLACE FUNCTION maybe_write_daily_journal_fallback()
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_tz TEXT := COALESCE(
        NULLIF(get_config_text('agent.timezone'), ''),
        NULLIF(get_config_text('heartbeat.timezone'), ''),
        'UTC'
    );
    v_local_ts TIMESTAMP;
    v_local_date DATE;
    v_local_hour INT;
    v_state JSONB;
    v_already_journaled BOOLEAN;
    v_min_count INT;
    v_count INT;
    v_samples TEXT[];
    v_content TEXT;
    v_entry_id UUID;
BEGIN
    IF NOT COALESCE(get_config_bool('maintenance.daily_journal_fallback_enabled'), true) THEN
        RETURN jsonb_build_object('skipped', true, 'reason', 'disabled');
    END IF;

    BEGIN
        v_local_ts := CURRENT_TIMESTAMP AT TIME ZONE v_tz;
    EXCEPTION WHEN OTHERS THEN
        v_local_ts := CURRENT_TIMESTAMP AT TIME ZONE 'UTC';
    END;
    v_local_date := v_local_ts::date;
    v_local_hour := EXTRACT(HOUR FROM v_local_ts)::int;

    -- Give the agent the whole local day to journal deliberately before this
    -- fallback ever considers stepping in.
    IF v_local_hour < COALESCE(get_config_int('maintenance.daily_journal_fallback_hour'), 21) THEN
        RETURN jsonb_build_object('skipped', true, 'reason', 'too_early', 'local_hour', v_local_hour);
    END IF;

    v_state := COALESCE(get_state('daily_journal_fallback'), '{}'::jsonb);
    IF v_state->>'last_checked_date' = v_local_date::text THEN
        RETURN jsonb_build_object('skipped', true, 'reason', 'already_checked_today');
    END IF;

    -- Record the attempt regardless of outcome -- a quiet day, or one already
    -- journaled, should not be re-evaluated every maintenance tick until midnight.
    PERFORM set_state('daily_journal_fallback', jsonb_build_object('last_checked_date', v_local_date::text));

    SELECT EXISTS (
        SELECT 1 FROM journal_entries
        WHERE (written_at AT TIME ZONE v_tz)::date = v_local_date
    ) INTO v_already_journaled;
    IF v_already_journaled THEN
        RETURN jsonb_build_object('skipped', true, 'reason', 'already_journaled', 'date', v_local_date);
    END IF;

    v_min_count := GREATEST(COALESCE(get_config_int('maintenance.daily_journal_fallback_min_episodes'), 3), 1);
    SELECT count(*)::int INTO v_count
    FROM memories
    WHERE type = 'episodic'
      AND status = 'active'
      AND importance >= 0.5
      AND (created_at AT TIME ZONE v_tz)::date = v_local_date;

    IF v_count < v_min_count THEN
        RETURN jsonb_build_object('skipped', true, 'reason', 'not_enough_activity',
                                   'episodes', v_count, 'threshold', v_min_count);
    END IF;

    SELECT array_agg(top.snippet) INTO v_samples
    FROM (
        SELECT left(content, 140) AS snippet
        FROM memories
        WHERE type = 'episodic'
          AND status = 'active'
          AND importance >= 0.5
          AND (created_at AT TIME ZONE v_tz)::date = v_local_date
        ORDER BY importance DESC, created_at DESC
        LIMIT 5
    ) top;

    v_content := 'A day worth remembering, though I never stopped to write it myself: '
        || array_to_string(v_samples, ' ~ ');

    v_entry_id := write_journal_entry(
        p_content  := v_content,
        p_title    := 'Auto-noted -- ' || to_char(v_local_date, 'FMMonth DD, YYYY'),
        p_tags     := ARRAY['auto-generated'],
        p_metadata := jsonb_build_object(
            'source', 'subconscious_daily_fallback',
            'episode_count', v_count,
            'date', v_local_date::text
        )
    );

    RETURN jsonb_build_object(
        'journaled', true, 'entry_id', v_entry_id,
        'episode_count', v_count, 'date', v_local_date
    );
END;
$$;
