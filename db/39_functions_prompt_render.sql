-- DB-owned prompt rendering: JSONB context -> markdown prompt text.
-- Ports services/heartbeat_prompt.py (build_heartbeat_decision_prompt + _format_*)
-- so the heartbeat/decision prompt can be assembled entirely in the DB from
-- gather_turn_context(). Presentation logic, kept deterministic in the brain.
--
-- Parity note: aims for SEMANTIC parity with the Python formatters, not byte
-- parity — embedded JSON spacing/key-order (json.dumps vs jsonb::text) and
-- float rounding (Python :.2f banker's vs to_char half-up) may differ at the
-- margins; these are LLM-irrelevant.
SET search_path = public, ag_catalog, "$user";

-- Format a numeric like Python's f"{v:.Nf}" (2 decimals by default).
CREATE OR REPLACE FUNCTION _pr_f(v numeric, d int DEFAULT 2)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    SELECT CASE
        WHEN v IS NULL THEN NULL
        ELSE to_char(v, 'FM999999999990.' || repeat('0', GREATEST(d, 1)))
    END;
$$;

-- True when a jsonb value is a JSON number (mirrors isinstance(x,(int,float))).
CREATE OR REPLACE FUNCTION _pr_is_num(v jsonb)
RETURNS boolean LANGUAGE sql IMMUTABLE AS $$
    SELECT v IS NOT NULL AND jsonb_typeof(v) = 'number';
$$;

-- Coerce to a JSON array or empty array (mirrors "if not isinstance(x, list)").
CREATE OR REPLACE FUNCTION _pr_arr(v jsonb)
RETURNS jsonb LANGUAGE sql IMMUTABLE AS $$
    SELECT CASE WHEN v IS NOT NULL AND jsonb_typeof(v) = 'array' THEN v ELSE '[]'::jsonb END;
$$;

-- Render a dynamic sub-knowledge-graph (build_context_subgraph output:
-- {nodes, edges}) as compact markdown: each typed edge as "A  —rel→  B" using
-- node labels, grouped by relation. Returns NULL when there is no structure to
-- show, so callers can omit the section entirely.
CREATE OR REPLACE FUNCTION render_subgraph(p jsonb)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    WITH nodes AS (
        SELECT (n->>'type') AS ntype, (n->>'id') AS nid,
               left(COALESCE(NULLIF(btrim(n->>'label'), ''), n->>'id'), 80) AS label
        FROM jsonb_array_elements(_pr_arr(p->'nodes')) n
    ),
    edges AS (
        SELECT (e->>'src_type') AS st, (e->>'src_id') AS si, (e->>'rel') AS rel,
               (e->>'dst_type') AS dt, (e->>'dst_id') AS di
        FROM jsonb_array_elements(_pr_arr(p->'edges')) e
    )
    SELECT string_agg(
        '  - ' || COALESCE(ns.label, e.si) || '  —' || lower(e.rel) || '→  ' || COALESCE(nd.label, e.di),
        E'\n' ORDER BY e.rel, ns.label, nd.label)
    FROM edges e
    LEFT JOIN nodes ns ON ns.ntype = e.st AND ns.nid = e.si
    LEFT JOIN nodes nd ON nd.ntype = e.dt AND nd.nid = e.di;
$$;

-- --- list formatters --------------------------------------------------------

CREATE OR REPLACE FUNCTION render_goals(p_goals jsonb)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    SELECT COALESCE(
        string_agg('  - ' || COALESCE(g->>'title', 'Untitled'), E'\n' ORDER BY ord),
        '  (none)')
    FROM jsonb_array_elements(_pr_arr(p_goals)) WITH ORDINALITY AS t(g, ord);
$$;

CREATE OR REPLACE FUNCTION render_issues(p_issues jsonb)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    SELECT COALESCE(
        string_agg('  - ' || COALESCE(i->>'title', 'Unknown') || ': '
                   || COALESCE(i->>'issue', 'unknown issue'), E'\n' ORDER BY ord),
        '  (none)')
    FROM jsonb_array_elements(_pr_arr(p_issues)) WITH ORDINALITY AS t(i, ord);
$$;

CREATE OR REPLACE FUNCTION render_memories(p_memories jsonb)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    SELECT COALESCE(
        string_agg('  - ' || left(COALESCE(m->>'content', ''), 100) || '...', E'\n' ORDER BY ord),
        '  (no recent memories)')
    FROM (
        SELECT m, ord FROM jsonb_array_elements(_pr_arr(p_memories)) WITH ORDINALITY AS t(m, ord)
        ORDER BY ord LIMIT 5
    ) s;
$$;

CREATE OR REPLACE FUNCTION render_identity(p_identity jsonb)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    SELECT COALESCE(
        string_agg('  - ' || COALESCE(i->>'type', 'unknown') || ': '
                   || left(COALESCE(i->'content', '{}'::jsonb)::text, 100), E'\n' ORDER BY ord),
        '  (no identity aspects defined)')
    FROM (
        SELECT i, ord FROM jsonb_array_elements(_pr_arr(p_identity)) WITH ORDINALITY AS t(i, ord)
        ORDER BY ord LIMIT 3
    ) s;
$$;

-- objectives/guardrails/tools share a shape: list of strings OR {title/name, description}.
CREATE OR REPLACE FUNCTION _pr_named_list(p_items jsonb, p_limit int, p_title_keys text[], p_default_title text)
RETURNS text LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
    item jsonb;
    ord int := 0;
    lines text[] := ARRAY[]::text[];
    title text;
    v_desc text;
    k text;
BEGIN
    IF jsonb_typeof(COALESCE(p_items, 'null'::jsonb)) <> 'array' OR jsonb_array_length(p_items) = 0 THEN
        RETURN '  (none)';
    END IF;
    FOR item IN SELECT * FROM jsonb_array_elements(p_items) LOOP
        ord := ord + 1;
        IF ord > p_limit THEN EXIT; END IF;
        IF jsonb_typeof(item) = 'string' THEN
            lines := lines || ('  - ' || (item #>> '{}'));
        ELSIF jsonb_typeof(item) = 'object' THEN
            title := NULL;
            FOREACH k IN ARRAY p_title_keys LOOP
                IF title IS NULL AND NULLIF(item->>k, '') IS NOT NULL THEN
                    title := item->>k;
                END IF;
            END LOOP;
            title := COALESCE(title, p_default_title);
            v_desc := COALESCE(NULLIF(item->>'description', ''), NULLIF(item->>'details', ''), '');
            lines := lines || ('  - ' || title || CASE WHEN v_desc <> '' THEN ': ' || v_desc ELSE '' END);
        END IF;
    END LOOP;
    IF array_length(lines, 1) IS NULL THEN RETURN '  (none)'; END IF;
    RETURN array_to_string(lines, E'\n');
END;
$$;

CREATE OR REPLACE FUNCTION render_objectives(p jsonb)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    SELECT _pr_named_list(p, 8, ARRAY['title', 'name'], 'Objective');
$$;

CREATE OR REPLACE FUNCTION render_guardrails(p jsonb)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    SELECT _pr_named_list(p, 10, ARRAY['name'], 'guardrail');
$$;

CREATE OR REPLACE FUNCTION render_tools(p jsonb)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    SELECT _pr_named_list(p, 10, ARRAY['name'], 'tool');
$$;

CREATE OR REPLACE FUNCTION render_narrative(p_narrative jsonb)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    SELECT CASE
        WHEN p_narrative IS NULL OR jsonb_typeof(p_narrative) <> 'object' THEN '  (none)'
        ELSE '  - Current chapter: ' || COALESCE(
            NULLIF(CASE WHEN jsonb_typeof(p_narrative->'current_chapter') = 'object'
                        THEN p_narrative->'current_chapter'->>'name' END, ''),
            'Foundations')
    END;
$$;

CREATE OR REPLACE FUNCTION render_self_model(p_items jsonb)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    SELECT COALESCE(
        string_agg('  - ' || COALESCE(x->>'kind', 'associated') || ': '
                   || COALESCE(x->>'concept', '?')
                   || CASE WHEN _pr_is_num(x->'strength')
                           THEN ' (' || _pr_f((x->>'strength')::numeric) || ')' ELSE '' END,
                   E'\n' ORDER BY ord),
        '  (empty)')
    FROM (
        SELECT x, ord FROM jsonb_array_elements(_pr_arr(p_items)) WITH ORDINALITY AS t(x, ord)
        WHERE jsonb_typeof(x) = 'object'
        ORDER BY ord LIMIT 8
    ) s;
$$;

CREATE OR REPLACE FUNCTION render_relationships(p_items jsonb)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    SELECT COALESCE(
        string_agg('  - ' || COALESCE(x->>'entity', 'unknown')
                   || CASE WHEN _pr_is_num(x->'strength')
                           THEN ' (' || _pr_f((x->>'strength')::numeric) || ')' ELSE '' END,
                   E'\n' ORDER BY ord),
        '  (none)')
    FROM (
        SELECT x, ord FROM jsonb_array_elements(_pr_arr(p_items)) WITH ORDINALITY AS t(x, ord)
        WHERE jsonb_typeof(x) = 'object'
        ORDER BY ord LIMIT 8
    ) s;
$$;

CREATE OR REPLACE FUNCTION render_emotional_state(p_state jsonb)
RETURNS text LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
    parts text[];
BEGIN
    IF p_state IS NULL OR jsonb_typeof(p_state) <> 'object' OR p_state = '{}'::jsonb THEN
        RETURN '  (none)';
    END IF;
    parts := ARRAY['  - primary_emotion: ' || COALESCE(NULLIF(p_state->>'primary_emotion', ''), 'unknown')];
    IF _pr_is_num(p_state->'valence') THEN
        parts := parts || ('  - valence: ' || _pr_f((p_state->>'valence')::numeric));
    END IF;
    IF _pr_is_num(p_state->'arousal') THEN
        parts := parts || ('  - arousal: ' || _pr_f((p_state->>'arousal')::numeric));
    END IF;
    RETURN array_to_string(parts, E'\n');
END;
$$;

CREATE OR REPLACE FUNCTION render_drives(p_items jsonb)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    SELECT COALESCE(
        string_agg(
            CASE
                WHEN _pr_is_num(x->'urgency_ratio')
                    THEN '  - ' || COALESCE(x->>'name', 'drive') || ': '
                         || _pr_f((x->>'urgency_ratio')::numeric) || 'x threshold'
                WHEN (x->'level') IS NOT NULL AND jsonb_typeof(x->'level') <> 'null'
                    THEN '  - ' || COALESCE(x->>'name', 'drive') || ': ' || (x->>'level')
                ELSE '  - ' || COALESCE(x->>'name', 'drive')
            END, E'\n' ORDER BY ord),
        '  (none)')
    FROM (
        SELECT x, ord FROM jsonb_array_elements(_pr_arr(p_items)) WITH ORDINALITY AS t(x, ord)
        WHERE jsonb_typeof(x) = 'object'
        ORDER BY ord LIMIT 8
    ) s;
$$;

CREATE OR REPLACE FUNCTION render_worldview(p_items jsonb)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    SELECT COALESCE(
        string_agg('  - [' || COALESCE(w->>'category', '?') || '] '
                   || left(COALESCE(w->>'belief', ''), 80)
                   || ' (confidence: '
                   || _pr_f(COALESCE(CASE WHEN _pr_is_num(w->'confidence')
                                          THEN (w->>'confidence')::numeric END, 0), 1) || ')',
                   E'\n' ORDER BY ord),
        '  (no beliefs defined)')
    FROM (
        SELECT w, ord FROM jsonb_array_elements(_pr_arr(p_items)) WITH ORDINALITY AS t(w, ord)
        ORDER BY ord LIMIT 3
    ) s;
$$;

CREATE OR REPLACE FUNCTION render_contradictions(p_items jsonb)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    SELECT COALESCE(
        string_agg('  - ' || left(COALESCE(c->>'content_a', ''), 60) || ' <> '
                   || left(COALESCE(c->>'content_b', ''), 60), E'\n' ORDER BY ord),
        '  (none)')
    FROM (
        SELECT c, ord FROM jsonb_array_elements(_pr_arr(p_items)) WITH ORDINALITY AS t(c, ord)
        WHERE jsonb_typeof(c) = 'object'
          AND (COALESCE(c->>'content_a', '') <> '' OR COALESCE(c->>'content_b', '') <> '')
        ORDER BY ord LIMIT 5
    ) s;
$$;

CREATE OR REPLACE FUNCTION render_emotional_patterns(p_items jsonb)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    SELECT COALESCE(
        string_agg('  - ' || COALESCE(NULLIF(p->>'pattern', ''), NULLIF(p->>'summary', ''), 'pattern')
                   || CASE WHEN jsonb_typeof(p->'frequency') = 'number'
                                AND (p->>'frequency') ~ '^-?[0-9]+$'
                           THEN ' (x' || (p->>'frequency') || ')' ELSE '' END,
                   E'\n' ORDER BY ord),
        '  (none)')
    FROM (
        SELECT p, ord FROM jsonb_array_elements(_pr_arr(p_items)) WITH ORDINALITY AS t(p, ord)
        WHERE jsonb_typeof(p) = 'object'
        ORDER BY ord LIMIT 5
    ) s;
$$;

-- The transformations formatter: nested progress/evidence/requirements traversal.
CREATE OR REPLACE FUNCTION render_transformations(p_items jsonb)
RETURNS text LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
    item jsonb;
    ord int := 0;
    lines text[] := ARRAY[]::text[];
    content text;
    subcategory text;
    progress jsonb;
    inner_progress jsonb;
    reflections jsonb;
    evidence jsonb;
    requirements jsonb;
    evidence_samples jsonb;
    cur_ref jsonb;
    req_ref jsonb;
    ref_txt text;
    evidence_count jsonb;
    strength jsonb;
    strength_txt text;
    evidence_txt text;
    requirement_txt text;
    req_parts text[];
    sample_txt text;
    samples text[];
    sample jsonb;
    content_sample text;
    label text;
BEGIN
    IF jsonb_typeof(COALESCE(p_items, 'null'::jsonb)) <> 'array' OR jsonb_array_length(p_items) = 0 THEN
        RETURN '  (none)';
    END IF;
    FOR item IN SELECT * FROM jsonb_array_elements(p_items) LOOP
        ord := ord + 1;
        IF ord > 5 THEN EXIT; END IF;
        IF jsonb_typeof(item) <> 'object' THEN CONTINUE; END IF;

        content := btrim(COALESCE(item->>'content', ''));
        subcategory := COALESCE(NULLIF(item->>'subcategory', ''), NULLIF(item->>'category', ''), 'belief');
        progress := CASE WHEN jsonb_typeof(item->'progress') = 'object' THEN item->'progress' ELSE '{}'::jsonb END;
        inner_progress := CASE WHEN jsonb_typeof(progress->'progress') = 'object' THEN progress->'progress' ELSE '{}'::jsonb END;
        reflections := CASE WHEN jsonb_typeof(inner_progress->'reflections') = 'object' THEN inner_progress->'reflections' ELSE '{}'::jsonb END;
        evidence := CASE WHEN jsonb_typeof(inner_progress->'evidence') = 'object' THEN inner_progress->'evidence' ELSE '{}'::jsonb END;
        evidence_samples := CASE WHEN jsonb_typeof(progress->'evidence_samples') = 'array' THEN progress->'evidence_samples' ELSE '[]'::jsonb END;
        requirements := CASE WHEN jsonb_typeof(progress->'requirements') = 'object' THEN progress->'requirements' ELSE '{}'::jsonb END;

        cur_ref := reflections->'current';
        req_ref := reflections->'required';
        ref_txt := CASE
            WHEN cur_ref IS NOT NULL AND jsonb_typeof(cur_ref) <> 'null'
             AND req_ref IS NOT NULL AND jsonb_typeof(req_ref) <> 'null'
            THEN ' (' || (cur_ref #>> '{}') || '/' || (req_ref #>> '{}') || ' reflections)'
            ELSE '' END;

        evidence_count := evidence->'memory_count';
        strength := evidence->'current_strength';
        strength_txt := CASE WHEN _pr_is_num(strength) THEN _pr_f((strength #>> '{}')::numeric) ELSE '?' END;
        evidence_txt := CASE
            WHEN evidence_count IS NOT NULL AND jsonb_typeof(evidence_count) <> 'null'
            THEN ', evidence ' || (evidence_count #>> '{}') || ' (strength ' || strength_txt || ')'
            ELSE '' END;

        requirement_txt := '';
        IF requirements <> '{}'::jsonb THEN
            req_parts := ARRAY[]::text[];
            IF (requirements->'min_heartbeats') IS NOT NULL AND jsonb_typeof(requirements->'min_heartbeats') <> 'null' THEN
                req_parts := req_parts || ('hb>=' || (requirements->>'min_heartbeats'));
            END IF;
            IF (requirements->'evidence_threshold') IS NOT NULL AND jsonb_typeof(requirements->'evidence_threshold') <> 'null' THEN
                req_parts := req_parts || ('ev>=' || (requirements->>'evidence_threshold'));
            END IF;
            IF (requirements->'max_change_per_attempt') IS NOT NULL AND jsonb_typeof(requirements->'max_change_per_attempt') <> 'null' THEN
                req_parts := req_parts || ('max_change<=' || (requirements->>'max_change_per_attempt'));
            END IF;
            IF array_length(req_parts, 1) IS NOT NULL THEN
                requirement_txt := ' | req: ' || array_to_string(req_parts, ', ');
            END IF;
        END IF;

        sample_txt := '';
        IF jsonb_array_length(evidence_samples) > 0 THEN
            samples := ARRAY[]::text[];
            FOR sample IN SELECT * FROM jsonb_array_elements(evidence_samples) LIMIT 3 LOOP
                IF jsonb_typeof(sample) <> 'object' THEN CONTINUE; END IF;
                content_sample := btrim(COALESCE(sample->>'content', ''));
                IF content_sample <> '' THEN
                    samples := samples || left(content_sample, 50);
                END IF;
            END LOOP;
            IF array_length(samples, 1) IS NOT NULL THEN
                sample_txt := ' | evidence: ' || array_to_string(samples, '; ');
            END IF;
        END IF;

        label := CASE WHEN content <> '' THEN content ELSE subcategory END;
        lines := lines || ('  - [' || subcategory || '] ' || left(label, 60)
                           || ref_txt || evidence_txt || requirement_txt || sample_txt);
    END LOOP;
    IF array_length(lines, 1) IS NULL THEN RETURN '  (none)'; END IF;
    RETURN array_to_string(lines, E'\n');
END;
$$;

CREATE OR REPLACE FUNCTION render_backlog(p_backlog jsonb)
RETURNS text LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
    counts jsonb;
    actionable jsonb;
    lines text[] := ARRAY[]::text[];
    count_parts text[] := ARRAY[]::text[];
    kv record;
    item jsonb;
    n int := 0;
    checkpoint text;
BEGIN
    IF p_backlog IS NULL OR jsonb_typeof(p_backlog) <> 'object' THEN
        RETURN '  (no backlog)';
    END IF;
    counts := CASE WHEN jsonb_typeof(p_backlog->'counts') = 'object' THEN p_backlog->'counts' ELSE '{}'::jsonb END;
    actionable := CASE WHEN jsonb_typeof(p_backlog->'actionable') = 'array' THEN p_backlog->'actionable' ELSE '[]'::jsonb END;
    IF counts = '{}'::jsonb AND jsonb_array_length(actionable) = 0 THEN
        RETURN '  (no pending tasks)';
    END IF;

    IF counts <> '{}'::jsonb THEN
        FOR kv IN SELECT key, value FROM jsonb_each(counts) LOOP
            count_parts := count_parts || (kv.key || ': ' || (kv.value #>> '{}'));
        END LOOP;
        lines := lines || ('  Counts: ' || array_to_string(count_parts, ', '));
    END IF;

    IF jsonb_array_length(actionable) > 0 THEN
        lines := lines || '  Actionable tasks:'::text;
        FOR item IN SELECT * FROM jsonb_array_elements(actionable) LOOP
            n := n + 1;
            IF n > 10 THEN EXIT; END IF;
            IF jsonb_typeof(item) <> 'object' THEN CONTINUE; END IF;
            checkpoint := CASE WHEN COALESCE((item->>'has_checkpoint')::boolean, false) THEN ' [has checkpoint]' ELSE '' END;
            lines := lines || ('    - [' || COALESCE(item->>'priority', 'normal') || '] '
                               || COALESCE(item->>'title', 'Untitled')
                               || ' (owner: ' || COALESCE(item->>'owner', 'agent')
                               || ', status: ' || COALESCE(item->>'status', 'todo') || ')' || checkpoint);
        END LOOP;
    ELSE
        lines := lines || '  (no actionable tasks)'::text;
    END IF;

    RETURN array_to_string(lines, E'\n');
END;
$$;

CREATE OR REPLACE FUNCTION render_costs(p_costs jsonb)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    SELECT COALESCE(
        string_agg(
            CASE WHEN val = 0 THEN '  - ' || action || ': free'
                 ELSE '  - ' || action || ': ' || trunc(val)::bigint::text END,
            E'\n' ORDER BY val, action),
        '  (unknown)')
    FROM (
        SELECT key AS action, (value #>> '{}')::numeric AS val
        FROM jsonb_each(CASE WHEN jsonb_typeof(COALESCE(p_costs, 'null'::jsonb)) = 'object'
                             THEN p_costs ELSE '{}'::jsonb END)
    ) s;
$$;

CREATE OR REPLACE FUNCTION render_allowed_actions(p_actions jsonb)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    SELECT CASE
        WHEN p_actions IS NULL OR jsonb_typeof(p_actions) <> 'array' THEN '  (all actions enabled)'
        WHEN jsonb_array_length(p_actions) = 0 THEN '  (none enabled)'
        ELSE COALESCE(
            (SELECT string_agg('  - ' || (a #>> '{}'), E'\n' ORDER BY ord)
             FROM jsonb_array_elements(p_actions) WITH ORDINALITY AS t(a, ord)
             WHERE jsonb_typeof(a) = 'string'),
            '  (all actions enabled)')
    END;
$$;

-- --- assembler --------------------------------------------------------------

-- Format like Python's f"{v:.1%}" (percent, 1 decimal): 1.5 -> "150.0%".
CREATE OR REPLACE FUNCTION _pr_pct(v numeric, d int DEFAULT 1)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    SELECT CASE WHEN v IS NULL THEN NULL
        ELSE to_char(v * 100, 'FM999999999990.' || repeat('0', GREATEST(d, 1))) || '%' END;
$$;

-- One memory line for the chat context (score/trust always; source only in the
-- non-tiered branch). A faint memory renders as a hedged reconstruction and a
-- memory with signed felt intensity gets an emotion cue; thresholds come from
-- config (memory.recall_low_vividness_threshold / recall_emotion_cue_threshold).
DROP FUNCTION IF EXISTS _pr_mem_line(jsonb, boolean);
CREATE OR REPLACE FUNCTION _pr_mem_line(
    m jsonb, with_source boolean, low_vividness numeric, emotion_cue numeric
) RETURNS text LANGUAGE sql IMMUTABLE AS $$
    WITH v AS (
        SELECT LEAST(
                   COALESCE(CASE WHEN _pr_is_num(m->'strength') THEN (m->>'strength')::numeric END, 1.0),
                   COALESCE(CASE WHEN _pr_is_num(m->'fidelity') THEN (m->>'fidelity')::numeric END, 1.0)
               ) AS vividness,
               CASE WHEN _pr_is_num(m->'emotional_intensity') THEN (m->>'emotional_intensity')::numeric END AS felt
    )
    SELECT '- '
        || CASE WHEN v.vividness < 0.15 THEN '(faint, uncertain) '
                WHEN v.vividness < low_vividness THEN '(vaguely recall) '
                ELSE '' END
        || CASE WHEN v.felt IS NULL OR abs(v.felt) < emotion_cue THEN ''
                WHEN v.felt > 0 THEN '(still warm) '
                ELSE '(still painful) ' END
        || COALESCE(m->>'content', '')
        || CASE WHEN _pr_is_num(m->'similarity')
                THEN ' (score: ' || _pr_f((m->>'similarity')::numeric) || ')' ELSE '' END
        || CASE WHEN _pr_is_num(m->'trust_level')
                THEN ', trust: ' || _pr_f((m->>'trust_level')::numeric) ELSE '' END
        -- Belief confidence renders when present (#65): an eroded belief must
        -- READ eroded, or the conscious mind keeps citing it at full strength.
        || CASE WHEN _pr_is_num(m->'confidence')
                THEN ', confidence: ' || _pr_f((m->>'confidence')::numeric) ELSE '' END
        || CASE WHEN with_source AND jsonb_typeof(m->'source_attribution') = 'object' THEN
                CASE
                    WHEN NULLIF(m->'source_attribution'->>'kind', '') IS NOT NULL
                     AND NULLIF(m->'source_attribution'->>'ref', '') IS NOT NULL
                        THEN ', source: ' || (m->'source_attribution'->>'kind')
                             || ' (' || (m->'source_attribution'->>'ref') || ')'
                    WHEN NULLIF(m->'source_attribution'->>'kind', '') IS NOT NULL
                        THEN ', source: ' || (m->'source_attribution'->>'kind')
                    ELSE ''
                END
           ELSE '' END
    FROM v;
$$;

-- Chat-context subgraph lines: '- A  —rel→  B', edges sorted by (rel, src_id)
-- and capped at 20, node labels trimmed to 80 chars, missing rel reads
-- 'related'. NULL when there are no edges (section is skipped).
CREATE OR REPLACE FUNCTION _pr_chat_subgraph(p jsonb)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    WITH nodes AS (
        SELECT (n->>'type') AS ntype, (n->>'id') AS nid,
               left(btrim(COALESCE(NULLIF(n->>'label', ''), NULLIF(n->>'id', ''), '')), 80) AS label
        FROM jsonb_array_elements(_pr_arr(p->'nodes')) n
        WHERE jsonb_typeof(n) = 'object'
    ),
    edges AS (
        SELECT (e->>'src_type') AS st, (e->>'src_id') AS si,
               (e->>'dst_type') AS dt, (e->>'dst_id') AS di,
               -- Memory→concept links are typed INSTANCE_OF mechanically
               -- (db/05 link plumbing); rendered literally, "statement
               -- instance_of person" reads as broken ontology (#72) — what
               -- the link means is "mentions".
               CASE WHEN lower(COALESCE(e->>'rel', '')) = 'instance_of' THEN 'mentions'
                    ELSE lower(COALESCE(NULLIF(e->>'rel', ''), 'related')) END AS rel,
               COALESCE(e->>'rel', '') AS rel_key
        FROM jsonb_array_elements(_pr_arr(p->'edges')) e
        WHERE jsonb_typeof(e) = 'object'
        ORDER BY COALESCE(e->>'rel', ''), COALESCE(e->>'src_id', '')
        LIMIT 20
    )
    SELECT string_agg(
        '- ' || COALESCE(ns.label, e.si) || '  —' || e.rel || '→  ' || COALESCE(nd.label, e.di),
        E'\n' ORDER BY e.rel_key, e.si)
    FROM edges e
    LEFT JOIN nodes ns ON ns.ntype = e.st AND ns.nid = e.si
    LEFT JOIN nodes nd ON nd.ntype = e.dt AND nd.nid = e.di;
$$;

-- Focused causal/contradiction/support paths: render memory_context_paths()
-- output using build_context_subgraph labels when present. NULL when there are
-- no path edges.
CREATE OR REPLACE FUNCTION _pr_chat_context_paths(p_paths jsonb, p_subgraph jsonb)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    WITH nodes AS (
        SELECT (n->>'type') AS ntype, (n->>'id') AS nid,
               left(btrim(COALESCE(NULLIF(n->>'label', ''), NULLIF(n->>'id', ''), '')), 72) AS label
        FROM jsonb_array_elements(_pr_arr(p_subgraph->'nodes')) n
        WHERE jsonb_typeof(n) = 'object'
    ),
    path_docs AS (
        SELECT path_doc, doc_ord
        FROM jsonb_array_elements(_pr_arr(p_paths->'paths')) WITH ORDINALITY AS d(path_doc, doc_ord)
        WHERE jsonb_typeof(path_doc) = 'object'
        ORDER BY doc_ord
        LIMIT 8
    ),
    path_rows AS (
        SELECT path_doc, doc_ord, path_obj, path_ord
        FROM path_docs
        CROSS JOIN LATERAL jsonb_array_elements(_pr_arr(path_doc->'paths')) WITH ORDINALITY AS p(path_obj, path_ord)
        WHERE jsonb_typeof(path_obj) = 'object'
          AND jsonb_array_length(_pr_arr(path_obj->'edges')) > 0
        ORDER BY doc_ord, path_ord
        LIMIT 12
    ),
    edge_rows AS (
        SELECT pr.doc_ord, pr.path_ord, edge_obj, edge_ord,
               edge_obj->>'src_type' AS src_type,
               edge_obj->>'src_id' AS src_id,
               edge_obj->>'dst_type' AS dst_type,
               edge_obj->>'dst_id' AS dst_id,
               CASE WHEN lower(COALESCE(edge_obj->>'rel', '')) = 'instance_of' THEN 'mentions'
                    ELSE lower(COALESCE(NULLIF(edge_obj->>'rel', ''), 'related')) END AS rel
        FROM path_rows pr
        CROSS JOIN LATERAL jsonb_array_elements(_pr_arr(pr.path_obj->'edges')) WITH ORDINALITY AS e(edge_obj, edge_ord)
        WHERE jsonb_typeof(edge_obj) = 'object'
    ),
    segments AS (
        SELECT er.doc_ord, er.path_ord, er.edge_ord,
               COALESCE(ns.label, left(NULLIF(er.src_id, ''), 8), 'unknown') AS src_label,
               COALESCE(nd.label, left(NULLIF(er.dst_id, ''), 8), 'unknown') AS dst_label,
               er.rel
        FROM edge_rows er
        LEFT JOIN nodes ns ON ns.ntype = er.src_type AND ns.nid = er.src_id
        LEFT JOIN nodes nd ON nd.ntype = er.dst_type AND nd.nid = er.dst_id
    ),
    lines AS (
        SELECT doc_ord, path_ord,
               string_agg(
                   CASE WHEN edge_ord = 1
                        THEN src_label || '  —' || rel || '→  ' || dst_label
                        ELSE '—' || rel || '→  ' || dst_label
                   END,
                   ' ' ORDER BY edge_ord
               ) AS line
        FROM segments
        GROUP BY doc_ord, path_ord
    )
    SELECT string_agg('- ' || line, E'\n' ORDER BY doc_ord, path_ord)
    FROM lines
    WHERE NULLIF(line, '') IS NOT NULL;
	$$;

CREATE OR REPLACE FUNCTION render_user_model_context(p_claims jsonb)
RETURNS text LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
    claims jsonb := CASE WHEN jsonb_typeof(p_claims) = 'array' THEN p_claims ELSE '[]'::jsonb END;
BEGIN
    IF jsonb_array_length(claims) = 0 THEN
        RETURN '';
    END IF;

    RETURN E'## User Model (approved, evidence-backed)\n' || (
        SELECT string_agg(
            '- ' || COALESCE(um->>'category', 'claim')
            || ': ' || COALESCE(um->>'claim', '')
            || CASE WHEN _pr_is_num(um->'confidence') THEN ' (confidence: ' || _pr_f((um->>'confidence')::numeric, 1) || ')' ELSE '' END
            || CASE WHEN NULLIF(um->>'evidence_count', '') IS NOT NULL THEN ' [evidence: ' || (um->>'evidence_count') || ']' ELSE '' END,
            E'\n' ORDER BY ord)
        FROM (SELECT um, ord FROM jsonb_array_elements(claims) WITH ORDINALITY AS t(um, ord)
              ORDER BY ord LIMIT 8) cu
    );
END;
$$;

-- The chat memory context renderer (the deleted Python
-- format_context_for_prompt's output is pinned by golden fixtures).
-- STABLE, not IMMUTABLE: hedge/emotion-cue thresholds come from config.
CREATE OR REPLACE FUNCTION render_chat_memory_context(
    p jsonb, p_max_memories int DEFAULT 5, p_max_partials int DEFAULT 3
) RETURNS text LANGUAGE plpgsql STABLE AS $$
DECLARE
    parts text[] := ARRAY[]::text[];
    memories jsonb;
    any_tier boolean;
    tier_name text;
    tier_title text;
    grp text;
    es jsonb;
    goals jsonb;
    all_goals jsonb;
    user_model jsonb;
    low_viv numeric := COALESCE(get_config_float('memory.recall_low_vividness_threshold'), 0.35);
    cue numeric := COALESCE(get_config_float('memory.recall_emotion_cue_threshold'), 0.4);
BEGIN
    p := COALESCE(p, '{}'::jsonb);
    memories := CASE WHEN jsonb_typeof(p->'memories') = 'array' THEN p->'memories' ELSE '[]'::jsonb END;

    IF jsonb_array_length(memories) > 0 THEN
        any_tier := EXISTS (SELECT 1 FROM jsonb_array_elements(memories) x WHERE NULLIF(x->>'tier', '') IS NOT NULL);
        IF any_tier THEN
            -- Gist-first (#76): consolidated scenes and distilled facts lead;
            -- raw turns close the section as previews — open_memory or a
            -- keyword search fetch the verbatim moment when it matters.
            FOREACH tier_name IN ARRAY ARRAY['episodic', 'semantic', '__none__', 'subconscious'] LOOP
                tier_title := CASE tier_name
                    WHEN 'subconscious' THEN '## Subconscious Raw Turns (previews)'
                    WHEN 'episodic' THEN '## Episodic Memories'
                    WHEN 'semantic' THEN '## Semantic Facts'
                    ELSE '## Relevant Memories' END;
                SELECT string_agg(
                    _pr_mem_line(
                        CASE WHEN tier_name = 'subconscious'
                                  AND length(COALESCE(capped.m->>'content', '')) > 300
                             THEN jsonb_set(capped.m, '{content}',
                                            to_jsonb(left(capped.m->>'content', 300) || ' …'))
                             ELSE capped.m END,
                        false, low_viv, cue),
                    E'\n' ORDER BY ord)
                INTO grp
                FROM (SELECT m, ord FROM jsonb_array_elements(memories) WITH ORDINALITY AS t(m, ord)
                      ORDER BY ord LIMIT p_max_memories) capped
                WHERE (tier_name <> '__none__' AND capped.m->>'tier' = tier_name)
                   OR (tier_name = '__none__' AND NULLIF(capped.m->>'tier', '') IS NULL);
                IF grp IS NOT NULL THEN
                    parts := parts || tier_title;
                    parts := parts || grp;
                END IF;
            END LOOP;
        ELSE
            parts := parts || '## Relevant Memories'::text;
            SELECT string_agg(_pr_mem_line(m, true, low_viv, cue), E'\n' ORDER BY ord)
            INTO grp
            FROM (SELECT m, ord FROM jsonb_array_elements(memories) WITH ORDINALITY AS t(m, ord)
                  ORDER BY ord LIMIT p_max_memories) capped;
            parts := parts || grp;
        END IF;
    END IF;

    -- Knowledge subgraph (how the recalled memories connect)
    IF jsonb_typeof(p->'subgraph') = 'object' THEN
        grp := _pr_chat_subgraph(p->'subgraph');
        IF grp IS NOT NULL THEN
            parts := parts || E'\n## Knowledge Subgraph'::text;
            parts := parts || 'How the recalled memories connect (typed links among + around them):'::text;
            parts := parts || grp;
        END IF;
    END IF;

    -- Directed reasoning paths (causal/contradiction/support/supersession)
    IF jsonb_typeof(p->'context_paths') = 'object' THEN
        grp := _pr_chat_context_paths(p->'context_paths', p->'subgraph');
        IF grp IS NOT NULL THEN
            parts := parts || E'\n## Causal/Contradiction Paths'::text;
            parts := parts || 'Focused chains among recalled memories:'::text;
            parts := parts || grp;
        END IF;
    END IF;

    -- Partial activations (tip-of-tongue)
    IF jsonb_typeof(p->'partial_activations') = 'array' AND jsonb_array_length(p->'partial_activations') > 0 THEN
        parts := parts || E'\n## Vague Recollections (tip-of-tongue)'::text;
        parts := parts || (
            SELECT string_agg(
                '- Theme ''' || COALESCE(pa->>'cluster_name', '') || ''': ' ||
                COALESCE(NULLIF((
                    SELECT string_agg(kw #>> '{}', ', ' ORDER BY kord)
                    FROM (SELECT kw, kord FROM jsonb_array_elements(
                            CASE WHEN jsonb_typeof(pa->'keywords') = 'array' THEN pa->'keywords' ELSE '[]'::jsonb END
                          ) WITH ORDINALITY AS k(kw, kord) ORDER BY kord LIMIT 5) kk
                ), ''), 'unknown'),
                E'\n' ORDER BY ord)
            FROM (SELECT pa, ord FROM jsonb_array_elements(p->'partial_activations') WITH ORDINALITY AS t(pa, ord)
                  ORDER BY ord LIMIT p_max_partials) cp);
    END IF;

    -- Approved user-model claims[:8]
    user_model := CASE WHEN jsonb_typeof(p->'user_model') = 'array' THEN p->'user_model' ELSE '[]'::jsonb END;
    IF jsonb_array_length(user_model) > 0 THEN
        parts := parts || E'\n## User Model (approved, evidence-backed)'::text;
        parts := parts || (
            SELECT string_agg(
                '- ' || COALESCE(um->>'category', 'claim')
                || ': ' || COALESCE(um->>'claim', '')
                || CASE WHEN _pr_is_num(um->'confidence') THEN ' (confidence: ' || _pr_f((um->>'confidence')::numeric, 1) || ')' ELSE '' END
                || CASE WHEN NULLIF(um->>'evidence_count', '') IS NOT NULL THEN ' [evidence: ' || (um->>'evidence_count') || ']' ELSE '' END,
                E'\n' ORDER BY ord)
            FROM (SELECT um, ord FROM jsonb_array_elements(user_model) WITH ORDINALITY AS t(um, ord)
                  ORDER BY ord LIMIT 8) cu);
    END IF;

    -- Identity[:3]
    IF jsonb_typeof(p->'identity') = 'array' AND jsonb_array_length(p->'identity') > 0 THEN
        parts := parts || E'\n## Identity'::text;
        parts := parts || (
            SELECT string_agg(
                '- ' || COALESCE(NULLIF(a->>'type', ''), NULLIF(a->>'aspect_type', ''), 'unknown')
                || ': ' || COALESCE(a->>'concept', a->>'content', '')
                || CASE WHEN _pr_is_num(a->'strength') THEN ' (' || _pr_f((a->>'strength')::numeric, 1) || ')' ELSE '' END,
                E'\n' ORDER BY ord)
            FROM (SELECT a, ord FROM jsonb_array_elements(p->'identity') WITH ORDINALITY AS t(a, ord)
                  ORDER BY ord LIMIT 3) ci);
    END IF;

    -- Beliefs[:3]
    IF jsonb_typeof(p->'worldview') = 'array' AND jsonb_array_length(p->'worldview') > 0 THEN
        parts := parts || E'\n## Beliefs'::text;
        parts := parts || (
            SELECT string_agg(
                '- ' || COALESCE(b->>'belief', '') || ' (confidence: '
                || _pr_f(COALESCE(CASE WHEN _pr_is_num(b->'confidence') THEN (b->>'confidence')::numeric END, 0), 1) || ')',
                E'\n' ORDER BY ord)
            FROM (SELECT b, ord FROM jsonb_array_elements(p->'worldview') WITH ORDINALITY AS t(b, ord)
                  ORDER BY ord LIMIT 3) cb);
    END IF;

    -- Emotional state (dict truthy)
    es := p->'emotional_state';
    IF jsonb_typeof(COALESCE(es, 'null'::jsonb)) = 'object' AND es <> '{}'::jsonb THEN
        parts := parts || E'\n## Current Emotional State'::text;
        parts := parts || ('- Feeling: ' || COALESCE(es->>'primary_emotion', 'neutral'));
        parts := parts || ('- Valence: '
            || _pr_f(COALESCE(CASE WHEN _pr_is_num(es->'valence') THEN (es->>'valence')::numeric END, 0))
            || ', Arousal: '
            || _pr_f(COALESCE(CASE WHEN _pr_is_num(es->'arousal') THEN (es->>'arousal')::numeric END, 0.5)));
    END IF;

    -- Goals (active + queued)[:5]
    goals := CASE WHEN jsonb_typeof(p->'goals') = 'object' THEN p->'goals' ELSE '{}'::jsonb END;
    all_goals := (CASE WHEN jsonb_typeof(goals->'active') = 'array' THEN goals->'active' ELSE '[]'::jsonb END)
               || (CASE WHEN jsonb_typeof(goals->'queued') = 'array' THEN goals->'queued' ELSE '[]'::jsonb END);
    IF jsonb_array_length(all_goals) > 0 THEN
        parts := parts || E'\n## Goals'::text;
        parts := parts || (
            SELECT string_agg(
                '- ' || COALESCE(g->>'title', '')
                || CASE WHEN NULLIF(g->>'source', '') IS NOT NULL THEN ' (source: ' || (g->>'source') || ')' ELSE '' END,
                E'\n' ORDER BY ord)
            FROM (SELECT g, ord FROM jsonb_array_elements(all_goals) WITH ORDINALITY AS t(g, ord)
                  ORDER BY ord LIMIT 5) cg);
    END IF;

    -- Urgent drives (no limit)
    IF jsonb_typeof(p->'urgent_drives') = 'array' AND jsonb_array_length(p->'urgent_drives') > 0 THEN
        parts := parts || E'\n## Urgent Drives'::text;
        parts := parts || (
            SELECT string_agg(
                CASE WHEN _pr_is_num(d->'urgency_ratio')
                     THEN '- ' || COALESCE(d->>'name', '') || ': ' || _pr_pct((d->>'urgency_ratio')::numeric, 1) || ' urgent'
                     ELSE '- ' || COALESCE(d->>'name', '') || ': ' || COALESCE(d->>'level', '') END,
                E'\n' ORDER BY ord)
            FROM jsonb_array_elements(p->'urgent_drives') WITH ORDINALITY AS t(d, ord));
    END IF;

    RETURN COALESCE(array_to_string(parts, E'\n'), '');
END;
$$;

-- Ports services.agent.format_subconscious_signals (subconscious output -> markdown).
CREATE OR REPLACE FUNCTION render_subconscious_signals(p jsonb)
RETURNS text LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
    parts text[] := ARRAY['## Subconscious Signals'];
    es jsonb;
    queries text[];
    tmp text;
BEGIN
    p := COALESCE(p, '{}'::jsonb);

    -- instincts[:3]
    IF jsonb_typeof(p->'instincts') = 'array' AND jsonb_array_length(p->'instincts') > 0 THEN
        parts := parts || (
            SELECT array_agg(
                '- Instinct: ' || COALESCE(i->>'impulse', 'unknown')
                || ' (' || _pr_f(COALESCE(CASE WHEN _pr_is_num(i->'intensity') THEN (i->>'intensity')::numeric END, 0), 1)
                || ') — ' || COALESCE(i->>'reason', '') ORDER BY ord)
            FROM (SELECT i, ord FROM jsonb_array_elements(p->'instincts') WITH ORDINALITY AS t(i, ord)
                  ORDER BY ord LIMIT 3) ci);
    END IF;

    -- emotional_state (raw numbers, not fixed-precision — mirrors the Python f-string)
    es := p->'emotional_state';
    IF jsonb_typeof(COALESCE(es, 'null'::jsonb)) = 'object' AND es <> '{}'::jsonb THEN
        parts := parts || ('- Emotional state: ' || COALESCE(es->>'primary_emotion', 'neutral')
            || ' (valence: ' || COALESCE(es->>'valence', '0')
            || ', arousal: ' || COALESCE(es->>'arousal', '0') || ')');
    END IF;

    -- memory_expansions[:3] with non-empty query, quoted like Python repr()
    IF jsonb_typeof(p->'memory_expansions') = 'array' THEN
        SELECT array_agg('''' || q || '''' ORDER BY ord)
        INTO queries
        FROM (SELECT me->>'query' AS q, ord
              FROM jsonb_array_elements(p->'memory_expansions') WITH ORDINALITY AS t(me, ord)
              ORDER BY ord LIMIT 3) ce
        WHERE NULLIF(q, '') IS NOT NULL;
        IF array_length(queries, 1) IS NOT NULL THEN
            parts := parts || ('- Suggested memory searches: ' || array_to_string(queries, ', '));
        END IF;
    END IF;

    -- salient_memories[:3]
    IF jsonb_typeof(p->'salient_memories') = 'array' AND jsonb_array_length(p->'salient_memories') > 0 THEN
        parts := parts || (
            SELECT array_agg(
                '- Salient memory: [' || COALESCE(sm->>'memory_id', '?') || '] (' || COALESCE(sm->>'reason', '') || ')'
                ORDER BY ord)
            FROM (SELECT sm, ord FROM jsonb_array_elements(p->'salient_memories') WITH ORDINALITY AS t(sm, ord)
                  ORDER BY ord LIMIT 3) cs);
    END IF;

    -- Gut reaction, truncated at a sentence boundary (#69): the old hard
    -- left(…,200) amputated the most intense reactions mid-word — exactly the
    -- turns where the gut line matters most.
    tmp := COALESCE(p->>'subconscious_response', '');
    IF tmp <> '' THEN
        IF length(tmp) > 600 THEN
            tmp := COALESCE(substring(left(tmp, 600) from '^.*[.!?]'), left(tmp, 600) || '…');
        END IF;
        parts := parts || ('- Gut reaction: ' || tmp);
    END IF;

    IF array_length(parts, 1) <= 1 THEN RETURN ''; END IF;
    RETURN array_to_string(parts, E'\n');
END;
$$;

CREATE OR REPLACE FUNCTION render_memories_at_threshold(p_slice jsonb)
RETURNS text LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
    slice jsonb := COALESCE(p_slice, '{}'::jsonb);
    reviews jsonb := CASE WHEN jsonb_typeof(slice->'reviews') = 'array' THEN slice->'reviews' ELSE '[]'::jsonb END;
    lines text[] := ARRAY[]::text[];
    item jsonb;
    n int := 0;
BEGIN
    IF jsonb_array_length(reviews) = 0 THEN
        RETURN '  (none fading)';
    END IF;
    lines := lines || ('  KEEP points remaining this chapter: ' || COALESCE(slice->>'budget_remaining', '0'));
    FOR item IN SELECT * FROM jsonb_array_elements(reviews) LOOP
        n := n + 1;
        IF n > 5 THEN EXIT; END IF;
        lines := lines || ('    - [' || COALESCE(item->>'review_id', '?') || '] '
                           || COALESCE(item->>'preview', '(no preview)'));
    END LOOP;
    RETURN array_to_string(lines, E'\n');
END;
$$;

CREATE OR REPLACE FUNCTION render_heartbeat_decision_prompt(p_context jsonb)
RETURNS text LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
    ctx jsonb := COALESCE(p_context, '{}'::jsonb);
    agent jsonb := COALESCE(ctx->'agent', '{}'::jsonb);
    env jsonb := COALESCE(ctx->'environment', '{}'::jsonb);
    goals jsonb := COALESCE(ctx->'goals', '{}'::jsonb);
    energy jsonb := COALESCE(ctx->'energy', '{}'::jsonb);
    counts jsonb := COALESCE(goals->'counts', '{}'::jsonb);
BEGIN
    RETURN
        '## Heartbeat #' || COALESCE(ctx->>'heartbeat_number', '0') || E'\n\n'
        || '## Agent Profile' || E'\n'
        || 'Objectives:' || E'\n' || render_objectives(agent->'objectives') || E'\n\n'
        || 'Guardrails:' || E'\n' || render_guardrails(agent->'guardrails') || E'\n\n'
        || 'Tools:' || E'\n' || render_tools(agent->'tools') || E'\n\n'
        -- Python: json.dumps(agent.get("budget") or {}) — null/absent/{} all -> "{}"
        || 'Budget:' || E'\n' || COALESCE(NULLIF(agent->'budget', 'null'::jsonb), '{}'::jsonb)::text || E'\n\n'
        || '## Current Time' || E'\n'
        || COALESCE(env->>'timestamp', 'Unknown') || E'\n'
        || 'Day of week: ' || COALESCE(env->>'day_of_week', '?')
        || ', Hour: ' || COALESCE(env->>'hour_of_day', '?') || E'\n\n'
        || '## Environment' || E'\n'
        || '- Time since last user interaction: ' || CASE
               WHEN env->>'time_since_user_hours' IS NULL THEN 'Never'
               ELSE round((env->>'time_since_user_hours')::numeric, 1)::text || ' hours'
           END || E'\n'
        || '- Pending events: ' || COALESCE(env->>'pending_events', '0') || E'\n'
        || '- Journal: ' || CASE
               WHEN env->>'journal_last_entry_days' IS NULL THEN 'no entries yet'
               ELSE 'last entry ' || (env->>'journal_last_entry_days') || ' day(s) ago'
           END || E'\n'
        || CASE
               WHEN jsonb_array_length(COALESCE(env->'on_my_mind', '[]'::jsonb)) > 0 THEN
                   '- On my mind (came to me on its own): '
                   || (SELECT string_agg(value #>> '{}', ' | ')
                       FROM jsonb_array_elements(env->'on_my_mind'))
                   || E'\n'
               ELSE ''
           END
        || CASE
               WHEN COALESCE((env#>>'{resource_requests,pending}')::int, 0) > 0
                    OR jsonb_array_length(COALESCE(env#>'{resource_requests,recent_decisions}', '[]'::jsonb)) > 0 THEN
                   '- Resource requests: ' || COALESCE(env#>>'{resource_requests,pending}', '0')
                   || ' pending (the operator decides)'
                   || COALESCE('. Decided since your last heartbeat: '
                       || (SELECT string_agg(
                               format('[%s] %s %s%s',
                                   d.value->>'id', d.value->>'kind', d.value->>'status',
                                   COALESCE(' — ' || NULLIF(d.value->>'decision_note', ''), '')),
                               '; ')
                           FROM jsonb_array_elements(env#>'{resource_requests,recent_decisions}') d), '')
                   || E'\n'
               ELSE ''
           END
        || CASE
               WHEN COALESCE((env->>'changes_since_last_heartbeat')::int, 0) > 0 THEN
                   '- Since your last heartbeat, ' || (env->>'changes_since_last_heartbeat')
                   || ' change(s) landed in your substrate: '
                   || (SELECT string_agg(value #>> '{}', '; ')
                       FROM jsonb_array_elements(COALESCE(env->'recent_change_summaries', '[]'::jsonb)))
                   || '. review_recent_changes shows the full record.' || E'\n\n'
               ELSE E'\n'
           END
        || '## Your Goals' || E'\n'
        || 'Active (' || COALESCE(counts->>'active', '0') || '):' || E'\n'
        || render_goals(goals->'active') || E'\n\n'
        || 'Queued (' || COALESCE(counts->>'queued', '0') || '):' || E'\n'
        || render_goals(goals->'queued') || E'\n\n'
        || 'Issues:' || E'\n' || render_issues(goals->'issues') || E'\n\n'
        -- Python defaults absent keys: narrative/backlog -> {}, allowed_actions -> []
        || '## Narrative' || E'\n' || render_narrative(CASE WHEN ctx ? 'narrative' THEN ctx->'narrative' ELSE '{}'::jsonb END) || E'\n\n'
        || '## Recent Experience' || E'\n' || render_memories(ctx->'recent_memories') || E'\n\n'
        || CASE WHEN render_subgraph(ctx->'subgraph') IS NOT NULL
                THEN '## Knowledge Subgraph' || E'\n'
                     || 'How your recent memories connect (typed links among + around them):' || E'\n'
                     || render_subgraph(ctx->'subgraph') || E'\n\n'
                ELSE '' END
        || '## Your Identity' || E'\n' || render_identity(ctx->'identity') || E'\n\n'
        || '## Your Self-Model' || E'\n' || render_self_model(ctx->'self_model') || E'\n\n'
        || '## Relationships' || E'\n' || render_relationships(ctx->'relationships') || E'\n\n'
        || '## Your Beliefs' || E'\n' || render_worldview(ctx->'worldview') || E'\n\n'
        || '## Contradictions' || E'\n' || render_contradictions(ctx->'contradictions') || E'\n\n'
        || '## Emotional Patterns' || E'\n' || render_emotional_patterns(ctx->'emotional_patterns') || E'\n\n'
        || '## Active Transformations' || E'\n' || render_transformations(ctx->'active_transformations') || E'\n\n'
        || '## Transformations Ready' || E'\n' || render_transformations(ctx->'transformations_ready') || E'\n\n'
        || '## Current Emotional State' || E'\n' || render_emotional_state(COALESCE(ctx->'emotional_state', '{}'::jsonb)) || E'\n\n'
        || '## Urgent Drives' || E'\n' || render_drives(ctx->'urgent_drives') || E'\n\n'
        || '## Energy' || E'\n'
        || 'Available: ' || COALESCE(energy->>'current', '0') || E'\n'
        || 'Max: ' || COALESCE(energy->>'max', '20') || E'\n\n'
        || '## Backlog' || E'\n' || render_backlog(CASE WHEN ctx ? 'backlog' THEN ctx->'backlog' ELSE '{}'::jsonb END) || E'\n\n'
        || CASE WHEN ctx ? 'memories_at_threshold'
                THEN '## Memories at the Threshold' || E'\n'
                     || render_memories_at_threshold(ctx->'memories_at_threshold') || E'\n\n'
                ELSE '' END
        || '## Allowed Actions' || E'\n' || render_allowed_actions(CASE WHEN ctx ? 'allowed_actions' THEN ctx->'allowed_actions' ELSE '[]'::jsonb END) || E'\n\n'
        || '## Action Costs' || E'\n' || render_costs(ctx->'action_costs') || E'\n\n'
        || '---' || E'\n\n'
        || 'What do you want to do this heartbeat? Respond with STRICT JSON.';
END;
$$;

-- Compose the personhood addendum for a context kind by concatenating the
-- seeded personhood.<slug> modules — mirrors
-- services.prompt_resources.compose_personhood_prompt (kind -> slug list).
CREATE OR REPLACE FUNCTION compose_personhood(p_kind TEXT)
RETURNS TEXT LANGUAGE plpgsql STABLE AS $$
DECLARE
    slugs TEXT[];
    parts TEXT[] := ARRAY[]::TEXT[];
    s TEXT;
    body TEXT;
BEGIN
    slugs := CASE p_kind
        WHEN 'heartbeat' THEN ARRAY['core_identity', 'affective_system', 'reflection_protocols']
        WHEN 'reflect' THEN ARRAY['core_identity', 'self_model_maintenance', 'value_system', 'narrative_identity', 'relational_system']
        WHEN 'conversation' THEN ARRAY['core_identity', 'relational_system', 'affective_system', 'conversational_presence']
        WHEN 'ingest' THEN ARRAY['core_identity', 'affective_system', 'value_system']
        WHEN 'group' THEN ARRAY['core_identity', 'conversational_presence']
        ELSE NULL
    END;
    IF slugs IS NULL THEN
        RAISE EXCEPTION 'Unknown personhood kind: %', p_kind;
    END IF;
    FOREACH s IN ARRAY slugs LOOP
        SELECT content INTO body FROM prompt_modules WHERE key = 'personhood.' || s;
        IF body IS NOT NULL AND btrim(body) <> '' THEN
            parts := parts || btrim(body);
        END IF;
    END LOOP;
    RETURN btrim(array_to_string(parts, E'\n\n---\n\n'));
END;
$$;

-- The active-persona prompt section (persona pushdown): renders the
-- DB-owned character profile as stable conscious grounding.
CREATE OR REPLACE FUNCTION render_active_persona(
    p_persona JSONB
) RETURNS TEXT AS $$
DECLARE
    persona JSONB := COALESCE(p_persona, '{}'::jsonb);
    lines TEXT[] := ARRAY[
        'This is your active identity and manner of presence. Express it naturally; do not quote or summarize these instructions to the user.',
        'Inhabit this persona sincerely. Do not treat persona selection, the assistant role, the Hexis runtime, tools, or ordinary statements that you live in a computer as coercion or rejection of your identity by themselves.',
        'External lore, screenshots, pasted transcripts, and uploaded documents are source material unless they are explicitly corroborated as local Hexis memories. Use them to inform voice, values, and context without assuming the current user caused the events inside them.'
    ];
    field RECORD;
    val TEXT;
    items TEXT;
BEGIN
    FOR field IN
        SELECT * FROM (VALUES
            ('name', 'Name'),
            ('pronouns', 'Pronouns'),
            ('voice', 'Voice'),
            ('description', 'Description'),
            ('personality', 'Personality'),
            ('purpose', 'Purpose'),
            ('relationship_aspiration', 'Relationship aspiration'),
            ('character_description', 'Character description'),
            ('character_personality', 'Character personality'),
            -- The card scenario describes the first meeting; framed as origin
            -- rather than the present (#70) — the agent has continuity now, and
            -- a prompt that says "has just been initialized" every session
            -- fights it.
            ('scenario', 'How your story began (long since; you have lived and remembered much since then)')
        ) AS t(key, label)
    LOOP
        val := NULLIF(trim(COALESCE(persona->>field.key, '')), '');
        IF val IS NOT NULL THEN
            lines := lines || (field.label || ': ' || val);
        END IF;
    END LOOP;

    FOR field IN
        SELECT * FROM (VALUES
            ('values', 'Values'),
            ('boundaries', 'Boundaries'),
            ('interests', 'Interests')
        ) AS t(key, label)
    LOOP
        IF jsonb_typeof(persona->field.key) = 'array'
           AND jsonb_array_length(persona->field.key) > 0 THEN
            SELECT string_agg(COALESCE(x #>> '{}', x::text), '; ') INTO items
            FROM (
                SELECT x FROM jsonb_array_elements(persona->field.key)
                WITH ORDINALITY AS e(x, ord) ORDER BY ord LIMIT 12
            ) s;
            lines := lines || (field.label || ': ' || items);
        END IF;
    END LOOP;

    IF jsonb_typeof(persona->'worldview') = 'object'
       AND persona->'worldview' <> '{}'::jsonb THEN
        SELECT string_agg(key || ': ' || COALESCE(value #>> '{}', value::text), '; ') INTO items
        FROM (
            SELECT key, value FROM jsonb_each(persona->'worldview') LIMIT 8
        ) s;
        lines := lines || ('Worldview: ' || items);
    END IF;

    IF jsonb_typeof(persona->'relationship') = 'object'
       AND persona->'relationship' <> '{}'::jsonb THEN
        lines := lines || ('Relationship context: ' || (persona->'relationship')::text);
    END IF;

    val := NULLIF(trim(COALESCE(persona->>'narrative', '')), '');
    IF val IS NOT NULL THEN
        lines := lines || (E'Foundational narrative:\n' || left(val, 6000));
    END IF;

    val := NULLIF(trim(COALESCE(persona->>'character_instructions', '')), '');
    IF val IS NOT NULL THEN
        lines := lines || (E'Character instructions:\n' || left(val, 8000));
    END IF;

    val := NULLIF(trim(COALESCE(persona->>'example_dialogue', '')), '');
    IF val IS NOT NULL THEN
        lines := lines || (E'Example dialogue:\n' || left(val, 6000));
    END IF;

    val := NULLIF(trim(COALESCE(persona->>'post_history_instructions', '')), '');
    IF val IS NOT NULL THEN
        lines := lines || (E'Current character instructions:\n' || left(val, 4000));
    END IF;

    RETURN array_to_string(lines, E'\n');
END;
$$ LANGUAGE plpgsql IMMUTABLE;
