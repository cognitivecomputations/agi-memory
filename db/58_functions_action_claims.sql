-- Action-claim guardrail (#38): detect assistant prose that claims an action
-- (stored / created / scheduled / sent / read file X) with no matching
-- successful tool call in the same turn or a durable receipt from a prior
-- turn in the hydrated chat session. Patterns are DATA, tunable live.
-- Advisory by design: detection never blocks a reply; the loop appends a
-- visible correction. Kill switch: config 'guardrails.action_claims.enabled'.
SET search_path = public, ag_catalog, "$user";

CREATE TABLE IF NOT EXISTS action_claim_patterns (
    id SERIAL PRIMARY KEY,
    claim_kind TEXT NOT NULL,
    pattern TEXT NOT NULL,               -- POSIX regex, evaluated per sentence, case-insensitive
    satisfied_by_tools TEXT[] NOT NULL,  -- LIKE patterns over tool names (backslash escapes _)
    require_arg_key TEXT,                -- when set, a file token in the sentence must match this call argument
    -- Patterns that inherently describe negative results ("I searched and
    -- found nothing") opt OUT of the negation suppression (#50).
    match_negated BOOLEAN NOT NULL DEFAULT FALSE,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO action_claim_patterns (claim_kind, pattern, satisfied_by_tools, require_arg_key, notes)
SELECT v.claim_kind, v.pattern, v.satisfied_by_tools, v.require_arg_key, v.notes
FROM (VALUES
    ('memory_write',
     '\mI(''ve| have)? ?((just|also|already|now|then) )?(stored|saved|recorded) (that|this|it|these|those|the|your|our|a )',
     ARRAY['remember','add_evidence','fast_ingest','slow_ingest','hybrid_ingest','url_ingest','git_ingest','fathom_ingest','import_memories'],
     NULL,
     'claims of a completed memory write'),
    ('memory_write',
     '\m(committed|added) (it|that|this|these|those) to (my |the )?memor',
     ARRAY['remember','add_evidence','fast_ingest','slow_ingest','hybrid_ingest','url_ingest','git_ingest'],
     NULL,
     'committed/added to memory phrasing'),
    ('memory_write',
     '\mI(''ve| have)? ?((just|also|already|now|then) )?created a (new )?memor',
     ARRAY['remember','add_evidence','fast_ingest','slow_ingest','hybrid_ingest'],
     NULL,
     'created-a-memory phrasing'),
    ('goal_backlog',
     '\mI(''ve| have)? ?((just|also|already|now|then) )?(created|added|filed|queued|logged) (a|an|the|this|that|another) (new |high.priority )?(goal|backlog|task|to.?do|item)',
     ARRAY['create_goal','manage_goals','manage_backlog','todoist_create_task','asana_create_task'],
     NULL,
     'claims of goal/backlog/task creation'),
    ('scheduled',
     '\mI(''ve| have)? ?((just|also|already|now|then) )?(scheduled|set up a (reminder|cron)|added a scheduled)',
     ARRAY['schedule_task','update_scheduled_task','manage_schedule','calendar_create','calendar_update'],
     NULL,
     'claims of scheduling'),
    ('external_send',
     '\mI(''ve| have)? ?((just|also|already|now|then) )?(sent|emailed|messaged|posted|filed|submitted|published|replied to)',
     ARRAY['email_send','email_send_sendgrid','discord_send','slack_send','telegram_send','queue_user_message','mcp\_%'],
     NULL,
     'claims of outward-facing sends; mcp\_% covers MCP-backed integrations'),
    ('source_inspection',
     '\mI (''ve |have )?((just|also|already|now|then) )?(read|inspected|examined|traced|reviewed|verified) [^.!?]*(\.(py|sql|md|ts|tsx|js|jsx|json|ya?ml|toml|sh|go|rs)\M|lines? [0-9])',
     ARRAY['inspect_source','read_file','grep','glob','list_directory'],
     'path',
     'claims of having read specific source files/lines'),
    -- Correction claims (#67): "I''ve corrected that in my memory" is only
    -- supported by a revision-class action — storing an unrelated new note
    -- with remember is an addition, not a correction.
    ('memory_correction',
     '\mI(''ve| have)? ?((just|also|already|now|then) )?(corrected|revised|updated|amended|fixed|reattributed|retracted) [^.!?]*(attribut|belief|record|memor|confidence|the fact)',
     ARRAY['add_evidence'],
     NULL,
     'memory-correction claims require a belief revision, not just any memory write')
) AS v(claim_kind, pattern, satisfied_by_tools, require_arg_key, notes)
WHERE NOT EXISTS (
    SELECT 1 FROM action_claim_patterns p
    WHERE p.claim_kind = v.claim_kind AND p.pattern = v.pattern
);

-- Negative search-result claims (#50): a false "nothing found" kills the
-- follow-up, so these are flagged when no search-capable tool ran this turn.
INSERT INTO action_claim_patterns (claim_kind, pattern, satisfied_by_tools, require_arg_key, match_negated, notes)
SELECT v.claim_kind, v.pattern, v.satisfied_by_tools, v.require_arg_key, v.match_negated, v.notes
FROM (VALUES
    ('search_negative',
     '\m(search(ed)?|scan(ned)?|looked|checked|recall(ed)?|queried)\M[^.!?]*(returns? no|no match(es|ing)?|found no|found nothing|nothing (matching|found|like)|not (present|found)|does ?not exist|doesn''t exist|no such (file|path|memory|record)|came up empty)',
     ARRAY['inspect_source','recall','search_history','grep','glob','list_directory','sense_memory_availability','inspect_database_schema','explore_concept'],
     NULL,
     TRUE,
     'negative search-result claims require an actual search this turn')
) AS v(claim_kind, pattern, satisfied_by_tools, require_arg_key, match_negated, notes)
WHERE NOT EXISTS (
    SELECT 1 FROM action_claim_patterns p
    WHERE p.claim_kind = v.claim_kind AND p.pattern = v.pattern
);

INSERT INTO config_defaults (key, value, description) VALUES
    ('guardrails.action_claims.enabled', 'true'::jsonb,
     'Detect unsupported action claims in final assistant text and append a visible correction'),
    ('guardrails.action_claims.llm_verifier_enabled', 'false'::jsonb,
     'Confirm/extend heuristic action-claim findings with an LLM pass (llm.guardrails, falls back to llm.subconscious)')
ON CONFLICT (key) DO NOTHING;

-- Detect action claims in p_text unsupported by the turn's successful tool
-- calls or hydrated prior-action receipts. Fail-soft: unknown
-- turn or empty text returns an empty report — the guardrail is advisory and
-- must never block the reply.
CREATE OR REPLACE FUNCTION detect_unsupported_action_claims(
    p_turn_id UUID,
    p_text TEXT
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    turn agent_turns%ROWTYPE;
    current_calls JSONB;
    prior_calls JSONB;
    calls JSONB;
    flagged JSONB := '[]'::jsonb;
    sentence TEXT;
    norm TEXT;
    is_negated BOOLEAN;
    historical_reference BOOLEAN;
    checked INT := 0;
    pat RECORD;
    satisfied BOOLEAN;
    sentence_flagged BOOLEAN;
    file_tokens TEXT[];
    call_elem JSONB;
    arg_value TEXT;
    tok TEXT;
    uuid_txt TEXT;
    success_count INT := 0;
    prior_receipt_count INT := 0;
BEGIN
    IF COALESCE(trim(p_text), '') = '' THEN
        RETURN jsonb_build_object('flagged', '[]'::jsonb, 'checked_sentences', 0, 'successful_tool_calls', 0);
    END IF;

    SELECT * INTO turn FROM agent_turns WHERE id = p_turn_id;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('flagged', '[]'::jsonb, 'checked_sentences', 0,
                                  'successful_tool_calls', 0, 'error', 'turn_not_found');
    END IF;

    current_calls := COALESCE(turn.runtime_state->'tool_calls_made', '[]'::jsonb);
    SELECT COALESCE(jsonb_agg(receipt), '[]'::jsonb)
    INTO prior_calls
    FROM jsonb_array_elements(COALESCE(turn.messages, '[]'::jsonb)) message
    CROSS JOIN LATERAL jsonb_array_elements(
        CASE
            WHEN jsonb_typeof(message#>'{metadata,action_receipts}') = 'array'
                THEN message#>'{metadata,action_receipts}'
            ELSE '[]'::jsonb
        END
    ) receipt
    WHERE message->>'role' = 'assistant'
      AND COALESCE((receipt->>'success')::boolean, FALSE);
    SELECT count(*) INTO success_count
    FROM jsonb_array_elements(current_calls) c
    WHERE COALESCE((c->>'success')::boolean, FALSE);
    prior_receipt_count := jsonb_array_length(prior_calls);

    -- Split on newlines, then on sentence enders followed by whitespace, so
    -- dots inside file paths ("core/agent_loop.py") never split a sentence.
    FOR sentence IN
        SELECT trim(s2)
        FROM regexp_split_to_table(p_text, '\n+') AS s1,
             LATERAL regexp_split_to_table(s1, '[.!?]+\s+') AS s2
        WHERE length(trim(s2)) > 8
    LOOP
        checked := checked + 1;
        -- Markdown emphasis defeated literal matching ("did **not**", `path`),
        -- producing a live false positive (#48): match against a normalized
        -- copy, report the original.
        norm := regexp_replace(sentence, '[*_`~]+', '', 'g');
        historical_reference := norm ~* '\m(earlier|previously|previous (turn|message|conversation|session|exchange)|prior turn|last (turn|time|session)|already|at the time|back then|originally|yesterday)\M';
        calls := current_calls || CASE
            WHEN historical_reference THEN prior_calls
            ELSE '[]'::jsonb
        END;

        -- Futurity / hypothetical / question suppression:
        -- false negatives are acceptable for an advisory check, false
        -- accusations are not. Claims about previous turns are checked against
        -- the durable receipts hydrated above.
        CONTINUE WHEN norm ~ '\?'
            OR norm ~* '\m(will|would|could|should|cannot|can(?!''t)|going to|about to|let me|want(ed)? to|plan(ning|ned)? to|intend to|try(ing)? to|need to|if|unless|whether|once|before I|when I|instead of)\M'
            OR position('[Correction]' in norm) > 0
            OR left(sentence, 1) = '>';

        -- Negation suppression is per-pattern (#50): patterns that describe
        -- negative results (match_negated) must still see negated sentences.
        is_negated := norm ~* '\m(didn''t|did not|couldn''t|could not|can''t|cannot|haven''t|hasn''t|have not|has not|do(es)? not|don''t|doesn''t|not yet|never|unable|failed|failing|no longer)\M';

        sentence_flagged := FALSE;
        FOR pat IN SELECT * FROM action_claim_patterns WHERE enabled ORDER BY id LOOP
            EXIT WHEN sentence_flagged;
            CONTINUE WHEN is_negated AND NOT pat.match_negated;
            CONTINUE WHEN norm !~* pat.pattern;

            satisfied := FALSE;
            IF pat.require_arg_key IS NOT NULL THEN
                file_tokens := ARRAY(
                    SELECT DISTINCT m[1]
                    FROM regexp_matches(norm, '([A-Za-z0-9_./-]+\.(?:py|sql|md|ts|tsx|js|jsx|json|ya?ml|toml|sh|go|rs))', 'g') AS m
                );
            END IF;

            FOR call_elem IN
                SELECT c FROM jsonb_array_elements(calls) c
                WHERE COALESCE((c->>'success')::boolean, FALSE)
            LOOP
                EXIT WHEN satisfied;
                CONTINUE WHEN NOT EXISTS (
                    SELECT 1 FROM unnest(pat.satisfied_by_tools) t
                    WHERE (call_elem->>'name') LIKE t
                );
                IF pat.require_arg_key IS NULL OR COALESCE(array_length(file_tokens, 1), 0) = 0 THEN
                    satisfied := TRUE;
                ELSE
                    arg_value := call_elem->'arguments'->>pat.require_arg_key;
                    IF arg_value IS NOT NULL THEN
                        FOREACH tok IN ARRAY file_tokens LOOP
                            IF position(lower(tok) in lower(arg_value)) > 0
                               OR position(lower(arg_value) in lower(tok)) > 0 THEN
                                satisfied := TRUE;
                                EXIT;
                            END IF;
                        END LOOP;
                    END IF;
                END IF;
            END LOOP;

            IF NOT satisfied THEN
                sentence_flagged := TRUE;
                flagged := flagged || jsonb_build_array(jsonb_build_object(
                    'kind', pat.claim_kind,
                    'sentence', left(sentence, 300),
                    'expected_tools', to_jsonb(pat.satisfied_by_tools)
                ));
            END IF;
        END LOOP;
    END LOOP;

    -- Fabricated-artifact check: UUID literals in the reply that appear in no
    -- tool/user/system message of the turn were invented by the model.
    FOR uuid_txt IN
        SELECT DISTINCT lower(m[1])
        FROM regexp_matches(p_text, '([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})', 'g') AS m
    LOOP
        IF NOT EXISTS (
            SELECT 1 FROM jsonb_array_elements(COALESCE(turn.messages, '[]'::jsonb)) msg
            WHERE (
                    msg->>'role' IN ('tool', 'user', 'system')
                    AND position(uuid_txt in lower(COALESCE(msg->>'content', ''))) > 0
                  )
               OR position(uuid_txt in lower(COALESCE(msg#>>'{metadata,action_receipts}', ''))) > 0
        ) THEN
            flagged := flagged || jsonb_build_array(jsonb_build_object(
                'kind', 'fabricated_artifact',
                'sentence', uuid_txt,
                'expected_tools', '[]'::jsonb
            ));
        END IF;
    END LOOP;

    RETURN jsonb_build_object(
        'flagged', flagged,
        'checked_sentences', checked,
        'successful_tool_calls', success_count,
        'prior_action_receipts', prior_receipt_count
    );
END;
$$;
