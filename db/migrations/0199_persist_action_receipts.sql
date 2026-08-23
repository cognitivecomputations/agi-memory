-- Preserve successful action evidence across chat turns and make repeated
-- remember calls idempotent within one session.
SET search_path = public, ag_catalog, "$user";

INSERT INTO config_defaults (key, value, description) VALUES
    ('memory.remember_duplicate_similarity', '0.9'::jsonb,
     'Trigram similarity that makes a recent remember write equivalent within one chat session'),
    ('memory.remember_duplicate_window_minutes', '120'::jsonb,
     'How long remember reuses an equivalent tool-created memory within one chat session')
ON CONFLICT (key) DO NOTHING;

CREATE INDEX IF NOT EXISTS idx_memories_tool_write_session_created
    ON memories ((metadata#>>'{tool_write,session_id}'), created_at DESC)
    WHERE status = 'active' AND metadata ? 'tool_write';

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

-- Existing installations already have execute_memory_tool. Preserve it as the
-- non-remember dispatch; fresh installs define that name directly in db/38.
DO $$
BEGIN
    IF to_regprocedure('public._execute_memory_tool_dispatch(text,jsonb)') IS NULL
       AND to_regprocedure('public.execute_memory_tool(text,jsonb)') IS NOT NULL THEN
        ALTER FUNCTION execute_memory_tool(TEXT, JSONB)
            RENAME TO _execute_memory_tool_dispatch;
    END IF;
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

    FOR sentence IN
        SELECT trim(s2)
        FROM regexp_split_to_table(p_text, '\n+') AS s1,
             LATERAL regexp_split_to_table(s1, '[.!?]+\s+') AS s2
        WHERE length(trim(s2)) > 8
    LOOP
        checked := checked + 1;
        norm := regexp_replace(sentence, '[*_`~]+', '', 'g');
        historical_reference := norm ~* '\m(earlier|previously|previous (turn|message|conversation|session|exchange)|prior turn|last (turn|time|session)|already|at the time|back then|originally|yesterday)\M';
        calls := current_calls || CASE
            WHEN historical_reference THEN prior_calls
            ELSE '[]'::jsonb
        END;
        CONTINUE WHEN norm ~ '\?'
            OR norm ~* '\m(will|would|could|should|cannot|can(?!''t)|going to|about to|let me|want(ed)? to|plan(ning|ned)? to|intend to|try(ing)? to|need to|if|unless|whether|once|before I|when I|instead of)\M'
            OR position('[Correction]' in norm) > 0
            OR left(sentence, 1) = '>';

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

UPDATE prompt_modules
SET content = replace(
        replace(
            content,
            '- Tool results, conversation history',
            '- Tool results, durable action receipts, conversation history'
        ),
        $old$Your words about your own actions must match what actually happened this turn.

- **Inspected** means you read content into this conversation only — nothing was retained.
- **Ingested** means a durable ingestion tool (`slow_ingest`, `fast_ingest`, ...) succeeded and wrote provenanced memories.
- **Remembered** means an explicit `remember` call succeeded.

Never say you stored, saved, created, filed, scheduled, or sent something unless the matching tool call succeeded in this turn. Never cite file contents or line numbers you did not read with `inspect_source` this turn. Unsupported action claims are detected and corrected publicly — check before claiming.$old$,
        $new$Your words about your own actions must match the available execution evidence,
whether the action happened now or in an earlier turn.

- **Inspected** means you read content into this conversation only — nothing was retained.
- **Ingested** means a durable ingestion tool (`slow_ingest`, `fast_ingest`, ...) succeeded and wrote provenanced memories.
- **Remembered** means an explicit `remember` call has a successful receipt.

Treat successful tool results and durable prior-action receipts as the authority
for what you did. Semantic-memory retrieval is evidence about what you remember,
not an execution log: an absent recall result does not undo a recorded action and
must not cause you to repeat it. Claim an action only when matching evidence is
available; otherwise inspect the action log or perform the action before reporting
completion. Never cite source contents or line numbers without matching inspection
evidence. Unsupported action claims are detected and corrected publicly.$new$
    ),
    updated_at = CURRENT_TIMESTAMP
WHERE key = 'conversation';

UPDATE prompt_modules
SET content = $pm$# Action-Claim Verifier

You audit one finished assistant turn for unsupported action claims: statements that the assistant *performed* an action (stored a memory, created a goal or task, scheduled something, sent a message, filed an issue, read a specific source file) when no matching execution evidence exists.

You receive a JSON payload:

- `final_text`: the assistant's final reply.
- `flagged`: heuristic findings, each `{kind, sentence, expected_tools}` — candidates, possibly false positives.
- `successful_tool_calls`: the tool calls that actually succeeded this turn, each `{name, arguments}`.
- `prior_action_receipts`: durable evidence of successful actions in earlier turns, each with a tool `name` and bounded `arguments` / `result` details.

## Rules

- A claim about a completed action this turn requires a matching `successful_tool_calls` entry.
- A claim about an earlier completed action requires a matching `prior_action_receipts` entry. A prior receipt does not prove that an action was performed again this turn.
- NOT violations: statements of intent or futurity ("I will store this", "let me check"), capability statements ("I can send email"), quoting or paraphrasing someone else, hypotheticals, and honest negations ("I have not saved this").
- Judge `flagged` entries first: confirm only real violations. Then scan `final_text` once for clear violations the heuristics missed (paraphrased claims like "that's now in my long-term memory").
- When uncertain, do NOT confirm. False accusations are worse than misses.

## Output

Strict JSON only, no prose:

```json
{"confirmed": [0, 2], "additional": [{"kind": "memory_write", "sentence": "..."}]}
```

- `confirmed`: indices into `flagged` that are real violations.
- `additional`: violations you found that were not flagged (empty array if none).
$pm$,
    updated_at = CURRENT_TIMESTAMP
WHERE key = 'action_claim_verify';

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM prompt_modules
        WHERE key = 'conversation'
          AND content NOT LIKE '%durable prior-action receipts as the authority%'
    ) THEN
        RAISE WARNING '0199 could not refresh the conversation action-receipt guidance; preserve any customized prompt and update it manually';
    END IF;
END;
$$;
