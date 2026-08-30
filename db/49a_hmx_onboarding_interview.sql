-- #97: HMX onboarding interviews -- a use case fully defined in one HMX
-- file. An import can carry an `onboarding_interview` section (schema in
-- schemas/hmx-1.7.schema.json); this table persists per-import interview
-- progress (keyed by the document's own export_id) so it survives restarts
-- and can be resumed, and each answer's binding execution is recorded for
-- provenance/audit. Binding execution itself lives in
-- apply_hmx_interview_binding() below -- currently 'remember' is wired to
-- real memory creation; the other documented binding actions (ingest,
-- init_identity, create_goal, set_guardrail, create_scheduled_task,
-- set_config) fail loud with 'not_implemented' rather than silently no-op.
SET search_path = public, ag_catalog, "$user";

CREATE TABLE IF NOT EXISTS hmx_interview_state (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    export_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'completed')),
    interview_version INT NOT NULL,
    questions JSONB NOT NULL,
    answers JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_hmx_interview_state_status
    ON hmx_interview_state (status);

-- Starts (or, idempotently, returns) the persisted interview for one import.
-- p_interview is the envelope's whole onboarding_interview object.
CREATE OR REPLACE FUNCTION start_hmx_interview(
    p_export_id TEXT,
    p_interview JSONB
) RETURNS JSONB AS $$
DECLARE
    existing hmx_interview_state%ROWTYPE;
BEGIN
    SELECT * INTO existing FROM hmx_interview_state WHERE export_id = p_export_id;
    IF FOUND THEN
        RETURN to_jsonb(existing);
    END IF;

    INSERT INTO hmx_interview_state (export_id, interview_version, questions)
    VALUES (
        p_export_id,
        COALESCE((p_interview->>'version')::int, 1),
        COALESCE(p_interview->'questions', '[]'::jsonb)
    )
    RETURNING * INTO existing;
    RETURN to_jsonb(existing);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_hmx_interview_state(p_export_id TEXT)
RETURNS JSONB AS $$
    SELECT to_jsonb(s) FROM hmx_interview_state s WHERE export_id = p_export_id;
$$ LANGUAGE sql STABLE;

-- Simple `{{answer}}` token substitution into a binding's params_template.
-- Deliberately minimal (no conditionals/loops): the HMX standard's contract
-- here is "one answer fills one template", not a general templating engine.
CREATE OR REPLACE FUNCTION _hmx_interview_fill_template(
    p_template JSONB,
    p_answer_text TEXT
) RETURNS JSONB AS $$
    SELECT CASE jsonb_typeof(p_template)
        WHEN 'string' THEN to_jsonb(replace(p_template #>> '{}', '{{answer}}', COALESCE(p_answer_text, '')))
        WHEN 'object' THEN COALESCE((
            SELECT jsonb_object_agg(key, _hmx_interview_fill_template(value, p_answer_text))
            FROM jsonb_each(p_template)
        ), '{}'::jsonb)
        WHEN 'array' THEN COALESCE((
            SELECT jsonb_agg(_hmx_interview_fill_template(value, p_answer_text))
            FROM jsonb_array_elements(p_template)
        ), '[]'::jsonb)
        ELSE p_template
    END;
$$ LANGUAGE sql IMMUTABLE;

-- Executes one question's binding against its recorded answer. Returns the
-- executed tool/action result, always with a 'success' key -- an
-- unimplemented action is a loud, structured failure, never a silent no-op.
CREATE OR REPLACE FUNCTION apply_hmx_interview_binding(
    p_action TEXT,
    p_params_template JSONB,
    p_answer JSONB,
    p_export_id TEXT,
    p_question_id TEXT
) RETURNS JSONB AS $$
DECLARE
    answer_text TEXT := CASE jsonb_typeof(p_answer)
        WHEN 'string' THEN p_answer #>> '{}'
        ELSE p_answer::text
    END;
    filled JSONB := _hmx_interview_fill_template(COALESCE(p_params_template, '{}'::jsonb), answer_text);
BEGIN
    IF p_action = 'remember' THEN
        -- Provenance always carries the interview's own identity, regardless
        -- of what the HMX author's params_template did or didn't set.
        filled := filled || jsonb_build_object(
            'sources', COALESCE(filled->'sources', '[]'::jsonb) || jsonb_build_array(
                jsonb_build_object('kind', 'hmx_interview', 'ref', p_export_id, 'label', p_question_id)
            )
        );
        RETURN execute_memory_tool('remember', filled);
    END IF;
    RETURN jsonb_build_object(
        'success', false,
        'error', format('HMX onboarding-interview binding action %L is not yet implemented', p_action),
        'error_type', 'not_implemented'
    );
END;
$$ LANGUAGE plpgsql;

-- Records one question's answer (or skip), applies its binding when
-- answered, and advances/completes the interview. p_status is 'answered' or
-- 'skipped'; skips never execute a binding.
CREATE OR REPLACE FUNCTION record_hmx_interview_answer(
    p_export_id TEXT,
    p_question_id TEXT,
    p_status TEXT,
    p_answer JSONB DEFAULT NULL
) RETURNS JSONB AS $$
DECLARE
    state hmx_interview_state%ROWTYPE;
    question JSONB;
    binding_result JSONB;
    entry JSONB;
    updated_answers JSONB;
    all_required_settled BOOLEAN;
BEGIN
    IF p_status NOT IN ('answered', 'skipped') THEN
        RAISE EXCEPTION 'invalid interview answer status: %', p_status;
    END IF;

    SELECT * INTO state FROM hmx_interview_state WHERE export_id = p_export_id FOR UPDATE;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('success', false, 'error', 'no interview in progress for this export_id');
    END IF;

    SELECT value INTO question
    FROM jsonb_array_elements(state.questions) value
    WHERE value->>'id' = p_question_id;
    IF question IS NULL THEN
        RETURN jsonb_build_object('success', false, 'error', format('unknown question id: %s', p_question_id));
    END IF;

    IF p_status = 'answered' THEN
        binding_result := apply_hmx_interview_binding(
            question->'binds'->>'action', question->'binds'->'params_template',
            p_answer, p_export_id, p_question_id
        );
    END IF;

    entry := jsonb_build_object(
        'question_id', p_question_id,
        'status', p_status,
        'answer', p_answer,
        'binding_result', binding_result,
        'answered_at', CURRENT_TIMESTAMP
    );
    -- Idempotent-ish: replace any prior entry for this question rather than
    -- accumulating duplicates if a caller retries.
    UPDATE hmx_interview_state
    SET answers = (
            SELECT COALESCE(jsonb_agg(a), '[]'::jsonb)
            FROM jsonb_array_elements(answers) a
            WHERE a->>'question_id' != p_question_id
        ) || jsonb_build_array(entry),
        updated_at = CURRENT_TIMESTAMP
    WHERE export_id = p_export_id
    RETURNING answers INTO updated_answers;

    SELECT NOT EXISTS (
        SELECT 1 FROM jsonb_array_elements(state.questions) q
        WHERE COALESCE((q->>'required')::boolean, FALSE)
          AND NOT EXISTS (
              SELECT 1 FROM jsonb_array_elements(updated_answers) a
              WHERE a->>'question_id' = q->>'id' AND a->>'status' = 'answered'
          )
    ) INTO all_required_settled;

    IF all_required_settled THEN
        UPDATE hmx_interview_state SET status = 'completed', updated_at = CURRENT_TIMESTAMP
        WHERE export_id = p_export_id;
    END IF;

    RETURN jsonb_build_object(
        'success', true,
        'question_id', p_question_id,
        'status', p_status,
        'binding_result', binding_result,
        'interview_status', CASE WHEN all_required_settled THEN 'completed' ELSE 'pending' END
    );
END;
$$ LANGUAGE plpgsql;
