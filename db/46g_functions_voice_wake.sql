-- Explicit companion-node wake word: server gate and metadata-only audit.
SET search_path = public, ag_catalog, "$user";

CREATE TABLE IF NOT EXISTS voice_wake_events (
    id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL UNIQUE,
    node_id TEXT NOT NULL REFERENCES hexis_nodes(node_id),
    session_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    detector_model TEXT,
    detector_score DOUBLE PRECISION CHECK (
        detector_score IS NULL OR detector_score BETWEEN 0 AND 1
    ),
    audio_bytes INTEGER NOT NULL DEFAULT 0 CHECK (audio_bytes >= 0),
    transcript_chars INTEGER CHECK (transcript_chars IS NULL OR transcript_chars >= 0),
    response_chars INTEGER CHECK (response_chars IS NULL OR response_chars >= 0),
    response_audio_bytes INTEGER CHECK (
        response_audio_bytes IS NULL OR response_audio_bytes >= 0
    ),
    outcome TEXT NOT NULL CHECK (
        outcome = 'completed' OR outcome LIKE 'failed\_%' ESCAPE '\'
    ),
    error_detail TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_voice_wake_events_created
    ON voice_wake_events (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_voice_wake_events_node
    ON voice_wake_events (node_id, created_at DESC);

INSERT INTO config_defaults (key, value, description) VALUES
    ('voice.wake.enabled', 'false'::jsonb,
     'Permit explicitly configured and paired nodes to submit wake-word utterances.'),
    ('voice.wake.max_audio_bytes', '4194304'::jsonb,
     'Maximum signed WAV payload accepted from one paired wake-word node (4 MiB).'),
    ('voice.wake.max_response_audio_bytes', '8388608'::jsonb,
     'Maximum synthesized response returned over the node WebSocket (8 MiB).')
ON CONFLICT (key) DO NOTHING;

CREATE OR REPLACE FUNCTION record_voice_wake_event(
    p_request_id UUID,
    p_node_id TEXT,
    p_session_id UUID,
    p_detector_model TEXT,
    p_detector_score DOUBLE PRECISION,
    p_audio_bytes INTEGER,
    p_transcript_chars INTEGER,
    p_response_chars INTEGER,
    p_response_audio_bytes INTEGER,
    p_outcome TEXT,
    p_error_detail TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE v_id BIGINT;
BEGIN
    IF p_request_id IS NULL OR NULLIF(btrim(COALESCE(p_node_id, '')), '') IS NULL THEN
        RAISE EXCEPTION 'wake request and node identity are required';
    END IF;
    IF p_outcome <> 'completed' AND p_outcome NOT LIKE 'failed\_%' ESCAPE '\' THEN
        RAISE EXCEPTION 'invalid wake outcome: %', p_outcome;
    END IF;
    INSERT INTO voice_wake_events (
        request_id, node_id, session_id, detector_model, detector_score,
        audio_bytes, transcript_chars, response_chars, response_audio_bytes,
        outcome, error_detail, metadata
    ) VALUES (
        p_request_id, btrim(p_node_id), p_session_id,
        NULLIF(left(btrim(p_detector_model), 200), ''),
        CASE WHEN p_detector_score IS NULL THEN NULL
             ELSE LEAST(GREATEST(p_detector_score, 0), 1) END,
        GREATEST(COALESCE(p_audio_bytes, 0), 0),
        CASE WHEN p_transcript_chars IS NULL THEN NULL ELSE GREATEST(p_transcript_chars, 0) END,
        CASE WHEN p_response_chars IS NULL THEN NULL ELSE GREATEST(p_response_chars, 0) END,
        CASE WHEN p_response_audio_bytes IS NULL THEN NULL ELSE GREATEST(p_response_audio_bytes, 0) END,
        p_outcome, NULLIF(left(btrim(p_error_detail), 500), ''),
        COALESCE(p_metadata, '{}'::jsonb)
    )
    ON CONFLICT (request_id) DO NOTHING
    RETURNING id INTO v_id;
    IF v_id IS NULL THEN
        SELECT id INTO v_id FROM voice_wake_events WHERE request_id = p_request_id;
    END IF;
    RETURN v_id;
END;
$$;

CREATE OR REPLACE FUNCTION reject_voice_wake_event_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION '% is append-only; % is not permitted', TG_TABLE_NAME, TG_OP;
END;
$$;

DROP TRIGGER IF EXISTS trg_voice_wake_events_immutable ON voice_wake_events;
CREATE TRIGGER trg_voice_wake_events_immutable
    BEFORE UPDATE OR DELETE ON voice_wake_events
    FOR EACH ROW EXECUTE FUNCTION reject_voice_wake_event_mutation();

COMMENT ON TABLE voice_wake_events IS
    'Append-only wake-turn audit; raw audio, transcript text, and response text are excluded.';
