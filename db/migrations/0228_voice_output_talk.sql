-- Opt-in local speech synthesis with metadata-only audit and ephemeral output.
SET search_path = public, ag_catalog, "$user";

CREATE TABLE IF NOT EXISTS voice_tts_events (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT,
    voice TEXT,
    outcome TEXT NOT NULL CHECK (
        outcome = 'synthesized'
        OR outcome LIKE 'skipped\_%' ESCAPE '\'
        OR outcome LIKE 'failed\_%' ESCAPE '\'
    ),
    input_chars INTEGER NOT NULL DEFAULT 0 CHECK (input_chars >= 0),
    audio_bytes INTEGER CHECK (audio_bytes IS NULL OR audio_bytes >= 0),
    duration_ms INTEGER CHECK (duration_ms IS NULL OR duration_ms >= 0),
    error_detail TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_voice_tts_events_created
    ON voice_tts_events (created_at DESC);

CREATE TABLE IF NOT EXISTS voice_tts_outputs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP + INTERVAL '1 hour',
    audio BYTEA NOT NULL,
    mime_type TEXT NOT NULL DEFAULT 'audio/wav',
    provider TEXT NOT NULL,
    model TEXT,
    voice TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_voice_tts_outputs_expiry
    ON voice_tts_outputs (expires_at);

INSERT INTO config_defaults (key, value, description) VALUES
    ('voice.tts.enabled', 'false'::jsonb,
     'Master gate for speech synthesis; disabled until the operator enables a local provider.'),
    ('voice.tts.provider', '"local_piper"'::jsonb,
     'Speech synthesis provider. OSS currently supports the local Piper-compatible sidecar.'),
    ('voice.tts.model', '"en_US-lessac-medium"'::jsonb,
     'Model requested from the selected speech synthesis provider.'),
    ('voice.tts.provider_models', '{"local_piper":"en_US-lessac-medium"}'::jsonb,
     'Live provider-to-default-model catalog used by voice setup.'),
    ('voice.tts.voice', '""'::jsonb,
     'Optional provider voice override; empty uses the sidecar model voice.'),
    ('voice.tts.max_chars', '4000'::jsonb,
     'Maximum characters accepted by one speech synthesis request.'),
    ('voice.tts.max_audio_bytes', '16777216'::jsonb,
     'Maximum synthesized audio response retained or returned (16 MiB by default).'),
    ('voice.tts.timeout_seconds', '60'::jsonb,
     'HTTP timeout for one local speech synthesis request.'),
    ('voice.tts.output_ttl_minutes', '60'::jsonb,
     'Minutes a tool-created speech output remains retrievable.'),
    ('voice.talk.enabled', 'false'::jsonb,
     'Allow foreground PWA talk mode after explicit per-session microphone activation.'),
    ('voice.talk.max_utterance_seconds', '60'::jsonb,
     'Maximum length of one foreground talk-mode utterance.')
ON CONFLICT (key) DO NOTHING;

CREATE OR REPLACE FUNCTION record_voice_tts_event(
    p_source TEXT,
    p_provider TEXT,
    p_model TEXT,
    p_voice TEXT,
    p_outcome TEXT,
    p_input_chars INTEGER DEFAULT 0,
    p_audio_bytes INTEGER DEFAULT NULL,
    p_duration_ms INTEGER DEFAULT NULL,
    p_error_detail TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE v_id BIGINT;
BEGIN
    IF NULLIF(btrim(COALESCE(p_source, '')), '') IS NULL THEN
        RAISE EXCEPTION 'speech source is required';
    END IF;
    IF p_outcome <> 'synthesized'
       AND p_outcome NOT LIKE 'skipped\_%' ESCAPE '\'
       AND p_outcome NOT LIKE 'failed\_%' ESCAPE '\' THEN
        RAISE EXCEPTION 'invalid speech synthesis outcome: %', p_outcome;
    END IF;
    INSERT INTO voice_tts_events (
        source, provider, model, voice, outcome, input_chars, audio_bytes,
        duration_ms, error_detail, metadata
    ) VALUES (
        btrim(p_source), COALESCE(NULLIF(btrim(p_provider), ''), 'unknown'),
        NULLIF(btrim(p_model), ''), NULLIF(btrim(p_voice), ''), p_outcome,
        GREATEST(COALESCE(p_input_chars, 0), 0),
        CASE WHEN p_audio_bytes IS NULL THEN NULL ELSE GREATEST(p_audio_bytes, 0) END,
        CASE WHEN p_duration_ms IS NULL THEN NULL ELSE GREATEST(p_duration_ms, 0) END,
        NULLIF(left(btrim(p_error_detail), 500), ''), COALESCE(p_metadata, '{}'::jsonb)
    ) RETURNING id INTO v_id;
    RETURN v_id;
END;
$$;

CREATE OR REPLACE FUNCTION reject_voice_tts_event_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION '% is append-only; % is not permitted', TG_TABLE_NAME, TG_OP;
END;
$$;

DROP TRIGGER IF EXISTS trg_voice_tts_events_immutable ON voice_tts_events;
CREATE TRIGGER trg_voice_tts_events_immutable
    BEFORE UPDATE OR DELETE ON voice_tts_events
    FOR EACH ROW EXECUTE FUNCTION reject_voice_tts_event_mutation();

CREATE OR REPLACE FUNCTION purge_expired_voice_tts_outputs()
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE v_count INTEGER;
BEGIN
    DELETE FROM voice_tts_outputs WHERE expires_at <= CURRENT_TIMESTAMP;
    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END;
$$;

COMMENT ON TABLE voice_tts_events IS
    'Append-only metadata audit of speech synthesis; input text and audio are excluded.';
COMMENT ON TABLE voice_tts_outputs IS
    'Ephemeral synthesized audio addressed by opaque id; input text is never copied here.';
