-- Explicit, privacy-preserving inbound voice-note configuration and audit.
SET search_path = public, ag_catalog, "$user";

INSERT INTO config_defaults (key, value, description) VALUES
    ('voice_notes.stt.enabled', 'false'::jsonb,
     'Master gate for inbound voice-note transcription; disabled until the user chooses a provider.'),
    ('voice_notes.stt.provider', '"local_whisper"'::jsonb,
     'Inbound STT provider: local_whisper or openai_whisper.'),
    ('voice_notes.stt.model', '"base"'::jsonb,
     'Model used by the selected voice-note transcription provider.'),
    ('voice_notes.stt.provider_models', '{"local_whisper":"base","openai_whisper":"whisper-1"}'::jsonb,
     'Live provider-to-default-model catalog used by voice-note setup.'),
    ('voice_notes.stt.channels', '[]'::jsonb,
     'Optional channel allowlist for STT; empty means every configured media-capable channel.'),
    ('voice_notes.stt.max_bytes', '26214400'::jsonb,
     'Maximum inbound audio size in bytes (25 MiB by default).'),
    ('voice_notes.stt.timeout_seconds', '60'::jsonb,
     'HTTP timeout in seconds for one cloud transcription attempt.'),
    ('voice_notes.stt.language', '""'::jsonb,
     'Optional language hint; empty means automatic detection.'),
    ('voice_notes.stt.prepend_marker', 'true'::jsonb,
     'Mark injected transcript text as a voice-note transcript.'),
    ('voice_notes.stt.cloud_disclosure_accepted', 'false'::jsonb,
     'Records the explicit choice to send voice-note audio to the configured cloud STT provider.'),
    ('audio_analysis.local.enabled', 'true'::jsonb,
     'Make the optional, approval-gated local audio analysis tool available.'),
    ('audio_analysis.local.allow_autonomous', 'false'::jsonb,
     'Allow a heartbeat to request local audio analysis; tool approval still applies.'),
    ('audio_analysis.local.model', '"pyannote/speaker-diarization-community-1"'::jsonb,
     'Hugging Face pyannote model used for device-local speaker diarization.'),
    ('audio_analysis.local.max_duration_seconds', '7200'::jsonb,
     'Maximum recording duration accepted for local diarization.'),
    ('audio_analysis.local.emotion.enabled', 'false'::jsonb,
     'Permit explicitly requested coarse local acoustic heuristics; disabled by default.')
ON CONFLICT (key) DO UPDATE SET
    value = EXCLUDED.value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

CREATE OR REPLACE FUNCTION record_voice_note_stt_event(
    p_channel_type TEXT,
    p_channel_id TEXT,
    p_sender_id TEXT,
    p_message_id TEXT,
    p_attachment_id TEXT,
    p_mime_type TEXT,
    p_filename TEXT,
    p_provider TEXT,
    p_model TEXT,
    p_outcome TEXT,
    p_transcript_chars INTEGER DEFAULT NULL,
    p_error_detail TEXT DEFAULT NULL,
    p_duration_ms INTEGER DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE
    event_id BIGINT;
BEGIN
    IF NULLIF(btrim(p_channel_type), '') IS NULL THEN
        RAISE EXCEPTION 'voice-note channel type is required';
    END IF;
    IF p_outcome <> 'transcribed'
       AND p_outcome NOT LIKE 'skipped\_%' ESCAPE '\'
       AND p_outcome NOT LIKE 'failed\_%' ESCAPE '\' THEN
        RAISE EXCEPTION 'invalid voice-note STT outcome: %', p_outcome;
    END IF;
    INSERT INTO voice_note_stt_events (
        channel_type, channel_id, sender_id, message_id, attachment_id,
        mime_type, filename, provider, model, outcome, transcript_chars,
        error_detail, duration_ms, metadata
    ) VALUES (
        btrim(p_channel_type), NULLIF(btrim(p_channel_id), ''),
        NULLIF(btrim(p_sender_id), ''), NULLIF(btrim(p_message_id), ''),
        NULLIF(btrim(p_attachment_id), ''), NULLIF(btrim(p_mime_type), ''),
        NULLIF(btrim(p_filename), ''),
        COALESCE(NULLIF(btrim(p_provider), ''), 'unknown'),
        NULLIF(btrim(p_model), ''), p_outcome,
        CASE WHEN p_transcript_chars IS NULL THEN NULL ELSE GREATEST(p_transcript_chars, 0) END,
        NULLIF(left(btrim(p_error_detail), 500), ''),
        CASE WHEN p_duration_ms IS NULL THEN NULL ELSE GREATEST(p_duration_ms, 0) END,
        COALESCE(p_metadata, '{}'::jsonb)
    ) RETURNING id INTO event_id;
    RETURN event_id;
END;
$$;

COMMENT ON TABLE voice_note_stt_events IS
    'Append-only metadata audit of inbound voice-note STT; transcript content is never stored here.';
