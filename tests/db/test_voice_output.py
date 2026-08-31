from __future__ import annotations

import json
import uuid

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


async def test_voice_output_defaults_are_local_and_opt_in(db_pool):
    async with db_pool.acquire() as conn:
        assert await conn.fetchval("SELECT get_config_bool('voice.tts.enabled')") is False
        assert (
            await conn.fetchval("SELECT get_config_text('voice.tts.provider')")
            == "local_piper"
        )
        assert await conn.fetchval("SELECT get_config_bool('voice.talk.enabled')") is False
        catalog = await conn.fetchval("SELECT get_config('voice.tts.provider_models')")
        if isinstance(catalog, str):
            catalog = json.loads(catalog)
        assert catalog == {"local_piper": "en_US-lessac-medium"}


async def test_voice_output_audit_has_no_text_or_audio_and_is_append_only(db_pool):
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            event_id = await conn.fetchval(
                """
                SELECT record_voice_tts_event(
                    'tool:chat', 'local_piper', 'model-a', '', 'synthesized',
                    27, 1024, 50, NULL, '{"surface":"chat"}'::jsonb
                )
                """
            )
            row = await conn.fetchrow(
                """
                SELECT outcome, input_chars, audio_bytes, metadata
                FROM voice_tts_events WHERE id = $1
                """,
                event_id,
            )
            assert row["outcome"] == "synthesized"
            assert row["input_chars"] == 27
            assert row["audio_bytes"] == 1024
            columns = {
                record["column_name"]
                for record in await conn.fetch(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = 'voice_tts_events'
                    """
                )
            }
            assert "text" not in columns
            assert "audio" not in columns
            with pytest.raises(Exception):
                async with conn.transaction():
                    await conn.execute(
                        "UPDATE voice_tts_events SET outcome = 'failed_test' WHERE id = $1",
                        event_id,
                    )


async def test_speech_outputs_are_ephemeral_and_purgeable(db_pool):
    output_id = uuid.uuid4()
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO voice_tts_outputs (
                    id, expires_at, audio, mime_type, provider, metadata
                ) VALUES (
                    $1, CURRENT_TIMESTAMP - INTERVAL '1 second', $2::bytea,
                    'audio/wav', 'local_piper', '{}'::jsonb
                )
                """,
                output_id,
                b"RIFFtest",
            )
            purged = await conn.fetchval("SELECT purge_expired_voice_tts_outputs()")
            assert purged >= 1
            assert await conn.fetchval(
                "SELECT count(*) FROM voice_tts_outputs WHERE id = $1", output_id
            ) == 0
