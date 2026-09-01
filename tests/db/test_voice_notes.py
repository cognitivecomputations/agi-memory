from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


async def test_voice_note_defaults_require_explicit_enablement(db_pool):
    async with db_pool.acquire() as conn:
        assert await conn.fetchval("SELECT get_config_bool('voice_notes.stt.enabled')") is False
        assert await conn.fetchval("SELECT get_config_text('voice_notes.stt.provider')") == "local_whisper"
        catalog = await conn.fetchval("SELECT get_config('voice_notes.stt.provider_models')")
        if isinstance(catalog, str):
            catalog = json.loads(catalog)
        assert catalog["local_whisper"] == "base"
        assert catalog["openai_whisper"] == "whisper-1"
        assert await conn.fetchval(
            "SELECT get_config_bool('voice_notes.stt.cloud_disclosure_accepted')"
        ) is False


async def test_voice_note_audit_is_metadata_only_and_append_only(db_pool):
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            event_id = await conn.fetchval(
                """
                SELECT record_voice_note_stt_event(
                    'telegram', 'chat', 'sender', 'message', 'attachment',
                    'audio/ogg', 'memo.ogg', 'local_whisper', 'base',
                    'transcribed', 42, NULL, 100, '{}'::jsonb
                )
                """
            )
            row = await conn.fetchrow(
                "SELECT outcome, transcript_chars, metadata FROM voice_note_stt_events WHERE id = $1",
                event_id,
            )
            assert row["outcome"] == "transcribed"
            assert row["transcript_chars"] == 42
            columns = {
                record["column_name"]
                for record in await conn.fetch(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = 'voice_note_stt_events'
                    """
                )
            }
            assert "transcript" not in columns
            with pytest.raises(Exception):
                async with conn.transaction():
                    await conn.execute(
                        "UPDATE voice_note_stt_events SET outcome = 'failed_test' WHERE id = $1",
                        event_id,
                    )
