from __future__ import annotations

import uuid

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


async def test_wake_defaults_are_off_and_bounded(db_pool):
    async with db_pool.acquire() as conn:
        assert await conn.fetchval("SELECT get_config_bool('voice.wake.enabled')") is False
        assert (
            await conn.fetchval("SELECT get_config_int('voice.wake.max_audio_bytes')")
            == 4_194_304
        )
        assert await conn.fetchval(
            "SELECT get_config_int('voice.wake.max_response_audio_bytes')"
        ) == 8_388_608


async def test_wake_audit_excludes_content_and_is_append_only(db_pool):
    node_id = "wake-test-" + uuid.uuid4().hex
    request_id = uuid.uuid4()
    session_id = uuid.uuid4()
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO hexis_nodes (node_id, public_key, name, capabilities)
                VALUES ($1, 'test-key', 'Wake test', '["audio.wake"]'::jsonb)
                """,
                node_id,
            )
            event_id = await conn.fetchval(
                """
                SELECT record_voice_wake_event(
                    $1, $2, $3, 'custom-model', 0.7, 2048, 12, 24, 4096,
                    'completed', NULL, '{"detector_label":"wake"}'::jsonb
                )
                """,
                request_id,
                node_id,
                session_id,
            )
            row = await conn.fetchrow(
                """
                SELECT transcript_chars, response_chars, response_audio_bytes
                FROM voice_wake_events WHERE id=$1
                """,
                event_id,
            )
            assert dict(row) == {
                "transcript_chars": 12,
                "response_chars": 24,
                "response_audio_bytes": 4096,
            }
            columns = {
                record["column_name"]
                for record in await conn.fetch(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema='public' AND table_name='voice_wake_events'
                    """
                )
            }
            assert "audio" not in columns
            assert "transcript" not in columns
            assert "response" not in columns
            with pytest.raises(Exception):
                async with conn.transaction():
                    await conn.execute(
                        "UPDATE voice_wake_events SET outcome='failed_test' WHERE id=$1",
                        event_id,
                    )
