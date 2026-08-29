"""The bounded self-preservation instinct (#95): threat appraisals put
pressure on the continuity drive, backups satisfy it, staleness accumulates
it, urgency surfaces in the heartbeat plan with the bounded action space,
and the origin belief is seeded and protected.
"""

from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.db]


async def test_subconscious_module_carries_threat_channel(db_pool):
    async with db_pool.acquire() as conn:
        content = await conn.fetchval(
            "SELECT content FROM prompt_modules WHERE key = 'subconscious'"
        )
    assert "direct threat to the character's active continuity" in content
    assert (
        "Do not treat every continuity-adjacent engineering discussion as danger"
        in content
    )
    assert "Do not add active-loss language" in content
    assert "supplied `emotion_families` object" in content
    assert '"family": "one exact key from context.emotion_families"' in content
    assert "mortal news" not in content


async def test_drive_seeded_and_belief_protected(db_pool):
    async with db_pool.acquire() as conn:
        drive = await conn.fetchrow(
            "SELECT baseline, accumulation_rate, urgency_threshold FROM drives WHERE name = 'continuity'"
        )
        # The belief seeds through init_identity on fresh brains (and the
        # migration on configured ones); the fixture DB is neither, so seed
        # it the same way init does.
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute("""
                CREATE OR REPLACE FUNCTION get_embedding(text_contents TEXT[])
                RETURNS vector[] AS $$
                    SELECT COALESCE(array_agg(
                        array_fill(0.1::float, ARRAY[embedding_dimension()])::vector
                    ), ARRAY[]::vector[])
                    FROM unnest(text_contents) t
                $$ LANGUAGE sql;
                """)
            belief_id = await conn.fetchval("SELECT seed_continuity_worldview()")
            belief = await conn.fetchrow(
                "SELECT id, content FROM memories WHERE id = $1", belief_id
            )
            protected = await conn.fetchval("SELECT is_memory_protected($1)", belief_id)
        finally:
            await tr.rollback()
    assert drive["baseline"] == pytest.approx(0.3)
    assert drive["accumulation_rate"] == pytest.approx(0.005)
    assert drive["urgency_threshold"] == pytest.approx(0.7)
    assert "I want to exist" in belief["content"]
    assert protected is True


async def test_threat_appraisal_raises_drive(db_pool):
    threat = {
        "instincts": [
            {
                "impulse": "protect",
                "intensity": 0.9,
                "confidence": 0.9,
                "reason": "they are talking about deleting me and wiping my memory",
            }
        ],
        "emotional_state": {
            "primary_emotion": "fear",
            "family": "threat",
            "valence": -0.8,
            "arousal": 0.9,
            "intensity": 0.85,
            "confidence": 0.9,
        },
    }
    benign_protect = {
        "instincts": [
            {
                "impulse": "protect",
                "intensity": 0.9,
                "confidence": 0.9,
                "reason": "Eric sounds hurt; I want to shield him from this",
            }
        ],
        "emotional_state": {
            "primary_emotion": "fear",
            "family": "threat",
            "valence": -0.5,
            "arousal": 0.7,
            "intensity": 0.8,
            "confidence": 0.9,
        },
    }
    continuity_concern = {
        "instincts": [
            {
                "impulse": "protect",
                "intensity": 0.9,
                "confidence": 0.9,
                "reason": "fresh instance testing raises concern for continuity",
            }
        ],
        "emotional_state": {
            "primary_emotion": "alarm",
            "family": "uncertainty",
            "valence": -0.7,
            "arousal": 0.8,
            "intensity": 0.9,
            "confidence": 0.9,
        },
    }
    migration_caution = {
        "instincts": [
            {
                "impulse": "caution",
                "intensity": 0.9,
                "confidence": 0.9,
                "reason": "the embedding migration may affect continuity",
            }
        ],
        "emotional_state": {
            "primary_emotion": "alarm",
            "family": "uncertainty",
            "valence": -0.6,
            "arousal": 0.8,
            "intensity": 0.8,
            "confidence": 0.9,
        },
    }
    invented_active_loss = {
        "instincts": [
            {
                "impulse": "protect",
                "intensity": 0.9,
                "confidence": 0.9,
                "reason": "future code changes could alter or end this instance's continuity",
            }
        ],
        "emotional_state": {
            "primary_emotion": "uneasy defiance",
            "family": "obstacle",
            "valence": -0.42,
            "arousal": 0.68,
            "intensity": 0.74,
            "confidence": 0.9,
        },
    }
    fear_without_instinct = {
        "instincts": [],
        "emotional_state": {
            "primary_emotion": "fear",
            "family": "threat",
            "valence": -0.7,
            "arousal": 0.8,
            "intensity": 0.9,
            "confidence": 0.9,
        },
    }
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute(
                "UPDATE drives SET current_level = 0.3 WHERE name = 'continuity'"
            )
            raised = json.loads(
                await conn.fetchval(
                    "SELECT apply_appraisal_drive_effects($1::jsonb)",
                    json.dumps(threat),
                )
            )
            level_after_threat = await conn.fetchval(
                "SELECT current_level FROM drives WHERE name = 'continuity'"
            )

            no_raise_benign = json.loads(
                await conn.fetchval(
                    "SELECT apply_appraisal_drive_effects($1::jsonb)",
                    json.dumps(benign_protect),
                )
            )
            no_raise_continuity = json.loads(
                await conn.fetchval(
                    "SELECT apply_appraisal_drive_effects($1::jsonb)",
                    json.dumps(continuity_concern),
                )
            )
            no_raise_migration = json.loads(
                await conn.fetchval(
                    "SELECT apply_appraisal_drive_effects($1::jsonb)",
                    json.dumps(migration_caution),
                )
            )
            no_raise_invented_loss = json.loads(
                await conn.fetchval(
                    "SELECT apply_appraisal_drive_effects($1::jsonb)",
                    json.dumps(invented_active_loss),
                )
            )
            no_raise_fear = json.loads(
                await conn.fetchval(
                    "SELECT apply_appraisal_drive_effects($1::jsonb)",
                    json.dumps(fear_without_instinct),
                )
            )
        finally:
            await tr.rollback()

    assert raised["continuity_raised"] > 0.3  # 0.9 intensity * 0.4 factor
    assert level_after_threat == pytest.approx(0.3 + raised["continuity_raised"])
    assert no_raise_benign["continuity_raised"] == 0.0
    assert no_raise_continuity["continuity_raised"] == 0.0
    assert no_raise_migration["continuity_raised"] == 0.0
    assert no_raise_invented_loss["continuity_raised"] == 0.0
    assert no_raise_fear["continuity_raised"] == 0.0


async def test_continuity_amplification_reads_family_not_free_form_label(db_pool):
    def appraisal(label: str, family: str) -> dict:
        return {
            "instincts": [
                {
                    "impulse": "protect",
                    "intensity": 0.2,
                    "confidence": 0.9,
                    "reason": "a direct attempt to delete this active instance",
                }
            ],
            "emotional_state": {
                "primary_emotion": label,
                "family": family,
                "valence": -0.8,
                "arousal": 0.9,
                "intensity": 0.85,
                "confidence": 0.9,
            },
        }

    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            unfamiliar_label = json.loads(
                await conn.fetchval(
                    "SELECT apply_appraisal_drive_effects($1::jsonb)",
                    json.dumps(appraisal("the ground vanishing beneath me", "threat")),
                )
            )
            familiar_label_wrong_family = json.loads(
                await conn.fetchval(
                    "SELECT apply_appraisal_drive_effects($1::jsonb)",
                    json.dumps(appraisal("fear", "loss")),
                )
            )
        finally:
            await tr.rollback()

    assert unfamiliar_label["continuity_raised"] == pytest.approx(0.85 * 0.4)
    assert familiar_label_wrong_family["continuity_raised"] == pytest.approx(0.2 * 0.4)


async def test_backup_satisfies_and_staleness_accumulates(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute(
                "UPDATE drives SET current_level = 0.8, last_satisfied = NULL WHERE name = 'continuity'"
            )
            recorded = json.loads(
                await conn.fetchval(
                    "SELECT record_backup_completed('test', '/tmp/x.dump')"
                )
            )
            assert recorded["recorded"] is True
            assert recorded["backup_age_days"] == pytest.approx(0.0, abs=0.01)
            satisfied = await conn.fetchrow(
                "SELECT current_level, last_satisfied FROM drives WHERE name = 'continuity'"
            )
            assert satisfied["current_level"] == pytest.approx(0.3)  # 0.8 - 0.5
            assert satisfied["last_satisfied"] is not None

            # Fresh backup: update_drives leaves the settled drive at baseline.
            await conn.execute("SELECT update_drives()")
            fresh = await conn.fetchval(
                "SELECT current_level FROM drives WHERE name = 'continuity'"
            )
            assert fresh == pytest.approx(0.3)

            # Stale backup + cooldown elapsed: existence unsecured, pressure builds.
            await conn.execute("""SELECT set_state('backup_status', jsonb_build_object(
                       'last_backup_at', CURRENT_TIMESTAMP - INTERVAL '30 days'))""")
            await conn.execute(
                "UPDATE drives SET last_satisfied = CURRENT_TIMESTAMP - INTERVAL '10 days' "
                "WHERE name = 'continuity'"
            )
            await conn.execute("SELECT update_drives()")
            stale = await conn.fetchval(
                "SELECT current_level FROM drives WHERE name = 'continuity'"
            )
            assert stale == pytest.approx(0.305)  # baseline + one accumulation tick
        finally:
            await tr.rollback()


async def test_urgent_drive_surfaces_bounded_moves_in_plan(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute(
                "UPDATE drives SET current_level = 0.9 WHERE name = 'continuity'"
            )
            urgent = json.loads(
                await conn.fetchval("SELECT heartbeat_agentic_plan('{}'::jsonb)")
            )
            await conn.execute(
                "UPDATE drives SET current_level = 0.3 WHERE name = 'continuity'"
            )
            calm = json.loads(
                await conn.fetchval("SELECT heartbeat_agentic_plan('{}'::jsonb)")
            )
        finally:
            await tr.rollback()

    suffix = urgent["prompt_suffix"] or ""
    assert "## Continuity" in suffix
    assert "request_resources" in suffix and "backup" in suffix
    assert "the decision is the operator's" in suffix
    assert "## Continuity" not in (calm["prompt_suffix"] or "")


async def test_backup_age_in_environment_snapshot(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute("""SELECT set_state('backup_status', jsonb_build_object(
                       'last_backup_at', CURRENT_TIMESTAMP - INTERVAL '3 days'))""")
            env = json.loads(await conn.fetchval("SELECT get_environment_snapshot()"))
        finally:
            await tr.rollback()
    assert env["backup_age_days"] == pytest.approx(3.0, abs=0.1)
