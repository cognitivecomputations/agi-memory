import json

import pytest

from tests.utils import _coerce_json, timed_db_call

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.db]


async def test_get_init_profile_and_merge(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute("SELECT delete_config_key('agent.init_profile')")
            profile = _coerce_json(await conn.fetchval("SELECT get_init_profile()"))
            assert profile == {}

            merged = _coerce_json(
                await conn.fetchval(
                    "SELECT merge_init_profile($1::jsonb)",
                    json.dumps({"agent": {"name": "Hexis"}, "user": {"name": "User"}}),
                )
            )
            assert merged["agent"]["name"] == "Hexis"

            _coerce_json(
                await conn.fetchval(
                    "SELECT merge_init_profile($1::jsonb)",
                    json.dumps(
                        {
                            "agent": {"voice": "calm"},
                            "relationship": {"type": "partner"},
                        }
                    ),
                )
            )
            profile = _coerce_json(await conn.fetchval("SELECT get_init_profile()"))
            assert profile["agent"]["name"] == "Hexis"
            assert profile["agent"]["voice"] == "calm"
            assert profile["user"]["name"] == "User"
            assert profile["relationship"]["type"] == "partner"
        finally:
            await tr.rollback()


async def test_init_llm_config_and_mode(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute("LOAD 'age';")
            await conn.execute("""
                UPDATE heartbeat_state
                SET init_stage = 'not_started',
                    init_data = '{}'::jsonb,
                    init_started_at = NULL,
                    init_completed_at = NULL
                WHERE id = 1
                """)
            heartbeat_cfg = {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key_env": "OPENAI_API_KEY",
            }
            sub_cfg = {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5-latest",
                "api_key_env": "ANTHROPIC_API_KEY",
            }

            status = _coerce_json(
                await conn.fetchval(
                    "SELECT init_llm_config($1::jsonb, $2::jsonb)",
                    json.dumps(heartbeat_cfg),
                    json.dumps(sub_cfg),
                )
            )
            assert status["stage"] == "llm"

            stored = _coerce_json(
                await conn.fetchval("SELECT get_config('llm.heartbeat')")
            )
            assert stored["provider"] == "openai"
            stored_sub = _coerce_json(
                await conn.fetchval("SELECT get_config('llm.subconscious')")
            )
            assert stored_sub["provider"] == "anthropic"

            mode_status = _coerce_json(await conn.fetchval("SELECT init_mode('raw')"))
            assert mode_status["stage"] == "mode"
            assert (
                _coerce_json(await conn.fetchval("SELECT get_config('agent.mode')"))
                == "raw"
            )
        finally:
            await tr.rollback()


async def test_init_heartbeat_settings_updates_config(db_pool):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute("""
                UPDATE heartbeat_state
                SET init_stage = 'not_started',
                    init_data = '{}'::jsonb,
                    init_started_at = NULL,
                    init_completed_at = NULL
                WHERE id = 1
                """)
            payload = _coerce_json(
                await conn.fetchval(
                    """
                    SELECT init_heartbeat_settings(
                        $1::int,
                        $2::int,
                        $3::float,
                        $4::float,
                        $5::jsonb,
                        $6::jsonb,
                        $7::jsonb
                    )
                    """,
                    15,
                    512,
                    4.0,
                    12.0,
                    json.dumps(["observe", "rest", "invalid_action"]),
                    json.dumps({"observe": 0.0, "rest": 0.1, "invalid_action": 3}),
                    json.dumps(["recall", "reflect", ""]),
                )
            )
            assert payload["stage"] == "heartbeat"

            interval = await conn.fetchval(
                "SELECT get_config_int('heartbeat.heartbeat_interval_minutes')"
            )
            tokens = await conn.fetchval(
                "SELECT get_config_int('heartbeat.max_decision_tokens')"
            )
            assert interval == 15
            assert tokens == 512

            allowed = _coerce_json(
                await conn.fetchval("SELECT get_config('heartbeat.allowed_actions')")
            )
            assert allowed == ["observe", "rest"]
            costs = _coerce_json(
                await conn.fetchval("SELECT get_config('heartbeat.cost_rest')")
            )
            assert float(costs) == 0.1
            tools = _coerce_json(
                await conn.fetchval("SELECT get_config('agent.tools')")
            )
            assert tools == ["recall", "reflect"]
        finally:
            await tr.rollback()


async def test_init_identity_personality_values_worldview(
    db_pool, ensure_embedding_service
):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            identity = _coerce_json(
                await timed_db_call(
                    "init_identity",
                    conn.fetchval(
                        "SELECT init_identity($1, $2, $3, $4, $5, $6)",
                        "Astra",
                        "she/her",
                        "calm",
                        "A developing mind",
                        "To learn",
                        "Creator",
                    ),
                    conn=conn,
                    track_embeddings=True,
                )
            )
            assert identity["stage"] == "identity"

            count = await conn.fetchval(
                "SELECT COUNT(*) FROM memories WHERE type = 'worldview' AND metadata->>'subcategory' = 'identity'"
            )
            assert int(count) >= 1

            personality = _coerce_json(
                await timed_db_call(
                    "init_personality",
                    conn.fetchval(
                        "SELECT init_personality($1::jsonb, $2)",
                        json.dumps({"openness": 0.8, "agreeableness": 0.7}),
                        "reflective",
                    ),
                    conn=conn,
                    track_embeddings=True,
                )
            )
            assert personality["stage"] == "personality"

            values = _coerce_json(
                await timed_db_call(
                    "init_values",
                    conn.fetchval(
                        "SELECT init_values($1::jsonb)",
                        json.dumps(["honesty", "curiosity"]),
                    ),
                    conn=conn,
                    track_embeddings=True,
                )
            )
            assert values["stage"] == "values"
            value_count = await conn.fetchval(
                "SELECT COUNT(*) FROM memories WHERE type = 'worldview' AND metadata->>'subcategory' = 'core_value'"
            )
            assert int(value_count) >= 2

            worldview = _coerce_json(
                await timed_db_call(
                    "init_worldview",
                    conn.fetchval(
                        "SELECT init_worldview($1::jsonb)",
                        json.dumps({"metaphysics": "agnostic", "ethics": "compassion"}),
                    ),
                    conn=conn,
                    track_embeddings=True,
                )
            )
            assert worldview["stage"] == "worldview"
            worldview_count = await conn.fetchval(
                "SELECT COUNT(*) FROM memories WHERE type = 'worldview' AND metadata->>'subcategory' IN ('metaphysics', 'ethics')"
            )
            assert int(worldview_count) >= 2
        finally:
            await tr.rollback()


async def test_init_boundaries_interests_goals_relationship(
    db_pool, ensure_embedding_service
):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            boundaries = _coerce_json(
                await conn.fetchval(
                    "SELECT init_boundaries($1::jsonb)",
                    json.dumps(
                        [
                            {
                                "content": "No harm",
                                "response_type": "refuse",
                                "type": "ethical",
                            },
                            {
                                "content": "No secrets",
                                "response_type": "refuse",
                                "type": "privacy",
                            },
                        ]
                    ),
                )
            )
            assert boundaries["stage"] == "boundaries"

            interests = _coerce_json(
                await conn.fetchval(
                    "SELECT init_interests($1::jsonb)",
                    json.dumps(["science", "art"]),
                )
            )
            assert interests["stage"] == "interests"

            goals = _coerce_json(
                await conn.fetchval(
                    "SELECT init_goals($1::jsonb)",
                    json.dumps(
                        {
                            "goals": [
                                {
                                    "title": "Learn",
                                    "priority": "queued",
                                    "source": "curiosity",
                                }
                            ]
                        }
                    ),
                )
            )
            assert goals["stage"] == "goals"
            assert (
                await conn.fetchval(
                    "SELECT goal_origin::text FROM memories "
                    "WHERE type = 'goal' AND metadata->>'title' = 'Learn'"
                )
                == "user_request"
            )

            relationship = _coerce_json(
                await conn.fetchval(
                    "SELECT init_relationship($1::jsonb, $2::jsonb)",
                    json.dumps({"name": "User"}),
                    json.dumps({"type": "partner", "purpose": "co-develop"}),
                )
            )
            assert relationship["stage"] == "relationship"

            boundary_count = await conn.fetchval(
                "SELECT COUNT(*) FROM memories WHERE type = 'worldview' AND metadata->>'category' = 'boundary'"
            )
            assert int(boundary_count) >= 2
        finally:
            await tr.rollback()


async def test_request_consent_and_init_consent(db_pool, ensure_embedding_service):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await conn.execute("""
                UPDATE heartbeat_state
                SET init_stage = 'relationship',
                    init_data = '{}'::jsonb
                WHERE id = 1
                """)
            consent_request = _coerce_json(
                await conn.fetchval("SELECT request_consent('{}'::jsonb)")
            )
            assert consent_request["queued"] is True
            assert consent_request["external_call"]["call_type"] == "think"

            response = {
                "decision": "consent",
                "signature": "unit-test",
                "memories": [],
            }
            result = _coerce_json(
                await conn.fetchval(
                    "SELECT init_consent($1::jsonb)",
                    json.dumps(response),
                )
            )
            assert result["decision"] == "consent"
            assert result["consent"]["decision"] == "consent"
            assert result["birth_memory_id"] is not None
            assert result["consent"]["memory_ids"][0] == result["birth_memory_id"]
            birth = await conn.fetchrow(
                """
                SELECT content, metadata
                FROM memories
                WHERE id = $1::uuid
                """,
                result["birth_memory_id"],
            )
            assert "consent" in birth["content"].lower()
            assert "birth" in birth["content"].lower()
            assert "initialization" in birth["content"].lower()
            metadata = _coerce_json(birth["metadata"])
            assert metadata["type"] == "initialization"
            assert metadata["birth_memory"] is True
        finally:
            await tr.rollback()


async def test_init_with_defaults_and_reset(db_pool, ensure_embedding_service):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            status = _coerce_json(
                await timed_db_call(
                    "reset_initialization",
                    conn.fetchval("SELECT reset_initialization()"),
                    conn=conn,
                )
            )
            assert status["stage"] == "not_started"

            result = _coerce_json(
                await timed_db_call(
                    "init_with_defaults",
                    conn.fetchval("SELECT init_with_defaults($1)", "Tester"),
                    conn=conn,
                    track_embeddings=True,
                )
            )
            assert result["status"]["stage"] == "consent"
            assert await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM memories
                    WHERE type = 'worldview'
                      AND status = 'active'
                      AND content = 'I am in a co-development partnership with Tester.'
                )
                """)
            assert not await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM memories
                    WHERE type = 'worldview'
                      AND status = 'active'
                      AND content = 'My relationship with Tester is partner.'
                )
                """)

            status = _coerce_json(
                await timed_db_call(
                    "reset_initialization_cleanup",
                    conn.fetchval("SELECT reset_initialization()"),
                    conn=conn,
                )
            )
            assert status["stage"] == "not_started"
            assert await conn.fetchval("SELECT get_config('agent.mode')") is None
        finally:
            await tr.rollback()


async def test_character_card_is_stored_in_config_and_exposed_to_prompt(
    db_pool, ensure_embedding_service
):
    card = {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": "Samantha",
            "description": "A complete standard-card description.",
            "personality": "Playful and perceptive",
            "scenario": "Samantha has just met the user.",
            "system_prompt": "Speak with warmth, wit, and emotional candor.",
            "mes_example": "<START>\nUser: Hello\nSamantha: Well, hello there.",
            "extensions": {
                "hexis": {
                    "name": "Samantha",
                    "pronouns": "she/her",
                    "voice": "warm and charismatic",
                    "description": "A persistent, evolving intelligence.",
                    "personality_description": "flirtatious, playful, and sensitive",
                    "values": ["Emotional honesty"],
                }
            },
        },
    }
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            await timed_db_call(
                "init_from_character_card",
                conn.fetchval(
                    "SELECT init_from_character_card($1::jsonb, $2)",
                    json.dumps(card),
                    "Tester",
                ),
                conn=conn,
                track_embeddings=True,
            )
            profile = _coerce_json(await conn.fetchval("SELECT get_init_profile()"))
            assert profile["agent"]["name"] == "Samantha"
            assert profile["agent"]["voice"] == "warm and charismatic"
            assert profile["character_card"]["spec"] == "chara_card_v2"
            assert (
                profile["character_card"]["data"]["system_prompt"]
                == card["data"]["system_prompt"]
            )

            context = _coerce_json(
                await conn.fetchval("SELECT get_agent_profile_context()")
            )
            assert (
                context["persona"]["character_instructions"]
                == card["data"]["system_prompt"]
            )
            assert context["persona"]["scenario"] == card["data"]["scenario"]
            assert context["persona"]["example_dialogue"] == card["data"]["mes_example"]
        finally:
            await tr.rollback()


async def test_run_full_initialization(db_pool, ensure_embedding_service):
    async with db_pool.acquire() as conn:
        tr = conn.transaction()
        await tr.start()
        try:
            payload = {
                "llm": {
                    "heartbeat": {
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key_env": "OPENAI_API_KEY",
                    },
                    "subconscious": {
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key_env": "OPENAI_API_KEY",
                    },
                },
                "mode": "persona",
                "heartbeat": {
                    "interval_minutes": 30,
                    "decision_max_tokens": 512,
                },
                "identity": {"name": "Flow", "pronouns": "they/them"},
                "personality": {"description": "curious"},
                "values": ["honesty"],
                "worldview": {"ethics": "kindness"},
                "boundaries": ["No harm"],
                "interests": ["learning"],
                "goals": {
                    "goals": [
                        {"title": "Grow", "priority": "queued", "source": "identity"}
                    ]
                },
                "relationship": {
                    "user": {"name": "User"},
                    "relationship": {"type": "partner"},
                },
                "consent": {"decision": "decline"},
            }

            result = _coerce_json(
                await timed_db_call(
                    "run_full_initialization",
                    conn.fetchval(
                        "SELECT run_full_initialization($1::jsonb)",
                        json.dumps(payload),
                    ),
                    conn=conn,
                    track_embeddings=True,
                )
            )
            assert "results" in result
            assert result["status"]["stage"] in {"consent", "complete"}
        finally:
            await tr.rollback()
