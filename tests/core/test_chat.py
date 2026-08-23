from contextlib import asynccontextmanager
import asyncio
from uuid import UUID, uuid4

import pytest

from services import chat as chat_mod
from core.agent_loop import AgentEvent, AgentEventData
from core.cognitive_memory_api import HydratedContext, Memory, MemoryType, RecallResult

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.core]


class _Acquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *_exc):
        return False


class _ConfigPool:
    def __init__(self, values):
        self.values = values

    def acquire(self):
        return _Acquire(self)

    async def fetchval(self, query):
        for key, value in self.values.items():
            if key in query:
                return value
        return None


class _RememberMem:
    def __init__(self, config=None):
        self._pool = _ConfigPool(config or {})
        self.raw_calls = []
        self.remember_calls = []
        self.link_calls = []
        self.record_chat_turn_memory_calls = []
        self.record_chat_session_turn_calls = []

    async def remember_turn_raw(self, *args, **kwargs):
        self.raw_calls.append((args, kwargs))
        return {"unit_id": str(uuid4()), "status": "stored"}

    async def remember(self, *args, **kwargs):
        self.remember_calls.append((args, kwargs))
        return uuid4()

    async def link_to_source_unit(self, *args, **kwargs):
        self.link_calls.append((args, kwargs))
        return True

    async def record_chat_turn_memory(self, *args, **kwargs):
        self.record_chat_turn_memory_calls.append((args, kwargs))
        return {"direct_promoted": True, "raw": {"status": "stored"}}

    async def record_chat_session_turn(self, *args, **kwargs):
        self.record_chat_session_turn_calls.append((args, kwargs))
        return {
            "session": {"session_id": kwargs.get("session_id")},
            "history": {"messages": []},
        }


async def test_remember_conversation_calls_record_chat_session_turn_for_uuid():
    mem = _RememberMem()
    session_id = str(uuid4())
    turn_id = str(uuid4())
    receipts = [
        {
            "name": "remember",
            "success": True,
            "result": {"memory_id": str(uuid4())},
        }
    ]

    await chat_mod._remember_conversation(  # noqa: SLF001
        mem,
        user_message="remember this important preference",
        assistant_message="noted",
        session_id=session_id,
        source_identity="chat:test",
        action_receipts=receipts,
        agent_turn_id=turn_id,
        surface="cli",
    )

    assert len(mem.record_chat_session_turn_calls) == 1
    assert (
        mem.record_chat_session_turn_calls[0][0][0]
        == "remember this important preference"
    )
    assert mem.record_chat_session_turn_calls[0][1]["session_id"] == session_id
    assert mem.record_chat_session_turn_calls[0][1]["surface"] == "cli"
    context = mem.record_chat_session_turn_calls[0][1]["context"]
    assert context["assistant_metadata"]["action_receipts"] == receipts
    assert context["assistant_metadata"]["agent_turn_id"] == turn_id


async def test_message_history_rehydrates_authoritative_action_receipts():
    memory_id = str(uuid4())
    receipt = {
        "name": "remember",
        "success": True,
        "arguments": {"content": "preserve the lighthouse agreement"},
        "result": {"memory_id": memory_id, "reused": False},
    }

    history = chat_mod._message_history_only(
        [
            {  # noqa: SLF001
                "role": "assistant",
                "content": "Stored it.",
                "metadata": {"action_receipts": [receipt]},
            }
        ]
    )

    assert history[0]["metadata"]["action_receipts"] == [receipt]
    assert history[1]["role"] == "system"
    assert "Authoritative prior-action evidence" in history[1]["content"]
    assert memory_id in history[1]["content"]


async def test_turn_id_survives_without_action_receipts():
    mem = _RememberMem()
    turn_id = str(uuid4())

    await chat_mod._remember_conversation(  # noqa: SLF001
        mem,
        user_message="hello",
        assistant_message="Hello.",
        session_id=str(uuid4()),
        agent_turn_id=turn_id,
    )

    context = mem.record_chat_session_turn_calls[0][1]["context"]
    assert context["assistant_metadata"] == {"agent_turn_id": turn_id}

    history = chat_mod._message_history_only(  # noqa: SLF001
        [
            {
                "role": "assistant",
                "content": "Hello.",
                "metadata": {"agent_turn_id": turn_id},
            }
        ]
    )
    assert history == [
        {
            "role": "assistant",
            "content": "Hello.",
            "metadata": {"agent_turn_id": turn_id},
        }
    ]


async def test_remember_conversation_falls_back_for_non_uuid_session():
    mem = _RememberMem()

    await chat_mod._remember_conversation(  # noqa: SLF001
        mem,
        user_message="remember this important preference",
        assistant_message="noted",
        session_id="channel:legacy:session",
        source_identity="chat:test",
    )

    assert len(mem.record_chat_turn_memory_calls) == 1
    assert (
        mem.record_chat_turn_memory_calls[0][0][0]
        == "remember this important preference"
    )


async def test_chat_turn_basic_flow(monkeypatch, db_pool):
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO config (key, value, description) VALUES ('chat.use_rlm', 'false', 'test override') "
            "ON CONFLICT (key) DO UPDATE SET value = 'false'"
        )

    class DummyMem:
        def __init__(self):
            self.remembered = []
            self.touched = []

        async def hydrate(self, *_args, **_kwargs):
            return HydratedContext(
                memories=[],
                partial_activations=[],
                identity=[],
                worldview=[],
                emotional_state=None,
                goals=None,
                urgent_drives=[],
            )

        async def touch_memories(self, ids):
            self.touched.extend(ids)

        async def remember(self, content, **_kwargs):
            self.remembered.append(content)
            return uuid4()

        async def record_chat_turn_memory(self, *_args, **_kwargs):
            self.remembered.append("turn_memory")
            return {"direct_promoted": False, "raw": {"status": "stored"}}

    mem = DummyMem()

    @asynccontextmanager
    async def fake_connect(_dsn, **_kwargs):
        yield mem

    async def fake_chat_completion(**_kwargs):
        return {"content": "hello there", "tool_calls": []}

    monkeypatch.setattr(chat_mod.CognitiveMemory, "connect", fake_connect)
    monkeypatch.setattr("core.agent_loop.chat_completion", fake_chat_completion)

    async def fake_agent_profile(_dsn=None, **_kwargs):
        return {}

    monkeypatch.setattr(chat_mod, "get_agent_profile_context", fake_agent_profile)

    result = await chat_mod.chat_turn(
        user_message="hi",
        history=[],
        llm_config={"provider": "openai", "model": "gpt-4o"},
        dsn="postgresql://unused",
        pool=db_pool,
    )

    assert result["assistant"] == "hello there"
    assert result["history"][-2] == {"role": "user", "content": "hi"}
    assistant_history = result["history"][-1]
    assert assistant_history["role"] == "assistant"
    assert assistant_history["content"] == "hello there"
    UUID(assistant_history["metadata"]["agent_turn_id"])


async def test_chat_turn_tool_loop(monkeypatch, db_pool):
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO config (key, value, description) VALUES ('chat.use_rlm', 'false', 'test override') "
            "ON CONFLICT (key) DO UPDATE SET value = 'false'"
        )

    class DummyMem:
        async def hydrate(self, *_args, **_kwargs):
            return HydratedContext(
                memories=[],
                partial_activations=[],
                identity=[],
                worldview=[],
                emotional_state=None,
                goals=None,
                urgent_drives=[],
            )

        async def touch_memories(self, _ids):
            return None

        async def remember(self, *_args, **_kwargs):
            return uuid4()

        async def record_chat_turn_memory(self, *_args, **_kwargs):
            return {"direct_promoted": False, "raw": {"status": "stored"}}

    mem = DummyMem()

    @asynccontextmanager
    async def fake_connect(_dsn, **_kwargs):
        yield mem

    responses = [
        {
            "content": "",
            "tool_calls": [
                {"id": "tool-1", "name": "recall", "arguments": {"query": "x"}}
            ],
        },
        {"content": "final response", "tool_calls": []},
    ]

    async def fake_chat_completion(**_kwargs):
        return responses.pop(0)

    monkeypatch.setattr(chat_mod.CognitiveMemory, "connect", fake_connect)
    monkeypatch.setattr("core.agent_loop.chat_completion", fake_chat_completion)

    async def fake_agent_profile(_dsn=None, **_kwargs):
        return {}

    monkeypatch.setattr(chat_mod, "get_agent_profile_context", fake_agent_profile)

    result = await chat_mod.chat_turn(
        user_message="hi",
        history=[],
        llm_config={"provider": "openai", "model": "gpt-4o"},
        dsn="postgresql://unused",
        pool=db_pool,
    )

    assert result["assistant"] == "final response"


async def test_chat_turn_hydrates_db_session_history_before_agent(monkeypatch, db_pool):
    session_id = str(uuid4())

    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO config (key, value, description) VALUES ('chat.use_rlm', 'false', 'test override') "
            "ON CONFLICT (key) DO UPDATE SET value = 'false'"
        )
        await conn.fetchval(
            "SELECT append_chat_message($1::uuid, 'user', 'db says alpha')", session_id
        )
        await conn.fetchval(
            "SELECT append_chat_message($1::uuid, 'assistant', 'db says beta')",
            session_id,
        )

    captured: dict[str, object] = {}

    class AgentResult:
        text = "fresh response"

    async def fake_run_agent(*_args, **kwargs):
        captured["history"] = kwargs["history"]
        return AgentResult()

    async def fake_agent_profile(_dsn=None, **_kwargs):
        return {}

    async def fake_remember_conversation(*_args, **_kwargs):
        return {}

    monkeypatch.setattr(chat_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(chat_mod, "get_agent_profile_context", fake_agent_profile)
    monkeypatch.setattr(chat_mod, "_remember_conversation", fake_remember_conversation)

    result = await chat_mod.chat_turn(
        user_message="continue",
        history=[{"role": "user", "content": "stale caller history"}],
        llm_config={"provider": "openai", "model": "gpt-4o"},
        dsn="postgresql://unused",
        pool=db_pool,
        session_id=session_id,
    )

    assert result["assistant"] == "fresh response"
    assert captured["history"] == [
        {"role": "user", "content": "db says alpha"},
        {"role": "assistant", "content": "db says beta"},
    ]


async def test_stream_chat_turn_reports_empty_timeout_without_memory(monkeypatch):
    async def fake_agent_profile(_dsn=None, **_kwargs):
        return {}

    async def fake_stream_agent(*_args, **_kwargs):
        yield AgentEventData(
            event=AgentEvent.LOOP_END,
            data={"stopped_reason": "timeout", "timed_out": True},
        )

    @asynccontextmanager
    async def fail_connect(*_args, **_kwargs):
        raise AssertionError("timeout notices must not be written as memories")
        yield

    monkeypatch.setattr(chat_mod, "get_agent_profile_context", fake_agent_profile)
    monkeypatch.setattr(chat_mod, "stream_agent", fake_stream_agent)
    monkeypatch.setattr(chat_mod.CognitiveMemory, "connect", fail_connect)

    chunks = [
        chunk
        async for chunk in chat_mod.stream_chat_turn(
            user_message="hi",
            history=[],
            llm_config={"provider": "openai", "model": "gpt-4o"},
            dsn="postgresql://unused",
            pool=object(),
        )
    ]

    text = "".join(chunks)
    assert "Request timed out before a response arrived" in text


async def test_stream_chat_turn_skips_memory_for_partial_timeout(monkeypatch):
    async def fake_agent_profile(_dsn=None, **_kwargs):
        return {}

    async def fake_stream_agent(*_args, **_kwargs):
        yield AgentEventData(event=AgentEvent.TEXT_DELTA, data={"text": "partial"})
        yield AgentEventData(
            event=AgentEvent.LOOP_END,
            data={"stopped_reason": "timeout", "timed_out": True},
        )

    @asynccontextmanager
    async def fail_connect(*_args, **_kwargs):
        raise AssertionError("partial timeout replies must not be written as memories")
        yield

    monkeypatch.setattr(chat_mod, "get_agent_profile_context", fake_agent_profile)
    monkeypatch.setattr(chat_mod, "stream_agent", fake_stream_agent)
    monkeypatch.setattr(chat_mod.CognitiveMemory, "connect", fail_connect)

    chunks = [
        chunk
        async for chunk in chat_mod.stream_chat_turn(
            user_message="hi",
            history=[],
            llm_config={"provider": "openai", "model": "gpt-4o"},
            dsn="postgresql://unused",
            pool=object(),
        )
    ]

    assert "".join(chunks) == "partial\n\n[Response timed out before completion.]"


async def test_stream_chat_events_keeps_token_streaming_when_rlm_streaming_disabled(
    monkeypatch,
):
    seen: dict[str, bool] = {}

    async def fake_agent_profile(*_args, **_kwargs):
        return {}

    async def fake_stream_agent(*_args, **_kwargs):
        seen["stream_agent"] = True
        yield AgentEventData(event=AgentEvent.LOOP_START, data={})
        yield AgentEventData(event=AgentEvent.TEXT_DELTA, data={"text": "Hello"})
        yield AgentEventData(event=AgentEvent.TEXT_DELTA, data={"text": " there"})
        yield AgentEventData(
            event=AgentEvent.LOOP_END,
            data={"stopped_reason": "completed", "timed_out": False},
        )

    async def fake_remember(*_args, **_kwargs):
        seen["remembered"] = True
        return {}

    async def fake_energy(*_args, **_kwargs):
        return {}

    monkeypatch.setattr(chat_mod, "get_agent_profile_context", fake_agent_profile)
    monkeypatch.setattr(chat_mod, "stream_agent", fake_stream_agent)
    monkeypatch.setattr(chat_mod, "_remember_conversation", fake_remember)
    monkeypatch.setattr(chat_mod, "_apply_chat_energy_effects", fake_energy)

    events = [
        event
        async for event in chat_mod.stream_chat_events(
            user_message="hi",
            history=[],
            llm_config={"provider": "openai", "model": "gpt-4o"},
            dsn="postgresql://unused",
            pool=_ConfigPool(
                {
                    "chat.use_rlm": True,
                    "rlm.chat.streaming_enabled": False,
                }
            ),
        )
    ]

    assert seen["stream_agent"] is True
    assert seen["remembered"] is True
    assert [
        event.data.get("text")
        for event in events
        if event.event == AgentEvent.TEXT_DELTA
    ] == ["Hello", " there"]
    assert not any(event.data.get("runtime") == "rlm" for event in events)


async def test_stream_chat_events_persists_successful_tool_receipt(monkeypatch):
    turn_id = str(uuid4())
    memory_id = str(uuid4())
    captured: dict[str, object] = {}

    async def fake_agent_profile(*_args, **_kwargs):
        return {}

    async def fake_stream_agent(*_args, **_kwargs):
        yield AgentEventData(
            event=AgentEvent.LOOP_START,
            data={"turn_id": turn_id},
        )
        yield AgentEventData(
            event=AgentEvent.TOOL_RESULT,
            data={
                "turn_id": turn_id,
                "call_id": "remember-1",
                "tool_name": "remember",
                "arguments": {
                    "content": "preserve the lighthouse agreement",
                    "type": "semantic",
                },
                "success": True,
                "output": {"memory_id": memory_id, "type": "semantic", "reused": False},
                "energy_spent": 1,
            },
        )
        yield AgentEventData(
            event=AgentEvent.TEXT_DELTA,
            data={"turn_id": turn_id, "text": "Stored it."},
        )
        yield AgentEventData(
            event=AgentEvent.LOOP_END,
            data={"turn_id": turn_id, "stopped_reason": "completed"},
        )

    async def fake_remember(*_args, **kwargs):
        captured.update(kwargs)
        return {}

    async def fake_energy(*_args, **_kwargs):
        return {}

    monkeypatch.setattr(chat_mod, "get_agent_profile_context", fake_agent_profile)
    monkeypatch.setattr(chat_mod, "stream_agent", fake_stream_agent)
    monkeypatch.setattr(chat_mod, "_remember_conversation", fake_remember)
    monkeypatch.setattr(chat_mod, "_apply_chat_energy_effects", fake_energy)

    events = [
        event
        async for event in chat_mod.stream_chat_events(
            user_message="Please remember this",
            history=[],
            llm_config={"provider": "openai", "model": "gpt-4o"},
            dsn="postgresql://unused",
            pool=_ConfigPool({"chat.use_rlm": False}),
            session_id=str(uuid4()),
        )
    ]

    assert any(event.event == AgentEvent.TOOL_RESULT for event in events)
    assert captured["agent_turn_id"] == turn_id
    receipts = captured["action_receipts"]
    assert receipts[0]["name"] == "remember"
    assert receipts[0]["result"]["memory_id"] == memory_id
