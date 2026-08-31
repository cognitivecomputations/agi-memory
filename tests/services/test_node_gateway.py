from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import uuid

import pytest

from core.node_identity import initialize_node_identity
from services.node_gateway import handle_node_websocket, request_node_invocation

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


class FakeWebSocket:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[dict] = asyncio.Queue()
        self.outgoing: asyncio.Queue[dict] = asyncio.Queue()
        self.accepted = False
        self.closed: tuple[int, str | None] | None = None

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, value: dict) -> None:
        await self.outgoing.put(value)

    async def receive_json(self) -> dict:
        return await self.incoming.get()

    async def receive_text(self) -> str:
        await asyncio.Future()
        raise AssertionError("unreachable")

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.closed = (code, reason)


async def _wait_for_status(db_pool, node_id: str, status: str) -> None:
    for _ in range(100):
        async with db_pool.acquire() as conn:
            current = await conn.fetchval(
                "SELECT status FROM hexis_nodes WHERE node_id=$1", node_id
            )
        if current == status:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"node {node_id} never reached {status}")


async def test_signed_gateway_pairing_and_invocation_round_trip(
    db_pool, tmp_path
) -> None:
    identity = initialize_node_identity(name="Round trip", path=tmp_path / "node.json")
    websocket = FakeWebSocket()
    gateway = asyncio.create_task(handle_node_websocket(websocket, db_pool))
    try:
        challenge = await asyncio.wait_for(websocket.outgoing.get(), timeout=2)
        assert challenge["type"] == "challenge"
        proof = {"challenge": challenge["challenge"], "node_id": identity.node_id}
        await websocket.incoming.put(
            {
                "type": "hello",
                "node_id": identity.node_id,
                "name": identity.name,
                "public_key": identity.public_key,
                "signature": identity.sign(proof),
                "capabilities": ["system.run"],
                "metadata": {"platform": "test", "command_aliases": ["hello"]},
            }
        )
        pairing = await asyncio.wait_for(websocket.outgoing.get(), timeout=2)
        assert pairing["status"] == "pairing_required"

        async with db_pool.acquire() as conn:
            decided = await conn.fetchval(
                "SELECT decide_node_pairing($1,'approve','test',NULL)",
                pairing["request_id"],
            )
        decided = json.loads(decided) if isinstance(decided, str) else decided
        assert decided["status"] == "approved"

        paired = await asyncio.wait_for(websocket.outgoing.get(), timeout=3)
        assert paired == {
            "type": "status",
            "status": "paired",
            "node_id": identity.node_id,
        }
        await _wait_for_status(db_pool, identity.node_id, "online")

        waiting = asyncio.create_task(
            request_node_invocation(
                db_pool,
                node_id=identity.node_id,
                action="system.run",
                arguments={"command": "hello", "args": [], "timeout": 10},
                requested_by="test",
                timeout_seconds=10,
            )
        )
        invocation = await asyncio.wait_for(websocket.outgoing.get(), timeout=3)
        assert invocation["type"] == "invoke"
        assert invocation["arguments"]["command"] == "hello"
        signed = {
            "invocation_id": invocation["invocation_id"],
            "success": True,
            "result": {"returncode": 0, "stdout": "hello\n", "stderr": ""},
            "error": None,
        }
        await websocket.incoming.put(
            {"type": "result", **signed, "signature": identity.sign(signed)}
        )
        terminal = await asyncio.wait_for(waiting, timeout=3)
        assert terminal["status"] == "succeeded"
        assert terminal["result"]["stdout"] == "hello\n"
    finally:
        gateway.cancel()
        await asyncio.gather(gateway, return_exceptions=True)
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM node_invocations WHERE node_id=$1", identity.node_id
            )
            await conn.execute(
                "DELETE FROM hexis_nodes WHERE node_id=$1", identity.node_id
            )
            await conn.execute(
                "DELETE FROM node_pairing_requests WHERE node_id=$1", identity.node_id
            )


async def test_gateway_rejects_invalid_identity_before_pairing(db_pool) -> None:
    websocket = FakeWebSocket()
    gateway = asyncio.create_task(handle_node_websocket(websocket, db_pool))
    challenge = await asyncio.wait_for(websocket.outgoing.get(), timeout=2)
    await websocket.incoming.put(
        {
            "type": "hello",
            "node_id": "0" * 64,
            "name": "Forgery",
            "public_key": "not-base64",
            "signature": "not-a-signature",
            "capabilities": ["system.run"],
        }
    )
    status = await asyncio.wait_for(websocket.outgoing.get(), timeout=2)
    await asyncio.wait_for(gateway, timeout=2)
    assert challenge["type"] == "challenge"
    assert status["status"] == "invalid_signature"
    assert websocket.closed == (4401, None)
    async with db_pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM node_pairing_requests WHERE node_id=$1", "0" * 64
            )
            == 0
        )


async def test_gateway_accepts_expanded_wave_c_capability_set(
    db_pool, tmp_path
) -> None:
    identity = initialize_node_identity(
        name="Wave C Mac", path=tmp_path / "wave-c-node.json"
    )
    websocket = FakeWebSocket()
    gateway = asyncio.create_task(handle_node_websocket(websocket, db_pool))
    try:
        challenge = await asyncio.wait_for(websocket.outgoing.get(), timeout=2)
        proof = {"challenge": challenge["challenge"], "node_id": identity.node_id}
        capabilities = [
            "apple.reminders.list",
            "apple.notes.search",
            "apple.calendar.list",
            "apple.shortcuts.list",
        ]
        await websocket.incoming.put(
            {
                "type": "hello",
                "node_id": identity.node_id,
                "name": identity.name,
                "public_key": identity.public_key,
                "signature": identity.sign(proof),
                "capabilities": capabilities,
            }
        )
        status = await asyncio.wait_for(websocket.outgoing.get(), timeout=2)
        assert status["status"] == "pairing_required"
        async with db_pool.acquire() as conn:
            stored = await conn.fetchval(
                "SELECT capabilities FROM node_pairing_requests WHERE id=$1::uuid",
                status["request_id"],
            )
        stored = json.loads(stored) if isinstance(stored, str) else stored
        assert stored == capabilities
    finally:
        gateway.cancel()
        await asyncio.gather(gateway, return_exceptions=True)
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM node_pairing_requests WHERE node_id=$1", identity.node_id
            )


async def test_gateway_accepts_only_signed_wake_audio_from_advertised_node(
    db_pool, tmp_path, monkeypatch
) -> None:
    identity = initialize_node_identity(
        name="Wake round trip", path=tmp_path / "wake-node.json"
    )
    websocket = FakeWebSocket()
    processed: list[dict] = []

    async def process(_pool, *, node_id, node_name, message):
        processed.append(
            {"node_id": node_id, "node_name": node_name, "message": message}
        )
        return {
            "type": "wake_response",
            "request_id": message["request_id"],
            "status": "succeeded",
            "assistant": "Hello from Hexis.",
            "audio_base64": base64.b64encode(b"RIFFreply").decode("ascii"),
        }

    monkeypatch.setattr("services.node_gateway.process_wake_utterance", process)
    gateway = asyncio.create_task(handle_node_websocket(websocket, db_pool))
    try:
        challenge = await asyncio.wait_for(websocket.outgoing.get(), timeout=2)
        proof = {"challenge": challenge["challenge"], "node_id": identity.node_id}
        await websocket.incoming.put(
            {
                "type": "hello",
                "node_id": identity.node_id,
                "name": identity.name,
                "public_key": identity.public_key,
                "signature": identity.sign(proof),
                "capabilities": ["audio.wake"],
                "metadata": {"platform": "test"},
            }
        )
        pairing = await asyncio.wait_for(websocket.outgoing.get(), timeout=2)
        async with db_pool.acquire() as conn:
            await conn.fetchval(
                "SELECT decide_node_pairing($1,'approve','test',NULL)",
                pairing["request_id"],
            )
        paired = await asyncio.wait_for(websocket.outgoing.get(), timeout=3)
        assert paired["status"] == "paired"

        audio = b"RIFFutterance"
        signed = {
            "request_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
            "mime_type": "audio/wav",
            "audio_bytes": len(audio),
            "audio_sha256": hashlib.sha256(audio).hexdigest(),
            "audio_base64": base64.b64encode(audio).decode("ascii"),
            "detector_model": "custom",
            "detector_label": "wake",
            "detector_score": 0.8,
        }
        await websocket.incoming.put(
            {
                "type": "wake_utterance",
                **signed,
                "signature": identity.sign(signed),
            }
        )
        response = await asyncio.wait_for(websocket.outgoing.get(), timeout=3)

        assert response["type"] == "wake_response"
        assert response["status"] == "succeeded"
        assert processed[0]["node_id"] == identity.node_id
        assert processed[0]["message"]["audio_sha256"] == signed["audio_sha256"]
    finally:
        gateway.cancel()
        await asyncio.gather(gateway, return_exceptions=True)
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM node_invocations WHERE node_id=$1", identity.node_id
            )
            await conn.execute(
                "DELETE FROM voice_wake_events WHERE node_id=$1", identity.node_id
            )
            await conn.execute(
                "DELETE FROM hexis_nodes WHERE node_id=$1", identity.node_id
            )
            await conn.execute(
                "DELETE FROM node_pairing_requests WHERE node_id=$1", identity.node_id
            )
