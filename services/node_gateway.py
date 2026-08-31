"""WebSocket gateway and durable request bridge for signed companion nodes."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import uuid
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from core.node_actions import MAX_NODE_CAPABILITIES, NODE_CAPABILITIES
from core.node_identity import node_id_for_public_key, verify_signature

logger = logging.getLogger(__name__)
_WAKE_SIGNED_FIELDS = (
    "request_id",
    "session_id",
    "mime_type",
    "audio_bytes",
    "audio_sha256",
    "audio_base64",
    "detector_model",
    "detector_label",
    "detector_score",
)


def _object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _wake_signed_payload(message: dict[str, Any]) -> dict[str, Any]:
    return {field: message.get(field) for field in _WAKE_SIGNED_FIELDS}


async def _record_wake_event(
    pool: Any,
    *,
    request_id: str,
    node_id: str,
    session_id: str | None,
    detector_model: str,
    detector_score: float | None,
    audio_bytes: int,
    transcript_chars: int | None,
    response_chars: int | None,
    response_audio_bytes: int | None,
    outcome: str,
    error_detail: str | None,
    metadata: dict[str, Any] | None = None,
) -> None:
    try:
        async with pool.acquire() as conn:
            await conn.fetchval(
                """
                SELECT record_voice_wake_event(
                    $1::uuid, $2, $3::uuid, $4, $5, $6, $7, $8, $9,
                    $10, $11, $12::jsonb
                )
                """,
                request_id,
                node_id,
                session_id,
                detector_model,
                detector_score,
                audio_bytes,
                transcript_chars,
                response_chars,
                response_audio_bytes,
                outcome,
                str(error_detail or "")[:500] or None,
                json.dumps(metadata or {}),
            )
    except Exception:
        logger.warning("Wake event audit write failed", exc_info=True)


async def process_wake_utterance(
    pool: Any,
    *,
    node_id: str,
    node_name: str,
    message: dict[str, Any],
) -> dict[str, Any]:
    """Run one verified node utterance through STT, canonical chat, and TTS."""
    raw_request_id = str(message.get("request_id") or "")
    try:
        request_id = str(uuid.UUID(raw_request_id))
        session_id = str(uuid.UUID(str(message.get("session_id") or "")))
    except (ValueError, TypeError, AttributeError):
        return {
            "type": "wake_response",
            "request_id": raw_request_id,
            "status": "failed",
            "error": "Wake request or session identity was invalid; wake Hexis again.",
        }

    detector_model = str(message.get("detector_model") or "")[:200]
    detector_label = str(message.get("detector_label") or "")[:200]
    try:
        detector_score = float(message.get("detector_score"))
    except (TypeError, ValueError):
        detector_score = None
    if detector_score is not None and not 0 <= detector_score <= 1:
        detector_score = None

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT COALESCE(get_config_bool('voice.wake.enabled'), false) AS enabled,
                   COALESCE(get_config_int('voice.wake.max_audio_bytes'), 4194304) AS max_audio_bytes,
                   COALESCE(get_config_int('voice.wake.max_response_audio_bytes'), 8388608) AS max_response_audio_bytes,
                   EXISTS(
                       SELECT 1 FROM voice_wake_events WHERE request_id=$1::uuid
                   ) AS already_processed
            """,
            request_id,
        )
    enabled = bool(row and row["enabled"])
    max_audio_bytes = min(max(int(row["max_audio_bytes"] or 4_194_304), 1), 8_388_608)
    max_response_audio_bytes = min(
        max(int(row["max_response_audio_bytes"] or 8_388_608), 1), 8_388_608
    )
    if row and row["already_processed"]:
        return {
            "type": "wake_response",
            "request_id": request_id,
            "status": "failed",
            "error": (
                "This signed wake request was already processed. Its conversation "
                "record was preserved; use the wake word again for a new turn."
            ),
        }

    metrics: dict[str, int | None] = {
        "audio_bytes": 0,
        "transcript_chars": None,
        "response_chars": None,
        "response_audio_bytes": None,
    }

    async def failed(outcome: str, detail: str) -> dict[str, Any]:
        await _record_wake_event(
            pool,
            request_id=request_id,
            node_id=node_id,
            session_id=session_id,
            detector_model=detector_model,
            detector_score=detector_score,
            audio_bytes=int(metrics["audio_bytes"] or 0),
            transcript_chars=metrics["transcript_chars"],
            response_chars=metrics["response_chars"],
            response_audio_bytes=metrics["response_audio_bytes"],
            outcome=outcome,
            error_detail=detail,
            metadata={"detector_label": detector_label},
        )
        return {
            "type": "wake_response",
            "request_id": request_id,
            "session_id": session_id,
            "status": "failed",
            "error": detail,
        }

    if not enabled:
        return await failed(
            "failed_disabled",
            "Wake-word turns are off. Open Settings → Voice, enable paired-node "
            "wake word, save, and use the wake word again.",
        )
    if message.get("mime_type") != "audio/wav":
        return await failed(
            "failed_invalid_audio", "Wake nodes must send 16 kHz mono WAV audio."
        )
    encoded = str(message.get("audio_base64") or "")
    if len(encoded) > ((max_audio_bytes + 2) // 3) * 4 + 4:
        return await failed(
            "failed_invalid_audio",
            f"Wake audio exceeded the configured {max_audio_bytes}-byte limit.",
        )
    try:
        audio = base64.b64decode(encoded, validate=True)
        declared_bytes = int(message.get("audio_bytes"))
    except (ValueError, TypeError):
        return await failed(
            "failed_invalid_audio", "Wake audio encoding or byte count was invalid."
        )
    metrics["audio_bytes"] = len(audio)
    if (
        not audio
        or len(audio) > max_audio_bytes
        or declared_bytes != len(audio)
        or not secrets.compare_digest(
            str(message.get("audio_sha256") or ""), hashlib.sha256(audio).hexdigest()
        )
    ):
        return await failed(
            "failed_invalid_audio",
            "Wake audio failed its size or SHA-256 integrity check.",
        )

    from services.voice_notes import transcribe_uploaded_voice

    transcription = await transcribe_uploaded_voice(
        pool,
        audio,
        filename=f"wake-{request_id}.wav",
        mime_type="audio/wav",
        channel_id=node_id,
        sender_id=node_id,
        channel_type="node_wake",
    )
    if not transcription.ok:
        return await failed(
            "failed_transcription",
            transcription.error_detail
            or "Wake transcription failed. Review Settings → Voice and retry.",
        )
    transcript = transcription.transcript.strip()
    metrics["transcript_chars"] = len(transcript)

    from core.agent_loop import AgentEvent
    from services.chat import stream_chat_events
    from services.speech import load_tts_config, synthesize_text

    async with pool.acquire() as conn:
        tts_cfg = await load_tts_config(conn)
    spoken_limit = min(max(tts_cfg.max_chars, 1), 1600)
    text_parts: list[str] = []
    agent_error: str | None = None
    try:
        async for event in stream_chat_events(
            user_message=transcript,
            history=[],
            session_id=session_id,
            pool=pool,
            user_label=f"Wake node: {node_name}",
            trusted_operator=False,
            operator_context={
                "channel_type": "node_wake",
                "channel_id": node_id,
                "sender_id": node_id,
                "disposition": "engage",
                "reason": "signed_paired_wake_turn",
            },
            surface="node_wake",
            prompt_addenda=[
                "This is a hands-free spoken turn. Answer in clear prose, omit URLs "
                f"unless requested, and stay under {spoken_limit} characters so the "
                "complete answer can be spoken."
            ],
            gateway_source_id=f"node:wake:{node_id}:{request_id}",
            gateway_payload={
                "request_id": request_id,
                "node_id": node_id,
                "transcript_chars": len(transcript),
            },
        ):
            if event.event == AgentEvent.TEXT_DELTA:
                text_parts.append(str(event.data.get("text") or ""))
            elif event.event == AgentEvent.ERROR:
                agent_error = str(event.data.get("error") or "agent response failed")
    except Exception as exc:
        logger.exception("Wake agent turn failed")
        agent_error = str(exc)
    assistant = "".join(text_parts).strip()
    metrics["response_chars"] = len(assistant)
    if not assistant:
        return await failed(
            "failed_agent",
            f"Hexis could not complete the wake response ({agent_error or 'no response text'}). "
            "The transcript remains in the conversation; use the wake word to retry.",
        )

    synthesis = await synthesize_text(
        pool,
        assistant,
        source=f"node_wake:{node_id[:12]}",
        cfg=tts_cfg,
    )
    if not synthesis.ok:
        result = await failed(
            "failed_synthesis",
            synthesis.error_detail
            or "The written wake response was saved, but speech synthesis failed.",
        )
        result["assistant"] = assistant
        result["transcript"] = transcript
        return result
    if len(synthesis.audio) > max_response_audio_bytes:
        result = await failed(
            "failed_synthesis",
            f"The spoken response exceeded the configured {max_response_audio_bytes}-byte node limit. "
            "The written response was preserved.",
        )
        result["assistant"] = assistant
        result["transcript"] = transcript
        return result
    metrics["response_audio_bytes"] = len(synthesis.audio)
    await _record_wake_event(
        pool,
        request_id=request_id,
        node_id=node_id,
        session_id=session_id,
        detector_model=detector_model,
        detector_score=detector_score,
        audio_bytes=len(audio),
        transcript_chars=len(transcript),
        response_chars=len(assistant),
        response_audio_bytes=len(synthesis.audio),
        outcome="completed",
        error_detail=None,
        metadata={"detector_label": detector_label},
    )
    return {
        "type": "wake_response",
        "request_id": request_id,
        "session_id": session_id,
        "status": "succeeded",
        "transcript": transcript,
        "assistant": assistant,
        "mime_type": synthesis.mime_type,
        "audio_base64": base64.b64encode(synthesis.audio).decode("ascii"),
    }


async def _mark_connection(
    pool: Any,
    node_id: str,
    public_key: str,
    connection_id: str,
    online: bool,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    async with pool.acquire() as conn:
        return _object(
            await conn.fetchval(
                "SELECT mark_node_connection($1, $2, $3::uuid, $4, $5::jsonb)",
                node_id,
                public_key,
                connection_id,
                online,
                json.dumps(metadata or {}),
            )
        )


async def _await_pairing(
    websocket: WebSocket,
    pool: Any,
    decision: dict[str, Any],
) -> bool:
    request_id = str(decision.get("request_id") or "")
    await websocket.send_json(
        {
            "type": "status",
            "status": "pairing_required",
            "request_id": request_id,
            "code": decision.get("code"),
            "expires_at": decision.get("expires_at"),
            "next_step": decision.get("next_step"),
        }
    )
    while True:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE node_pairing_requests
                SET status='expired', decided_at=CURRENT_TIMESTAMP,
                    decided_by='system',
                    decision_note='Pairing request expired before a decision.'
                WHERE id=$1::uuid AND status='pending'
                  AND expires_at <= CURRENT_TIMESTAMP
                """,
                request_id,
            )
            row = await conn.fetchrow(
                "SELECT status, expires_at FROM node_pairing_requests WHERE id=$1::uuid",
                request_id,
            )
        if row is None:
            await websocket.send_json(
                {
                    "type": "status",
                    "status": "not_found",
                    "reason": "Pairing request disappeared.",
                }
            )
            return False
        status = str(row["status"])
        if status == "approved":
            return True
        if status in {"denied", "expired"}:
            await websocket.send_json(
                {
                    "type": "status",
                    "status": status,
                    "reason": f"Node pairing was {status}.",
                }
            )
            return False
        try:
            # The node does not send application messages before pairing. A
            # bounded receive keeps disconnects observable while SQL remains
            # the source of truth for the operator decision.
            await asyncio.wait_for(websocket.receive_text(), timeout=1)
        except asyncio.TimeoutError:
            pass


async def handle_node_websocket(websocket: WebSocket, pool: Any) -> None:
    """Authenticate a node, wait in-place for approval, then dispatch work."""
    await websocket.accept()
    node_id = ""
    public_key = ""
    connection_id = str(uuid.uuid4())
    send_lock = asyncio.Lock()
    wake_task: asyncio.Task[Any] | None = None

    async def send(payload: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    try:
        challenge = secrets.token_urlsafe(32)
        await send({"type": "challenge", "challenge": challenge})
        hello = await asyncio.wait_for(websocket.receive_json(), timeout=15)
        if not isinstance(hello, dict) or hello.get("type") != "hello":
            await send(
                {
                    "type": "status",
                    "status": "invalid_handshake",
                    "reason": "Expected a node hello.",
                }
            )
            await websocket.close(code=4400)
            return
        node_id = str(hello.get("node_id") or "")
        public_key = str(hello.get("public_key") or "")
        signature = str(hello.get("signature") or "")
        try:
            fingerprint_matches = node_id_for_public_key(public_key) == node_id
        except (TypeError, ValueError):
            fingerprint_matches = False
        proof = {"challenge": challenge, "node_id": node_id}
        if not fingerprint_matches or not verify_signature(
            public_key, proof, signature
        ):
            await send(
                {
                    "type": "status",
                    "status": "invalid_signature",
                    "reason": "The node identity signature or fingerprint was invalid.",
                }
            )
            await websocket.close(code=4401)
            return
        name = str(hello.get("name") or "Unnamed node").strip()
        capabilities = hello.get("capabilities")
        if (
            not isinstance(capabilities, list)
            or not all(item in NODE_CAPABILITIES for item in capabilities)
            or len(capabilities) > MAX_NODE_CAPABILITIES
            or len(capabilities) != len(set(capabilities))
        ):
            await send(
                {
                    "type": "status",
                    "status": "invalid_capabilities",
                    "reason": "Node capabilities were invalid.",
                }
            )
            await websocket.close(code=4400)
            return
        metadata = (
            hello.get("metadata") if isinstance(hello.get("metadata"), dict) else {}
        )
        if not name or len(name) > 100 or len(json.dumps(metadata)) > 16_384:
            await send(
                {
                    "type": "status",
                    "status": "invalid_metadata",
                    "reason": "Node name or metadata exceeded the safe handshake limit.",
                }
            )
            await websocket.close(code=4400)
            return
        async with pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT register_node_handshake($1, $2, $3, $4::jsonb, $5::jsonb)",
                node_id,
                public_key,
                name,
                json.dumps(capabilities),
                json.dumps(metadata),
            )
        decision = _object(raw)
        if not decision.get("approved"):
            if decision.get("status") != "pairing_required" or not await _await_pairing(
                websocket, pool, decision
            ):
                await websocket.close(code=4403)
                return
            async with pool.acquire() as conn:
                raw = await conn.fetchval(
                    "SELECT register_node_handshake($1, $2, $3, $4::jsonb, $5::jsonb)",
                    node_id,
                    public_key,
                    name,
                    json.dumps(capabilities),
                    json.dumps(metadata),
                )
            decision = _object(raw)
            if not decision.get("approved"):
                await send(
                    {
                        "type": "status",
                        "status": decision.get("status") or "not_approved",
                        "reason": decision.get("reason") or "Pairing was not accepted.",
                    }
                )
                await websocket.close(code=4403)
                return

        acquired = await _mark_connection(
            pool, node_id, public_key, connection_id, True, metadata
        )
        if not acquired.get("updated"):
            await send(
                {
                    "type": "status",
                    "status": "already_connected",
                    "reason": (
                        "This signed node identity already has a live connection. "
                        "Stop the other node process or wait 30 seconds, then retry."
                    ),
                }
            )
            await websocket.close(code=4409)
            return
        await send({"type": "status", "status": "paired", "node_id": node_id})
        inflight: str | None = None
        while True:
            if wake_task is not None and wake_task.done():
                try:
                    wake_task.result()
                except Exception:
                    logger.warning("Wake response task failed", exc_info=True)
                wake_task = None
            if inflight is None:
                async with pool.acquire() as conn:
                    claimed = _object(
                        await conn.fetchval("SELECT claim_node_invocation($1)", node_id)
                    )
                if claimed.get("claimed"):
                    inflight = str(claimed.get("invocation_id") or "")
                    await send(
                        {
                            "type": "invoke",
                            "invocation_id": inflight,
                            "action": claimed.get("action"),
                            "arguments": claimed.get("arguments") or {},
                            "expires_at": claimed.get("expires_at"),
                        }
                    )
            try:
                message = await asyncio.wait_for(websocket.receive_json(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if not isinstance(message, dict):
                continue
            if message.get("type") == "heartbeat":
                refreshed = await _mark_connection(
                    pool, node_id, public_key, connection_id, True
                )
                if not refreshed.get("updated"):
                    await websocket.close(
                        code=4409, reason="Node connection no longer owns this identity"
                    )
                    return
                continue
            if message.get("type") == "wake_utterance":
                request_id = str(message.get("request_id") or "")
                if "audio.wake" not in capabilities:
                    await websocket.close(
                        code=4400,
                        reason="Node sent wake audio without advertising audio.wake",
                    )
                    return
                signed = _wake_signed_payload(message)
                if not verify_signature(
                    public_key, signed, str(message.get("signature") or "")
                ):
                    await websocket.close(code=4401, reason="Invalid wake signature")
                    return
                if wake_task is not None:
                    await send(
                        {
                            "type": "wake_response",
                            "request_id": request_id,
                            "status": "failed",
                            "error": (
                                "This node already has a wake turn in progress. Wait "
                                "for its response, then use the wake word again."
                            ),
                        }
                    )
                    continue

                async def answer_wake(
                    wake_message: dict[str, Any] = dict(message),
                ) -> None:
                    try:
                        response = await process_wake_utterance(
                            pool,
                            node_id=node_id,
                            node_name=name,
                            message=wake_message,
                        )
                    except Exception as exc:
                        logger.exception("Wake request failed before a response")
                        response = {
                            "type": "wake_response",
                            "request_id": str(wake_message.get("request_id") or ""),
                            "status": "failed",
                            "error": (
                                f"Hexis could not process the wake turn ({exc}). Run "
                                "`hexis doctor`, then use the wake word again."
                            ),
                        }
                    try:
                        await send(response)
                    except Exception:
                        logger.warning(
                            "Wake response completed after the node disconnected",
                            exc_info=True,
                        )

                wake_task = asyncio.create_task(answer_wake())
                continue
            if message.get("type") != "result":
                continue
            invocation_id = str(message.get("invocation_id") or "")
            if not inflight or invocation_id != inflight:
                await websocket.close(code=4400, reason="Unexpected invocation result")
                return
            signed = {
                "invocation_id": invocation_id,
                "success": bool(message.get("success")),
                "result": message.get("result"),
                "error": message.get("error"),
            }
            result_signature = str(message.get("signature") or "")
            if not verify_signature(public_key, signed, result_signature):
                await websocket.close(code=4401, reason="Invalid result signature")
                return
            result = message.get("result")
            if result is not None and not isinstance(result, dict):
                result = {"value": result}
            async with pool.acquire() as conn:
                await conn.fetchval(
                    "SELECT complete_node_invocation($1::uuid, $2, $3, $4::jsonb, $5, $6)",
                    invocation_id,
                    node_id,
                    bool(message.get("success")),
                    json.dumps(result) if result is not None else None,
                    str(message.get("error") or "") or None,
                    result_signature,
                )
            inflight = None
            await _mark_connection(pool, node_id, public_key, connection_id, True)
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        raise
    finally:
        if node_id and public_key:
            try:
                await _mark_connection(pool, node_id, public_key, connection_id, False)
            except Exception:
                pass


async def request_node_invocation(
    pool: Any,
    *,
    node_id: str,
    action: str,
    arguments: dict[str, Any],
    requested_by: str,
    timeout_seconds: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Queue one invocation and wait for its signed terminal record."""
    async with pool.acquire() as conn:
        created = _object(
            await conn.fetchval(
                "SELECT create_node_invocation($1, $2, $3::jsonb, $4, $5, $6::jsonb)",
                node_id,
                action,
                json.dumps(arguments),
                requested_by,
                timeout_seconds,
                json.dumps(metadata or {}),
            )
        )
    if not created.get("queued"):
        return created
    invocation_id = str(created["invocation_id"])
    wait_seconds = int(created.get("timeout_seconds") or 120) + 2
    loop = asyncio.get_running_loop()
    deadline = loop.time() + wait_seconds
    while loop.time() < deadline:
        async with pool.acquire() as conn:
            current = _object(
                await conn.fetchval(
                    "SELECT get_node_invocation($1::uuid)", invocation_id
                )
            )
        if current.get("status") in {
            "succeeded",
            "failed",
            "expired",
            "cancelled",
        }:
            return current
        await asyncio.sleep(0.25)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE node_invocations
            SET status='expired', completed_at=CURRENT_TIMESTAMP,
                error='Node invocation timed out before a signed result arrived.'
            WHERE id=$1::uuid AND status IN ('queued', 'dispatched')
            """,
            invocation_id,
        )
        current = _object(
            await conn.fetchval("SELECT get_node_invocation($1::uuid)", invocation_id)
        )
    return current
