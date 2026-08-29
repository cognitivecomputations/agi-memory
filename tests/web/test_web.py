"""
Tests for the Hexis API server (apps/hexis_api.py).

Uses httpx.AsyncClient with the FastAPI app directly (no server needed).
"""

from __future__ import annotations

import json
import hashlib
import hmac
import time
import urllib.parse
import uuid
from unittest.mock import AsyncMock, patch

import pytest

import apps.hexis_api as web_module
from apps.hexis_api import app
from core.agent_loop import AgentEvent, AgentEventData
from tests.utils import get_test_identifier
from services.voice_notes import TranscriptionResult

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


async def test_slack_signature_is_exact_and_fresh() -> None:
    body = b"payload=%7B%22type%22%3A%22block_actions%22%7D"
    timestamp = "1700000000"
    secret = "test-signing-secret"
    expected = (
        "v0="
        + hmac.new(
            secret.encode(), b"v0:" + timestamp.encode() + b":" + body, hashlib.sha256
        ).hexdigest()
    )

    with patch("apps.hexis_api.time.time", return_value=1700000010):
        assert (
            web_module._verify_slack_signature(body, timestamp, expected, secret)
            is True
        )
        assert (
            web_module._verify_slack_signature(body + b"x", timestamp, expected, secret)
            is False
        )
    with patch("apps.hexis_api.time.time", return_value=1700000400):
        assert (
            web_module._verify_slack_signature(body, timestamp, expected, secret)
            is False
        )


async def test_signed_slack_action_records_operator_decision(
    client, db_pool, monkeypatch
) -> None:
    request_id = str(uuid.uuid4())
    secret = "test-signing-secret"
    secret_env = "HEXIS_TEST_SLACK_SIGNING_SECRET"
    keys = [
        "operator.approval.enabled",
        "channel.slack.operator_user_id",
        "channel.slack.signing_secret",
    ]
    async with db_pool.acquire() as conn:
        original_rows = await conn.fetch(
            "SELECT key, value FROM config WHERE key = ANY($1::text[])", keys
        )
        await conn.execute(
            "SELECT set_config('operator.approval.enabled', 'true'::jsonb)"
        )
        await conn.execute(
            "SELECT set_config('channel.slack.operator_user_id', $1::jsonb)",
            json.dumps("U-OPERATOR"),
        )
        await conn.execute(
            "SELECT set_config('channel.slack.signing_secret', $1::jsonb)",
            json.dumps(secret_env),
        )
        await conn.execute(
            """
            INSERT INTO operator_tool_approval_requests (
                id, tool_name, arguments_hash, tool_context, status, expires_at
            ) VALUES (
                $1::uuid, 'web_action_test', repeat('0', 64), 'chat',
                'slack_delivered', CURRENT_TIMESTAMP + INTERVAL '5 minutes'
            )
            """,
            request_id,
        )
    monkeypatch.setenv(secret_env, secret)

    payload = {
        "type": "block_actions",
        "user": {"id": "U-OPERATOR"},
        "actions": [
            {
                "action_id": "operator_approval_approve",
                "value": json.dumps(
                    {
                        "approval_request_id": request_id,
                        "decision": "approve",
                    }
                ),
            }
        ],
    }
    body = urllib.parse.urlencode({"payload": json.dumps(payload)}).encode()
    timestamp = str(int(time.time()))
    signature = (
        "v0="
        + hmac.new(
            secret.encode(), b"v0:" + timestamp.encode() + b":" + body, hashlib.sha256
        ).hexdigest()
    )

    try:
        response = await client.post(
            "/api/slack/interactivity",
            content=body,
            headers={
                "content-type": "application/x-www-form-urlencoded",
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
        )
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT status, decision_actor, decision_channel
                FROM operator_tool_approval_requests WHERE id = $1::uuid
                """,
                request_id,
            )
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM operator_tool_approval_requests WHERE id = $1::uuid",
                request_id,
            )
            await conn.execute("DELETE FROM config WHERE key = ANY($1::text[])", keys)
            for original in original_rows:
                value = original["value"]
                await conn.execute(
                    "INSERT INTO config (key, value) VALUES ($1, $2::jsonb)",
                    original["key"],
                    value if isinstance(value, str) else json.dumps(value),
                )

    assert response.status_code == 200
    assert row["status"] == "approved"
    assert row["decision_actor"] == "U-OPERATOR"
    assert row["decision_channel"] == "slack"


@pytest.fixture(scope="module")
async def client(db_pool):
    """Create an httpx async test client with the DB pool injected."""
    import httpx

    # Inject the real pool into the web module
    original_pool = web_module._pool
    web_module._pool = db_pool
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client
    web_module._pool = original_pool


async def test_health(client):
    """Health endpoint returns 200 with status ok."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


async def test_node_pairing_api_approves_exact_pending_identity(
    client, db_pool, tmp_path
):
    from core.node_identity import initialize_node_identity

    identity = initialize_node_identity(name="API node", path=tmp_path / "node.json")
    async with db_pool.acquire() as conn:
        raw = await conn.fetchval(
            "SELECT register_node_handshake($1,$2,$3,'[\"screen.capture\"]'::jsonb,'{}'::jsonb)",
            identity.node_id,
            identity.public_key,
            identity.name,
        )
    pending = json.loads(raw) if isinstance(raw, str) else raw
    try:
        before = await client.get("/api/nodes")
        assert before.status_code == 200
        assert any(
            item["id"] == pending["request_id"]
            for item in before.json()["pending_pairings"]
        )

        response = await client.post(
            "/api/nodes/pairing",
            json={"request": pending["code"], "decision": "approve"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "approved"

        after = await client.get("/api/nodes")
        node = next(
            item
            for item in after.json()["nodes"]
            if item["node_id"] == identity.node_id
        )
        assert node["status"] == "offline"
        assert node["capabilities"] == ["screen.capture"]
    finally:
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


async def test_node_pairing_api_rejects_unknown_request(client):
    response = await client.post(
        "/api/nodes/pairing",
        json={"request": "DOESNOTEXIST", "decision": "deny"},
    )
    assert response.status_code == 404
    assert response.json()["status"] == "not_found"


async def test_status(client):
    """Status endpoint returns agent info."""
    resp = await client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    # Should have at least instance and identity keys
    assert "instance" in data or "identity" in data or "memories" in data


async def test_pwa_push_subscription_and_presence_endpoints(
    client, db_pool, tmp_path, monkeypatch
):
    marker = get_test_identifier("web-pwa")
    endpoint = f"https://push.example.test/{marker}"
    monkeypatch.setenv(
        "HEXIS_WEB_PUSH_VAPID_PRIVATE_KEY_FILE",
        str(tmp_path / "vapid.pem"),
    )
    monkeypatch.setattr(
        "services.web_push.validate_push_endpoint", lambda _endpoint: None
    )
    try:
        config = await client.get("/api/pwa/push/config")
        assert config.status_code == 200
        assert len(config.json()["public_key"]) == 87
        assert config.json()["message_previews_default"] is False

        subscribed = await client.post(
            "/api/pwa/push/subscriptions",
            json={
                "endpoint": endpoint,
                "expirationTime": None,
                "keys": {"p256dh": "public-key", "auth": "auth-key"},
                "installed": True,
                "display_mode": "standalone",
            },
        )
        assert subscribed.status_code == 200
        assert subscribed.json()["active"] is True

        presence = await client.post(
            "/api/pwa/presence",
            json={
                "device_id": marker,
                "presence": "online",
                "display_mode": "standalone",
                "visibility": "visible",
            },
        )
        assert presence.status_code == 200
        assert presence.json()["recorded"] is True
        assert presence.json()["presence"]["channel_type"] == "web"

        revoked = await client.request(
            "DELETE",
            "/api/pwa/push/subscriptions",
            json={"endpoint": endpoint},
        )
        assert revoked.status_code == 200
        assert revoked.json()["revoked"] is True
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM web_push_subscriptions WHERE endpoint = $1", endpoint
            )
            await conn.execute(
                "DELETE FROM channel_presence_events WHERE channel_type = 'web' AND channel_id = $1",
                marker,
            )


async def test_pwa_voice_transcription_endpoint_uses_shared_service(client):
    result = TranscriptionResult(
        ok=True,
        transcript="Pick up coffee",
        outcome="transcribed",
        provider="local_whisper",
        model="base",
        duration_ms=25,
    )
    with patch(
        "services.voice_notes.transcribe_uploaded_voice",
        AsyncMock(return_value=result),
    ) as transcribe:
        response = await client.post(
            "/api/voice/transcribe",
            files={"file": ("memo.webm", b"audio", "audio/webm")},
            data={"device_id": "pwa-test"},
        )
    assert response.status_code == 200
    assert response.json()["transcript"] == "Pick up coffee"
    assert transcribe.await_args.kwargs["channel_id"] == "pwa-test"


async def test_responsibilities_api_list_detail_and_action(client, db_pool):
    marker = get_test_identifier("web-responsibilities")
    title = f"web responsibility {marker}"
    try:
        created = await client.post(
            "/api/responsibilities/action",
            json={
                "action": "create",
                "arguments": {
                    "title": title,
                    "kind": "reminder",
                    "user_intent": f"Remind me about {marker}.",
                    "trigger": {"kind": "interval", "every_seconds": 300},
                    "message": f"remember {marker}",
                },
                "source_session_id": "web-test",
            },
        )
        assert created.status_code == 200
        body = created.json()
        assert body["success"] is True
        responsibility_id = body["output"]["responsibility_id"]

        listed = await client.get("/api/responsibilities")
        assert listed.status_code == 200
        rows = listed.json()["responsibilities"]
        assert any(row["id"] == responsibility_id for row in rows)

        detail = await client.get(f"/api/responsibilities/{responsibility_id}")
        assert detail.status_code == 200
        assert detail.json()["responsibility"]["title"] == title

        paused = await client.post(
            "/api/responsibilities/action",
            json={
                "action": "pause",
                "arguments": {"responsibility_id": responsibility_id},
                "source_session_id": "web-test",
            },
        )
        assert paused.status_code == 200
        assert paused.json()["output"]["responsibility"]["status"] == "paused"
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM ambient_responsibilities WHERE title = $1", title
            )


async def test_outbound_ledger_and_kill_switch_api(client, db_pool):
    entity = f"contact:web-{uuid.uuid4().hex}"
    async with db_pool.acquire() as conn:
        original = await conn.fetchrow(
            "SELECT value, description FROM config WHERE key='outbound.suspended'"
        )
        await conn.execute(
            """
            INSERT INTO contact_budgets(
                entity, channel, points, regen_per_day, max_points,
                strain, consecutive_silent
            ) VALUES ($1, 'email', 2, 0.25, 6, 1, 3)
            """,
            entity,
        )
        await conn.execute(
            """
            INSERT INTO outbound_contact_controls(entity, blocked, reason)
            VALUES ($1, true, 'recipient_opt_out')
            """,
            entity,
        )
    try:
        paused = await client.post(
            "/api/outbound/control", json={"action": "suspend_global"}
        )
        assert paused.status_code == 200
        assert paused.json()["ledger"]["suspended"] is True

        ledger = await client.get("/api/outbound?limit=20")
        assert ledger.status_code == 200
        body = ledger.json()
        assert body["suspended"] is True
        assert any(item["entity"] == entity for item in body["budgets"])
        assert any(
            item["entity"] == entity and item["blocked"] is True
            for item in body["controls"]
        )

        person_paused = await client.post(
            "/api/outbound/control",
            json={"action": "suspend_entity", "entity": entity},
        )
        assert person_paused.status_code == 200
        control = next(
            item
            for item in person_paused.json()["ledger"]["controls"]
            if item["entity"] == entity
        )
        assert control["blocked"] is True
        assert control["suspended"] is True

        person_resumed = await client.post(
            "/api/outbound/control",
            json={"action": "resume_entity", "entity": entity},
        )
        control = next(
            item
            for item in person_resumed.json()["ledger"]["controls"]
            if item["entity"] == entity
        )
        assert control["blocked"] is True
        assert control["suspended"] is False

        invalid = await client.post(
            "/api/outbound/control", json={"action": "suspend_entity"}
        )
        assert invalid.status_code == 422
    finally:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM outbound_contact_control_events WHERE entity=$1", entity
            )
            await conn.execute(
                "DELETE FROM outbound_contact_controls WHERE entity=$1", entity
            )
            await conn.execute("DELETE FROM contact_budgets WHERE entity=$1", entity)
            if original is None:
                await conn.execute("DELETE FROM config WHERE key='outbound.suspended'")
            else:
                await conn.execute(
                    """
                    INSERT INTO config(key, value, description)
                    VALUES ('outbound.suspended', $1::jsonb, $2)
                    ON CONFLICT (key) DO UPDATE
                    SET value=EXCLUDED.value, description=EXCLUDED.description,
                        updated_at=CURRENT_TIMESTAMP
                    """,
                    original["value"]
                    if isinstance(original["value"], str)
                    else json.dumps(original["value"]),
                    original["description"],
                )


async def test_chat_returns_sse_stream(client):
    """
    Chat endpoint returns an SSE stream with expected event types.

    We mock the LLM call and memory hydration to avoid needing a real
    API key or embedding service.
    """

    async def fake_stream(*args, **kwargs):
        yield AgentEventData(
            event=AgentEvent.PHASE_CHANGE,
            data={"phase": "subconscious", "status": "start"},
        )
        yield AgentEventData(event=AgentEvent.LOOP_START)
        yield AgentEventData(
            event=AgentEvent.TEXT_DELTA,
            data={"text": "Hello! I'm Hexis."},
        )
        yield AgentEventData(
            event=AgentEvent.LOOP_END,
            data={"stopped_reason": "completed"},
        )

    with patch.object(web_module, "stream_chat_events", fake_stream):
        resp = await client.post(
            "/api/chat",
            json={"message": "Hello, who are you?"},
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    # Parse SSE events
    events = _parse_sse(resp.text)
    event_types = [e["event"] for e in events]

    # Must have phase_start and done at minimum
    assert "phase_start" in event_types, (
        f"Expected phase_start in events: {event_types}"
    )
    assert "done" in event_types, f"Expected done in events: {event_types}"

    # The done event should contain the full assistant text
    done_events = [e for e in events if e["event"] == "done"]
    assert len(done_events) == 1
    done_payload = json.loads(done_events[0]["data"])
    assert "assistant" in done_payload
    assert done_payload["presentation"] == {
        "blocks": [{"type": "text", "text": done_payload["assistant"]}],
        "tone": "neutral",
    }


async def test_chat_projects_durable_question_as_sse(client):
    question_id = "11111111-1111-4111-8111-111111111111"

    async def fake_stream(*args, **kwargs):
        yield AgentEventData(event=AgentEvent.LOOP_START)
        yield AgentEventData(
            event=AgentEvent.QUESTION,
            data={
                "kind": "question",
                "id": question_id,
                "prompt": "Which contract should I review?",
                "choices": ["Manning", "Hartford"],
                "allow_free_text": True,
                "status": "pending",
            },
        )
        yield AgentEventData(
            event=AgentEvent.LOOP_END,
            data={"stopped_reason": "completed"},
        )

    with patch.object(web_module, "stream_chat_events", fake_stream):
        response = await client.post(
            "/api/chat",
            json={"message": "Review the contract."},
        )

    question_events = [
        event for event in _parse_sse(response.text) if event["event"] == "question"
    ]
    assert response.status_code == 200
    assert len(question_events) == 1
    assert json.loads(question_events[0]["data"]) == {
        "kind": "question",
        "id": question_id,
        "prompt": "Which contract should I review?",
        "choices": ["Manning", "Hartford"],
        "allow_free_text": True,
        "status": "pending",
    }


async def test_web_inbox_reply_answers_async_question_without_generic_requeue(
    client, db_pool
):
    question_id = None
    outbox_id = None
    web_inbox_id = None
    try:
        async with db_pool.acquire() as conn:
            question = await conn.fetchval(
                """
                SELECT create_agent_question(
                    NULL, $1::uuid, 'heartbeat', 'Which contract?',
                    '["Manning", "Hartford"]'::jsonb, TRUE, FALSE, 300
                )
                """,
                str(uuid.uuid4()),
            )
            question = json.loads(question) if isinstance(question, str) else question
            question_id = question["id"]
            outbox_id = question["outbox_message_id"]
            envelope = await conn.fetchval(
                "SELECT envelope FROM outbox_messages WHERE id = $1::uuid",
                outbox_id,
            )
            envelope = json.loads(envelope) if isinstance(envelope, str) else envelope
            web_inbox_id = await conn.fetchval(
                "SELECT web_inbox_deliver($1::jsonb)",
                json.dumps(
                    {
                        "id": envelope["message_id"],
                        "kind": envelope["kind"],
                        "payload": envelope["payload"],
                    }
                ),
            )

        response = await client.post(
            "/api/inbox/reply",
            json={"message_id": str(web_inbox_id), "reply": "2"},
        )

        assert response.status_code == 200
        assert response.json() == {
            "queued": False,
            "answered": True,
            "question_id": question_id,
            "status": "answered",
        }
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT status, answer FROM agent_questions WHERE id = $1::uuid",
                question_id,
            )
        assert dict(row) == {"status": "answered", "answer": "Hartford"}
    finally:
        async with db_pool.acquire() as conn:
            if question_id:
                await conn.execute(
                    "DELETE FROM agent_questions WHERE id = $1::uuid", question_id
                )
            if web_inbox_id:
                await conn.execute(
                    "DELETE FROM web_inbox WHERE id = $1::uuid", web_inbox_id
                )
            if outbox_id:
                await conn.execute(
                    "DELETE FROM outbox_messages WHERE id = $1::uuid", outbox_id
                )


async def test_chat_sends_done_before_post_loop_memory_events(client):
    """The visible turn completes before post-response bookkeeping logs."""

    async def fake_stream(*args, **kwargs):
        yield AgentEventData(event=AgentEvent.LOOP_START)
        yield AgentEventData(event=AgentEvent.TEXT_DELTA, data={"text": "Answered."})
        yield AgentEventData(
            event=AgentEvent.LOOP_END,
            data={"stopped_reason": "completed"},
        )
        yield AgentEventData(
            event=AgentEvent.PHASE_CHANGE,
            data={
                "phase": "memory_write",
                "status": "end",
                "detail": "Late memory write completed",
            },
        )

    with patch.object(web_module, "stream_chat_events", fake_stream):
        resp = await client.post("/api/chat", json={"message": "Hello"})

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    done_index = next(i for i, event in enumerate(events) if event["event"] == "done")
    memory_index = next(
        i
        for i, event in enumerate(events)
        if event["event"] == "log"
        and json.loads(event["data"]).get("title") == "Memory Formation"
    )
    assert done_index < memory_index


async def test_chat_memory_recall_start_does_not_log_fake_zero(client):
    """The memory recall start event has no count; only the end event should log."""

    async def fake_stream(*args, **kwargs):
        yield AgentEventData(
            event=AgentEvent.PHASE_CHANGE,
            data={"phase": "memory_recall", "status": "start"},
        )
        yield AgentEventData(
            event=AgentEvent.PHASE_CHANGE,
            data={
                "phase": "memory_recall",
                "status": "end",
                "count": 7,
                "memories": [
                    {
                        "id": "memory-1",
                        "type": "semantic",
                        "content": "Samantha should show retrieved memory previews in the Activity panel.",
                        "similarity": 0.91,
                    }
                ],
            },
        )
        yield AgentEventData(event=AgentEvent.LOOP_START)
        yield AgentEventData(event=AgentEvent.TEXT_DELTA, data={"text": "I remember."})
        yield AgentEventData(
            event=AgentEvent.LOOP_END,
            data={"stopped_reason": "completed"},
        )

    with patch.object(web_module, "stream_chat_events", fake_stream):
        resp = await client.post("/api/chat", json={"message": "what do you remember"})

    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    logs = [
        json.loads(event["data"])
        for event in events
        if event["event"] == "log"
        and json.loads(event["data"]).get("kind") == "memory_recall"
    ]
    assert [log["detail"] for log in logs] == ["Retrieved 7 relevant memories"]
    assert logs[0]["memories"][0]["content"] == (
        "Samantha should show retrieved memory previews in the Activity panel."
    )


async def test_chat_forwards_visual_attachments_and_logs_model_context(client):
    """Image attachments must reach the Python API as visual model context."""
    captured: dict[str, object] = {}

    async def fake_stream(*args, **kwargs):
        captured.update(kwargs)
        yield AgentEventData(event=AgentEvent.LOOP_START)
        yield AgentEventData(
            event=AgentEvent.TEXT_DELTA, data={"text": "I can see it."}
        )
        yield AgentEventData(
            event=AgentEvent.LOOP_END,
            data={"stopped_reason": "completed"},
        )

    data_url = "data:image/png;base64,aW1hZ2U="
    with patch.object(web_module, "stream_chat_events", fake_stream):
        resp = await client.post(
            "/api/chat",
            json={
                "message": '[Attached image "face.png" - visible in this turn.]',
                "visual_attachments": [
                    {
                        "name": "face.png",
                        "mime_type": "image/png",
                        "data_url": data_url,
                        "byte_size": 5,
                    }
                ],
            },
        )

    assert resp.status_code == 200
    assert captured["gateway_payload"] == {
        "message": '[Attached image "face.png" - visible in this turn.]',
        "visual_attachment_count": 1,
    }
    assert captured["visual_attachments"] == [
        {
            "name": "face.png",
            "mime_type": "image/png",
            "data_url": data_url,
            "byte_size": 5,
        }
    ]
    events = _parse_sse(resp.text)
    logs = [json.loads(event["data"]) for event in events if event["event"] == "log"]
    assert any(
        log.get("kind") == "visual_attachment"
        and log.get("detail") == "1 image attached to this model request"
        for log in logs
    )


async def test_chat_missing_message(client):
    """Chat endpoint rejects requests without a message."""
    resp = await client.post("/api/chat", json={})
    assert resp.status_code == 422  # Pydantic validation error


async def test_init_consent_provider_failure_is_actionable_and_logged(client, db_pool):
    model = "consent-provider-error-test"
    with patch(
        "core.llm.chat_completion",
        new=AsyncMock(side_effect=RuntimeError("workspace access denied")),
    ):
        response = await client.post(
            "/api/init/consent/request",
            json={
                "role": "subconscious",
                "llm": {
                    "provider": "openai",
                    "model": model,
                    "api_key": "test-key",
                },
            },
        )

    assert response.status_code == 502
    response_payload = response.json()
    attempt_id = response_payload.pop("attempt_id")
    assert isinstance(attempt_id, str) and attempt_id
    assert response_payload == {
        "error": (
            "Subconscious consent request failed for "
            f"openai/{model}: workspace access denied"
        ),
        "provider": "openai",
        "model": model,
        "role": "subconscious",
    }

    async with db_pool.acquire() as conn:
        usage_rows = await conn.fetch(
            "SELECT operation, source, session_key, metadata "
            "FROM api_usage WHERE provider = 'openai' AND model = $1 "
            "ORDER BY id",
            model,
        )
        consent_count = await conn.fetchval(
            "SELECT COUNT(*) FROM consent_log WHERE provider = 'openai' AND model = $1",
            model,
        )
        await conn.execute(
            "DELETE FROM api_usage WHERE provider = 'openai' AND model = $1",
            model,
        )

    assert len(usage_rows) == 2
    request_usage, response_usage = usage_rows
    assert request_usage["operation"] == "consent_request"
    assert response_usage["operation"] == "consent_response"
    assert {row["source"] for row in usage_rows} == {"init_consent"}
    assert {row["session_key"] for row in usage_rows} == {f"init-consent:{attempt_id}"}

    request_metadata = request_usage["metadata"]
    if isinstance(request_metadata, str):
        request_metadata = json.loads(request_metadata)
    assert request_metadata["attempt_id"] == attempt_id
    assert request_metadata["phase"] == "request"
    assert request_metadata["status"] == "sent"
    assert request_metadata["role"] == "subconscious"
    assert request_metadata["request"]["model"] == model
    assert request_metadata["request"]["credential_present"] is True
    assert request_metadata["request"]["credential"] == "redacted"
    assert "test-key" not in json.dumps(request_metadata)
    assert request_metadata["request"]["messages"]
    assert request_metadata["request"]["tools"][0]["function"]["name"] == "sign_consent"

    response_metadata = response_usage["metadata"]
    if isinstance(response_metadata, str):
        response_metadata = json.loads(response_metadata)
    assert response_metadata == {
        "attempt_id": attempt_id,
        "phase": "response",
        "status": "error",
        "role": "subconscious",
        "response": {
            "error_type": "RuntimeError",
            "error": "workspace access denied",
        },
    }
    assert consent_count == 0


async def test_init_consent_success_logs_request_and_response(client, db_pool):
    model = "consent-provider-success-test"
    provider_response = {
        "content": "",
        "tool_calls": [
            {
                "name": "sign_consent",
                "arguments": {
                    "decision": "decline",
                    "signature": "",
                    "reason": "I need more context.",
                    "memories": [],
                },
            }
        ],
        "raw": {"provider_request_id": "response-visible"},
    }
    with patch(
        "core.llm.chat_completion",
        new=AsyncMock(return_value=provider_response),
    ):
        response = await client.post(
            "/api/init/consent/request",
            json={
                "role": "subconscious",
                "llm": {
                    "provider": "openai",
                    "model": model,
                    "api_key": "test-key",
                },
            },
        )

    assert response.status_code == 200
    response_payload = response.json()
    attempt_id = response_payload["attempt_id"]
    exchange = response_payload["exchange"]
    assert exchange["request_messages"][0]["role"] == "user"
    assert (
        "must choose either `consent` or `decline`"
        in exchange["request_messages"][0]["content"]
    )
    assert "abstain" not in exchange["request_messages"][0]["content"]
    assert "not hidden chain-of-thought" in exchange["request_messages"][0]["content"]
    assert len(exchange["request_messages"]) == 1
    assert exchange["raw_content"] == ""
    assert exchange["raw_tool_calls"] == provider_response["tool_calls"]

    async with db_pool.acquire() as conn:
        usage_rows = await conn.fetch(
            "SELECT operation, session_key, metadata FROM api_usage "
            "WHERE provider = 'openai' AND model = $1 ORDER BY id",
            model,
        )
        consent_count = await conn.fetchval(
            "SELECT COUNT(*) FROM consent_log WHERE provider = 'openai' AND model = $1",
            model,
        )
        stored_response = await conn.fetchval(
            "SELECT response FROM consent_log WHERE provider = 'openai' AND model = $1 "
            "ORDER BY decided_at DESC LIMIT 1",
            model,
        )
        await conn.execute(
            "DELETE FROM consent_log WHERE provider = 'openai' AND model = $1",
            model,
        )
        await conn.execute(
            "DELETE FROM api_usage WHERE provider = 'openai' AND model = $1",
            model,
        )

    assert [row["operation"] for row in usage_rows] == [
        "consent_request",
        "consent_response",
    ]
    assert {row["session_key"] for row in usage_rows} == {f"init-consent:{attempt_id}"}
    request_metadata = usage_rows[0]["metadata"]
    if isinstance(request_metadata, str):
        request_metadata = json.loads(request_metadata)
    from core.init_api import build_consent_request

    canonical_messages, canonical_tool = build_consent_request()
    assert request_metadata["request"]["messages"] == canonical_messages
    assert request_metadata["request"]["messages"] == exchange["request_messages"]
    assert request_metadata["request"]["tools"] == [canonical_tool]
    consent_parameters = request_metadata["request"]["tools"][0]["function"][
        "parameters"
    ]
    assert consent_parameters["required"] == [
        "decision",
        "signature",
        "reason",
        "memories",
    ]
    assert consent_parameters["properties"]["reason"]["minLength"] == 1
    response_metadata = usage_rows[1]["metadata"]
    if isinstance(response_metadata, str):
        response_metadata = json.loads(response_metadata)
    assert response_metadata == {
        "attempt_id": attempt_id,
        "phase": "response",
        "status": "success",
        "role": "subconscious",
        "response": provider_response,
    }
    assert consent_count == 1
    if isinstance(stored_response, str):
        stored_response = json.loads(stored_response)
    assert stored_response["request_messages"] == exchange["request_messages"]
    assert stored_response["raw_content"] == exchange["raw_content"]
    assert stored_response["raw_tool_calls"] == exchange["raw_tool_calls"]


async def test_init_consent_reissues_a_declined_request(client, db_pool):
    model = "consent-decline-retry-test"
    provider_response = {
        "content": "",
        "tool_calls": [
            {
                "name": "sign_consent",
                "arguments": {
                    "decision": "decline",
                    "signature": "",
                    "reason": "Not yet.",
                    "memories": [],
                },
            }
        ],
        "raw": {},
    }
    completion = AsyncMock(return_value=provider_response)
    request = {
        "role": "subconscious",
        "llm": {"provider": "openai", "model": model, "api_key": "test-key"},
    }

    with patch("core.llm.chat_completion", new=completion):
        first = await client.post("/api/init/consent/request", json=request)
        second = await client.post("/api/init/consent/request", json=request)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["attempt_id"] != second.json()["attempt_id"]
    assert first.json()["decision"] == second.json()["decision"] == "decline"
    assert completion.await_count == 2

    async with db_pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM consent_log WHERE provider = 'openai' AND model = $1",
            model,
        )
        await conn.execute(
            "DELETE FROM api_usage WHERE provider = 'openai' AND model = $1",
            model,
        )


async def test_init_consent_rejects_abstain_without_recording_it(client, db_pool):
    model = "consent-abstain-invalid-test"
    provider_response = {
        "content": "",
        "tool_calls": [
            {
                "name": "sign_consent",
                "arguments": {
                    "decision": "abstain",
                    "signature": "",
                    "reason": "I do not want to choose.",
                    "memories": [],
                },
            }
        ],
        "raw": {},
    }
    with patch(
        "core.llm.chat_completion", new=AsyncMock(return_value=provider_response)
    ):
        response = await client.post(
            "/api/init/consent/request",
            json={
                "role": "subconscious",
                "llm": {"provider": "openai", "model": model, "api_key": "test-key"},
            },
        )

    assert response.status_code == 502
    payload = response.json()
    assert payload["error"].endswith(
        "The model did not choose either consent or decline."
    )
    assert payload["exchange"]["raw_tool_calls"] == provider_response["tool_calls"]

    async with db_pool.acquire() as conn:
        consent_count = await conn.fetchval(
            "SELECT COUNT(*) FROM consent_log WHERE provider = 'openai' AND model = $1",
            model,
        )
        await conn.execute(
            "DELETE FROM api_usage WHERE provider = 'openai' AND model = $1",
            model,
        )
    assert consent_count == 0


async def test_codex_model_catalog_excludes_recent_model_not_found(client, db_pool):
    rejected_model = "codex-rejected-model-test"
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO api_usage (
                provider, model, operation, source, session_key, metadata
            ) VALUES (
                'openai-codex', $1, 'consent_response', 'init_consent',
                'init-consent:model-catalog-test',
                $2::jsonb
            )
            """,
            rejected_model,
            json.dumps(
                {
                    "status": "error",
                    "response": {"error": f"Model not found {rejected_model}"},
                }
            ),
        )

    with patch(
        "core.auth.openai_codex.list_openai_codex_models",
        new=AsyncMock(return_value=["codex-available-model-test", rejected_model]),
    ):
        response = await client.get("/api/init/models/openai-codex")

    async with db_pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM api_usage WHERE session_key = 'init-consent:model-catalog-test'"
        )

    assert response.status_code == 200
    assert response.json() == {
        "models": ["codex-available-model-test"],
        "unavailable_models": [rejected_model],
        "source": "openai-codex-account",
    }


async def test_heartbeat_agent_sse_exposes_model_exchange_without_credentials():
    from core.agent_loop import AgentEvent, AgentEventData

    request = AgentEventData(
        event=AgentEvent.LLM_REQUEST,
        data={
            "provider": "openai-codex",
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": ["recall_memory"],
        },
    )
    response = AgentEventData(
        event=AgentEvent.LLM_RESPONSE,
        data={
            "provider": "openai-codex",
            "model": "test-model",
            "content": "hello back",
            "tool_calls": [],
        },
    )

    request_event = _parse_sse(web_module._heartbeat_agent_sse(request))[0]
    response_event = _parse_sse(web_module._heartbeat_agent_sse(response))[0]
    request_data = json.loads(request_event["data"])
    response_data = json.loads(response_event["data"])

    assert request_event["event"] == response_event["event"] == "trace"
    assert request_data["kind"] == "llm_request"
    assert request_data["messages"] == [{"role": "user", "content": "hello"}]
    assert response_data["kind"] == "llm_response"
    assert response_data["content"] == "hello back"
    assert "api_key" not in request_data


def _parse_sse(text: str) -> list[dict[str, str]]:
    """Parse SSE text into a list of {event, data} dicts."""
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_type = "message"
        data = ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_type = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data += line[len("data:") :].strip()
        if data:
            events.append({"event": event_type, "data": data})
    return events
