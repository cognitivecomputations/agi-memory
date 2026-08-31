"""Portable channel presentation contract and delivery tests."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from channels.base import ChannelAdapter, ChannelCapabilities
from channels.manager import ChannelManager
from channels.outbox import ChannelOutboxConsumer, _resolve_payload_message
from channels.presentation import (
    ActionButton,
    ActionsBlock,
    CitationBlock,
    ContextBlock,
    DividerBlock,
    MarkdownDialect,
    MessagePresentation,
    TextBlock,
    agent_question_presentation,
    citations_from_tool_output,
    normalize_message_presentation,
    presentation_from_text,
    render_presentation,
)
from channels.slack_adapter import SlackAdapter
from services.approval_slack_actions import build_approval_presentation
from services.operator_approval import redact_approval_arguments
from services.outbound_safety import OutboundPreparation


def _presentation() -> MessagePresentation:
    return MessagePresentation(
        title="Deployment",
        tone="success",
        blocks=(
            TextBlock("**Ready** for review."),
            DividerBlock(),
            ContextBlock("Derived from the live check."),
        ),
    )


def test_presentation_wire_round_trip() -> None:
    presentation = _presentation()

    assert normalize_message_presentation(presentation.to_dict()) == presentation


def test_action_presentation_round_trip_and_plain_fallback() -> None:
    presentation = MessagePresentation(
        title="Decision",
        blocks=(
            ActionsBlock(
                actions=(
                    ActionButton("approve", "Approve", "request-1", "primary"),
                    ActionButton("deny", "Deny", "request-1", "danger"),
                ),
                block_id="decision-1",
            ),
        ),
    )

    assert normalize_message_presentation(presentation.to_dict()) == presentation
    assert render_presentation(presentation).endswith("Actions: Approve | Deny")


def test_citation_presentation_round_trip_and_low_trust_fallback() -> None:
    citation = CitationBlock(
        citation_id="mem-12345678-1234-4234-8234-123456789abc",
        label="Manning agreement",
        href="/documents?document=12345678-1234-4234-8234-123456789abc",
        trust_level=0.42,
        low_trust=True,
        source_kind="document",
        locator={"page_start": 4, "page_end": 5},
    )
    presentation = presentation_from_text(
        "The retainer was quarterly.[^mem-12345678-1234-4234-8234-123456789abc]",
        [citation],
    )

    assert normalize_message_presentation(presentation.to_dict()) == presentation
    rendered = render_presentation(presentation, MarkdownDialect.MARKDOWN)
    assert "page 4–5 · trust 42%, low trust" in rendered
    assert "](/documents?document=" in rendered


def test_citations_are_collected_from_tool_envelopes_without_trusting_prose() -> None:
    tool_output = {
        "memories": [
            {
                "content": "claim",
                "citation": {
                    "citation_id": "mem-12345678-1234-4234-8234-123456789abc",
                    "label": "A memory",
                    "href": "/memories?memory=12345678-1234-4234-8234-123456789abc",
                    "trust_level": 0.8,
                    "low_trust": False,
                },
            },
            {
                "content": "model-invented",
                "citation": {
                    "citation_id": "evil",
                    "label": "Unsafe",
                    "href": "javascript:alert(1)",
                    "trust_level": 1,
                },
            },
        ]
    }

    citations = citations_from_tool_output(tool_output)
    assert [citation.citation_id for citation in citations] == [
        "mem-12345678-1234-4234-8234-123456789abc"
    ]


def test_temporal_diff_citations_are_found_recursively_and_deduplicated() -> None:
    envelope = {
        "citation_id": "mem-12345678-1234-4234-8234-123456789abc",
        "label": "Historical memory",
        "href": "/memories?memory=12345678-1234-4234-8234-123456789abc",
        "trust_level": 0.8,
        "low_trust": False,
    }
    output = {
        "from_snapshot": {"memories": [{"citation": envelope}]},
        "to_snapshot": {"memories": [{"citation": envelope}]},
        "expired": [{"citation": envelope}],
    }

    citations = citations_from_tool_output(output)

    assert [citation.citation_id for citation in citations] == [envelope["citation_id"]]


def test_agent_question_presentation_has_numbered_choices_and_other() -> None:
    presentation = agent_question_presentation(
        {
            "id": "12345678-1234-4234-8234-123456789abc",
            "prompt": "Which contract should I review?",
            "choices": ["Manning", "Hartford"],
            "allow_free_text": True,
        }
    )

    rendered = render_presentation(presentation)
    assert rendered.startswith("Question 12345678")
    assert "1. Manning" in rendered
    assert "2. Hartford" in rendered
    assert "3. Other (type your answer)" in rendered
    assert "Reply with a number" in rendered


def test_approval_presentation_can_disable_interactive_controls() -> None:
    presentation = build_approval_presentation(
        approval_request_id="12345678-1234-1234-1234-123456789abc",
        message="Approve this exact action.",
        interactive=False,
    )

    assert not any(isinstance(block, ActionsBlock) for block in presentation.blocks)


def test_approval_preview_redacts_secrets_and_bounds_content() -> None:
    preview = redact_approval_arguments(
        {
            "recipient": "person@example.com",
            "access-token": "do-not-store",
            "nested": {"password": "do-not-store", "body": "x" * 700},
        }
    )

    assert preview["recipient"] == "person@example.com"
    assert preview["access-token"] == "[redacted]"
    assert preview["nested"]["password"] == "[redacted]"
    assert len(preview["nested"]["body"]) == 500


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"blocks": "wrong"}, "presentation.blocks must be a list"),
        (
            {"blocks": [{"type": "buttons", "buttons": []}]},
            "presentation.blocks[0].type is unsupported",
        ),
        (
            {"blocks": [{"type": "text", "text": ""}]},
            "presentation.blocks[0].text must be non-blank text",
        ),
    ],
)
def test_malformed_presentation_fails_with_path(value, message: str) -> None:
    with pytest.raises(
        ValueError, match=message.replace("[", r"\[").replace("]", r"\]")
    ):
        normalize_message_presentation(value)


def test_renderers_preserve_order_and_degrade_context() -> None:
    presentation = _presentation()

    assert render_presentation(presentation) == (
        "Deployment\n\n**Ready** for review.\n\n"
        "----------------------------------------\n\nDerived from the live check."
    )
    assert render_presentation(presentation, MarkdownDialect.MARKDOWN) == (
        "**Deployment**\n\n**Ready** for review.\n\n---\n\n"
        "> Derived from the live check."
    )
    assert render_presentation(presentation, MarkdownDialect.SLACK).startswith(
        "*Deployment*\n\n"
    )
    assert render_presentation(presentation, MarkdownDialect.TELEGRAM) == (
        "*Deployment*\n\n**Ready** for review.\n\n---\n\nDerived from the live check."
    )


class _PresentationAdapter(ChannelAdapter):
    def __init__(self) -> None:
        self.sent: list[dict] = []

    @property
    def channel_type(self) -> str:
        return "presentation-test"

    @property
    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(max_message_length=30)

    async def start(self, on_message) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send(
        self,
        channel_id,
        text,
        *,
        reply_to=None,
        thread_id=None,
    ):
        self.sent.append(
            {
                "channel_id": channel_id,
                "text": text,
                "reply_to": reply_to,
                "thread_id": thread_id,
            }
        )
        return f"message-{len(self.sent)}"

    async def send_typing(self, channel_id) -> None:
        return None


async def test_adapter_chunks_presentation_without_losing_reply_context() -> None:
    adapter = _PresentationAdapter()
    presentation = MessagePresentation(
        blocks=(TextBlock("First paragraph.\n\nSecond paragraph that is longer."),)
    )

    message_id = await adapter.send_presentation(
        "channel-1", presentation, reply_to="source-1", thread_id="thread-1"
    )

    assert message_id == "message-1"
    assert "".join(item["text"] for item in adapter.sent).replace(" ", "") == (
        "Firstparagraph.Secondparagraphthatislonger."
    )
    assert adapter.sent[0]["reply_to"] == "source-1"
    assert all(item["thread_id"] == "thread-1" for item in adapter.sent)
    assert all(item["reply_to"] is None for item in adapter.sent[1:])


async def test_manager_dispatches_portable_presentation() -> None:
    adapter = _PresentationAdapter()
    manager = ChannelManager(pool=MagicMock())
    manager.register(adapter)

    message_id = await manager.send("presentation-test", "channel-1", _presentation())

    assert message_id == "message-1"
    assert adapter.sent[0]["text"].startswith("Deployment")


async def test_slack_renders_approval_controls_as_private_block_kit() -> None:
    adapter = SlackAdapter()
    client = MagicMock()
    client.conversations_open = AsyncMock(return_value={"channel": {"id": "D123"}})
    client.chat_postMessage = AsyncMock(return_value={"ts": "171.2"})
    adapter._app = MagicMock(client=client)
    presentation = build_approval_presentation(
        approval_request_id="12345678-1234-1234-1234-123456789abc",
        message="Approve this exact action.",
    )

    message_id = await adapter.send_presentation("U123", presentation)

    assert message_id == "171.2"
    client.conversations_open.assert_awaited_once_with(users="U123")
    payload = client.chat_postMessage.await_args.kwargs
    assert payload["channel"] == "D123"
    action_block = next(
        block for block in payload["blocks"] if block["type"] == "actions"
    )
    assert [item["action_id"] for item in action_block["elements"]] == [
        "operator_approval_approve",
        "operator_approval_deny",
    ]


async def test_outbox_routes_presentation_and_logs_plain_mirror() -> None:
    manager = MagicMock()
    manager.send = AsyncMock(return_value="sent-1")
    consumer = ChannelOutboxConsumer(manager, MagicMock())
    consumer._log_delivery = AsyncMock()
    body = {
        "kind": "channel_message",
        "id": "outbox-1",
        "payload": {
            "presentation": _presentation().to_dict(),
            "delivery_mode": "direct",
            "target_channel": "discord",
            "target_id": "channel-1",
        },
    }

    with (
        patch(
            "services.outbound_safety.prepare_outbox_outbound",
            AsyncMock(
                return_value=OutboundPreparation(
                    allowed=True,
                    arguments={"message": render_presentation(_presentation())},
                )
            ),
        ),
        patch(
            "services.outbound_safety.finalize_outbox_outbound",
            AsyncMock(),
        ),
    ):
        await consumer._process_message(body)

    outbound = manager.send.await_args.args[2]
    assert isinstance(outbound, MessagePresentation)
    assert outbound == _presentation()
    assert consumer._log_delivery.await_args.args[4].startswith("Deployment")


def test_outbox_text_payload_remains_backward_compatible() -> None:
    message, mirror = _resolve_payload_message({"content": "Legacy text"})

    assert message == "Legacy text"
    assert mirror == "Legacy text"


def test_live_adapter_dialects_match_native_send_paths() -> None:
    from channels.discord_adapter import DiscordAdapter
    from channels.matrix_adapter import MatrixAdapter
    from channels.slack_adapter import SlackAdapter
    from channels.telegram_adapter import TelegramAdapter

    assert DiscordAdapter().capabilities.markdown_dialect is MarkdownDialect.MARKDOWN
    assert TelegramAdapter().capabilities.markdown_dialect is MarkdownDialect.TELEGRAM
    assert SlackAdapter().capabilities.markdown_dialect is MarkdownDialect.SLACK
    assert MatrixAdapter().capabilities.markdown_dialect is MarkdownDialect.PLAIN
