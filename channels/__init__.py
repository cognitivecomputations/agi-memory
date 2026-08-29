"""
Hexis Channel System

Multi-channel messaging adapters that let users talk to the agent
from Discord, Telegram, and other platforms.
"""

from .base import (
    ChannelAdapter,
    ChannelCapabilities,
    ChannelMessage,
)
from .conversation import process_channel_message
from .manager import ChannelManager
from .media import Attachment
from .presentation import (
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

__all__ = [
    "Attachment",
    "ActionButton",
    "ActionsBlock",
    "ChannelAdapter",
    "ChannelCapabilities",
    "ChannelMessage",
    "ChannelManager",
    "CitationBlock",
    "ContextBlock",
    "DividerBlock",
    "MarkdownDialect",
    "MessagePresentation",
    "TextBlock",
    "agent_question_presentation",
    "citations_from_tool_output",
    "normalize_message_presentation",
    "presentation_from_text",
    "process_channel_message",
    "render_presentation",
]
