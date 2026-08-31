"""Portable, presentation-only message blocks for every Hexis surface.

Presentation never replaces the canonical conversation text stored by the
agent.  It gives each delivery surface enough structure to render that text
without learning platform-specific payloads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Literal, TypeAlias


class MarkdownDialect(str, Enum):
    """Text formatting understood by a delivery surface."""

    PLAIN = "plain"
    MARKDOWN = "markdown"
    SLACK = "slack-mrkdwn"
    TELEGRAM = "telegram-markdown"


PresentationTone: TypeAlias = Literal["neutral", "info", "success", "warning", "danger"]


@dataclass(frozen=True)
class TextBlock:
    """Primary markdown-ish message text."""

    text: str


@dataclass(frozen=True)
class ContextBlock:
    """Lower-emphasis supporting text."""

    text: str


@dataclass(frozen=True)
class DividerBlock:
    """A semantic break between adjacent blocks."""


@dataclass(frozen=True)
class CitationBlock:
    """A stable memory/document footnote with a local drill-down target."""

    citation_id: str
    label: str
    href: str
    trust_level: float
    low_trust: bool = False
    source_kind: str | None = None
    locator: dict[str, Any] | None = None
    memory_id: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None

    def __post_init__(self) -> None:
        if not self.citation_id.strip():
            raise ValueError("citation id must not be blank")
        if not re.fullmatch(r"[A-Za-z0-9:_-]+", self.citation_id):
            raise ValueError("citation id contains unsupported characters")
        if not self.label.strip():
            raise ValueError("citation label must not be blank")
        if not self.href.startswith(("/memories?", "/documents?")):
            raise ValueError("citation href must target a local memory or document")
        if not 0.0 <= float(self.trust_level) <= 1.0:
            raise ValueError("citation trust_level must be between 0 and 1")
        if self.locator is not None and not isinstance(self.locator, dict):
            raise ValueError("citation locator must be an object")


@dataclass(frozen=True)
class ActionButton:
    """One explicit user choice, rendered interactively where supported."""

    action_id: str
    label: str
    value: str
    style: Literal["default", "primary", "danger"] = "default"

    def __post_init__(self) -> None:
        if not self.action_id.strip():
            raise ValueError("action button id must not be blank")
        if not self.label.strip():
            raise ValueError("action button label must not be blank")
        if not self.value:
            raise ValueError("action button value must not be blank")
        if self.style not in {"default", "primary", "danger"}:
            raise ValueError(f"unsupported action button style: {self.style!r}")


@dataclass(frozen=True)
class ActionsBlock:
    """A row of choices; non-interactive surfaces degrade to labeled text."""

    actions: tuple[ActionButton, ...]
    block_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.actions, tuple):
            object.__setattr__(self, "actions", tuple(self.actions))
        if not self.actions:
            raise ValueError("actions block requires at least one action")
        if len(self.actions) > 5:
            raise ValueError("actions block supports at most five actions")
        ids = [action.action_id for action in self.actions]
        if len(set(ids)) != len(ids):
            raise ValueError("action ids must be unique within a block")


PresentationBlock: TypeAlias = (
    TextBlock | ContextBlock | DividerBlock | CitationBlock | ActionsBlock
)


@dataclass(frozen=True)
class MessagePresentation:
    """Ordered portable blocks plus optional presentation metadata."""

    blocks: tuple[PresentationBlock, ...] = ()
    title: str | None = None
    tone: PresentationTone = "neutral"

    def __post_init__(self) -> None:
        if not isinstance(self.blocks, tuple):
            object.__setattr__(self, "blocks", tuple(self.blocks))
        if self.title is not None and not self.title.strip():
            raise ValueError("presentation title must not be blank")
        if not self.title and not self.blocks:
            raise ValueError("presentation requires a title or at least one block")
        if self.tone not in {"neutral", "info", "success", "warning", "danger"}:
            raise ValueError(f"unsupported presentation tone: {self.tone!r}")
        for index, block in enumerate(self.blocks):
            if not isinstance(
                block,
                (TextBlock, ContextBlock, DividerBlock, CitationBlock, ActionsBlock),
            ):
                raise TypeError(f"unsupported presentation block at index {index}")
            if isinstance(block, (TextBlock, ContextBlock)) and not block.text.strip():
                raise ValueError(f"presentation block {index} text must not be blank")

    def to_dict(self) -> dict[str, Any]:
        blocks: list[dict[str, Any]] = []
        for block in self.blocks:
            if isinstance(block, TextBlock):
                blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ContextBlock):
                blocks.append({"type": "context", "text": block.text})
            elif isinstance(block, ActionsBlock):
                action_block: dict[str, Any] = {
                    "type": "actions",
                    "actions": [
                        {
                            "action_id": action.action_id,
                            "label": action.label,
                            "value": action.value,
                            "style": action.style,
                        }
                        for action in block.actions
                    ],
                }
                if block.block_id:
                    action_block["block_id"] = block.block_id
                blocks.append(action_block)
            elif isinstance(block, CitationBlock):
                citation_block: dict[str, Any] = {
                    "type": "citation",
                    "citation_id": block.citation_id,
                    "label": block.label,
                    "href": block.href,
                    "trust_level": block.trust_level,
                    "low_trust": block.low_trust,
                }
                for key in (
                    "source_kind",
                    "locator",
                    "memory_id",
                    "document_id",
                    "chunk_id",
                ):
                    value = getattr(block, key)
                    if value is not None:
                        citation_block[key] = value
                blocks.append(citation_block)
            else:
                blocks.append({"type": "divider"})
        result: dict[str, Any] = {"blocks": blocks, "tone": self.tone}
        if self.title:
            result["title"] = self.title
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MessagePresentation":
        raw_blocks = value.get("blocks", [])
        if not isinstance(raw_blocks, list):
            raise ValueError("presentation.blocks must be a list")

        blocks: list[PresentationBlock] = []
        for index, raw in enumerate(raw_blocks):
            if not isinstance(raw, dict):
                raise ValueError(f"presentation.blocks[{index}] must be an object")
            block_type = raw.get("type")
            if block_type in {"text", "context"}:
                text = raw.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"presentation.blocks[{index}].text must be non-blank text"
                    )
                block = TextBlock(text) if block_type == "text" else ContextBlock(text)
                blocks.append(block)
            elif block_type == "divider":
                blocks.append(DividerBlock())
            elif block_type == "actions":
                raw_actions = raw.get("actions")
                if not isinstance(raw_actions, list) or not raw_actions:
                    raise ValueError(
                        f"presentation.blocks[{index}].actions must be a non-empty list"
                    )
                actions: list[ActionButton] = []
                for action_index, raw_action in enumerate(raw_actions):
                    if not isinstance(raw_action, dict):
                        raise ValueError(
                            f"presentation.blocks[{index}].actions[{action_index}] must be an object"
                        )
                    try:
                        actions.append(
                            ActionButton(
                                action_id=str(raw_action.get("action_id") or ""),
                                label=str(raw_action.get("label") or ""),
                                value=str(raw_action.get("value") or ""),
                                style=str(raw_action.get("style") or "default"),  # type: ignore[arg-type]
                            )
                        )
                    except ValueError as exc:
                        raise ValueError(
                            f"presentation.blocks[{index}].actions[{action_index}]: {exc}"
                        ) from exc
                block_id = raw.get("block_id")
                if block_id is not None and not isinstance(block_id, str):
                    raise ValueError(
                        f"presentation.blocks[{index}].block_id must be text"
                    )
                blocks.append(ActionsBlock(tuple(actions), block_id=block_id))
            elif block_type == "citation":
                try:
                    trust_level = float(raw.get("trust_level"))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"presentation.blocks[{index}].trust_level must be a number"
                    ) from exc
                try:
                    blocks.append(
                        CitationBlock(
                            citation_id=str(raw.get("citation_id") or ""),
                            label=str(raw.get("label") or ""),
                            href=str(raw.get("href") or ""),
                            trust_level=trust_level,
                            low_trust=bool(raw.get("low_trust")),
                            source_kind=(
                                str(raw["source_kind"])
                                if raw.get("source_kind") is not None
                                else None
                            ),
                            locator=raw.get("locator"),
                            memory_id=(
                                str(raw["memory_id"])
                                if raw.get("memory_id") is not None
                                else None
                            ),
                            document_id=(
                                str(raw["document_id"])
                                if raw.get("document_id") is not None
                                else None
                            ),
                            chunk_id=(
                                str(raw["chunk_id"])
                                if raw.get("chunk_id") is not None
                                else None
                            ),
                        )
                    )
                except ValueError as exc:
                    raise ValueError(f"presentation.blocks[{index}]: {exc}") from exc
            else:
                raise ValueError(
                    f"presentation.blocks[{index}].type is unsupported: {block_type!r}"
                )

        title = value.get("title")
        if title is not None and not isinstance(title, str):
            raise ValueError("presentation.title must be text")
        tone = value.get("tone", "neutral")
        if not isinstance(tone, str):
            raise ValueError("presentation.tone must be text")
        return cls(blocks=tuple(blocks), title=title, tone=tone)  # type: ignore[arg-type]


def normalize_message_presentation(value: Any) -> MessagePresentation:
    """Normalize a wire payload without silently dropping malformed blocks."""

    if isinstance(value, MessagePresentation):
        return value
    if not isinstance(value, dict):
        raise ValueError("presentation must be an object")
    return MessagePresentation.from_dict(value)


def _citation_from_mapping(value: dict[str, Any]) -> CitationBlock | None:
    try:
        return CitationBlock(
            citation_id=str(value.get("citation_id") or ""),
            label=str(value.get("label") or ""),
            href=str(value.get("href") or ""),
            trust_level=float(value.get("trust_level")),
            low_trust=bool(value.get("low_trust")),
            source_kind=(
                str(value["source_kind"])
                if value.get("source_kind") is not None
                else None
            ),
            locator=value.get("locator")
            if isinstance(value.get("locator"), dict)
            else None,
            memory_id=(str(value["memory_id"]) if value.get("memory_id") else None),
            document_id=(
                str(value["document_id"]) if value.get("document_id") else None
            ),
            chunk_id=(str(value["chunk_id"]) if value.get("chunk_id") else None),
        )
    except (TypeError, ValueError):
        return None


def citations_from_tool_output(value: Any) -> tuple[CitationBlock, ...]:
    """Collect trusted DB-shaped citation envelopes from a tool result."""

    found: dict[str, CitationBlock] = {}

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            raw_citation = item.get("citation")
            if isinstance(raw_citation, dict):
                citation = _citation_from_mapping(raw_citation)
                if citation is not None:
                    found.setdefault(citation.citation_id, citation)
            for key, child in item.items():
                if key != "citation":
                    visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return tuple(found.values())


def presentation_from_text(
    text: str,
    citations: Iterable[CitationBlock | dict[str, Any]] | None = None,
) -> MessagePresentation:
    """Wrap canonical text and its actual memory/document footnotes."""

    normalized: dict[str, CitationBlock] = {}
    for value in citations or ():
        citation = (
            value if isinstance(value, CitationBlock) else _citation_from_mapping(value)
        )
        if citation is not None:
            normalized.setdefault(citation.citation_id, citation)

    reference_order = list(dict.fromkeys(re.findall(r"\[\^([A-Za-z0-9:_-]+)\]", text)))
    ordered = [
        normalized[citation_id]
        for citation_id in reference_order
        if citation_id in normalized
    ]
    if not ordered:
        # A model can omit markers; the structural layer still exposes the
        # sources it consulted rather than silently presenting unsourced prose.
        ordered = list(normalized.values())
    blocks: list[PresentationBlock] = [TextBlock(text)]
    blocks.extend(ordered[:20])
    return MessagePresentation(blocks=tuple(blocks))


def agent_question_presentation(payload: dict[str, Any]) -> MessagePresentation:
    """Render one ask_user event as a numbered, replyable channel card."""
    question_id = str(payload.get("id") or "")
    code = question_id.replace("-", "")[:8].upper()
    prompt = str(payload.get("prompt") or "I need your input.").strip()
    choices = [
        str(item).strip() for item in payload.get("choices", []) if str(item).strip()
    ][:4]
    allow_free_text = payload.get("allow_free_text") is not False
    choice_lines = [f"{index}. {choice}" for index, choice in enumerate(choices, 1)]
    if allow_free_text:
        choice_lines.append(
            f"{len(choices) + 1}. Other (type your answer)"
            if choices
            else "Type your answer."
        )
    guidance = "Reply with a number." if choices else "Reply with your answer."
    if code:
        guidance += f" If more than one question is waiting, include code {code}."
    blocks: list[PresentationBlock] = [TextBlock(prompt)]
    if choice_lines:
        blocks.append(TextBlock("\n".join(choice_lines)))
    blocks.append(ContextBlock(guidance))
    return MessagePresentation(
        title=f"Question {code}" if code else "Question",
        blocks=tuple(blocks),
        tone="info",
    )


def render_presentation(
    presentation: MessagePresentation,
    dialect: MarkdownDialect | str = MarkdownDialect.PLAIN,
) -> str:
    """Render portable blocks for a surface's declared text dialect."""

    try:
        resolved = MarkdownDialect(dialect)
    except ValueError:
        resolved = MarkdownDialect.PLAIN

    sections: list[str] = []
    if presentation.title:
        if resolved is MarkdownDialect.MARKDOWN:
            sections.append(f"**{presentation.title}**")
        elif resolved in {MarkdownDialect.SLACK, MarkdownDialect.TELEGRAM}:
            sections.append(f"*{presentation.title}*")
        else:
            sections.append(presentation.title)

    for block in presentation.blocks:
        if isinstance(block, DividerBlock):
            sections.append(
                "---" if resolved is not MarkdownDialect.PLAIN else "-" * 40
            )
        elif isinstance(block, ActionsBlock):
            sections.append(
                "Actions: " + " | ".join(action.label for action in block.actions)
            )
        elif isinstance(block, CitationBlock):
            locator = _render_citation_locator(block.locator)
            trust = f"trust {round(block.trust_level * 100)}%"
            if block.low_trust:
                trust += ", low trust"
            label = f"[{block.citation_id}] {block.label}"
            if resolved is MarkdownDialect.MARKDOWN:
                label = f"[{label}]({block.href})"
            detail = " · ".join(part for part in (locator, trust) if part)
            sections.append(f"{label} — {detail}" if detail else label)
        elif isinstance(block, ContextBlock) and resolved in {
            MarkdownDialect.MARKDOWN,
            MarkdownDialect.SLACK,
        }:
            sections.append("\n".join(f"> {line}" for line in block.text.splitlines()))
        else:
            sections.append(block.text)

    return "\n\n".join(sections)


def _render_citation_locator(locator: dict[str, Any] | None) -> str:
    if not locator:
        return ""
    page_start = locator.get("page_start")
    page_end = locator.get("page_end")
    if page_start:
        return f"page {page_start}" + (
            f"–{page_end}" if page_end and page_end != page_start else ""
        )
    sheet = locator.get("sheet_name")
    row_start = locator.get("row_start")
    row_end = locator.get("row_end")
    if sheet:
        rows = ""
        if row_start:
            rows = f", row {row_start}" + (
                f"–{row_end}" if row_end and row_end != row_start else ""
            )
        return f"sheet {sheet}{rows}"
    headings = locator.get("heading_path")
    if isinstance(headings, list) and headings:
        return " › ".join(str(item) for item in headings if str(item).strip())
    chunk_index = locator.get("chunk_index")
    return f"chunk {chunk_index}" if chunk_index is not None else ""
