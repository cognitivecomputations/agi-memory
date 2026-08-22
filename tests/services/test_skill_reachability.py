"""Ordinary requests must reach the tools that serve them.

Tool exposure is skill-gated: a tool outside the active skill set is hard-refused
(`core/agent_loop.py`). Selection scored literal token overlap against the skill's
*name*, so a person asking about "email" scored zero against `gmail-actions`, and
"book time with Sarah next week" scored zero against `calendar`. Seven of ten
ordinary requests reached `core-memory` and nothing else.

This is the regression test for that. It runs the production selection path.
"""

from __future__ import annotations

import pytest

from skills.base import SkillContext
from skills.loader import load_skills
from services.skill_runtime import (
    ALWAYS_AVAILABLE_TOOL_NAMES,
    AUTO_ACTIVATE_SCORE_THRESHOLD,
    DEFAULT_SKILL_NAMES,
    DISCOVERY_TOOL_NAMES,
    _passes_specialized_gate,
    _score_skill,
    _tokens,
    skill_bound_tools,
)

class _EveryTool(frozenset):
    """Stands in for the live registry so this test needs no database.

    `load_skills` drops a skill whose `requires.tools` are unavailable; here we
    are testing *selection*, not availability, so every tool is present.
    """

    def __contains__(self, item) -> bool:  # noqa: D105
        return True


ALL_TOOLS = _EveryTool()


# Requests a person would actually make, paired with a skill that must activate.
REQUESTS = [
    ("what's on my calendar today?", "calendar"),
    ("book time with Sarah next week", "calendar"),
    ("remind me to call Bob at 4pm tomorrow", "calendar"),
    ("did I get anything important in email?", None),  # any email-shaped skill
    ("summarize this contract and tell me the risky clauses", "knowledge-ingest"),
    ("how much have I spent on the API this week?", "cost-report"),
    ("who is Manning and what do we owe them?", "crm-lookup"),
    ("should I take this deal or walk away?", "council"),
]


def _select(skills, query, max_skills=4):
    tokens = _tokens(query)
    selected = [s for s in skills if s.name in DEFAULT_SKILL_NAMES]
    names = {s.name for s in selected}
    scored = sorted(
        ((_score_skill(s, tokens), s) for s in skills
         if s.name not in names and _passes_specialized_gate(s, tokens)),
        key=lambda item: (-item[0], item[1].name),
    )
    for score, skill in scored:
        if score < AUTO_ACTIVATE_SCORE_THRESHOLD:
            continue
        selected.append(skill)
        names.add(skill.name)
        if len(selected) >= max_skills:
            break
    return names


@pytest.fixture(scope="module")
def chat_skills():
    return load_skills(SkillContext.CHAT, available_tools=ALL_TOOLS, available_config=None)


@pytest.mark.parametrize("query,expected", REQUESTS)
def test_ordinary_requests_activate_a_skill_that_can_serve_them(chat_skills, query, expected):
    names = _select(chat_skills, query)
    assert names != {"core-memory"}, f"{query!r} reached core-memory alone"
    if expected:
        assert expected in names, f"{query!r} did not activate {expected}: got {sorted(names)}"


def test_the_council_fires_on_decisions_rather_than_on_the_word_council(chat_skills):
    # It scored 1, 4, 1 on exactly these before aliases existed; threshold is 5.
    for q in ("should I take this deal or walk away?",
              "help me think through a hard decision",
              "weigh the tradeoffs on hiring"):
        assert "council" in _select(chat_skills, q), q


def test_read_only_everyday_tools_are_always_reachable(chat_skills):
    """Even a turn that activates nothing can still look things up."""
    names = _select(chat_skills, "tell me a joke")
    allowed = set(DISCOVERY_TOOL_NAMES) | set(ALWAYS_AVAILABLE_TOOL_NAMES)
    for skill in chat_skills:
        if skill.name in names:
            allowed.update(skill_bound_tools(skill))
    for tool in ("web_search", "web_fetch", "calendar_events", "search_contacts"):
        assert tool in allowed, f"{tool} should be in the always-on floor"


def test_every_tool_is_reachable_through_some_skill():
    """A tool bound to no skill can never be unlocked by use_skill."""
    bound: set[str] = set()
    for ctx in (SkillContext.CHAT, SkillContext.HEARTBEAT):
        for skill in load_skills(ctx, available_tools=ALL_TOOLS, available_config=None):
            bound.update(skill_bound_tools(skill))
    # manage_sessions is delegation; explore_concept/_subgraph are memory acts.
    for tool in ("manage_sessions", "explore_concept", "explore_subgraph",
                 "database_backup", "post_process_output"):
        assert tool in bound, f"{tool} is bound to no skill and cannot be reached"
