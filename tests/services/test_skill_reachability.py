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
    select_skills,
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


WAVE_A_REQUESTS = [
    ("help me prepare for my trip to Lisbon", "travel-prep"),
    ("triage my inbox and show what can be archived", "inbox-triage"),
    ("follow up after today's meeting", "meeting-follow-up"),
    ("run my weekly review", "weekly-review"),
    ("capture this receipt as an expense", "expense-capture"),
    ("add this note to my Obsidian vault", "personal-notes"),
]

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
] + WAVE_A_REQUESTS

WAVE_A_SKILL_NAMES = {
    "travel-prep",
    "inbox-triage",
    "meeting-follow-up",
    "weekly-review",
    "expense-capture",
    "personal-notes",
}

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


def test_the_everyday_floor_does_not_reach_autonomous_turns():
    """The floor is for answering someone who asked, not for acting unprompted.

    `integrations.gmail.heartbeat_digest_enabled` exists to authorize reading
    mail during a heartbeat, when nobody asked. A convenience floor must not
    become a way around that gate.
    """
    from core.tools.base import ToolContext
    import inspect
    import services.skill_runtime as sr

    src = inspect.getsource(sr.select_skills)
    assert "if tool_context != ToolContext.HEARTBEAT:" in src, (
        "the always-available floor must be scoped to live turns"
    )
    assert "email_list" in sr.ALWAYS_AVAILABLE_TOOL_NAMES
    assert ToolContext.HEARTBEAT is not None


def test_wave_a_skills_are_chat_only_and_keep_mutations_explicit(chat_skills):
    """Everyday convenience must not become unrequested background action."""
    by_name = {skill.name: skill for skill in chat_skills}
    assert WAVE_A_SKILL_NAMES <= by_name.keys()
    for name in WAVE_A_SKILL_NAMES:
        skill = by_name[name]
        assert skill.contexts == [SkillContext.CHAT]
        assert skill.bound_tools
        assert "explicit" in skill.content.lower(), name


def test_wave_a_manifests_only_name_real_core_or_plugin_tools(chat_skills):
    """A typo in a declarative binding is an invisible dead capability."""
    from core.tools.registry import create_default_registry
    from plugins.installed.asana.tools import create_asana_tools
    from plugins.installed.fathom.tools import create_fathom_tools
    from plugins.installed.todoist.tools import create_todoist_tools

    registry = create_default_registry(None)
    known = set(registry.list_names())
    for factory in (create_asana_tools, create_fathom_tools, create_todoist_tools):
        known.update(handler.spec.name for handler in factory())

    by_name = {skill.name: skill for skill in chat_skills}
    for name in WAVE_A_SKILL_NAMES:
        unknown = set(skill_bound_tools(by_name[name])) - known
        assert not unknown, f"{name} binds unknown tools: {sorted(unknown)}"


@pytest.mark.parametrize("query,expected", WAVE_A_REQUESTS)
@pytest.mark.asyncio
async def test_wave_a_requests_activate_through_the_full_selector(query, expected):
    """Exercise selection as the agent loop calls it, including tool exposure."""
    from core.tools.base import ToolContext
    from core.tools.registry import create_default_registry
    from plugins.installed.asana.tools import create_asana_tools
    from plugins.installed.fathom.tools import create_fathom_tools
    from plugins.installed.todoist.tools import create_todoist_tools

    registry = create_default_registry(None)
    for factory in (create_asana_tools, create_fathom_tools, create_todoist_tools):
        registry.register_all(factory())

    selection = await select_skills(registry, ToolContext.CHAT, query=query)
    selected = {skill.name: skill for skill in selection.skills}
    assert expected in selected
    expected_tools = set(skill_bound_tools(selected[expected])) & set(registry.list_names())
    assert expected_tools <= selection.allowed_tool_names


class TestSemanticSelection:
    """Selection ranks by what a request means, not which words it shares.

    Token overlap needed a hand-written alias list per skill, and the lists
    misfired in both directions: `council` never activated on the decisions it
    exists for until `decide` was added, and then fired on "what did we decide
    last time", which is recall. Embeddings answer "most like which" directly.
    """

    def test_the_gate_is_the_shape_of_the_distribution_not_a_fixed_cutoff(self):
        """Absolute cosine cannot separate signal from noise here.

        Measured over the probe: genuine matches span 0.46–0.73 and noise spans
        0.40–0.54, so any fixed line cuts through both. A peaked distribution is
        what "about something in particular" looks like.
        """
        from services.skill_runtime import _select_by_distribution

        # One clear outlier among many mediocre scores → it is selected.
        peaked = {f"s{i}": 0.42 + (i % 5) * 0.01 for i in range(25)}
        peaked["winner"] = 0.75
        picked = [n for _s, n in _select_by_distribution(peaked, z_threshold=2.0, floor=0.40)]
        assert picked == ["winner"]

        # A flat distribution means nothing in particular — a greeting.
        flat = {f"s{i}": 0.50 + (i % 3) * 0.005 for i in range(25)}
        assert _select_by_distribution(flat, z_threshold=2.0, floor=0.40) == []

    def test_the_absolute_floor_still_backstops_a_peaked_but_weak_run(self):
        from services.skill_runtime import _select_by_distribution

        weak = {f"s{i}": 0.10 for i in range(25)}
        weak["tallest_dwarf"] = 0.30
        assert _select_by_distribution(weak, z_threshold=2.0, floor=0.40) == []

    def test_embedding_text_carries_aliases_as_prose_not_as_tokens(self):
        """Aliases survive as example phrasings that enrich the vector — they
        are no longer a lookup table anyone has to populate correctly."""
        from services.skill_runtime import skill_embedding_text

        class _S:
            name = "gmail-actions"
            description = "Read and send mail"
            aliases = ["email", "inbox"]

        text = skill_embedding_text(_S())
        assert "gmail actions" in text
        assert "Read and send mail" in text
        assert "Also called: email, inbox" in text


class TestJargonBackstop:
    """Embeddings are weakest exactly where identifiers are strongest.

    "pending protected replacement decision for worldview" scores flat against
    every skill description — max z = 1.53, nothing stands out — because a
    general embedding model has no representation for Hexis-internal jargon.
    But it names `protected_replacement_review` almost verbatim, and a tool name
    is an identifier the system defines rather than a keyword anyone guessed.
    """

    def _skill(self, name, tools):
        class _S:
            pass
        s = _S()
        s.name = name
        s.description = ""
        s.aliases = []
        s.bound_tools = tools
        s.requires_tools = []
        return s

    def test_internal_jargon_reaches_its_skill_by_tool_name(self):
        from services.skill_runtime import _explicitly_named, _tokens

        skill = self._skill("memory-exchange", ["protected_replacement_review"])
        q = "pending protected replacement decision for worldview"
        assert _explicitly_named(skill, q, _tokens(q))

    def test_naming_the_skill_outright_always_works(self):
        from services.skill_runtime import _explicitly_named, _tokens

        skill = self._skill("council", ["run_council"])
        q = "use the council on this"
        assert _explicitly_named(skill, q, _tokens(q))

    def test_ordinary_english_does_not_trip_it(self):
        """A weak leading pair must not turn every sentence into a match."""
        from services.skill_runtime import _explicitly_named, _tokens

        # "promote to staged" → leading pair "promote to"; "to" is not a signal.
        skill = self._skill("memory-exchange", ["promote_to_staged"])
        q = "I need to promote to the team that we shipped"
        assert not _explicitly_named(skill, q, _tokens(q))

    def test_single_word_tools_are_never_a_match(self):
        from services.skill_runtime import _explicitly_named, _tokens

        skill = self._skill("shell-access", ["shell"])
        q = "she sells sea shell trinkets"
        assert not _explicitly_named(skill, q, _tokens(q))
