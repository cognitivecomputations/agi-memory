"""The agent is always told who it is speaking with.

Discretion is judgment, not a stored flag: a person decides what to repeat at the
moment of speaking, from who is listening. That decision is impossible without
knowing who is listening, so the block is rendered on every chat turn — including
when the honest answer is "someone unidentified."
"""

from __future__ import annotations

from services.agent import render_interlocutor_block


def test_cli_and_dashboard_are_the_primary_user():
    for surface in ("chat", "cli", "api", "ui", "repl"):
        block = render_interlocutor_block(interlocutor="Eric", surface=surface, is_group=False)
        assert "Eric" in block
        assert "authority over you" in block
        # No warning to withhold: everything the agent knows is already theirs.
        assert "not yours to repeat" not in block


def test_unnamed_primary_user_still_reads_as_the_principal():
    block = render_interlocutor_block(interlocutor=None, surface="api", is_group=False)
    assert "your primary user" in block
    assert "not yours to repeat" not in block


def test_a_named_third_party_gets_the_discretion_instruction():
    block = render_interlocutor_block(interlocutor="Sarah Chen", surface="slack", is_group=False)
    assert "Sarah Chen" in block
    assert "**This is not your primary user.**" in block
    assert "not yours to repeat" in block
    assert "it is not" in block  # when unsure, it is not yours to share


def test_an_unidentified_sender_is_never_assumed_to_be_the_principal():
    block = render_interlocutor_block(interlocutor=None, surface="telegram", is_group=False)
    assert "someone you have not identified" in block
    assert "Do not assume it is your primary user" in block


def test_a_group_says_who_else_can_read_it():
    block = render_interlocutor_block(interlocutor="Sarah Chen", surface="slack", is_group=True)
    assert "group conversation" in block
    assert "Others can read everything you say" in block


def test_a_group_on_a_primary_surface_is_not_treated_as_private():
    # Surface alone does not confer principal status; a shared room never does.
    block = render_interlocutor_block(interlocutor=None, surface="api", is_group=True)
    assert "**This is not your primary user.**" in block
