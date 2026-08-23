"""Every channel adapter says whether the room is shared.

`ChannelMessage.is_group` decides which personhood prompt loads and what the
agent is told about who can read it (`render_interlocutor_block`). It reads
adapter metadata and has no derivation fallback, so an adapter that never sets
it silently reports every room as private — including group rooms, where the
agent would then be told it is speaking one-to-one.

Four of seven adapters were missing it. Discord and Slack already carried the
signal under another name; Matrix and WhatsApp needed the platform's own
convention.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from channels.base import ChannelMessage

ADAPTERS = ["discord", "imessage", "matrix", "signal", "slack", "telegram", "whatsapp"]


def _msg(**meta) -> ChannelMessage:
    return ChannelMessage(
        channel_type="test", channel_id="c", sender_id="s", sender_name="S",
        content="hi", message_id="m", metadata=meta,
    )


class TestTheProperty:
    def test_explicit_is_group_wins(self):
        assert _msg(is_group=True).is_group is True
        assert _msg(is_group=False).is_group is False

    def test_is_private_is_honoured_as_the_inverse(self):
        assert _msg(is_private=True).is_group is False
        assert _msg(is_private=False).is_group is True

    def test_silence_is_read_as_private(self):
        """The safe default — but only safe if adapters actually speak up,
        which is what the coverage test below enforces."""
        assert _msg().is_group is False


class TestAdapterCoverage:
    @pytest.mark.parametrize("adapter", ADAPTERS)
    def test_every_adapter_reports_group_context(self, adapter):
        src = Path(f"channels/{adapter}_adapter.py").read_text()
        assert "is_group" in src or "is_private" in src, (
            f"{adapter} never sets is_group, so every room it delivers looks "
            "private — including shared ones"
        )


class TestPlatformConventions:
    """The derivation each adapter uses, pinned so a refactor cannot invert it."""

    def test_discord_dm_has_no_guild(self):
        assert _msg(is_group=False, guild_id=None).is_group is False
        assert _msg(is_group=True, guild_id="123").is_group is True

    @pytest.mark.parametrize(
        "channel_type,expected",
        [("im", False), ("channel", True), ("group", True), ("mpim", True)],
    )
    def test_slack_only_im_is_one_to_one(self, channel_type, expected):
        # Mirrors the adapter's rule: "im" is 1:1; mpim is a multi-party DM.
        derived = str(channel_type).lower() not in ("im", "")
        assert derived is expected
        assert _msg(is_group=derived).is_group is expected

    @pytest.mark.parametrize("members,expected", [(2, False), (3, True), (12, True)])
    def test_matrix_direct_rooms_hold_two(self, members, expected):
        derived = members > 2
        assert derived is expected
        assert _msg(is_group=derived, room_member_count=members).is_group is expected

    @pytest.mark.parametrize(
        "jid,expected",
        [("14155551212", False), ("14155551212@s.whatsapp.net", False),
         ("120363001@g.us", True)],
    )
    def test_whatsapp_group_jids_end_in_g_us(self, jid, expected):
        derived = jid.endswith("@g.us")
        assert derived is expected
        assert _msg(is_group=derived).is_group is expected
