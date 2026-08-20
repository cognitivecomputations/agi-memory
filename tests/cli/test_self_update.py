"""Unit tests for the `hexis upgrade` self-update helpers.

These guard the logic that decides HOW to move the hexis package (pip vs
pipx vs uv) and WHETHER a newer release exists — the two spots where a
broken decision silently re-installs the old stack.
"""

import sys
from unittest.mock import patch

import pytest

from apps import hexis_cli

pytestmark = [pytest.mark.cli]


# _is_newer: gates both upgrade nagging and the fail-loud stop, so unknown
# versions must never register as an available update.

@pytest.mark.parametrize(
    "candidate,current,expected",
    [
        ("1.0.12", "1.0.11", True),
        ("1.0.11", "1.0.11", False),
        ("1.0.11", "1.0.12", False),
        ("1.1.0", "1.0.99", True),
        ("2.0.0", "1.9.9", True),
        ("1.0.10", "1.0.9", True),  # numeric, not lexicographic
        ("1.0.13rc1", "1.0.12", False),  # unparseable -> not newer
        ("1.0.13", "dev", False),  # source checkout marker -> not newer
        ("", "1.0.11", False),
    ],
)
def test_is_newer(candidate, current, expected):
    assert hexis_cli._is_newer(candidate, current) is expected


# _installed_via: marker files at the venv root identify the tool manager.

def test_installed_via_uv(tmp_path):
    (tmp_path / "uv-receipt.toml").touch()
    with patch.object(sys, "prefix", str(tmp_path)):
        assert hexis_cli._installed_via() == "uv"


def test_installed_via_pipx(tmp_path):
    (tmp_path / "pipx_metadata.json").touch()
    with patch.object(sys, "prefix", str(tmp_path)):
        assert hexis_cli._installed_via() == "pipx"


def test_installed_via_plain_pip(tmp_path):
    with patch.object(sys, "prefix", str(tmp_path)):
        assert hexis_cli._installed_via() == "pip"


def test_self_update_hint_covers_every_installer():
    # uv and pipx must use `install --force`, not `upgrade`: their upgrade
    # commands honor the pin recorded at install time and no-op with exit 0.
    assert hexis_cli._self_update_hint("uv") == "uv tool install --force hexis"
    assert hexis_cli._self_update_hint("pipx") == "pipx install --force hexis"
    assert hexis_cli._self_update_hint("pip") == "pip install --upgrade hexis"


# _pypi_latest: any failure must read as "unknown", never as "up to date".

def test_pypi_latest_swallows_network_errors():
    with patch("urllib.request.urlopen", side_effect=OSError("offline")):
        assert hexis_cli._pypi_latest() is None
