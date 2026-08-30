"""Tests for core/auth/github_pat.py (#102): the interim GitHub credential
store for the github-issues skill, replacing a dead-end "set an env var"
message with a productized, verified, terminal-only setup flow.
"""
from __future__ import annotations

import pytest

from core.auth.github_pat import (
    GITHUB_PAT_CONFIG_KEY,
    GithubPatCredentials,
    credentials_from_value,
    credentials_to_dict,
    delete_credentials,
    load_credentials,
    resolve_github_token,
    save_credentials,
    validate_pat_format,
    verify_pat,
)

pytestmark = [pytest.mark.cli]


def test_validate_pat_format_accepts_fine_grained_and_classic():
    assert validate_pat_format("github_pat_" + "a" * 20) is None
    assert validate_pat_format("ghp_" + "a" * 20) is None


def test_validate_pat_format_rejects_empty_and_wrong_prefix():
    assert validate_pat_format("") is not None
    assert "empty" in validate_pat_format("").lower()
    assert validate_pat_format("sk-not-a-github-token") is not None


def test_credentials_round_trip_through_dict():
    creds = GithubPatCredentials(token="github_pat_abc", login="octocat")
    restored = credentials_from_value(credentials_to_dict(creds))
    assert restored == creds


def test_credentials_from_value_handles_garbage():
    assert credentials_from_value(None) is None
    assert credentials_from_value("not json") is None
    assert credentials_from_value({"login": "octocat"}) is None  # no token
    assert credentials_from_value({"token": ""}) is None


def test_save_load_delete_round_trip(tmp_path, monkeypatch):
    import core.auth.store as auth_store

    monkeypatch.setattr(auth_store, "AUTH_DIR", tmp_path / "auth")

    assert load_credentials() is None

    creds = GithubPatCredentials(token="github_pat_" + "z" * 20, login="octocat")
    save_credentials(creds)
    loaded = load_credentials()
    assert loaded == creds

    delete_credentials()
    assert load_credentials() is None


def test_resolve_github_token_never_falls_back_to_ambient_credentials(tmp_path, monkeypatch):
    """#102 Non-Goal: never silently consume `gh auth token`, `GITHUB_TOKEN`,
    or any other ambient credential just because it exists."""
    import core.auth.store as auth_store

    monkeypatch.setattr(auth_store, "AUTH_DIR", tmp_path / "auth")
    monkeypatch.setenv("GITHUB_TOKEN", "ambient-should-be-ignored")
    monkeypatch.setenv("GITHUB_PERSONAL_ACCESS_TOKEN", "also-ambient-ignored")

    token, login = resolve_github_token()
    assert token is None
    assert login == ""

    save_credentials(GithubPatCredentials(token="github_pat_stored", login="octocat"))
    token, login = resolve_github_token()
    assert token == "github_pat_stored"
    assert login == "octocat"


async def test_verify_pat_reports_401_clearly(monkeypatch):
    import urllib.error

    def fake_urlopen(request, timeout=10):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    login, error = await verify_pat("github_pat_bad")
    assert login is None
    assert "401" in error


async def test_verify_pat_returns_login_on_success(monkeypatch):
    import json as _json

    class _FakeResponse:
        def __init__(self, payload: dict) -> None:
            self._body = _json.dumps(payload).encode("utf-8")

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=10):
        assert request.get_header("Authorization") == "Bearer github_pat_good"
        return _FakeResponse({"login": "octocat"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    login, error = await verify_pat("github_pat_good")
    assert error is None
    assert login == "octocat"


def test_config_key_is_namespaced():
    assert GITHUB_PAT_CONFIG_KEY == "token.github_pat"
