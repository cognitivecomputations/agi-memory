"""Tests for apps.cli_auth — CLI auth subparser wiring and dispatch."""

import pytest

from apps.cli_auth import dispatch_auth_command, register_auth_subparsers

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.cli]


def test_register_auth_subparsers_creates_providers():
    """Verify all 8 provider subparsers are registered."""
    import argparse

    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--dsn")

    top = argparse.ArgumentParser()
    auth_sub = top.add_subparsers(dest="auth_command")
    register_auth_subparsers(auth_sub, parent)

    # Parse a few provider commands to verify they exist
    for provider in [
        "openai-codex",
        "anthropic",
        "chutes",
        "github",
        "github-copilot",
        "qwen-portal",
        "minimax-portal",
        "google-gemini-cli",
        "google-antigravity",
    ]:
        args = top.parse_args([provider, "status"])
        assert hasattr(args, "func"), f"{provider} status should set func"
        assert "status" in args.func, f"{provider} func should contain 'status': {args.func}"


def test_register_auth_subparsers_login_variants():
    """Verify login subcommands exist for all providers."""
    import argparse

    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--dsn")

    top = argparse.ArgumentParser()
    auth_sub = top.add_subparsers(dest="auth_command")
    register_auth_subparsers(auth_sub, parent)

    # OAuth providers have "login"
    for provider in ["openai-codex", "chutes", "github-copilot", "qwen-portal",
                     "minimax-portal", "google-gemini-cli", "google-antigravity"]:
        args = top.parse_args([provider, "login"])
        assert hasattr(args, "func"), f"{provider} login should set func"

    # Anthropic and GitHub use "setup-token" instead of "login"
    args = top.parse_args(["anthropic", "setup-token"])
    assert hasattr(args, "func")
    args = top.parse_args(["github", "setup-token"])
    assert hasattr(args, "func")


def test_register_auth_subparsers_logout():
    """Verify logout subcommands exist."""
    import argparse

    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--dsn")

    top = argparse.ArgumentParser()
    auth_sub = top.add_subparsers(dest="auth_command")
    register_auth_subparsers(auth_sub, parent)

    for provider in [
        "openai-codex", "anthropic", "chutes", "github", "github-copilot",
        "qwen-portal", "minimax-portal", "google-gemini-cli", "google-antigravity",
    ]:
        args = top.parse_args([provider, "logout"])
        assert hasattr(args, "func")
