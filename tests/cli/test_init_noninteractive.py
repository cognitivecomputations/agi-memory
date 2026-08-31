"""Tests for non-interactive hexis init mode."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from apps.hexis_init import (
    _PROVIDER_ENV_VARS,
    _resolve_noninteractive_api_key_env,
    _resolve_noninteractive_endpoint,
    _resolve_noninteractive_provider,
    _write_env_var,
    build_parser,
    detect_provider,
)

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.cli]


# ---------------------------------------------------------------------------
# detect_provider unit tests
# ---------------------------------------------------------------------------


def test_detect_provider_anthropic():
    assert detect_provider("sk-ant-abc123") == "anthropic"


def test_detect_provider_openai():
    assert detect_provider("sk-abc123def") == "openai"


def test_detect_provider_grok():
    assert detect_provider("gsk_abc123") == "grok"


def test_detect_provider_gemini():
    assert detect_provider("AIzaSyAbc123") == "gemini"


def test_detect_provider_unknown():
    with pytest.raises(ValueError, match="Cannot detect provider"):
        detect_provider("xyz-unknown-key")


def test_detect_provider_ordering():
    """sk-ant- must match anthropic, not openai."""
    assert detect_provider("sk-ant-api03-xxxx") == "anthropic"


# ---------------------------------------------------------------------------
# _write_env_var unit tests
# ---------------------------------------------------------------------------


def test_write_env_var_creates_file(tmp_path):
    env_path = tmp_path / ".env"
    _write_env_var(env_path, "MY_KEY", "my_value")
    content = env_path.read_text()
    assert "MY_KEY=my_value" in content
    assert content.endswith("\n")


def test_write_env_var_updates_existing(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("MY_KEY=old_value\nOTHER=keep\n")
    _write_env_var(env_path, "MY_KEY", "new_value")
    content = env_path.read_text()
    assert "MY_KEY=new_value" in content
    assert "old_value" not in content
    assert "OTHER=keep" in content


def test_write_env_var_appends_new(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("EXISTING=value\n")
    _write_env_var(env_path, "NEW_KEY", "new_value")
    content = env_path.read_text()
    assert "EXISTING=value" in content
    assert "NEW_KEY=new_value" in content


# ---------------------------------------------------------------------------
# build_parser tests
# ---------------------------------------------------------------------------


def test_build_parser_noninteractive_flags():
    parser = build_parser()
    args = parser.parse_args([
        "--api-key", "sk-ant-test",
        "--character", "hexis",
        "--provider", "anthropic",
        "--model", "claude-sonnet-4-20250514",
        "--endpoint", "https://llm.example/v1",
        "--name", "Alice",
        "--no-docker",
        "--no-pull",
    ])
    assert args.api_key == "sk-ant-test"
    assert args.character == "hexis"
    assert args.provider == "anthropic"
    assert args.model == "claude-sonnet-4-20250514"
    assert args.endpoint == "https://llm.example/v1"
    assert args.name == "Alice"
    assert args.no_docker is True
    assert args.no_pull is True


def test_build_parser_defaults():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.api_key is None
    assert args.api_key_env is None
    assert args.character is None
    assert args.provider is None
    assert args.model is None
    assert args.endpoint is None
    assert args.name is None
    assert args.no_docker is False
    assert args.no_pull is False


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------


def test_default_models_derive_from_live_catalog():
    """Defaults are no longer hardcoded per provider — they come from the live
    models.dev catalog via model_catalog.recommended_default (Bar #1)."""
    from apps.tui import model_catalog

    # Catalog-backed providers map to models.dev. A custom endpoint cannot:
    # its model list is defined by that server, so callers pass --model.
    for provider in set(_PROVIDER_ENV_VARS) - {"openai_compatible"}:
        assert provider in model_catalog.PROVIDER_SLUG, f"No catalog slug for {provider}"
    # The flagship heuristic picks a sensible non-variant default.
    assert model_catalog.recommended_default(
        "openai", ["gpt-5.5-pro", "gpt-5.5", "gpt-5.4-mini"]) == "gpt-5.5"


# ---------------------------------------------------------------------------
# Safe custom-endpoint configuration
# ---------------------------------------------------------------------------


def test_openai_compatible_resolves_explicit_endpoint_and_key_env(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_KEY", "test-key")
    args = build_parser().parse_args([
        "--provider", "openai_compatible",
        "--model", "served-model",
        "--endpoint", "https://llm.example/v1",
        "--api-key-env", "INFERENCE_API_KEY",
    ])

    provider = _resolve_noninteractive_provider(args)
    endpoint, source = _resolve_noninteractive_endpoint(args, provider)

    assert provider == "openai_compatible"
    assert endpoint == "https://llm.example/v1"
    assert source == "--endpoint"
    assert _resolve_noninteractive_api_key_env(args, provider) == "INFERENCE_API_KEY"


def test_openai_compatible_uses_documented_env_endpoint(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    args = build_parser().parse_args([
        "--provider", "openai_compatible",
        "--model", "served-model",
        "--api-key-env", "OPENAI_API_KEY",
    ])

    endpoint, source = _resolve_noninteractive_endpoint(args, "openai_compatible")

    assert endpoint == "https://llm.example/v1"
    assert source == "OPENAI_BASE_URL"
    assert _resolve_noninteractive_api_key_env(args, "openai_compatible") == "OPENAI_API_KEY"


def test_openai_compatible_requires_endpoint(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    args = build_parser().parse_args([
        "--provider", "openai_compatible",
        "--model", "served-model",
        "--api-key-env", "OPENAI_API_KEY",
    ])

    with pytest.raises(ValueError, match="--endpoint URL or set OPENAI_BASE_URL"):
        _resolve_noninteractive_endpoint(args, "openai_compatible")


def test_api_key_env_must_be_set(monkeypatch):
    monkeypatch.delenv("MISSING_API_KEY", raising=False)
    args = build_parser().parse_args([
        "--provider", "openai_compatible",
        "--model", "served-model",
        "--endpoint", "https://llm.example/v1",
        "--api-key-env", "MISSING_API_KEY",
    ])

    with pytest.raises(ValueError, match="MISSING_API_KEY is not set"):
        _resolve_noninteractive_api_key_env(args, "openai_compatible")


def test_ambient_key_requires_explicit_selection(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key")
    args = build_parser().parse_args([
        "--provider", "openai",
        "--model", "served-model",
    ])

    with pytest.raises(ValueError, match="--api-key-env OPENAI_API_KEY"):
        _resolve_noninteractive_api_key_env(args, "openai")


def test_standard_openai_does_not_inherit_custom_endpoint(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://unrelated.example/v1")
    args = build_parser().parse_args([
        "--provider", "openai",
        "--model", "served-model",
        "--api-key", "sk-test-key",
    ])

    endpoint, source = _resolve_noninteractive_endpoint(args, "openai")

    assert endpoint == ""
    assert source == ""


# ---------------------------------------------------------------------------
# CLI smoke tests (subprocess)
# ---------------------------------------------------------------------------

_CWD = str(Path(__file__).resolve().parents[2])


async def test_cli_init_help():
    """hexis init --help includes non-interactive flags."""
    p = subprocess.run(
        [sys.executable, "-m", "apps.hexis_init", "--help"],
        capture_output=True, text=True,
        env=os.environ.copy(),
        cwd=_CWD,
    )
    assert p.returncode == 0
    assert "--api-key" in p.stdout
    assert "--api-key-env" in p.stdout
    assert "--endpoint" in p.stdout
    assert "--character" in p.stdout
    assert "--provider" in p.stdout
    assert "--no-docker" in p.stdout
    assert "--no-pull" in p.stdout


async def test_cli_init_bad_key_prefix():
    """Unrecognised key prefix exits with error."""
    p = subprocess.run(
        [sys.executable, "-m", "apps.hexis_init",
         "--api-key", "xyz-unknown",
         "--no-docker", "--no-pull"],
        capture_output=True, text=True,
        env=os.environ.copy(),
        cwd=_CWD,
    )
    assert p.returncode != 0
    combined = p.stdout + p.stderr
    assert "Cannot detect provider" in combined or "init failed" in combined


# ---------------------------------------------------------------------------
# Embedding-unavailable handling (Experience Bar: no dead-ends)
# ---------------------------------------------------------------------------


class _StubConn:
    """Conn stub: _embedding_service_info falls back to defaults on failure."""

    async def fetchval(self, *args, **kwargs):
        raise RuntimeError("no db in this test")


async def test_embedding_step_noninteractive_fails_with_guidance(capsys):
    """A down embedding service yields sidecar guidance + a clean error, not a bare traceback."""
    from apps.hexis_init import _run_embedding_step

    async def down_step():
        raise RuntimeError(
            "Failed to get embeddings: Embedding service not available after 30 seconds: "
            "Failed to connect to host.docker.internal port 42666: Connection refused"
        )

    with pytest.raises(RuntimeError, match="start the local embedding service"):
        await _run_embedding_step(_StubConn(), down_step, interactive=False)
    err = capsys.readouterr().err
    assert "embeddinggemma" in err
    assert "embeddinggemma-metal" not in err
    assert "EMBEDDING_SERVICE_URL" in err


async def test_embedding_step_interactive_retries_then_succeeds(monkeypatch):
    """Answering yes to 'Try again?' reruns only the failed DB step."""
    import builtins

    from apps.hexis_init import _run_embedding_step

    monkeypatch.setattr(builtins, "input", lambda *a, **k: "y")
    attempts = {"n": 0}

    async def flaky_step():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("Embedding service not available after 30 seconds: refused")
        return "ok"

    assert await _run_embedding_step(_StubConn(), flaky_step) == "ok"
    assert attempts["n"] == 2


async def test_embedding_step_unrelated_errors_pass_through():
    """Only embedding-service failures get the retry flow; other errors raise as-is."""
    from apps.hexis_init import _run_embedding_step

    async def broken_step():
        raise ValueError("some other init problem")

    with pytest.raises(ValueError, match="some other init problem"):
        await _run_embedding_step(_StubConn(), broken_step, interactive=False)
