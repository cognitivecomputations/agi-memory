"""Regression coverage for local OpenAI-compatible init."""

from __future__ import annotations

import json
from typing import Any

import pytest

from apps import hexis_init
from apps.tui import model_catalog
from core import cli_api, init_api

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.cli]


class _WizardConn:
    def __init__(self) -> None:
        self.heartbeat: dict[str, Any] | None = None
        self.subconscious: dict[str, Any] | None = None

    async def fetchval(self, query: str, *args: Any) -> None:
        assert "init_llm_config" in query
        self.heartbeat = json.loads(args[0])
        self.subconscious = json.loads(args[1])

    async def execute(self, _query: str, *_args: Any) -> None:
        return None


class _ConfigConn:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    async def fetch(self, _query: str) -> list[dict[str, Any]]:
        return [{"key": key, "value": value} for key, value in self.config.items()]

    async def fetchval(self, query: str) -> float:
        assert "heartbeat_interval_minutes" in query
        return 60.0

    async def close(self) -> None:
        return None


async def test_loopback_endpoint_explains_docker_reachability() -> None:
    error = hexis_init._openai_compatible_endpoint_error("http://localhost:11434/v1")

    assert error is not None
    assert "workers run in Docker" in error
    assert (
        hexis_init._openai_compatible_endpoint_error("http://100.100.205.29:11434/v1")
        is None
    )


async def test_openai_compatible_wizard_discovers_saves_and_tests_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reported Ollama journey never asks the user to invent a provider id."""
    endpoint = "http://100.100.205.29:11434/v1"
    model = "qwen3.8:latest"
    conn = _WizardConn()
    tested: dict[str, Any] = {}

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def fake_select_value(
        _message: str,
        pairs: list[tuple[str, object]],
        *,
        default_value: object = None,
    ) -> str:
        del default_value
        values = [value for _label, value in pairs]
        assert set(hexis_init._PROVIDER_ENV_VARS) | {"openai_compatible"} <= set(values)
        assert "__custom__" not in values
        return "openai_compatible"

    async def fake_autocomplete(
        _message: str,
        options: list[str],
        *,
        default: str = "",
    ) -> str:
        del default
        assert model in options
        return model

    def fake_prompt(label: str, **_kwargs: Any) -> str:
        if label.startswith("OpenAI-compatible endpoint"):
            return endpoint
        if label.startswith("API key env var name"):
            return ""
        raise AssertionError(f"Unexpected prompt: {label}")

    def fake_yes_no(label: str, *, default: bool) -> bool:
        del default
        return label == "Test the connection now?"

    async def fake_fetch_models(
        provider: str,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
    ) -> list[str]:
        assert provider == "openai_compatible"
        assert endpoint == "http://100.100.205.29:11434/v1"
        assert api_key is None
        return ["another-model", model]

    async def fake_load_config(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        assert conn.heartbeat is not None
        return {
            **conn.heartbeat,
            "api_key": None,
        }

    async def fake_test_connection(config: dict[str, Any]) -> dict[str, Any]:
        tested.update(config)
        return {"ok": True, "status": "ok", "message": "Connected."}

    monkeypatch.setattr("apps.cli_prompts.select_value", fake_select_value)
    monkeypatch.setattr("apps.cli_prompts.autocomplete", fake_autocomplete)
    monkeypatch.setattr(hexis_init, "_prompt", fake_prompt)
    monkeypatch.setattr(hexis_init, "_prompt_yes_no", fake_yes_no)
    monkeypatch.setattr(model_catalog, "fetch_models", fake_fetch_models)
    monkeypatch.setattr(hexis_init, "_load_llm_config_for_consent", fake_load_config)
    monkeypatch.setattr(init_api, "test_llm_connection", fake_test_connection)

    resolved = await hexis_init._configure_llm(
        conn,
        dsn="postgresql://unused",
        wait_seconds=1,
    )

    expected = {
        "provider": "openai_compatible",
        "model": model,
        "endpoint": endpoint,
        "api_key_env": "",
    }
    assert conn.heartbeat == expected
    assert conn.subconscious == expected
    assert tested == {**expected, "api_key": None}
    assert resolved == tested


async def test_openai_compatible_model_discovery_uses_endpoint_and_optional_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str, dict[str, Any]]] = []

    async def fake_request_json(
        provider: str,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append((provider, method, url, kwargs))
        return {
            "object": "list",
            "data": [
                {"id": "qwen3.8:latest"},
                {"id": "qwen3.8:latest"},
                {"id": "nomic-embed-text:latest"},
                {"id": "llama3.1:latest"},
            ],
        }

    monkeypatch.setattr(model_catalog, "request_json", fake_request_json)

    models = await model_catalog.fetch_models(
        "openai_compatible",
        endpoint="http://ollama.test:11434/v1/",
        api_key="secret",
    )

    assert models == ["qwen3.8:latest", "llama3.1:latest"]
    assert calls == [
        (
            "openai_compatible_models",
            "GET",
            "http://ollama.test:11434/v1/models",
            {
                "headers": {"Authorization": "Bearer secret"},
                "timeout": model_catalog._TIMEOUT,
                "attempts": 2,
                "max_delay": 2.0,
            },
        )
    ]


async def test_keyless_remote_endpoint_passes_config_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm_config = {
        "provider": "openai_compatible",
        "model": "qwen3.8:latest",
        "endpoint": "http://100.100.205.29:11434/v1",
        "api_key_env": "",
    }
    conn = _ConfigConn(
        {
            "agent.is_configured": True,
            "agent.objectives": ["Be helpful"],
            "llm.heartbeat": llm_config,
            "llm.chat": llm_config,
            "llm.subconscious": llm_config,
        }
    )

    async def fake_connect(*_args: Any, **_kwargs: Any) -> _ConfigConn:
        return conn

    monkeypatch.setattr(cli_api, "_connect_with_retry", fake_connect)

    errors, warnings = await cli_api.config_validate("postgresql://unused")

    assert errors == []
    assert warnings == []
