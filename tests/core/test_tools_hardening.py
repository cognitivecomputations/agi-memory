import asyncio
import sys
import types
from types import SimpleNamespace

import pytest

from core.tools.base import ToolExecutionContext, ToolContext, ToolErrorType
from core.tools.registry import ToolRegistry
from core.tools.config import ToolsConfig
from core.tools.policy import PolicyCheckResult
from core.tools.base import ToolHandler, ToolResult, ToolSpec, ToolCategory
from core.tools.web import WebFetchHandler, WebSearchHandler, WebSummarizeHandler
class DummyHandler(ToolHandler):
    def __init__(self, name: str, cost: int, supports_parallel: bool = True):
        self._spec = ToolSpec(
            name=name,
            description="dummy",
            parameters={"type": "object", "properties": {}},
            category=ToolCategory.MEMORY,
            energy_cost=cost,
            is_read_only=True,
            supports_parallel=supports_parallel,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, arguments, context):
        return ToolResult.success_result({"ok": True})


@pytest.mark.core
def test_is_path_allowed_blocks_prefix_escape(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = ToolExecutionContext(tool_context=ToolContext.CHAT, call_id="x", workspace_path=str(workspace))

    assert ctx.is_path_allowed(str(workspace / "file.txt")) is True

    sneaky = tmp_path / "workspace_evil" / "file.txt"
    sneaky.parent.mkdir()
    assert ctx.is_path_allowed(str(sneaky)) is False


@pytest.mark.core
def test_is_path_allowed_blocks_symlink_escape(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")

    link = workspace / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")

    ctx = ToolExecutionContext(tool_context=ToolContext.CHAT, call_id="x", workspace_path=str(workspace))
    assert ctx.is_path_allowed(str(link / "secret.txt")) is False


@pytest.mark.core
def test_is_path_allowed_allows_nonexistent_within_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = ToolExecutionContext(tool_context=ToolContext.CHAT, call_id="x", workspace_path=str(workspace))

    assert ctx.is_path_allowed(str(workspace / "new.txt")) is True


@pytest.mark.core
def test_web_url_private_ip_validation():
    handler = WebFetchHandler()
    errs = handler.validate({"url": "http://172.16.0.1/test"})
    assert any("internal" in e.lower() for e in errs)

    errs = handler.validate({"url": "http://127.0.0.1/test"})
    assert any("localhost" in e.lower() or "internal" in e.lower() for e in errs)

    errs = handler.validate({"url": "http://172.0.0.1/test"})
    assert not any("internal" in e.lower() for e in errs)


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.core
async def test_web_search_uses_keyless_fallback_without_tavily(monkeypatch):
    class FakeResponse:
        status_code = 200
        headers: dict = {}
        text = """
        <html><body>
          <a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fprofile&amp;rut=x">
            Example Result
          </a>
          <td class="result-snippet">Useful search snippet.</td>
        </body></html>
        """
        content = text.encode("utf-8")

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def request(self, method, url, *, params=None, **_kwargs):
            assert method == "GET"
            assert url == "https://lite.duckduckgo.com/lite/"
            assert params == {"q": "example query"}
            return FakeResponse()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    handler = WebSearchHandler(api_key_resolver=lambda: None)
    context = ToolExecutionContext(tool_context=ToolContext.CHAT, call_id="x")

    result = await handler.execute({"query": "example query", "max_results": 3}, context)

    assert result.success is True
    assert result.output["provider"] == {
        "id": "duckduckgo_lite",
        "label": "DuckDuckGo Lite",
    }
    assert result.output["external_content"] == {
        "untrusted": True,
        "source": "web_search",
        "provider": "duckduckgo_lite",
    }
    assert result.output["results"] == [
        {
            "title": "Example Result",
            "url": "https://example.com/profile",
            "snippet": "Useful search snippet.",
            "score": None,
            "site_name": "example.com",
        }
    ]


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.core
async def test_web_search_falls_back_to_bing_when_duckduckgo_is_empty(monkeypatch):
    class FakeResponse:
        def __init__(self, text):
            self.status_code = 200
            self.headers: dict = {}
            self.text = text
            self.content = text.encode("utf-8")

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def request(self, method, url, *, params=None, **_kwargs):
            assert method == "GET"
            if url == "https://lite.duckduckgo.com/lite/":
                assert params == {"q": "example query"}
                return FakeResponse("<html><body>no results</body></html>")
            assert url == "https://www.bing.com/search"
            assert params == {"q": "example query", "format": "rss"}
            return FakeResponse(
                """<?xml version="1.0" encoding="utf-8" ?>
                <rss version="2.0"><channel>
                  <item>
                    <title>Fallback Result</title>
                    <link>https://example.net/profile</link>
                    <description>Fallback snippet.</description>
                  </item>
                </channel></rss>"""
            )

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    handler = WebSearchHandler(api_key_resolver=lambda: None)
    context = ToolExecutionContext(tool_context=ToolContext.CHAT, call_id="x")

    result = await handler.execute({"query": "example query", "max_results": 3}, context)

    assert result.success is True
    assert result.output["provider"]["id"] == "bing_rss"
    assert result.output["provider_errors"] == [
        "DuckDuckGo Lite: DuckDuckGo Lite returned no parseable results."
    ]
    assert result.output["results"][0]["title"] == "Fallback Result"


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.core
async def test_web_search_explicit_provider_does_not_silently_fallback(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    class FakeRegistry:
        async def get_config(self):
            return ToolsConfig(web_search={"provider": "tavily"})

    handler = WebSearchHandler(api_key_resolver=lambda: None)
    context = ToolExecutionContext(tool_context=ToolContext.CHAT, call_id="x")
    context.registry = FakeRegistry()

    result = await handler.execute({"query": "example query"}, context)

    assert result.success is False
    assert result.error_type == ToolErrorType.MISSING_CONFIG
    assert "TAVILY_API_KEY" in (result.error or "")


@pytest.mark.core
def test_tools_config_preserves_web_search_provider():
    config = ToolsConfig(web_search={"provider": "searxng", "searxng_url": "http://localhost:8080"})
    restored = ToolsConfig.from_json(config.to_json())

    assert restored.web_search == {
        "provider": "searxng",
        "searxng_url": "http://localhost:8080",
    }


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.core
async def test_execute_batch_parallel_energy_budget(monkeypatch):
    registry = ToolRegistry(pool=object())
    registry.register(DummyHandler("tool_a", cost=3))
    registry.register(DummyHandler("tool_b", cost=3))

    async def _allow(*args, **kwargs):
        return PolicyCheckResult.allow()

    async def _config():
        return ToolsConfig()

    monkeypatch.setattr(registry._policy, "check_all", _allow)
    monkeypatch.setattr(registry, "get_config", _config)

    context = ToolExecutionContext(
        tool_context=ToolContext.HEARTBEAT,
        call_id="x",
        energy_available=3,
    )

    results = await registry.execute_batch(
        [("tool_a", {}), ("tool_b", {})],
        context,
        parallel=True,
    )

    assert results[0].success is True
    assert results[1].success is False
    assert results[1].error_type == ToolErrorType.INSUFFICIENT_ENERGY


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.core
async def test_registry_propagates_turn_cancellation(monkeypatch):
    handler = DummyHandler("blocking_tool", cost=0, supports_parallel=False)
    handler._spec.execution_timeout_seconds = None

    async def _block(_arguments, _context):
        await asyncio.Event().wait()
        return ToolResult.success_result({"ok": True})

    monkeypatch.setattr(handler, "execute", _block)
    registry = ToolRegistry(pool=object())
    registry.register(handler)

    async def _allow(*args, **kwargs):
        return PolicyCheckResult.allow()

    async def _config():
        return ToolsConfig()

    monkeypatch.setattr(registry._policy, "check_all", _allow)
    monkeypatch.setattr(registry, "get_config", _config)
    task = asyncio.create_task(
        registry.execute(
            "blocking_tool",
            {},
            ToolExecutionContext(tool_context=ToolContext.CHAT, call_id="cancel"),
        )
    )
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.core
async def test_web_summarize_uses_direct_llm(monkeypatch):
    """web_summarize summarizes via a single in-process LLM call (the broken
    external_calls queue+poll path is gone)."""
    dummy_trafilatura = types.SimpleNamespace(
        fetch_url=lambda url: "<html></html>",
        extract=lambda downloaded, include_tables=True: "content",
        extract_metadata=lambda downloaded: types.SimpleNamespace(title="Title"),
    )
    monkeypatch.setitem(sys.modules, "trafilatura", dummy_trafilatura)

    async def _fake_load_llm_config(pool, **kwargs):
        return {"provider": "openai", "model": "gpt-4o", "endpoint": None, "api_key": "t"}

    captured: dict = {}

    async def _fake_chat_completion(**kwargs):
        captured.update(kwargs)
        return {"content": "A concise summary.", "raw": None}

    monkeypatch.setattr("core.llm_config.load_llm_config", _fake_load_llm_config)
    monkeypatch.setattr("core.llm.chat_completion", _fake_chat_completion)

    context = ToolExecutionContext(tool_context=ToolContext.CHAT, call_id="x")
    context.registry = SimpleNamespace(pool=SimpleNamespace())  # only handed to load_llm_config

    handler = WebSummarizeHandler()
    result = await handler.execute({"url": "http://example.com"}, context)

    assert result.success is True
    assert result.output["summary"] == "A concise summary."
    assert captured["max_tokens"] == 500
    assert captured["messages"][0]["role"] == "user"
