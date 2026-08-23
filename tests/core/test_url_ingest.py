"""Tests for URLIngestHandler and multi-extractor pipeline (url_ingest tool)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.tools.base import ToolCategory, ToolContext, ToolErrorType, ToolExecutionContext
from core.tools.ingest import (
    URLIngestHandler,
    _detect_url_source_type,
    create_ingest_tools,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context():
    registry = MagicMock()
    registry.pool = MagicMock()
    return ToolExecutionContext(
        tool_context=ToolContext.CHAT,
        call_id="test-call",
        registry=registry,
    )


# ---------------------------------------------------------------------------
# Spec tests
# ---------------------------------------------------------------------------

class TestURLIngestSpec:
    def test_spec_name(self):
        spec = URLIngestHandler().spec
        assert spec.name == "url_ingest"

    def test_spec_category(self):
        spec = URLIngestHandler().spec
        assert spec.category == ToolCategory.INGEST

    def test_spec_energy(self):
        spec = URLIngestHandler().spec
        assert spec.energy_cost == 3

    def test_spec_not_read_only(self):
        spec = URLIngestHandler().spec
        assert spec.is_read_only is False

    def test_spec_required_params(self):
        spec = URLIngestHandler().spec
        assert "url" in spec.parameters["required"]

    def test_spec_has_mode(self):
        spec = URLIngestHandler().spec
        props = spec.parameters["properties"]
        assert "mode" in props
        assert props["mode"]["enum"] == ["fast", "slow", "hybrid"]

    def test_spec_has_title(self):
        spec = URLIngestHandler().spec
        assert "title" in spec.parameters["properties"]

    def test_spec_description_mentions_source_types(self):
        spec = URLIngestHandler().spec
        desc = spec.description.lower()
        assert "pdf" in desc
        assert "youtube" in desc
        assert "rss" in desc


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestURLIngestValidation:
    def test_valid_url(self):
        handler = URLIngestHandler()
        errors = handler.validate({"url": "https://example.com/article"})
        assert errors == []

    def test_empty_url(self):
        handler = URLIngestHandler()
        errors = handler.validate({"url": ""})
        assert any("empty" in e.lower() for e in errors)

    def test_missing_url(self):
        handler = URLIngestHandler()
        errors = handler.validate({})
        assert any("empty" in e.lower() for e in errors)

    def test_no_scheme(self):
        handler = URLIngestHandler()
        errors = handler.validate({"url": "example.com"})
        assert any("http" in e.lower() for e in errors)

    def test_localhost_blocked(self):
        handler = URLIngestHandler()
        errors = handler.validate({"url": "http://localhost:8080/secret"})
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# URL source type detection tests
# ---------------------------------------------------------------------------

class TestDetectURLSourceType:
    def test_youtube_watch_url(self):
        assert _detect_url_source_type("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "youtube"

    def test_youtube_short_url(self):
        assert _detect_url_source_type("https://youtu.be/dQw4w9WgXcQ") == "youtube"

    def test_youtube_with_params(self):
        assert _detect_url_source_type("https://youtube.com/watch?v=abc123def45&t=120") == "youtube"

    def test_pdf_url(self):
        assert _detect_url_source_type("https://example.com/paper.pdf") == "pdf"

    def test_pdf_url_case_insensitive(self):
        assert _detect_url_source_type("https://example.com/paper.PDF") == "pdf"

    def test_arxiv_pdf_path(self):
        assert _detect_url_source_type("https://arxiv.org/pdf/2509.25149") == "pdf"

    def test_rss_url(self):
        assert _detect_url_source_type("https://example.com/feed.rss") == "rss"

    def test_atom_url(self):
        assert _detect_url_source_type("https://example.com/feed.atom") == "rss"

    def test_xml_url(self):
        assert _detect_url_source_type("https://example.com/rss.xml") == "rss"

    def test_regular_web_url(self):
        assert _detect_url_source_type("https://example.com/blog/article") == "web"

    def test_web_url_with_query(self):
        assert _detect_url_source_type("https://docs.python.org/3/library/asyncio.html") == "web"


# ---------------------------------------------------------------------------
# YouTube transcript reader tests
# ---------------------------------------------------------------------------

class TestYouTubeTranscriptReader:
    def test_extract_video_id_watch(self):
        from services.ingest import YouTubeTranscriptReader
        assert YouTubeTranscriptReader.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_extract_video_id_short(self):
        from services.ingest import YouTubeTranscriptReader
        assert YouTubeTranscriptReader.extract_video_id("https://youtu.be/abc123def45") == "abc123def45"

    def test_extract_video_id_none(self):
        from services.ingest import YouTubeTranscriptReader
        assert YouTubeTranscriptReader.extract_video_id("https://example.com") is None

    def test_can_handle_youtube(self):
        from services.ingest import YouTubeTranscriptReader
        assert YouTubeTranscriptReader.can_handle("https://www.youtube.com/watch?v=xyz12345678") is True

    def test_cannot_handle_other(self):
        from services.ingest import YouTubeTranscriptReader
        assert YouTubeTranscriptReader.can_handle("https://example.com/article") is False


# ---------------------------------------------------------------------------
# Execution tests (mocked)
# ---------------------------------------------------------------------------

class TestURLIngestExecution:
    class _FakeAcquire:
        def __init__(self, conn):
            self.conn = conn

        async def __aenter__(self):
            return self.conn

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakePool:
        def __init__(self, conn):
            self.conn = conn

        def acquire(self):
            return TestURLIngestExecution._FakeAcquire(self.conn)

    class _FakeConn:
        def __init__(self, job_id="11111111-1111-4111-8111-111111111111", exc=None):
            self.job_id = job_id
            self.exc = exc
            self.calls = []

        async def fetchval(self, query, *args):
            self.calls.append((query, args))
            if self.exc is not None:
                raise self.exc
            return self.job_id

    @pytest.mark.asyncio
    async def test_enqueue_failure(self):
        handler = URLIngestHandler()
        conn = self._FakeConn(exc=RuntimeError("database unavailable"))
        ctx = _make_context()
        ctx.registry.pool = self._FakePool(conn)

        result = await handler.execute({"url": "https://example.com/article"}, ctx)

        assert not result.success
        assert result.error_type == ToolErrorType.EXECUTION_FAILED
        assert "could not be queued" in result.error

    @pytest.mark.asyncio
    async def test_successful_enqueue(self):
        handler = URLIngestHandler()
        conn = self._FakeConn()
        ctx = _make_context()
        ctx.registry.pool = self._FakePool(conn)

        result = await handler.execute(
            {
                "url": "https://example.com/article",
                "mode": "fast",
                "title": "Example Article",
                "keep_reason": "Useful technical background",
            },
            ctx,
        )

        assert result.success
        assert result.output["accepted"] is True
        assert result.output["job_id"] == "11111111-1111-4111-8111-111111111111"
        assert result.output["url"] == "https://example.com/article"
        assert result.output["mode"] == "fast"
        assert result.output["source_type"] == "web"
        assert "Queued URL ingestion" in result.display_output
        assert len(conn.calls) == 1
        query, args = conn.calls[0]
        assert "enqueue_ingestion_job('url'" in query
        import json

        payload = json.loads(args[0])
        assert payload["title"] == "Example Article"
        # Sensitivity marking was removed: discretion is the agent's judgment at
        # speaking time, informed by who it is talking to, not a stored flag.
        assert "sensitivity" not in payload
        assert payload["acquired_reason"] == "Useful technical background"
        assert args[1].startswith("url:")

    @pytest.mark.asyncio
    async def test_youtube_source_type_in_output(self):
        handler = URLIngestHandler()
        conn = self._FakeConn()
        ctx = _make_context()
        ctx.registry.pool = self._FakePool(conn)

        result = await handler.execute(
            {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}, ctx
        )

        assert result.success
        assert result.output["source_type"] == "youtube"


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestFactory:
    def test_url_ingest_in_factory(self):
        tools = create_ingest_tools()
        names = {t.spec.name for t in tools}
        assert "url_ingest" in names

    def test_factory_count(self):
        tools = create_ingest_tools()
        assert len(tools) == 5  # fast, slow, hybrid, git, url
