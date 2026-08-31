"""Tests for core.tools.ingest -- fast, slow, hybrid ingestion tools."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.tools.base import ToolCategory, ToolContext, ToolExecutionContext


# ---------------------------------------------------------------------------
# FastIngestHandler
# ---------------------------------------------------------------------------

class TestFastIngestHandler:
    """Unit tests for FastIngestHandler."""

    def test_spec_properties(self):
        from core.tools.ingest import FastIngestHandler

        handler = FastIngestHandler()
        spec = handler.spec
        assert spec.name == "fast_ingest"
        assert spec.category == ToolCategory.INGEST
        assert spec.energy_cost == 2
        assert ToolContext.HEARTBEAT in spec.allowed_contexts
        assert ToolContext.CHAT in spec.allowed_contexts
        assert ToolContext.MCP in spec.allowed_contexts
        assert spec.is_read_only is False

    def test_validate_missing_path(self):
        from core.tools.ingest import FastIngestHandler

        handler = FastIngestHandler()
        errors = handler.validate({})
        assert any("path" in e.lower() for e in errors)

    def test_validate_empty_path(self):
        from core.tools.ingest import FastIngestHandler

        handler = FastIngestHandler()
        errors = handler.validate({"path": ""})
        assert any("path" in e.lower() for e in errors)

    def test_validate_valid_path(self):
        from core.tools.ingest import FastIngestHandler

        handler = FastIngestHandler()
        errors = handler.validate({"path": "/tmp/test.md"})
        assert errors == []


# ---------------------------------------------------------------------------
# SlowIngestHandler
# ---------------------------------------------------------------------------

class TestSlowIngestHandler:
    """Unit tests for SlowIngestHandler."""

    def test_spec_properties(self):
        from core.tools.ingest import SlowIngestHandler

        handler = SlowIngestHandler()
        spec = handler.spec
        assert spec.name == "slow_ingest"
        assert spec.category == ToolCategory.INGEST
        assert spec.energy_cost == 5
        assert ToolContext.HEARTBEAT in spec.allowed_contexts
        assert ToolContext.CHAT in spec.allowed_contexts
        assert ToolContext.MCP in spec.allowed_contexts
        assert spec.is_read_only is False

    def test_validate_missing_path(self):
        from core.tools.ingest import SlowIngestHandler

        handler = SlowIngestHandler()
        errors = handler.validate({})
        assert any("path" in e.lower() for e in errors)

    def test_validate_valid_path(self):
        from core.tools.ingest import SlowIngestHandler

        handler = SlowIngestHandler()
        errors = handler.validate({"path": "/tmp/test.md"})
        assert errors == []


# ---------------------------------------------------------------------------
# HybridIngestHandler
# ---------------------------------------------------------------------------

class TestHybridIngestHandler:
    """Unit tests for HybridIngestHandler."""

    def test_spec_properties(self):
        from core.tools.ingest import HybridIngestHandler

        handler = HybridIngestHandler()
        spec = handler.spec
        assert spec.name == "hybrid_ingest"
        assert spec.category == ToolCategory.INGEST
        assert spec.energy_cost == 3
        assert ToolContext.HEARTBEAT in spec.allowed_contexts
        assert ToolContext.CHAT in spec.allowed_contexts
        assert ToolContext.MCP in spec.allowed_contexts
        assert spec.is_read_only is False

    def test_validate_missing_path(self):
        from core.tools.ingest import HybridIngestHandler

        handler = HybridIngestHandler()
        errors = handler.validate({})
        assert any("path" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# URLIngestHandler
# ---------------------------------------------------------------------------

class TestURLIngestHandler:
    """Unit tests for URLIngestHandler."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_queues_url_ingestion_with_url_as_source_path(self):
        from core.tools.ingest import URLIngestHandler

        url = "https://example.com/with/specification"

        class FakeAcquire:
            def __init__(self, conn):
                self.conn = conn

            async def __aenter__(self):
                return self.conn

            async def __aexit__(self, exc_type, exc, tb):
                return False

        class FakeConn:
            def __init__(self):
                self.calls = []

            async def fetchval(self, query, *args):
                self.calls.append((query, args))
                return "11111111-1111-4111-8111-111111111111"

        class FakePool:
            def __init__(self, conn):
                self.conn = conn

            def acquire(self):
                return FakeAcquire(self.conn)

        conn = FakeConn()
        registry = MagicMock()
        registry.pool = FakePool(conn)
        context = ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id="url-ingest-test",
            registry=registry,
        )

        result = await URLIngestHandler().execute({"url": url, "mode": "fast"}, context)

        assert result.success, result.error
        assert result.output["accepted"] is True
        assert result.output["url"] == url
        assert len(conn.calls) == 1

        payload = json.loads(conn.calls[0][1][0])
        assert payload["url"] == url
        assert payload["mode"] == "fast"
        assert payload["source_type"] == "web"
        assert payload["acquisition"] == "user"


# ---------------------------------------------------------------------------
# GitIngestHandler
# ---------------------------------------------------------------------------

class TestGitIngestHandler:
    """Unit tests for GitIngestHandler."""

    def test_spec_properties(self):
        from core.tools.ingest import GitIngestHandler

        handler = GitIngestHandler()
        spec = handler.spec
        assert spec.name == "git_ingest"
        assert spec.category == ToolCategory.INGEST
        assert spec.energy_cost == 4
        assert ToolContext.HEARTBEAT in spec.allowed_contexts
        assert ToolContext.CHAT in spec.allowed_contexts
        assert ToolContext.MCP in spec.allowed_contexts
        assert spec.is_read_only is False

    def test_validate_missing_url(self):
        from core.tools.ingest import GitIngestHandler

        handler = GitIngestHandler()
        errors = handler.validate({})
        assert any("url" in e.lower() for e in errors)

    def test_validate_empty_url(self):
        from core.tools.ingest import GitIngestHandler

        handler = GitIngestHandler()
        errors = handler.validate({"url": ""})
        assert any("url" in e.lower() for e in errors)

    def test_validate_valid_url(self):
        from core.tools.ingest import GitIngestHandler

        handler = GitIngestHandler()
        errors = handler.validate({"url": "https://github.com/owner/repo"})
        assert errors == []

    def test_validate_shorthand(self):
        from core.tools.ingest import GitIngestHandler

        handler = GitIngestHandler()
        errors = handler.validate({"url": "owner/repo"})
        assert errors == []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestCreateIngestTools:
    """Test the create_ingest_tools factory."""

    def test_returns_all_handlers(self):
        from core.tools.ingest import create_ingest_tools

        tools = create_ingest_tools()
        assert len(tools) == 5
        names = {t.spec.name for t in tools}
        assert names == {"fast_ingest", "slow_ingest", "hybrid_ingest", "git_ingest", "url_ingest"}

    def test_all_have_ingest_category(self):
        from core.tools.ingest import create_ingest_tools

        tools = create_ingest_tools()
        for t in tools:
            assert t.spec.category == ToolCategory.INGEST


# ---------------------------------------------------------------------------
# _build_ingest_config resolution (requires running DB)
# ---------------------------------------------------------------------------

class TestBuildIngestConfig:
    """Verify _build_ingest_config resolves LLM config from DB."""

    async def test_resolves_provider_from_db(self, db_pool):
        from core.tools.ingest import _build_ingest_config
        from services.ingest import IngestionMode

        config = await _build_ingest_config(db_pool, mode=IngestionMode.FAST)
        assert config.llm_config is not None
        assert "provider" in config.llm_config
        assert "model" in config.llm_config
        assert config.dsn is not None

    async def test_overrides_applied(self, db_pool):
        from core.tools.ingest import _build_ingest_config
        from services.ingest import IngestionMode

        config = await _build_ingest_config(
            db_pool, mode=IngestionMode.FAST, max_section_chars=5000
        )
        assert config.max_section_chars == 5000


class TestResolveLlmConfig:
    """Verify resolve_llm_config works with pools and connections."""

    async def test_with_pool(self, db_pool):
        from core.llm_config import resolve_llm_config

        cfg = await resolve_llm_config(db_pool, "llm.chat", fallback_key="llm")
        assert "provider" in cfg
        assert "model" in cfg

    async def test_with_connection(self, db_pool):
        from core.llm_config import resolve_llm_config

        async with db_pool.acquire() as conn:
            cfg = await resolve_llm_config(conn, "llm.chat", fallback_key="llm")
        assert "provider" in cfg
        assert "model" in cfg

    async def test_overrides_merged(self, db_pool):
        from core.llm_config import resolve_llm_config

        cfg = await resolve_llm_config(
            db_pool, "llm.chat", fallback_key="llm",
            overrides={"model": "custom-model"},
        )
        assert cfg["model"] == "custom-model"


# ---------------------------------------------------------------------------
# Slow ingest assessment parsing
# ---------------------------------------------------------------------------

class TestSlowIngestAssessment:
    """Test assessment parsing and safe defaults."""

    def test_safe_assessment_valid(self):
        from services.slow_ingest_rlm import _safe_assessment

        raw = {
            "acceptance": "accept",
            "analysis": "Good content.",
            "emotional_reaction": {"valence": 0.5, "arousal": 0.3, "primary_emotion": "curious"},
            "worldview_impact": "extends",
            "importance": 0.8,
            "trust_assessment": 0.9,
            "extracted_facts": ["Fact one", "Fact two"],
            "connections": ["abc-123"],
            "rejection_reasons": [],
        }
        result = _safe_assessment(raw)
        assert result["acceptance"] == "accept"
        assert result["importance"] == 0.8
        assert result["trust_assessment"] == 0.9
        assert len(result["extracted_facts"]) == 2

    def test_safe_assessment_bad_acceptance(self):
        from services.slow_ingest_rlm import _safe_assessment

        raw = {"acceptance": "invalid_value"}
        result = _safe_assessment(raw)
        assert result["acceptance"] == "question"

    def test_safe_assessment_not_dict(self):
        from services.slow_ingest_rlm import _safe_assessment

        result = _safe_assessment("not a dict")
        assert result["acceptance"] == "question"
        assert result["importance"] == 0.5

    def test_safe_assessment_clamps_values(self):
        from services.slow_ingest_rlm import _safe_assessment

        raw = {"importance": 5.0, "trust_assessment": -1.0}
        result = _safe_assessment(raw)
        assert result["importance"] == 1.0
        assert result["trust_assessment"] == 0.0

    def test_safe_assessment_missing_keys(self):
        from services.slow_ingest_rlm import _safe_assessment

        result = _safe_assessment({})
        assert "acceptance" in result
        assert "analysis" in result
        assert "emotional_reaction" in result
        assert "extracted_facts" in result
        assert "rejection_reasons" in result

    @pytest.mark.asyncio(loop_scope="session")
    async def test_trust_multipliers_retired(self, db_pool):
        """#83 sources-are-authority: the acceptance multiplier is gone —
        stance lives in edges/metadata; trust derives from sources."""
        async with db_pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT get_config('memory.slow_ingest_trust_multipliers')"
            )
        assert raw is None


# ---------------------------------------------------------------------------
# IngestionMode enum extension
# ---------------------------------------------------------------------------

class TestIngestionModeEnum:
    """Verify ingestion modes exist."""

    def test_fast_mode_exists(self):
        from services.ingest import IngestionMode

        assert IngestionMode.FAST.value == "fast"

    def test_slow_mode_exists(self):
        from services.ingest import IngestionMode

        assert IngestionMode.SLOW.value == "slow"

    def test_hybrid_mode_exists(self):
        from services.ingest import IngestionMode

        assert IngestionMode.HYBRID.value == "hybrid"

    def test_normalize_fast(self):
        from services.ingest import _normalize_mode, IngestionMode

        assert _normalize_mode("fast") == IngestionMode.FAST

    def test_normalize_legacy_auto(self):
        from services.ingest import _normalize_mode, IngestionMode

        assert _normalize_mode("auto") == IngestionMode.FAST

    def test_normalize_legacy_standard(self):
        from services.ingest import _normalize_mode, IngestionMode

        assert _normalize_mode("standard") == IngestionMode.FAST

    def test_normalize_legacy_deep(self):
        from services.ingest import _normalize_mode, IngestionMode

        assert _normalize_mode("deep") == IngestionMode.FAST

    def test_normalize_legacy_shallow(self):
        from services.ingest import _normalize_mode, IngestionMode

        assert _normalize_mode("shallow") == IngestionMode.FAST

    def test_normalize_legacy_archive(self):
        from services.ingest import _normalize_mode, IngestionMode

        assert _normalize_mode("archive") == IngestionMode.FAST

    def test_normalize_slow(self):
        from services.ingest import _normalize_mode, IngestionMode

        assert _normalize_mode("slow") == IngestionMode.SLOW

    def test_normalize_hybrid(self):
        from services.ingest import _normalize_mode, IngestionMode

        assert _normalize_mode("hybrid") == IngestionMode.HYBRID

    def test_normalize_none_defaults_fast(self):
        from services.ingest import _normalize_mode, IngestionMode

        assert _normalize_mode(None) == IngestionMode.FAST


# ---------------------------------------------------------------------------
# ToolCategory enum extension
# ---------------------------------------------------------------------------

class TestToolCategoryEnum:
    """Verify INGEST category exists."""

    def test_ingest_category_exists(self):
        assert ToolCategory.INGEST.value == "ingest"


# ---------------------------------------------------------------------------
# DB config keys (requires running DB)
# ---------------------------------------------------------------------------

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


class TestIngestConfigKeys:
    """Verify ingest energy costs are configured in DB."""

    async def test_fast_ingest_cost(self, db_pool):
        async with db_pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT get_config_float('heartbeat.cost_fast_ingest')"
            )
            assert result == 2.0

    async def test_slow_ingest_cost(self, db_pool):
        async with db_pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT get_config_float('heartbeat.cost_slow_ingest')"
            )
            assert result == 5.0

    async def test_hybrid_ingest_cost(self, db_pool):
        async with db_pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT get_config_float('heartbeat.cost_hybrid_ingest')"
            )
            assert result == 3.0

    async def test_retired_ingest_actions_are_not_heartbeat_actions(self, db_pool):
        async with db_pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT get_config('heartbeat.allowed_actions')"
            )
            actions = json.loads(result) if isinstance(result, str) else result
            assert {"fast_ingest", "slow_ingest", "hybrid_ingest"}.isdisjoint(actions)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

class TestSlowIngestPrompt:
    """Verify the slow ingest prompt loads correctly."""

    def test_prompt_loads(self):
        from services.prompt_resources import load_rlm_slow_ingest_prompt

        prompt = load_rlm_slow_ingest_prompt()
        assert "conscious reading" in prompt.lower() or "REPL" in prompt



# ---------------------------------------------------------------------------
# Reader coverage
# ---------------------------------------------------------------------------

class TestReaderExtensionCoverage:
    """Verify get_reader() returns the correct reader for new extensions."""

    def test_docx_reader(self):
        from services.ingest import DocxReader, get_reader

        reader = get_reader(Path("test.docx"))
        assert isinstance(reader, DocxReader)

    def test_rtf_reader(self):
        from services.ingest import RtfReader, get_reader

        reader = get_reader(Path("test.rtf"))
        assert isinstance(reader, RtfReader)

    def test_tex_reader(self):
        from services.ingest import LatexReader, get_reader

        reader = get_reader(Path("test.tex"))
        assert isinstance(reader, LatexReader)

    def test_bib_reader(self):
        from services.ingest import LatexReader, get_reader

        reader = get_reader(Path("test.bib"))
        assert isinstance(reader, LatexReader)

    def test_eml_reader(self):
        from services.ingest import EmailReader, get_reader

        reader = get_reader(Path("test.eml"))
        assert isinstance(reader, EmailReader)

    def test_mbox_reader(self):
        from services.ingest import EmailReader, get_reader

        reader = get_reader(Path("test.mbox"))
        assert isinstance(reader, EmailReader)

    def test_epub_reader(self):
        from services.ingest import EpubReader, get_reader

        reader = get_reader(Path("test.epub"))
        assert isinstance(reader, EpubReader)

    def test_pptx_reader(self):
        from services.ingest import PptxReader, get_reader

        reader = get_reader(Path("test.pptx"))
        assert isinstance(reader, PptxReader)

    def test_xlsx_reader(self):
        from services.ingest import XlsxReader, get_reader

        reader = get_reader(Path("test.xlsx"))
        assert isinstance(reader, XlsxReader)

    def test_xls_fails_loud_with_conversion_hint(self):
        # Legacy .xls was never actually readable (openpyxl can't parse it);
        # it now fails loud with the fix instead of crashing in the parser.
        import pytest

        from services.ingest import get_reader

        with pytest.raises(RuntimeError, match="convert it to .xlsx"):
            get_reader(Path("test.xls"))

    def test_ipynb_reader(self):
        from services.ingest import NotebookReader, get_reader

        reader = get_reader(Path("test.ipynb"))
        assert isinstance(reader, NotebookReader)


class TestSupportedExtensions:
    """Verify new extensions are in SUPPORTED_EXTENSIONS."""

    def test_new_extensions_included(self):
        from services.ingest import IngestionPipeline

        expected = {".docx", ".rtf", ".tex", ".bib", ".eml", ".mbox",
                    ".epub", ".pptx", ".xlsx", ".ipynb"}
        for ext in expected:
            assert ext in IngestionPipeline.SUPPORTED_EXTENSIONS, f"{ext} not in SUPPORTED_EXTENSIONS"
        # Legacy .xls is deliberately unsupported (fail-loud in get_reader).
        assert ".xls" not in IngestionPipeline.SUPPORTED_EXTENSIONS


class TestInferSourceType:
    """Verify _infer_source_type for new extensions."""

    def test_pptx_is_presentation(self):
        from services.ingest import _infer_source_type

        assert _infer_source_type(Path("test.pptx")) == "presentation"

    def test_xlsx_is_spreadsheet(self):
        from services.ingest import _infer_source_type

        assert _infer_source_type(Path("test.xlsx")) == "spreadsheet"

    def test_xls_is_spreadsheet(self):
        from services.ingest import _infer_source_type

        assert _infer_source_type(Path("test.xls")) == "spreadsheet"

    def test_ipynb_is_code(self):
        from services.ingest import _infer_source_type

        assert _infer_source_type(Path("test.ipynb")) == "code"

    def test_eml_is_email(self):
        from services.ingest import _infer_source_type

        assert _infer_source_type(Path("test.eml")) == "email"

    def test_mbox_is_email(self):
        from services.ingest import _infer_source_type

        assert _infer_source_type(Path("test.mbox")) == "email"

    def test_epub_is_document(self):
        from services.ingest import _infer_source_type

        assert _infer_source_type(Path("test.epub")) == "document"

    def test_tex_is_document(self):
        from services.ingest import _infer_source_type

        assert _infer_source_type(Path("test.tex")) == "document"

    def test_docx_is_document(self):
        from services.ingest import _infer_source_type

        assert _infer_source_type(Path("test.docx")) == "document"


class TestNotebookReaderParse:
    """Verify NotebookReader can parse a simple notebook."""

    def test_simple_notebook(self, tmp_path):
        import json as json_mod

        from services.ingest import NotebookReader

        nb = {
            "cells": [
                {"cell_type": "markdown", "source": ["# Title\n", "Some text"]},
                {"cell_type": "code", "source": ["print('hello')"]},
            ],
            "metadata": {"kernelspec": {"language": "python"}},
        }
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text(json_mod.dumps(nb))
        result = NotebookReader.read(nb_path)
        assert "[Format: Jupyter Notebook]" in result
        assert "[Cells: 2]" in result
        assert "# Title" in result
        assert "```python" in result
        assert "print('hello')" in result


class TestLatexReaderParse:
    """Verify LatexReader can parse simple LaTeX content."""

    def test_simple_tex(self, tmp_path):
        from services.ingest import LatexReader

        tex = r"""
\documentclass{article}
\begin{document}
\section{Introduction}
Hello world.
\textbf{Bold text} and \emph{italic text}.
\end{document}
"""
        tex_path = tmp_path / "test.tex"
        tex_path.write_text(tex)
        result = LatexReader.read(tex_path)
        assert "[Format: LaTeX]" in result
        assert "Hello world" in result
        assert "Bold text" in result
        assert "italic text" in result

    def test_simple_bib(self, tmp_path):
        from services.ingest import LatexReader

        bib = """
@article{smith2024,
  author = {John Smith},
  title = {A Great Paper},
  year = {2024},
  abstract = {This is the abstract.}
}
"""
        bib_path = tmp_path / "test.bib"
        bib_path.write_text(bib)
        result = LatexReader.read(bib_path)
        assert "[Format: BibTeX]" in result
        assert "John Smith" in result
        assert "A Great Paper" in result
        assert "[Entries: 1]" in result


class TestEmailReaderParse:
    """Verify EmailReader can parse a simple .eml file."""

    def test_simple_eml(self, tmp_path):
        from services.ingest import EmailReader

        eml_content = (
            "From: alice@example.com\r\n"
            "To: bob@example.com\r\n"
            "Subject: Test Email\r\n"
            "Date: Mon, 01 Jan 2024 00:00:00 +0000\r\n"
            "\r\n"
            "Hello Bob,\r\n"
            "This is a test.\r\n"
        )
        eml_path = tmp_path / "test.eml"
        eml_path.write_bytes(eml_content.encode("utf-8"))
        result = EmailReader.read(eml_path)
        assert "[Email]" in result
        assert "[Subject: Test Email]" in result
        assert "Hello Bob" in result


class TestIngestPolicyConfig:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_settings_hydrate_from_db(self, db_pool):
        """#91: ingest.* config keys reach the Config; explicit overrides win."""
        from core.tools.ingest import _build_ingest_config
        from services.ingest import IngestionMode

        async with db_pool.acquire() as conn:
            tr = conn.transaction()
            await tr.start()
            try:
                await conn.execute(
                    "SELECT set_config('ingest.max_parallel_llm', '7'::jsonb)"
                )
                await conn.execute(
                    "SELECT set_config('ingest.max_facts_per_section', '11'::jsonb)"
                )
            finally:
                # set_config must be visible to the builder's own conn —
                # commit inside the fixture DB (module-scoped, disposable).
                await tr.commit()

        config = await _build_ingest_config(
            db_pool, mode=IngestionMode.FAST, max_facts_per_section=3
        )
        assert config.max_parallel_llm == 7          # from DB
        assert config.max_facts_per_section == 3     # explicit override wins
        assert config.deep_max_words == 2000         # seeded default
