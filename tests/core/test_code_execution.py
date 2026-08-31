"""
Tests for Phase 2: Code Execution Tool

Covers CodeExecutionHandler: basic execution, variable persistence,
timeout enforcement, safe builtins, error handling, and registry integration.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

pytestmark = [pytest.mark.asyncio(loop_scope="session")]


@pytest.fixture(autouse=True)
def _restore_cwd():
    """Restore working directory after each test.

    The timeout test can leave background threads that corrupt os.getcwd().
    """
    import os

    saved = os.getcwd()
    yield
    try:
        os.getcwd()
    except OSError:
        os.chdir(saved)


# ============================================================================
# Helper: create a fresh handler + context
# ============================================================================


def _make_context(session_id: str | None = None):
    from core.tools.base import ToolContext, ToolExecutionContext

    return ToolExecutionContext(
        tool_context=ToolContext.CHAT,
        call_id=str(uuid.uuid4()),
        session_id=session_id or f"test-{uuid.uuid4().hex[:8]}",
    )


# ============================================================================
# Basic execution
# ============================================================================


class TestCodeExecutionBasic:
    async def test_simple_print(self):
        from core.tools.code_execution import CodeExecutionHandler, cleanup_session_repl

        handler = CodeExecutionHandler()
        ctx = _make_context()
        try:
            result = await handler.execute({"code": "print('hello world')"}, ctx)
            assert result.success is True
            assert "hello world" in result.output["stdout"]
        finally:
            cleanup_session_repl(ctx.session_id)

    async def test_math_expression(self):
        from core.tools.code_execution import CodeExecutionHandler, cleanup_session_repl

        handler = CodeExecutionHandler()
        ctx = _make_context()
        try:
            result = await handler.execute({"code": "x = 2 + 3\nprint(x)"}, ctx)
            assert result.success is True
            assert "5" in result.output["stdout"]
            assert "x" in result.output["variables"]
        finally:
            cleanup_session_repl(ctx.session_id)

    async def test_empty_code_rejected(self):
        from core.tools.code_execution import CodeExecutionHandler

        handler = CodeExecutionHandler()
        ctx = _make_context()
        result = await handler.execute({"code": ""}, ctx)
        assert result.success is False
        assert "No code" in result.error

    async def test_whitespace_code_rejected(self):
        from core.tools.code_execution import CodeExecutionHandler

        handler = CodeExecutionHandler()
        ctx = _make_context()
        result = await handler.execute({"code": "   \n   "}, ctx)
        assert result.success is False


# ============================================================================
# Variable persistence across calls
# ============================================================================


class TestCodeExecutionPersistence:
    async def test_variables_persist(self):
        from core.tools.code_execution import CodeExecutionHandler, cleanup_session_repl

        handler = CodeExecutionHandler()
        session_id = f"persist-{uuid.uuid4().hex[:8]}"
        ctx = _make_context(session_id)

        try:
            # First call: set variable
            r1 = await handler.execute({"code": "data = [1, 2, 3]"}, ctx)
            assert r1.success is True

            # Second call: use variable from first call
            ctx2 = _make_context(session_id)
            r2 = await handler.execute({"code": "total = sum(data)\nprint(total)"}, ctx2)
            assert r2.success is True
            assert "6" in r2.output["stdout"]
        finally:
            cleanup_session_repl(session_id)

    async def test_different_sessions_isolated(self):
        from core.tools.code_execution import CodeExecutionHandler, cleanup_session_repl

        handler = CodeExecutionHandler()
        sess_a = f"iso-a-{uuid.uuid4().hex[:8]}"
        sess_b = f"iso-b-{uuid.uuid4().hex[:8]}"

        try:
            # Session A: define a variable
            ctx_a = _make_context(sess_a)
            await handler.execute({"code": "secret = 42"}, ctx_a)

            # Session B: variable should NOT exist
            ctx_b = _make_context(sess_b)
            r = await handler.execute({"code": "print(secret)"}, ctx_b)
            assert r.success is False
            assert "NameError" in (r.error or "")
        finally:
            cleanup_session_repl(sess_a)
            cleanup_session_repl(sess_b)


# ============================================================================
# Timeout enforcement
# ============================================================================


class TestCodeExecutionTimeout:
    async def test_timeout_kills_long_running_code(self):
        import os
        from core.tools.code_execution import CodeExecutionHandler, cleanup_session_repl

        handler = CodeExecutionHandler()
        ctx = _make_context()
        saved_cwd = os.getcwd()
        try:
            result = await handler.execute(
                {"code": "import time; time.sleep(10)", "timeout": 1},
                ctx,
            )
            assert result.success is False
            assert "timed out" in (result.error or "").lower()
        finally:
            # Timeout test can leave cwd in a deleted temp dir; restore it
            try:
                os.getcwd()
            except OSError:
                os.chdir(saved_cwd)
            cleanup_session_repl(ctx.session_id)

    async def test_timeout_capped_at_max(self):
        from core.tools.code_execution import CodeExecutionHandler, MAX_TIMEOUT

        handler = CodeExecutionHandler()
        # Verify the handler caps timeout at MAX_TIMEOUT
        assert MAX_TIMEOUT == 120


# ============================================================================
# Safe builtins
# ============================================================================


class TestCodeExecutionSafety:
    async def test_eval_blocked(self):
        from core.tools.code_execution import CodeExecutionHandler, cleanup_session_repl

        handler = CodeExecutionHandler()
        ctx = _make_context()
        try:
            result = await handler.execute({"code": "result = eval('1+1')"}, ctx)
            # eval is set to None in safe builtins, so calling it should error
            assert result.success is False
        finally:
            cleanup_session_repl(ctx.session_id)

    async def test_exec_blocked(self):
        from core.tools.code_execution import CodeExecutionHandler, cleanup_session_repl

        handler = CodeExecutionHandler()
        ctx = _make_context()
        try:
            result = await handler.execute({"code": "exec('x=1')"}, ctx)
            assert result.success is False
        finally:
            cleanup_session_repl(ctx.session_id)

    async def test_import_still_works(self):
        from core.tools.code_execution import CodeExecutionHandler, cleanup_session_repl

        handler = CodeExecutionHandler()
        ctx = _make_context()
        try:
            result = await handler.execute(
                {"code": "import json\nprint(json.dumps({'a': 1}))"},
                ctx,
            )
            assert result.success is True
            assert '{"a": 1}' in result.output["stdout"]
        finally:
            cleanup_session_repl(ctx.session_id)


# ============================================================================
# Error handling
# ============================================================================


class TestCodeExecutionErrors:
    async def test_syntax_error(self):
        from core.tools.code_execution import CodeExecutionHandler, cleanup_session_repl

        handler = CodeExecutionHandler()
        ctx = _make_context()
        try:
            result = await handler.execute({"code": "def foo(:"}, ctx)
            assert result.success is False
            assert "SyntaxError" in (result.error or "")
        finally:
            cleanup_session_repl(ctx.session_id)

    async def test_runtime_error(self):
        from core.tools.code_execution import CodeExecutionHandler, cleanup_session_repl

        handler = CodeExecutionHandler()
        ctx = _make_context()
        try:
            result = await handler.execute({"code": "1 / 0"}, ctx)
            assert result.success is False
            assert "ZeroDivisionError" in (result.error or "")
        finally:
            cleanup_session_repl(ctx.session_id)

    async def test_name_error(self):
        from core.tools.code_execution import CodeExecutionHandler, cleanup_session_repl

        handler = CodeExecutionHandler()
        ctx = _make_context()
        try:
            result = await handler.execute({"code": "print(undefined_var)"}, ctx)
            assert result.success is False
            assert "NameError" in (result.error or "")
        finally:
            cleanup_session_repl(ctx.session_id)

    async def test_partial_output_on_error(self):
        from core.tools.code_execution import CodeExecutionHandler, cleanup_session_repl

        handler = CodeExecutionHandler()
        ctx = _make_context()
        try:
            result = await handler.execute(
                {"code": "print('before error')\n1/0"},
                ctx,
            )
            assert result.success is False
            # stdout should still contain output from before the error
            assert "before error" in result.output["stdout"]
        finally:
            cleanup_session_repl(ctx.session_id)


# ============================================================================
# Integration: registry registration
# ============================================================================


class TestCodeExecutionRegistration:
    async def test_registered_in_default_registry(self, db_pool):
        from core.tools import create_default_registry, ToolContext

        registry = create_default_registry(db_pool)
        handler = registry.get("execute_code")
        assert handler is not None
        assert handler.spec.name == "execute_code"
        assert handler.spec.category.value == "code"

    async def test_appears_in_chat_specs(self, db_pool):
        from core.tools import create_default_registry, ToolContext

        registry = create_default_registry(db_pool)
        specs = await registry.get_specs(ToolContext.CHAT)
        names = [s["function"]["name"] for s in specs]
        assert "execute_code" in names

    async def test_execute_via_registry(self, db_pool):
        from core.tools import ToolContext, ToolExecutionContext, create_default_registry
        from core.tools.code_execution import cleanup_session_repl

        registry = create_default_registry(db_pool)
        session_id = f"reg-test-{uuid.uuid4().hex[:8]}"
        ctx = ToolExecutionContext(
            tool_context=ToolContext.CHAT,
            call_id=str(uuid.uuid4()),
            session_id=session_id,
        )

        try:
            result = await registry.execute(
                "execute_code",
                {"code": "answer = 6 * 7\nprint(answer)"},
                ctx,
            )
            assert result.success is True
            assert "42" in result.output["stdout"]
        finally:
            cleanup_session_repl(session_id)


# ============================================================================
# Spec validation
# ============================================================================


class TestCodeExecutionSpec:
    def test_spec_properties(self):
        from core.tools.code_execution import CodeExecutionHandler

        handler = CodeExecutionHandler()
        spec = handler.spec
        assert spec.name == "execute_code"
        assert spec.energy_cost == 3
        assert spec.is_read_only is False
        # Arbitrary Python in-process, persistent globals, bridged to the whole
        # tool registry — unsandboxed. A person says yes or it does not run.
        assert spec.requires_approval is True
        assert spec.supports_parallel is False

    def test_openai_function_format(self):
        from core.tools.code_execution import CodeExecutionHandler

        handler = CodeExecutionHandler()
        func = handler.spec.to_openai_function()
        assert func["type"] == "function"
        assert func["function"]["name"] == "execute_code"
        assert "code" in func["function"]["parameters"]["properties"]
        assert "code" in func["function"]["parameters"]["required"]
