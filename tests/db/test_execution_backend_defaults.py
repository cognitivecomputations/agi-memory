from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_execution_defaults_are_local_and_bounded(db_pool):
    async with db_pool.acquire() as conn:
        raw = await conn.fetchval("SELECT get_config('execution.backends')")
        max_output = await conn.fetchval(
            "SELECT get_config_int('execution.max_output_chars')"
        )
        max_timeout = await conn.fetchval(
            "SELECT get_config_int('execution.max_timeout_seconds')"
        )
        ttl = await conn.fetchval(
            "SELECT get_config_int('execution.repl_state_ttl_hours')"
        )
    config = json.loads(raw) if isinstance(raw, str) else raw
    assert config == {
        "active": "local",
        "profiles": {"local": {"type": "local"}},
    }
    assert max_output == 50_000
    assert max_timeout == 300
    assert ttl == 168


async def test_execution_migration_is_self_contained(db_pool):
    async with db_pool.acquire() as conn, conn.transaction():
        for key in (
            "execution.backends",
            "execution.max_output_chars",
            "execution.max_timeout_seconds",
            "execution.repl_state_ttl_hours",
        ):
            await conn.execute("DELETE FROM config_defaults WHERE key = $1", key)
        migration = Path("db/migrations/0231_execution_backends.sql").read_text()
        await conn.execute(migration)
        assert await conn.fetchval(
            "SELECT get_config('execution.backends') IS NOT NULL"
        )


async def test_live_local_profile_routes_shell_script_and_code(db_pool, tmp_path):
    from core.tools.base import ToolContext, ToolExecutionContext
    from core.tools.code_execution import CodeExecutionHandler, cleanup_session_repl
    from core.tools.shell import ScriptRunnerHandler, ShellHandler

    script = tmp_path / "journey.py"
    script.write_text("print('script-local-ok')\n")
    context = ToolExecutionContext(
        tool_context=ToolContext.CHAT,
        call_id="execution-local-journey",
        session_id="execution-local-journey",
        workspace_path=str(tmp_path),
        allow_shell=True,
        registry=SimpleNamespace(pool=db_pool),
    )
    shell = await ShellHandler().execute({"command": "printf shell-local-ok"}, context)
    script_result = await ScriptRunnerHandler().execute({"path": "journey.py"}, context)
    try:
        code = await CodeExecutionHandler().execute(
            {"code": "print('code-local-ok')"}, context
        )
    finally:
        cleanup_session_repl(context.session_id or "")
    assert shell.success and shell.output["backend"] == "local"
    assert shell.output["stdout"] == "shell-local-ok"
    assert script_result.success and script_result.output["backend"] == "local"
    assert "script-local-ok" in script_result.output["stdout"]
    assert code.success and code.output["backend"] == "local"
    assert code.output["stdout"] == "code-local-ok"
