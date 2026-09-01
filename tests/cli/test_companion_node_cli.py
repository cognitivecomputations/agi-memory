from __future__ import annotations

import asyncio
import base64
import json
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from apps.cli_node import _invoke
from apps.hexis_cli import main


def test_node_local_cli_journey_keeps_identity_and_allowlist_explicit(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    assert main(["node", "init", "--name", "Laptop"]) == 0
    created = capsys.readouterr().out
    assert "Created signed node identity for Laptop" in created
    assert "Pairing completes in place" in created

    assert (
        main(
            [
                "node",
                "allow",
                "hello",
                "--allow-args",
                "--",
                sys.executable,
                "-c",
                "print('hello')",
            ]
        )
        == 0
    )
    allowed = capsys.readouterr().out
    assert "Invocation-time args are allowed" in allowed

    assert main(["node", "allow", "hello", "/bin/false"]) == 1
    collision = capsys.readouterr().err
    assert "--replace" in collision

    assert main(["node", "status", "--local-only", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["local"]["name"] == "Laptop"
    assert status["local"]["commands"]["hello"]["allow_args"] is True
    assert "system.run" in status["local"]["capabilities"]
    assert "private_key" not in json.dumps(status)

    assert main(["node", "disallow", "hello"]) == 0
    assert "Removed host-command alias" in capsys.readouterr().out


def test_node_invoke_parser_requires_an_exact_action_shape(capsys):
    assert (
        main(
            [
                "node",
                "invoke",
                "a" * 64,
                "system.run",
                "--yes",
            ]
        )
        == 1
    )
    assert "requires --command" in capsys.readouterr().err


def test_node_wake_cli_is_explicit_and_status_never_opens_microphone(
    monkeypatch, tmp_path, capsys
):
    from apps import cli_node

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    assert main(["node", "init", "--name", "Wake laptop"]) == 0
    capsys.readouterr()
    model = tmp_path / "custom-wake.onnx"
    model.write_bytes(b"model")
    monkeypatch.setattr(cli_node, "_install_wake_support", lambda **_kwargs: True)

    assert (
        main(
            [
                "node",
                "wake",
                "setup",
                "--model",
                str(model),
                "--threshold",
                "0.6",
                "--max-utterance-seconds",
                "20",
                "--silence-ms",
                "900",
                "--session-idle-minutes",
                "10",
                "-y",
            ]
        )
        == 0
    )
    assert "microphone is still off" in capsys.readouterr().out

    assert main(["node", "wake", "status", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["enabled"] is True
    assert status["microphone_active"] is False
    assert status["model_name"] == "custom-wake"

    assert main(["node", "wake", "disable"]) == 0
    assert "No model or identity was deleted" in capsys.readouterr().out
    assert main(["node", "wake", "status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["enabled"] is False


def test_screen_capture_refuses_to_overwrite_without_explicit_choice(
    tmp_path: Path, capsys
) -> None:
    target = tmp_path / "screen.png"
    target.write_bytes(b"original")
    args = Namespace(
        action="screen.capture",
        node_id="a" * 64,
        command_alias=None,
        invoke_args=[],
        timeout=30,
        output=str(target),
        overwrite=False,
        yes=True,
    )
    result = {
        "status": "succeeded",
        "invocation_id": "11111111-1111-4111-8111-111111111111",
        "result": {"data_base64": base64.b64encode(b"new").decode("ascii")},
    }
    pool = AsyncMock()
    pool.close = AsyncMock()
    asyncpg = pytest.importorskip("asyncpg")
    invoke = AsyncMock(return_value=result)
    with (
        patch.object(asyncpg, "create_pool", AsyncMock(return_value=pool)),
        patch(
            "services.node_gateway.request_node_invocation",
            invoke,
        ),
    ):
        assert asyncio.run(_invoke(args, "postgresql://unused")) == 1

    invoke.assert_not_awaited()
    assert target.read_bytes() == b"original"
    assert "Refusing to overwrite" in capsys.readouterr().err
