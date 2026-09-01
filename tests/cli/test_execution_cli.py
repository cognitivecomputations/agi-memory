from __future__ import annotations

import json

from apps.hexis_cli import build_parser, main


def test_execution_parser_defaults_to_read_only_status():
    args = build_parser().parse_args(["execution"])
    assert args.func == "execution_status"
    assert args.json is False


def test_execution_local_status_and_probe_journey(capsys):
    assert main(["execution", "status", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["active"] == "local"
    assert status["profiles"] == [
        {
            "name": "local",
            "type": "local",
            "active": True,
            "locally_ready": True,
            "issues": [],
            "config": {"type": "local"},
        }
    ]
    assert "read-only" in status["note"].lower()

    assert main(["execution", "test", "local", "--json"]) == 0
    probe = json.loads(capsys.readouterr().out)
    assert probe["success"] is True
    assert probe["profile"] == "local"
    assert probe["type"] == "local"
    assert "hexis-execution-ok" in probe["stdout"]


def test_execution_setup_refuses_missing_identity_before_database_write(
    tmp_path, capsys
):
    missing = tmp_path / "missing-key"
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("build.example ssh-ed25519 AAAA\n")
    assert (
        main(
            [
                "execution",
                "add-ssh",
                "build",
                "--host",
                "build.example",
                "--user",
                "runner",
                "--workspace",
                "/srv/project",
                "--identity-file",
                str(missing),
                "--known-hosts-file",
                str(known_hosts),
            ]
        )
        == 1
    )
    error = capsys.readouterr().err
    assert "identity file is unavailable" in error
    assert str(missing) in error


def test_execution_status_reports_database_failure_without_traceback(
    monkeypatch, capsys
):
    from apps import cli_execution

    async def unavailable(_dsn):
        raise RuntimeError("database offline")

    monkeypatch.setattr(cli_execution, "_open_pool", unavailable)
    assert main(["execution", "status"]) == 1
    error = capsys.readouterr().err
    assert "Could not read execution profiles: database offline" in error
    assert "Traceback" not in error
