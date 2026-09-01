"""`hexis init` leaves a stack that can actually run the agent.

A database-only stack looks fine and does nothing: the heartbeat and
maintenance loops live in their own containers, so an init that returns early
because Postgres happens to be up produces an agent that never wakes up on its
own. These tests pin the readiness check to the whole default profile.
"""

from pathlib import Path

import pytest

import apps.hexis_cli as cli
from apps.hexis_init import _default_stack_services, _ensure_stack_running

pytestmark = [pytest.mark.cli]

ALL_SERVICES = "db\nrabbitmq\nheartbeat_worker\nmaintenance_worker"


@pytest.fixture
def compose(monkeypatch, tmp_path):
    """Stub the compose plumbing; record what init decided to start."""
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}\n")
    calls: dict[str, list] = {"capture": [], "run": []}
    responses = {"config": ALL_SERVICES, "ps": ""}

    def fake_capture(compose_cmd, path, root, args, env_file):
        calls["capture"].append(args)
        if "config" in args:
            return 0, responses["config"]
        return 0, responses["ps"]

    def fake_run(compose_cmd, path, root, args, env_file):
        calls["run"].append(args)
        return 0

    monkeypatch.setattr(cli, "_find_compose_file", lambda: (compose_file, True))
    monkeypatch.setattr(cli, "_stack_root_from_compose", lambda p: tmp_path)
    monkeypatch.setattr(cli, "ensure_docker", lambda: "docker")
    monkeypatch.setattr(cli, "ensure_compose", lambda docker: ["docker", "compose"])
    monkeypatch.setattr(cli, "resolve_env_file", lambda root: None)
    monkeypatch.setattr(cli, "_run_compose_capture", fake_capture)
    monkeypatch.setattr(cli, "run_compose", fake_run)
    return calls, responses


def test_starts_the_workers_when_only_the_database_is_up(compose):
    calls, responses = compose
    responses["ps"] = "db rabbitmq"

    _ensure_stack_running(object())

    # The workers were missing, so init brought the stack up rather than
    # declaring victory on a database that cannot think.
    assert ["up", "-d"] in calls["run"]


def test_does_not_restart_a_stack_that_is_fully_up(compose):
    calls, responses = compose
    responses["ps"] = "db rabbitmq heartbeat_worker maintenance_worker"

    _ensure_stack_running(object())

    assert calls["run"] == []


def test_starts_everything_when_nothing_is_running(compose):
    calls, responses = compose
    responses["ps"] = ""

    _ensure_stack_running(object())

    assert ["up", "-d"] in calls["run"]


def test_default_services_come_from_compose_not_a_hardcoded_list(compose):
    _calls, responses = compose
    responses["config"] = "db\nrabbitmq\nheartbeat_worker\nmaintenance_worker\nnew_service"

    services = _default_stack_services(["docker", "compose"], Path("x"), Path("y"), None)

    assert services == ["db", "rabbitmq", "heartbeat_worker", "maintenance_worker", "new_service"]


def test_compose_warnings_are_not_mistaken_for_services(compose):
    _calls, responses = compose
    responses["config"] = (
        "WARN[0000] The \"OPENAI_API_KEY\" variable is not set. Defaulting to a blank string.\n"
        "db\nheartbeat_worker\nmaintenance_worker"
    )

    services = _default_stack_services(["docker", "compose"], Path("x"), Path("y"), None)

    assert services == ["db", "heartbeat_worker", "maintenance_worker"]


def test_falls_back_to_the_essential_services_when_compose_cannot_answer(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_run_compose_capture", lambda *a, **k: (1, "docker not found"))

    services = _default_stack_services(["docker", "compose"], tmp_path, tmp_path, None)

    assert services == ["db", "heartbeat_worker", "maintenance_worker"]


def test_local_dsns_auto_start_the_stack_but_remote_ones_do_not():
    from apps.hexis_init import _dsn_is_local

    assert _dsn_is_local("postgresql://u:p@localhost:43815/hexis_memory")
    assert _dsn_is_local("postgresql://u:p@127.0.0.1:43815/hexis_memory")
    assert _dsn_is_local("postgresql://u:p@db:5432/hexis_memory")
    # Someone pointed at a database they host elsewhere keeps their own stack.
    assert not _dsn_is_local("postgresql://u:p@brain.example.com:5432/hexis_memory")
    assert not _dsn_is_local("postgresql://u:p@10.0.0.5:5432/hexis_memory")
