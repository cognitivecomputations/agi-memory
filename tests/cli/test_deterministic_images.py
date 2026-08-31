from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_python_runtime_images_install_only_from_committed_uv_lock():
    for relative in ("ops/Dockerfile.worker", "ops/Dockerfile.channels"):
        dockerfile = (ROOT / relative).read_text()

        assert "COPY pyproject.toml uv.lock /app/" in dockerfile
        assert "uv sync --locked --no-install-project" in dockerfile or (
            "uv sync --locked --extra channels --no-install-project" in dockerfile
        )
        assert "uv sync --locked" in dockerfile
        assert "pip install" not in dockerfile
        assert "ghcr.io/astral-sh/uv:0.12.7@sha256:" in dockerfile
        assert "FROM python:3.12-slim@sha256:" in dockerfile
        assert "date -u" not in dockerfile
        assert "sha256sum" in dockerfile


def test_uv_lock_covers_runtime_and_channel_only_dependencies():
    lock = (ROOT / "uv.lock").read_text()

    assert lock.startswith("version = 1\n")
    for package in (
        "hexis",
        "pywebpush",
        "tiktoken",
        "discord-py",
        "python-telegram-bot",
        "slack-bolt",
        "matrix-nio",
    ):
        assert f'name = "{package}"' in lock


def test_source_setup_and_watch_treat_lock_as_authoritative():
    mise = (ROOT / "mise.toml").read_text()
    compose = (ROOT / "docker-compose.yml").read_text()

    assert "uv sync --locked --inexact --python 3.12" in mise
    assert "uv pip install --python .venv/bin/python -e ." not in mise
    assert "path: ./uv.lock" in compose
