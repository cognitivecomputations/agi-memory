from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _service_block(compose_path: str, service: str) -> str:
    text = (ROOT / compose_path).read_text(encoding="utf-8")
    marker = f"  {service}:\n"
    start = text.index(marker)
    next_service = text.find("\n  ", start + len(marker))
    while next_service != -1 and text[next_service + 3] in {" ", "#"}:
        next_service = text.find("\n  ", next_service + 1)
    return text[start:] if next_service == -1 else text[start:next_service]


def test_source_compose_starts_core_workers_by_default():
    for name in (
        "rabbitmq",
        "heartbeat_worker",
        "maintenance_worker",
        "api",
        "channel_worker",
        "ui",
    ):
        assert "\n    profiles:" not in _service_block("docker-compose.yml", name)

    assert 'command: ["hexis-worker", "--mode", "heartbeat"]' in _service_block(
        "docker-compose.yml", "heartbeat_worker"
    )
    assert 'command: ["hexis-worker", "--mode", "maintenance"]' in _service_block(
        "docker-compose.yml", "maintenance_worker"
    )
    assert 'command: ["hexis-channels"]' in _service_block(
        "docker-compose.yml", "channel_worker"
    )


def test_source_compose_can_use_published_images_without_building():
    expected = {
        "db": "ghcr.io/quixiai/hexis-brain",
        "heartbeat_worker": "ghcr.io/quixiai/hexis-worker",
        "maintenance_worker": "ghcr.io/quixiai/hexis-worker",
        "api": "ghcr.io/quixiai/hexis-worker",
        "channel_worker": "ghcr.io/quixiai/hexis-channels",
        "ui": "ghcr.io/quixiai/hexis-ui",
    }
    for service, image in expected.items():
        block = _service_block("docker-compose.yml", service)
        assert f"image: {image}:${{HEXIS_IMAGE_TAG:-latest}}" in block
        assert "\n    build:" in block


def test_runtime_compose_starts_core_workers_by_default():
    for name in (
        "rabbitmq",
        "heartbeat_worker",
        "maintenance_worker",
        "api",
        "channel_worker",
        "ui",
    ):
        assert "\n    profiles:" not in _service_block(
            "ops/docker-compose.runtime.yml", name
        )

    assert 'command: ["hexis-worker", "--mode", "heartbeat"]' in _service_block(
        "ops/docker-compose.runtime.yml", "heartbeat_worker"
    )
    assert 'command: ["hexis-worker", "--mode", "maintenance"]' in _service_block(
        "ops/docker-compose.runtime.yml", "maintenance_worker"
    )
    assert 'command: ["hexis-channels"]' in _service_block(
        "ops/docker-compose.runtime.yml", "channel_worker"
    )


def test_api_can_reach_loopback_voice_sidecar_on_every_supported_docker_host():
    for compose_path in ("docker-compose.yml", "ops/docker-compose.runtime.yml"):
        block = _service_block(compose_path, "api")
        assert '"host.docker.internal:host-gateway"' in block
        assert (
            "HEXIS_TTS_URL: "
            "${HEXIS_TTS_URL:-http://host.docker.internal:42667}" in block
        )
