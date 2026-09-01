from __future__ import annotations

import wave
from array import array
from pathlib import Path

from core.node_daemon import advertised_capabilities
from core.node_identity import (
    disable_node_wake,
    initialize_node_identity,
    set_node_wake,
)
from core.node_wake import _rms_pcm16, _wav_bytes


def test_wake_policy_is_explicit_and_advertised_only_while_enabled(tmp_path: Path):
    config = tmp_path / "node.json"
    model = tmp_path / "hey-hexis.onnx"
    model.write_bytes(b"model")
    initialize_node_identity(name="Wake node", path=config)

    enabled = set_node_wake(
        model_path=str(model),
        model_name="hey-hexis",
        threshold=0.61,
        input_device="Built-in Mic",
        max_utterance_seconds=20,
        silence_ms=900,
        session_idle_minutes=10,
        path=config,
    )

    assert enabled.wake["enabled"] is True
    assert enabled.wake["threshold"] == 0.61
    assert "audio.wake" in advertised_capabilities(enabled)

    disabled = disable_node_wake(config)
    assert disabled.wake["enabled"] is False
    assert disabled.wake["model_path"] == str(model.resolve())
    assert "audio.wake" not in advertised_capabilities(disabled)


def test_wake_pcm_helpers_emit_bounded_standard_wav():
    samples = array("h", [1000, -1000] * 1280)

    assert 999 <= _rms_pcm16(samples.tobytes()) <= 1001
    payload = _wav_bytes([samples.tobytes()])

    with wave.open(__import__("io").BytesIO(payload), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16_000
        assert wav.getnframes() == len(samples)
