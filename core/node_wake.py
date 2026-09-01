"""Explicit local wake-word capture for a paired companion node.

The detector and microphone dependencies are optional and imported only after
the operator enables wake capture. Detection and VAD stay on the node; only the
bounded post-chime utterance is sent over the already authenticated connection.
"""

from __future__ import annotations

import io
import math
import os
import threading
import time
import wave
from array import array
from pathlib import Path
from typing import Any, Callable

WAKE_REQUIREMENTS = ("openwakeword==0.6.0", "sounddevice>=0.5,<0.6")
_SAMPLE_RATE = 16_000
_CHUNK_SAMPLES = 1_280  # 80 ms, openWakeWord's documented streaming frame.
_NO_SPEECH_TIMEOUT_SECONDS = 8


class NodeWakeError(RuntimeError):
    """A local wake failure with an actionable recovery step."""


def wake_model_cache() -> Path:
    root = Path(os.getenv("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    return root / "hexis" / "wake-models"


def pretrained_model_catalog() -> dict[str, dict[str, str]]:
    """Read model names and URLs from the installed detector package."""
    try:
        import openwakeword
    except ImportError as exc:
        raise NodeWakeError(
            "Wake support is not installed. Run `hexis node wake setup` to install "
            "it and choose a model."
        ) from exc
    catalog: dict[str, dict[str, str]] = {}
    for name, raw in dict(getattr(openwakeword, "MODELS", {})).items():
        if not isinstance(raw, dict):
            continue
        download_url = str(raw.get("download_url") or "")
        if download_url:
            catalog[str(name)] = {"download_url": download_url}
    if not catalog:
        raise NodeWakeError(
            "The installed openWakeWord package exposed no pretrained model catalog. "
            "Use `hexis node wake setup --model /absolute/custom-model.onnx`."
        )
    return catalog


def download_pretrained_model(name: str) -> Path:
    """Download one explicitly selected upstream model into the Hexis cache."""
    catalog = pretrained_model_catalog()
    selected = str(name or "").strip()
    if selected not in catalog:
        available = ", ".join(sorted(catalog))
        raise NodeWakeError(
            f"Unknown pretrained wake model {selected!r}. Available models: {available}."
        )
    target = wake_model_cache()
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        os.chmod(target, 0o700)
    try:
        from openwakeword.utils import download_models

        download_models(model_names=[selected], target_directory=str(target))
    except Exception as exc:
        raise NodeWakeError(
            f"Wake model {selected!r} could not be downloaded ({exc}). Existing "
            f"cache files were preserved at {target}; retry the setup when ready."
        ) from exc
    filename = Path(catalog[selected]["download_url"]).name
    expected = target / filename.replace(".tflite", ".onnx")
    if not expected.is_file():
        matches = sorted(target.glob(f"{selected}*.onnx"))
        if not matches:
            raise NodeWakeError(
                f"The download completed but no ONNX model for {selected!r} appeared "
                f"under {target}. Review that directory and retry."
            )
        expected = matches[0]
    return expected.resolve()


def _rms_pcm16(raw: bytes) -> float:
    if len(raw) < 2:
        return 0.0
    samples = array("h")
    samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
    if not samples:
        return 0.0
    return math.sqrt(sum(value * value for value in samples) / len(samples))


def _wav_bytes(frames: list[bytes]) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(_SAMPLE_RATE)
        wav.writeframes(b"".join(frames))
    return output.getvalue()


class WakeListener:
    """Blocking local audio loop intended to run in one worker thread."""

    def __init__(self, wake: dict[str, Any]) -> None:
        model_path = Path(str(wake.get("model_path") or "")).expanduser().resolve()
        if not model_path.is_file():
            raise NodeWakeError(
                f"Configured wake model is missing: {model_path}. Run `hexis node "
                "wake setup` to select it again."
            )
        self.model_path = model_path
        self.model_name = str(wake.get("model_name") or model_path.stem)[:200]
        self.threshold = float(wake.get("threshold") or 0.5)
        self.input_device = _device_value(wake.get("input_device"))
        self.max_utterance_seconds = min(
            max(int(wake.get("max_utterance_seconds") or 30), 5), 60
        )
        self.silence_ms = min(max(int(wake.get("silence_ms") or 1200), 500), 3000)
        try:
            import numpy as np
            import sounddevice as sd
            from openwakeword.model import Model
        except ImportError as exc:
            raise NodeWakeError(
                "Wake dependencies are incomplete. Run `hexis node wake setup` in "
                "this Hexis environment, then retry."
            ) from exc
        try:
            framework = "onnx" if model_path.suffix.lower() == ".onnx" else "tflite"
            self._model = Model(
                wakeword_models=[str(model_path)],
                inference_framework=framework,
            )
        except Exception as exc:
            raise NodeWakeError(
                f"Wake model {model_path} could not be loaded ({exc}). Run `hexis "
                "node wake setup` to select a compatible model."
            ) from exc
        self._np = np
        self._sd = sd

    def run(
        self,
        *,
        stop_event: threading.Event,
        on_utterance: Callable[[bytes, dict[str, Any]], dict[str, Any]],
        status_callback: Callable[[str], Any] = print,
    ) -> None:
        status_callback(
            f"Wake listening is active for {self.model_name!r}. The microphone stays "
            "local until detection; Ctrl+C exits."
        )
        while not stop_event.is_set():
            try:
                detection = self._wait_for_detection(stop_event)
                if detection is None:
                    return
                status_callback(
                    f"Wake word detected ({detection['score']:.2f}). Listen for the cue, then speak."
                )
                self._tone(frequency=880, duration=0.12)
                utterance = self._capture_utterance(stop_event)
                if stop_event.is_set():
                    return
                if utterance is None:
                    status_callback(
                        "No post-wake speech was detected. Nothing was uploaded; wake listening resumed."
                    )
                    self._tone(frequency=260, duration=0.12)
                    continue
                status_callback(
                    "Utterance captured; the microphone is off while Hexis responds."
                )
                reply = on_utterance(utterance, detection)
                if stop_event.is_set():
                    return
                assistant = str(reply.get("assistant") or "").strip()
                transcript = str(reply.get("transcript") or "").strip()
                if transcript:
                    status_callback(f"You: {transcript}")
                if assistant:
                    status_callback(f"Hexis: {assistant}")
                audio = reply.get("audio")
                if isinstance(audio, bytes) and audio:
                    self._play_wav(audio)
                elif reply.get("error"):
                    status_callback(str(reply["error"]))
                    self._tone(frequency=260, duration=0.16)
            except NodeWakeError:
                raise
            except Exception as exc:
                raise NodeWakeError(
                    f"Wake audio stopped ({exc}). Check microphone/speaker access and "
                    "run `hexis node wake status` before retrying."
                ) from exc

    def _input_stream(self) -> Any:
        try:
            return self._sd.RawInputStream(
                samplerate=_SAMPLE_RATE,
                blocksize=_CHUNK_SAMPLES,
                device=self.input_device,
                channels=1,
                dtype="int16",
            )
        except Exception as exc:
            raise NodeWakeError(
                f"The wake microphone could not open ({exc}). Grant microphone "
                "permission or choose a device with `hexis node wake setup --device NAME`."
            ) from exc

    def _wait_for_detection(
        self, stop_event: threading.Event
    ) -> dict[str, Any] | None:
        with self._input_stream() as stream:
            while not stop_event.is_set():
                raw, _overflowed = stream.read(_CHUNK_SAMPLES)
                frame = self._np.frombuffer(bytes(raw), dtype=self._np.int16)
                predictions = self._model.predict(frame)
                if not isinstance(predictions, dict) or not predictions:
                    continue
                label, raw_score = max(
                    predictions.items(), key=lambda item: float(item[1])
                )
                score = float(raw_score)
                if score >= self.threshold:
                    self._model.reset()
                    return {
                        "model": self.model_name,
                        "label": str(label)[:200],
                        "score": min(max(score, 0.0), 1.0),
                    }
        return None

    def _capture_utterance(self, stop_event: threading.Event) -> bytes | None:
        started = time.monotonic()
        speech_started: float | None = None
        silence_started: float | None = None
        noise_floor = 120.0
        calibration = 0
        pre_roll: list[bytes] = []
        frames: list[bytes] = []
        with self._input_stream() as stream:
            while not stop_event.is_set():
                raw_buffer, _overflowed = stream.read(_CHUNK_SAMPLES)
                raw = bytes(raw_buffer)
                now = time.monotonic()
                level = _rms_pcm16(raw)
                if speech_started is None:
                    if now - started < 0.5:
                        noise_floor = (
                            (noise_floor * calibration + level) / (calibration + 1)
                        )
                        calibration += 1
                    threshold = max(350.0, noise_floor * 2.8)
                    pre_roll.append(raw)
                    pre_roll = pre_roll[-4:]
                    if now - started >= 0.25 and level >= threshold:
                        speech_started = now
                        frames.extend(pre_roll)
                        silence_started = None
                    elif now - started >= _NO_SPEECH_TIMEOUT_SECONDS:
                        return None
                    continue

                frames.append(raw)
                threshold = max(350.0, noise_floor * 2.8)
                if level >= threshold:
                    silence_started = None
                elif silence_started is None:
                    silence_started = now
                spoken_ms = (now - speech_started) * 1000
                if (
                    spoken_ms >= 450
                    and silence_started is not None
                    and (now - silence_started) * 1000 >= self.silence_ms
                ):
                    break
                if now - speech_started >= self.max_utterance_seconds:
                    break
        return _wav_bytes(frames) if frames else None

    def _tone(self, *, frequency: int, duration: float) -> None:
        count = max(1, int(_SAMPLE_RATE * duration))
        times = self._np.arange(count, dtype=self._np.float32) / _SAMPLE_RATE
        samples = (0.12 * self._np.sin(2 * self._np.pi * frequency * times)).astype(
            self._np.float32
        )
        self._sd.play(samples, _SAMPLE_RATE, blocking=True)

    def _play_wav(self, raw: bytes) -> None:
        try:
            with wave.open(io.BytesIO(raw), "rb") as wav:
                if wav.getsampwidth() != 2:
                    raise NodeWakeError(
                        "The voice provider returned a WAV format this node cannot play."
                    )
                channels = wav.getnchannels()
                rate = wav.getframerate()
                samples = self._np.frombuffer(
                    wav.readframes(wav.getnframes()), dtype=self._np.int16
                )
            if channels > 1:
                samples = samples.reshape((-1, channels))
            self._sd.play(samples, rate, blocking=True)
        except NodeWakeError:
            raise
        except Exception as exc:
            raise NodeWakeError(
                f"The spoken response could not play ({exc}). Its text was printed; "
                "check the node speaker and retry."
            ) from exc


def _device_value(value: Any) -> str | int | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text
