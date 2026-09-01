"""Background, device-local speaker diarization and transcript labeling."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATUS_FILENAME = "diarize_status.json"
PID_FILENAME = "diarize.pid"
LOG_FILENAME = "diarize.log"
DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def audio_analysis_cache_root() -> Path:
    base = os.getenv("XDG_CACHE_HOME")
    return (
        (Path(base).expanduser() if base else Path.home() / ".cache")
        / "hexis"
        / "audio-analysis"
    )


def default_output_dir(audio_path: str | Path) -> Path:
    audio = Path(audio_path).expanduser().resolve()
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", audio.stem).strip("-.") or "audio"
    fingerprint = hashlib.sha256(str(audio).encode("utf-8")).hexdigest()[:12]
    return audio_analysis_cache_root() / f"{safe_stem}-{fingerprint}"


def resolve_hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = str(os.getenv(key) or "").strip()
        if value:
            return value
    return None


def status_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / STATUS_FILENAME


def write_status(output_dir: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = status_path(output)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({**payload, "updated_at": _utc_now()}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def status_for(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    path = status_path(output)
    if not path.exists():
        return {
            "status": "missing",
            "output_dir": str(output),
            "error": "No job exists yet. Start analysis with this audio path first.",
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "failed",
            "output_dir": str(output),
            "error": f"Invalid status file: {exc}",
        }
    pid = data.get("pid")
    if data.get("status") == "running" and pid:
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            data["status"] = "failed"
            data["error"] = (
                data.get("error")
                or f"Worker {pid} stopped without a completion record. Check {output / LOG_FILENAME}."
            )
        except PermissionError:
            pass
    data["output_dir"] = str(output)
    return data


def assign_speakers_to_segments(
    segments: list[dict[str, Any]], turns: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    for segment in segments:
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start)
        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        for turn in turns:
            overlap = max(
                0.0,
                min(end, float(turn["end"])) - max(start, float(turn["start"])),
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = str(turn.get("speaker") or "UNKNOWN")
        if best_overlap <= 0 and turns:
            midpoint = (start + end) / 2
            nearest = min(
                turns,
                key=lambda turn: min(
                    abs(midpoint - float(turn["start"])),
                    abs(midpoint - float(turn["end"])),
                ),
            )
            best_speaker = str(nearest.get("speaker") or "UNKNOWN")
        labeled.append(
            {
                **segment,
                "speaker": best_speaker,
                "speaker_overlap_s": round(best_overlap, 3),
            }
        )
    return labeled


def turns_from_annotation(output: Any) -> list[dict[str, Any]]:
    annotation = output
    if not hasattr(annotation, "itertracks"):
        annotation = getattr(output, "exclusive_speaker_diarization", None) or getattr(
            output, "speaker_diarization", None
        )
    if annotation is None or not hasattr(annotation, "itertracks"):
        raise RuntimeError(f"Unsupported pyannote output type: {type(output)!r}")
    turns = [
        {
            "start": float(segment.start),
            "end": float(segment.end),
            "speaker": str(label),
        }
        for segment, _, label in annotation.itertracks(yield_label=True)
    ]
    return sorted(turns, key=lambda turn: turn["start"])


def _load_audio(audio_path: Path) -> dict[str, Any]:
    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(audio_path))
        return {"waveform": waveform, "sample_rate": int(sample_rate)}
    except Exception:
        try:
            import numpy as np
            import soundfile as sound_file
            import torch

            data, sample_rate = sound_file.read(str(audio_path), always_2d=True)
            waveform = torch.from_numpy(np.asarray(data, dtype=np.float32).T)
            return {"waveform": waveform, "sample_rate": int(sample_rate)}
        except Exception as exc:
            raise RuntimeError(
                "Audio decoding failed. Install ffmpeg plus the hexis audio_analysis extra."
            ) from exc


def run_pyannote_diarization(audio_path: Path, model_id: str) -> list[dict[str, Any]]:
    token = resolve_hf_token()
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Accept the pyannote model terms on Hugging Face, then expose HF_TOKEN to Hexis."
        )
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Install local diarization with: pip install 'hexis[audio_analysis]'"
        ) from exc
    try:
        pipeline = Pipeline.from_pretrained(model_id, token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)
    try:
        import torch

        if not torch.cuda.is_available():
            pipeline.to(torch.device("cpu"))
    except Exception:
        logger.debug("Could not explicitly select a pyannote device", exc_info=True)
    return turns_from_annotation(pipeline(_load_audio(audio_path)))


def emotion_heuristics_for_segments(
    audio_path: Path, segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Add explicitly labeled coarse local heuristics, never inferred emotions."""

    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Install emotion heuristics with: pip install 'hexis[audio_analysis]'"
        ) from exc
    samples, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
    output: list[dict[str, Any]] = []
    for segment in segments:
        start = max(0, int(float(segment.get("start") or 0) * sample_rate))
        end = min(len(samples), int(float(segment.get("end") or 0) * sample_rate))
        chunk = samples[start:end] if end > start else np.zeros(1, dtype=samples.dtype)
        rms = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))
        try:
            pitches = librosa.yin(chunk, fmin=50, fmax=400, sr=sample_rate)
            pitch = float(np.nanmedian(pitches)) if len(pitches) else 0.0
            if np.isnan(pitch):
                pitch = 0.0
        except Exception:
            pitch = 0.0
        output.append(
            {
                **segment,
                "emotion": {
                    "source": "heuristic_local",
                    "notice": "Acoustic estimate, not a reliable reading of emotion.",
                    "valence": round(min(1.0, max(-1.0, (pitch - 150) / 150)), 3),
                    "arousal": round(min(1.0, max(0.0, rms * 8)), 3),
                    "features": {"rms": round(rms, 5), "pitch_hz": round(pitch, 2)},
                },
            }
        )
    return output


def _srt_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_diarized_srt(segments: list[dict[str, Any]], path: Path) -> None:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        start = float(segment.get("start") or 0)
        end = float(segment.get("end") or start)
        lines.extend(
            [
                str(index),
                f"{_srt_timestamp(start)} --> {_srt_timestamp(end)}",
                f"[{segment.get('speaker') or 'UNKNOWN'}]: {str(segment.get('text') or '').strip()}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def diarize_and_label(
    audio_path: str | Path,
    *,
    whisper_json_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_id: str = DEFAULT_MODEL,
    emotion_heuristics: bool = False,
) -> dict[str, Any]:
    audio = Path(audio_path).expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio}")
    output = Path(output_dir).resolve() if output_dir else default_output_dir(audio)
    started_at = _utc_now()
    write_status(
        output,
        {
            "status": "running",
            "phase": "diarize",
            "audio_path": str(audio),
            "model": model_id,
            "pid": os.getpid(),
            "started_at": started_at,
        },
    )
    turns = run_pyannote_diarization(audio, model_id)
    turns_path = output / f"{audio.stem}_diarization_turns.json"
    turns_path.write_text(
        json.dumps({"turns": turns}, indent=2) + "\n", encoding="utf-8"
    )
    outputs = {"turns": str(turns_path)}
    segments: list[dict[str, Any]] = []
    transcript_path = (
        Path(whisper_json_path).expanduser().resolve()
        if whisper_json_path
        else audio.with_suffix(".json")
    )
    if transcript_path.is_file():
        raw = json.loads(transcript_path.read_text(encoding="utf-8"))
        segments = assign_speakers_to_segments(list(raw.get("segments") or []), turns)
        if emotion_heuristics:
            segments = emotion_heuristics_for_segments(audio, segments)
        diarized = {
            **raw,
            "segments": segments,
            "diarization": {
                "pipeline": "pyannote",
                "model": model_id,
                "emotion": "heuristic_local" if emotion_heuristics else "disabled",
            },
        }
        json_path = output / f"{audio.stem}_diarized.json"
        srt_path = output / f"{audio.stem}_diarized.srt"
        json_path.write_text(
            json.dumps(diarized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        write_diarized_srt(segments, srt_path)
        outputs.update({"diarized_json": str(json_path), "diarized_srt": str(srt_path)})
    speakers = sorted({str(turn["speaker"]) for turn in turns})
    result = {
        "status": "completed",
        "phase": "done",
        "audio_path": str(audio),
        "whisper_json_path": (
            str(transcript_path) if transcript_path.is_file() else None
        ),
        "model": model_id,
        "emotion_heuristics": emotion_heuristics,
        "pid": os.getpid(),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "speaker_count": len(speakers),
        "speakers": speakers,
        "turn_count": len(turns),
        "segment_count": len(segments),
        "outputs": outputs,
        "error": None,
    }
    write_status(output, result)
    return result


def start_diarization_job(
    audio_path: str | Path,
    *,
    whisper_json_path: str | Path | None = None,
    model_id: str = DEFAULT_MODEL,
    emotion_heuristics: bool = False,
) -> dict[str, Any]:
    audio = Path(audio_path).expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio}")
    output = default_output_dir(audio)
    existing = status_for(output)
    if existing.get("status") == "running" and existing.get("pid"):
        try:
            os.kill(int(existing["pid"]), 0)
            return {
                "status": "already_running",
                "pid": existing["pid"],
                "output_dir": str(output),
            }
        except ProcessLookupError:
            pass
    command = [
        sys.executable,
        "-m",
        "services.local_audio_analysis",
        "--audio",
        str(audio),
        "--output-dir",
        str(output),
        "--model",
        model_id,
    ]
    if whisper_json_path:
        command.extend(
            ["--whisper-json", str(Path(whisper_json_path).expanduser().resolve())]
        )
    if emotion_heuristics:
        command.append("--emotion-heuristics")
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / LOG_FILENAME
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n--- start {_utc_now()} ---\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=os.environ.copy(),
        )
    (output / PID_FILENAME).write_text(str(process.pid), encoding="utf-8")
    payload = {
        "status": "running",
        "phase": "started",
        "audio_path": str(audio),
        "model": model_id,
        "emotion_heuristics": emotion_heuristics,
        "pid": process.pid,
        "started_at": _utc_now(),
        "output_dir": str(output),
        "status_path": str(status_path(output)),
        "log_path": str(log_path),
    }
    write_status(output, payload)
    return payload


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local pyannote diarization worker")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--whisper-json")
    parser.add_argument("--output-dir")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--emotion-heuristics", action="store_true")
    arguments = parser.parse_args(argv)
    output = (
        Path(arguments.output_dir)
        if arguments.output_dir
        else default_output_dir(arguments.audio)
    )
    try:
        result = diarize_and_label(
            arguments.audio,
            whisper_json_path=arguments.whisper_json,
            output_dir=output,
            model_id=arguments.model,
            emotion_heuristics=arguments.emotion_heuristics,
        )
        print(json.dumps({"ok": True, "result": result}, indent=2))
        return 0
    except Exception as exc:
        logger.exception("Local audio analysis failed")
        write_status(
            output,
            {
                "status": "failed",
                "phase": "error",
                "audio_path": str(Path(arguments.audio).expanduser().resolve()),
                "model": arguments.model,
                "pid": os.getpid(),
                "error": str(exc)[:2000],
                "failed_at": _utc_now(),
            },
        )
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
