from __future__ import annotations

import json
from pathlib import Path

from services.local_audio_analysis import (
    assign_speakers_to_segments,
    default_output_dir,
    status_for,
    turns_from_annotation,
    write_diarized_srt,
    write_status,
)


class Segment:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class Annotation:
    def itertracks(self, yield_label: bool = False):
        assert yield_label
        yield Segment(0, 1), None, "SPEAKER_00"
        yield Segment(1, 2), None, "SPEAKER_01"


def test_turns_and_segments_are_labeled_by_overlap():
    turns = turns_from_annotation(Annotation())
    labeled = assign_speakers_to_segments(
        [{"start": 0.1, "end": 0.9, "text": "hello"}], turns
    )
    assert labeled[0]["speaker"] == "SPEAKER_00"
    assert labeled[0]["speaker_overlap_s"] == 0.8


def test_generated_artifacts_default_to_hexis_cache(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    audio = tmp_path / "source" / "meeting.wav"
    expected = tmp_path / "cache" / "hexis" / "audio-analysis"
    output = default_output_dir(audio)
    assert output.parent == expected
    assert output.name.startswith("meeting-")


def test_status_round_trip_and_missing_next_step(tmp_path: Path):
    missing = status_for(tmp_path / "missing")
    assert missing["status"] == "missing"
    assert "Start analysis" in missing["error"]
    write_status(tmp_path, {"status": "completed", "speaker_count": 2})
    status = status_for(tmp_path)
    assert status["speaker_count"] == 2
    assert "updated_at" in json.loads((tmp_path / "diarize_status.json").read_text())


def test_srt_keeps_speaker_labels(tmp_path: Path):
    output = tmp_path / "meeting.srt"
    write_diarized_srt(
        [{"start": 0, "end": 1.25, "speaker": "SPEAKER_00", "text": " Hello "}],
        output,
    )
    text = output.read_text()
    assert "00:00:00,000 --> 00:00:01,250" in text
    assert "[SPEAKER_00]: Hello" in text
