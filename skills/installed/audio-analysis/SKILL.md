---
name: audio-analysis
description: Analyze a local recording for speakers and label timestamped transcript segments without uploading audio
category: knowledge
requires:
  tools: [transcribe, analyze_local_audio]
contexts: [heartbeat, chat]
aliases: [audio, recording, diarize, diarization, speakers, transcript, voice]
bound_tools: [transcribe, analyze_local_audio]
---

# Local Audio Analysis

Use device-local speaker diarization when the user asks who spoke when in a recording.

For a plain speech-to-text request, use `transcribe`; it follows the local/cloud
choice in Settings and does not require a diarization job.

## Method

1. Confirm the exact local audio path and, when available, a Whisper JSON path with timestamped segments.
2. Call `analyze_local_audio` with `action=start`. The tool requires operator approval and never uploads audio.
3. Report the returned cache directory and immediately explain that analysis continues in the background.
4. Poll with `action=status` and the same `audio_path` until the job completes or fails.
5. On completion, use the returned diarized JSON or SRT artifact. Preserve `SPEAKER_*` labels unless the user supplies identities.
6. If the job fails, report its cause and exact next step from the status. Never present acoustic heuristics as a reliable reading of emotion.

## Boundaries

- Do not enable autonomous audio analysis without the user's explicit choice.
- Do not copy generated artifacts into a source repository; they belong in the Hexis cache.
- Only request `emotion_heuristics` when the user asks for them. Label the result as a coarse local acoustic estimate.
