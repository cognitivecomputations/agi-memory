---
name: speech
description: Speak a response aloud through the explicitly configured local voice provider
category: communication
requires:
  tools: [speak]
contexts: [chat]
aliases: [speak, speech, aloud, say, voice, read aloud, pronounce]
bound_tools: [speak]
---

# Speech Output

Use `speak` only when the user asks to hear text aloud or is actively using a
voice conversation. Ordinary text replies do not need duplicate audio.

## Method

1. Keep spoken text concise and natural; remove Markdown syntax that would sound awkward.
2. Call `speak` with the exact words to render.
3. Keep the normal text response available as the accessible transcript.
4. If synthesis is disabled or unavailable, preserve the text and relay the exact setup or recovery step returned by the tool.

## Boundaries

- Never enable voice output or microphone access on the user's behalf.
- Never treat a voice choice as consent to cloud processing; OSS speech output uses the configured local endpoint only.
- Never loop, replay, or start always-on listening from this tool.
