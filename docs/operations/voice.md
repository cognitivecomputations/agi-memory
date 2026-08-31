<!--
title: Voice and Talk Mode
summary: Local speech output, foreground conversation, and voice-note transcription
read_when:
  - "You want Hexis to speak"
  - "You want a hands-free foreground conversation"
  - "Local voice setup or playback is failing"
section: operations
-->

# Voice and Talk Mode

Hexis has three independently useful voice layers:

- inbound voice-note and PWA transcription;
- opt-in local speech output through Piper;
- foreground Talk mode in the web app.

The browser has no background microphone. Talk mode starts only after you press
**Talk**, releases the microphone while Hexis thinks and speaks, and stops when
the page leaves the foreground. Always-on detection is a separate, explicitly
configured companion-node capability described below.

## Set up local speech

1. Start Hexis with `hexis up`.
2. Open **Settings → Voice**.
3. Enable **Local speech output**. Enable **Foreground Talk mode** too if wanted,
   then save.
4. In a terminal, run:

   ```bash
   hexis voice setup
   ```

   Confirm the optional dependency install. Hexis reads the selected model from
   the live database, starts Piper on `127.0.0.1:42667`, and waits for its real
   readiness endpoint. The first start may take longer while Piper downloads the
   selected voice; the process and download are left running if the CLI wait
   expires, so no progress is discarded.
5. Verify the result with `hexis voice status`, then press **Refresh status** in
   Settings.

`hexis up` and `hexis dev` start the sidecar on later runs when speech remains
enabled. `hexis down` stops only the exact process Hexis recorded as its own.
An already-running compatible provider is never adopted or stopped.

## Hear a response

Ask Hexis to read or say something aloud. The optional `speak` tool adds an
audio player to the conversation while keeping the written response available.
Tool-created audio is addressed by an opaque ID, is never cached by the HTTP
layer, and expires after the configured retention period. The synthesis audit
stores counts, provider, model, timing, and outcome—not input text or audio.

## Use Talk mode

Talk mode also needs transcription enabled under **Settings → Voice**. Cloud
transcription remains a separate disclosed choice; local Whisper is the default.

Open Chat and press **Talk**. The active card shows whether Hexis is listening,
transcribing, thinking, or speaking. Natural silence sends an utterance; **Send
now** ends it manually. **Stop** always releases capture. A failure pauses the
loop in place, preserves the written transcript/response, and offers **Resume**.

Microphone APIs require a secure context away from localhost. For a phone or
another computer, use `hexis tunnel start` and open the private HTTPS tailnet URL;
see [Secure Phone and PWA Access](secure-remote-access.md).

## Lifecycle commands

| Command | Effect |
|---------|--------|
| `hexis voice` | Read-only status |
| `hexis voice setup [-y]` | Confirm/install optional Piper support and start the live configured model |
| `hexis voice start` | Start without changing installed packages |
| `hexis voice status [--json]` | Inspect provider, model, ownership, endpoint, and log path |
| `hexis voice stop` | Stop only the exact Hexis-owned process |

Ownership state is mode `0600` at `~/.hexis/voice-sidecar.json`; logs are at
`~/.hexis/voice-sidecar.log`. A stale PID or mismatched launch token fails closed
and leaves the process alone for review.

## Optional paired-node wake word

On the companion device, initialize its identity and run `hexis node wake setup`.
Setup installs optional openWakeWord/sounddevice support but does not open the
microphone. It derives pretrained choices from the installed catalog and requires
explicit acceptance of their CC BY-NC-SA 4.0, English-only terms; an absolute
custom ONNX/TFLite path can be selected instead.

In **Settings → Voice**, enable voice-note transcription, local speech output,
and **Paired-node wake-word turns**. Then run or restart:

```bash
hexis node run --gateway <private-hexis-url>
```

Pairing shows `audio.wake`; adding it to an already-paired identity requires a
fresh approval. Detection is local. A cue separates the wake phrase from the
bounded utterance sent over the signed outward WebSocket, and the microphone is
closed while Hexis transcribes, thinks, synthesizes, and plays the answer. The
complete text turn remains in conversation history. See [Companion
Nodes](companion-nodes.md) for model, device, and revocation details.

## Troubleshooting

- **Piper is missing:** run `hexis voice setup`; it installs into the exact Python
  environment that owns the `hexis` command.
- **First start is still loading:** follow `~/.hexis/voice-sidecar.log`, then rerun
  `hexis voice status`. Do not start a second copy.
- **Settings says the provider is unavailable:** verify status, run `hexis voice
  setup`, then press **Refresh status**.
- **A remote device cannot open the microphone:** use the trusted HTTPS URL from
  `hexis tunnel start`, not a LAN HTTP address.
- **Talk mode is paused:** read the in-place error, apply its stated fix, and press
  **Resume Talk mode**. The text conversation remains usable throughout.

`HEXIS_TTS_URL` is an advanced override and is intentionally restricted to
credential-free HTTP endpoints on loopback or `host.docker.internal`. Hexis
refuses remote hosts so private response text cannot silently leave the device.
