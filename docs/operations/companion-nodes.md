<!--
title: Companion Nodes
summary: Pair an outward-only device for Apple apps, local secrets, commands, screen capture, or wake word
read_when:
  - "You want Hexis to run an explicitly allowlisted command on your computer"
  - "You want Hexis to use Apple Reminders, Notes, Calendar, Shortcuts, or 1Password on a Mac"
  - "You want to give Hexis fresh visual context from a screen"
  - "You want an explicitly enabled local wake-word listener"
  - "A companion-node pairing or invocation failed"
section: operations
-->

# Companion Nodes

A companion node is a headless `hexis node run` process that can give Hexis
private-host capabilities a browser cannot provide:

- `system.run` executes one locally allowlisted executable and its fixed arguments.
- `screen.capture` takes one screenshot and returns it as visual model context.
- Structured actions use Apple Reminders, Notes, Calendar, and Shortcuts on a Mac.
- 1Password exposes redacted item metadata or copies one field to the Mac's local
  clipboard without returning the secret to Hexis.
- `audio.wake` detects a locally selected wake model, captures one bounded
  post-cue utterance, and plays the response.

The node opens an outbound WebSocket to the Hexis API. It never listens on a port.
Its Ed25519 identity must be approved before it can connect, every model-initiated
invocation remains behind the normal tool-approval gate, and `system.run` is also
restricted by the allowlist stored on the node itself. Commands use direct argv
execution; there is no shell expansion.

## Before you start

The Hexis API must be running and migration 0224 or later must be applied:

```bash
hexis up
hexis migrate
```

For a node on the Hexis host, the default gateway is
`http://127.0.0.1:43817`. For a different device, keep port 43817 bound to
loopback and put it behind a private tailnet HTTPS route or a TLS reverse proxy
with its own authentication. OSS Hexis has no application authentication layer:
never bind the API publicly, forward its port from a router, or put it behind
Tailscale Funnel. See [Secure Phone and PWA Access](secure-remote-access.md) for
the same network-boundary requirements.

`hexis tunnel start` manages the dashboard route only. A remote node can use the
same tailnet hostname as its gateway, but its signed identity still requires the
separate comparison-and-approval flow below; tailnet membership never grants host
capabilities by itself.

On macOS, the node derives its advertised capabilities from the executables that
are actually installed. `osascript` enables Reminders, Notes, and Calendar;
`shortcuts` enables Shortcuts; and `op` enables redacted 1Password item listing.
Local secret copy also needs `pbcopy`. `hexis node status --local-only` shows the
current derived set. Installing or removing one of these programs does not change
a running process: restart `hexis node run` and approve any added capabilities.

The first Apple action may prompt macOS for Automation access to the relevant app.
Approve it for the terminal or host service that runs the node. For 1Password,
install the official `op` CLI and complete `op signin` locally before starting the
node. Hexis never accepts a 1Password account password or session token.

## 1. Create this device's identity

Run this on the device that will provide host capabilities:

```bash
hexis node init --name "Eric's MacBook"
hexis node status --local-only
```

The private identity is stored at
`${XDG_CONFIG_HOME:-~/.config}/hexis/node.json` with mode `0600`. Back it up as
sensitive device state. `hexis node status` shows the public fingerprint but
never prints the private key. Initialization refuses to overwrite an existing
identity.

## 2. Allow only the commands you need

Each alias pins an executable to the absolute path resolved at allow time.
Arguments after the executable are fixed and are stored in the node file, so do
not put passwords, tokens, or other secrets there.

For a command with no invocation-time arguments:

```bash
hexis node allow show-calendar -- /usr/bin/osascript /Users/you/bin/show-calendar.scpt
```

To permit additional arguments explicitly:

```bash
hexis node allow shortcuts --allow-args -- /usr/bin/shortcuts run
```

Inspect or remove policy locally:

```bash
hexis node status --local-only
hexis node disallow shortcuts
```

An existing alias is never changed silently. Review the new argv, then use
`--replace` if replacement is intentional. Invocation arguments remain disabled
unless `--allow-args` was chosen.

## 3. Connect and approve the exact identity

Keep this process open on the node:

```bash
hexis node run --gateway http://127.0.0.1:43817
```

For a remote private HTTPS endpoint, pass its `https://…` base URL; Hexis derives
the `wss://…/api/nodes/connect` endpoint. You can also set
`HEXIS_NODE_GATEWAY_URL` (then `HEXIS_API_URL` is the fallback).

The first signed connection displays an eight-character pairing code and waits
in place. Approve the card in the Conversation inbox after comparing its full
node fingerprint, device name, code, and capabilities. Or review it on the
Hexis host:

```bash
hexis node pairing list
hexis node pairing approve ABCD1234
```

To reject it instead:

```bash
hexis node pairing deny ABCD1234 --note "Device not recognized"
```

Approval applies only to that signing identity and the capabilities shown. A
later capability addition—such as enabling `audio.wake` on an existing node—files
a fresh approval and cannot connect with the expanded access until accepted. A second live process cannot
take over the same identity; after an unclean disconnect, a stale session can be
reclaimed after 30 seconds.

## Optional: enable wake word on this node

Wake listening is off by default and is local policy, separate from pairing. Run:

```bash
hexis node wake setup
```

The setup installs the optional detector/audio packages, then derives the
pretrained choices from the installed openWakeWord catalog. It does not choose a
model silently. Upstream pretrained models are English-only and licensed CC
BY-NC-SA 4.0; setup displays that constraint and requires an explicit acceptance
before downloading one. To use a model under different terms, pass an absolute
custom `.onnx` or `.tflite` path with `--model`.

Then enable **Paired-node wake-word turns**, voice-note transcription, and local
speech under **Settings → Voice**. Restart `hexis node run`. The new
`audio.wake` capability may require a fresh pairing approval even when the signed
device was already paired.

Detection stays on the node. After detection, Hexis closes the detector stream,
plays a cue, opens the microphone for one silence/max-duration-bounded utterance,
then closes it while the signed audio is transcribed, answered, synthesized, and
returned. Raw audio, transcript text, and response text are excluded from the wake
audit. Use `hexis node wake status` for a read-only check and `hexis node wake
disable` to prevent capture on future node starts. Ctrl+C closes a currently
running foreground node and its microphone immediately.

## 4. Use the node

In chat, ask Hexis to use the paired device. The `host-node` skill exposes six
approval-gated tools. Every call names the exact paired node, operation, and
arguments; even list and search operations require a fresh approval because they
read private host data.

| Tool | Operations | Important arguments |
|---|---|---|
| `apple_reminders` | `list`, `create` | list name, title, notes, timezone-bearing due time |
| `apple_notes` | `search`, `create` | query or title/body, optional exact folder |
| `apple_calendar` | `list`, `create` | timezone-bearing start/end, optional exact calendar |
| `apple_shortcuts` | `list`, `run` | exact Shortcut name for `run` |
| `onepassword_local` | `list_items`, `copy_field` | redacted query/vault, or exact `op://vault/item/field` reference |
| `node_invoke` | `system.run`, `screen.capture` | local command alias, or no capture arguments |

The Apple tools choose fixed source-controlled JXA programs and pass user text as
direct process arguments. They do not accept AppleScript or JavaScript from the
model. Shortcuts are invoked by exact name with the macOS CLI; Hexis does not feed
arbitrary text or temporary files into them.

`onepassword_local list_items` returns only item id, title, category, vault, and
update time. It never returns fields. `copy_field` runs `op read` on the node and
pipes the bytes directly to `pbcopy`; the result sent through the gateway contains
only a success receipt. By default the node clears that clipboard value after 60
seconds, and only if the clipboard still contains the same value, so newer clipboard
content is never erased. The approved call can choose 10–300 seconds.

A screen capture is attached directly to the model's turn as visual context; raw
image bytes are not copied into ordinary tool text or events.

For an operator-driven diagnostic, copy the complete node id from `hexis node
status` and invoke it directly. Both commands require typed confirmation unless
`--yes` is explicit:

```bash
hexis node invoke NODE_ID system.run --command show-calendar
hexis node invoke NODE_ID screen.capture --output ./screen.png
```

Repeat `--arg VALUE` only for an alias created with `--allow-args`. A CLI capture
without `--output` is stored under
`${XDG_CACHE_HOME:-~/.cache}/hexis/node/captures/` with mode `0600`.
An existing output is never replaced unless `--overwrite` is explicit.

On macOS, grant Screen Recording permission to the terminal or host service that
runs the node. On Linux, install `grim` or `gnome-screenshot`. Restart the node
after changing its command policy, installing a capability provider, or changing
host permissions.

## Stop or revoke access

Ctrl+C stops the process and takes the node offline without changing its pairing
or local allowlist. To permanently reject that signed identity, run on the Hexis
host:

```bash
hexis node revoke NODE_ID --reason "Device retired"
```

Revocation cancels queued work and is intentionally irreversible for that
identity. If the device should return later, remove the local identity only after
revocation, run `hexis node init` to create a new one, and approve the new
fingerprint explicitly.

## Troubleshooting

| Symptom | Cause and exact next step |
|---|---|
| `No node identity exists` | Run `hexis node init --name <device-name>` on the node. |
| Identity is readable by other users | Run `chmod 600 ~/.config/hexis/node.json`, or use the printed XDG path. |
| Pairing stays pending | Keep `hexis node run` open, then approve the matching code in Conversation or with `hexis node pairing approve <code>`. New capabilities require a fresh approval; requests expire after 24 hours by default. |
| `already_connected` | Stop the other process using this identity, or wait 30 seconds after its last heartbeat and retry. Do not copy one node file to several devices. |
| Node is offline | Start `hexis node run --gateway <url>` on that device, wait until it says connected, then retry the invocation. |
| Command is not allowlisted | Run `hexis node allow <alias> -- <executable> ...` locally and restart the node. |
| Allowlisted executable no longer runs | Reinstall or locate it, then explicitly update the alias with `hexis node allow <alias> --replace -- <executable> ...`. |
| Screenshot fails on macOS | Grant Screen Recording permission to the process running the node, restart it, and retry. |
| An Apple app action is absent | Run `hexis node status --local-only`. Verify the required macOS executable exists, restart `hexis node run`, and approve the newly advertised capability if prompted. |
| An Apple app denies automation | Grant Automation access to the terminal or installed Hexis host service in macOS Privacy & Security, restart the node, and retry the exact call. |
| 1Password is absent or signed out | Install the official `op` CLI, run `op signin` locally on the node, restart `hexis node run`, and approve the added capability. Do not send credentials through chat. |
| A copied secret is not returned in chat | This is intentional: the secret exists only on the node clipboard. Paste it locally before its approved expiry. |
| Remote connection fails | Verify the private HTTPS route reaches the loopback API health endpoint, then pass that base URL with `--gateway`; do not expose the API publicly. |
| Wake support cannot open audio on Linux | Install the PortAudio runtime (often `libportaudio2`), verify the input device with `hexis node wake setup --device NAME`, then restart the node. |
| Wake detection works but no answer plays | Enable `node_wake` in any non-empty STT channel allowlist, verify `hexis voice status`, and use the exact recovery printed by the node. The written turn remains in conversation history. |
| Identity was denied or revoked | Reusing it is intentionally blocked. Revoke any superseded record, create a new local identity, and approve its new fingerprint. |
