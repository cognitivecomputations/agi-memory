<!--
title: Secure Phone and PWA Access
summary: Reach the installable Hexis dashboard over private Tailscale HTTPS
read_when:
  - "You want to install Hexis on a phone or another computer"
  - "Push notifications or microphone capture say HTTPS is required"
  - "You need a private remote-access path for the OSS dashboard"
section: operations
-->

# Secure Phone and PWA Access

The open-source dashboard has no application authentication layer. Keep every
Hexis port bound to loopback and put remote access behind a network boundary.
The recommended path is `hexis tunnel`, which configures Tailscale Serve to
terminate browser-trusted HTTPS, proxy only to the live loopback dashboard port,
and remain reachable only from your tailnet.

Do not set `HEXIS_BIND_ADDRESS=0.0.0.0`, forward port 3477 on a router, or use
Tailscale Funnel. Those make an unauthenticated personal dashboard reachable
outside the intended private boundary. A reverse proxy is also acceptable only
when the proxy supplies its own authentication.

## Why HTTPS is required

`http://localhost:3477` is a browser secure-context exception, so local
development works. A phone opening `http://192.168.x.x:3477` is not a secure
context: the service worker, Web Push, install prompt, and microphone are all
unavailable. Tailscale Serve gives the phone a trusted `https://…ts.net` origin
without changing Hexis's loopback binds.

## 1. Join the host and phone to one tailnet

1. Install Tailscale on the Hexis host and the phone.
2. Sign both devices into the same tailnet.
3. In the Tailscale admin console's DNS page, enable MagicDNS and HTTPS
   certificates.
4. Review the certificate-transparency disclosure before enabling HTTPS. The
   machine's full `*.ts.net` name is published in the public certificate log,
   although the machine itself remains private to the tailnet. Rename a machine
   first if its hostname contains sensitive information.

Tailscale's current certificate and Serve behavior is documented in
[Enabling HTTPS](https://tailscale.com/docs/how-to/set-up-https-certificates) and
the [`tailscale serve` reference](https://tailscale.com/docs/reference/tailscale-cli/serve).

Tailnet admission is the PWA device-approval boundary: a phone must be approved
in Tailscale before it can reach Hexis. This does not add an OSS application-auth
layer. Companion nodes additionally require their independent signed-identity
pairing flow.

## 2. Start Hexis and private HTTPS in one command

Run this on the Hexis host:

```bash
hexis tunnel start
hexis tunnel status
hexis doctor
```

If the local dashboard is down, `hexis tunnel start` starts the normal stack and
waits for the real HTTP endpoint before changing Tailscale. It derives
`HEXIS_UI_PORT`, pins the proxy target to `127.0.0.1`, and records only the route
it created in mode-0600 ownership state. It refuses to:

- operate while `HEXIS_BIND_ADDRESS` is broader than loopback;
- enable or preserve Tailscale Funnel for the Hexis target;
- replace an unrelated root Serve handler;
- claim an equivalent route that someone else configured.

Serve provisions and renews its TLS certificate through the Tailscale daemon.
The command prints the private URL, normally:

```text
https://machine-name.your-tailnet.ts.net
```

Open that exact URL on the phone. `hexis doctor` reports **Dashboard HTTPS: OK**
only after a trusted HTTPS request reaches the dashboard. It reports **Remote
exposure: FAIL** if Hexis is bound beyond loopback or its target is on Funnel.
If automatic discovery is unavailable, set the authoritative URL for the
diagnostic:

```bash
HEXIS_UI_PUBLIC_URL=https://machine-name.your-tailnet.ts.net hexis doctor
```

Use `--no-start-stack` only when you want the command to leave a stopped local
stack untouched. Do not run `tailscale cert` just to use Serve: Serve manages the certificate.
Use `tailscale cert` only for a different TLS terminator, where you also take
responsibility for renewal.

## 3. Install and enable the capabilities you want

In the HTTPS dashboard:

1. Open **Settings → App**.
2. Choose **Install Hexis**, or use the browser's Install/Add to Home Screen
   command when the platform owns the prompt.
3. Choose **Enable notifications** only if wanted. The browser asks permission
   in response to that click; message previews are hidden by default.
4. Configure **Settings → Voice notes** before using the microphone button in
   Conversation. Local and cloud transcription retain the same explicit
   provider choice and cloud disclosure as channel voice notes.

On iOS 16.4 and later, push requires Add to Home Screen and opening the installed
app first. The PWA has no background listening, wake word, Siri integration, host
commands, or Web Share Target. An uninstalled iOS website can also lose storage
after an extended period of non-use.

## Turn access or notifications off

- In **Settings → App**, choose **Turn off notifications** for each browser.
- To remove the private HTTPS route created by Hexis, run `hexis tunnel stop` on
  the host. It removes only the owned `/` handler and preserves other Serve paths.
  If a matching route was configured outside Hexis, the command refuses to
  remove it; inspect it with `tailscale serve status` and disable that exact route manually.
- To remove a device completely, sign it out of Tailscale or revoke it in the
  tailnet admin console.

These actions do not delete the Hexis database, identity, memories, or local
dashboard. Re-enabling is an explicit choice on the same screens.

## Troubleshooting

| Symptom | Cause and next step |
|---|---|
| Install, push, or microphone controls are unavailable | Confirm the address begins with `https://` (or is localhost), then run `hexis doctor`. |
| `hexis tunnel start` says Tailscale is disconnected | Run `tailscale up`, complete sign-in in place, then rerun the Hexis command. |
| `hexis doctor` says Tailscale is connected but Serve is missing | Enable MagicDNS/HTTPS in the admin console, then run `hexis tunnel start`. |
| `hexis doctor` reports Remote exposure **FAIL** | Restore `HEXIS_BIND_ADDRESS=127.0.0.1`. If Funnel names the Hexis target, run `hexis tunnel stop` for an owned route or disable that exact Funnel route manually. |
| `hexis tunnel start` finds an unrelated root handler | Hexis will not overwrite it. Keep that handler and configure an authenticated reverse proxy manually, or explicitly remove it with Tailscale before retrying. |
| HTTPS opens but the dashboard errors | Run `hexis logs -f api ui`; verify the derived `http://127.0.0.1:${HEXIS_UI_PORT:-3477}/chat` target locally. |
| iPhone receives no push | Add the site to the Home Screen, open that installed app, then enable notifications from Settings → App. |
| Voice capture records but transcription fails | Open Settings → Voice notes; choose a provider and follow the exact dependency/key guidance shown there. |
