<!--
title: Installation
summary: Install Hexis with the install script (recommended), uv, pipx, or pip, or from source; configure environment
read_when:
  - "You want to install Hexis"
  - "You need to set up your .env file"
  - "You want to run from source"
section: start
-->

# Installation

## Install Script (Recommended)

```bash
curl -LsSf https://quixi.ai/hexis.sh | sh
```

One command, works on a machine with nothing but curl and a shell. The script:

1. Installs [uv](https://docs.astral.sh/uv/) if it isn't already present (uv downloads its own Python — no Python install needed)
2. Runs `uv tool install hexis` to put the CLI in an isolated environment
3. Tells you if PATH needs a new terminal, and whether Docker is ready

It is safe to re-run: an existing install is upgraded to the latest release.

## Install with uv

```bash
uv tool install hexis
```

This installs the `hexis` CLI and all dependencies into an isolated environment that uv creates and owns — nothing to activate, and no conflicts with your system Python. If Python 3.10+ isn't on the machine, uv downloads it automatically.

Don't have uv? It's a one-liner: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `brew install uv`).

If `hexis` isn't found afterward, uv's tool directory isn't on your PATH yet — run `uv tool update-shell` and open a new terminal.

To update later: `hexis upgrade` — it updates the CLI package itself (via uv, pipx, or pip, whichever installed it), then refreshes the Docker images and migrates the database. To move just the CLI package by hand: `uv tool install --force hexis`.

The CLI manages Docker containers, the database, and agent configuration.

## Uninstall

Use the CLI so the Python tool and Docker resources are handled together:

```bash
hexis uninstall
```

The default is reversible: Hexis stops and removes its containers, network, and
images, then removes the CLI using the tool that installed it (`uv`, `pipx`, or
`pip`). The brain's Docker volumes and `~/.hexis` configuration, credentials,
skills, artifacts, and backups are preserved. Reinstall Hexis and run
`hexis up` to use that data again.

For a permanent clean removal:

```bash
hexis backup --output "$HOME/hexis-backups"  # optional; keep it outside ~/.hexis
hexis uninstall --purge
```

`--purge` requires an explicit confirmation and permanently deletes the brain
database volumes plus the Hexis data directory, including its default backups
directory. It also removes the standalone `embeddinggemma` binary and model
cache when durable ownership records prove Hexis created them. A legacy,
changed, or independently started companion is surfaced and retained rather
than guessed at and deleted. If Docker is unavailable and you intentionally
want to remove only the CLI while leaving all Docker resources untouched, use
`hexis uninstall --cli-only`.

## Install with pipx

Already a pipx user? It gives you the same isolated-tool experience:

```bash
pipx install hexis
```

Update later with `pipx upgrade hexis`.

## Install with pip

Plain `pip install hexis` only works inside an activated virtualenv — on modern macOS (Homebrew Python) and Debian/Ubuntu, running it against the system Python fails with `error: externally-managed-environment`. If you manage your own environments:

```bash
python3 -m venv ~/.venvs/hexis
source ~/.venvs/hexis/bin/activate
pip install hexis
```

Note that `hexis` is then only on PATH while that virtualenv is active.

## Install from Source

For development or contributing:

### With Mise

The repository's [`mise.toml`](../../mise.toml) provides Python, uv, Bun, the Docker CLI, and Docker Compose. Install [Mise](https://mise.jdx.dev/getting-started.html), then run:

```bash
git clone https://github.com/QuixiAI/Hexis.git && cd Hexis
mise install
mise run setup
mise run docker:check
```

`mise run setup` links the Mise-managed Compose binary into Docker's user plugin directory only when `docker compose` is otherwise unavailable; it never replaces an existing plugin. `docker:check` then verifies Compose through the standard `docker compose` command and uses whichever Docker daemon is active.

On macOS 13 or newer without Docker Desktop, install and start the optional VZ/VirtioFS Colima runtime with `mise run docker:colima`, then rerun the check. On older macOS releases, use Docker Desktop or configure a compatible Docker daemon manually.

### Without Mise

```bash
git clone https://github.com/QuixiAI/Hexis.git && cd Hexis
uv sync --locked --inexact
source .venv/bin/activate
cp .env.example .env   # edit with your settings; never commit .env
```

No uv? A plain virtualenv works too: `python3 -m venv .venv && source .venv/bin/activate && pip install -e .`

If build isolation fails in a restricted environment:

```bash
pip install -e . --no-build-isolation
```

## Environment Configuration

Copy the tracked template, then edit only the settings you need. The resulting
`.env` is ignored by Git:

```bash
cp .env.example .env
```

If port `43815` is already in use, set `POSTGRES_PORT` to any free port.

For a custom OpenAI-compatible server, set its base URL and key in `.env`:

```dotenv
OPENAI_BASE_URL=https://your-inference-server.example/v1
OPENAI_API_KEY=replace-with-your-key
```

Then initialize with the model ID exposed by that server. `--api-key-env` tells
Hexis which variable to read without putting the secret in shell history:

```bash
hexis init --provider openai_compatible --model your-model-id --character hexis \
  --api-key-env OPENAI_API_KEY
```

If the server does not authenticate, use a non-secret placeholder such as
`OPENAI_API_KEY=not-needed`; the OpenAI client still requires a value. Hexis
stores the endpoint and the environment variable's **name** in PostgreSQL, not
the key itself.

See [Environment Variables](../operations/environment-variables.md) for the complete reference.

## Start the Stack

```bash
hexis up         # starts the brain, workers, API, dashboard, and delivery relay
hexis doctor     # verify everything is healthy
```

The CLI auto-detects whether you're running from source or a packaged install and uses the appropriate Docker Compose file.
It starts cached or published images by default; a source checkout only builds
images when you explicitly run `hexis up --build` or enter watch mode with
`hexis dev`.

The dashboard stays available on loopback at `http://127.0.0.1:3477`. To install
it on a phone, keep that loopback bind and run `hexis tunnel start`; the private
[Tailscale HTTPS runbook](../operations/secure-remote-access.md) covers device
approval and verification. Plain LAN HTTP cannot provide the service worker, push,
or microphone.

## Verify It Worked

```bash
hexis status     # should show database connected, agent not yet configured
hexis doctor     # checks Docker, DB, and embedding service health
```

## Next Steps

- [First Agent](first-agent.md) -- configure your agent's identity and personality
