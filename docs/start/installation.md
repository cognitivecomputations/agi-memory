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

To update later: `uv tool upgrade hexis` (or use `hexis upgrade`, which also refreshes the Docker images and migrates the database).

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

```bash
git clone https://github.com/QuixiAI/Hexis.git && cd Hexis
uv venv && source .venv/bin/activate
uv pip install -e .
cp .env.local .env   # edit with your settings
```

No uv? A plain virtualenv works too: `python3 -m venv .venv && source .venv/bin/activate && pip install -e .`

If build isolation fails in a restricted environment:

```bash
pip install -e . --no-build-isolation
```

## Environment Configuration

Create a `.env` file (automatically created by `hexis init` for packaged installs):

```bash
POSTGRES_DB=hexis_memory
POSTGRES_USER=hexis_user
POSTGRES_PASSWORD=hexis_password
POSTGRES_HOST=localhost
POSTGRES_PORT=43815
HEXIS_BIND_ADDRESS=127.0.0.1    # Set to 0.0.0.0 to expose services
```

If port `43815` is already in use, set `POSTGRES_PORT` to any free port.

For LLM API keys (if using API-key providers):

```bash
OPENAI_API_KEY=sk-...           # OpenAI Platform
ANTHROPIC_API_KEY=sk-ant-...    # Anthropic
```

See [Environment Variables](../operations/environment-variables.md) for the complete reference.

## Start the Stack

```bash
hexis up         # starts PostgreSQL, RabbitMQ, heartbeat worker, and maintenance worker
hexis doctor     # verify everything is healthy
```

The CLI auto-detects whether you're running from source or a packaged install and uses the appropriate Docker Compose file.

## Verify It Worked

```bash
hexis status     # should show database connected, agent not yet configured
hexis doctor     # checks Docker, DB, and embedding service health
```

## Next Steps

- [First Agent](first-agent.md) -- configure your agent's identity and personality
