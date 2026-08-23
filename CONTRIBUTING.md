# Contributing to Hexis

Thank you for wanting to make Hexis better. The full contributor guide lives in
the docs:

- **[Contributing overview](docs/contributing/index.md)** — dev setup, project
  layout, coding style, commit conventions
- **[Testing](docs/contributing/testing.md)** — running and writing tests
  (`pytest tests -q` against the Docker stack)
- **[Repository guidelines](CLAUDE.md)** — architecture principles, schema
  migration workflow, and the Experience Bar every user-facing change is held to

Quick version:

```bash
git clone https://github.com/QuixiAI/Hexis.git && cd Hexis
pip install -e . && cp .env.example .env
hexis up && hexis doctor
pytest tests -q
```

Schema changes go through forward-only migrations (`db/migrations/NNNN_*.sql` +
`hexis migrate`) — never ask users to wipe their agent's memories. Bugs and
ideas: [GitHub Issues](https://github.com/QuixiAI/Hexis/issues) /
[Discussions](https://github.com/QuixiAI/Hexis/discussions).

## Where new capability belongs — the footprint ladder

Core is the mind; capability lives at the edges. Before adding anything to
`core/`, walk this ladder top to bottom and stop at the first rung that fits:

1. **Extend something that exists.** A new argument, branch, or category on an
   existing tool or skill beats a parallel one. Check `core/tools/` and
   `skills/installed/` first.
2. **A skill.** If the capability is a *method* — a way of using tools that
   already exist — it's a `SKILL.md`, not code. Skills load from disk; no
   release needed.
3. **A gated tool.** If it needs code and carries risk or cost, it's a registry
   tool with an energy cost, `requires_approval`, and/or a config gate — and it
   must be bound by at least one skill (`tests/core/test_tool_coverage.py`
   enforces reachability; an unbound tool is a hand the agent can't use).
4. **A plugin.** Anything that talks to an external product (task managers,
   social platforms, SaaS APIs) lives in `plugins/installed/<name>/` — a
   `plugin.json` manifest with a tool-ownership contract, a `tools.py`, and a
   bundled skill. `plugins/installed/todoist/` is the reference shape.
5. **An MCP server.** Capability that already exists as an MCP server — or that
   should be usable outside Hexis — connects via MCP rather than being
   vendored in.
6. **A new core tool — last resort.** Only mechanisms of the mind itself
   (memory, identity, consent, energy, heartbeat) belong in `core/`. If the
   capability would make equal sense for an agent with a different personality
   and job, it is not core.

### Not in core — and where it goes instead

Every "not here" is a redirect to its sanctioned home, never a flat refusal:

| You want to add… | Its home |
|---|---|
| An integration with an external product/SaaS | A plugin under `plugins/installed/` |
| A workflow, method, or prompt technique | A skill (`SKILL.md`) |
| A one-off script or computation | The code-execution skill's existing tools (`shell`, `execute_code`) |
| A messaging platform | A channel adapter under `channels/` |
| A new LLM provider | `core/llm.py` (this one *is* core: the mind's voice) |
| A capability the agent should grow herself | Runtime self-extension: `create_tool` + `author_skill` |
