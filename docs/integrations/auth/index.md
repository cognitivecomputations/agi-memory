<!--
title: Auth Providers
summary: OAuth, device code, and token-based LLM authentication
read_when:
  - "You want to set up LLM authentication"
  - "You want to use a free LLM provider"
section: integrations
-->

# Auth Providers

Hexis supports multiple LLM providers via OAuth, device-code, or token-based authentication. All credentials are stored in the Postgres `config` table and refreshed automatically.

## Provider Matrix

| Provider | Auth Type | DB Config Key | Cost |
|----------|-----------|---------------|------|
| `openai-codex` | OAuth PKCE | `oauth.openai_codex` | ChatGPT Plus/Pro subscription |
| `anthropic` | Setup token | `token.anthropic_setup_token` | Claude subscription or API key |
| `chutes` | OAuth PKCE | `oauth.chutes` | Free |
| `github-copilot` | Device code | `oauth.github_copilot` | Copilot subscription |
| `qwen-portal` | Device code + PKCE | `oauth.qwen_portal` | Free tier available |
| `minimax-portal` | User code + PKCE | `oauth.minimax_portal` | Free tier available |
| `google-gemini-cli` | OAuth PKCE | `oauth.google_gemini_cli` | Free tier available |
| `google-antigravity` | OAuth PKCE | `oauth.google_antigravity` | Free tier available |

All providers can be used for any LLM slot (`llm.chat`, `llm.heartbeat`, `llm.subconscious`).

API-key providers (`openai`, `anthropic`, `grok`, `gemini`) can also use traditional environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) instead of OAuth.

For a custom OpenAI-compatible endpoint, put the endpoint and key in the
project's ignored `.env` file:

```dotenv
OPENAI_BASE_URL=https://your-inference-server.example/v1
OPENAI_API_KEY=replace-with-your-key
```

Then configure Hexis without exposing the key in shell history:

```bash
hexis init --provider openai_compatible --model your-model-id --character hexis \
  --api-key-env OPENAI_API_KEY
```

The endpoint and environment variable name are stored in the brain; the key
itself remains in `.env`. Use `OPENAI_API_KEY=not-needed` for an unauthenticated
server because the OpenAI client still requires a value.

---

## OpenAI Codex (ChatGPT Subscription)

Uses your ChatGPT Plus/Pro subscription via OAuth + PKCE. **Not** OpenAI Platform API-key auth.

```bash
hexis auth openai-codex login           # opens browser for OAuth
hexis auth openai-codex status [--json]
hexis auth openai-codex logout [--yes]
```

Options: `--no-open` (print URL instead), `--timeout-seconds N` (default: 60)

Configure: `llm.chat.provider = "openai-codex"`, `llm.chat.model = "gpt-5.2"`

---

## Anthropic (Setup Token)

Uses a Claude Code setup token for Bearer auth. No paid API key required with a Claude subscription.

```bash
hexis auth anthropic setup-token        # prompts for token (sk-ant-oat01-...)
hexis auth anthropic status [--json]
hexis auth anthropic logout [--yes]
```

Options: `--token TOKEN` (pass directly)

Configure: `llm.chat.provider = "anthropic"`, `llm.chat.model = "claude-sonnet-4-20250514"`

When no `ANTHROPIC_API_KEY` env var is set, Hexis automatically uses the stored setup token.

---

## Chutes

Free LLM inference via Chutes OAuth.

```bash
hexis auth chutes login
hexis auth chutes status [--json]
hexis auth chutes logout [--yes]
```

Options: `--no-open`, `--timeout-seconds N` (default: 120), `--manual` (paste URL), `--client-id ID`

Configure: `llm.chat.provider = "chutes"`, `llm.chat.model = "deepseek-ai/DeepSeek-V3-0324"`

Default endpoint: `https://api.chutes.ai/v1`

---

## GitHub Copilot

Device code auth using your GitHub Copilot subscription.

```bash
hexis auth github-copilot login         # prints user code, opens github.com/login/device
hexis auth github-copilot status [--json]
hexis auth github-copilot logout [--yes]
```

Options: `--enterprise-domain DOMAIN` (default: `github.com`), `--timeout-seconds N` (default: 120)

Configure: `llm.chat.provider = "github-copilot"`, `llm.chat.model = "gpt-4o"`

---

## Qwen Portal

Free access to Qwen models via device code auth.

```bash
hexis auth qwen-portal login
hexis auth qwen-portal status [--json]
hexis auth qwen-portal logout [--yes]
```

Options: `--timeout-seconds N` (default: 120)

Configure: `llm.chat.provider = "qwen-portal"`, `llm.chat.model = "qwen-max-latest"`

Default endpoint: `https://portal.qwen.ai/v1`

---

## MiniMax Portal

Access MiniMax models via user-code + PKCE auth.

```bash
hexis auth minimax-portal login
hexis auth minimax-portal status [--json]
hexis auth minimax-portal logout [--yes]
```

Options: `--region global|cn` (default: `global`), `--timeout-seconds N` (default: 120)

Configure: `llm.chat.provider = "minimax-portal"`, `llm.chat.model = "MiniMax-M1"`

Default endpoints: Global `https://api.minimax.io/anthropic`, CN `https://api.minimaxi.com/anthropic`

---

## Google Gemini CLI

Uses Google Cloud Code Assist via OAuth PKCE (same auth as `gemini` CLI).

**Prerequisites**: Set `GEMINI_CLI_OAUTH_CLIENT_ID` and `GEMINI_CLI_OAUTH_CLIENT_SECRET` env vars.

```bash
hexis auth google-gemini-cli login
hexis auth google-gemini-cli status [--json]
hexis auth google-gemini-cli logout [--yes]
```

Options: `--no-open`, `--timeout-seconds N` (default: 120), `--manual`

Configure: `llm.chat.provider = "google-gemini-cli"`, `llm.chat.model = "gemini-2.5-flash"`

---

## Google Antigravity

Uses Google Cloud Code Assist sandbox via OAuth PKCE.

**Prerequisites**: Set `ANTIGRAVITY_OAUTH_CLIENT_ID` and `ANTIGRAVITY_OAUTH_CLIENT_SECRET` env vars.

```bash
hexis auth google-antigravity login
hexis auth google-antigravity status [--json]
hexis auth google-antigravity logout [--yes]
```

Options: `--no-open`, `--timeout-seconds N` (default: 120), `--manual`

Configure: `llm.chat.provider = "google-antigravity"`, `llm.chat.model = "gemini-2.5-flash"`

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| "Provider X is not configured" | Run `hexis auth <provider> login` |
| "Credentials expired" | Re-run `hexis auth <provider> login` |
| Callback server won't bind | Use `--manual` to paste redirect URL |
| `hexis doctor` shows auth warnings | Run `hexis auth <provider> status` |

## Related

- [Quickstart](../../start/quickstart.md) -- provider setup during init
- [Environment Variables](../../operations/environment-variables.md) -- API key env vars
