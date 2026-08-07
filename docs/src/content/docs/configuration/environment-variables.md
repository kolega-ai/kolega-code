---
title: Environment Variables
description: Every environment variable Kolega Code reads, and how precedence works.
---

Kolega Code reads configuration from CLI flags, exported environment variables,
and saved Settings. It does **not** automatically load the target project's
`.env` file for its own provider/model/API-key configuration. This page lists the
environment variables it understands when they are explicitly present in the
Kolega Code process environment.

## Precedence

For any given setting, the first available source wins:

1. **CLI flags** (e.g. `--provider`, `--model`)
2. **Exported process environment variables**
3. **Saved Settings** (`settings.json`)

Kolega Code requires an explicit provider/model selection from flags, exported
environment variables, or saved settings. API key variables alone are not a model
selection source.

:::note
A project `.env` file is treated as part of the project being edited. Kolega Code
will not read it for its own LLM configuration. For persistent credentials, use
Settings. For automation, export environment variables in the shell or CI process
that launches Kolega Code.
:::

## API keys

Set the variable for each provider you use. Only the providers backing your active
model roles are required.

API key variables provide credentials only. They do not select the active
provider or model.

| Variable | Provider |
| --- | --- |
| `ANTHROPIC_API_KEY` | Anthropic |
| `OPENAI_API_KEY` | OpenAI |
| `GOOGLE_API_KEY` | Google |
| `GROQ_API_KEY` | Groq |
| `TOGETHER_API_KEY` | Together.ai |
| `FIREWORKS_API_KEY` | Fireworks.ai |
| `XAI_API_KEY` | x.ai |
| `DASHSCOPE_API_KEY` | DashScope (Alibaba) |
| `MOONSHOT_API_KEY` | Moonshot |
| `DEEPSEEK_API_KEY` | DeepSeek |
| `ZAI_API_KEY` | Z.AI (GLM Coding Plan) |
| `KIMI_CODING_API_KEY` | Kimi Coding Plan |
| `OLLAMA_API_KEY` | Ollama Cloud |
| `OPENROUTER_API_KEY` | OpenRouter |

The local `llama` provider needs no key. The `zai` key authenticates against Z.AI's
Anthropic-compatible endpoint (it is the key Z.AI also documents as `ANTHROPIC_AUTH_TOKEN`).
The `kimi_coding` key authenticates against the Kimi Coding Plan's separate
Anthropic-compatible endpoint (`https://api.kimi.com/coding/`), which is distinct from the
standard Moonshot API used by the `moonshot` provider.
The `OLLAMA_API_KEY` key authenticates against Ollama Cloud's direct API (`https://ollama.com/v1` for OpenAI-compatible requests).
The `OPENROUTER_API_KEY` key authenticates against the OpenRouter gateway (`https://openrouter.ai/api/v1`).

### OpenRouter tuning

All optional; unset means Kolega Code sends nothing for that field.

| Variable | Purpose |
| --- | --- |
| `KOLEGA_CODE_OPENROUTER_PROVIDER_ORDER` | Comma-separated upstream provider slugs to try in order (OpenRouter `provider.order`). |
| `KOLEGA_CODE_OPENROUTER_PROVIDER_ONLY` | Comma-separated upstream provider slugs to restrict routing to (OpenRouter `provider.only`). |
| `KOLEGA_CODE_OPENROUTER_CATALOG` | Read the refreshed model cache from this path instead of the state directory. |
| `KOLEGA_CODE_DISABLE_OPENROUTER_CATALOG` | Ignore any refreshed cache and use only the catalog bundled with this release. |

## Model selection

Each role can be configured independently. **Provider and model are always set
together** — naming one without the other is an error. Nothing is inferred from a
model id and no model is defaulted from a provider, because the same id can be
served by different providers on different credentials.

| Variable | Role |
| --- | --- |
| `KOLEGA_CODE_PROVIDER` / `KOLEGA_CODE_MODEL` | Main (long-context) coding model |
| `KOLEGA_CODE_FAST_PROVIDER` / `KOLEGA_CODE_FAST_MODEL` | Fast utility model |
| `KOLEGA_CODE_THINKING_EFFORT` | Model-specific thinking effort |
| `KOLEGA_CODE_EDIT_PROTOCOL` | Optional model-facing edit override: `search_replace`, `codex_apply_patch`, or `claude_code`; otherwise the model catalogue preference or `search_replace` fallback is used |

These outrank the Fast slot saved in Settings. Full precedence per role:
CLI flag > environment variable > saved slot > the active model.

See [Providers & Models](../providers-and-models/) for what each role does.

## Per-agent models

Give an individual agent its own model. `<ROLE>` is one of `PLANNING`, `BUILDING`,
`INVESTIGATION`, `GENERAL`, or `BROWSER`. A role with no override inherits the main
model. These also override anything saved in the Settings screen's **Agent Models**
category.

| Variable | Purpose |
| --- | --- |
| `KOLEGA_CODE_<ROLE>_PROVIDER` | Provider for that agent (e.g. `KOLEGA_CODE_INVESTIGATION_PROVIDER`) |
| `KOLEGA_CODE_<ROLE>_MODEL` | Model for that agent |
| `KOLEGA_CODE_<ROLE>_EFFORT` | Thinking effort for that agent |

The `BROWSER` role requires a vision-capable model. An incompatible explicit or
inherited model is never replaced automatically; dispatching the browser agent
fails with a configuration error until `KOLEGA_CODE_BROWSER_PROVIDER` and
`KOLEGA_CODE_BROWSER_MODEL` (or the equivalent Settings override) select a model
with vision support.

## State & environment

| Variable | Purpose |
| --- | --- |
| `KOLEGA_CODE_STATE_DIR` | Override where settings and sessions are stored |
| `KOLEGA_CODE_ENVIRONMENT` | Environment label attached to tracing/metadata (default `development`) |
| `KOLEGA_CODE_NO_DIAGNOSTICS` | Set to any value to disable the local [diagnostics](../../troubleshooting/diagnostics/) log and responsiveness watchdog |
| `KOLEGA_CODE_COMPRESSION_THRESHOLD` | Context-window usage percent that triggers automatic history compression (`10`–`100`, default `80`; `100` effectively disables automatic compression) |
| `KOLEGA_CODE_SUBAGENTS` | Sub-agent dispatch (`dispatch_agent`) master switch: `on` or `off` (default `on`). Useful for headless or CI runs; the `--subagents` flag takes precedence |

## Web search

The `web_search` tool defaults to the keyless DuckDuckGo backend. Set a backend
explicitly when you want a cloud provider or a self-hosted SearXNG instance.

| Variable | Purpose |
| --- | --- |
| `KOLEGA_CODE_WEB_SEARCH_MODE` | Web tool mode: `auto` (hosted server-side search when the model supports it, else client tools), `hosted`, `client`, or `off` |
| `KOLEGA_CODE_WEB_SEARCH_BACKEND` | Backend for `web_search`: `duckduckgo`, `firecrawl`, `tavily`, or `searxng` |
| `FIRECRAWL_API_KEY` | Optional Firecrawl key for higher rate limits |
| `TAVILY_API_KEY` | Tavily API key |
| `SEARXNG_BASE_URL` | Base URL for a self-hosted SearXNG instance |

## Browser automation

Local browser automation needs no credentials. Hosted sandbox deployments can
connect the same browser agent to Browserless.

| Variable | Purpose |
| --- | --- |
| `BROWSERLESS_API_KEY` | Browserless cloud API token |
| `BROWSERLESS_WS_ENDPOINT` | Optional cloud or self-hosted WebSocket endpoint |
| `BROWSERLESS_REGION` | Cloud region used when no endpoint is supplied: `sfo`, `lon`, or `ams` |
| `BROWSERLESS_PROTOCOL` | Connection protocol: `cdp` (default) or `playwright` |
| `BROWSERLESS_TIMEOUT_MS` | Optional maximum Browserless session duration; must fit the account plan |
| `BROWSER_CONNECT_TIMEOUT_MS` | Browser transport connection timeout (default 30000 ms) |

Custom endpoint query parameters are preserved, including Browserless proxy,
stealth, profile, and recording options. If `BROWSERLESS_TIMEOUT_MS` is unset,
Browserless applies the account's default session duration.

## Telemetry (Langfuse)

Optional [Langfuse](https://langfuse.com/) tracing of LLM usage.

| Variable | Purpose |
| --- | --- |
| `LANGFUSE_HOST` | Langfuse host URL |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |

## About project `.env` files

Kolega Code does not automatically load a `.env` file from the project directory.
That file commonly belongs to the app being edited, so consuming it as Kolega
Code configuration can accidentally pick up unrelated application secrets such as
`OPENAI_API_KEY`.

Use one of these instead:

```bash title="one-off shell configuration"
export MOONSHOT_API_KEY=...
export KOLEGA_CODE_PROVIDER=moonshot
export KOLEGA_CODE_MODEL=kimi-k2.7-code
kolega-code
```

Or save the provider, model, and API key in the TUI Settings screen. Settings are
the recommended persistent configuration path for interactive use.

:::caution
Keep project `.env` files out of version control — they often hold application
secrets. Add them to your `.gitignore`.
:::
