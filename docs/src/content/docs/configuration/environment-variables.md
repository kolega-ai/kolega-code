---
title: Environment Variables
description: Every environment variable Kolega Code reads, and how precedence works.
---

Kolega Code reads configuration from environment variables and from a project-local
`.env` file. This page lists everything it understands.

## Precedence

For any given setting, the first available source wins:

1. **CLI flags** (e.g. `--provider`, `--model`)
2. **Shell environment variables**
3. **Project `.env` file** (in the project directory)
4. **Saved Settings** (`settings.json`)

Kolega Code requires an explicit provider/model selection from flags,
environment variables, a project `.env` file, or saved settings. API key
variables alone are not a model selection source.

:::note
Within the env layer, your **shell environment takes priority over the `.env`
file** — values exported in your shell override the same key in `.env`.
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

The local `llama` provider needs no key. The `zai` key authenticates against Z.AI's
Anthropic-compatible endpoint (it is the key Z.AI also documents as `ANTHROPIC_AUTH_TOKEN`).
The `kimi_coding` key authenticates against the Kimi Coding Plan's separate
Anthropic-compatible endpoint (`https://api.kimi.com/coding/`), which is distinct from the
standard Moonshot API used by the `moonshot` provider.

## Model selection

Each role can be configured independently. Set just the provider to use that
provider's default model, or set both provider and model.

| Variable | Role |
| --- | --- |
| `KOLEGA_CODE_PROVIDER` / `KOLEGA_CODE_MODEL` | Main (long-context) coding model |
| `KOLEGA_CODE_FAST_PROVIDER` / `KOLEGA_CODE_FAST_MODEL` | Fast utility model |
| `KOLEGA_CODE_THINKING_PROVIDER` / `KOLEGA_CODE_THINKING_MODEL` | Thinking model |
| `KOLEGA_CODE_THINKING_EFFORT` | Model-specific thinking effort |

See [Providers & Models](../providers-and-models/) for what each role does.

## State & environment

| Variable | Purpose |
| --- | --- |
| `KOLEGA_CODE_STATE_DIR` | Override where settings and sessions are stored |
| `KOLEGA_CODE_ENVIRONMENT` | Environment label attached to tracing/metadata (default `development`) |

## Telemetry (Langfuse)

Optional [Langfuse](https://langfuse.com/) tracing of LLM usage.

| Variable | Purpose |
| --- | --- |
| `LANGFUSE_HOST` | Langfuse host URL |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |

## Using a `.env` file

Kolega Code automatically loads a `.env` file from the project directory. A good
starting point:

```bash title=".env"
# Pick one provider's key (or several)
MOONSHOT_API_KEY=
DEEPSEEK_API_KEY=
ANTHROPIC_API_KEY=

# Optional: choose models per role
KOLEGA_CODE_PROVIDER=moonshot
KOLEGA_CODE_MODEL=kimi-k2.7-code
KOLEGA_CODE_THINKING_EFFORT=auto

# Optional: Langfuse tracing
LANGFUSE_HOST=https://us.cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

:::caution
Keep `.env` out of version control — it holds secrets. Add it to your
`.gitignore`.
:::
