# Kolega Code

**An AI coding agent that runs in your terminal — local-first, model-agnostic, plan-then-build.**

[![PyPI version](https://img.shields.io/pypi/v/kolega-code)](https://pypi.org/project/kolega-code/)
[![Python versions](https://img.shields.io/pypi/pyversions/kolega-code)](https://pypi.org/project/kolega-code/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![CI](https://github.com/kolega-ai/kolega-code/actions/workflows/ci.yml/badge.svg)](https://github.com/kolega-ai/kolega-code/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-kolega--ai.github.io-blue)](https://kolega-ai.github.io/kolega-code/)

Point Kolega Code at a project directory and it opens an interactive UI where you talk to an
agent that can read and edit your code, run shell commands, search the codebase, browse the
web, and dispatch specialized sub-agents to get work done.

It's **local-first**: the agent operates on your actual filesystem and terminal, and your
sessions, settings, and API keys stay on your own machine.

![Kolega Code in action](docs/src/assets/demo.gif)

## Why Kolega Code

- **Plans before it builds.** A read-only **Plan mode** produces a reviewable plan and a
  shared task list before anything changes; **Build mode** implements it. Toggle with `Shift+Tab`.
- **Orchestrates many agents.** For larger jobs, the main agent dispatches specialized
  sub-agents (investigation, browser, coding, general) and can fan them out in parallel for
  broad audits, large migrations, or implementing a plan's independent parts at once.
- **Local-first by design.** It works on your real files and terminal; sessions, settings,
  and credentials live on your machine, not in someone else's cloud.
- **Bring your own model.** Talks to a range of providers and lets you assign different models
  to different roles.

## What it does

- **Reads and edits your code.** Opens files, searches across the codebase, creates new files,
  and applies precise edits.
- **Runs commands.** Executes shell commands and watches their output in a dedicated terminal view.
- **Plans before it builds.** Plan mode investigates and proposes; Build mode executes.
- **Searches and browses the web.** The agent can search for relevant pages, fetch URLs
  directly, or use the built-in browser agent (powered by Playwright) when a task needs
  site interaction.
- **Dispatches sub-agents.** Hands work off to specialized agents and tracks their activity live.
- **Orchestrates workflows.** With [gigacode](https://kolega-ai.github.io/kolega-code/gigacode/),
  it fans out many sub-agents in parallel.
- **Works non-interactively too.** Run a single prompt with `kolega-code ask`, get JSON output,
  and save or resume sessions.

## Quick start

**1. Install** with the script:

```bash
curl -fsSL https://kolega.dev/install-kolega-code.sh | sh
```

Or with [uv](https://docs.astral.sh/uv/) (or `pip`):

```bash
uv tool install kolega-code
# or: pip install kolega-code
```

Verify the install:

```bash
kolega-code --version
```

**2. Start a session** in your project:

```bash
kolega-code .
```

**3. Add a provider key.** Open the **Settings** tab to pick a provider and model and save your
API key. Then press `Shift+Tab` anytime to switch between **Plan** and **Build** mode.

Resume a previous conversation:

```bash
kolega-code . --resume            # latest session
kolega-code . --resume <id>       # a specific thread or session
```

## Two ways to use it

| Mode | Command | Best for |
| --- | --- | --- |
| **Interactive TUI** | `kolega-code .` | Day-to-day development, exploration, pair-programming |
| **One-shot** | `kolega-code ask "…"` | Scripting, automation, quick questions, CI |

There are also helper commands for managing sessions and checking your setup:

```bash
kolega-code ask "summarize this repository" --project .
kolega-code sessions list --project .
kolega-code doctor --project .
```

## Bring your own model

Kolega Code talks to a range of LLM providers — including Anthropic, OpenAI, Google, Moonshot,
and DeepSeek — and lets you assign **different models to different roles**: a strong
long-context model for coding, a fast cheap model for small utility calls, and one for extended
"thinking". See [Providers & Models](https://kolega-ai.github.io/kolega-code/configuration/providers-and-models/).

## Sign in with ChatGPT

If you have a paid **ChatGPT** plan (Plus, Pro, or Business), you can use it to run OpenAI models
instead of a separate API key. Run `/login chatgpt` in the TUI, complete the browser sign-in, and
Kolega Code switches to the **OpenAI (ChatGPT subscription)** provider (e.g. `gpt-5-codex`). Tokens
are stored locally (chmod `600`) and refreshed automatically; `/logout chatgpt` removes them.

> **Heads-up:** This reuses the OpenAI Codex OAuth client and ChatGPT backend, the same approach
> other third-party agents use. It is **not** an officially sanctioned integration — OpenAI's
> supported "Sign in with ChatGPT" is identity-only and does not grant model access to third-party
> tools. Using it is at your own risk and may violate OpenAI's terms; a Free plan cannot run models
> this way. If in doubt, use an `OPENAI_API_KEY` instead.

## Configuration

Set your provider, model, and API keys from the **Settings** tab in the UI, or via environment
variables and flags for non-interactive use:

```bash
export KOLEGA_CODE_PROVIDER=deepseek
export DEEPSEEK_API_KEY=...
kolega-code ask "summarize this repository" --project . --provider deepseek --model deepseek-v4-pro
```

API key variables only provide credentials — pick a provider/model explicitly or save one in
Settings. Local session state lives under your platform's state directory unless
`KOLEGA_CODE_STATE_DIR` is set. See the
[Configuration docs](https://kolega-ai.github.io/kolega-code/configuration/settings-and-api-keys/)
for the full story.

The `web_search` tool uses DuckDuckGo by default without a key. To choose another backend,
set it in **Settings** or export `KOLEGA_CODE_WEB_SEARCH_BACKEND` as `firecrawl`, `tavily`,
or `searxng`; use `FIRECRAWL_API_KEY`, `TAVILY_API_KEY`, or `SEARXNG_BASE_URL` as needed.

## Requirements

- **Python 3.11+**
- An **API key** for at least one supported provider
- A terminal that supports a modern TUI (most do)

## Documentation

Full documentation lives at **[kolega-ai.github.io/kolega-code](https://kolega-ai.github.io/kolega-code/)**:

- [Quick Start](https://kolega-ai.github.io/kolega-code/getting-started/quick-start/)
- [CLI overview](https://kolega-ai.github.io/kolega-code/cli/overview/)
- [How it works & concepts](https://kolega-ai.github.io/kolega-code/concepts/how-it-works/)
- [Configuration](https://kolega-ai.github.io/kolega-code/configuration/settings-and-api-keys/)

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for local setup, running the
test suite, and building the docs site. Please report security issues privately per
[SECURITY.md](SECURITY.md).

## License

Released under the [MIT License](LICENSE).
