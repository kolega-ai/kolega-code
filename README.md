# Kolega Code

**Multi-agent coding in the terminal.**

[![PyPI version](https://img.shields.io/pypi/v/kolega-code)](https://pypi.org/project/kolega-code/)
[![Python versions](https://img.shields.io/pypi/pyversions/kolega-code)](https://pypi.org/project/kolega-code/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![CI](https://github.com/kolega-ai/kolega-code/actions/workflows/ci.yml/badge.svg)](https://github.com/kolega-ai/kolega-code/actions/workflows/ci.yml)
[![Coverage](docs/src/assets/coverage.svg)](https://github.com/kolega-ai/kolega-code/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-kolega--ai.github.io-blue)](https://kolega-ai.github.io/kolega-code/)
[![Changelog](https://img.shields.io/badge/changelog-keep%20up-blue)](CHANGELOG.md)

Kolega Code is a local-first terminal coding agent built for work that is too wide
for one loop: fan out specialized sub-agents with **Gigacode**, route different
models to different jobs, search the web, drive a browser, and keep sessions,
settings, permissions, and credentials on your machine.

![Kolega Code in action](docs/src/assets/demo.gif)

## Built for work one agent cannot cover

Most terminal agents are strongest when one model can reason through one task at a
time. Kolega Code keeps that familiar workflow, then adds orchestration for broad work:
large audits, sweeping migrations, cross-file checks, adversarial reviews, and
implementation plans whose pieces can run independently.

With [Gigacode](https://kolega-ai.github.io/kolega-code/gigacode/), Kolega Code can:

- **Fan out many sub-agents at once.** Split a wide codebase review by package,
  assign independent implementation tasks, or run checks across many directories in
  parallel.
- **Use real workflow shapes.** The agent can generate parallel phases, pipelines,
  loops, judge panels, and synthesis steps instead of only delegating one task at a
  time.
- **Keep orchestration visible.** Workflow phase headers and progress lines appear
  in the transcript; the sub-agent inspector shows each agent's live trajectory.
- **Save inspectable artifacts.** Each run keeps the generated workflow script,
  result files, a Markdown transcript, raw JSONL, a resume journal, and debug
  sub-agent transcripts under Kolega Code's state directory.
- **Run in either mode.** In Plan mode, workflow sub-agents stay read-only for
  parallel investigation. In Build mode, they can use the full coding toolset.
- **Resume interrupted runs.** Finished workflow steps are journaled so a deliberate
  resume does not have to restart the whole fan-out.

Use normal chat for focused changes. Turn on Gigacode when the problem is broad
enough that one serial agent pass would be the bottleneck.

## Features

- **Gigacode orchestration:** parallel, pipelined, looped, judged, and synthesized
  multi-agent workflows with saved artifacts and resume support.
- **Specialized sub-agents:** planning, building/coder, investigation, general, and
  browser agents, with live activity tracking in the TUI.
- **Repo tools:** read and search code, create files, apply precise edits, and inspect
  session changes/diffs.
- **Terminal execution:** run shell commands with streamed output and project-level
  permission controls.
- **Plan/build workflow:** use read-only Plan mode for investigation and a reviewable
  task list, then Build mode to implement.
- **Web search and browsing:** DuckDuckGo works by default with no key; Firecrawl,
  Tavily, and SearXNG are configurable search backends. Kolega Code can also fetch URLs
  directly and use a Playwright-powered browser agent for interactive sites.
- **MCP servers:** connect verified `streamable_http`, `sse`, and `stdio` MCP servers
  (including OAuth-enabled HTTP servers) as permission-gated tools.
- **Model routing:** choose provider/model combinations, set thinking effort, split
  long-context/fast/thinking roles, and override models per agent role.
- **Interactive or scriptable:** use the Textual TUI, queue follow-up prompts
  while the agent is working, run `kolega-code ask`, request JSON output,
  list/export/resume sessions, and diagnose setup with `doctor`.
- **Extensibility:** add agent skills, override prompts with project templates, run
  lifecycle hooks, and persist project permission rules.
- **Local-first state:** sessions, settings, permissions, OAuth tokens, and API-key
  settings stay on your machine with restrictive permissions where applicable.

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

**3. Connect a model.** Complete the first-run wizard with ChatGPT sign-in or an
API key. Later, open the categorized Settings screen from the sidebar or with
`/settings`. Use `Shift+Tab` to switch between **Plan** and **Build** mode, or run
`/gigacode on` when a task is broad enough for fan-out.

Resume a previous conversation:

```bash
kolega-code . --resume            # latest session
kolega-code . --resume <id>       # a specific thread or session
```

## Two ways to use it

| Mode | Command | Best for |
| --- | --- | --- |
| **Interactive TUI** | `kolega-code .` | Day-to-day development, exploration, orchestration |
| **One-shot** | `kolega-code ask "…"` | Scripting, automation, quick questions, CI |

There are also helper commands for managing sessions and checking your setup:

```bash
kolega-code ask "summarize this repository" --project .
kolega-code sessions list --project .
kolega-code doctor --project .
```

## Supported providers

Kolega Code supports a broad model-provider catalog and lets you route models by
role instead of forcing one model to do every job.

Supported model providers:

- Anthropic
- OpenAI API
- OpenAI via ChatGPT subscription sign-in
- Google
- Groq
- Together.ai
- Fireworks.ai
- xAI / Grok
- DashScope / Qwen
- Moonshot / Kimi
- DeepSeek
- Z.AI / GLM Coding Plan
- Kimi Coding Plan
- Ollama Cloud
- local Llama

Supported web-search backends:

- DuckDuckGo — default, no key required
- Firecrawl
- Tavily
- SearXNG — self-hosted option

See [Providers & Models](https://kolega-ai.github.io/kolega-code/configuration/providers-and-models/)
for model IDs, role configuration, API-key variables, and thinking-effort options.

## Model routing

Kolega Code can assign different models to different operational roles: a strong
long-context model for the main coding loop, a faster model for utility calls, and
a dedicated model for extended thinking. You can also override models per agent
role — planning, building, investigation, general, and browser — so wide workflows
can use cheaper models where they fit and stronger models where they matter.

## Sign in with ChatGPT

If you have a paid **ChatGPT** plan (Plus, Pro, or Business), you can use it to run
OpenAI models instead of a separate API key. Run `/login chatgpt` in the TUI,
complete the browser sign-in, and Kolega Code switches to the **OpenAI (ChatGPT
subscription)** provider (default `gpt-5.6-sol`). Tokens are stored locally (chmod
`600`) and refreshed automatically; `/logout chatgpt` removes them. See
[Sign in with ChatGPT](https://kolega-ai.github.io/kolega-code/configuration/sign-in-with-chatgpt/).

## Configuration

Set your provider, model, and API keys from first-run onboarding or the full-screen
Settings editor in the UI, or via environment variables and flags for
non-interactive use:

```bash
export KOLEGA_CODE_PROVIDER=deepseek
export DEEPSEEK_API_KEY=...
kolega-code ask "summarize this repository" --project . --provider deepseek --model deepseek-v4-pro
```

API key variables only provide credentials — pick a provider/model explicitly or
save one in Settings. Local session state lives under your platform's state
directory unless `KOLEGA_CODE_STATE_DIR` is set. See the
[Configuration docs](https://kolega-ai.github.io/kolega-code/configuration/settings-and-api-keys/)
for the full story.

The `web_search` tool uses DuckDuckGo by default without a key. To choose another
backend, set it in **Settings** or export `KOLEGA_CODE_WEB_SEARCH_BACKEND` as
`firecrawl`, `tavily`, or `searxng`; use `FIRECRAWL_API_KEY`, `TAVILY_API_KEY`, or
`SEARXNG_BASE_URL` as needed.

Projects can override Kolega Code's base prompts with uppercase Markdown templates
in `.kolega/prompts/`. Generate editable starters with Jinja replacement tags using
`/prompts dump` in the TUI or `kolega-code prompts dump --project .` in a terminal.
To dump only selected starters, pass prompt names such as `coder`, `planning`, or
`compaction` (filename aliases like `CODER.md` work too). Validate existing
overrides with `/prompts validate` or `kolega-code prompts validate --project .`.

## Requirements

- **Python 3.11+**
- An **API key**, ChatGPT sign-in, or local model for at least one supported model provider
- A terminal that supports a modern TUI (most do)

## Documentation

Full documentation lives at **[kolega-ai.github.io/kolega-code](https://kolega-ai.github.io/kolega-code/)**:

- [Quick Start](https://kolega-ai.github.io/kolega-code/getting-started/quick-start/)
- [CLI overview](https://kolega-ai.github.io/kolega-code/cli/overview/)
- [How it works & concepts](https://kolega-ai.github.io/kolega-code/concepts/how-it-works/)
- [Configuration](https://kolega-ai.github.io/kolega-code/configuration/settings-and-api-keys/)

## Project resources

- [Documentation](https://kolega-ai.github.io/kolega-code/)
- [Releases](https://github.com/kolega-ai/kolega-code/releases)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Issue tracker](https://github.com/kolega-ai/kolega-code/issues)

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for local setup,
running the test suite, and building the docs site. Please report security issues
privately per [SECURITY.md](SECURITY.md).

## License

Released under the [Apache License 2.0](LICENSE).
