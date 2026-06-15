# kolega-code

Kolega Code is a local-first AI coding agent for the terminal.

The package owns the `kolega_code` import namespace and provides the
`kolega-code` command.

## Install

Install with the public installer:

```bash
curl -fsSL https://kolega.dev/install-kolega-code.sh | sh
```

Or install directly from PyPI with uv:

```bash
uv tool install kolega-code
```

Verify the command is available:

```bash
kolega-code --version
```

Upgrade or uninstall:

```bash
uv tool upgrade kolega-code
uv tool uninstall kolega-code
```

Run the Textual UI and open the Settings tab to select Moonshot Kimi K2.6 or DeepSeek V4 Pro and save your API key:

```bash
kolega-code .
```

In the Textual UI, press `Shift+Tab` to switch between build mode and planning mode. Planning mode uses a standalone read-only planning agent; when it submits a complete plan, choose whether to implement it or keep discussing the plan.

All CLI sessions use the CLI-specific coding-agent prompt, including resumed sessions. Launching the UI starts a fresh thread by default. Resume an existing thread explicitly:

```bash
kolega-code . --resume
kolega-code . --resume <thread-or-session-id>
```

You can also set `MOONSHOT_API_KEY`, `DEEPSEEK_API_KEY`, or keep using env/flag based configuration for non-UI commands:

```bash
kolega-code ask "summarize this repository" --project .
kolega-code ask "summarize this repository" --project . --provider deepseek --model deepseek-v4-pro
kolega-code sessions list --project .
kolega-code doctor --project .
```

The Settings UI supports Moonshot `kimi-k2.6` and DeepSeek `deepseek-v4-pro`. A saved UI selection is used for all agent model roles and API keys are stored in the local CLI settings file with restrictive permissions. Existing environment and model/provider flag overrides continue to work. Local session state is stored under the platform state directory unless `KOLEGA_CODE_STATE_DIR` is set.

## From source

```bash
git clone https://github.com/kolega-ai/kolega-code.git
cd kolega-code
uv sync --extra dev
uv run kolega-code --version
```

## Tests

Fast tests run by default:

```bash
./run_tests.sh
```

Some slow and integration tests require real provider credentials. To run them locally, create an ignored `.env` file from the example and fill only the keys you need:

```bash
cp .env.example .env
./run_tests.sh --all
```

The test runner loads `.env` through pytest and keeps existing shell environment variables higher priority than values in the file. You can pass additional pytest arguments through the wrapper:

```bash
./run_tests.sh kolega_code/agent/tests/llm/test_client.py -ra
```
