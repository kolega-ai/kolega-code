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
kolega-code update
uv tool uninstall kolega-code
```

Running the installer again also updates an existing install to the latest
released version.

Run the Textual UI and open the Settings tab to pick any provider and model from the catalog, choose the model's thinking effort, and save your API key:

```bash
kolega-code .
```

In the Textual UI, press `Shift+Tab` to switch between build mode and planning mode. Planning mode uses a standalone read-only planning agent; when it submits a complete plan, choose whether to implement it or keep discussing the plan.

All CLI sessions use the CLI-specific coding-agent prompt, including resumed sessions. Launching the UI starts a fresh thread by default. Resume an existing thread explicitly:

```bash
kolega-code . --resume
kolega-code . --resume <thread-or-session-id>
```

You can also use env/flag based configuration for non-UI commands. API key
variables provide credentials only; set a provider/model explicitly or save one
in the Settings UI:

```bash
export KOLEGA_CODE_PROVIDER=deepseek
export DEEPSEEK_API_KEY=...
kolega-code ask "summarize this repository" --project .
kolega-code ask "summarize this repository" --project . --provider deepseek --model deepseek-v4-pro
kolega-code sessions list --project .
kolega-code doctor --project .
```

The Settings UI exposes every model in the catalog (`kolega_code/llm/specs.py`) across all supported providers — the provider and model dropdowns are derived from that catalog, so adding a model there makes it selectable in the UI automatically. A saved UI selection is used for all agent model roles, and switching models resets thinking effort to that model's default. API keys are stored in the local CLI settings file with restrictive permissions. Existing environment and model/provider flag overrides continue to work, but API key variables alone never select a provider or model. Local session state is stored under the platform state directory unless `KOLEGA_CODE_STATE_DIR` is set.

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
