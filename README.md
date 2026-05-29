# kolega-code

Shared Python package for Kolega agent runtime code.

The package owns the `kolega_code` import namespace and is intended to stay local-filesystem first. Provider-specific sandbox integrations such as E2B live outside this package.

## CLI

Install the optional CLI extra to use the Textual terminal interface:

```bash
pip install "kolega-code[cli]"
```

Run the Textual UI and open the Settings tab to select Moonshot Kimi K2.6 or DeepSeek V4 Pro and save your API key:

```bash
kolega-code .
```

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
