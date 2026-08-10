# kolega-extension-example

The minimal working Kolega Code CLI extension: one prompt section, one harmless
tool, and every lifecycle hook exercised. No domain logic.

An extension is arbitrary Python code running with the same authority as
Kolega Code itself — only load code you trust.

## Install and run

```bash
pip install -e examples/extension

kolega-code /path/to/project \
  --extension kolega_extension_example:create_extension

kolega-code ask "Use the extension_echo tool on the word hello" \
  --extension kolega_extension_example:create_extension \
  --extension-config /tmp/example-config.json
```

The `--extension-config` path is optional and opaque: this example only records
it. See `docs/src/content/docs/concepts/extensions.md` for the full contract.
