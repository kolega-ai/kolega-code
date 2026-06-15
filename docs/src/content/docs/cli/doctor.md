---
title: doctor
description: Check your local Kolega Code configuration and API-key status.
---

`kolega-code doctor` runs a quick health check of your local setup. Use it after
installing or whenever the agent won't start, to see what's configured and what's
missing.

```bash
kolega-code doctor --project .
```

| Option | Description |
| --- | --- |
| `--project <PATH>` | Project directory to check (default `.`) |
| `--state-dir <PATH>` | Directory for CLI session state |

It also accepts the [global model options](../overview/#global-model-options), so
you can validate a specific provider/model combination before using it.

## What it checks

- **Project** — the resolved project path.
- **State dir** — where settings and sessions are stored.
- **Textual installed** — whether the `[cli]` extra (the interactive UI) is
  available.
- **Stored active model** — the provider/model saved in Settings, or
  `not configured`.
- **Stored thinking effort** — the saved effort value, or `model default`.
- **Stored API key** — for the active provider, whether the key is
  `present via <ENV_VAR>`, `present in local settings`, or `missing` (the key
  itself is never printed).
- **Configuration** — whether a valid `AgentConfig` can be built. If valid, it
  prints the resolved **long**, **fast**, **edit**, and **thinking** models.

## Example output

```text
Project: /Users/you/code/my-app
State dir: /Users/you/Library/Application Support/kolega-code
Textual installed: True
Stored active model: moonshot/kimi-k2.7-code
Stored thinking effort: auto
Stored API key: present in local settings
✓ Configuration: valid
Long model: moonshot/kimi-k2.7-code
Fast model: moonshot/kimi-k2.7-code
Edit model: moonshot/kimi-k2.7-code
Thinking model: moonshot/kimi-k2.7-code
Thinking effort: auto
```

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Configuration is valid |
| `2` | Configuration is invalid (e.g. missing API key, unknown model) — the reason is printed |

If `doctor` reports a missing key or invalid configuration, head to
[Providers & Models](../../configuration/providers-and-models/) and
[Environment Variables](../../configuration/environment-variables/) to fix it.
