---
title: Extensions
description: Load one installed Python extension into the interactive CLI or ask at launch.
---

An extension lets host-authored Python code contribute prompt sections and
tools to Kolega Code's top-level agent, observe structured LLM trace records,
and bind to each constructed agent's live state — without forking the runtime.

:::caution
An extension is **arbitrary Python code running with the same authority as
Kolega Code itself**. Only load extensions you trust. Nothing is loaded
implicitly: an extension runs only when you name it on the command line.
:::

## Loading an extension

Both the interactive CLI (either launch form) and `ask` accept the same flags:

```bash
kolega-code /path/to/project \
  --extension acme_kolega_extension:create_extension \
  --extension-config /absolute/path/to/extension.json

kolega-code ask "Inspect the repository" \
  --extension acme_kolega_extension:create_extension \
  --extension-config /absolute/path/to/extension.json
```

- `--extension MODULE:FACTORY` names an importable module and a factory
  callable inside it. The package must already be installed in the active
  environment; Kolega Code never installs anything or modifies `sys.path`.
- `--extension-config PATH` is optional and opaque: the path is resolved to an
  absolute path and handed to the factory unread. It is extension input, never
  a source of Kolega Code settings. Passing it without `--extension` is an
  error.

Load and validation failures terminate before the first model request with a
concise error and a nonzero exit status. A tool name contributed by the
extension that conflicts with a built-in or already-provisioned tool also
refuses to start.

## The factory and bundle API

Everything an extension needs is importable from the package root:

```python
from pathlib import Path

from kolega_code import (
    KolegaExtensionBundle,
    KolegaExtensionHost,
    PromptExtension,
    ToolExtension,
)


def create_extension(host: KolegaExtensionHost, config_path: Path | None) -> KolegaExtensionBundle:
    return KolegaExtensionBundle(
        prompt_extensions=[...],
        tool_extensions=[...],
        llm_trace_sink=None,
        bind_agent=None,
        cleanup=None,
    )
```

The module and factory are resolved once at launch. The factory is then called
once for **every top-level agent generation**: `ask` has one, while the
interactive CLI rebuilds its agent on model switches, plan/build transitions,
workspace changes, and settings changes — each rebuild cleans up the previous
bundle and requests a fresh one. `host` carries the resolved project path,
workspace/thread IDs, the read-only `AgentConfig`, and the `AgentMode`
(`CLI` or `ASK`).

### Prompt and tool contributions

`prompt_extensions` and `tool_extensions` are ordinary
[`PromptExtension`/`ToolExtension`](../tools/) objects, appended to the host's
normal inventories before the agent is constructed. All existing behavior
applies: prompt filtering by `agent_types`/`modes`, explicit tool schemas,
`propagate_to_sub_agents`, `exclusive_tools` whole-batch rejection, and
per-extension `cleanup` through the tool collection.

Tool callbacks follow the existing `ToolExtension` contract: each callback is
an **async** callable (the executor awaits it), and its docstring is the
model-visible tool description. By default the input schema is **derived from
the callback**: parameter types from annotations, required-ness from missing
defaults, and per-parameter descriptions from the docstring's `Args:` section
(which is then stripped from the wire description). A `tool_schemas` entry —
the bare JSON schema, `{"type": "object", "properties": ..., "required":
...}`, not a full tool-definition envelope — replaces the derived schema
verbatim, for shapes a signature cannot express (arrays with bounds, enums,
nested objects). If an explicit schema doesn't describe every property, the
`Args:` block is kept in the description so the parameters stay documented.

### Agent binding

`bind_agent` is called exactly once per bundle, after the agent and its tool
collection are fully constructed and before that generation's first model
request (it may return an awaitable). Tool callbacks receive only
model-supplied arguments, so an extension that needs the calling agent's state
captures the bound agent in a closure:

```python
def create_extension(host, config_path):
    state = {}

    async def probe() -> str:  # tool callbacks are awaited: make them async
        agent = state["agent"]
        return f"history has {len(agent.history)} messages"

    return KolegaExtensionBundle(
        tool_extensions=[ToolExtension(name="probe-ext", tools={"probe": probe},
                                       propagate_to_sub_agents=False)],
        bind_agent=lambda agent: state.__setitem__("agent", agent),
    )
```

When a callback runs, the bound agent's conversation already contains the
assistant message with that tool call. A tool that depends on the top-level
binding should set `propagate_to_sub_agents=False`, since sub-agents are not
the bound agent.

### LLM trace sink

`llm_trace_sink` is an optional callable forwarded to every LLM client created
from the bound agent's context — the top-level loop, helpers, and sub-agents
that inherit the context, exactly like the usage ledger. Providers that emit
structured trace records (today: the native Tinker provider's
`TinkerTraceRecord`) pass each record to the sink; other providers ignore it.
Records carry their own attribution (`request_role`, from
`llm_call_origin`), so one sink separates top-level, sub-agent, and named
helper calls without extra plumbing.

### Cleanup

`cleanup` runs exactly once per bundle (awaited when awaitable), after the
corresponding agent's own cleanup, on every exit path: completion, interactive
agent rebuild, failure, or interrupt. A cleanup failure is reported without
masking the primary error.

## Minimal example

A complete installable example package lives in
[`examples/extension`](https://github.com/kolega-ai/kolega-code/tree/main/examples/extension):
one prompt section and one harmless tool, with no domain logic.
