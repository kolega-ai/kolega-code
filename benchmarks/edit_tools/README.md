# Edit-tool benchmark harness

This is repository-only research infrastructure for comparing model-facing edit
protocols. It is deliberately outside `kolega_code`, has no installed console
command, and is excluded from the published wheel.

## Quick start

Validate the checked-in core corpus and an example model matrix without making
network calls:

```bash
uv run python -m benchmarks.edit_tools validate \
  --suite benchmarks/edit_tools/suites/core.yaml \
  --matrix benchmarks/edit_tools/matrices/example.yaml
```

Inspect the exact planned calls before spending model tokens:

```bash
uv run python -m benchmarks.edit_tools run \
  --suite benchmarks/edit_tools/suites/smoke.yaml \
  --matrix benchmarks/edit_tools/matrices/example.yaml \
  --dry-run
```

Run them only after reviewing the count:

```bash
uv run python -m benchmarks.edit_tools run \
  --suite benchmarks/edit_tools/suites/smoke.yaml \
  --matrix benchmarks/edit_tools/matrices/example.yaml \
  --confirm-live
```

Use `--task`, `--tag`, `--provider`, `--protocol`, and `--lane` to filter a
matrix. Each filter can be repeated or comma-separated. `--max-trials` provides
an additional hard call-count bound. A stopped run can be continued with the
same suite, matrix, and filters using `--resume .benchmark-runs/<run-id>`.

## Lanes

- `controlled` sends a fixed system prompt through the real provider client and
  exposes production read tools plus only the selected edit protocol. It has no
  terminal, browser, web, sub-agent, MCP, hook, skill, or LSP access.
- `coder_agent` uses the real CoderAgent prompt and loop, while retaining the
  same safe read/edit-only tool boundary. This lane accepts production protocols
  only; research candidates stay controlled until promoted.

Every trial starts from a fresh temporary workspace. Edit snapshots are stored
under that trial's artifact directory, never in normal Kolega Code user state.
Trusted verifier commands are executed as argv arrays without an implicit shell.

## Corpus

- `smoke.yaml`: four curated shared-capability cases, one repetition.
- `core.yaml`: twelve curated cases and thirty-six deterministic synthetic
  cases, three repetitions.
- `coder-agent.yaml`: the twelve curated cases, one repetition.

Synthetic cases use seeded inverse mutations and are fully materialized into the
run directory. The saved before and expected trees remain reproducible even if a
future generator version changes.

The main suites compare capabilities shared by both current protocols: file
updates, creation, and multi-file edits. Delete and move tasks belong in a
separate capability suite so they do not silently bias the shared score against
search/replace.

## Results

Runs are written under `.benchmark-runs/<run-id>/` and contain:

- an immutable manifest with git, suite, matrix, task, and protocol digests;
- materialized before/expected cases;
- append-only `trials.jsonl` records;
- normalized transcripts, event timelines, diffs, and oracle output per trial;
- `summary.json`, `summary.csv`, and `summary.md` reports.

Credentials and common token-shaped values are redacted. Provider, credential,
quota, timeout, and harness failures are displayed separately and excluded from
the model success denominator.

Success rates include Wilson intervals. Protocol comparisons pair identical
task/repetition outcomes and bootstrap over tasks with a fixed seed. The report
names a leader only when the paired 95% interval excludes zero.

## Complete provider smoke

The catalog gate runs one create-file task through both production protocols for
one model from every provider in `MODEL_SPECS`:

```bash
uv run python -m benchmarks.edit_tools provider-smoke \
  --confirm-live \
  --require-complete
```

This currently covers Anthropic, OpenAI, OpenAI ChatGPT OAuth, Google,
Moonshot, Kimi Coding, Z.AI, DeepSeek, xAI, Fireworks, Together, DashScope, and
Ollama Cloud. Missing credentials are recorded as `not_run`; `--require-complete`
returns non-zero unless every provider/protocol trial passes. Groq and local
Llama remain available for explicit research matrices but are not part of this
gate because they have no current `MODEL_SPECS` entries.

Live pytest coverage is separately opt-in with
`KOLEGA_RUN_LIVE_EDIT_BENCHMARKS=1`.

## Adding a protocol candidate

Add a versioned adapter to the benchmark protocol registry with its definitions,
capabilities, call normalization, and executor. Keep candidates benchmark-local;
do not add them to the production `EditProtocol` enum until controlled and
CoderAgent results justify promotion. Automatic candidate generation and search
is intentionally a later layer built on these stable tasks, records, and reports.
