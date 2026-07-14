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

For the full DeepSeek V4 Pro/Flash comparison, the checked-in matrix plans 600
trials (100 tasks × 3 protocols × 2 models). Codex apply-patch is deliberately
excluded because it is not a viable default for these models:

```bash
uv run python -m benchmarks.edit_tools run \
  --suite benchmarks/edit_tools/suites/core.yaml \
  --matrix benchmarks/edit_tools/matrices/deepseek-v4.yaml \
  --dry-run
```

Replace `--dry-run` with `--confirm-live` only after reviewing that plan. The
matrix allows 30 model iterations so tasks with many regions and files are not
artificially capped below their required read/edit sequence.

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

- `smoke.yaml`: four tiny curated cases for cheap integration checks.
- `core.yaml`: 100 exact mechanical edit tasks over pinned files from public
  repositories, across twelve programming languages.
- `coder-agent.yaml`: the legacy curated cases for end-to-end agent checks.

Top-level files in `suites/` are runnable suite definitions only. Their task
data is deliberately separated by purpose:

- `suites/corpora/edit-core.yaml` contains the real-file benchmark corpus.
- `suites/fixtures/smoke-tasks.yaml` contains tiny fixtures used only by the
  smoke and CoderAgent integration suites.

The tiny fixtures are never included in `edit-core` or its model matrices.

The core suite measures edit-tool use, not solution design. Each prompt names
the exact files and original locations, and supplies every replacement or
insertion verbatim. The primary score is whether the resulting workspace is
byte-for-byte equal to the expected tree. Per-operation completion is a
diagnostic for partial failures.

The selected primary files range from short (20–99 lines) through oversized
(2,000–6,000 lines), with 350–899 lines treated as medium. Tasks cover one to
fifteen edit regions and one to eight changed files. Reports break results down
by language, edit family, target length, payload size, and changed-file count.
Payload size counts the larger of the removed and inserted sides of each edit,
so a substantial deletion is not mislabeled as a tiny task.

Source snapshots include repository URL, immutable commit, license, Git blob
ID, byte count, line count, and SHA-256 for every copied file. Snapshot-backed
run manifests record task and tree digests without copying the same source trees
into every run.

### Browse the test cases

Build and open the static corpus browser:

```bash
uv run python -m benchmarks.edit_tools browse
```

The browser provides searchable task metadata, the exact model instruction,
the neutral operation recipe, pinned source provenance, per-file unified diffs,
and full before/expected source views. It contains corpus definitions only, not
benchmark results. The generated site is written under
`.corpus-builds/browser/edit-core/` and ignored by Git. Use `--no-open` to build
it without launching a browser, or `--output` to choose another build root.

### Rebuilding the corpus

Check out the repositories and commits listed in `fixtures/sources.yaml` under
one directory, then import the deterministic source selection:

```bash
uv run python -m benchmarks.edit_tools corpus-import \
  --checkout-root /path/to/pinned-checkouts
```

Author missing exact edit recipes into `suites/corpora/edit-core.yaml` with the
live, retryable pipeline:

```bash
uv run python -m benchmarks.edit_tools corpus-author \
  --confirm-live \
  --concurrency 4
```

Authoring uses Claude Opus with the `edit`/`write` search-and-replace surface,
then retries in a clean workspace and falls back to an OpenAI apply-patch model.
Candidates are accepted only if they change exactly the scheduled files,
round-trip through the neutral recipe, stay within the requested operation
bucket, and do not increase tree-sitter parse errors. Attempt artifacts are
written under `.corpus-builds/`, which is ignored by Git.

## Multilingual verifier

Legacy semantic tasks that require a compiler or runtime use a pinned,
network-disabled verifier container. Build it before validating those suites:

```bash
docker build \
  -t kolega-edit-verifier:1 \
  -f benchmarks/edit_tools/verifier/Dockerfile \
  benchmarks/edit_tools/verifier
```

The image includes Python, Node/TypeScript, Go, Rust, Java, C/C++, C#, Ruby,
PHP, Swift, and Kotlin toolchains. Verification runs against a disposable copy
of the edited workspace, so compiler artifacts cannot affect collateral-change
scoring. Override the image with `KOLEGA_EDIT_BENCHMARK_VERIFIER_IMAGE`.

Execute every corpus oracle against both its before and expected workspaces:

```bash
uv run python -m benchmarks.edit_tools validate \
  --suite benchmarks/edit_tools/suites/core.yaml \
  --verify-oracles
```

The core suite compares capabilities shared by all current protocols: file
updates, creation, and multi-file edits. Recipe operations may remove regions
inside a file, but they do not delete or move files.

Production protocol IDs are `search_replace`, `codex_apply_patch`,
`claude_code`, and `hashline_v2`. Hashline v2 uses JSON line-anchor edits and
adds anchors to read/search output only while its `edit` tool is exposed.

## Results

Runs are written under `.benchmark-runs/<run-id>/` and contain:

- an immutable manifest with git, suite, matrix, task, and protocol digests;
- compact snapshot references and tree digests (inline smoke cases still store
  materialized before/expected trees);
- append-only `trials.jsonl` records;
- normalized transcripts, event timelines, diffs, and oracle output per trial;
- `summary.json`, `summary.csv`, and `summary.md` reports.

Credentials and common token-shaped values are redacted. Provider, credential,
quota, timeout, and harness failures are displayed separately and excluded from
the model success denominator.

Success rates include Wilson intervals. Protocol comparisons pair identical
task/repetition outcomes and bootstrap over tasks with a fixed seed. The report
names a leader only when the paired 95% interval excludes zero.

Reports keep task success and exact workspace matching separate. They also
record first-edit-attempt success per changed file as a primary outcome: the
earliest edit call targeting each file must parse and apply. A failed call that
is fixed on a later attempt remains a first-attempt failure for that file, and a
target file with no edit call is unsuccessful. Reports separately show the
share of trials where every changed file succeeded on its first attempt, a
conditional first-call rate, and `no_edit_rate` as diagnostics.

## Complete provider smoke

The catalog gate runs one create-file task through every production protocol for
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
