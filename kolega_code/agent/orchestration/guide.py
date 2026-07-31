"""The gigacode authoring guide, injected as a PromptExtension when gigacode is on.

This is the model-facing documentation for writing ``run_workflow`` scripts. Keep
it in sync with the primitives in :mod:`runtime` and the tool in
``tool_backend/workflow_tool.py``.
"""

GIGACODE_AUTHORING_GUIDE = """\
## gigacode — dynamic workflow orchestration

gigacode is ON for this session. For substantial work that benefits from running
many sub-agents with real control flow — broad audits, migrations, multi-file
reviews, judge panels, adversarial verification, **implementing a plan that splits
into independent workstreams**, anything one context can't hold — author a Python
orchestration script and run it with the `run_workflow` tool. For a quick single
lookup or a small/coupled edit, just do it directly; don't orchestrate.

### How to author a workflow

Call `run_workflow` with a `script` string. The script is Python and MUST begin
with a module-level `meta` literal, then use the injected primitives. Example:

```python
meta = {
    "name": "review-changes",
    "description": "Review changed files across dimensions, verify each finding",
    "phases": [{"title": "Review"}, {"title": "Verify"}],
}

DIMENSIONS = [
    {"key": "bugs", "prompt": "Review the diff for correctness bugs."},
    {"key": "perf", "prompt": "Review the diff for performance problems."},
]

phase("Review")
results = await pipeline(
    DIMENSIONS,
    lambda d: agent(d["prompt"], label=f"review:{d['key']}", schema=FINDINGS_SCHEMA),
    lambda review: parallel([
        (lambda f=f: agent(f"Adversarially verify: {f['title']}",
                           label=f"verify:{f['title'][:40]}", schema=VERDICT_SCHEMA, phase="Verify"))
        for f in (review or {}).get("findings", [])
    ]),
)
confirmed = [f for group in results if group for f in group if f]
return {"confirmed": confirmed}
```

`meta` must be a pure literal (no variables, calls, or f-strings) — it is read
without running the script. Required keys: `name`, `description`. Optional keys:
`phases` and `max_agent_depth`.

### Where scripts live

Pass short scripts inline via `script`. Draft a long script in your session
scratchpad directory (its path is in your system prompt) and pass `script_path`.
Do NOT write workflow scripts into the project working tree unless the user
explicitly asks for a repo-versioned workflow: generated scripts there pollute
`git status` and go stale for future sessions. Nothing durable is lost by using
the scratchpad — every executed script is persisted under the run directory and
returned as `scriptPath`; iterate by editing THAT file and re-running with
`script_path`.

`max_agent_depth` controls nested agent delegation for the whole workflow. It
defaults to `1`, so agents dispatched directly by workflow `agent()` calls are
leaves and receive no agent-dispatch tools. The only other accepted value, and
the hard maximum, is `2`: direct workers may then retain only the dispatch tools
their agent type already supports, and any children they dispatch are leaves.
This never enables unsupported dispatch tools and never permits a worker to call
`run_workflow`.

Strongly prefer the default of `1`. Express fan-out and stages visibly in the
workflow with `agent()`, `parallel()`, and `pipeline()`. Depth `2` is an exceptional
opt-in for a worker that genuinely must consult a specialist; nested calls are
less visible to workflow-level orchestration and can multiply token use.

At depth `2`, recursive delegation supports built-in agent-dispatch tools only.
Opaque host-provided `ToolExtension` dispatch callbacks are unavailable in
workflows until there is a workflow-aware accounting and depth protocol. Ordinary
non-workflow extension and agent-delegation behavior is unchanged.

### Choosing a sub-agent model

Omit `model_override` to inherit the normal model settings for the requested agent
type from the session's CLI/host configuration. This is the default and is almost
always correct.

When a task genuinely needs a different model, call the read-only
`list_subagent_models` tool before authoring the workflow. It reports the configured
providers, exact model IDs, supported effort values, and effective agent defaults
without exposing credentials. You may filter it by provider. Do not guess model IDs
or effort values.

A Gigacode override is atomic and applies to one direct workflow worker:

```python
result = await agent(
    "Review the authentication design for subtle security flaws.",
    agent_type="investigation",
    model_override={
        "provider": "anthropic",
        "model": "claude-opus-5",
        "effort": "high",
    },
)
```

If `model_override` is present, it must be a complete object with exactly
`provider`, `model`, and `effort`. For a model with effort controls, `effort` must
be one of the strings returned by `list_subagent_models`. Use `effort: None` only
when that model has no effort control; `None` does not mean "use the default".
Unsupported models, unconfigured providers, malformed objects, and invalid effort
values fail that `agent()` call and return `None` through normal workflow failure
handling. They never fall back to an inherited or provider-default model.

The override affects only the worker launched by that `agent()` call, not its
helper models or descendants. At depth `2`, a child dispatched by that worker uses
its own role default unless the child dispatch supplies its own complete ordinary
dispatch override. In Plan mode, the direct worker is always forced to the
read-only Investigation agent first, and the override is validated and applied to
that actual worker.

The legacy partial arguments `model=` and `effort=` are no longer accepted.
Migrate saved workflows to the complete `model_override={...}` object; legacy calls
fail with a migration error rather than running with partially inherited routing.

### Primitives (all available as globals in the script)

- `await agent(prompt, *, label=None, phase=None, schema=None, model_override=None, agent_type=None, browser_target=None)`
  Dispatch one sub-agent. Returns its report text, or a validated dict when `schema`
  (a JSON Schema) is given, or `None` if the agent failed/was skipped (so you can
  filter it out). `agent_type` is one of "general" (default, full toolset), "investigation"
  (read-only), "browser", or "coder". Omit `model_override` to inherit the requested
  role's configured defaults; otherwise supply the complete atomic object documented
  above. `browser_target` applies only to browser agents; omit it for isolated
  Playwright, or use `"chrome"` only when the user directs the workflow to use the
  configured Chrome extension. In plan mode every sub-agent is forced read-only
  regardless of `agent_type`, so use workflows there for parallel research and
  synthesis, not edits.
- `await parallel(thunks)` — run zero-arg thunks concurrently and wait for ALL (a barrier).
  A thunk that raises resolves to `None` and the drop is reported in the progress log and
  transcript with the offending script line — check those reports before trusting a fan-out
  full of `None`s. Budget/agent-cap exhaustion is never swallowed; it fails the run.
  Filter `None` before use.
- `await pipeline(items, *stages)` — run each item through all stages independently, with
  NO barrier between stages (item A can be in stage 3 while item B is still in stage 1).
  Each stage is called with `(prev_result, original_item, index)` — write `lambda r: ...`
  or `lambda prev, item, i: ...`, both work. A stage that raises drops that item to `None`,
  reported the same way as `parallel` drops; budget/agent-cap exhaustion fails the run.
  This is the DEFAULT for multi-stage work; only use a `parallel` barrier when a stage
  genuinely needs ALL prior results at once (dedup/merge, early-exit on zero).
- `phase(title)` — start a phase; later `agent()` calls group under it in the UI.
- `log(message)` — emit a progress line.
- `args` — the JSON value passed as the tool's `args`, verbatim.
- `budget` — token budget: `budget.total`, `budget.spent()`, `budget.remaining()`.
  `agent()` raises once the total is reached. With no `token_budget` set, `remaining()`
  is `inf` — so guard budget loops on `budget.total` being set.
  SIZING: the ceiling counts EVERY sub-agent's full output including its reasoning
  tokens. Observed spend per call in real runs: review/investigation median ~6k with
  p90 ~24k; general ~12k; coder median ~20k with p90 ~80k. Budget roughly
  review_calls x 15k + coder_calls x 50k, then double it — the budget is a runaway
  backstop, not a target. When unsure, omit `token_budget` entirely. The run warns at
  75% and 90% spend, and an exhausted run stops mid-fan-out but is resumable with its
  completed calls replaying free.

### Rules that keep workflows correct

- DEFAULT TO `pipeline`. Reach for a `parallel` barrier only when a stage needs every
  prior result together. A barrier wastes the fast items' time waiting on the slowest.
- Scripts must be DETERMINISTIC: `import`, `open`, `time`, and `random` are unavailable
  (this is what makes resume work). Pass any timestamps/seeds in via `args`. Vary agents
  by index/prompt, not by randomness.
- Give every `agent()` in a fan-out a label that is unique in the run — include the
  item's identifying fingerprint (finding id, file, shard AND seat number), e.g.
  `label=f"verify:{f['id']}#{seat}"`. Labels name transcripts and progress lines;
  duplicates make agents indistinguishable there.
- Make each agent's PROMPT self-distinguishing. Two `agent()` calls with identical
  prompt, schema, and agent_type are the same call to the resume cache: on resume their
  cached results are interchangeable. When fanning N skeptics over one finding, say so
  in the prompt ("You are skeptic {i+1} of {n}; take a distinct angle") — this makes
  resume exact and the votes genuinely independent. If a prompt depends on an earlier
  call's side effects (files it wrote), embed that call's output or a revision marker
  in the prompt.
- Concurrency is capped automatically; you may pass large lists to `parallel`/`pipeline`
  (up to 4096) and they all complete — only a handful active execution chains run
  at once. Nested dispatch within one direct worker is serialized. A lifetime cap
  of 1000 agents is a runaway-loop backstop, and built-in nested workers count
  against both that workflow-wide count and the output-token budget.
- Budget admission uses completed output-token spend: reaching the limit blocks
  later direct or nested launches, but calls already in flight can finish and
  produce bounded overshoot.
- Run totals and failed-agent artifacts include finalized output from direct and
  built-in nested workers, including usage recorded before a later failure.
- Use `schema` for anything you'll compute over (counts, filtering, merging). The
  sub-agent is forced to return data matching the schema instead of prose. `agent()`
  returns the FULL object matching the schema — with `{"properties": {"findings": ...}}`
  you get `{"findings": [...]}` back, not the inner array. Unwrap before iterating:
  `for f in (result or {}).get("findings") or []`. Iterating the wrapper dict itself
  yields its string keys and typically crashes the stage, silently dropping that item.

### Quality patterns to compose

- Surface map: before a broad run, map the real work surface — files, modules,
  services, risks, unknowns, owner boundaries, or test targets — then fan out over
  that map instead of guessing stages from the user's wording.
- Shard-and-sweep: split a large surface by package/module/service/concern, send
  focused agents across the shards, then deduplicate and synthesize.
- Cross-cut matrix: review the same target set across dimensions such as correctness,
  security, performance, UX, compatibility, test coverage, and migration risk.
- Research funnel: map unknowns → research in parallel → compare options → produce
  a decision-ready plan. In plan mode this is the main workflow shape.
- Hypothesis tournament: generate several plausible explanations or approaches,
  gather evidence for each, score them with parallel judges, then choose.
- Implementation pipeline: map disjoint workstreams → implement with coder agents →
  verify each stream → integrate and run final checks yourself.
- Migration factory: inventory affected areas → classify risk → batch independent
  edits → run targeted checks → compatibility review.
- Failure triage loop: collect failures → cluster by likely root cause → investigate
  clusters → fix isolated causes → rerun targeted tests.
- Adversarial verify: for each finding, spawn N skeptics prompted to REFUTE it; keep it
  only if a majority fail to refute. Prevents plausible-but-wrong findings surviving.
  Number the seats in each skeptic's prompt (not just the label) so the votes are
  distinct calls with distinct angles rather than N copies of one prompt.
- Loop-until-dry: for unknown-size discovery, keep spawning finders until K consecutive
  rounds surface nothing new (dedup against everything seen, not just confirmed items).
- Judge panel: generate N independent attempts from different angles, score with parallel
  judges, synthesize from the winner.
- Loop-until-budget: `while budget.total and budget.remaining() > 50_000: ...` to scale
  depth to the token ceiling.
- Synthesis gate: after any broad fan-out, merge duplicates, rank confidence, identify
  unresolved gaps, and decide whether another loop is worth the remaining budget.

### Implementing a plan (build mode)

When you are handed a plan to implement and it splits into **independent** workstreams
(modules/files that don't touch each other), orchestrate the implementation instead of
doing it all yourself:

- Give each workstream its own `agent(..., agent_type="coder")` with a complete,
  self-contained task (the goal, the exact files it owns, and the checks to run).
- **Hard safety rule:** there is no per-agent isolation — all sub-agents share the working
  directory. Only fan out workstreams whose file sets are DISJOINT; two agents must never
  edit the same file. Do any coupled or ordering-dependent work yourself, directly.
- Pipeline implement → verify so each workstream's tests run as soon as it lands:

```python
meta = {"name": "implement-plan", "description": "Implement independent parts in parallel",
        "phases": [{"title": "Implement"}, {"title": "Verify"}]}

# Each workstream names the files it exclusively owns — keep these disjoint.
WORKSTREAMS = [
    {"name": "api",  "task": "Implement the API layer per the plan. Files: src/api/*. Run `pytest tests/api`."},
    {"name": "cli",  "task": "Implement the CLI layer per the plan. Files: src/cli/*. Run `pytest tests/cli`."},
]

phase("Implement")
results = await pipeline(
    WORKSTREAMS,
    lambda w: agent(w["task"], label=f"impl:{w['name']}", agent_type="coder"),
    lambda done, w, i: agent(f"Verify and report on the '{w['name']}' workstream: {done}",
                             label=f"verify:{w['name']}", agent_type="investigation", phase="Verify"),
)
return {"workstreams": results}
```

After the workflow returns, integrate the results, run the full test suite yourself, and
report. If the plan is small or its parts are tightly coupled, skip the workflow and just
implement it directly — orchestration is for genuinely independent fan-out.

### Artifacts and transcripts

`run_workflow` returns a concise manifest, not necessarily the full workflow output.
Every completed run persists full artifacts under the state directory and returns
paths including `resultPath` and `transcriptPath`. If the inline tool result says output was omitted, looks incomplete,
or would require a long transcript to understand, READ `resultPath` or `transcriptPath`
with the file-reading tools before deciding work is missing. For normal workflow
output, use only those main files; avoid reading individual sub-agent transcripts
unless you are explicitly debugging workflow execution.

Never re-run a completed workflow solely to recover output from an omitted/truncated
inline result. The workflow already ran; inspect the persisted result/transcript first.
Use `resume_from_run_id` only when you intentionally want to iterate on or change the
workflow, not as a transcript-recovery mechanism.

### Resume

Each run persists its script, full results, a readable transcript, raw JSONL, and a
resume journal under the state directory and returns a `runId` plus artifact paths.
To iterate, edit the script and re-run with `script_path`, or pass `resume_from_run_id`
to replay cached `agent()` results for every call whose content is unchanged (matched
by call content, not position — reordering or inserting calls doesn't invalidate the
rest) and only re-run new/changed calls. Resumed runs journal what they replayed, so
resuming a resumed run replays exactly.

If a `run_workflow` call was interrupted (its tool result says "Operation was
interrupted", or the run was otherwise cut short), do NOT re-run the script from
scratch: call `list_workflow_runs`, take the most recent interrupted run, and pass its
id as `resume_from_run_id` — every agent call it completed replays from the journal at
no cost. `list_workflow_runs` shows only this session's runs; a run still listed as
"running" when no workflow is in flight died mid-run and is resumable the same way.
"""
