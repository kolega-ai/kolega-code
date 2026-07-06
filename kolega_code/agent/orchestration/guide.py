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
        (lambda f=f: agent(f"Adversarially verify: {f['title']}", schema=VERDICT_SCHEMA, phase="Verify"))
        for f in (review or {}).get("findings", [])
    ]),
)
confirmed = [f for group in results if group for f in group if f]
return {"confirmed": confirmed}
```

`meta` must be a pure literal (no variables, calls, or f-strings) — it is read
without running the script. Required keys: `name`, `description`. Optional `phases`.

### Primitives (all available as globals in the script)

- `await agent(prompt, *, label=None, phase=None, schema=None, model=None, effort=None, agent_type=None)`
  Dispatch one sub-agent. Returns its report text, or a validated dict when `schema`
  (a JSON Schema) is given, or `None` if the agent failed/was skipped (so you can
  filter it out). `agent_type` is one of "general" (default, full toolset), "investigation"
  (read-only), "browser", or "coder". `model`/`effort` override the model and thinking
  effort for that one call — omit them to inherit the session's settings (almost always correct).
  In plan mode every sub-agent is forced read-only regardless of `agent_type`, so use
  workflows there for parallel research and synthesis, not edits.
- `await parallel(thunks)` — run zero-arg thunks concurrently and wait for ALL (a barrier).
  A thunk that raises resolves to `None`; the call never rejects. Filter `None` before use.
- `await pipeline(items, *stages)` — run each item through all stages independently, with
  NO barrier between stages (item A can be in stage 3 while item B is still in stage 1).
  Each stage is called with `(prev_result, original_item, index)` — write `lambda r: ...`
  or `lambda prev, item, i: ...`, both work. A stage that raises drops that item to `None`.
  This is the DEFAULT for multi-stage work; only use a `parallel` barrier when a stage
  genuinely needs ALL prior results at once (dedup/merge, early-exit on zero).
- `phase(title)` — start a phase; later `agent()` calls group under it in the UI.
- `log(message)` — emit a progress line.
- `args` — the JSON value passed as the tool's `args`, verbatim.
- `budget` — token budget: `budget.total`, `budget.spent()`, `budget.remaining()`.
  `agent()` raises once the total is reached. With no `token_budget` set, `remaining()`
  is `inf` — so guard budget loops on `budget.total` being set.

### Rules that keep workflows correct

- DEFAULT TO `pipeline`. Reach for a `parallel` barrier only when a stage needs every
  prior result together. A barrier wastes the fast items' time waiting on the slowest.
- Scripts must be DETERMINISTIC: `import`, `open`, `time`, and `random` are unavailable
  (this is what makes resume work). Pass any timestamps/seeds in via `args`. Vary agents
  by index/prompt, not by randomness.
- Concurrency is capped automatically; you may pass large lists to `parallel`/`pipeline`
  (up to 4096) and they all complete — only a handful run at once. A lifetime cap of
  1000 agents is a runaway-loop backstop.
- Use `schema` for anything you'll compute over (counts, filtering, merging). The
  sub-agent is forced to return data matching the schema instead of prose.

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
to replay cached `agent()` results for the unchanged prefix and only re-run new/changed
calls.
"""
