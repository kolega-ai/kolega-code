"""Unit tests for the gigacode workflow runtime, executor, journal, and budget.

These exercise the orchestration package in isolation with a stub ``dispatch``,
so they need none of the agent/LLM stack.
"""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import pytest

from kolega_code.agent.orchestration import (
    AgentRunResult,
    AgentRunSpec,
    Budget,
    ResumeCache,
    RunJournal,
    WorkflowAgentCapExceeded,
    WorkflowBudgetExceeded,
    WorkflowRuntime,
    WorkflowScriptError,
    DispatchFn,
    extract_meta,
)
from kolega_code.agent.orchestration.accounting import WorkflowRunAccounting
from kolega_code.agent.model_routing import AtomicModelOverride

META = 'meta = {"name": "t", "description": "d"}\n'


def make_runtime(
    tmp_path: Path,
    *,
    dispatch: Optional[DispatchFn] = None,
    budget: Optional[Budget] = None,
    accounting: Optional[WorkflowRunAccounting] = None,
    resume_cache: Optional[ResumeCache] = None,
    concurrency: int = 4,
    agent_cap: int = 1000,
    max_agent_depth: int = 1,
    run_id: str = "run",
) -> tuple[WorkflowRuntime, list[AgentRunSpec], list[tuple[str, dict[str, Any]]], RunJournal]:
    """Build a runtime backed by a recording stub dispatch.

    Resume tests must give the resumed runtime its own ``run_id``: replayed
    calls are re-recorded, so sharing a journal with the run being resumed
    would append rows into the file the cache was loaded from.
    """
    calls = []

    async def default_dispatch(spec: AgentRunSpec) -> AgentRunResult:
        calls.append(spec)
        if spec.schema:
            return AgentRunResult(structured={"prompt": spec.prompt, "idx": spec.call_index}, tokens=10)
        return AgentRunResult(text=f"recap:{spec.prompt}", tokens=5)

    events = []

    async def emit(kind: str, content: dict[str, Any]) -> None:
        events.append((kind, content))

    journal = RunJournal.for_run(tmp_path, run_id)
    runtime = WorkflowRuntime(
        dispatch=dispatch or default_dispatch,
        emit=emit,
        journal=journal,
        budget=budget if budget is not None else Budget(),
        accounting=accounting,
        resume_cache=resume_cache,
        concurrency=concurrency,
        agent_cap=agent_cap,
        max_agent_depth=max_agent_depth,
    )
    return runtime, calls, events, journal


# --------------------------------------------------------------------- executor
def test_extract_meta_valid():
    meta = extract_meta(META + "return 1")
    assert meta["name"] == "t" and meta["description"] == "d"
    assert meta["max_agent_depth"] == 1


@pytest.mark.parametrize("max_agent_depth", [1, 2])
def test_extract_meta_accepts_supported_agent_depths(max_agent_depth: int) -> None:
    source = f'meta = {{"name": "t", "description": "d", "max_agent_depth": {max_agent_depth}}}\nreturn 1'
    assert extract_meta(source)["max_agent_depth"] == max_agent_depth


@pytest.mark.parametrize("max_agent_depth", [0, 3, True, "2", 1.5, None])
def test_extract_meta_rejects_invalid_agent_depths(max_agent_depth: Any) -> None:
    source = f'meta = {{"name": "t", "description": "d", "max_agent_depth": {max_agent_depth!r}}}\nreturn 1'
    with pytest.raises(WorkflowScriptError, match="max_agent_depth"):
        extract_meta(source)


@pytest.mark.parametrize(
    "source",
    [
        "phase('x')\nreturn 1",  # no meta
        'meta = {"name": "t"}\nreturn 1',  # missing description
        'n = "t"\nmeta = {"name": n, "description": "d"}\nreturn 1',  # non-literal
        'meta = ["not", "a", "dict"]\nreturn 1',  # wrong type
    ],
)
def test_extract_meta_rejects_bad(source):
    with pytest.raises(WorkflowScriptError):
        extract_meta(source)


@pytest.mark.asyncio
async def test_import_and_open_are_blocked(tmp_path):
    runtime, _, _, _ = make_runtime(tmp_path)
    with pytest.raises(Exception):
        await runtime.execute(META + "import os\nreturn os.getcwd()", args=None)
    with pytest.raises(Exception):
        await runtime.execute(META + "return open('/etc/hosts').read()", args=None)


@pytest.mark.asyncio
async def test_args_passthrough_and_multiline_strings(tmp_path):
    runtime, _, _, _ = make_runtime(tmp_path)
    assert await runtime.execute(META + "return args['x'] * 2", args={"x": 21}) == 42
    # AST-level wrapping must not corrupt multi-line string literals.
    out = await runtime.execute(META + 'x = """a\nb"""\nreturn x', args=None)
    assert out == "a\nb"


# ---------------------------------------------------------------------- agent()
@pytest.mark.asyncio
async def test_agent_returns_text_and_structured(tmp_path):
    runtime, calls, _, _ = make_runtime(tmp_path)
    script = META + ("a = await agent('hello')\nb = await agent('world', schema={'type': 'object'})\nreturn [a, b]\n")
    out = await runtime.execute(script, args=None)
    assert out[0] == "recap:hello"
    assert out[1] == {"prompt": "world", "idx": 1}
    assert len(calls) == 2
    assert all(call.max_agent_depth == 1 for call in calls)


@pytest.mark.asyncio
async def test_agent_spec_carries_explicit_max_agent_depth(tmp_path: Path) -> None:
    runtime, calls, _, _ = make_runtime(tmp_path, max_agent_depth=2)
    await runtime.execute(META + "return await agent('hello')", args=None)
    assert calls[0].max_agent_depth == 2


@pytest.mark.asyncio
async def test_agent_normalizes_atomic_model_override(tmp_path: Path) -> None:
    runtime, calls, _, _ = make_runtime(tmp_path)
    script = (
        META
        + "return await agent('hello', model_override={"
        + "'provider': 'DEEPSEEK', 'model': 'deepseek-v4-flash', 'effort': 'HIGH'})"
    )

    assert await runtime.execute(script, args=None) == "recap:hello"
    assert calls[0].model_override == AtomicModelOverride(
        provider="deepseek",
        model="deepseek-v4-flash",
        effort="high",
    )
    with pytest.raises(Exception):
        calls[0].model_override.effort = "low"  # type: ignore[misc]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override",
    [
        "{'provider': 'anthropic', 'model': 'claude-opus-4-8'}",
        "{'provider': 'anthropic', 'model': 'claude-opus-4-8', 'effort': 'high', 'extra': 1}",
        "['anthropic', 'claude-opus-4-8', 'high']",
    ],
)
async def test_agent_rejects_non_atomic_model_override_before_dispatch(
    tmp_path: Path,
    override: str,
) -> None:
    runtime, calls, _, _ = make_runtime(tmp_path)

    with pytest.raises(WorkflowScriptError, match="model_override"):
        await runtime.execute(META + f"return await agent('hello', model_override={override})", args=None)
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", ["model='claude-opus-4-8'", "effort='high'", "model='x', effort='high'"])
async def test_agent_legacy_routing_kwargs_have_targeted_migration_error(
    tmp_path: Path,
    legacy: str,
) -> None:
    runtime, calls, _, _ = make_runtime(tmp_path)

    with pytest.raises(WorkflowScriptError, match="Migrate.*model_override"):
        await runtime.execute(META + f"return await agent('hello', {legacy})", args=None)
    assert calls == []


@pytest.mark.asyncio
async def test_agent_reservation_is_absent_from_spec_serialization(tmp_path: Path) -> None:
    serialized_specs: list[dict[str, Any]] = []

    async def dispatch(spec: AgentRunSpec) -> AgentRunResult:
        serialized_specs.append(asdict(spec))
        return AgentRunResult(text="ok", tokens=3)

    runtime, _, _, _ = make_runtime(tmp_path, dispatch=dispatch)
    assert await runtime.execute(META + "return await agent('hello')", args=None) == "ok"
    assert "reservation" not in serialized_specs[0]


@pytest.mark.asyncio
async def test_failed_agent_returns_none(tmp_path):
    async def dispatch(spec):
        return AgentRunResult(status="failed", error="boom")

    runtime, _, _, _ = make_runtime(tmp_path, dispatch=dispatch)
    out = await runtime.execute(META + "return await agent('x')", args=None)
    assert out is None


# -------------------------------------------------------------------- parallel()
@pytest.mark.asyncio
async def test_parallel_runs_all_and_isolates_failures(tmp_path):
    async def dispatch(spec):
        if "fail" in spec.prompt:
            raise RuntimeError("nope")
        return AgentRunResult(text=spec.prompt, tokens=1)

    runtime, _, _, _ = make_runtime(tmp_path, dispatch=dispatch)
    script = META + (
        "res = await parallel([\n"
        "    (lambda: agent('ok1')),\n"
        "    (lambda: agent('fail')),\n"
        "    (lambda: agent('ok2')),\n"
        "])\n"
        "return res\n"
    )
    out = await runtime.execute(script, args=None)
    assert out == ["ok1", None, "ok2"]


# -------------------------------------------------------------------- pipeline()
@pytest.mark.asyncio
async def test_pipeline_stage_arity_and_failure_isolation(tmp_path):
    async def dispatch(spec):
        if spec.prompt == "stage1:bad":
            raise RuntimeError("bad")
        return AgentRunResult(text=spec.prompt, tokens=1)

    runtime, _, _, _ = make_runtime(tmp_path, dispatch=dispatch)
    # Stage 1 uses 3 args; stage 2 uses 1 arg — both must be callable.
    script = META + (
        "out = await pipeline(\n"
        "    ['good', 'bad'],\n"
        "    lambda item, orig, idx: agent(f'stage1:{item}'),\n"
        "    lambda prev: agent(f'stage2:{prev}'),\n"
        ")\n"
        "return out\n"
    )
    out = await runtime.execute(script, args=None)
    assert out[0] == "stage2:stage1:good"
    assert out[1] is None  # 'bad' dropped at stage 1


# ------------------------------------------------- fan-out drop reporting
@pytest.mark.asyncio
async def test_pipeline_stage_exception_recorded_with_script_line(tmp_path):
    script = META + (
        "def busted(review):\n"
        "    return review.get('missing')\n"  # script line 3: AttributeError on a str
        "out = await pipeline(\n"
        "    ['a', 'b'],\n"
        "    lambda item: agent('review ' + item),\n"
        "    lambda prev: busted(prev),\n"
        ")\n"
        "return out\n"
    )

    async def dispatch(spec: AgentRunSpec) -> AgentRunResult:
        return AgentRunResult(text="not a dict", tokens=1)

    runtime, _, events, journal = make_runtime(tmp_path, dispatch=dispatch)
    assert await runtime.execute(script, args=None) == [None, None]

    drops = runtime.dropped_items
    assert len(drops) == 2
    assert all(d["where"] == "pipeline" and d["stage"] == 1 for d in drops)
    assert all("AttributeError" in d["error"] for d in drops)
    assert all(d["script_line"] == 3 for d in drops)

    transcript_events = [json.loads(line) for line in journal.transcript_jsonl_path.read_text().splitlines()]
    dropped_events = [e for e in transcript_events if e.get("type") == "fanout_item_dropped"]
    assert len(dropped_events) == 2

    logs = [c["message"] for kind, c in events if kind == "workflow_log"]
    assert any("item dropped" in m and "AttributeError" in m and "script line 3" in m for m in logs)
    # Every item came back None with drops recorded — the script-bug hint fires.
    assert any("script bug" in m for m in logs)


@pytest.mark.asyncio
async def test_parallel_thunk_exception_recorded_without_all_none_hint(tmp_path):
    runtime, _, events, _ = make_runtime(tmp_path)
    script = META + (
        "res = await parallel([\n"
        "    (lambda: agent('ok')),\n"
        "    (lambda: [][5]),\n"  # script line 4: IndexError
        "])\n"
        "return res\n"
    )
    assert await runtime.execute(script, args=None) == ["recap:ok", None]

    drops = runtime.dropped_items
    assert len(drops) == 1
    assert drops[0]["where"] == "parallel"
    assert drops[0]["item"] == 1
    assert drops[0]["stage"] is None
    assert "IndexError" in drops[0]["error"]
    assert drops[0]["script_line"] == 4

    logs = [c["message"] for kind, c in events if kind == "workflow_log"]
    assert any("item dropped" in m for m in logs)
    assert not any("script bug" in m for m in logs)  # one item succeeded


@pytest.mark.asyncio
async def test_budget_exhaustion_propagates_out_of_fanout(tmp_path):
    """Budget exhaustion fails the run instead of shredding the fan-out into Nones."""
    budget = Budget(total=5)
    accounting = WorkflowRunAccounting(budget, agent_cap=1000)
    runtime, calls, _, _ = make_runtime(tmp_path, budget=budget, accounting=accounting)
    script = META + (
        "await agent('first')\n"  # spends 5 of 5
        "return await parallel([(lambda: agent('starved-1')), (lambda: agent('starved-2'))])\n"
    )
    with pytest.raises(WorkflowBudgetExceeded):
        await runtime.execute(script, args=None)
    assert [c.prompt for c in calls] == ["first"]
    assert runtime.dropped_items == []  # exhaustion is not a per-item drop


@pytest.mark.asyncio
async def test_agent_cap_exhaustion_propagates_out_of_fanout(tmp_path):
    runtime, calls, _, _ = make_runtime(tmp_path, agent_cap=1)
    script = META + ("await agent('first')\nreturn await parallel([(lambda: agent('over-cap'))])\n")
    with pytest.raises(WorkflowAgentCapExceeded):
        await runtime.execute(script, args=None)
    assert [c.prompt for c in calls] == ["first"]


@pytest.mark.asyncio
async def test_workflow_script_error_in_fanout_still_drops_but_is_recorded(tmp_path):
    """Per-call authoring errors keep the documented drop-to-None contract
    (docs/gigacode.mdx promises this for malformed overrides) — but loudly."""
    runtime, calls, _, _ = make_runtime(tmp_path)
    script = META + "return await parallel([(lambda: agent('x', model='legacy'))])\n"
    assert await runtime.execute(script, args=None) == [None]
    assert calls == []
    assert len(runtime.dropped_items) == 1
    assert "WorkflowScriptError" in runtime.dropped_items[0]["error"]


# ------------------------------------------------------------------- phase/log
@pytest.mark.asyncio
async def test_phase_and_log_emit_events(tmp_path):
    runtime, _, events, _ = make_runtime(tmp_path)
    await runtime.execute(META + "phase('Find')\nlog('hi')\nreturn 1", args=None)
    kinds = [k for k, _ in events]
    assert "workflow_phase" in kinds
    assert "workflow_log" in kinds


@pytest.mark.asyncio
async def test_phase_sets_default_phase_on_agents(tmp_path):
    runtime, calls, _, _ = make_runtime(tmp_path)
    await runtime.execute(META + "phase('P')\nawait agent('x')\nreturn 1", args=None)
    assert calls[0].phase == "P"


# ---------------------------------------------------------------------- budget
@pytest.mark.asyncio
async def test_budget_accounting_and_ceiling(tmp_path):
    runtime, _, _, _ = make_runtime(tmp_path, budget=Budget(total=10))
    # Two agents at 5 tokens each bring spent to 10 == total; the third must raise.
    script = META + ("await agent('a')\nawait agent('b')\nawait agent('c')\nreturn budget.spent()\n")
    with pytest.raises(WorkflowBudgetExceeded):
        await runtime.execute(script, args=None)


@pytest.mark.asyncio
async def test_unbounded_budget_remaining_is_inf(tmp_path):
    runtime, _, _, _ = make_runtime(tmp_path, budget=Budget())
    out = await runtime.execute(META + "return budget.remaining() == float('inf')", args=None)
    assert out is True


# ----------------------------------------------------------------------- caps
@pytest.mark.asyncio
async def test_agent_cap_raises(tmp_path):
    runtime, _, _, _ = make_runtime(tmp_path, agent_cap=2)
    script = META + "for i in range(5):\n    await agent(str(i))\nreturn 1"
    with pytest.raises(WorkflowAgentCapExceeded):
        await runtime.execute(script, args=None)


@pytest.mark.asyncio
async def test_cached_agent_does_not_reserve_or_spend(tmp_path: Path) -> None:
    budget = Budget(total=5)
    accounting = WorkflowRunAccounting(budget, agent_cap=1)
    cached_spec = AgentRunSpec(prompt="cached")
    prior = RunJournal.for_run(tmp_path, "prior")
    prior.record(0, cached_spec.cache_key(), None, "from-cache", tokens=10)
    runtime, calls, _, journal = make_runtime(
        tmp_path,
        budget=budget,
        accounting=accounting,
        resume_cache=prior.load_cache(),
    )

    assert await runtime.agent("cached") == "from-cache"
    assert calls == []
    assert accounting.agent_count == 0
    assert budget.spent() == 0
    # The replay is re-recorded under the new run — marked, and without the
    # original token count (replays spend nothing).
    row = json.loads(journal.journal_path.read_text().splitlines()[0])
    assert row["replayed"] is True
    assert row["status"] == "completed"
    assert "tokens" not in row


@pytest.mark.asyncio
async def test_queued_cancellation_does_not_reserve_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kolega_code.agent.orchestration import runtime as runtime_module

    monkeypatch.setattr(runtime_module, "START_STAGGER_SECONDS", 0)
    started = asyncio.Event()
    release = asyncio.Event()

    async def dispatch(spec: AgentRunSpec) -> AgentRunResult:
        if spec.prompt == "first":
            started.set()
            await release.wait()
        return AgentRunResult(text=spec.prompt, tokens=1)

    budget = Budget()
    accounting = WorkflowRunAccounting(budget, agent_cap=2)
    runtime, _, _, _ = make_runtime(
        tmp_path,
        dispatch=dispatch,
        budget=budget,
        accounting=accounting,
        concurrency=1,
    )

    first = asyncio.create_task(runtime.agent("first"))
    await started.wait()
    queued = asyncio.create_task(runtime.agent("queued"))
    await asyncio.sleep(0)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued

    assert accounting.agent_count == 1
    release.set()
    assert await first == "first"
    assert budget.spent() == 1


@pytest.mark.asyncio
async def test_parallel_fanout_cap(tmp_path):
    runtime, _, _, _ = make_runtime(tmp_path)
    script = META + "return await parallel([(lambda: agent('x'))] * 5000)"
    with pytest.raises(WorkflowScriptError):
        await runtime.execute(script, args=None)


# --------------------------------------------------------------------- journal
def test_journal_round_trip_and_cache(tmp_path):
    journal = RunJournal.for_run(tmp_path, "j")
    journal.write_script("meta = {}\n")
    journal.record(0, "key0", "label0", "value0")
    journal.record(1, "key1", "label1", {"k": "v"})
    journal.record(2, "key2", None, None, status="failed")
    cache = journal.load_cache()
    row0 = cache.take("key0")
    assert row0 is not None and row0["value"] == "value0"
    assert cache.take("key0") is None  # each row is consumed once
    row1 = cache.take("key1")
    assert row1 is not None and row1["value"] == {"k": "v"}
    assert cache.take("key2") is None  # failed entries are not cached
    assert journal.read_script() == "meta = {}\n"


@pytest.mark.asyncio
async def test_resume_replays_cached_prefix(tmp_path):
    runtime, calls, _, journal = make_runtime(tmp_path)
    script = META + ("a = await agent('one')\nb = await agent('two', schema={'type': 'object'})\nreturn [a, b]\n")
    first = await runtime.execute(script, args=None)
    assert len(calls) == 2

    # Resume with the recorded cache: same script => zero new dispatches, same result.
    cache = journal.load_cache()
    runtime2, calls2, _, _ = make_runtime(tmp_path, resume_cache=cache, run_id="run2")
    second = await runtime2.execute(script, args=None)
    assert len(calls2) == 0
    assert second == first


@pytest.mark.asyncio
async def test_resume_reruns_after_change(tmp_path):
    runtime, calls, _, journal = make_runtime(tmp_path)
    await runtime.execute(META + "await agent('one')\nawait agent('two')\nreturn 1", args=None)
    cache = journal.load_cache()

    # Change the second call's prompt: 'one' replays, the changed call re-runs.
    runtime2, calls2, _, _ = make_runtime(tmp_path, resume_cache=cache, run_id="run2")
    await runtime2.execute(META + "await agent('one')\nawait agent('CHANGED')\nreturn 1", args=None)
    assert [c.prompt for c in calls2] == ["CHANGED"]


@pytest.mark.asyncio
async def test_resume_reruns_when_max_agent_depth_changes(tmp_path: Path) -> None:
    runtime, _, _, journal = make_runtime(tmp_path, max_agent_depth=1)
    await runtime.execute(META + "return await agent('one')", args=None)

    runtime2, calls2, _, _ = make_runtime(tmp_path, resume_cache=journal.load_cache(), max_agent_depth=2, run_id="run2")
    await runtime2.execute(META + "return await agent('one')", args=None)

    assert [call.prompt for call in calls2] == ["one"]
    assert calls2[0].max_agent_depth == 2


@pytest.mark.asyncio
async def test_resume_matches_by_content_not_index(tmp_path: Path) -> None:
    """Replay is keyed by call content, so reordered calls still hit the cache."""
    runtime, _, _, journal = make_runtime(tmp_path)
    await runtime.execute(META + "await agent('x')\nawait agent('y')\nreturn 1", args=None)

    runtime2, calls2, _, _ = make_runtime(tmp_path, resume_cache=journal.load_cache(), run_id="run2")
    await runtime2.execute(META + "await agent('y')\nawait agent('x')\nreturn 1", args=None)
    assert calls2 == []


@pytest.mark.asyncio
async def test_resume_fifo_orders_identical_keys_by_original_index(tmp_path: Path) -> None:
    """Identical-key rows replay in original issue (index) order even though the
    journal file is written in completion order."""
    key = AgentRunSpec(prompt="same").cache_key()
    prior = RunJournal.for_run(tmp_path, "prior")
    prior.record(2, key, None, "v2")
    prior.record(0, key, None, "v0")
    prior.record(1, key, None, "v1")

    runtime, calls, _, _ = make_runtime(tmp_path, resume_cache=prior.load_cache())
    script = META + "out = []\nfor _ in range(3):\n    out.append(await agent('same'))\nreturn out"
    assert await runtime.execute(script, args=None) == ["v0", "v1", "v2"]
    assert calls == []


@pytest.mark.asyncio
async def test_resume_insertion_only_runs_new_call(tmp_path: Path) -> None:
    runtime, _, _, journal = make_runtime(tmp_path)
    await runtime.execute(META + "await agent('a')\nawait agent('b')\nreturn 1", args=None)

    runtime2, calls2, _, _ = make_runtime(tmp_path, resume_cache=journal.load_cache(), run_id="run2")
    await runtime2.execute(META + "await agent('new')\nawait agent('a')\nawait agent('b')\nreturn 1", args=None)
    assert [c.prompt for c in calls2] == ["new"]


@pytest.mark.asyncio
async def test_resume_exhausted_key_runs_live(tmp_path: Path) -> None:
    prior = RunJournal.for_run(tmp_path, "prior")
    prior.record(0, AgentRunSpec(prompt="same").cache_key(), None, "cached")

    runtime, calls, _, _ = make_runtime(tmp_path, resume_cache=prior.load_cache())
    result = await runtime.execute(META + "return [await agent('same'), await agent('same')]", args=None)
    assert result == ["cached", "recap:same"]
    assert [c.prompt for c in calls] == ["same"]


@pytest.mark.asyncio
async def test_resume_identical_key_skips_failed_rows(tmp_path: Path) -> None:
    key = AgentRunSpec(prompt="same").cache_key()
    prior = RunJournal.for_run(tmp_path, "prior")
    prior.record(0, key, None, "v0")
    prior.record(1, key, None, None, status="failed")
    prior.record(2, key, None, "v2")

    runtime, calls, _, _ = make_runtime(tmp_path, resume_cache=prior.load_cache())
    script = META + "out = []\nfor _ in range(3):\n    out.append(await agent('same'))\nreturn out"
    assert await runtime.execute(script, args=None) == ["v0", "v2", "recap:same"]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_replayed_row_written_to_new_journal(tmp_path: Path) -> None:
    runtime, _, _, journal = make_runtime(tmp_path)
    await runtime.execute(META + "return await agent('a', label='orig-label')", args=None)
    original = json.loads(journal.journal_path.read_text().splitlines()[0])

    runtime2, calls2, _, journal2 = make_runtime(tmp_path, resume_cache=journal.load_cache(), run_id="run2")
    await runtime2.execute(META + "return await agent('a', label='new-label')", args=None)
    assert calls2 == []

    row = json.loads(journal2.journal_path.read_text().splitlines()[0])
    assert row["replayed"] is True
    assert row["status"] == "completed"
    assert row["index"] == 0
    assert row["key"] == original["key"]
    assert row["value"] == original["value"]
    assert row["label"] == "new-label"  # the current script's label, not the original's
    assert "tokens" not in row


@pytest.mark.asyncio
async def test_resume_of_resume_replays_everything(tmp_path: Path) -> None:
    """Replayed rows are journaled, so a chain of resumes never re-dispatches."""
    runtime, _, _, journal = make_runtime(tmp_path)
    first = await runtime.execute(META + "return [await agent('a'), await agent('b')]", args=None)

    runtime2, calls2, _, journal2 = make_runtime(tmp_path, resume_cache=journal.load_cache(), run_id="run2")
    second = await runtime2.execute(META + "return [await agent('a'), await agent('b')]", args=None)
    assert calls2 == []

    runtime3, calls3, _, _ = make_runtime(tmp_path, resume_cache=journal2.load_cache(), run_id="run3")
    third = await runtime3.execute(META + "return [await agent('a'), await agent('b')]", args=None)
    assert calls3 == []
    assert first == second == third


@pytest.mark.asyncio
async def test_replayed_completed_none_value_still_replays(tmp_path: Path) -> None:
    prior = RunJournal.for_run(tmp_path, "prior")
    prior.record(0, AgentRunSpec(prompt="a").cache_key(), None, None)

    runtime, calls, _, _ = make_runtime(tmp_path, resume_cache=prior.load_cache())
    assert await runtime.execute(META + "return await agent('a')", args=None) is None
    assert calls == []


@pytest.mark.asyncio
async def test_resume_after_interrupt_with_colliding_labels(tmp_path: Path) -> None:
    """Labels play no role in replay identity: after an interrupt, a later call
    sharing an earlier call's label can never be served the earlier result."""
    script = META + (
        "a = await agent('verify finding A', label='verify:seat')\n"
        "b = await agent('verify finding B', label='verify:seat')\n"
        "return {'A': a, 'B': b}\n"
    )

    async def interrupting(spec: AgentRunSpec) -> AgentRunResult:
        if spec.prompt == "verify finding B":
            raise asyncio.CancelledError()
        return AgentRunResult(text="verdict-A", tokens=5)

    runtime, _, _, journal = make_runtime(tmp_path, dispatch=interrupting)
    with pytest.raises(asyncio.CancelledError):
        await runtime.execute(script, args=None)

    runtime2, calls2, _, _ = make_runtime(tmp_path, resume_cache=journal.load_cache(), run_id="run2")
    result = await runtime2.execute(script, args=None)
    assert result == {"A": "verdict-A", "B": "recap:verify finding B"}
    assert [c.prompt for c in calls2] == ["verify finding B"]


@pytest.mark.asyncio
async def test_duplicate_label_warns_once(tmp_path: Path) -> None:
    runtime, _, events, journal = make_runtime(tmp_path)
    script = META + (
        "await agent('a', label='dup')\n"
        "await agent('b', label='dup')\n"
        "await agent('c', label='dup')\n"
        "await agent('d', label='unique')\n"
        "return 1\n"
    )
    await runtime.execute(script, args=None)
    warnings = [c for kind, c in events if kind == "workflow_log" and "duplicate agent label" in c["message"]]
    assert len(warnings) == 1
    assert "'dup'" in warnings[0]["message"]

    # The hint also fires on a fully-replayed resume.
    runtime2, calls2, events2, _ = make_runtime(tmp_path, resume_cache=journal.load_cache(), run_id="run2")
    await runtime2.execute(script, args=None)
    assert calls2 == []
    warnings2 = [c for kind, c in events2 if kind == "workflow_log" and "duplicate agent label" in c["message"]]
    assert len(warnings2) == 1


def test_cache_key_uses_atomic_override_or_inherited_routing_fingerprint() -> None:
    no_override_a = AgentRunSpec(prompt="same", routing_fingerprint="routing-a")
    no_override_b = AgentRunSpec(prompt="same", routing_fingerprint="routing-b")
    with_override_a = AgentRunSpec(
        prompt="same",
        model_override=AtomicModelOverride("anthropic", "claude-sonnet-4-5-20250929", None),
        routing_fingerprint="routing-a",
    )
    with_override_b = AgentRunSpec(
        prompt="same",
        model_override=AtomicModelOverride("anthropic", "claude-sonnet-4-5-20250929", None),
        routing_fingerprint="routing-b",
    )

    assert no_override_a.cache_key() != no_override_b.cache_key()
    assert with_override_a.cache_key() == with_override_b.cache_key()
    assert no_override_a.cache_key() != with_override_a.cache_key()


def test_cache_key_includes_actual_execution_class_and_read_only_mode() -> None:
    writable = AgentRunSpec(
        prompt="same",
        agent_type="coder",
        actual_agent_type="CoderAgent",
        read_only_mode=False,
    )
    forced_plan = AgentRunSpec(
        prompt="same",
        agent_type="coder",
        actual_agent_type="InvestigationAgent",
        read_only_mode=True,
    )

    assert writable.cache_key() != forced_plan.cache_key()


def test_cache_key_includes_browser_target() -> None:
    playwright = AgentRunSpec(prompt="same", agent_type="browser")
    chrome = AgentRunSpec(prompt="same", agent_type="browser", browser_target="chrome")

    assert playwright.cache_key() != chrome.cache_key()


@pytest.mark.asyncio
async def test_runtime_passes_browser_target_to_dispatch(tmp_path: Path) -> None:
    runtime, calls, _, _ = make_runtime(tmp_path)

    await runtime.agent("browse", agent_type="browser", browser_target="chrome")

    assert calls[0].browser_target == "chrome"


@pytest.mark.asyncio
async def test_runtime_rejects_browser_target_for_non_browser_agent(tmp_path: Path) -> None:
    runtime, _, _, _ = make_runtime(tmp_path)

    with pytest.raises(WorkflowScriptError, match="only valid"):
        await runtime.agent("investigate", agent_type="investigation", browser_target="chrome")


@pytest.mark.asyncio
async def test_runtime_records_dispatch_routing_metadata(tmp_path: Path) -> None:
    async def dispatch(_spec: AgentRunSpec) -> AgentRunResult:
        return AgentRunResult(
            text="ok",
            requested_routing={"provider": "deepseek", "model": "deepseek-v4-flash", "effort": "high"},
            effective_routing={"provider": "deepseek", "model": "deepseek-v4-flash", "effort": "high"},
            actual_agent_type="InvestigationAgent",
        )

    runtime, _, _, journal = make_runtime(tmp_path, dispatch=dispatch)
    await runtime.agent(
        "hello",
        model_override={"provider": "deepseek", "model": "deepseek-v4-flash", "effort": "high"},
    )

    journal_entry = json.loads(journal.journal_path.read_text().splitlines()[0])
    transcript_entry = json.loads(journal.transcript_jsonl_path.read_text().splitlines()[0])
    assert journal_entry["requested_routing"]["provider"] == "deepseek"
    assert journal_entry["effective_routing"]["model"] == "deepseek-v4-flash"
    assert journal_entry["actual_agent_type"] == "InvestigationAgent"
    assert transcript_entry["requested_routing"] == journal_entry["requested_routing"]
    assert transcript_entry["actual_agent_type"] == "InvestigationAgent"


@pytest.mark.asyncio
async def test_runtime_preserves_null_requested_routing_metadata(tmp_path: Path) -> None:
    async def dispatch(_spec: AgentRunSpec) -> AgentRunResult:
        return AgentRunResult(
            text="ok",
            requested_routing=None,
            effective_routing={"provider": "anthropic", "model": "inherited", "effort": None},
            actual_agent_type="GeneralAgent",
        )

    runtime, _, _, journal = make_runtime(tmp_path, dispatch=dispatch)
    await runtime.agent("hello")

    journal_entry = json.loads(journal.journal_path.read_text().splitlines()[0])
    transcript_entry = json.loads(journal.transcript_jsonl_path.read_text().splitlines()[0])
    assert "requested_routing" in journal_entry
    assert journal_entry["requested_routing"] is None
    assert "requested_routing" in transcript_entry
    assert transcript_entry["requested_routing"] is None


def test_old_and_malformed_journal_rows_remain_readable(tmp_path: Path) -> None:
    journal = RunJournal.for_run(tmp_path, "legacy")
    journal.ensure_dirs()
    journal.journal_path.write_text(
        "\n".join(
            [
                '{"index": 0, "key": "legacy-key", "value": "legacy-value"}',
                "[]",
                '{"not": "a resume row"}',
                "{malformed",
            ]
        )
        + "\n"
    )

    cache = journal.load_cache()
    assert len(cache) == 1
    row = cache.take("legacy-key")
    assert row is not None and row["value"] == "legacy-value"
    assert cache.take("legacy-key") is None
