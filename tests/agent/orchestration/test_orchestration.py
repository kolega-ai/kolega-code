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
    resume_cache: Optional[dict[int, Any]] = None,
    concurrency: int = 4,
    agent_cap: int = 1000,
    max_agent_depth: int = 1,
) -> tuple[WorkflowRuntime, list[AgentRunSpec], list[tuple[str, dict[str, Any]]], RunJournal]:
    """Build a runtime backed by a recording stub dispatch."""
    calls = []

    async def default_dispatch(spec: AgentRunSpec) -> AgentRunResult:
        calls.append(spec)
        if spec.schema:
            return AgentRunResult(structured={"prompt": spec.prompt, "idx": spec.call_index}, tokens=10)
        return AgentRunResult(text=f"recap:{spec.prompt}", tokens=5)

    events = []

    async def emit(kind: str, content: dict[str, Any]) -> None:
        events.append((kind, content))

    journal = RunJournal.for_run(tmp_path, "run")
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
    runtime, calls, _, _ = make_runtime(
        tmp_path,
        budget=budget,
        accounting=accounting,
        resume_cache={0: (cached_spec.cache_key(), "from-cache")},
    )

    assert await runtime.agent("cached") == "from-cache"
    assert calls == []
    assert accounting.agent_count == 0
    assert budget.spent() == 0


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
    assert cache[0] == ("key0", "value0")
    assert cache[1] == ("key1", {"k": "v"})
    assert 2 not in cache  # failed entries are not cached
    assert journal.read_script() == "meta = {}\n"


@pytest.mark.asyncio
async def test_resume_replays_cached_prefix(tmp_path):
    runtime, calls, _, journal = make_runtime(tmp_path)
    script = META + ("a = await agent('one')\nb = await agent('two', schema={'type': 'object'})\nreturn [a, b]\n")
    first = await runtime.execute(script, args=None)
    assert len(calls) == 2

    # Resume with the recorded cache: same script => zero new dispatches, same result.
    cache = journal.load_cache()
    runtime2, calls2, _, _ = make_runtime(tmp_path, resume_cache=cache)
    second = await runtime2.execute(script, args=None)
    assert len(calls2) == 0
    assert second == first


@pytest.mark.asyncio
async def test_resume_reruns_after_change(tmp_path):
    runtime, calls, _, journal = make_runtime(tmp_path)
    await runtime.execute(META + "await agent('one')\nawait agent('two')\nreturn 1", args=None)
    cache = journal.load_cache()

    # Change the second call's prompt: prefix (index 0) is cached, index 1 re-runs.
    runtime2, calls2, _, _ = make_runtime(tmp_path, resume_cache=cache)
    await runtime2.execute(META + "await agent('one')\nawait agent('CHANGED')\nreturn 1", args=None)
    assert [c.prompt for c in calls2] == ["CHANGED"]


@pytest.mark.asyncio
async def test_resume_reruns_when_max_agent_depth_changes(tmp_path: Path) -> None:
    runtime, _, _, journal = make_runtime(tmp_path, max_agent_depth=1)
    await runtime.execute(META + "return await agent('one')", args=None)

    runtime2, calls2, _, _ = make_runtime(tmp_path, resume_cache=journal.load_cache(), max_agent_depth=2)
    await runtime2.execute(META + "return await agent('one')", args=None)

    assert [call.prompt for call in calls2] == ["one"]
    assert calls2[0].max_agent_depth == 2


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

    assert journal.load_cache() == {0: ("legacy-key", "legacy-value")}
