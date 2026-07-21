"""Unit tests for the gigacode workflow runtime, executor, journal, and budget.

These exercise the orchestration package in isolation with a stub ``dispatch``,
so they need none of the agent/LLM stack.
"""

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
    extract_meta,
)

META = 'meta = {"name": "t", "description": "d"}\n'


def make_runtime(
    tmp_path,
    *,
    dispatch=None,
    budget=None,
    resume_cache=None,
    concurrency=4,
    agent_cap=1000,
    max_agent_depth=1,
):
    """Build a runtime backed by a recording stub dispatch."""
    calls = []

    async def default_dispatch(spec: AgentRunSpec) -> AgentRunResult:
        calls.append(spec)
        if spec.schema:
            return AgentRunResult(structured={"prompt": spec.prompt, "idx": spec.call_index}, tokens=10)
        return AgentRunResult(text=f"recap:{spec.prompt}", tokens=5)

    events = []

    async def emit(kind, content):
        events.append((kind, content))

    journal = RunJournal.for_run(tmp_path, "run")
    runtime = WorkflowRuntime(
        dispatch=dispatch or default_dispatch,
        emit=emit,
        journal=journal,
        budget=budget if budget is not None else Budget(),
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
def test_extract_meta_accepts_supported_agent_depths(max_agent_depth):
    source = f'meta = {{"name": "t", "description": "d", "max_agent_depth": {max_agent_depth}}}\nreturn 1'
    assert extract_meta(source)["max_agent_depth"] == max_agent_depth


@pytest.mark.parametrize("max_agent_depth", [0, 3, True, "2", 1.5, None])
def test_extract_meta_rejects_invalid_agent_depths(max_agent_depth):
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
async def test_agent_spec_carries_explicit_max_agent_depth(tmp_path):
    runtime, calls, _, _ = make_runtime(tmp_path, max_agent_depth=2)
    await runtime.execute(META + "return await agent('hello')", args=None)
    assert calls[0].max_agent_depth == 2


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
async def test_resume_reruns_when_max_agent_depth_changes(tmp_path):
    runtime, _, _, journal = make_runtime(tmp_path, max_agent_depth=1)
    await runtime.execute(META + "return await agent('one')", args=None)

    runtime2, calls2, _, _ = make_runtime(tmp_path, resume_cache=journal.load_cache(), max_agent_depth=2)
    await runtime2.execute(META + "return await agent('one')", args=None)

    assert [call.prompt for call in calls2] == ["one"]
    assert calls2[0].max_agent_depth == 2
