from __future__ import annotations

import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from kolega_code.cli.session_journal import SessionEvent
from kolega_code.cli.session_usage import derive_session_usage
from kolega_code.events import AgentEvent
from kolega_code.session.projection import PresentationState, ProjectionError, fold, replay

FIXTURE = json.loads((Path(__file__).parents[1] / "fixtures" / "usage.json").read_text(encoding="utf-8"))
FOLD_JS = Path(__file__).resolve().parents[2] / "kolega_code/web/assets/fold.js"
COUNTS = ("requests", "responses", "reported", "unreported", "failed", "open", "abandoned", "partial", "missing")
TOKENS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_input_tokens",
    "cache_write_input_tokens",
    "reasoning_output_tokens",
)


def _event(data: dict[str, Any]) -> AgentEvent:
    return AgentEvent(sender="fixture", **deepcopy(data))


def _node(tmp_path: Path, payload: Any, script: str) -> Any:
    if shutil.which("node") is None:
        pytest.skip("node is required for JavaScript usage fold checks")
    data = tmp_path / "usage-input.json"
    harness = tmp_path / "usage-check.mjs"
    data.write_text(json.dumps(payload), encoding="utf-8")
    harness.write_text(
        f"""
import assert from "node:assert/strict";
import {{ readFileSync }} from "node:fs";
import {{ emptyState, fold, replay, toDict }} from {json.dumps(str(FOLD_JS))};
const input = JSON.parse(readFileSync(process.argv[2], "utf8"));
{script}
""",
        encoding="utf-8",
    )
    result = subprocess.run(["node", str(harness), str(data)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout or "null")


def test_golden_usage_at_every_prefix_and_seek(tmp_path: Path) -> None:
    events = [_event(step["event"]) for step in FIXTURE["steps"]]
    expected = [FIXTURE["initial_usage"], *(step["expected_usage"] for step in FIXTURE["steps"])]
    state = PresentationState()
    prefixes = [state.to_dict()]
    assert state.usage is None
    for cut, event in enumerate(events, 1):
        fold(state, event)
        assert state.usage == expected[cut], FIXTURE["steps"][cut - 1]["name"]
        assert state.to_dict()["usage"] == expected[cut]
        assert state.sub_agents == {}
        assert "usage_updated" not in state.unknown_event_types
        prefixes.append(deepcopy(state.to_dict()))
    for cut in reversed(range(len(events) + 1)):
        sought = replay(events[:cut]).to_dict()
        assert sought["usage"] == expected[cut]
        assert sought == prefixes[cut]
    actual = _node(
        tmp_path,
        {"events": [event.model_dump(mode="json") for event in events], "expected": expected},
        """
const state = emptyState();
const prefixes = [toDict(state)];
assert.equal(state.usage, null);
for (let index = 0; index < input.events.length; index++) {
  fold(state, input.events[index]);
  assert.deepEqual(state.usage, input.expected[index + 1], `incremental prefix ${index + 1}`);
  assert.deepEqual(toDict(state).usage, input.expected[index + 1]);
  assert.equal(state.subAgents.size, 0);
  assert.equal(state.unknownEventTypes.has("usage_updated"), false);
  prefixes.push(toDict(state));
}
for (let cut = input.events.length; cut >= 0; cut--) {
  const sought = toDict(replay(input.events.slice(0, cut)));
  assert.deepEqual(sought.usage, input.expected[cut], `seek prefix ${cut}`);
  assert.deepEqual(sought, prefixes[cut]);
}
process.stdout.write(JSON.stringify(prefixes));
""",
    )
    assert actual == prefixes


def test_fixture_oracles_are_compatible_with_recorded_evidence() -> None:
    journal = [SessionEvent.from_dict(event) for event in FIXTURE["journal"]]
    for step in FIXTURE["steps"]:
        if "derive" not in step:
            continue
        evidence = step["derive"]
        aggregate = derive_session_usage(journal[: evidence["journal_prefix"]], live_run_id=evidence["live_run_id"])
        assert step["event"]["content"]["usage"] == aggregate, step["name"]
        assert step["expected_usage"] == {"source_seq": evidence["journal_prefix"], **aggregate}
    by_name = {step["name"]: step["expected_usage"] for step in FIXTURE["steps"]}
    assert by_name["known empty session"]["complete"] is True
    unknown = by_name["unknown live usage"]
    assert unknown["total_tokens"] == unknown["known_requests"]["total_tokens"] == 0
    assert unknown["open"] == unknown["missing"] == 1
    closed = by_name["same source sequence closes the live run"]
    assert closed["source_seq"] == unknown["source_seq"]
    assert closed["open"] == 0 and closed["abandoned"] == 1
    partial = by_name["partial known subtotal"]
    assert partial["input_tokens"] == 7 and partial["known_requests"]["input_tokens"] == 1
    assert partial["output_tokens"] == partial["known_requests"]["output_tokens"] == 0
    zero = by_name["explicit known zero replaces unknown"]
    assert zero["total_tokens"] == 0 and zero["known_requests"]["total_tokens"] == 1
    assert zero["complete"] is True
    refined = by_name["late high precision reported refinement"]
    assert refined["costs"]["currencies"]["USD"]["total"] == "0.1234567890123456789012345678"
    multi = by_name["credits stay separate from currencies"]
    assert set(multi["costs"]["currencies"]) == {"USD", "EUR"}
    assert multi["costs"]["currencies"]["EUR"]["total"] == "2.50000000000000000001"
    assert multi["costs"]["non_monetary"]["provider_credits"]["total"] == "3.141592653589793238462643383"
    correction = by_name["downward correction replaces tokens and costs"]
    assert correction["total_tokens"] == 12 < multi["total_tokens"]
    assert correction["costs"]["currencies"]["USD"]["total"] == "1E-28"


def _baseline() -> dict[str, Any]:
    return deepcopy(
        next(step["event"] for step in FIXTURE["steps"] if step["name"] == "closed run with complete evidence")
    )


def _malformed_contents() -> list[dict[str, Any]]:
    content = _baseline()["content"]
    cases: list[dict[str, Any]] = []
    missing = object()

    def variants(path: tuple[str, ...], values: tuple[Any, ...]) -> None:
        for value in values:
            candidate = deepcopy(content)
            candidate["source_seq"] = 1000
            target = candidate
            for key in path[:-1]:
                target = target[key]
            if value is missing:
                del target[path[-1]]
            else:
                target[path[-1]] = value
            cases.append(candidate)

    variants(("scope",), (missing, None, "", "worker", "turn"))
    variants(("semantics",), (missing, None, "", "delta", "total"))
    variants(("source_seq",), (missing, None, True, False, -1, 0.5, "10", 9_007_199_254_740_992))
    variants(("usage",), (missing, None, [], "", 0, True))
    bad_counts = (missing, None, True, False, -1, 0.5, "1", 9_007_199_254_740_992)
    for name in (*COUNTS, *TOKENS):
        variants(("usage", name), bad_counts)
    variants(("usage", "known_requests"), (missing, None, [], "", 0))
    for name in TOKENS:
        variants(("usage", "known_requests", name), bad_counts)
    for path in (("usage", "complete"), ("usage", "costs", "complete")):
        variants(path, (missing, None, 0, 1, "true", []))
    variants(("usage", "costs"), (missing, None, [], "", True))
    for name in ("reported", "estimated", "missing", "partial", "missing_requests", "partial_requests"):
        variants(("usage", "costs", name), bad_counts)
    for bucket, denomination in (("currencies", "USD"), ("non_monetary", "provider_credits")):
        path = ("usage", "costs", bucket)
        variants(path, (missing, None, [], "", True, {"": {}}, {"USD": []}))
        for name in ("reported", "estimated", "total"):
            variants(
                (*path, denomination, name),
                (missing, None, True, 0, 1.5, "", "NaN", "Infinity", "-0.1", "1\n", "1.2.3", " 1"),
            )
        for name in ("requests", "reported_requests", "estimated_requests"):
            variants((*path, denomination, name), bad_counts)
        variants((*path, denomination, "token_cost_subtotal"), (None, 0, "NaN", "-1"))
    return cases


def test_malformed_snapshot_fields_are_inert_and_do_not_advance_watermark(tmp_path: Path) -> None:
    baseline = _baseline()
    expected = next(
        step["expected_usage"] for step in FIXTURE["steps"] if step["name"] == "closed run with complete evidence"
    )
    cases = _malformed_contents()
    later = deepcopy(baseline)
    later["seq"] += 2
    later["content"]["source_seq"] += 1
    expected_later = {**deepcopy(expected), "source_seq": later["content"]["source_seq"]}
    for index, content in enumerate(cases):
        malformed = {"event_type": "usage_updated", "seq": baseline["seq"] + 1, "content": content}
        assert replay([_event(malformed)]).usage is None, index
        state = replay([_event(baseline), _event(malformed)])
        assert state.usage == expected, index
        assert state.sub_agents == {}
        assert state.unknown_event_types == set()
        fold(state, _event(later))
        assert state.usage == expected_later, index
    _node(
        tmp_path,
        {"baseline": baseline, "cases": cases, "expected": expected, "later": later, "expectedLater": expected_later},
        """
for (const [index, content] of input.cases.entries()) {
  const malformed = {event_type: "usage_updated", seq: input.baseline.seq + 1, content};
  assert.equal(replay([malformed]).usage, null, `empty case ${index}`);
  const state = replay([input.baseline, malformed]);
  assert.deepEqual(state.usage, input.expected, `case ${index}`);
  assert.equal(state.subAgents.size, 0);
  assert.deepEqual([...state.unknownEventTypes], []);
  fold(state, input.later);
  assert.deepEqual(state.usage, input.expectedLater, `watermark case ${index}`);
}
""",
    )


def test_snapshot_retains_metadata_without_aliasing_payload(tmp_path: Path) -> None:
    baseline = _baseline()
    event = _event(baseline)
    state = replay([event])
    expected = deepcopy(state.usage)
    usage = event.content["usage"]
    usage["input_tokens"] = 999
    usage["known_requests"]["total_tokens"] = 999
    usage["costs"]["currencies"]["USD"]["total"] = "999"
    usage["costs"]["non_monetary"]["provider_credits"]["total"] = "999"
    usage["coverage"]["full"] = False
    usage["persistence"]["healthy"] = False
    usage["detail"]["reported_missing_reasoning"] = 999
    usage["breakdown"][0]["total_tokens"] = 999
    assert state.usage == expected
    assert state.to_dict()["usage"] == expected
    _node(
        tmp_path,
        {"event": baseline, "expected": expected},
        """
const state = replay([input.event]);
const usage = input.event.content.usage;
usage.input_tokens = 999;
usage.known_requests.total_tokens = 999;
usage.costs.currencies.USD.total = "999";
usage.costs.non_monetary.provider_credits.total = "999";
usage.coverage.full = false;
usage.persistence.healthy = false;
usage.detail.reported_missing_reasoning = 999;
usage.breakdown[0].total_tokens = 999;
assert.deepEqual(state.usage, input.expected);
assert.deepEqual(toDict(state).usage, input.expected);
""",
    )


def test_worker_snapshots_never_create_activity_or_replace_root_usage(tmp_path: Path) -> None:
    baseline = _baseline()
    cases = [{"agent_id": "worker"}, {"agent_name": "worker"}, {"dispatch_id": "worker"}, {"depth": 1}]
    for info in cases:
        worker = deepcopy(baseline)
        worker["seq"] += 1
        worker["sub_agent_info"] = info
        worker["content"]["source_seq"] = 1000
        assert replay([_event(worker)]).usage is None
        state = replay([_event(baseline), _event(worker)])
        assert state.usage == {"source_seq": baseline["content"]["source_seq"], **baseline["content"]["usage"]}
        assert state.sub_agents == {}
    _node(
        tmp_path,
        {"baseline": baseline, "cases": cases},
        """
for (const info of input.cases) {
  const worker = structuredClone(input.baseline);
  worker.seq++;
  worker.sub_agent_info = info;
  worker.content.source_seq = 1000;
  assert.equal(replay([worker]).usage, null);
  const state = replay([input.baseline, worker]);
  assert.deepEqual(state.usage, {
    ...input.baseline.content.usage, source_seq: input.baseline.content.source_seq,
  });
  assert.equal(state.subAgents.size, 0);
}
""",
    )


def test_usage_keeps_event_sequence_ordering_contract(tmp_path: Path) -> None:
    baseline = _baseline()
    for seq in (baseline["seq"] - 1, baseline["seq"]):
        duplicate = deepcopy(baseline)
        duplicate["seq"] = seq
        with pytest.raises(ProjectionError):
            replay([_event(baseline), _event(duplicate)])
    _node(
        tmp_path,
        baseline,
        """
for (const seq of [input.seq - 1, input.seq]) {
  assert.throws(() => replay([input, {...input, seq}]), /requires ascending order/);
}
""",
    )
