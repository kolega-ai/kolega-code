"""Sub-agent streamed steps must accumulate O(n), not O(n^2).

`_accumulate_sub_agent_stream` used to do `step.content += text` on every streamed
delta. Because `step.content` is an instance attribute (no CPython in-place concat
optimization), that is O(n^2) over the stream — on DeepSeek-class sub-agents emitting
long reasoning it grew to seconds of event-loop CPU and froze scrolling while
sub-agents ran. The fix mirrors the main transcript's `_apply_stream_chunk`: deltas
land in `stream_parts` (O(1)) and are folded once via `materialize()`.

These tests pin:
  1. correctness - the materialized step equals the naive concatenation of all deltas,
                   and content is deferred (stays empty) until materialize.
  2. scaling     - a large stream materializes in well under a generous wall-clock bound
                   (it was seconds of O(n^2) before; milliseconds after).
"""

import time

import pytest

from kolega_code.cli.tui.state import ConversationEntry, SubAgentActivity
from kolega_code.cli.tui.transcript import TranscriptRenderingMixin
from kolega_code.events import AgentEvent


@pytest.fixture
def mixin() -> TranscriptRenderingMixin:
    # _accumulate_sub_agent_stream uses only the passed `activity`, no instance state.
    return TranscriptRenderingMixin.__new__(TranscriptRenderingMixin)


def _activity() -> SubAgentActivity:
    return SubAgentActivity(
        agent_id="agent-1",
        agent_name="sub",
        task="task",
        index=1,
        entry=ConversationEntry(kind="sub_agent", content=""),
    )


def _feed(mixin, activity, kind, uuid, text, *, streaming=True):
    event = AgentEvent(event_type="chat_message", sender="sub", uuid=uuid, is_streaming=streaming, content={})
    mixin._accumulate_sub_agent_stream(activity, kind, event, text)


def test_materialized_step_equals_concatenation(mixin):
    activity = _activity()
    deltas = [f"reasoning token {i} " for i in range(2000)]
    for d in deltas:
        _feed(mixin, activity, "thinking", "u1", d, streaming=True)

    step = activity.stream_steps["u1"]
    # Deferred: nothing is folded into content until materialize().
    assert step.content == ""
    assert step.stream_parts  # deltas are buffered as parts

    _feed(mixin, activity, "thinking", "u1", "", streaming=False)  # completion
    assert step.complete is True
    assert step.content == "".join(deltas)  # materialized on completion
    assert step.stream_parts == []


def test_thinking_and_response_stay_separate_steps(mixin):
    activity = _activity()
    _feed(mixin, activity, "thinking", "u1", "thinking part ", streaming=True)
    _feed(mixin, activity, "assistant", "u2", "answer part ", streaming=True)
    _feed(mixin, activity, "thinking", "u1", "more thinking", streaming=False)
    _feed(mixin, activity, "assistant", "u2", "more answer", streaming=False)

    assert activity.stream_steps["u1"].content == "thinking part more thinking"
    assert activity.stream_steps["u2"].content == "answer part more answer"
    assert activity.stream_steps["u1"].kind == "thinking"
    assert activity.stream_steps["u2"].kind == "assistant"


def test_large_stream_is_linear_not_quadratic(mixin):
    activity = _activity()
    # ~1 MB of reasoning in tiny deltas: O(n^2) `+=` would be seconds; O(n) is ms.
    delta = "x" * 18
    n = (1024 * 1024) // len(delta)

    start = time.perf_counter()
    for _ in range(n):
        _feed(mixin, activity, "thinking", "u1", delta, streaming=True)
    _feed(mixin, activity, "thinking", "u1", "", streaming=False)
    elapsed = time.perf_counter() - start

    step = activity.stream_steps["u1"]
    assert len(step.content) == n * len(delta)
    # Generous bound: the O(n^2) version took multiple seconds at this size.
    assert elapsed < 1.0, f"accumulation took {elapsed:.2f}s — likely regressed to O(n^2)"
