import asyncio
import gc
import json
from types import SimpleNamespace
from typing import Any, AsyncIterator, cast

import pytest

from kolega_code.llm.accounting import UsageObservation, publish_observation, reported_cost
from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import LLMError
from kolega_code.llm.ledger import (
    LedgerStreamAdapter,
    UsageLedger,
    helper_origin,
    llm_call_origin,
    logical_prompt_scope,
)
from kolega_code.llm.models import Message, MessageHistory
from kolega_code.llm.providers.anthropic import AnthropicStreamWrapper
from kolega_code.llm.providers.openai import OpenAIStreamWrapper
from kolega_code.llm.providers.responses_common import ResponsesStreamWrapper


class Resolver:
    def __init__(self) -> None:
        self.rate = "1"
        self.pins: list[dict[str, Any]] = []

    def pin(self, provider: str, model: str | None, context: dict[str, Any] | None = None) -> dict[str, Any]:
        result = {"provider": provider, "model": model, "context": dict(context or {}), "rate": self.rate}
        self.pins.append(result)
        return result

    def calculate(
        self, pinned: dict[str, Any], usage: Any, observation: UsageObservation | None = None
    ) -> dict[str, Any]:
        if observation is not None and observation.cost is not None:
            return observation.cost
        return {
            "source": "unavailable",
            "amount": None,
            "currency": None,
            "complete": False,
            "missing_components": ["test_unpriced"],
        }


class Observer:
    def __init__(self) -> None:
        self.begins: list[Any] = []
        self.responses: list[Any] = []
        self.failures: list[Any] = []
        self.observations: list[Any] = []

    def on_begin(self, request: Any) -> None:
        self.begins.append(request)

    def on_response(self, request: Any, message: Any) -> None:
        self.responses.append(request)

    def on_failure(self, request: Any) -> None:
        self.failures.append(request)

    def on_observation(self, request: Any) -> None:
        self.observations.append(request)


def test_failure_can_have_complete_tokens_and_reported_cost() -> None:
    ledger = UsageLedger(Resolver())
    observer = Observer()
    ledger.observer = observer
    request = ledger.begin("openrouter", "m")
    ledger.record_observation(
        request,
        UsageObservation(
            "openrouter",
            "m",
            input_tokens=0,
            output_tokens=5,
            complete=True,
            cost=reported_cost("openrouter", {"cost": "0.2"}),
        ),
    )
    ledger.record_failure(request, "cancelled")
    snap = ledger.snapshot()
    assert snap.failed == 1 and snap.responses == 0 and snap.total_tokens == 5
    assert snap.complete and snap.costs["complete"]
    assert snap.costs["currencies"]["USD"]["total"] == "0.2"
    assert len(observer.failures) == 1 and len(observer.begins) == 1 and not observer.observations
    payload = observer.failures[0].to_payload()
    assert payload["state"] == "failed" and payload["observation"]["complete"]
    json.dumps(payload, allow_nan=False)


def test_partial_unknown_counts_and_genuine_empty_are_different() -> None:
    ledger = UsageLedger(Resolver())
    empty = ledger.snapshot().to_payload()
    assert empty["input_tokens"] == 0 and empty["complete"]
    request = ledger.begin("openai", "m")
    ledger.record_observation(request, UsageObservation("openai", input_tokens=0))
    ledger.record_failure(request, "failed")
    payload = ledger.snapshot().to_payload()
    assert payload["input_tokens"] == 0 and payload["output_tokens"] is None
    assert payload["partial"] == 1 and not payload["complete"]
    assert payload["detail_complete"]["cache_read_input_tokens"] is False


def test_late_evidence_replaces_same_identity_and_notifies_only_when_changed() -> None:
    ledger = UsageLedger(Resolver())
    observer = Observer()
    ledger.observer = observer
    request = ledger.begin("openai", "m")
    ledger.record_observation(request, UsageObservation("openai", "m", input_tokens=10))
    ledger.record_failure(request, "broken")
    revision = ledger.revision
    final = UsageObservation("openai", "m", input_tokens=12, output_tokens=3, complete=True)
    ledger.record_observation(request, final)
    ledger.record_observation(request, final)
    snap = ledger.snapshot()
    assert (snap.requests, snap.failed, snap.responses, snap.total_tokens) == (1, 1, 0, 15)
    assert snap.complete and snap.revision == revision + 1
    assert len(observer.failures) == len(observer.observations) == 1
    assert observer.observations[0].request_id == request
    assert observer.observations[0].state == "failed"
    message = Message("assistant", "late")
    ledger.record_response(request, None, message)
    assert message.usage_metadata is not None
    assert message.usage_metadata["llm_call"]["llm_call_id"] == request
    assert ledger.snapshot().responses == 0


@pytest.mark.asyncio
async def test_child_task_captures_launch_prompt_and_pins_pricing_before_later_scope() -> None:
    resolver = Resolver()
    ledger = UsageLedger(resolver)
    launched = asyncio.Event()
    finish = asyncio.Event()

    async def child() -> str:
        with llm_call_origin(helper_origin("compression")):
            request = ledger.begin("openai", "m", {"service_tier": "priority"})
        launched.set()
        await finish.wait()
        ledger.record_observation(
            request, UsageObservation("openai", "m", input_tokens=1, output_tokens=2, complete=True)
        )
        ledger.record_failure(request, "cancelled after usage")
        return request

    with logical_prompt_scope("A"):
        task = asyncio.create_task(child())
        await launched.wait()
    resolver.rate = "99"
    with logical_prompt_scope("B"):
        finish.set()
        request = await task
    assert ledger.snapshot_for_prompt("A").total_tokens == 3
    assert ledger.snapshot_for_prompt("B").requests == 0
    fact = ledger.get_request(request)
    assert fact is not None and fact.prompt_id == "A"
    assert fact.origin is not None and fact.origin.helper == "compression"
    assert fact.pinned_pricing["rate"] == "1"
    assert fact.pricing_context["requested_at"] == fact.started_at
    assert fact.pricing_context["service_tier"] == "priority"
    with pytest.raises(TypeError):
        fact.pinned_pricing["rate"] = "20"


def test_transport_listener_failures_do_not_count_as_sink_failures() -> None:
    ledger = UsageLedger(Resolver())
    calls: list[int] = []

    def broken() -> None:
        raise RuntimeError("transport")

    unsubscribe = ledger.subscribe(broken)
    ledger.subscribe(lambda: calls.append(ledger.revision))
    ledger.begin("openai", "m")
    assert calls == [1] and ledger.persist_failures == 0
    unsubscribe()
    unsubscribe()
    ledger.begin("openai", "m")
    assert calls == [1, 2]


class Wrapper:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.observation = UsageObservation("openai", "m", input_tokens=7)

    async def __aenter__(self) -> Any:
        if self.stage == "entry":
            raise RuntimeError("entry")
        return self

    async def __aexit__(self, *args: Any) -> bool:
        if self.stage == "cleanup":
            raise RuntimeError("cleanup")
        return False

    async def __anext__(self) -> Any:
        if self.stage == "cancel":
            raise asyncio.CancelledError()
        raise RuntimeError("iteration")

    def get_usage_observation(self) -> UsageObservation:
        return self.observation

    async def get_final_message(self) -> Message:
        return Message("assistant", "ok", usage_observation=self.observation)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["entry", "cleanup", "iteration", "cancel"])
async def test_all_stream_failure_boundaries_preserve_observations(stage: str) -> None:
    ledger = UsageLedger(Resolver())
    request = ledger.begin("openai", "m")
    adapter = LedgerStreamAdapter(Wrapper(stage), ledger, request)
    with pytest.raises(asyncio.CancelledError if stage == "cancel" else RuntimeError):
        async with adapter:
            if stage not in ("entry", "cleanup"):
                await adapter.__anext__()
    assert ledger.snapshot().failed == 1 and ledger.snapshot().input_tokens == 7
    assert ledger.snapshot().partial == 1


@pytest.mark.asyncio
async def test_clean_abandonment_and_explicit_close_preserve_partial_evidence() -> None:
    ledger = UsageLedger(Resolver())
    request = ledger.begin("openai", "m")
    adapter = LedgerStreamAdapter(Wrapper("abandon"), ledger, request)
    async with adapter:
        pass
    await adapter.aclose()
    snapshot = ledger.snapshot()
    assert snapshot.abandoned == 1 and snapshot.open == 0 and snapshot.requests == 1
    assert snapshot.input_tokens == 7 and not snapshot.complete


async def events(*values: Any) -> AsyncIterator[Any]:
    for value in values:
        yield value


@pytest.mark.asyncio
async def test_workers_ai_partial_deltas_then_authoritative_total_replace() -> None:
    def event(inp: int, out: int, finish: str | None = None) -> Any:
        return SimpleNamespace(
            model="m",
            choices=[SimpleNamespace(delta=None, finish_reason=finish)],
            usage=SimpleNamespace(prompt_tokens=inp, completion_tokens=out, total_tokens=inp + out),
        )

    wrapper = OpenAIStreamWrapper(
        events(event(10, 0), event(0, 3), event(0, 4), event(0, 0, "stop"), event(10, 9, "stop")),
        provider_name="cloudflare_workers_ai",
        model="m",
    )
    assert (await wrapper.__anext__()).type == "ignore"
    first = wrapper.get_usage_observation()
    assert first.input_tokens == 10 and first.output_tokens == 0 and not first.complete
    for _ in range(4):
        await wrapper.__anext__()
    final = wrapper.get_usage_observation()
    assert final.input_tokens == 10 and final.output_tokens == 9 and final.total_tokens == 19
    assert final.complete


@pytest.mark.asyncio
async def test_anthropic_component_merge_and_snapshot_do_not_drain_stream() -> None:
    wrapper = AnthropicStreamWrapper(None, model="m")
    wrapper.generator = events(
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                model="resolved", usage=SimpleNamespace(input_tokens=2, output_tokens=0, cache_read_input_tokens=10)
            ),
        ),
        SimpleNamespace(
            type="message_delta", usage=SimpleNamespace(output_tokens=5), delta=SimpleNamespace(stop_reason="end_turn")
        ),
        SimpleNamespace(type="message_stop"),
    )
    await wrapper.__anext__()
    before = wrapper.get_usage_observation()
    assert before.input_tokens == 12 and before.output_tokens == 0 and not before.complete
    assert wrapper.get_usage_observation() == before
    await wrapper.__anext__()
    await wrapper.__anext__()
    final = wrapper.get_usage_observation()
    assert final.input_tokens == 12 and final.output_tokens == 5 and final.complete
    assert final.model == "resolved"


@pytest.mark.asyncio
async def test_responses_failure_keeps_charge_and_partial_core_independently() -> None:
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=0, cost={"total_cost": "0.05"}))
    wrapper = ResponsesStreamWrapper(
        events(SimpleNamespace(type="response.failed", response=response, message="failed")),
        provider_name="perplexity_agent",
        model="m",
    )
    ledger = UsageLedger(Resolver())
    adapter = LedgerStreamAdapter(wrapper, ledger, ledger.begin("perplexity_agent", "m"))
    with pytest.raises(RuntimeError):
        async with adapter:
            await adapter.__anext__()
    snapshot = ledger.snapshot()
    assert snapshot.failed == 1 and snapshot.input_tokens == 0
    assert snapshot.to_payload()["output_tokens"] is None
    assert snapshot.costs["currencies"]["USD"]["total"] == "0.05"
    assert snapshot.costs["complete"] and not snapshot.complete


def test_reported_zero_replaces_estimate_instead_of_adding_to_it() -> None:
    class Estimator(Resolver):
        def calculate(
            self, pinned: dict[str, Any], usage: Any, observation: UsageObservation | None = None
        ) -> dict[str, Any]:
            if observation is not None and observation.cost is not None:
                return observation.cost
            return {"source": "estimated", "amount": "2", "currency": "USD", "complete": True}

    ledger = UsageLedger(Estimator())
    request = ledger.begin("openrouter", "m")
    ledger.record_observation(request, UsageObservation("openrouter", "m", input_tokens=10, output_tokens=2))
    partial = ledger.snapshot()
    assert not partial.costs["complete"] and partial.costs["currencies"]["USD"]["total"] == "2"
    ledger.record_failure(request, "closed")
    ledger.record_observation(
        request,
        UsageObservation(
            "openrouter",
            "m",
            input_tokens=12,
            output_tokens=2,
            complete=True,
            cost=reported_cost("openrouter", {"cost": "0"}),
        ),
    )
    final = ledger.snapshot()
    assert final.costs["complete"] and final.costs["currencies"]["USD"]["total"] == "0"
    assert final.costs["currencies"]["USD"]["estimated"] == "0"
    assert partial.costs["currencies"]["USD"]["total"] == "2"


@pytest.mark.asyncio
async def test_setup_and_generate_failures_do_not_repeat_provider_calls() -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        def stream(self, *args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            raise RuntimeError("synchronous setup failure")

        async def generate(self, *args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            publish_observation(UsageObservation("openai", "m", input_tokens=2, output_tokens=3, complete=True))
            raise RuntimeError("message parsing failure")

    ledger = UsageLedger(Resolver())
    client = LLMClient("openai", api_key="fake-key", usage_ledger=ledger)
    provider = Provider()
    client.provider = cast(Any, provider)
    messages = MessageHistory([Message("user", "hello")])
    with pytest.raises(RuntimeError, match="synchronous setup"):
        await cast(Any, client.stream(messages, model="m"))
    with pytest.raises(LLMError):
        await client.generate(messages, model="m")
    snapshot = ledger.snapshot()
    assert provider.calls == snapshot.requests == snapshot.failed == 2
    assert snapshot.total_tokens == 5


@pytest.mark.asyncio
async def test_exhausted_unfinalized_adapter_is_abandoned_when_released() -> None:
    class Exhausted(Wrapper):
        async def __anext__(self) -> Any:
            raise StopAsyncIteration

    ledger = UsageLedger(Resolver())
    adapter = LedgerStreamAdapter(Exhausted("exhausted"), ledger, ledger.begin("openai", "m"))
    async with adapter:
        with pytest.raises(StopAsyncIteration):
            await adapter.__anext__()
    assert ledger.snapshot().open == 1
    del adapter
    gc.collect()
    assert ledger.snapshot().abandoned == 1 and ledger.snapshot().input_tokens == 7


@pytest.mark.asyncio
async def test_successful_chunk_publishes_consumed_snapshot_while_request_remains_open() -> None:
    class Partial:
        calls = 0

        async def __anext__(self) -> str:
            self.calls += 1
            return "chunk"

        def get_usage_observation(self) -> UsageObservation:
            return UsageObservation("openai", "m", input_tokens=self.calls)

    ledger = UsageLedger(Resolver())
    inner = Partial()
    adapter = LedgerStreamAdapter(inner, ledger, ledger.begin("openai", "m"))
    assert await adapter.__anext__() == "chunk"
    snapshot = ledger.snapshot()
    assert snapshot.open == 1 and snapshot.input_tokens == inner.calls == 1
    assert snapshot.to_payload()["output_tokens"] is None
    assert not snapshot.complete


def test_checkpoint_breakdown_preserves_partial_evidence_metadata() -> None:
    ledger = UsageLedger(Resolver())
    first = ledger.begin("openai", "m")
    ledger.record_observation(first, UsageObservation("openai", "m", input_tokens=1, output_tokens=2, complete=True))
    ledger.record_response(first, None)
    mark = ledger.snapshot()
    assert mark.reported == 1 and mark.complete
    second = ledger.begin("openai", "m")
    ledger.record_observation(second, UsageObservation("openai", "m", input_tokens=0))
    ledger.record_abandonment(second)
    delta = ledger.snapshot().since(mark)
    row = delta.breakdown[0]
    assert delta.abandoned == row.abandoned == 1
    assert delta.partial == row.partial == 1
    assert row.known_requests["input_tokens"] == 1 and row.known_requests["output_tokens"] == 0
    assert not row.complete and not delta.complete
    assert delta.to_payload()["breakdown"][0]["output_tokens"] is None


@pytest.mark.asyncio
async def test_early_openai_final_message_does_not_promote_interim_usage_to_complete() -> None:
    ledger = UsageLedger(Resolver())
    wrapper = OpenAIStreamWrapper(None, provider_name="cloudflare_workers_ai", model="m")
    wrapper.usage_data = {"prompt_tokens": 2, "completion_tokens": 1}
    adapter = LedgerStreamAdapter(wrapper, ledger, ledger.begin("cloudflare_workers_ai", "m"))
    message = await adapter.get_final_message()
    assert message.usage is not None and message.usage.reported
    assert message.usage_observation is not None and not message.usage_observation.complete
    snapshot = ledger.snapshot()
    assert snapshot.responses == 1 and snapshot.partial == 1
    assert snapshot.total_tokens == 3 and not snapshot.complete
