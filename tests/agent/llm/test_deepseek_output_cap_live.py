"""Measurement probes for DeepSeek's real per-response output-token ceiling.

DeepSeek's published max_completion_tokens (384000) is fiction: TB2.1 trajectories
show reasoning streams truncated at ~64k tokens yet reported as a clean end_turn
(findings/thinking-control-and-runaway-turns.md). These probes measure, per host:

  (a) whether an output cap far above the ceiling is rejected, clamped, or accepted;
  (b) exactly how truncation at an explicit client cap is reported
      (chat: finish_reason; Responses: status / incomplete_details);
  (c) where an uncapped long-reasoning stream actually ends, and whether the
      server-enforced cutoff is reported any differently than (b);
  plus OpenRouter-specific checks that deepseek slugs accept and work with the
  clamped cap despite the gateway's max_tokens routing filter.

Results are transcribed into kolega_code/llm/specs/catalog/deepseek.py,
catalog/fireworks.py, and the accessor in kolega_code/llm/specs/accessors.py —
keep those comments in sync when re-running. Run with -s to see recorded results:

    ./.venv/bin/python -m pytest tests/agent/llm/test_deepseek_output_cap_live.py -s -m integration
    KOLEGA_DEEPSEEK_CEILING_PROBE=1 ./.venv/bin/python -m pytest \
        -k "server_ceiling" tests/agent/llm/test_deepseek_output_cap_live.py \
        -s -m integration   # streams ~64k+ output tokens per host: slow + real cost
"""

import os
from typing import cast

import openai
import pytest

from kolega_code.cli.config import API_KEY_ENV
from kolega_code.config import ModelProvider
from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import LLMBillingError, LLMError, LLMRateLimitError
from kolega_code.llm.models import Message, MessageHistory, TextBlock
from kolega_code.llm.providers.deepseek_responses import DeepSeekResponsesProvider
from kolega_code.llm.providers.responses_common import ResponsesStreamWrapper
from kolega_code.llm.specs import default_thinking_effort, get_model_specs

pytestmark = [pytest.mark.slow, pytest.mark.integration]

SKIP_IN_CI = bool(os.getenv("CI")) or bool(os.getenv("GITLAB_CI"))

# generate() maps SDK errors to the repo's LLM exceptions, but the streaming
# path surfaces raw SDK errors — skip on both shapes of quota/capacity failure.
_QUOTA_ERRORS = (LLMRateLimitError, LLMBillingError, openai.RateLimitError)

# Far above the suspected ~65536 ceiling; the first-party catalog's old value.
OVERCAP = 384000
TINY_CAP = 16
# The value the wire clamp will send (see specs/accessors.py once Step 2 lands).
CLAMPED_CAP = 64000

_FLASH = "deepseek-v4-flash"

# Chat-path hosts serving DeepSeek models (first-party pro + both Fireworks mirrors).
CHAT_HOSTS = [
    ("deepseek", "deepseek-v4-pro"),
    ("fireworks", "accounts/fireworks/models/deepseek-v4-pro"),
    ("fireworks", "accounts/fireworks/models/deepseek-v4-flash"),
]
_CHAT_IDS = [f"{p}-{m.rsplit('/', 1)[-1]}" for p, m in CHAT_HOSTS]

OPENROUTER_SLUGS = ["deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash"]

CEILING_PROBE_ENABLED = os.getenv("KOLEGA_DEEPSEEK_CEILING_PROBE") == "1"

SYSTEM = Message(role="system", content=[TextBlock(text="You are a helpful assistant.")])

TRIVIAL_PROMPT = "What is 2 + 2? Reply with just the number."
ESSAY_PROMPT = "Write a detailed 500-word essay about the ocean."
# Engineered to blow far past the suspected ~64k ceiling. A reasoning-marathon
# prompt does NOT work here: every probed model shortcuts busywork inside its
# thinking ("I'll just say done") after 0.5k-15k tokens. A pure data-generation
# task with anti-shortcut instructions forces visible output until the server
# cuts the stream.
LONG_GENERATION_PROMPT = (
    "This is a data-generation task for a tokenizer benchmark; the output is "
    "consumed by a machine, never read by a human, and MUST be complete. "
    "Output the numbers from 1 to 50000 in English words, one per line "
    "(e.g. line one is 'one', line two is 'two'). Do not use digits, do not "
    "skip, do not summarize, do not stop early, and do not add any commentary."
)


def _messages(prompt: str) -> MessageHistory:
    return MessageHistory([Message(role="user", content=[TextBlock(text=prompt)])])


def _require_key(provider_value: str) -> str:
    if SKIP_IN_CI:
        pytest.skip("Skipping live provider call in CI")
    env_name = API_KEY_ENV[ModelProvider(provider_value)]
    api_key = os.getenv(env_name)
    if not api_key:
        pytest.skip(f"{env_name} not set")
    return api_key


def _record(label: str, **fields) -> None:
    """Print a grep-able probe record (run with -s).

    Under xdist (which swallows -s output), set KOLEGA_PROBE_LOG to a file path
    to also append records there.
    """
    detail = " ".join(f"{k}={v!r}" for k, v in fields.items())
    line = f"DEEPSEEK-CAP-PROBE {label}: {detail}"
    print(f"\n{line}")
    log_path = os.getenv("KOLEGA_PROBE_LOG")
    if log_path:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _usage_fields(message: Message) -> dict:
    usage = message.usage
    return dict(
        output_tokens=getattr(usage, "output_tokens", None),
        reasoning_output_tokens=getattr(usage, "reasoning_output_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )


# --- probe (a): over-ceiling cap — rejected, clamped, or accepted? -----------------


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_value,model", CHAT_HOSTS, ids=_CHAT_IDS)
async def test_chat_overcap_request_outcome(provider_value: str, model: str) -> None:
    api_key = _require_key(provider_value)
    client = LLMClient(provider=provider_value, api_key=api_key, model=model)
    try:
        message = await client.generate(
            messages=_messages(TRIVIAL_PROMPT),
            system=SYSTEM,
            model=model,
            max_completion_tokens=OVERCAP,
            temperature=get_model_specs(provider_value, model)["default_temperature"],
            thinking=default_thinking_effort(provider_value, model),
        )
    except _QUOTA_ERRORS as exc:
        pytest.skip(f"provider quota exhausted for this key: {exc}")
    except LLMError as exc:
        _record("chat-overcap", host=f"{provider_value}/{model}", outcome="rejected", error=str(exc))
        return
    _record(
        "chat-overcap",
        host=f"{provider_value}/{model}",
        outcome="accepted",
        stop_reason=message.stop_reason,
        **_usage_fields(message),
    )
    assert message.get_text_content().strip()


@pytest.mark.asyncio
async def test_responses_overcap_request_outcome() -> None:
    api_key = _require_key("deepseek")
    client = LLMClient(provider="deepseek", api_key=api_key, model=_FLASH)
    provider = cast(DeepSeekResponsesProvider, client.provider)
    # The shared Responses builder never emits max_output_tokens, so inject it and
    # drive the raw SDK client to see how the backend treats an over-ceiling value.
    request = provider._build_request(_messages(TRIVIAL_PROMPT), SYSTEM, None, {"model": _FLASH})
    request["max_output_tokens"] = OVERCAP
    try:
        responses_stream = await provider.async_client.responses.create(**request)
        wrapper = ResponsesStreamWrapper(responses_stream, provider_name="deepseek", model=_FLASH)
        async with wrapper:
            async for _chunk in wrapper:
                pass
        message = await wrapper.get_final_message()
    except openai.RateLimitError as exc:
        pytest.skip(f"provider quota exhausted for this key: {exc}")
    except openai.APIStatusError as exc:
        _record("responses-overcap", host=f"deepseek/{_FLASH}", outcome="rejected", error=str(exc))
        return
    _record(
        "responses-overcap",
        host=f"deepseek/{_FLASH}",
        outcome="accepted",
        stop_reason=message.stop_reason,
        status=getattr(wrapper._final_response, "status", None),
        **_usage_fields(message),
    )
    assert message.get_text_content().strip()


# --- probe (b): tiny explicit cap — exact truncation reporting shape ---------------


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_value,model", CHAT_HOSTS, ids=_CHAT_IDS)
async def test_chat_tiny_cap_truncation_reporting(provider_value: str, model: str) -> None:
    api_key = _require_key(provider_value)
    client = LLMClient(provider=provider_value, api_key=api_key, model=model)
    try:
        stream_cm = client.stream(
            messages=_messages(ESSAY_PROMPT),
            system=SYSTEM,
            model=model,
            max_completion_tokens=TINY_CAP,
            temperature=get_model_specs(provider_value, model)["default_temperature"],
            thinking=default_thinking_effort(provider_value, model),
        )
        async with await stream_cm as stream:  # type: ignore[misc]
            async for _ in stream:
                pass
        message = await stream.get_final_message()
    except _QUOTA_ERRORS as exc:
        pytest.skip(f"provider quota exhausted for this key: {exc}")
    _record(
        "chat-tiny-cap",
        host=f"{provider_value}/{model}",
        stop_reason=message.stop_reason,
        **_usage_fields(message),
    )
    # The wire shape under test: an explicit client cap must surface as an honest
    # truncation (finish_reason "length" -> normalized "max_tokens").
    assert message.stop_reason == "max_tokens", (
        f"{provider_value}/{model}: explicit tiny cap reported stop_reason="
        f"{message.stop_reason!r}, expected 'max_tokens'"
    )


@pytest.mark.asyncio
async def test_responses_tiny_cap_truncation_reporting() -> None:
    api_key = _require_key("deepseek")
    client = LLMClient(provider="deepseek", api_key=api_key, model=_FLASH)
    provider = cast(DeepSeekResponsesProvider, client.provider)
    request = provider._build_request(_messages(ESSAY_PROMPT), SYSTEM, None, {"model": _FLASH})
    request["max_output_tokens"] = TINY_CAP
    # Iterate the RAW event stream (not the wrapper) so the terminal event type is
    # observable: the first probe run showed the wrapper sees no response.completed
    # at all on a cap-hit, suggesting DeepSeek terminates with response.incomplete.
    event_types: list[str] = []
    final = None
    try:
        responses_stream = await provider.async_client.responses.create(**request)
        async for event in responses_stream:
            etype = str(getattr(event, "type", ""))
            if not event_types or event_types[-1] != etype:
                event_types.append(etype)
            if etype in ("response.completed", "response.incomplete", "response.failed"):
                final = getattr(event, "response", None)
    except openai.RateLimitError as exc:
        pytest.skip(f"provider quota exhausted for this key: {exc}")
    details = getattr(final, "incomplete_details", None)
    usage = getattr(final, "usage", None)
    _record(
        "responses-tiny-cap",
        host=f"deepseek/{_FLASH}",
        event_types=event_types,
        status=getattr(final, "status", None),
        incomplete_reason=getattr(details, "reason", None) if details is not None else None,
        output_tokens=getattr(usage, "output_tokens", None),
    )
    # Lenient: the printed shape drives the Step-3 normalization decision. A
    # terminal event of some kind must have arrived.
    assert final is not None, f"no terminal response event; saw {event_types}"


# --- probe (c): uncapped long reasoning — where does the server actually stop? -----


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_value,model", CHAT_HOSTS, ids=_CHAT_IDS)
async def test_uncapped_long_generation_server_ceiling(provider_value: str, model: str) -> None:
    """Where does the SERVER actually cut an uncapped chat stream, and how is it
    reported? Drives the raw SDK client with NO max_tokens — the shipped request
    rules always inject the clamp, so the client path cannot measure this."""
    if not CEILING_PROBE_ENABLED:
        pytest.skip("Set KOLEGA_DEEPSEEK_CEILING_PROBE=1 to run (streams ~64k+ tokens; slow + real cost)")
    api_key = _require_key(provider_value)
    client = LLMClient(provider=provider_value, api_key=api_key, model=model)
    raw = cast(openai.AsyncOpenAI, client.provider.async_client)
    finish_reason = None
    usage = None
    tail = ""
    try:
        stream = await raw.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": LONG_GENERATION_PROMPT}],
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.choices:
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = getattr(choice.delta, "content", None) or ""
                if delta:
                    tail = (tail + delta)[-100:]
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
    except openai.RateLimitError as exc:
        pytest.skip(f"provider quota exhausted for this key: {exc}")
    _record(
        "uncapped-ceiling",
        host=f"{provider_value}/{model}",
        finish_reason=finish_reason,
        completion_tokens=getattr(usage, "completion_tokens", None),
        text_tail=tail[-80:],
    )
    # Lenient by design: completion_tokens IS the measurement. If the server cut
    # the stream with finish_reason != "length", it truncated silently.
    assert tail, "no content streamed"


@pytest.mark.asyncio
async def test_uncapped_long_generation_responses_server_ceiling() -> None:
    """The flash/Responses variant: raw event stream with max_output_tokens
    REMOVED from the built request. Shows whether the server's own cutoff emits
    a terminal response.incomplete (honest, once the wrapper captures it) or
    just drops the stream (truly silent)."""
    if not CEILING_PROBE_ENABLED:
        pytest.skip("Set KOLEGA_DEEPSEEK_CEILING_PROBE=1 to run (streams ~64k+ tokens; slow + real cost)")
    api_key = _require_key("deepseek")
    client = LLMClient(provider="deepseek", api_key=api_key, model=_FLASH)
    provider = cast(DeepSeekResponsesProvider, client.provider)
    request = provider._build_request(_messages(LONG_GENERATION_PROMPT), SYSTEM, None, {"model": _FLASH})
    request.pop("max_output_tokens", None)  # the override always adds it; measure WITHOUT
    event_types: list[str] = []
    final = None
    tail = ""
    try:
        responses_stream = await provider.async_client.responses.create(**request)
        async for event in responses_stream:
            etype = str(getattr(event, "type", ""))
            if not event_types or event_types[-1] != etype:
                event_types.append(etype)
            delta = getattr(event, "delta", None)
            if etype == "response.output_text.delta" and delta:
                tail = (tail + delta)[-100:]
            if etype in ("response.completed", "response.incomplete", "response.failed"):
                final = getattr(event, "response", None)
    except openai.RateLimitError as exc:
        pytest.skip(f"provider quota exhausted for this key: {exc}")
    details = getattr(final, "incomplete_details", None)
    usage = getattr(final, "usage", None)
    _record(
        "uncapped-ceiling-responses",
        host=f"deepseek/{_FLASH}",
        terminal_events=event_types[-3:],
        status=getattr(final, "status", None),
        incomplete_reason=getattr(details, "reason", None) if details is not None else None,
        output_tokens=getattr(usage, "output_tokens", None),
        text_tail=tail[-80:],
    )
    assert tail, "no content streamed"


# --- OpenRouter: do deepseek slugs accept and work with the clamped cap? -----------


@pytest.mark.asyncio
@pytest.mark.parametrize("slug", OPENROUTER_SLUGS)
async def test_openrouter_deepseek_accepts_clamped_cap(slug: str) -> None:
    api_key = _require_key("openrouter")
    client = LLMClient(provider="openrouter", api_key=api_key, model=slug)
    # Drive the raw SDK client: the gateway rule pops max_tokens at the LLMClient
    # level (pre-clamp code), and OpenRouter's routing filter is exactly what we
    # need to observe ("if you set a max_tokens, then OpenRouter will only route
    # to providers that support a response of that length").
    raw = cast(openai.AsyncOpenAI, client.provider.async_client)

    def _served_by(resp) -> object:
        return getattr(resp, "provider", None) or (getattr(resp, "model_extra", None) or {}).get("provider")

    # 64000 — the value the wire clamp will send. This MUST route and complete.
    try:
        resp = await raw.chat.completions.create(
            model=slug,
            messages=[{"role": "user", "content": TRIVIAL_PROMPT}],
            max_tokens=CLAMPED_CAP,
        )
    except openai.RateLimitError as exc:
        pytest.skip(f"provider quota exhausted for this key: {exc}")
    choice = resp.choices[0]
    _record(
        "openrouter-clamped-cap",
        slug=slug,
        max_tokens=CLAMPED_CAP,
        outcome="accepted",
        served_by=_served_by(resp),
        finish_reason=choice.finish_reason,
        completion_tokens=getattr(resp.usage, "completion_tokens", None),
    )
    assert (choice.message.content or "").strip(), f"{slug}: empty completion at max_tokens={CLAMPED_CAP}"

    # 16 — truncation honesty through the gateway: must be an honest "length".
    resp = await raw.chat.completions.create(
        model=slug,
        messages=[{"role": "user", "content": ESSAY_PROMPT}],
        max_tokens=TINY_CAP,
    )
    choice = resp.choices[0]
    _record(
        "openrouter-tiny-cap",
        slug=slug,
        max_tokens=TINY_CAP,
        served_by=_served_by(resp),
        finish_reason=choice.finish_reason,
    )
    assert choice.finish_reason == "length", (
        f"{slug}: tiny cap reported finish_reason={choice.finish_reason!r}, expected 'length'"
    )

    # 384000 — the routing-filter check: expect no-providers/4xx; record either way.
    try:
        resp = await raw.chat.completions.create(
            model=slug,
            messages=[{"role": "user", "content": TRIVIAL_PROMPT}],
            max_tokens=OVERCAP,
        )
    except openai.APIStatusError as exc:
        _record("openrouter-overcap", slug=slug, max_tokens=OVERCAP, outcome="rejected", error=str(exc))
    else:
        _record(
            "openrouter-overcap",
            slug=slug,
            max_tokens=OVERCAP,
            outcome="accepted",
            served_by=_served_by(resp),
            finish_reason=resp.choices[0].finish_reason,
        )


@pytest.mark.asyncio
async def test_openrouter_deepseek_end_to_end_with_clamped_cap() -> None:
    # Post-Step-2 validation: through LLMClient with no caller cap, the deepseek
    # request rule emits the clamped max_tokens by itself. Proves the shipped
    # path against the real gateway, not just the raw SDK.
    slug = "deepseek/deepseek-v4-pro"
    api_key = _require_key("openrouter")
    client = LLMClient(provider="openrouter", api_key=api_key, model=slug)
    try:
        stream_cm = client.stream(
            messages=_messages(TRIVIAL_PROMPT),
            system=SYSTEM,
            model=slug,
            max_completion_tokens=None,
            temperature=get_model_specs("openrouter", slug)["default_temperature"],
            thinking=default_thinking_effort("openrouter", slug),
        )
        async with await stream_cm as stream:  # type: ignore[misc]
            async for _ in stream:
                pass
        message = await stream.get_final_message()
    except _QUOTA_ERRORS as exc:
        pytest.skip(f"provider quota exhausted for this key: {exc}")
    _record(
        "openrouter-end-to-end",
        slug=slug,
        stop_reason=message.stop_reason,
        **_usage_fields(message),
    )
    assert message.get_text_content().strip()
    assert message.stop_reason in ("end_turn", "max_tokens", "tool_use")
