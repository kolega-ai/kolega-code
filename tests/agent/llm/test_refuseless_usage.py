from contextlib import nullcontext
from functools import partial
import inspect
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
from openai import AsyncOpenAI
import pytest

from kolega_code.llm.instrumented_client import InstrumentedLLMClient, get_output_tokens
from kolega_code.llm.models import Message, MessageHistory, TextBlock
from kolega_code.llm.providers import openai as transport
from kolega_code.llm.usage import OPENAI_USAGE_PROVIDERS, normalize_usage, usage_token_fields

pytestmark = pytest.mark.usefixtures("isolated_cli_env")

FAKE_KEY = "abl_refuseless_hermetic_test_not_a_real_key"
MODEL = "glm-5.3-abliterated"
CORE_USAGE = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}


def test_refuseless_has_openai_usage_family_and_output_count() -> None:
    assert "refuseless" in OPENAI_USAGE_PROVIDERS
    assert usage_token_fields("refuseless") == ("prompt_tokens", "completion_tokens")
    assert get_output_tokens({"provider": "refuseless", **CORE_USAGE}) == 20


def test_refuseless_normalization_keeps_cache_and_reasoning_as_subsets() -> None:
    usage = normalize_usage(
        {
            "prompt_tokens": 95,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cache_read_input_tokens": 40,
            "cache_write_input_tokens": 5,
            "reasoning_output_tokens": 7,
        },
        "refuseless",
        MODEL,
    )
    assert usage.reported
    assert usage.provider == "refuseless"
    assert usage.model == MODEL
    assert usage.input_tokens == 100
    assert usage.output_tokens == 20
    assert usage.total_tokens == 120
    assert usage.cache_read_input_tokens == 40
    assert usage.cache_write_input_tokens == 5
    assert usage.reasoning_output_tokens == 7


def test_refuseless_missing_details_are_not_fabricated() -> None:
    usage = normalize_usage(CORE_USAGE, "refuseless", MODEL)
    assert usage.reported
    assert usage.total_tokens == 120
    assert usage.cache_read_input_tokens is None
    assert usage.cache_write_input_tokens is None
    assert usage.reasoning_output_tokens is None
    absent = normalize_usage(None, "refuseless", MODEL)
    assert not absent.reported
    assert absent.input_tokens is None
    assert absent.output_tokens is None
    assert absent.total_tokens is None


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "stream"])
@pytest.mark.parametrize("detail_mode", ["subsets", "absent", "zero", "no_usage"])
async def test_refuseless_sdk_usage_and_both_langfuse_paths(
    monkeypatch: pytest.MonkeyPatch, method: str, detail_mode: str
) -> None:
    usage: dict[str, Any] | None = dict(CORE_USAGE)
    expected: dict[str, Any] | None = {"input": 100, "output": 20, "total": 120}
    if detail_mode in ("subsets", "zero"):
        cached, written, reasoning = (40, 5, 7) if detail_mode == "subsets" else (0, 0, 0)
        usage.update(
            prompt_tokens_details={"cached_tokens": cached, "cache_write_tokens": written},
            completion_tokens_details={"reasoning_tokens": reasoning},
        )
        expected.update(
            cache_read_input_tokens=cached,
            cache_creation_input_tokens=written,
            reasoning_output_tokens=reasoning,
        )
    elif detail_mode == "no_usage":
        usage = expected = None

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        response = {
            "id": "chatcmpl-usage-test",
            "object": "chat.completion",
            "created": 0,
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok", "reasoning_content": "checked"},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }
        if not body.get("stream"):
            return httpx.Response(200, json=response)
        base = {**response, "object": "chat.completion.chunk"}
        events = [
            {
                **base,
                "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}],
                "usage": None,
            },
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": None},
            {**base, "choices": [], "usage": usage},
        ]
        sse = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse)

    langfuse, trace, generation = MagicMock(), MagicMock(), MagicMock()
    langfuse.start_as_current_observation.return_value.__enter__.return_value = trace
    trace.start_observation.return_value = generation
    recorder = AsyncMock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        monkeypatch.setattr(transport, "AsyncOpenAI", partial(AsyncOpenAI, http_client=http_client))
        client = InstrumentedLLMClient(
            "refuseless", FAKE_KEY, model=MODEL, langfuse_client=langfuse, usage_recorder=recorder
        )
        monkeypatch.setattr(client, "_trace_attributes", lambda _: nullcontext())
        monkeypatch.delenv("DISABLE_USAGE_RECORDING", raising=False)
        messages = MessageHistory([Message("user", [TextBlock("hello")])])
        if method == "generate":
            message = await client.generate(messages, model=MODEL)
        else:
            stream: Any = client.stream(messages, model=MODEL)
            assert inspect.isawaitable(stream)
            stream = await stream
            async with stream:
                async for _ in stream:
                    pass
                message = await stream.get_final_message()

    assert message.get_text_content() == "ok"
    assert message.usage is not None
    assert message.usage.provider == "refuseless"
    assert message.usage.model == MODEL
    assert message.usage.reported is (detail_mode != "no_usage")
    update = generation.update.call_args.kwargs
    assert update["level"] == "DEFAULT"
    assert update["status_message"] == "Success"
    assert update.get("usage_details") == expected
    generation.end.assert_called_once()
    trace.end.assert_called_once()
    recorder.record_usage.assert_awaited_once()
    recorded = recorder.record_usage.await_args.args[0]
    assert recorded["provider"] == "refuseless"
    assert recorded["success"]
    assert recorded["metadata"]["raw_usage"] == message.usage_metadata
    if expected is not None:
        assert (message.usage.input_tokens, message.usage.output_tokens, message.usage.total_tokens) == (100, 20, 120)
    if detail_mode in ("subsets", "zero"):
        assert expected is not None
        assert usage is not None
        assert message.usage.cache_read_input_tokens == expected["cache_read_input_tokens"]
        assert message.usage.cache_write_input_tokens == expected["cache_creation_input_tokens"]
        assert message.usage.reasoning_output_tokens == expected["reasoning_output_tokens"]
        assert message.usage_metadata["prompt_tokens_details"] == usage["prompt_tokens_details"]
        assert message.usage_metadata["completion_tokens_details"] == usage["completion_tokens_details"]
    else:
        assert message.usage.cache_read_input_tokens is None
        assert message.usage.cache_write_input_tokens is None
        assert message.usage.reasoning_output_tokens is None
