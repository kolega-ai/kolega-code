from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from functools import partial
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import httpx
from openai import APIStatusError, AsyncOpenAI
import pytest

from kolega_code.llm.exceptions import LLMNotFoundError
from kolega_code.llm.providers import openai as transport
from tests.agent.llm import test_refuseless_live as live

pytestmark = pytest.mark.usefixtures("isolated_cli_env")

FAKE_KEY = "abl_refuseless_live_contract_not_a_real_key"
MODELS = (
    "glm-5.3-abliterated",
    "glm-5.3-flash-abliterated",
    "deepseek-v4.1-flash-abliterated",
    "deepseek-v4.1-flash-fast",
    "minimax-m3.1-flash",
)


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None) -> None:
    monkeypatch.setenv("KOLEGA_TEST_LIVE_API", "0")
    monkeypatch.setattr(live, "dotenv_values", Mock(side_effect=AssertionError("Real dotenv reads are forbidden")))
    monkeypatch.setattr(
        httpx.HTTPTransport, "handle_request", Mock(side_effect=AssertionError("Real network is forbidden"))
    )
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", Mock(side_effect=AssertionError("Real network is forbidden"))
    )


class _StreamBody(httpx.AsyncByteStream):
    def __init__(self, model: str) -> None:
        self.model = model

    async def __aiter__(self) -> AsyncIterator[bytes]:
        chunk = {
            "id": "chatcmpl-offline",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": self.model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }
        yield f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()


def _success(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if body.get("stream"):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_StreamBody(body["model"]))
    message: dict[str, Any] = {"role": "assistant", "content": "OK"}
    if body.get("tool_choice"):
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_offline_add",
                    "type": "function",
                    "function": {"name": "add_numbers", "arguments": '{"a":37,"b":58}'},
                }
            ],
        }
    elif body["messages"][-1]["role"] == "tool":
        message["content"] = "95"
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-offline",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [
                {"index": 0, "message": message, "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        },
    )


async def _acceptance(
    monkeypatch: pytest.MonkeyPatch, model: str, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
    monkeypatch.setenv("KOLEGA_TEST_LIVE_API", "1")
    monkeypatch.setenv("REFUSELESS_API_KEY", FAKE_KEY)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        monkeypatch.setattr(transport, "AsyncOpenAI", partial(AsyncOpenAI, http_client=http_client))
        factory = live.client_factory.__wrapped__()
        create = await anext(factory)
        try:
            await live.test_live_text_stream_and_tool_round_trip(model, create)
        finally:
            await factory.aclose()
        assert http_client.is_closed


def test_live_markers_and_literal_model_lineup() -> None:
    assert {mark.name for mark in live.pytestmark} == {"integration", "live_api"}
    assert live.MODELS == MODELS


@pytest.mark.parametrize("opt_in", [None, "", "0", "true", "yes"])
def test_live_requires_exact_explicit_opt_in(monkeypatch: pytest.MonkeyPatch, opt_in: str | None) -> None:
    monkeypatch.setenv("REFUSELESS_API_KEY", FAKE_KEY)
    if opt_in is None:
        monkeypatch.delenv("KOLEGA_TEST_LIVE_API", raising=False)
    else:
        monkeypatch.setenv("KOLEGA_TEST_LIVE_API", opt_in)
    with pytest.raises(pytest.skip.Exception, match="KOLEGA_TEST_LIVE_API=1"):
        live._api_key()


@pytest.mark.parametrize("dotenv_key", [None, ""])
def test_explicit_live_missing_key_fails(monkeypatch: pytest.MonkeyPatch, dotenv_key: str | None) -> None:
    monkeypatch.setenv("KOLEGA_TEST_LIVE_API", "1")
    dotenv = Mock(return_value={"REFUSELESS_API_KEY": dotenv_key})
    monkeypatch.setattr(live, "dotenv_values", dotenv)
    with pytest.raises(pytest.fail.Exception, match="requires REFUSELESS_API_KEY"):
        live._api_key()
    dotenv.assert_called_once_with(Path(live.__file__).resolve().parents[3] / ".env")


def test_explicit_live_uses_environment_key_without_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KOLEGA_TEST_LIVE_API", "1")
    monkeypatch.setenv("REFUSELESS_API_KEY", FAKE_KEY)
    assert live._api_key() == FAKE_KEY


def test_explicit_live_accepts_mocked_dotenv_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KOLEGA_TEST_LIVE_API", "1")
    monkeypatch.setattr(live, "dotenv_values", Mock(return_value={"REFUSELESS_API_KEY": FAKE_KEY}))
    assert live._api_key() == FAKE_KEY


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
async def test_available_models_run_entire_acceptance(monkeypatch: pytest.MonkeyPatch, model: str) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return _success(request)

    await _acceptance(monkeypatch, model, handler)
    assert len(requests) == 4
    assert all(request["model"] == model for request in requests)
    assert requests[1]["stream"] is True
    assert requests[1]["stream_options"]["include_usage"] is True
    assert requests[2]["tool_choice"] == {"type": "function", "function": {"name": "add_numbers"}}
    assert requests[3]["messages"][-2]["tool_calls"][0]["id"] == "call_offline_add"
    assert requests[3]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_offline_add",
        "content": "95",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("failure_at", [1, 2, 3, 4])
async def test_unavailable_models_fail_acceptance(monkeypatch: pytest.MonkeyPatch, model: str, failure_at: int) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == failure_at:
            return httpx.Response(404, json={"error": {"code": "model_not_found", "message": "Unknown model."}})
        return _success(request)

    with pytest.raises((LLMNotFoundError, APIStatusError)):
        await _acceptance(monkeypatch, model, handler)
    assert len(requests) == failure_at
