import base64
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from google import genai
from google.genai import types as genai_types

from kolega_code.llm.models import (
    Message,
    MessageHistory,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from kolega_code.llm.providers.google import GoogleProvider, GoogleStreamWrapper
from kolega_code.llm.providers.models import GenerationParams


class _FakeGoogleModels:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object]] = []
        # A real Message: the provider attaches normalized usage to the message
        # produced by (the monkeypatched, identity) Message.from_google.
        self.response = Message(role="assistant", content="")
        self.stream = SimpleNamespace()

    async def generate_content(self, *, model: str, contents: object, config: object) -> object:
        del contents
        self.calls.append(("generate", model, config))
        return self.response

    async def generate_content_stream(self, *, model: str, contents: object, config: object) -> object:
        del contents
        self.calls.append(("stream", model, config))
        return self.stream


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["generate", "stream"])
@pytest.mark.parametrize(
    "model,thinking,expected_temperature",
    [
        ("gemini-3.7-flash", "medium", None),
        ("gemini-3.6-flash", "medium", None),
        ("gemini-3.5-flash-lite", "minimal", None),
        ("gemini-3.1-pro-preview", "high", 0.7),
    ],
)
async def test_google_request_config_respects_model_temperature_support(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    model: str,
    thinking: str,
    expected_temperature: float | None,
) -> None:
    provider = GoogleProvider(api_key="test-key")
    fake_models = _FakeGoogleModels()
    cast(Any, provider).async_client = SimpleNamespace(aio=SimpleNamespace(models=fake_models))
    cast(Any, provider).rate_limiter = SimpleNamespace(acquire=AsyncMock())
    monkeypatch.setattr(Message, "from_google", lambda response: response)

    messages = MessageHistory([Message(role="user", content=[TextBlock(text="Hello")])])
    system = Message(role="system", content=[TextBlock(text="Be concise")])
    params = GenerationParams(
        temperature=0.7,
        max_completion_tokens=1024,
        tools=[
            ToolDefinition(
                name="lookup",
                description="Look something up.",
                parameters=[ToolParameter(name="query", type="string", description="Search query.", required=True)],
            )
        ],
        thinking=thinking,
    )

    result = await getattr(provider, method)(messages, system=system, params=params, model=model)

    assert fake_models.calls[0][:2] == (method, model)
    config = cast(Any, fake_models.calls[0][2])
    serialized = config.model_dump(exclude_none=True)
    if expected_temperature is None:
        assert "temperature" not in serialized
    else:
        assert serialized["temperature"] == expected_temperature
    assert serialized["thinking_config"]["thinking_level"].lower() == thinking
    assert config.tools is not None
    assert config.tools[0].function_declarations is not None
    declaration = config.tools[0].function_declarations[0]
    assert type(declaration) is not genai_types.FunctionDeclaration
    assert isinstance(declaration.parameters, genai_types.Schema)
    assert declaration.parameters.properties is not None
    assert set(declaration.parameters.properties) == {"query"}

    if method == "generate":
        assert result is fake_models.response
    else:
        assert isinstance(result, GoogleStreamWrapper)
        assert result.gemini_stream is fake_models.stream


@pytest.mark.asyncio
async def test_google_mldev_serializes_function_response_as_user_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real 2.12.1 ML Dev request path for a complete tool-use round trip."""

    thought_signature = b"\x01\x02gemini-signature"
    history = MessageHistory(
        [
            Message(role="user", content=[TextBlock(text="Inspect the repository with gh.")]),
            Message(
                role="assistant",
                content=[
                    ToolCall(
                        id="call-gh-1",
                        name="gh",
                        input={"args": ["repo", "view"]},
                        thought_signature=thought_signature,
                    )
                ],
                usage_metadata={"provider": "google"},
            ),
            Message(
                role="user",
                content=[
                    ToolResult(
                        tool_use_id="call-gh-1",
                        content="kolega-ai/kolega-code",
                        name="gh",
                        is_error=False,
                    )
                ],
            ),
        ]
    )

    client = genai.Client(api_key="test-key")
    request = AsyncMock(return_value=genai_types.HttpResponse(headers={}, body="{}"))
    monkeypatch.setattr(client._api_client, "async_request", request)

    try:
        await client.aio.models.generate_content(
            model="gemini-test",
            contents=history.to_google(),
        )
    finally:
        await client.aio.aclose()
        client.close()

    request_call = request.await_args
    assert request_call is not None
    request_payload = cast(dict[str, Any], request_call.args[2])
    contents = cast(list[dict[str, Any]], request_payload["contents"])

    assert [content["role"] for content in contents] == ["user", "model", "user"]
    assert all(content["role"] != "tool" for content in contents)

    function_call_part = contents[1]["parts"][0]
    assert function_call_part["functionCall"] == {
        "id": "call-gh-1",
        "name": "gh",
        "args": {"args": ["repo", "view"]},
    }
    assert function_call_part["thoughtSignature"] == base64.b64encode(thought_signature).decode("ascii")

    function_response_part = contents[2]["parts"][0]
    assert function_response_part["functionResponse"] == {
        "id": "call-gh-1",
        "name": "gh",
        "response": {"output": "kolega-ai/kolega-code"},
    }
