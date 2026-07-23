from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from kolega_code.llm.models import Message, MessageHistory, TextBlock
from kolega_code.llm.providers.google import GoogleProvider, GoogleStreamWrapper
from kolega_code.llm.providers.models import GenerationParams


class _FakeGoogleModels:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object]] = []
        self.response = object()
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

    if method == "generate":
        assert result is fake_models.response
    else:
        assert isinstance(result, GoogleStreamWrapper)
        assert result.gemini_stream is fake_models.stream
