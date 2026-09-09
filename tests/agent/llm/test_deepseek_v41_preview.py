"""Preview regression tests; live checks require KOLEGA_TEST_DEEPSEEK_PREVIEW=1.

The model ID advertises September 10 expiry. Keep live probes explicitly opt-in
so ordinary integration runs do not depend on a retired preview endpoint.
"""

import base64
import io
import os

import pytest
from PIL import Image

from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import (
    ImageBlock,
    Message,
    MessageHistory,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    WebSearchCallBlock,
)
from kolega_code.llm.providers.deepseek_responses import DeepSeekResponsesProvider
from kolega_code.llm.providers.models import GenerationParams

MODEL = "deepseek-v4.1-flash-expires-on-0910"


@pytest.mark.parametrize("effort", ["none", "low", "high", "max"])
@pytest.mark.parametrize("requested,expected", [(None, 384000), (128, 128), (500000, 384000)])
def test_preview_request(effort: str, requested: int | None, expected: int) -> None:
    client = LLMClient(provider="deepseek", api_key="sk-test", model=MODEL)
    assert isinstance(client.provider, DeepSeekResponsesProvider)
    request = client.provider._build_request(
        MessageHistory([Message("user", [TextBlock("Hello")])]),
        None,
        GenerationParams(thinking=effort, max_completion_tokens=requested, hosted_web_search=True),
        {"model": MODEL},
    )
    assert request["model"] == MODEL
    assert request["reasoning"] == {"effort": effort, "summary": "auto"}
    assert request["max_output_tokens"] == expected
    assert request["stream"] is True
    assert {"type": "web_search"} in request["tools"]


@pytest.fixture
def preview_client() -> LLMClient:
    if os.getenv("KOLEGA_TEST_DEEPSEEK_PREVIEW") != "1":
        pytest.skip("Expiring preview: set KOLEGA_TEST_DEEPSEEK_PREVIEW=1 to probe explicitly")
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    return LLMClient(provider="deepseek", api_key=key, model=MODEL)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["none", "low", "high", "max"])
async def test_preview_live_vision_and_effort(preview_client: LLMClient, effort: str) -> None:
    png = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(png, format="PNG")
    image = ImageBlock(
        image_type="base64", media_type="image/png", data=base64.b64encode(png.getvalue()).decode("ascii")
    )
    response = await preview_client.generate(
        messages=MessageHistory(
            [Message("user", [TextBlock("Name the color in this image. Reply with one word."), image])]
        ),
        system=None,
        model=MODEL,
        thinking=effort,
        max_completion_tokens=512,
    )
    assert isinstance(response, Message)
    assert response.get_text_content().strip().lower().rstrip(".") == "red"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_preview_live_tool_round_trip(preview_client: LLMClient) -> None:
    tool = ToolDefinition(
        name="get_secret",
        description="Get the secret word.",
        parameters=[ToolParameter(name="key", type="string", description="Lookup key", required=True)],
    )
    history = MessageHistory(
        [Message("user", [TextBlock("Call get_secret with key demo, then reply with only the returned secret word.")])]
    )
    response = await preview_client.generate(
        messages=history, system=None, model=MODEL, tools=[tool], thinking="high", max_completion_tokens=1024
    )
    assert isinstance(response, Message)
    calls = [block for block in response.content if isinstance(block, ToolCall)]
    assert len(calls) == 1
    call = calls[0]
    assert call.name == "get_secret"
    assert call.input == {"key": "demo"}
    replay = MessageHistory(
        [
            *history,
            response,
            Message("user", [ToolResult(tool_use_id=call.id, name=call.name, content="tangerine", is_error=False)]),
        ]
    )
    answer = await preview_client.generate(
        messages=replay, system=None, model=MODEL, tools=[tool], thinking="high", max_completion_tokens=1024
    )
    assert isinstance(answer, Message)
    assert answer.get_text_content().strip().lower().rstrip(".") == "tangerine"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_preview_live_search_and_full_budget(preview_client: LLMClient) -> None:
    response = await preview_client.generate(
        messages=MessageHistory(
            [Message("user", [TextBlock("Search the web for DeepSeek's official website. Reply in one sentence.")])]
        ),
        system=None,
        model=MODEL,
        thinking="none",
        hosted_web_search=True,
        # No override: exercise the catalog's full output budget on the wire.
    )
    assert isinstance(response, Message)
    assert any(isinstance(block, WebSearchCallBlock) for block in response.content)
    assert response.get_text_content().strip()
