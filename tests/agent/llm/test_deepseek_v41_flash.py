"""DeepSeek V4.1 Flash (released) regression tests.

The released model is served under the canonical ``deepseek-flash`` id; the
expiring preview id was retired. Live probes follow the standard integration
gate (``DEEPSEEK_API_KEY`` set), like the rest of the DeepSeek Responses tests.
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
)
from kolega_code.llm.providers.deepseek_responses import DeepSeekResponsesProvider
from kolega_code.llm.providers.models import GenerationParams
from kolega_code.llm.specs import get_model_specs

MODEL = "deepseek-flash"


@pytest.mark.parametrize("effort", ["none", "low", "high", "max"])
@pytest.mark.parametrize("requested,expected", [(None, 384000), (128, 128), (500000, 384000)])
def test_v41_flash_request(effort: str, requested: int | None, expected: int) -> None:
    client = LLMClient(provider="deepseek", api_key="sk-test", model=MODEL)
    assert isinstance(client.provider, DeepSeekResponsesProvider)
    request = client.provider._build_request(
        MessageHistory([Message("user", [TextBlock("Hello")])]),
        None,
        GenerationParams(thinking=effort, max_completion_tokens=requested),
        {"model": MODEL},
    )
    assert request["model"] == MODEL
    assert request["reasoning"] == {"effort": effort, "summary": "auto"}
    assert request["max_output_tokens"] == expected
    assert request["stream"] is True
    # DeepSeek's Responses surface ignores the built-in web_search tool (docs tool
    # table + live probes), so the catalog declares no hosted search. The provider
    # builder still honors an explicit hosted_web_search=True, but the agent gate
    # (supports_hosted_web_search) never turns it on for this model — assert the flag
    # in test_v41_flash_spec_has_no_hosted_web_search below.
    assert "tools" not in request or {"type": "web_search"} not in request["tools"]


def test_v41_flash_spec_has_no_hosted_web_search() -> None:
    specs = get_model_specs("deepseek", MODEL)
    assert specs["supports_hosted_web_search"] is False


@pytest.fixture
def v41_flash_client() -> LLMClient:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    return LLMClient(provider="deepseek", api_key=key, model=MODEL)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["none", "low", "high", "max"])
async def test_v41_flash_live_vision_and_effort(v41_flash_client: LLMClient, effort: str) -> None:
    png = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(png, format="PNG")
    image = ImageBlock(
        image_type="base64", media_type="image/png", data=base64.b64encode(png.getvalue()).decode("ascii")
    )
    response = await v41_flash_client.generate(
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
async def test_v41_flash_live_tool_round_trip(v41_flash_client: LLMClient) -> None:
    tool = ToolDefinition(
        name="get_secret",
        description="Get the secret word.",
        parameters=[ToolParameter(name="key", type="string", description="Lookup key", required=True)],
    )
    history = MessageHistory(
        [Message("user", [TextBlock("Call get_secret with key demo, then reply with only the returned secret word.")])]
    )
    response = await v41_flash_client.generate(
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
    answer = await v41_flash_client.generate(
        messages=replay, system=None, model=MODEL, tools=[tool], thinking="high", max_completion_tokens=1024
    )
    assert isinstance(answer, Message)
    assert answer.get_text_content().strip().lower().rstrip(".") == "tangerine"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_v41_flash_live_full_budget(v41_flash_client: LLMClient) -> None:
    response = await v41_flash_client.generate(
        messages=MessageHistory(
            [Message("user", [TextBlock("What is DeepSeek's official website? Reply in one sentence.")])]
        ),
        system=None,
        model=MODEL,
        thinking="none",
        # No override: exercise the catalog's full output budget on the wire.
    )
    assert isinstance(response, Message)
    assert response.get_text_content().strip()
