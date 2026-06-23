"""
Comprehensive tests comparing local vs API token counting for OpenAI provider.

These tests verify that local tiktoken-based token counting is within reasonable accuracy
of OpenAI's official API token counting, using real system prompts and tool definitions.
"""

import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from dotenv import load_dotenv

from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.llm.models import (
    ImageBlock,
    Message,
    MessageHistory,
    TextBlock,
    ToolCall,
    ToolResult,
)
from kolega_code.llm.providers.openai import OpenAIProvider
from kolega_code.agent.prompt_provider import AgentMode, AgentType, PromptContext, PromptProvider
from kolega_code.agent.tools import ToolCollection, ToolCollectionConfig

# Load environment variables from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]
dotenv_path = REPO_ROOT / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)


@pytest.fixture
def api_key():
    """Get OpenAI API key from environment."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture
def openai_provider(api_key):
    """Create OpenAI provider for testing."""
    return OpenAIProvider(api_key=api_key)


@pytest.fixture
def simple_messages():
    """Simple test messages."""
    return MessageHistory([Message("user", [TextBlock("Hello, how are you?")])])


@pytest.fixture
def simple_system():
    """Simple system message."""
    return Message("system", [TextBlock("You are a helpful assistant.")])


@pytest.fixture
def real_system_prompt():
    """Get real system prompt from CoderAgent."""
    prompt_provider = PromptProvider()
    context = PromptContext(
        system_name="Kolega Studio",
        project_path="/test/project",
        is_git_repo=True,
        platform="Linux",
        date_today="2025-01-01",
        model_name="gpt-4o",
        available_ports=[3000, 8000],
        kolega_md="",
        workspace_id="test-workspace",
        workspace_environment_variables=None,
        memories=None,
    )

    prompt_text = prompt_provider.get_system_prompt(
        agent_type=AgentType.CODER,
        mode=AgentMode.CLI,
        template_slug=None,
        context=context,
    )

    return Message("system", [TextBlock(prompt_text)])


@pytest.fixture
def real_tools(tmp_path):
    """Get real tool definitions from ToolCollection."""
    mock_connection_manager = Mock(spec=AgentConnectionManager)
    mock_config = AgentConfig(
        anthropic_api_key="test",
        openai_api_key="test",
        long_context_config=ModelConfig(
            provider=ModelProvider.OPENAI,
            model="gpt-4o",
            rate_limits=RateLimitConfig(),
        ),
        fast_config=ModelConfig(
            provider=ModelProvider.OPENAI,
            model="gpt-4o",
            rate_limits=RateLimitConfig(),
        ),
        thinking_config=ModelConfig(
            provider=ModelProvider.OPENAI,
            model="gpt-4o",
            rate_limits=RateLimitConfig(),
        ),
    )

    tool_config = ToolCollectionConfig(
        custom_tool_groups=["coder_agent_tools"],
        tool_exclusions=[
            "read_memory",
            "write_memory",
            "execute_terminal_command",
            "replace_lines",
            "get_tool_list",
            "log_error",
            "log_info",
            "run_command",
            "dispatch_coding_agent",
        ],
    )

    tool_collection = ToolCollection(
        project_path=tmp_path,
        workspace_id="test-workspace",
        thread_id="test-thread",
        connection_manager=mock_connection_manager,
        config=mock_config,
        caller=None,
        tool_config=tool_config,
    )

    return tool_collection.get_tool_list()


@pytest.fixture
def complex_messages():
    """Multi-turn conversation with various content types."""
    return MessageHistory(
        [
            Message("user", [TextBlock("Can you help me write a Python function?")]),
            Message(
                "assistant",
                [
                    TextBlock(
                        "Of course! I'd be happy to help you write a Python function. What would you like the function to do?"
                    )
                ],
            ),
            Message("user", [TextBlock("I need a function that calculates the factorial of a number recursively.")]),
            Message(
                "assistant",
                [
                    TextBlock(
                        "Here's a recursive factorial function:\n\n```python\ndef factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n - 1)\n```"
                    )
                ],
            ),
        ]
    )


@pytest.fixture
def messages_with_tool_calls():
    """Messages containing tool calls and results."""
    return MessageHistory(
        [
            Message("user", [TextBlock("Can you read the README.md file?")]),
            Message(
                "assistant",
                [
                    TextBlock("I'll read that file for you."),
                    ToolCall(
                        id="call_123",
                        name="read_file",
                        input={"target_file": "README.md"},
                    ),
                ],
            ),
            Message(
                "user",
                [
                    ToolResult(
                        tool_use_id="call_123",
                        name="read_file",
                        content="# My Project\n\nThis is a sample README file.",
                        is_error=False,
                    )
                ],
            ),
            Message(
                "assistant",
                [TextBlock('I\'ve read the README.md file. It contains information about "My Project".')],
            ),
        ]
    )


def calculate_percentage_difference(local_count: int, api_count: int) -> float:
    """Calculate percentage difference between local and API token counts."""
    if api_count == 0:
        return 0.0
    return abs(local_count - api_count) / api_count * 100


def get_accuracy_threshold(api_count: int, has_tools: bool = False) -> float:
    """Get appropriate accuracy threshold based on token count.

    Small token counts (<200) have higher variance due to fixed overhead,
    so we use a more lenient threshold. For realistic agent contexts (>200 tokens),
    we enforce a stricter threshold. Tool definitions have additional variance.
    """
    if api_count < 200:
        return 15.0  # Lenient threshold for small samples
    if has_tools:
        return 20.0  # More lenient for tool definitions (OpenAI uses compact internal format)
    return 10.0  # Moderate threshold for realistic contexts (OpenAI less predictable than Anthropic)


@pytest.mark.asyncio
async def test_nested_image_tool_result_contributes_image_tokens():
    provider = OpenAIProvider(api_key="test")
    image = ImageBlock(image_type="base64", media_type="image/png", data="A" * 100)
    messages = MessageHistory(
        [
            Message(
                "user",
                [ToolResult(tool_use_id="call_img", name="read_image", content=[image], is_error=False)],
            )
        ]
    )

    result = await provider.count_tokens(messages=messages, tools=[])

    assert result.input_tokens >= provider._estimate_image_tokens(len(image.data))


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_simple_message_comparison(
    openai_provider,
    simple_messages,
    simple_system,
):
    """Compare token counts for basic user/system messages."""
    model = "gpt-4o"

    # Get local count
    local_result = await openai_provider.count_tokens(
        messages=simple_messages,
        system=simple_system,
        model=model,
        tools=[],
    )

    # Get API count by making a real call
    combined_messages = MessageHistory([simple_system] + list(simple_messages))
    response = await openai_provider.async_client.chat.completions.create(
        model=model,
        messages=combined_messages.to_openai(),
        max_tokens=1,  # Minimal completion to save costs
    )
    api_count = response.usage.prompt_tokens

    # Calculate percentage difference
    diff_pct = calculate_percentage_difference(local_result.input_tokens, api_count)
    threshold = get_accuracy_threshold(api_count)

    print("\nSimple message comparison:")
    print(f"  Local count: {local_result.input_tokens}")
    print(f"  API count: {api_count}")
    print(f"  Difference: {diff_pct:.2f}%")
    print(f"  Threshold: {threshold:.1f}%")

    # Assert within threshold
    assert diff_pct <= threshold, (
        f"Difference {diff_pct:.2f}% exceeds {threshold:.1f}% threshold (local={local_result.input_tokens}, api={api_count})"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_with_real_system_prompt(
    openai_provider,
    simple_messages,
    real_system_prompt,
):
    """Test with actual CoderAgent system prompt."""
    model = "gpt-4o"

    # Get local count
    local_result = await openai_provider.count_tokens(
        messages=simple_messages,
        system=real_system_prompt,
        model=model,
        tools=[],
    )

    # Get API count by making a real call
    combined_messages = MessageHistory([real_system_prompt] + list(simple_messages))
    response = await openai_provider.async_client.chat.completions.create(
        model=model,
        messages=combined_messages.to_openai(),
        max_tokens=1,
    )
    api_count = response.usage.prompt_tokens

    # Calculate percentage difference
    diff_pct = calculate_percentage_difference(local_result.input_tokens, api_count)
    threshold = get_accuracy_threshold(api_count)

    print("\nReal system prompt comparison:")
    print(f"  Local count: {local_result.input_tokens}")
    print(f"  API count: {api_count}")
    print(f"  Difference: {diff_pct:.2f}%")
    print(f"  Threshold: {threshold:.1f}%")

    # Assert within threshold (realistic context size)
    assert diff_pct <= threshold, (
        f"Difference {diff_pct:.2f}% exceeds {threshold:.1f}% threshold (local={local_result.input_tokens}, api={api_count})"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_with_tools(
    openai_provider,
    simple_messages,
    simple_system,
    real_tools,
):
    """Test with real tool definitions from ToolCollection."""
    model = "gpt-4o"

    # Get local count
    local_result = await openai_provider.count_tokens(
        messages=simple_messages,
        system=simple_system,
        model=model,
        tools=real_tools,
    )

    # Get API count by making a real call
    combined_messages = MessageHistory([simple_system] + list(simple_messages))
    response = await openai_provider.async_client.chat.completions.create(
        model=model,
        messages=combined_messages.to_openai(),
        tools=[t.to_openai() for t in real_tools],
        max_tokens=1,
    )
    api_count = response.usage.prompt_tokens

    # Calculate percentage difference
    diff_pct = calculate_percentage_difference(local_result.input_tokens, api_count)
    threshold = get_accuracy_threshold(api_count, has_tools=True)

    print("\nWith tools comparison:")
    print(f"  Tool count: {len(real_tools)}")
    print(f"  Local count: {local_result.input_tokens}")
    print(f"  API count: {api_count}")
    print(f"  Difference: {diff_pct:.2f}%")
    print(f"  Threshold: {threshold:.1f}%")

    # Assert within threshold (realistic context with tools)
    assert diff_pct <= threshold, (
        f"Difference {diff_pct:.2f}% exceeds {threshold:.1f}% threshold (local={local_result.input_tokens}, api={api_count})"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_with_complex_conversation(
    openai_provider,
    complex_messages,
    simple_system,
):
    """Test with multi-turn conversation."""
    model = "gpt-4o"

    # Get local count
    local_result = await openai_provider.count_tokens(
        messages=complex_messages,
        system=simple_system,
        model=model,
        tools=[],
    )

    # Get API count by making a real call
    combined_messages = MessageHistory([simple_system] + list(complex_messages))
    response = await openai_provider.async_client.chat.completions.create(
        model=model,
        messages=combined_messages.to_openai(),
        max_tokens=1,
    )
    api_count = response.usage.prompt_tokens

    # Calculate percentage difference
    diff_pct = calculate_percentage_difference(local_result.input_tokens, api_count)
    threshold = get_accuracy_threshold(api_count)

    print("\nComplex conversation comparison:")
    print(f"  Message count: {len(complex_messages)}")
    print(f"  Local count: {local_result.input_tokens}")
    print(f"  API count: {api_count}")
    print(f"  Difference: {diff_pct:.2f}%")
    print(f"  Threshold: {threshold:.1f}%")

    # Assert within threshold
    assert diff_pct <= threshold, (
        f"Difference {diff_pct:.2f}% exceeds {threshold:.1f}% threshold (local={local_result.input_tokens}, api={api_count})"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_with_images(
    openai_provider,
    simple_system,
):
    """Test token counting with image attachments."""
    model = "gpt-4o"

    # Create a small test image (1x1 pixel PNG as base64)
    # This is a tiny 1x1 transparent PNG
    tiny_image_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )

    # Create a message with image
    messages_with_image = MessageHistory(
        [
            Message(
                "user",
                [
                    TextBlock("What do you see in this image?"),
                    ImageBlock(image_type="base64", media_type="image/png", data=tiny_image_base64),
                ],
            )
        ]
    )

    # Get local count
    local_result = await openai_provider.count_tokens(
        messages=messages_with_image,
        system=simple_system,
        model=model,
        tools=[],
    )

    # Get API count by making a real call
    combined_messages = MessageHistory([simple_system] + list(messages_with_image))
    response = await openai_provider.async_client.chat.completions.create(
        model=model,
        messages=combined_messages.to_openai(),
        max_tokens=1,
    )
    api_count = response.usage.prompt_tokens

    # Calculate percentage difference
    diff_pct = calculate_percentage_difference(local_result.input_tokens, api_count)

    print("\nWith images comparison:")
    print(f"  Image size: {len(tiny_image_base64)} chars (base64)")
    print(f"  Local count: {local_result.input_tokens}")
    print(f"  API count: {api_count}")
    print(f"  Difference: {diff_pct:.2f}%")

    # Images are harder to estimate precisely without decoding, but we verify:
    # 1. Both methods counted more than text-only (proving images are counted)
    # 2. Both counts are non-zero (images aren't ignored)
    # Text-only would be ~14 tokens, so >20 proves image was counted
    assert local_result.input_tokens > 20, "Local counting should include image tokens"
    assert api_count > 20, "API counting should include image tokens"

    # For images, allow higher variance since:
    # - We estimate without decoding (no actual pixel dimensions)
    # - OpenAI has complex image token calculation based on detail level
    # - This tiny 1x1 test image is an edge case
    image_threshold = 100.0
    assert diff_pct <= image_threshold, (
        f"Difference {diff_pct:.2f}% exceeds {image_threshold:.1f}% threshold for images (local={local_result.input_tokens}, api={api_count})"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_with_tool_calls(
    openai_provider,
    messages_with_tool_calls,
    simple_system,
):
    """Test token counting with tool calls and results."""
    model = "gpt-4o"

    # Get local count
    local_result = await openai_provider.count_tokens(
        messages=messages_with_tool_calls,
        system=simple_system,
        model=model,
        tools=[],
    )

    # Get API count by making a real call
    combined_messages = MessageHistory([simple_system] + list(messages_with_tool_calls))
    response = await openai_provider.async_client.chat.completions.create(
        model=model,
        messages=combined_messages.to_openai(),
        max_tokens=1,
    )
    api_count = response.usage.prompt_tokens

    # Calculate percentage difference
    diff_pct = calculate_percentage_difference(local_result.input_tokens, api_count)
    # Tool calls/results have higher variance in token counting, similar to images
    threshold = 25.0 if api_count < 200 else 15.0

    print("\nWith tool calls comparison:")
    print(f"  Message count: {len(messages_with_tool_calls)}")
    print(f"  Local count: {local_result.input_tokens}")
    print(f"  API count: {api_count}")
    print(f"  Difference: {diff_pct:.2f}%")
    print(f"  Threshold: {threshold:.1f}%")

    # Assert within threshold (tool calls have higher variance)
    assert diff_pct <= threshold, (
        f"Difference {diff_pct:.2f}% exceeds {threshold:.1f}% threshold (local={local_result.input_tokens}, api={api_count})"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_agent_context(
    openai_provider,
    complex_messages,
    real_system_prompt,
    real_tools,
):
    """Test with full agent context: real system prompt, complex messages, and tools."""
    model = "gpt-4o"

    # Get local count
    local_result = await openai_provider.count_tokens(
        messages=complex_messages,
        system=real_system_prompt,
        model=model,
        tools=real_tools,
    )

    # Get API count by making a real call
    combined_messages = MessageHistory([real_system_prompt] + list(complex_messages))
    response = await openai_provider.async_client.chat.completions.create(
        model=model,
        messages=combined_messages.to_openai(),
        tools=[t.to_openai() for t in real_tools],
        max_tokens=1,
    )
    api_count = response.usage.prompt_tokens

    # Calculate percentage difference
    diff_pct = calculate_percentage_difference(local_result.input_tokens, api_count)
    threshold = get_accuracy_threshold(api_count, has_tools=True)

    print("\nFull agent context comparison:")
    print(f"  Message count: {len(complex_messages)}")
    print(f"  Tool count: {len(real_tools)}")
    print(f"  Local count: {local_result.input_tokens}")
    print(f"  API count: {api_count}")
    print(f"  Difference: {diff_pct:.2f}%")
    print(f"  Threshold: {threshold:.1f}%")

    # Assert within threshold (realistic full agent context)
    assert diff_pct <= threshold, (
        f"Difference {diff_pct:.2f}% exceeds {threshold:.1f}% threshold (local={local_result.input_tokens}, api={api_count})"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_accuracy_threshold_summary(
    openai_provider,
    simple_messages,
    simple_system,
    complex_messages,
    real_system_prompt,
    real_tools,
    messages_with_tool_calls,
):
    """Run all comparison scenarios and verify all are within their thresholds."""
    model = "gpt-4o"

    # Create message with image for testing
    tiny_image_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    messages_with_image = MessageHistory(
        [
            Message(
                "user",
                [
                    TextBlock("What do you see?"),
                    ImageBlock(image_type="base64", media_type="image/png", data=tiny_image_base64),
                ],
            )
        ]
    )

    test_scenarios = [
        ("Simple", simple_messages, simple_system, []),
        ("Real System", simple_messages, real_system_prompt, []),
        ("With Tools", simple_messages, simple_system, real_tools),
        ("Complex Messages", complex_messages, simple_system, []),
        ("With Tool Calls", messages_with_tool_calls, simple_system, []),
        ("With Images", messages_with_image, simple_system, []),
        ("Full Context", complex_messages, real_system_prompt, real_tools),
    ]

    results = []

    for name, messages, system, tools in test_scenarios:
        # Get local count
        local_result = await openai_provider.count_tokens(
            messages=messages,
            system=system,
            model=model,
            tools=tools,
        )

        # Get API count
        combined_messages = MessageHistory([system] + list(messages))
        if tools:
            response = await openai_provider.async_client.chat.completions.create(
                model=model,
                messages=combined_messages.to_openai(),
                tools=[t.to_openai() for t in tools],
                max_tokens=1,
            )
        else:
            response = await openai_provider.async_client.chat.completions.create(
                model=model,
                messages=combined_messages.to_openai(),
                max_tokens=1,
            )
        api_count = response.usage.prompt_tokens

        diff_pct = calculate_percentage_difference(local_result.input_tokens, api_count)
        results.append((name, local_result.input_tokens, api_count, diff_pct))

    # Print summary
    print("\n" + "=" * 80)
    print("Token Counting Accuracy Summary (OpenAI)")
    print("=" * 80)
    print(f"{'Scenario':<20} {'Local':<10} {'API':<10} {'Diff %':<10} {'Status':<10}")
    print("-" * 80)

    all_within_threshold = True
    for name, local_count, api_count, diff_pct in results:
        # Images get special handling - they're estimated without decoding
        if "Images" in name:
            threshold = 100.0
        elif "Tool Calls" in name:
            # Tool calls/results have higher variance, especially in small contexts
            threshold = 25.0 if api_count < 200 else 15.0
        elif "Tools" in name or "Full Context" in name:
            threshold = get_accuracy_threshold(api_count, has_tools=True)
        else:
            threshold = get_accuracy_threshold(api_count)
        status = "✓ PASS" if diff_pct <= threshold else "✗ FAIL"
        if diff_pct > threshold:
            all_within_threshold = False
        print(f"{name:<20} {local_count:<10} {api_count:<10} {diff_pct:<10.2f} {status:<10}")

    print("=" * 80)
    print("Note: Realistic agent contexts (>200 tokens) must be within 10%.")
    print("      Contexts with tools allowed up to 20% due to OpenAI's compact format.")
    print("      Contexts with tool calls/results allowed up to 25% for small samples.")
    print("      Small samples (<200 tokens) allowed up to 15% due to fixed overhead.")
    print("      Images allowed up to 100% variance (estimated without decoding).")
    print("=" * 80)

    # Assert all scenarios pass their respective thresholds
    assert all_within_threshold, "One or more scenarios exceeded their accuracy threshold"
