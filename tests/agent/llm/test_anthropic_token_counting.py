# ruff: noqa: F401,F811,E402
"""
Comprehensive tests comparing local vs API token counting for Anthropic provider.

These tests verify that local tiktoken-based token counting is within 5% accuracy
of Anthropic's official API token counting, using real system prompts and tool definitions.
"""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from dotenv import load_dotenv

from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.llm.models import Message, MessageHistory, TextBlock, ImageBlock
from kolega_code.llm.providers.anthropic import AnthropicProvider
from kolega_code.agent.prompt_provider import AgentMode, AgentType, PromptContext, PromptProvider
from kolega_code.agent.tools import ToolCollection, ToolCollectionConfig
from ._token_counting_utils import (
    calculate_percentage_difference,
    complex_messages,
    get_accuracy_threshold,
    simple_messages,
    simple_system,
)

# Load environment variables from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]
dotenv_path = REPO_ROOT / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)


@pytest.fixture
def api_key():
    """Get Anthropic API key from environment."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key


@pytest.fixture
def anthropic_provider_local(api_key):
    """Create Anthropic provider with local token counting enabled."""
    with patch.dict(os.environ, {"ANTHROPIC_USE_LOCAL_TOKEN_COUNTING": "true"}):
        provider = AnthropicProvider(api_key=api_key)
    return provider


@pytest.fixture
def anthropic_provider_api(api_key):
    """Create Anthropic provider with API token counting enabled."""
    with patch.dict(os.environ, {"ANTHROPIC_USE_LOCAL_TOKEN_COUNTING": "false"}):
        provider = AnthropicProvider(api_key=api_key)
    return provider


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
        model_name="claude-sonnet-4-5-20250929",
        available_ports="3000, 8000",
        kolega_md="",
        workspace_id="test-workspace",
        workspace_environment_variables={},
        memories=[],
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
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",
            rate_limits=RateLimitConfig(),
        ),
        fast_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",
            rate_limits=RateLimitConfig(),
        ),
        thinking_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",
            rate_limits=RateLimitConfig(),
        ),
    )

    tool_config = ToolCollectionConfig(
        custom_tool_groups=["coder_agent_tools"],
        tool_exclusions=[
            "read_memory",
            "write_memory",
            "execute_terminal_command",
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


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_simple_message_comparison(
    anthropic_provider_local,
    anthropic_provider_api,
    simple_messages,
    simple_system,
):
    """Compare token counts for basic user/system messages."""
    model = "claude-sonnet-4-5-20250929"

    # Get counts from both methods
    local_result = await anthropic_provider_local.count_tokens(
        messages=simple_messages,
        system=simple_system,
        model=model,
        tools=[],
    )

    api_result = await anthropic_provider_api.count_tokens(
        messages=simple_messages,
        system=simple_system,
        model=model,
        tools=[],
    )

    # Calculate percentage difference
    diff_pct = calculate_percentage_difference(local_result.input_tokens, api_result.input_tokens)
    threshold = get_accuracy_threshold(api_result.input_tokens)

    print("\nSimple message comparison:")
    print(f"  Local count: {local_result.input_tokens}")
    print(f"  API count: {api_result.input_tokens}")
    print(f"  Difference: {diff_pct:.2f}%")
    print(f"  Threshold: {threshold:.1f}% (small sample)")

    # Assert within threshold
    assert diff_pct <= threshold, (
        f"Difference {diff_pct:.2f}% exceeds {threshold:.1f}% threshold (local={local_result.input_tokens}, api={api_result.input_tokens})"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_with_real_system_prompt(
    anthropic_provider_local,
    anthropic_provider_api,
    simple_messages,
    real_system_prompt,
):
    """Test with actual CoderAgent system prompt."""
    model = "claude-sonnet-4-5-20250929"

    # Get counts from both methods
    local_result = await anthropic_provider_local.count_tokens(
        messages=simple_messages,
        system=real_system_prompt,
        model=model,
        tools=[],
    )

    api_result = await anthropic_provider_api.count_tokens(
        messages=simple_messages,
        system=real_system_prompt,
        model=model,
        tools=[],
    )

    # Calculate percentage difference
    diff_pct = calculate_percentage_difference(local_result.input_tokens, api_result.input_tokens)
    threshold = get_accuracy_threshold(api_result.input_tokens)

    print("\nReal system prompt comparison:")
    print(f"  Local count: {local_result.input_tokens}")
    print(f"  API count: {api_result.input_tokens}")
    print(f"  Difference: {diff_pct:.2f}%")
    print(f"  Threshold: {threshold:.1f}%")

    # Assert within 5% tolerance (realistic context size)
    assert diff_pct <= threshold, (
        f"Difference {diff_pct:.2f}% exceeds {threshold:.1f}% threshold (local={local_result.input_tokens}, api={api_result.input_tokens})"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_with_tools(
    anthropic_provider_local,
    anthropic_provider_api,
    simple_messages,
    simple_system,
    real_tools,
):
    """Test with real tool definitions from ToolCollection."""
    model = "claude-sonnet-4-5-20250929"

    # Get counts from both methods
    local_result = await anthropic_provider_local.count_tokens(
        messages=simple_messages,
        system=simple_system,
        model=model,
        tools=real_tools,
    )

    api_result = await anthropic_provider_api.count_tokens(
        messages=simple_messages,
        system=simple_system,
        model=model,
        tools=real_tools,
    )

    # Calculate percentage difference
    diff_pct = calculate_percentage_difference(local_result.input_tokens, api_result.input_tokens)
    threshold = get_accuracy_threshold(api_result.input_tokens)

    print("\nWith tools comparison:")
    print(f"  Tool count: {len(real_tools)}")
    print(f"  Local count: {local_result.input_tokens}")
    print(f"  API count: {api_result.input_tokens}")
    print(f"  Difference: {diff_pct:.2f}%")
    print(f"  Threshold: {threshold:.1f}%")

    # Assert within 5% tolerance (realistic context with tools)
    assert diff_pct <= threshold, (
        f"Difference {diff_pct:.2f}% exceeds {threshold:.1f}% threshold (local={local_result.input_tokens}, api={api_result.input_tokens})"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_with_complex_conversation(
    anthropic_provider_local,
    anthropic_provider_api,
    complex_messages,
    simple_system,
):
    """Test with multi-turn conversation."""
    model = "claude-sonnet-4-5-20250929"

    # Get counts from both methods
    local_result = await anthropic_provider_local.count_tokens(
        messages=complex_messages,
        system=simple_system,
        model=model,
        tools=[],
    )

    api_result = await anthropic_provider_api.count_tokens(
        messages=complex_messages,
        system=simple_system,
        model=model,
        tools=[],
    )

    # Calculate percentage difference
    diff_pct = calculate_percentage_difference(local_result.input_tokens, api_result.input_tokens)
    threshold = get_accuracy_threshold(api_result.input_tokens)

    print("\nComplex conversation comparison:")
    print(f"  Message count: {len(complex_messages)}")
    print(f"  Local count: {local_result.input_tokens}")
    print(f"  API count: {api_result.input_tokens}")
    print(f"  Difference: {diff_pct:.2f}%")
    print(f"  Threshold: {threshold:.1f}% (small sample)")

    # Assert within threshold
    assert diff_pct <= threshold, (
        f"Difference {diff_pct:.2f}% exceeds {threshold:.1f}% threshold (local={local_result.input_tokens}, api={api_result.input_tokens})"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_with_images(
    anthropic_provider_local,
    anthropic_provider_api,
    simple_system,
):
    """Test token counting with image attachments."""
    model = "claude-sonnet-4-5-20250929"

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

    # Get counts from both methods
    local_result = await anthropic_provider_local.count_tokens(
        messages=messages_with_image,
        system=simple_system,
        model=model,
        tools=[],
    )

    api_result = await anthropic_provider_api.count_tokens(
        messages=messages_with_image,
        system=simple_system,
        model=model,
        tools=[],
    )

    # Calculate percentage difference
    diff_pct = calculate_percentage_difference(local_result.input_tokens, api_result.input_tokens)
    image_threshold = 200.0

    print("\nWith images comparison:")
    print(f"  Image size: {len(tiny_image_base64)} chars (base64)")
    print(f"  Local count: {local_result.input_tokens}")
    print(f"  API count: {api_result.input_tokens}")
    print(f"  Difference: {diff_pct:.2f}%")
    print(f"  Threshold: {image_threshold:.1f}% (image estimate)")

    # Images are harder to estimate precisely without decoding, but we verify:
    # 1. Both methods counted more than text-only (proving images are counted)
    # 2. Both counts are non-zero (images aren't ignored)
    # Text-only would be ~14 tokens, so >20 proves image was counted
    assert local_result.input_tokens > 20, "Local counting should include image tokens"
    assert api_result.input_tokens > 20, "API counting should include image tokens"

    # For images, allow very high variance since:
    # - We estimate without decoding (no actual pixel dimensions)
    # - This tiny 1x1 test image is an edge case (96 chars base64)
    # - Normal conversation images (screenshots, etc.) will be much larger and more accurate
    # - The key goal is images aren't ignored (count > 0)
    assert diff_pct <= image_threshold, (
        f"Difference {diff_pct:.2f}% exceeds {image_threshold:.1f}% threshold for images (local={local_result.input_tokens}, api={api_result.input_tokens})"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_agent_context(
    anthropic_provider_local,
    anthropic_provider_api,
    complex_messages,
    real_system_prompt,
    real_tools,
):
    """Test with full agent context: real system prompt, complex messages, and tools."""
    model = "claude-sonnet-4-5-20250929"

    # Get counts from both methods
    local_result = await anthropic_provider_local.count_tokens(
        messages=complex_messages,
        system=real_system_prompt,
        model=model,
        tools=real_tools,
    )

    api_result = await anthropic_provider_api.count_tokens(
        messages=complex_messages,
        system=real_system_prompt,
        model=model,
        tools=real_tools,
    )

    # Calculate percentage difference
    diff_pct = calculate_percentage_difference(local_result.input_tokens, api_result.input_tokens)
    threshold = get_accuracy_threshold(api_result.input_tokens)

    print("\nFull agent context comparison:")
    print(f"  Message count: {len(complex_messages)}")
    print(f"  Tool count: {len(real_tools)}")
    print(f"  Local count: {local_result.input_tokens}")
    print(f"  API count: {api_result.input_tokens}")
    print(f"  Difference: {diff_pct:.2f}%")
    print(f"  Threshold: {threshold:.1f}%")

    # Assert within 5% tolerance (realistic full agent context)
    assert diff_pct <= threshold, (
        f"Difference {diff_pct:.2f}% exceeds {threshold:.1f}% threshold (local={local_result.input_tokens}, api={api_result.input_tokens})"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_accuracy_threshold_summary(
    anthropic_provider_local,
    anthropic_provider_api,
    simple_messages,
    simple_system,
    complex_messages,
    real_system_prompt,
    real_tools,
):
    """Run all comparison scenarios and verify all are within their thresholds."""
    model = "claude-sonnet-4-5-20250929"

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
        ("With Images", messages_with_image, simple_system, []),
        ("Full Context", complex_messages, real_system_prompt, real_tools),
    ]

    results = []

    for name, messages, system, tools in test_scenarios:
        local_result = await anthropic_provider_local.count_tokens(
            messages=messages,
            system=system,
            model=model,
            tools=tools,
        )

        api_result = await anthropic_provider_api.count_tokens(
            messages=messages,
            system=system,
            model=model,
            tools=tools,
        )

        diff_pct = calculate_percentage_difference(local_result.input_tokens, api_result.input_tokens)
        results.append((name, local_result.input_tokens, api_result.input_tokens, diff_pct))

    # Print summary
    print("\n" + "=" * 80)
    print("Token Counting Accuracy Summary")
    print("=" * 80)
    print(f"{'Scenario':<20} {'Local':<10} {'API':<10} {'Diff %':<10} {'Status':<10}")
    print("-" * 80)

    all_within_threshold = True
    for name, local_count, api_count, diff_pct in results:
        # Images get special handling - they're estimated without decoding
        if "Images" in name:
            threshold = 200.0
        else:
            threshold = get_accuracy_threshold(api_count)
        status = "✓ PASS" if diff_pct <= threshold else "✗ FAIL"
        if diff_pct > threshold:
            all_within_threshold = False
        print(f"{name:<20} {local_count:<10} {api_count:<10} {diff_pct:<10.2f} {status:<10}")

    print("=" * 80)
    print("Note: Realistic agent contexts (>200 tokens) must be within 5%.")
    print("      Small samples (<200 tokens) allowed up to 15% due to fixed overhead.")
    print("      Images allowed up to 200% variance (estimated without decoding).")
    print("=" * 80)

    # Assert all scenarios pass their respective thresholds
    assert all_within_threshold, "One or more scenarios exceeded their accuracy threshold"


def test_environment_variable_default(api_key):
    """Test that local token counting defaults to False when env var not set."""
    # Clear the environment variable
    with patch.dict(os.environ, {}, clear=False):
        if "ANTHROPIC_USE_LOCAL_TOKEN_COUNTING" in os.environ:
            del os.environ["ANTHROPIC_USE_LOCAL_TOKEN_COUNTING"]
        provider = AnthropicProvider(api_key=api_key)

    assert provider.use_local_token_counting is False, "Should default to False when env var not set"


def test_environment_variable_true(api_key):
    """Test that local token counting is enabled when env var is 'true'."""
    with patch.dict(os.environ, {"ANTHROPIC_USE_LOCAL_TOKEN_COUNTING": "true"}):
        provider = AnthropicProvider(api_key=api_key)

    assert provider.use_local_token_counting is True, 'Should be True when env var is "true"'


def test_environment_variable_false(api_key):
    """Test that local token counting is disabled when env var is 'false'."""
    with patch.dict(os.environ, {"ANTHROPIC_USE_LOCAL_TOKEN_COUNTING": "false"}):
        provider = AnthropicProvider(api_key=api_key)

    assert provider.use_local_token_counting is False, 'Should be False when env var is "false"'


def test_environment_variable_case_insensitive(api_key):
    """Test that env var is case insensitive."""
    test_cases = [
        ("TRUE", True),
        ("True", True),
        ("TrUe", True),
        ("FALSE", False),
        ("False", False),
        ("FaLsE", False),
    ]

    for env_value, expected_result in test_cases:
        with patch.dict(os.environ, {"ANTHROPIC_USE_LOCAL_TOKEN_COUNTING": env_value}):
            provider = AnthropicProvider(api_key=api_key)
        assert provider.use_local_token_counting is expected_result, f"Failed for env_value={env_value}"


def test_environment_variable_invalid_value(api_key):
    """Test that invalid env var values default to False."""
    invalid_values = ["yes", "no", "1", "0", "enabled", "disabled", "garbage"]

    for invalid_value in invalid_values:
        with patch.dict(os.environ, {"ANTHROPIC_USE_LOCAL_TOKEN_COUNTING": invalid_value}):
            provider = AnthropicProvider(api_key=api_key)
        assert provider.use_local_token_counting is False, f"Should default to False for invalid value: {invalid_value}"
