"""Small, explicit model-connection probe used by TUI setup surfaces."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from kolega_code.config import AgentConfig
from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import LLMError, llm_error_message
from kolega_code.llm.models import Message, MessageHistory, TextBlock


CONNECTION_TEST_TIMEOUT_SECONDS = 30.0
CONNECTION_TEST_MAX_COMPLETION_TOKENS = 32


@dataclass(frozen=True)
class ModelConnectionResult:
    ok: bool
    message: str


async def test_model_connection(
    config: AgentConfig,
    *,
    client_factory: Callable[..., Any] = LLMClient,
    timeout: float = CONNECTION_TEST_TIMEOUT_SECONDS,
) -> ModelConnectionResult:
    """Send a tiny no-tool prompt through the selected model.

    This is intentionally opt-in: providers do not expose one uniform, free auth
    endpoint, so a real generation is the only dependable cross-provider probe.
    """

    model_config = config.long_context_config
    try:
        client = client_factory(
            provider=model_config.provider,
            api_key=config.get_api_key(model_config.provider) or "",
            max_retries=0,
            requests_per_minute=model_config.rate_limits.requests_per_minute,
            tokens_per_minute=model_config.rate_limits.tokens_per_minute,
            token_manager=config.get_chatgpt_token_manager(),
        )
        messages = MessageHistory([Message(role="user", content=[TextBlock(text="Reply with OK.")])])
        await asyncio.wait_for(
            client.generate(
                messages=messages,
                model=model_config.model,
                max_completion_tokens=CONNECTION_TEST_MAX_COMPLETION_TOKENS,
                tools=[],
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return ModelConnectionResult(False, f"Connection test timed out after {timeout:g} seconds.")
    except LLMError as exc:
        return ModelConnectionResult(False, llm_error_message(exc, model=model_config.model))
    except Exception:
        return ModelConnectionResult(
            False, "The connection test failed unexpectedly. Check the provider and try again."
        )
    return ModelConnectionResult(
        True,
        f"Connected to {model_config.provider.value}/{model_config.model}.",
    )
