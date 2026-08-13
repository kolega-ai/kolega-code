"""Small, explicit model-connection probe used by TUI setup surfaces."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Optional

from kolega_code.config import ModelProvider, RateLimitConfig
from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import LLMError, llm_error_message
from kolega_code.llm.ledger import helper_origin, llm_call_origin
from kolega_code.llm.models import Message, MessageHistory, TextBlock


CONNECTION_TEST_TIMEOUT_SECONDS = 30.0
CONNECTION_TEST_MAX_COMPLETION_TOKENS = 32


@dataclass(frozen=True)
class ModelConnectionResult:
    ok: bool
    message: str


async def _probe_generate(client: Any, model: str, messages: MessageHistory) -> Any:
    with llm_call_origin(helper_origin("model_connection")):
        return await client.generate(
            messages=messages,
            # The Anthropic provider family (anthropic, moonshot, zai,
            # kimi_coding, thinking_machines) requires a system message, so the
            # probe must carry one even though it is not the point of the test.
            system=Message(role="system", content=[TextBlock(text="Reply with OK.")]),
            model=model,
            max_completion_tokens=CONNECTION_TEST_MAX_COMPLETION_TOKENS,
            tools=[],
        )


async def test_model_connection(
    provider: ModelProvider,
    model: str,
    *,
    api_key: str = "",
    token_manager: Any = None,
    rate_limits: Optional[RateLimitConfig] = None,
    client_factory: Callable[..., Any] = LLMClient,
    timeout: float = CONNECTION_TEST_TIMEOUT_SECONDS,
    usage_ledger: Any = None,
    base_url: Optional[str] = None,
    api_style: Optional[str] = None,
) -> ModelConnectionResult:
    """Send a tiny no-tool prompt through one provider/model pair.

    This is intentionally opt-in: providers do not expose one uniform, free auth
    endpoint, so a real generation is the only dependable cross-provider probe.

    The target is passed in rather than read off an ``AgentConfig`` so a credential can
    be probed on its own — the Providers screen tests a key for a provider that may not
    be in use, and must keep working when the rest of the model configuration is broken.
    """

    limits = rate_limits or RateLimitConfig()
    try:
        factory_kwargs: dict[str, Any] = {
            "provider": provider,
            "api_key": api_key,
            "max_retries": 0,
            "requests_per_minute": limits.requests_per_minute,
            "tokens_per_minute": limits.tokens_per_minute,
            "token_manager": token_manager,
        }
        # The probe is a real paid request; account for it when the host has a
        # ledger. Omitted when None so injected fake factories keep their shape.
        if usage_ledger is not None:
            factory_kwargs["usage_ledger"] = usage_ledger
        if base_url:
            factory_kwargs["base_url"] = base_url
        if api_style:
            factory_kwargs["api_style"] = api_style
        client = client_factory(**factory_kwargs)
        messages = MessageHistory([Message(role="user", content=[TextBlock(text="Reply with OK.")])])
        await asyncio.wait_for(
            _probe_generate(client, model, messages),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return ModelConnectionResult(False, f"Connection test timed out after {timeout:g} seconds.")
    except LLMError as exc:
        return ModelConnectionResult(False, llm_error_message(exc, model=model))
    except Exception:
        return ModelConnectionResult(
            False, "The connection test failed unexpectedly. Check the provider and try again."
        )
    return ModelConnectionResult(True, f"Connected to {provider.value}/{model}.")
