"""Client library for interacting with Large Language Model (LLM) providers.

This module provides a unified interface for making requests to various LLM services
including Anthropic, OpenAI, and Google. The main class LLMClient handles:

- Provider-specific API initialization and authentication
- Rate limiting and retry logic
- Message formatting and parsing
- Streaming and non-streaming completions
- Token counting and budget management
- Tool/function calling capabilities

The client abstracts away provider differences to give applications a clean, consistent
API for using any supported LLM service interchangeably.

Example:
    client = LLMClient(
        provider='openai',
        api_key='sk-...',
        max_retries=3,
        requests_per_minute=60
    )

    response = await client.generate(
        messages=message_history,
        system=system_message,
        temperature=0.7
    )

The module also provides supporting classes and types for working with messages,
tools, and provider-specific parameters in a standardized way.
"""

import dataclasses
import inspect
from typing import TYPE_CHECKING, Any, AsyncContextManager, Coroutine, Dict, List, Optional, Union

from .exceptions import LLMContextWindowExceededError, map_to_llm_error
from .ledger import LedgerStreamAdapter, UsageLedger, describe_error
from .models import Message, MessageHistory, ToolDefinition
from .providers.models import GenerationParams, TokenCount
from .specs import validate_thinking_effort
from kolega_code.auth import constants as chatgpt_constants

if TYPE_CHECKING:
    # Provider classes are imported lazily in _initialize_provider so a session
    # only loads the vendor SDK for the provider it actually uses (each provider
    # module imports its own SDK at module load).
    from .providers.anthropic import AnthropicProvider
    from .providers.google import GoogleProvider
    from .providers.openai import OpenAIProvider
    from .providers.tinker import TinkerProvider


class LLMClient:
    """A unified client for interacting with different LLM providers.

    This class provides a consistent interface for making requests to various LLM providers
    including Anthropic, OpenAI, Google, and others. It handles:

    - Provider-specific API initialization and authentication
    - Rate limiting and retry logic
    - Message formatting and parsing
    - Streaming and non-streaming completions
    - Token counting and budget management
    - Tool/function calling capabilities

    The client abstracts away provider differences to give a clean, unified API for
    applications to use any supported LLM service interchangeably.
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        max_retries: int = 3,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        token_manager: Optional[Any] = None,
        model: Optional[str] = None,
        usage_ledger: Optional[UsageLedger] = None,
        trace_sink: Optional[Any] = None,
        context_window_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        base_url: Optional[str] = None,
        api_style: Optional[str] = None,
    ):
        if (context_window_tokens is None) != (max_output_tokens is None):
            raise ValueError("context_window_tokens and max_output_tokens must be supplied together")
        self.provider_name = provider.lower()
        self._api_key = api_key  # Store API key privately
        # Refreshing OAuth token manager, used only by the ChatGPT-subscription provider.
        self._token_manager = token_manager
        # Strict per-run context budget (the paired AgentConfig fields, propagated
        # by AgentContext.create_llm_client). When set, generate()/stream() cap the
        # output maximum at max_output_tokens and reject, before any provider call,
        # a request whose counted input exceeds context_window_tokens - output.
        self._context_window_tokens = context_window_tokens
        self._max_output_tokens = max_output_tokens
        # Optional structured trace sink for the native Tinker provider (the
        # on-policy RL trajectory record). Ignored by every other provider.
        self._trace_sink = trace_sink
        # Process-wide usage accounting; every invocation on this client settles
        # into it exactly once (see kolega_code/llm/ledger.py). None disables it.
        self._usage_ledger = usage_ledger
        # Routing hint only: nearly every model shares its provider's API surface, so
        # the client is provider-scoped and the model id is passed per call. The one
        # exception is deepseek-v4-flash (Responses API vs the deepseek Chat default),
        # which must be known at construction to pick the right provider class. The
        # per-call `model=` kwarg still governs the request body.
        self._model = model
        self.provider = self._initialize_provider(
            provider,
            max_retries=max_retries,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            base_url=base_url,
            api_style=api_style,
        )

    @staticmethod
    def _provider_class(provider: str):
        """Import and return the provider class for ``provider`` (lazy).

        Each provider module imports its own vendor SDK at module load, so we
        import only the one this session uses. This keeps the unused vendor SDKs
        (each tens of MB) out of the process. Returns ``None`` for an unknown
        provider so the caller can raise a clear error.
        """
        p = provider.lower()
        if p in ("anthropic", "moonshot", "zai", "kimi_coding", "thinking_machines"):
            from .providers.anthropic import AnthropicProvider

            return AnthropicProvider
        if p == "openai":
            # api-key OpenAI uses the Responses API (gpt-5.x reject tools +
            # reasoning_effort on Chat Completions).
            from .providers.openai_responses import OpenAIResponsesProvider

            return OpenAIResponsesProvider
        if p in (
            "together",
            "groq",
            "fireworks",
            "llama",
            "xai",
            "dashscope",
            "deepseek",
            "ollama_cloud",
            "openrouter",
        ):
            from .providers.openai import OpenAIProvider

            return OpenAIProvider
        if p == "google":
            from .providers.google import GoogleProvider

            return GoogleProvider
        if p == "tinker":
            from .providers.tinker import TinkerProvider

            return TinkerProvider
        return None

    def _initialize_provider(
        self,
        provider: str,
        max_retries: int = 3,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        base_url: Optional[str] = None,
        api_style: Optional[str] = None,
    ) -> "Union[AnthropicProvider, OpenAIProvider, GoogleProvider, TinkerProvider]":
        """Initialize the appropriate LLM provider based on the provider name.

        Args:
            provider (str): Name of the LLM provider to initialize (e.g. 'anthropic', 'openai', 'google')
            max_retries (int, optional): Maximum number of retries for failed API calls. Defaults to 3.
            requests_per_minute (int, optional): Maximum number of requests allowed per minute. Defaults to None.
            tokens_per_minute (int, optional): Maximum number of tokens allowed per minute. Defaults to None.

        Returns:
            Union[AnthropicProvider, OpenAIProvider, GoogleProvider]: Initialized provider instance

        Raises:
            LLMError: If an unsupported provider name is specified or initialization fails
        """
        try:
            # Custom endpoints: the provider value is "custom:<id>" and the wire
            # dialect comes from api_style, resolved by the config layer.
            if provider.lower().startswith("custom:"):
                if not base_url or not api_style:
                    raise ValueError(f"Custom endpoint provider {provider} requires base_url and api_style.")
                if api_style == "openai_chat":
                    from .providers.openai import OpenAIProvider as CustomProviderClass
                elif api_style == "openai_responses":
                    from .providers.openai_responses import OpenAIResponsesProvider as CustomProviderClass
                elif api_style == "anthropic":
                    from .providers.anthropic import AnthropicProvider as CustomProviderClass
                else:
                    raise ValueError(f"Unsupported api_style {api_style!r} for custom endpoint {provider}.")
                return CustomProviderClass(
                    # Keyless local servers still require a non-empty credential at
                    # SDK construction; a dummy satisfies them (servers ignore it).
                    api_key=self._api_key or "local",
                    max_retries=max_retries,
                    requests_per_minute=requests_per_minute,
                    tokens_per_minute=tokens_per_minute,
                    base_url=base_url,
                    provider_name=provider.lower(),
                )

            # ChatGPT-subscription OAuth provider: distinct base URL + Responses API,
            # authenticated by a refreshing token manager rather than an api key.
            if provider.lower() == chatgpt_constants.PROVIDER_KEY:
                if self._token_manager is None:
                    raise ValueError("ChatGPT provider requires sign-in; run /login chatgpt to sign in.")
                from .providers.chatgpt_oauth import ChatGPTOAuthProvider

                return ChatGPTOAuthProvider(
                    token_manager=self._token_manager,
                    max_retries=max_retries,
                    requests_per_minute=requests_per_minute,
                    tokens_per_minute=tokens_per_minute,
                    base_url=chatgpt_constants.INFERENCE_BASE_URL,
                    provider_name=chatgpt_constants.PROVIDER_KEY,
                )

            # TEMPORARY single exception: deepseek-v4-flash speaks the Responses API
            # (bare host, no /v1) while the rest of the deepseek catalog stays on Chat
            # Completions. Special-cased here — like the ChatGPT provider above —
            # because it needs a distinct base URL. Delete this branch once DeepSeek
            # moves its catalog to Responses (then route "deepseek" wholesale).
            if provider.lower() == "deepseek" and self._model == "deepseek-v4-flash":
                from .providers.deepseek_responses import DeepSeekResponsesProvider

                return DeepSeekResponsesProvider(
                    api_key=self._api_key,
                    max_retries=max_retries,
                    requests_per_minute=requests_per_minute,
                    tokens_per_minute=tokens_per_minute,
                    base_url="https://api.deepseek.com",
                    provider_name="deepseek",
                )

            base_urls: Dict[str, str] = {
                "openai": "https://api.openai.com/v1/",
                "together": "https://api.together.xyz/v1",
                "groq": "https://api.groq.com/openai/v1",
                "fireworks": "https://api.fireworks.ai/inference/v1",
                "llama": "http://localhost:8000/v1",
                "google": "https://generativelanguage.googleapis.com",
                "xai": "https://api.x.ai/v1",
                "dashscope": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                "moonshot": "https://api.moonshot.ai/anthropic",
                "deepseek": "https://api.deepseek.com/v1",
                "zai": "https://api.z.ai/api/anthropic",
                "kimi_coding": "https://api.kimi.com/coding",
                "ollama_cloud": "https://ollama.com/v1",
                "openrouter": "https://openrouter.ai/api/v1",
                # Tinker's Anthropic-compatible endpoint: the Anthropic SDK appends
                # /v1/messages (and /v1/messages/count_tokens) to this base.
                "thinking_machines": "https://tinker.thinkingmachines.dev/services/tinker-prod/anthropic/api",
            }

            provider_class = self._provider_class(provider)
            if not provider_class:
                raise ValueError(f"Unsupported provider: {provider}")

            base_url = base_urls.get(provider.lower())

            # Every provider class except GoogleProvider takes a provider_name (the
            # Anthropic/OpenAI/OpenAIResponses families share a class across several
            # provider keys and need it to disambiguate behavior).
            provider_kwargs = {}
            if provider.lower() != "google":
                provider_kwargs["provider_name"] = provider.lower()
            if provider.lower() == "tinker":
                provider_kwargs["trace_sink"] = self._trace_sink

            return provider_class(
                api_key=self._api_key,
                max_retries=max_retries,
                requests_per_minute=requests_per_minute,
                tokens_per_minute=tokens_per_minute,
                base_url=base_url,
                **provider_kwargs,
            )
        except Exception as e:
            raise map_to_llm_error(e, provider) from e

    async def count_tokens(
        self,
        messages: MessageHistory,
        system: Optional[Message] = None,
        tools: List[ToolDefinition] = [],
        **kwargs: Any,
    ) -> TokenCount:
        """Count tokens for a list of messages and optional system message.

        Args:
            messages (MessageHistory): The message history to count tokens for
            system (Optional[Message]): Optional system message to include in token count
            tools (List[ToolDefinition]): List of tool definitions to include in token count
            **kwargs (Dict[str, Any]): Additional provider-specific arguments

        Returns:
            TokenCount: Object containing input token count and optionally output token count
                       depending on provider capabilities

        Raises:
            LLMError: Any LLM-related error that occurs during token counting
        """
        try:
            model: Optional[str] = str(kwargs.pop("model", None))
            return await self.provider.count_tokens(
                messages=messages, system=system, model=model, tools=tools, **kwargs
            )
        except Exception as e:
            raise map_to_llm_error(e, self.provider_name) from e

    def _prepare_thinking_param(
        self, thinking: Optional[Union[int, str]] = None, model: Optional[str] = None
    ) -> Optional[str]:
        """Validate a model-specific thinking effort value."""
        if thinking is None:
            return None

        if isinstance(thinking, int):
            raise ValueError("Numeric thinking token budgets have been replaced by named thinking effort levels.")
        if not model:
            raise ValueError("A model is required when setting thinking effort.")
        return validate_thinking_effort(self.provider_name, model, thinking)

    @property
    def _has_run_context_budget(self) -> bool:
        return self._context_window_tokens is not None and self._max_output_tokens is not None

    async def _enforce_run_context_budget(
        self,
        messages: MessageHistory,
        system: Optional[Message],
        params: GenerationParams,
        kwargs: Dict[str, Any],
        precomputed_input_tokens: Optional[int],
    ) -> GenerationParams:
        """Apply the strict per-run context budget to one request.

        Returns the params to send: a copy of ``params`` carrying the effective
        output maximum (the smaller of the caller's request and the run-wide cap,
        or the run-wide cap when the caller requested none). The supplied params
        are never mutated. Raises LLMContextWindowExceededError before any
        provider generation when the request's input — the fully rendered
        request, counted with the same messages/system/tools/model and
        renderer-affecting thinking setting — exceeds
        ``context_window_tokens - effective_output`` (equality is allowed).
        ``precomputed_input_tokens`` is the caller's own count of this exact
        request (BaseAgent already counts its main-loop request for the context
        gauge); when given, no second count is performed.
        """
        assert self._context_window_tokens is not None and self._max_output_tokens is not None
        requested = params.max_completion_tokens
        effective_output = self._max_output_tokens if requested is None else min(requested, self._max_output_tokens)

        if precomputed_input_tokens is not None:
            input_tokens = precomputed_input_tokens
        else:
            counted = await self.provider.count_tokens(
                messages=messages,
                system=system,
                model=self._request_model(kwargs),
                tools=params.tools or [],
                thinking=params.thinking,
            )
            input_tokens = counted.input_tokens

        max_input = self._context_window_tokens - effective_output
        if input_tokens > max_input:
            raise LLMContextWindowExceededError(
                f"Request needs {input_tokens} input tokens, exceeding this run's maximum input of "
                f"{max_input} (context window {self._context_window_tokens} tokens minus "
                f"{effective_output} reserved output tokens).",
                provider=self.provider_name,
            )

        if effective_output == params.max_completion_tokens:
            return params
        return dataclasses.replace(params, max_completion_tokens=effective_output)

    async def generate(
        self,
        messages: MessageHistory,
        system: Optional[Message] = None,
        temperature: float = 1.0,
        max_completion_tokens: Optional[int] = None,
        tools: Optional[List[ToolDefinition]] = None,
        thinking: Optional[Union[int, str]] = None,
        params: Optional[GenerationParams] = None,
        hosted_web_search: bool = False,
        *,
        _precomputed_input_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Message:
        """Generate a complete response from the LLM provider.

        Args:
            messages (MessageHistory): The conversation history to generate from
            system (Optional[Message]): Optional system message to prepend
            temperature (float): Sampling temperature, higher is more random (default: 1.0)
            max_completion_tokens (Optional[int]): Maximum tokens to generate in response
            tools (Optional[List[Dict[str, Any]]]): List of tool definitions for function calling
            thinking (Optional[Union[int, str]]): Model-specific thinking effort string.
            params (Optional[GenerationParams]): Override all parameters with a GenerationParams object
            hosted_web_search (bool): Expose the provider's server-side web_search
                tool (Responses APIs only; ignored elsewhere).
            _precomputed_input_tokens: Internal. The caller's own input count for
                this exact request; used by the strict per-run context budget to
                avoid counting the same request twice. Never forwarded to
                providers.
            **kwargs: Additional provider-specific parameters

        Returns:
            Message: The complete generated response message

        Raises:
            LLMError: Any LLM-related error that occurs during generation
        """
        try:
            if params is None:
                params = GenerationParams(
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    tools=tools,
                    thinking=self._prepare_thinking_param(
                        thinking, str(kwargs.get("model")) if kwargs.get("model") else None
                    ),
                    hosted_web_search=hosted_web_search,
                )
            if self._has_run_context_budget:
                params = await self._enforce_run_context_budget(
                    messages, system, params, kwargs, _precomputed_input_tokens
                )
            if self._usage_ledger is None:
                return await self.provider.generate(messages, system, params, **kwargs)
            request_id = self._usage_ledger.begin(self.provider_name, self._request_model(kwargs))
            try:
                message = await self.provider.generate(messages, system, params, **kwargs)
            except BaseException as exc:
                # Includes CancelledError: a cancelled call may already have cost
                # tokens, so it settles as failed; propagation is unchanged (the
                # outer clause maps Exception only).
                self._usage_ledger.record_failure(request_id, describe_error(exc))
                raise
            self._usage_ledger.record_response(request_id, getattr(message, "usage", None), message=message)
            return message
        except Exception as e:
            raise map_to_llm_error(e, self.provider_name) from e

    def stream(
        self,
        messages: MessageHistory,
        system: Optional[Message] = None,
        temperature: float = 1.0,
        max_completion_tokens: Optional[int] = None,
        tools: Optional[List[ToolDefinition]] = None,
        thinking: Optional[Union[int, str]] = None,
        params: Optional[GenerationParams] = None,
        hosted_web_search: bool = False,
        *,
        _precomputed_input_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[AsyncContextManager[Any], Coroutine[Any, Any, AsyncContextManager[Any]]]:
        """Generate a streaming response from the LLM provider.

        Args:
            messages (MessageHistory): The conversation history to generate from
            system (Optional[Message]): Optional system message to prepend
            temperature (float): Sampling temperature, higher is more random (default: 1.0)
            max_completion_tokens (Optional[int]): Maximum tokens to generate in response
            tools (Optional[List[Dict[str, Any]]]): List of tool definitions for function calling
            thinking (Optional[Union[int, str]]): Model-specific thinking effort string.
            params (Optional[GenerationParams]): Override all parameters with a GenerationParams object
            hosted_web_search (bool): Expose the provider's server-side web_search
                tool (Responses APIs only; ignored elsewhere).
            _precomputed_input_tokens: Internal. The caller's own input count for
                this exact request; used by the strict per-run context budget to
                avoid counting the same request twice. Never forwarded to
                providers.
            **kwargs: Additional provider-specific parameters

        Returns:
            AsyncContextManager: A context manager that yields message chunks when streamed

        Raises:
            LLMError: Any LLM-related error that occurs during stream initialization
        """
        try:
            if params is None:
                params = GenerationParams(
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    tools=tools,
                    thinking=self._prepare_thinking_param(
                        thinking, str(kwargs.get("model")) if kwargs.get("model") else None
                    ),
                    hosted_web_search=hosted_web_search,
                )

            if self._has_run_context_budget:
                # The budget check is async (token counting), and stream() is not:
                # defer both into a coroutine so the over-limit error surfaces when
                # the caller awaits — still before any provider generation.
                return self._budgeted_stream(messages, system, params, kwargs, _precomputed_input_tokens)

            # Return the appropriate stream type for the provider
            inner = self.provider.stream(messages, system, params, **kwargs)
            if self._usage_ledger is None:
                return inner
            return self._ledgered_stream(inner, self._request_model(kwargs))
        except Exception as e:
            raise map_to_llm_error(e, self.provider_name) from e

    async def _budgeted_stream(
        self,
        messages: MessageHistory,
        system: Optional[Message],
        params: GenerationParams,
        kwargs: Dict[str, Any],
        precomputed_input_tokens: Optional[int],
    ) -> AsyncContextManager[Any]:
        """Enforce the run context budget, then open the provider stream.

        Runs when the caller awaits the stream — the point the request starts —
        so the over-limit error is raised before any provider generation.
        """
        params = await self._enforce_run_context_budget(messages, system, params, kwargs, precomputed_input_tokens)
        inner = self.provider.stream(messages, system, params, **kwargs)
        if self._usage_ledger is not None:
            return await self._ledgered_stream(inner, self._request_model(kwargs))
        if inspect.iscoroutine(inner):
            return await inner
        return inner

    def _request_model(self, kwargs: Dict[str, Any]) -> Optional[str]:
        model = kwargs.get("model")
        return str(model) if model else self._model

    async def _ledgered_stream(self, inner: Any, model: Optional[str]) -> LedgerStreamAdapter:
        # Runs when the CALLER awaits the stream — the point the request starts.
        assert self._usage_ledger is not None
        request_id = self._usage_ledger.begin(self.provider_name, model)
        try:
            wrapper = await inner
        except BaseException as exc:
            # Stream-setup errors propagate unmapped today; that is preserved.
            self._usage_ledger.record_failure(request_id, describe_error(exc))
            raise
        return LedgerStreamAdapter(wrapper, self._usage_ledger, request_id)
