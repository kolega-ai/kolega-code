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

from typing import TYPE_CHECKING, Any, AsyncContextManager, Coroutine, Dict, List, Optional, Union

from .exceptions import map_to_llm_error
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
    ):
        self.provider_name = provider.lower()
        self._api_key = api_key  # Store API key privately
        # Refreshing OAuth token manager, used only by the ChatGPT-subscription provider.
        self._token_manager = token_manager
        self.provider = self._initialize_provider(
            provider,
            max_retries=max_retries,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
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
        if p in ("anthropic", "moonshot", "zai", "kimi_coding"):
            from .providers.anthropic import AnthropicProvider

            return AnthropicProvider
        if p == "openai":
            # api-key OpenAI uses the Responses API (gpt-5.x reject tools +
            # reasoning_effort on Chat Completions).
            from .providers.openai_responses import OpenAIResponsesProvider

            return OpenAIResponsesProvider
        if p in ("together", "groq", "fireworks", "llama", "xai", "dashscope", "deepseek", "ollama_cloud"):
            from .providers.openai import OpenAIProvider

            return OpenAIProvider
        if p == "google":
            from .providers.google import GoogleProvider

            return GoogleProvider
        return None

    def _initialize_provider(
        self,
        provider: str,
        max_retries: int = 3,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
    ) -> "Union[AnthropicProvider, OpenAIProvider, GoogleProvider]":
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
        **kwargs: Dict[str, Any],
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

    async def generate(
        self,
        messages: MessageHistory,
        system: Optional[Message] = None,
        temperature: float = 1.0,
        max_completion_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        thinking: Optional[Union[int, str]] = None,
        params: Optional[GenerationParams] = None,
        **kwargs: Dict[str, Any],
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
                )
            return await self.provider.generate(messages, system, params, **kwargs)
        except Exception as e:
            raise map_to_llm_error(e, self.provider_name) from e

    def stream(
        self,
        messages: MessageHistory,
        system: Optional[Message] = None,
        temperature: float = 1.0,
        max_completion_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        thinking: Optional[Union[int, str]] = None,
        params: Optional[GenerationParams] = None,
        **kwargs: Dict[str, Any],
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
                )

            # Return the appropriate stream type for the provider
            return self.provider.stream(messages, system, params, **kwargs)
        except Exception as e:
            raise map_to_llm_error(e, self.provider_name) from e
