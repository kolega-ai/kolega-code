from abc import ABC, abstractmethod
from typing import Any, AsyncContextManager, Dict, List, Optional, Union

from anthropic import APIError as AnthropicAPIError
from google.genai.errors import APIError as GeminiAPIError
from openai import APIError as OpenAIAPIError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..models import Message, MessageHistory, ToolDefinition
from ..ratelimit import RateLimiter
from .models import GenerationParams, ReasoningEffort, ThinkingConfig, TokenCount


class BaseLLMProvider(ABC):
    """Abstract base class defining the interface for LLM providers"""

    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter(requests_per_minute, tokens_per_minute)
        self.base_url = base_url

    @abstractmethod
    async def count_tokens(
        self,
        messages: MessageHistory,
        system: Message = None,
        model: Optional[str] = None,
        tools: List[ToolDefinition] = None,
        **kwargs,
    ) -> TokenCount:
        pass

    @abstractmethod
    def stream(
        self,
        messages: MessageHistory,
        system: Optional[Message] = None,
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> AsyncContextManager:
        pass

    @abstractmethod
    async def generate(
        self, messages: MessageHistory, system: Message = None, params: Optional[GenerationParams] = None, **kwargs
    ) -> Message:
        pass

    def _prepare_generation_params(self, params: Optional[GenerationParams] = None) -> Dict[str, Any]:
        """Convert common parameters to provider-specific format"""
        return {}

    def _prepare_thinking_params(self, thinking: Optional[Union[ThinkingConfig, ReasoningEffort]]) -> Dict[str, Any]:
        """Convert thinking parameters to provider-specific format"""
        return {}

    def get_retry_decorator(self):
        """Get retry decorator with exponential backoff"""
        return retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type((AnthropicAPIError, OpenAIAPIError, GeminiAPIError)),
        )
