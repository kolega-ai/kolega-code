"""
Instrumented LLM client that adds Langfuse tracing to all LLM operations.
"""

import os
from typing import Any, Optional, List, Dict, Union, AsyncContextManager, Coroutine
from datetime import datetime, timezone
import logging

from langfuse import Langfuse

from .client import LLMClient
from .models import Message, MessageHistory
from .providers.models import GenerationParams

logger = logging.getLogger(__name__)


class InstrumentedLLMClient(LLMClient):
    """LLMClient with Langfuse instrumentation for observability."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        max_retries: int = 3,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        langfuse_client: Optional[Langfuse] = None,
        workspace_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        environment: Optional[str] = None,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        usage_recorder: Optional[Any] = None,
        token_manager: Optional[Any] = None,
    ):
        super().__init__(provider, api_key, max_retries, requests_per_minute, tokens_per_minute, token_manager)
        self.langfuse = langfuse_client
        self.workspace_id = workspace_id
        self.thread_id = thread_id
        self.agent_type = agent_type
        self.environment = environment or os.environ.get("ENVIRONMENT", "development")
        self.user_id = user_id
        self.user_email = user_email
        self.usage_recorder = usage_recorder

    def _create_generation_metadata(self, **kwargs) -> Dict[str, Any]:
        """Create metadata for Langfuse generation."""
        metadata = {
            "provider": self.provider_name,
            "workspace_id": self.workspace_id,
            "thread_id": self.thread_id,
            "agent_type": self.agent_type,
            "environment": self.environment,
            "user_id": self.user_id,
            "user_email": self.user_email,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Add any additional kwargs as metadata
        for key, value in kwargs.items():
            if key not in ["messages", "system", "params", "model"]:
                metadata[key] = value

        return metadata

    def _extract_usage_details(self, response: Message) -> Dict[str, Any]:
        """Extract usage details from provider response"""
        if not response or not hasattr(response, "usage_metadata"):
            return {}

        return response.usage_metadata

    def _normalize_usage_data(
        self, usage_metadata: Dict[str, Any], model: str, success: bool = True, error_message: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Normalize provider usage metadata for host-provided usage recorders.

        Args:
            usage_metadata: Usage metadata from LLM response
            model: Model name used
            success: Whether the request was successful
            error_message: Error message if request failed
        """
        if not usage_metadata:
            return None

        provider = usage_metadata.get("provider", self.provider_name)

        if provider in ["anthropic", "moonshot", "deepseek", "kimi_coding"]:
            input_tokens = usage_metadata.get("input_tokens", 0)
            output_tokens = usage_metadata.get("output_tokens", 0)
            cache_read_tokens = usage_metadata.get("cache_read_input_tokens", 0)
            cache_write_tokens = usage_metadata.get("cache_write_input_tokens", 0)
        elif provider in ["openai", "openai_chatgpt", "together", "groq", "fireworks", "llama", "xai", "dashscope"]:
            input_tokens = usage_metadata.get("prompt_tokens", 0)
            output_tokens = usage_metadata.get("completion_tokens", 0)
            cache_read_tokens = usage_metadata.get("cache_read_input_tokens", 0)
            cache_write_tokens = usage_metadata.get("cache_write_input_tokens", 0)
        elif provider == "google":
            input_tokens = usage_metadata.get("prompt_token_count", 0)
            output_tokens = usage_metadata.get("candidates_token_count", 0)
            cache_read_tokens = usage_metadata.get("cache_read_input_tokens", 0)
            cache_write_tokens = usage_metadata.get("cache_write_input_tokens", 0)
        else:
            logger.warning(f"Unknown provider for usage recording: {provider}")
            return None

        return {
            "user_id": self.user_id,
            "workspace_id": self.workspace_id,
            "thread_id": self.thread_id,
            "agent_type": self.agent_type,
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read_tokens,
            "cache_write_input_tokens": cache_write_tokens,
            "success": success,
            "error_message": error_message,
            "timestamp": datetime.now(timezone.utc),
            "metadata": {
                "environment": self.environment,
                "raw_usage": usage_metadata,
            },
        }

    async def _record_usage(
        self, usage_metadata: Dict[str, Any], model: str, success: bool = True, error_message: Optional[str] = None
    ) -> None:
        """Record token usage through a host-provided recorder, when configured."""
        if os.environ.get("DISABLE_USAGE_RECORDING"):
            logger.debug("Usage recording disabled by DISABLE_USAGE_RECORDING env var")
            return

        if not self.usage_recorder:
            return

        usage_data = self._normalize_usage_data(usage_metadata, model, success, error_message)
        if not usage_data:
            return

        try:
            if hasattr(self.usage_recorder, "record_usage"):
                result = self.usage_recorder.record_usage(usage_data)
            elif callable(self.usage_recorder):
                result = self.usage_recorder(usage_data)
            else:
                logger.warning("Usage recorder is not callable and has no record_usage method")
                return

            import inspect

            if inspect.isawaitable(result):
                await result
        except Exception as e:
            logger.warning(f"Failed to record token usage: {e}")

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
        """Generate with Langfuse tracing."""
        if not self.langfuse:
            # Fallback to non-instrumented if Langfuse not configured
            return await super().generate(
                messages, system, temperature, max_completion_tokens, tools, thinking, params, **kwargs
            )

        # Extract model from kwargs
        model = kwargs.get("model", "unknown")

        # Format input for Langfuse
        input_data = {
            "messages": [msg.to_dict() for msg in messages],
            "system": system.to_dict() if system else None,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "tools": tools,
        }

        # Create metadata for the generation
        metadata = self._create_generation_metadata(**kwargs)

        # Create trace first (v3 API)
        trace = self.langfuse.start_span(
            name=f"{self.agent_type or 'agent'}-llm-call",
            input=input_data,
            metadata=metadata,
        )

        # Create session name with user context
        session_name = f"{self.workspace_id}/{self.thread_id}"

        # Update trace with attributes
        trace.update_trace(
            user_id=self.user_id or self.workspace_id,  # Use actual user_id if available, fallback to workspace
            session_id=session_name,
            tags=[
                tag
                for tag in [
                    self.environment,
                    f"workspace:{self.workspace_id}",
                    f"thread:{self.thread_id}",
                    f"agent:{self.agent_type}",
                    f"provider:{self.provider_name}",
                    f"user:{self.user_id}" if self.user_id else None,
                ]
                if tag is not None
            ],
        )

        # Create generation as child of trace
        generation = trace.start_generation(
            name=f"{self.agent_type or 'agent'}-llm-generation",
            model=model,
            model_parameters={
                "temperature": temperature,
                "max_completion_tokens": max_completion_tokens,
                "provider": self.provider_name,
            },
            input=input_data,
            metadata=metadata,
        )

        try:
            # Call parent generate method
            response = await super().generate(
                messages, system, temperature, max_completion_tokens, tools, thinking, params, **kwargs
            )

            # Extract token usage from response
            usage_details = self._extract_usage_details(response)

            # Normalize usage data to Langfuse format
            normalized_usage = None
            if usage_details:
                provider = usage_details.get("provider", self.provider_name)
                if provider in ["anthropic", "moonshot", "deepseek", "kimi_coding"]:
                    normalized_usage = {
                        "input": usage_details.get("input_tokens", 0),
                        "output": usage_details.get("output_tokens", 0),
                        "total": usage_details.get("input_tokens", 0) + usage_details.get("output_tokens", 0),
                        "cache_read_input_tokens": usage_details.get("cache_read_input_tokens", 0),
                        "cache_creation_input_tokens": usage_details.get("cache_write_input_tokens", 0),
                    }
                elif provider in ["openai", "openai_chatgpt", "fireworks"]:
                    normalized_usage = {
                        "input": usage_details.get("prompt_tokens", 0),
                        "output": usage_details.get("completion_tokens", 0),
                        "total": usage_details.get("total_tokens", 0),
                        "cache_read_input_tokens": usage_details.get("cache_read_input_tokens", 0),
                        "cache_creation_input_tokens": usage_details.get("cache_write_input_tokens", 0),
                    }
                elif provider == "google":
                    normalized_usage = {
                        "input": usage_details.get("prompt_token_count", 0),
                        "output": usage_details.get("candidates_token_count", 0),
                        "total": usage_details.get("total_token_count", 0),
                        "cache_read_input_tokens": usage_details.get("cache_read_input_tokens", 0),
                        "cache_creation_input_tokens": usage_details.get("cache_write_input_tokens", 0),
                    }

            # Update generation with success
            generation.update(
                output=response.to_dict(),
                usage_details=normalized_usage,
                level="DEFAULT",
                status_message="Success",
            )
            generation.end()

            # End the trace
            trace.end()

            await self._record_usage(usage_details, model, success=True)

            return response

        except Exception as e:
            # Update generation with error
            generation.update(
                level="ERROR",
                status_message=str(e),
            )
            generation.end()
            # End the trace
            trace.update(level="ERROR", status_message=str(e))
            trace.end()
            raise

    def stream(
        self,
        messages: MessageHistory,
        system: Optional[Message] = None,
        temperature: float = 1.0,
        max_completion_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        thinking: Optional[Union[int, str]] = None,
        params: Optional[GenerationParams] = None,
        **kwargs,
    ) -> Union[AsyncContextManager[Any], Coroutine[Any, Any, AsyncContextManager[Any]]]:
        """Stream a response with Langfuse tracing"""
        if not self.langfuse:
            # Fallback to non-instrumented if Langfuse not configured
            return super().stream(
                messages, system, temperature, max_completion_tokens, tools, thinking, params, **kwargs
            )

        # Since we need to create langfuse metadata synchronously but the stream
        # might be a coroutine, we return a coroutine that creates the wrapper
        async def create_instrumented_stream():
            # Extract model from kwargs
            model = kwargs.get("model", "unknown")

            # Format input for Langfuse (same format as generate method)
            input_data = {
                "messages": [msg.to_dict() for msg in messages],
                "system": system.to_dict() if system else None,
                "temperature": temperature,
                "max_completion_tokens": max_completion_tokens,
                "tools": tools,
            }

            # Create metadata for the generation
            metadata = self._create_generation_metadata(**kwargs)

            # Create trace first (v3 API)
            trace = self.langfuse.start_span(
                name=f"{self.agent_type or 'agent'}-llm-stream",
                input=input_data,
                metadata=metadata,
            )

            # Create session name with user context
            session_name = f"{self.workspace_id}/{self.thread_id}"

            # Update trace with attributes
            trace.update_trace(
                user_id=self.user_id or self.workspace_id,  # Use actual user_id if available, fallback to workspace
                session_id=session_name,
                tags=[
                    tag
                    for tag in [
                        self.environment,
                        f"workspace:{self.workspace_id}",
                        f"thread:{self.thread_id}",
                        f"agent:{self.agent_type}",
                        f"provider:{self.provider_name}",
                        "streaming",
                        f"user:{self.user_id}" if self.user_id else None,
                    ]
                    if tag is not None
                ],
            )

            # Create generation as child of trace
            generation = trace.start_generation(
                name=f"{self.agent_type or 'agent'}-llm-stream-generation",
                model=model,
                model_parameters={
                    "temperature": temperature,
                    "max_completion_tokens": max_completion_tokens,
                    "streaming": True,
                    "provider": self.provider_name,
                },
                input=input_data,
                metadata=metadata,
            )

            # Get stream from underlying client
            stream = LLMClient.stream(
                self, messages, system, temperature, max_completion_tokens, tools, thinking, params, **kwargs
            )

            # Check if stream is a coroutine (needs to be awaited)
            import inspect

            if inspect.iscoroutine(stream):
                stream = await stream

            # Wrap with minimal instrumented wrapper
            return MinimalLangfuseStreamWrapper(stream, generation, trace, self, model)

        return create_instrumented_stream()


class MinimalLangfuseStreamWrapper:
    """Minimal wrapper for any provider's stream with Langfuse tracing"""

    def __init__(self, stream, generation, trace, instrumented_client, model):
        self.stream = stream
        self.generation = generation
        self.trace = trace
        self.instrumented_client = instrumented_client
        self.model = model

    async def __aenter__(self):
        # Enter the underlying stream's context
        await self.stream.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Let stream clean up first
        await self.stream.__aexit__(exc_type, exc_val, exc_tb)

        # Try to get final message for usage data
        output = None
        usage = None

        try:
            if hasattr(self.stream, "get_final_message"):
                final_message = await self.stream.get_final_message()

                # The Message already has usage_metadata populated by from_anthropic/from_openai
                output = final_message.get_text_content() if hasattr(final_message, "get_text_content") else None
                usage = self._extract_langfuse_usage(final_message)
        except Exception as e:
            logger.debug(f"Error getting final message: {e}")

        # Update generation with available data
        gen_update_kwargs = {
            "level": "ERROR" if exc_type else "DEFAULT",
            "status_message": str(exc_val) if exc_val else "Success",
        }

        if output is not None:
            gen_update_kwargs["output"] = output
        if usage is not None:
            gen_update_kwargs["usage_details"] = usage

        try:
            self.generation.update(**gen_update_kwargs)
            self.generation.end()

            # Update and end trace
            self.trace.update(
                level="ERROR" if exc_type else "DEFAULT",
                status_message=str(exc_val) if exc_val else "Success",
            )
            self.trace.end()
        except Exception as ex:
            logger.debug(f"Error updating langfuse generation: {ex}")

        # Record usage for streaming response
        try:
            if hasattr(self.stream, "get_final_message"):
                final_message = await self.stream.get_final_message()
                if hasattr(final_message, "usage_metadata") and final_message.usage_metadata:
                    # Get a mutable copy of usage metadata
                    usage_metadata = dict(final_message.usage_metadata)
                    await self.instrumented_client._record_usage(
                        usage_metadata,
                        self.model,
                        success=(exc_type is None),
                        error_message=str(exc_val) if exc_val else None,
                    )
        except Exception as e:
            logger.debug(f"Error recording usage for stream: {e}")

        return False  # Don't suppress exceptions

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self.stream.__anext__()
            return chunk
        except StopAsyncIteration:
            raise

    async def get_final_message(self):
        """Delegate to underlying stream's get_final_message method."""
        if hasattr(self.stream, "get_final_message"):
            return await self.stream.get_final_message()
        else:
            raise AttributeError(f"Underlying stream {type(self.stream).__name__} has no attribute 'get_final_message'")

    def _extract_langfuse_usage(self, message: Message) -> Optional[Dict[str, Any]]:
        """Extract and normalize usage data for Langfuse from message."""
        if not hasattr(message, "usage_metadata") or not message.usage_metadata:
            return None

        usage_metadata = message.usage_metadata
        provider = usage_metadata.get("provider", "")

        if provider in ["anthropic", "moonshot", "deepseek", "kimi_coding"]:
            return {
                "input": usage_metadata.get("input_tokens", 0),
                "output": usage_metadata.get("output_tokens", 0),
                "total": usage_metadata.get("input_tokens", 0) + usage_metadata.get("output_tokens", 0),
                "cache_read_input_tokens": usage_metadata.get("cache_read_input_tokens", 0),
                "cache_creation_input_tokens": usage_metadata.get("cache_write_input_tokens", 0),
            }
        elif provider in ["openai", "openai_chatgpt", "fireworks"]:
            return {
                "input": usage_metadata.get("prompt_tokens", 0),
                "output": usage_metadata.get("completion_tokens", 0),
                "total": usage_metadata.get("total_tokens", 0),
                "cache_read_input_tokens": usage_metadata.get("cache_read_input_tokens", 0),
                "cache_creation_input_tokens": usage_metadata.get("cache_write_input_tokens", 0),
            }
        elif provider == "google":
            return {
                "input": usage_metadata.get("prompt_token_count", 0),
                "output": usage_metadata.get("candidates_token_count", 0),
                "total": usage_metadata.get("total_token_count", 0),
            }

        return None
