"""
Instrumented LLM client that adds Langfuse tracing to all LLM operations.
"""

import asyncio
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional, List, Dict, Union, AsyncContextManager, Coroutine
from datetime import datetime, timezone
import logging

if TYPE_CHECKING:
    # Type hints only — langfuse is a heavy optional dependency. The instrumented
    # client operates on a Langfuse *instance* passed in as langfuse_client.
    from langfuse import Langfuse

from .client import LLMClient
from .models import Message, MessageHistory, ToolDefinition
from .providers.models import GenerationParams

# The provider-family vocabulary lives in kolega_code/llm/usage.py, shared with
# normalized usage attachment so both consumers stay covered by one guard test.
from .usage import normalize_usage, usage_token_fields as _usage_token_fields

logger = logging.getLogger(__name__)


def _end_observations(generation: Any, trace: Any, **updates: Any) -> None:
    trace_updates = {key: updates[key] for key in ("level", "status_message") if key in updates}
    for observation, values in ((generation, updates), (trace, trace_updates)):
        try:
            observation.update(**values)
        except Exception:
            logger.debug("Error updating Langfuse observation", exc_info=True)
        try:
            observation.end()
        except Exception:
            logger.debug("Error ending Langfuse observation", exc_info=True)


def _normalized_langfuse_usage(usage_metadata: Mapping[str, Any], provider: str) -> Optional[Dict[str, Any]]:
    usage = normalize_usage(usage_metadata, provider, None)
    if not usage.reported:
        return None
    details = {"input": usage.input_tokens, "output": usage.output_tokens, "total": usage.total_tokens}
    for source, target in (
        ("cache_read_input_tokens", "cache_read_input_tokens"),
        ("cache_write_input_tokens", "cache_creation_input_tokens"),
        ("reasoning_output_tokens", "reasoning_output_tokens"),
    ):
        value = getattr(usage, source)
        if value is not None:
            details[target] = value
    return details


def get_output_tokens(usage_metadata: Optional[Mapping[str, Any]], provider: Optional[str] = None) -> int:
    """Extract a provider-shaped, non-negative output-token count."""
    if not usage_metadata:
        return 0
    fields = _usage_token_fields(usage_metadata.get("provider", provider), metadata=usage_metadata)
    if fields is None:
        return 0
    value = usage_metadata.get(fields[1], 0)
    return value if type(value) is int and value >= 0 else 0


class InstrumentedLLMClient(LLMClient):
    """LLMClient with Langfuse instrumentation for observability."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        max_retries: int = 3,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        langfuse_client: Optional["Langfuse"] = None,
        workspace_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        environment: Optional[str] = None,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        usage_recorder: Optional[Any] = None,
        token_manager: Optional[Any] = None,
        model: Optional[str] = None,
        usage_ledger: Optional[Any] = None,
        trace_sink: Optional[Any] = None,
        context_window_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        base_url: Optional[str] = None,
        api_style: Optional[str] = None,
        default_headers: Optional[Mapping[str, str]] = None,
    ):
        super().__init__(
            provider,
            api_key,
            max_retries,
            requests_per_minute,
            tokens_per_minute,
            token_manager,
            model=model,
            usage_ledger=usage_ledger,
            trace_sink=trace_sink,
            context_window_tokens=context_window_tokens,
            max_output_tokens=max_output_tokens,
            base_url=base_url,
            api_style=api_style,
            default_headers=default_headers,
        )
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

    def _trace_attributes(self, tags: List[Optional[str]]) -> Any:
        """Trace-level attributes for the active Kolega span and its children.

        Langfuse v4 removed ``span.update_trace()``: user, session, tags and
        environment are applied through the current observation context.
        """
        from langfuse import propagate_attributes

        return propagate_attributes(
            user_id=self.user_id or self.workspace_id,  # Actual user_id when available, else workspace
            session_id=f"{self.workspace_id}/{self.thread_id}",
            tags=[tag for tag in tags if tag is not None],
            environment=self.environment,
        )

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

        fields = _usage_token_fields(provider, metadata=usage_metadata)
        if fields is None:
            logger.warning(f"Unknown provider for usage recording: {provider}")
            return None
        input_tokens = usage_metadata.get(fields[0], 0)
        output_tokens = usage_metadata.get(fields[1], 0)
        cache_read_tokens = usage_metadata.get("cache_read_input_tokens", 0)
        cache_write_tokens = usage_metadata.get("cache_write_input_tokens", 0)

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
        tools: Optional[List[ToolDefinition]] = None,
        thinking: Optional[Union[int, str]] = None,
        params: Optional[GenerationParams] = None,
        *,
        _precomputed_input_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Message:
        """Generate with Langfuse tracing."""
        if not self.langfuse:
            # Fallback to non-instrumented if Langfuse not configured
            return await super().generate(
                messages,
                system,
                temperature,
                max_completion_tokens,
                tools,
                thinking,
                params,
                _precomputed_input_tokens=_precomputed_input_tokens,
                **kwargs,
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

        model_parameters: Dict[str, Any] = {
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "provider": self.provider_name,
        }

        # Make the Kolega span current before propagating attributes so an
        # enclosing host observation is never relabelled with Kolega metadata.
        with self.langfuse.start_as_current_observation(
            name=f"{self.agent_type or 'agent'}-llm-call",
            as_type="span",
            input=input_data,
            metadata=metadata,
            end_on_exit=False,
        ) as trace:
            with self._trace_attributes(
                [
                    self.environment,
                    f"workspace:{self.workspace_id}",
                    f"thread:{self.thread_id}",
                    f"agent:{self.agent_type}",
                    f"provider:{self.provider_name}",
                    f"user:{self.user_id}" if self.user_id else None,
                ]
            ):
                generation = trace.start_observation(
                    name=f"{self.agent_type or 'agent'}-llm-generation",
                    as_type="generation",
                    model=model,
                    model_parameters=model_parameters,
                    input=input_data,
                    metadata=metadata,
                )

        try:
            # Call parent generate method
            response = await super().generate(
                messages,
                system,
                temperature,
                max_completion_tokens,
                tools,
                thinking,
                params,
                _precomputed_input_tokens=_precomputed_input_tokens,
                **kwargs,
            )

            # Extract token usage from response
            usage_details = self._extract_usage_details(response)

            # Normalize usage data to Langfuse format
            normalized_usage = None
            if usage_details:
                provider = usage_details.get("provider", self.provider_name)
                if provider == "refuseless":
                    normalized_usage = _normalized_langfuse_usage(usage_details, provider)
                elif provider in ["anthropic", "moonshot", "minimax", "kimi_coding", "thinking_machines", "tinker"]:
                    normalized_usage = {
                        "input": usage_details.get("input_tokens", 0),
                        "output": usage_details.get("output_tokens", 0),
                        "total": usage_details.get("input_tokens", 0) + usage_details.get("output_tokens", 0),
                        "cache_read_input_tokens": usage_details.get("cache_read_input_tokens", 0),
                        "cache_creation_input_tokens": usage_details.get("cache_write_input_tokens", 0),
                    }
                elif provider in [
                    "openai",
                    "openai_chatgpt",
                    "fireworks",
                    "deepseek",
                    "mistral",
                    "cloudflare_workers_ai",
                ]:
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
        tools: Optional[List[ToolDefinition]] = None,
        thinking: Optional[Union[int, str]] = None,
        params: Optional[GenerationParams] = None,
        *,
        _precomputed_input_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[AsyncContextManager[Any], Coroutine[Any, Any, AsyncContextManager[Any]]]:
        """Stream a response with Langfuse tracing"""
        if not self.langfuse:
            # Fallback to non-instrumented if Langfuse not configured
            return super().stream(
                messages,
                system,
                temperature,
                max_completion_tokens,
                tools,
                thinking,
                params,
                _precomputed_input_tokens=_precomputed_input_tokens,
                **kwargs,
            )

        # ``self.langfuse`` is guaranteed non-None here (we returned early above).
        # Capture it in a local so the closure below sees a non-optional type.
        langfuse = self.langfuse

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

            model_parameters: Dict[str, Any] = {
                "temperature": temperature,
                "max_completion_tokens": max_completion_tokens,
                "streaming": True,
                "provider": self.provider_name,
            }

            # Keep attribute propagation scoped to the Kolega span, not any
            # host observation that was current when stream creation began.
            with langfuse.start_as_current_observation(
                name=f"{self.agent_type or 'agent'}-llm-stream",
                as_type="span",
                input=input_data,
                metadata=metadata,
                end_on_exit=False,
            ) as trace:
                with self._trace_attributes(
                    [
                        self.environment,
                        f"workspace:{self.workspace_id}",
                        f"thread:{self.thread_id}",
                        f"agent:{self.agent_type}",
                        f"provider:{self.provider_name}",
                        "streaming",
                        f"user:{self.user_id}" if self.user_id else None,
                    ]
                ):
                    generation = trace.start_observation(
                        name=f"{self.agent_type or 'agent'}-llm-stream-generation",
                        as_type="generation",
                        model=model,
                        model_parameters=model_parameters,
                        input=input_data,
                        metadata=metadata,
                    )

            try:
                stream = LLMClient.stream(
                    self,
                    messages,
                    system,
                    temperature,
                    max_completion_tokens,
                    tools,
                    thinking,
                    params,
                    _precomputed_input_tokens=_precomputed_input_tokens,
                    **kwargs,
                )

                import inspect

                if inspect.iscoroutine(stream):
                    stream = await stream
            except BaseException as error:
                _end_observations(generation, trace, level="ERROR", status_message=str(error) or type(error).__name__)
                raise

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
        self._exited: bool = False

    async def __aenter__(self):
        try:
            await self.stream.__aenter__()
        except BaseException as error:
            await self.__aexit__(type(error), error, error.__traceback__)
            raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._exited:
            return False
        self._exited = True
        try:
            await self.stream.__aexit__(exc_type, exc_val, exc_tb)
        except BaseException as error:
            if exc_val is not None and not isinstance(error, asyncio.CancelledError):
                logger.debug("Stream cleanup failed while handling an exception", exc_info=True)
                await self._finalize(exc_type, exc_val)
            else:
                await self._finalize(type(error), error)
                raise
        else:
            await self._finalize(exc_type, exc_val)
        return False

    async def _finalize(self, exc_type: Optional[type[BaseException]], exc_val: Optional[BaseException]) -> None:
        output = None
        usage = None

        try:
            if hasattr(self.stream, "get_final_message"):
                final_message = await self.stream.get_final_message()

                # The Message already has usage_metadata populated by from_anthropic/from_openai
                output = final_message.get_text_content() if hasattr(final_message, "get_text_content") else None
                usage = self._extract_langfuse_usage(final_message)
        except asyncio.CancelledError as error:
            _end_observations(
                self.generation, self.trace, level="ERROR", status_message=str(error) or type(error).__name__
            )
            raise
        except Exception as e:
            logger.debug(f"Error getting final message: {e}")

        # Update generation with available data
        gen_update_kwargs: Dict[str, Any] = {
            "level": "ERROR" if exc_type else "DEFAULT",
            "status_message": (str(exc_val) or type(exc_val).__name__) if exc_val is not None else "Success",
        }

        if output is not None:
            gen_update_kwargs["output"] = output
        if usage is not None:
            gen_update_kwargs["usage_details"] = usage

        _end_observations(self.generation, self.trace, **gen_update_kwargs)

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
                        error_message=(str(exc_val) or type(exc_val).__name__) if exc_val is not None else None,
                    )
        except Exception as e:
            logger.debug(f"Error recording usage for stream: {e}")

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

        if provider == "refuseless":
            return _normalized_langfuse_usage(usage_metadata, provider)
        elif provider in ["anthropic", "moonshot", "minimax", "kimi_coding"]:
            return {
                "input": usage_metadata.get("input_tokens", 0),
                "output": usage_metadata.get("output_tokens", 0),
                "total": usage_metadata.get("input_tokens", 0) + usage_metadata.get("output_tokens", 0),
                "cache_read_input_tokens": usage_metadata.get("cache_read_input_tokens", 0),
                "cache_creation_input_tokens": usage_metadata.get("cache_write_input_tokens", 0),
            }
        elif provider in ["openai", "openai_chatgpt", "fireworks", "deepseek", "mistral", "cloudflare_workers_ai"]:
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
