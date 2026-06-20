import asyncio
from pathlib import Path
from typing import Union

import trafilatura

from kolega_code.config import AgentConfig
from kolega_code.llm.client import LLMClient
from kolega_code.llm.instrumented_client import InstrumentedLLMClient
from kolega_code.llm.models import Message, MessageHistory, TextBlock
from kolega_code.llm.specs import get_model_specs
from .streaming_tool import StreamingTool


class WebFetchTool(StreamingTool):
    """Tool for fetching web page content and delegating lightweight processing to the fast model."""

    FETCH_TIMEOUT_SECONDS = 20
    MAX_CONTENT_CHARS = 100_000
    DEFAULT_RESPONSE_CHAR_LIMIT = 512
    WEB_FETCH_MAX_COMPLETION_TOKENS = 4096

    def __init__(
        self,
        project_path: Union[str, Path],
        workspace_id: str,
        thread_id: str,
        connection_manager,
        config: AgentConfig,
        caller,
        filesystem=None,
    ):
        super().__init__(project_path, workspace_id, thread_id, connection_manager, config, caller, filesystem)

    async def web_fetch(self, url: str, instruction: str) -> str:
        """
        Fetch web content from a URL, process it with the fast model, and return a concise answer.

        This tool downloads the page, extracts clean text via Trafilatura, and asks the fast LLM
        to follow the provided instruction. The model is asked to keep the output compact (≈512
        characters), but the result is only trimmed if it well exceeds that limit.

        Args:
            url: Fully qualified URL to fetch (http/https).
            instruction: Guidance for how the extracted content should be used.

        Returns:
            The model's response derived from the fetched content, truncated to the character limit if necessary.
        """
        if not url or not url.lower().startswith(("http://", "https://")):
            return "Error: Provide a valid http(s) URL."

        tool_call_id = getattr(self.caller, "current_tool_call_id", None)

        if tool_call_id:
            await self.send_streaming_update(
                f"Fetching content from {url}...", tool_call_id, "web_fetch", is_complete=False
            )

        try:
            downloaded_html = await asyncio.wait_for(
                asyncio.to_thread(trafilatura.fetch_url, url), timeout=self.FETCH_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            error_message = f"Error: Timed out fetching {url} after {self.FETCH_TIMEOUT_SECONDS} seconds."
            if tool_call_id:
                await self.send_streaming_update(error_message, tool_call_id, "web_fetch", is_complete=True)
            return error_message
        except Exception as exc:  # pragma: no cover - defensive logging branch
            error_message = f"Error: Failed to fetch {url}: {exc}"
            if tool_call_id:
                await self.send_streaming_update(error_message, tool_call_id, "web_fetch", is_complete=True)
            return error_message

        if not downloaded_html:
            message = f"Error: No content retrieved from {url}."
            if tool_call_id:
                await self.send_streaming_update(message, tool_call_id, "web_fetch", is_complete=True)
            return message

        try:
            extracted_text = await asyncio.to_thread(
                trafilatura.extract,
                downloaded_html,
                include_comments=False,
                include_tables=True,
            )
        except Exception as exc:  # pragma: no cover - defensive logging branch
            error_message = f"Error: Failed to extract content from {url}: {exc}"
            if tool_call_id:
                await self.send_streaming_update(error_message, tool_call_id, "web_fetch", is_complete=True)
            return error_message

        if not extracted_text or not extracted_text.strip():
            message = f"Error: Extracted page content for {url} is empty."
            if tool_call_id:
                await self.send_streaming_update(message, tool_call_id, "web_fetch", is_complete=True)
            return message

        content = extracted_text.strip()
        truncated_note = ""
        if len(content) > self.MAX_CONTENT_CHARS:
            content = content[: self.MAX_CONTENT_CHARS]
            truncated_note = (
                f"\n\n[Web content truncated to first {self.MAX_CONTENT_CHARS} characters to fit token limits.]"
            )

        if tool_call_id:
            await self.send_streaming_update(
                "Processing content with fast model...", tool_call_id, "web_fetch", is_complete=False
            )

        provider = self.config.fast_config.provider
        api_key = self.config.get_api_key(provider)
        rate_limits = self.config.fast_config.rate_limits

        client_kwargs = {
            "provider": provider.value,
            "api_key": api_key,
            "max_retries": rate_limits.max_retries,
            "requests_per_minute": rate_limits.requests_per_minute,
            "tokens_per_minute": rate_limits.tokens_per_minute,
            "token_manager": self.config.get_chatgpt_token_manager(),
        }

        if hasattr(self.caller, "llm") and isinstance(self.caller.llm, InstrumentedLLMClient):
            client = InstrumentedLLMClient(
                langfuse_client=self.caller.llm.langfuse,
                workspace_id=getattr(self.caller, "workspace_id", None),
                thread_id=getattr(self.caller, "thread_id", None),
                agent_type=f"{self.caller.agent_name}-web-fetch",
                environment=self.config.environment,
                user_id=getattr(self.caller, "user_id", None),
                user_email=getattr(self.caller, "user_email", None),
                **client_kwargs,
            )
        else:
            client = LLMClient(**client_kwargs)

        try:
            model_specs = get_model_specs(provider, self.config.fast_config.model)
            max_completion_tokens = min(
                int(model_specs["max_completion_tokens"]),
                self.WEB_FETCH_MAX_COMPLETION_TOKENS,
            )

            target_chars = self.DEFAULT_RESPONSE_CHAR_LIMIT

            system_prompt = Message(
                role="system",
                content=[
                    TextBlock(
                        text=(
                            "You see extracted web page content and an instruction. Follow the instruction faithfully"
                            f" and keep the response around {target_chars} characters when possible—concise but clear."
                            " If more detail is required, stay well-structured and call out when the content is"
                            " insufficient."
                        )
                    )
                ],
            )

            user_prompt = Message(
                role="user",
                content=[
                    TextBlock(
                        text=f"Instruction:\n{instruction.strip()}\n\nWeb content:\n{content}{truncated_note}"
                    )
                ],
            )

            response_message = await client.generate(
                model=self.config.fast_config.model,
                max_completion_tokens=max_completion_tokens,
                system=system_prompt,
                messages=MessageHistory([user_prompt]),
                temperature=0.0,
            )

            response_text = (response_message.get_text_content() or "").strip()
            if not response_text:
                error_message = "Error: Fast model returned an empty response for fetched content."
                if tool_call_id:
                    await self.send_streaming_update(error_message, tool_call_id, "web_fetch", is_complete=True)
                return error_message

            hard_cut_threshold = target_chars * 2
            if len(response_text) > hard_cut_threshold:
                # Prefer trimming on word boundaries to avoid mid-word truncation.
                trimmed = response_text[:target_chars].rstrip()
                cut_index = trimmed.rfind(" ")
                if cut_index > 0:
                    trimmed = trimmed[:cut_index]
                if not trimmed:
                    trimmed = response_text[:target_chars]
                response_text = trimmed.rstrip(" ,.;:-") + "…"

            if tool_call_id:
                await self.send_streaming_update(response_text, tool_call_id, "web_fetch", is_complete=True)

            return response_text
        except Exception as exc:
            error_message = f"Error: Failed to process content with fast model: {exc}"
            if tool_call_id:
                await self.send_streaming_update(error_message, tool_call_id, "web_fetch", is_complete=True)
            return error_message
