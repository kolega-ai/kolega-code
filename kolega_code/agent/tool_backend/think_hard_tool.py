from .. import prompts
from kolega_code.llm.client import LLMClient
from kolega_code.llm.instrumented_client import InstrumentedLLMClient
from kolega_code.llm.models import Message, MessageHistory, TextBlock, ThinkingBlock
from kolega_code.llm.specs import get_model_specs
from .streaming_tool import StreamingTool


class ThinkHardTool(StreamingTool):
    async def think_hard(self, problem_statement: str) -> str:
        """
        Uses Claude 3.7 Sonnet in extended thinking mode to analyze a problem deeply.

        This tool leverages Claude's extended thinking capabilities to perform in-depth
        analysis on complex problems. It sends the problem statement to the Claude API
        with specific parameters to enable extended thinking and returns the detailed response.

        Args:
            problem_statement: A clear statement of the problem to be analyzed, including ALL relevant details.

        Returns:
            The detailed analysis from Claude, including its extended thinking process
        """
        await self.log_info(f"Thinking hard about: {problem_statement[:100]}...", sender=self.caller.agent_name)

        provider = self.config.thinking_config.provider
        api_key = self.config.get_api_key(provider)
        rate_limits = self.config.thinking_config.rate_limits

        # Check if the caller has an instrumented client we can leverage
        if hasattr(self.caller, "llm") and isinstance(self.caller.llm, InstrumentedLLMClient):
            # Create a new instrumented client with the same Langfuse instance but for thinking
            client = InstrumentedLLMClient(
                provider=provider.value,
                api_key=api_key,
                max_retries=rate_limits.max_retries,
                requests_per_minute=rate_limits.requests_per_minute,
                tokens_per_minute=rate_limits.tokens_per_minute,
                langfuse_client=self.caller.llm.langfuse,
                workspace_id=self.caller.workspace_id,
                thread_id=self.caller.thread_id,
                agent_type=f"{self.caller.agent_name}-thinking",
                environment=self.caller.llm.environment,
                user_id=self.caller.user_id,
                user_email=self.caller.user_email,
                token_manager=self.config.get_chatgpt_token_manager(),
            )
        else:
            # Fallback to regular client
            client = LLMClient(
                provider=provider.value,
                api_key=api_key,
                max_retries=rate_limits.max_retries,
                requests_per_minute=rate_limits.requests_per_minute,
                tokens_per_minute=rate_limits.tokens_per_minute,
                token_manager=self.config.get_chatgpt_token_manager(),
            )

        try:
            model_specs = get_model_specs(self.config.thinking_config.provider, self.config.thinking_config.model)
            # Call LLM with the model-specific thinking effort configured for this session.
            thinking_param = self.config.thinking_config.thinking_effort

            system_message = Message(role="system", content=[TextBlock(text=prompts.THINK_HARD_PROMPT)])

            messages = MessageHistory(
                [
                    Message(
                        role="user",
                        content=[
                            TextBlock(
                                text=f"Think deeply and comprehensively about this problem:\n\n{problem_statement}"
                            )
                        ],
                    )
                ]
            )

            # Get tool_call_id from caller if available for streaming
            tool_call_id = getattr(self.caller, "current_tool_call_id", None)

            # Use streaming to avoid timeout issues
            thinking_content = []
            text_content = []
            accumulated_thinking = ""
            accumulated_text = ""
            has_sent_thinking_header = False
            has_sent_analysis_header = False

            max_completion = model_specs["max_completion_tokens"]

            # Use the stream and process chunks for streaming updates
            async with await client.stream(
                model=self.config.thinking_config.model,
                max_completion_tokens=max_completion,
                system=system_message,
                messages=messages,
                thinking=thinking_param,
            ) as stream:
                # Process chunks for streaming if we have a tool_call_id
                if tool_call_id:
                    async for chunk in stream:
                        # Check if this is a thinking chunk
                        if hasattr(chunk, "thinking") and chunk.thinking:
                            # Send header if first thinking content
                            if not has_sent_thinking_header:
                                await self.send_streaming_update(
                                    "# Extended Thinking Process\n\n",
                                    tool_call_id,
                                    "think_hard",
                                    is_complete=False,
                                    stream_mode="append",
                                )
                                has_sent_thinking_header = True

                            accumulated_thinking += chunk.thinking
                            # Stream thinking content periodically
                            if len(accumulated_thinking) >= 50:
                                await self.send_streaming_update(
                                    accumulated_thinking,
                                    tool_call_id,
                                    "think_hard",
                                    is_complete=False,
                                    stream_mode="append",
                                )
                                accumulated_thinking = ""

                        # Check if this is a text chunk
                        elif hasattr(chunk, "text") and chunk.text:
                            # Send any remaining thinking content and analysis header
                            if accumulated_thinking:
                                await self.send_streaming_update(
                                    accumulated_thinking + "\n\n",
                                    tool_call_id,
                                    "think_hard",
                                    is_complete=False,
                                    stream_mode="append",
                                )
                                accumulated_thinking = ""

                            if not has_sent_analysis_header:
                                await self.send_streaming_update(
                                    "# Final Analysis\n\n",
                                    tool_call_id,
                                    "think_hard",
                                    is_complete=False,
                                    stream_mode="append",
                                )
                                has_sent_analysis_header = True

                            accumulated_text += chunk.text
                            # Stream text content periodically
                            if len(accumulated_text) >= 50:
                                await self.send_streaming_update(
                                    accumulated_text,
                                    tool_call_id,
                                    "think_hard",
                                    is_complete=False,
                                    stream_mode="append",
                                )
                                accumulated_text = ""

                    # Send any remaining accumulated content
                    remaining_content = accumulated_thinking + accumulated_text
                    if remaining_content:
                        await self.send_streaming_update(
                            remaining_content,
                            tool_call_id,
                            "think_hard",
                            is_complete=False,
                            stream_mode="append",
                        )

                # Get the final message regardless of streaming
                final_message = await stream.get_final_message()

            # Extract thinking and text content from the final message
            for block in final_message.content:
                if isinstance(block, ThinkingBlock):
                    thinking_content.append(block.thinking)
                elif isinstance(block, TextBlock):
                    text_content.append(block.text)

            # Build the complete result
            result = ""
            if thinking_content:
                result += "# Extended Thinking Process\n\n"
                result += "\n".join(thinking_content) + "\n\n"
            result += "# Final Analysis\n\n"
            result += "\n".join(text_content)

            # Send final complete update if streaming
            if tool_call_id:
                await self.send_streaming_update(
                    result, tool_call_id, "think_hard", is_complete=True, stream_mode="replace"
                )

            return result

        except Exception as e:
            error_message = f"Error during extended thinking: {str(e)}"
            await self.log_error(error_message, sender=self.caller.agent_name)
            return error_message
