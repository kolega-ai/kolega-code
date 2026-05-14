from .. import prompts
from ..llm.client import LLMClient
from ..llm.models import Message, MessageHistory, TextBlock
from ..llm.specs import get_model_specs
from .base_tool import BaseTool


class ApplyEditTool(BaseTool):

    async def edit_file(self, relative_path: str, instructions: str, code_edit: str) -> str:
        await self.log_info(f"Applying edits to: {relative_path}", sender=self.caller.agent_name)

        provider = self.config.edit_model_config.provider
        api_key = self.config.get_api_key(provider)
        rate_limits = self.config.edit_model_config.rate_limits
        model_specs = get_model_specs(self.config.edit_model_config.provider, self.config.edit_model_config.model)

        client = LLMClient(
            provider=provider.value,
            api_key=api_key,
            max_retries=rate_limits.max_retries,
            requests_per_minute=rate_limits.requests_per_minute,
            tokens_per_minute=rate_limits.tokens_per_minute,
        )

        try:
            # Read the original file content
            if not self.filesystem.exists(relative_path):
                raise FileNotFoundError(f"File not found: {relative_path}")

            if not self.filesystem.is_file(relative_path):
                raise ValueError(f"Not a file: {relative_path}")

            # Read the original code from the file
            original_code = self.filesystem.read_text(relative_path)

            system_prompt = prompts.APPLY_EDIT_USER_PROMPT
            user_prompt = prompts.APPLY_EDIT_USER_PROMPT.format(
                original_code=original_code, code_edit=code_edit, instructions=instructions
            )

            system_message = Message(role="system", content=[TextBlock(text=system_prompt)])

            messages = MessageHistory([Message(role="user", content=[TextBlock(text=user_prompt)])])

            # Count tokens in the messages to ensure they're within model limits
            tokens = await client.count_tokens(messages=messages, system=system_message)

            if tokens.input_tokens > model_specs["max_completion_tokens"]:
                await self.log_warning(
                    f"The input tokens are higher than the max completion tokens in the model. ({tokens} vs {model_specs['max_completion_tokens']})",
                    sender=self.caller.agent_name,
                )

            response = await client.generate(
                model=self.config.edit_model_config.model,
                max_completion_tokens=model_specs["max_completion_tokens"],
                system=system_message,
                messages=messages,
            )

            response_text = response.get_text_content()

            if "<updated-code>" not in response_text:
                raise ValueError("Malformed LLM response.")

            updated_code = response_text.split("<updated-code>\n")[1].split("</updated-code>")[0]
            if not updated_code:
                raise ValueError("Updated code is empty.")

            # If we are here we have the updated code.
            # Write the updated code to the file (with vibe policy enforcement)
            try:
                blocked_msg = self._enforce_vibe_edit_policy(relative_path)
                if blocked_msg:
                    return blocked_msg
                self.filesystem.write_text(relative_path, updated_code)
                success_msg = f"Successfully updated file: {relative_path}"
                await self.log_info(success_msg, sender=self.caller.agent_name)
                return f"# {relative_path} has been updated.\n\n```\n{updated_code}\n```"
            except PermissionError:
                error_msg = f"Permission denied when writing to file: {relative_path}"
                await self.log_error(error_msg, sender=self.caller.agent_name)
                raise
            except Exception as e:
                error_msg = f"Failed to write to file {relative_path}: {str(e)}"
                await self.log_error(error_msg, sender=self.caller.agent_name)
                raise

        except Exception as e:
            error_message = f"Error while applying edit: {str(e)}"
            await self.log_error(error_message, sender=self.caller.agent_name)

            import traceback

            traceback_str = traceback.format_exc()
            print(f"Traceback:\n{traceback_str}")
            return error_message
