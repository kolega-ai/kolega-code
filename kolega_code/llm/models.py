import base64
import json
import logging
from typing import Any, Dict, List, Optional, Union

from google.genai import types as genai_types

from .tool_execution_ids import ToolExecutionIdRegistry, new_tool_execution_id

# Mapping from type string to class
CONTENT_BLOCK_CLASSES = {}
logger = logging.getLogger(__name__)


def _remove_trailing_commas(payload: str) -> str:
    """Remove trailing commas before } or ] which frequently cause JSON errors."""
    # Simple, conservative fixes
    payload = payload.replace(",}\n", "}\n").replace(", }", " }")
    payload = payload.replace(",]\n", "]\n").replace(", ]", " ]")
    # Handle edge cases without newlines
    payload = payload.replace(",}", "}")
    payload = payload.replace(",]", "]")
    return payload


def _balance_brackets(payload: str) -> str:
    """If there is an off-by-one bracket mismatch, try to balance it by appending the closing bracket."""
    opens = payload.count("{")
    closes = payload.count("}")
    if opens == closes:
        return payload
    if opens == closes + 1:
        return payload + "}"
    return payload


def safe_parse_tool_arguments(raw: str) -> Dict[str, Any]:
    """Parse OpenAI tool function.arguments into a dict safely.

    - Tries strict json.loads first
    - Applies minimal, conservative repairs (trim, remove trailing commas, balance one missing brace)
    - Returns a fallback dict with _raw_arguments and _parse_error on failure
    """
    try:
        if not raw:
            return {}
        return json.loads(raw)
    except Exception as first_err:
        repaired = raw.strip()
        repaired = _remove_trailing_commas(repaired)
        repaired = _balance_brackets(repaired)
        try:
            return json.loads(repaired)
        except Exception as second_err:
            # Last resort: do not crash; surface raw args for downstream handling
            snippet = raw if len(raw) <= 200 else raw[:200] + "..."
            logger.warning(
                f"Failed to parse tool arguments as JSON. Using fallback. error1={first_err} error2={second_err} raw_snippet={snippet}"
            )
            return {"_raw_arguments": raw, "_parse_error": "json_decode_error"}



def register_content_block(cls):
    CONTENT_BLOCK_CLASSES[cls.TYPE_NAME] = cls
    return cls


class ContentBlock:
    """Base class for content blocks in messages"""

    TYPE_NAME = "base"  # Should be overridden by subclasses

    def __init__(self, type: str, cache_checkpoint: bool = False):
        self.type = type
        self._cache_checkpoint = cache_checkpoint

    @property
    def cache_checkpoint(self) -> bool:
        return self._cache_checkpoint

    @cache_checkpoint.setter
    def cache_checkpoint(self, value: bool):
        self._cache_checkpoint = value

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the content block to a dictionary."""
        raise NotImplementedError("Subclasses must implement to_dict")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContentBlock":
        """Deserializes a content block from a dictionary."""
        block_type = data.get("type")
        if not block_type or block_type not in CONTENT_BLOCK_CLASSES:
            raise ValueError(f"Unknown or missing content block type: {block_type}")
        target_class = CONTENT_BLOCK_CLASSES[block_type]
        # We assume the target class's from_dict knows how to handle the data
        return target_class.from_dict(data)


@register_content_block
class TextBlock(ContentBlock):
    """Text content block for messages"""

    TYPE_NAME = "text"

    def __init__(self, text: str, cache_checkpoint: bool = False):
        super().__init__(type=self.TYPE_NAME, cache_checkpoint=cache_checkpoint)
        self.text = text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "text": self.text,
            "cache_checkpoint": self.cache_checkpoint,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextBlock":
        return cls(text=data["text"], cache_checkpoint=data.get("cache_checkpoint", False))

    def to_anthropic(self) -> Dict[str, Any]:
        """
        Converts the text block into the Anthropic format.

        Returns:
            Dict[str, Any]: A dictionary with the structure expected by Anthropic API
        """
        result = {"type": "text", "text": self.text}

        if self.cache_checkpoint:
            result["cache_control"] = {"type": "ephemeral"}

        return result

    def to_openai(self) -> Dict[str, Any]:
        """
        Converts the text block into the OpenAI format.

        Returns:
            Dict[str, Any]: A dictionary with the structure expected by OpenAI API
        """
        return {"type": "text", "text": self.text}

    def to_google(self) -> genai_types.Part:
        return genai_types.Part.from_text(text=self.text)

    def to_markdown(self) -> str:
        """
        Converts the text block into a markdown string.

        Returns:
            str: The text content formatted as markdown
        """
        return self.text


@register_content_block
class ImageBlock(ContentBlock):
    TYPE_NAME = "image_url"  # Consistent with OpenAI type for simplicity

    def __init__(self, image_type: str, media_type: str, data: str, cache_checkpoint: bool = False):
        super().__init__(type=self.TYPE_NAME, cache_checkpoint=cache_checkpoint)

        self.image_type = image_type  # e.g., 'base64' or 'url'
        self.media_type = media_type  # e.g., 'image/jpeg'
        self.data = data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "image_type": self.image_type,
            "media_type": self.media_type,
            "data": self.data,
            "cache_checkpoint": self.cache_checkpoint,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageBlock":
        return cls(
            image_type=data["image_type"],
            media_type=data["media_type"],
            data=data["data"],
            cache_checkpoint=data.get("cache_checkpoint", False),
        )

    def to_anthropic(self) -> Dict[str, Any]:
        """
        Converts the image block into the Anthropic format.

        The method formats the image data according to Anthropic's API requirements,
        including the image type (base64 or url), media type (MIME type), and the
        actual image data.

        Returns:
            Dict[str, Any]: A dictionary with the structure expected by Anthropic API
        """
        result = {
            "type": "image",
            "source": {"type": self.image_type, "media_type": self.media_type, "data": self.data},
        }

        if self.cache_checkpoint:
            result["cache_control"] = {"type": "ephemeral"}

        return result

    def to_openai(self) -> Dict[str, Any]:
        """
        Converts the image block into the OpenAI format.

        The method formats the image data according to OpenAI's API requirements,
        including the image type (base64 or url), media type (MIME type), and the
        actual image data.

        Returns:
            Dict[str, Any]: A dictionary with the structure expected by OpenAI API
        """
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{self.media_type};base64,{self.data}" if self.image_type == "base64" else self.data
            },
        }

    def to_google(self) -> genai_types.Part:
        return genai_types.Part.from_bytes(data=base64.b64decode(self.data), mime_type=self.media_type)

    def to_markdown(self) -> str:
        """
        Converts the image block into a markdown string with the image embedded.

        For base64 images, this creates a markdown image tag with the data URI scheme,
        allowing the image to be displayed directly in markdown without external hosting.

        Returns:
            str: The image formatted as a markdown image tag
        """

        if self.image_type == "base64":
            return f"data:{self.media_type};base64,{self.data}"
        else:
            return self.data


@register_content_block
class ThinkingBlock(ContentBlock):
    """Thinking content block for messages"""

    TYPE_NAME = "thinking"

    def __init__(self, thinking: str, cache_checkpoint: bool = False, signature: Optional[str] = None):
        super().__init__(type=self.TYPE_NAME, cache_checkpoint=cache_checkpoint)
        self.thinking = thinking
        self.signature = signature

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": self.type,
            "thinking": self.thinking,
            "cache_checkpoint": self.cache_checkpoint,
        }
        if self.signature:
            result["signature"] = self.signature
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThinkingBlock":
        return cls(
            thinking=data["thinking"],
            cache_checkpoint=data.get("cache_checkpoint", False),
            signature=data.get("signature"),
        )

    def to_anthropic(self) -> Dict[str, Any]:
        """
        Converts the text block into the Anthropic format.

        Returns:
            Dict[str, Any]: A dictionary with the structure expected by Anthropic API
        """
        result = {"type": "thinking", "thinking": self.thinking}
        if self.signature:
            result["signature"] = self.signature

        if self.cache_checkpoint:
            result["cache_control"] = {"type": "ephemeral"}

        return result

    def to_openai(self) -> Dict[str, Any]:
        """
        Converts the thinking block into the OpenAI format.

        Returns:
            Dict[str, Any]: A dictionary with the structure expected by OpenAI API
        """
        # OpenAI doesn't have a direct equivalent for thinking blocks
        # Convert to a text block with formatting to indicate it's thinking
        return {"type": "text", "text": f"*Thinking:*\n{self.thinking}"}

    def to_google(self) -> genai_types.Part:
        return genai_types.Part.from_text(text=f"*Thinking:*\n{self.thinking}")

    def to_markdown(self) -> str:
        """
        Converts the thinking block into a markdown string.

        Returns:
            str: The thinking content formatted as markdown with code block
        """
        return f"*Thinking:*\n\n```\n{self.thinking}\n```"


@register_content_block
class RedactedThinkingBlock(ContentBlock):
    """Redacted thinking content block for provider-managed encrypted reasoning."""

    TYPE_NAME = "redacted_thinking"

    def __init__(self, data: str, cache_checkpoint: bool = False):
        super().__init__(type=self.TYPE_NAME, cache_checkpoint=cache_checkpoint)
        self.data = data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "data": self.data,
            "cache_checkpoint": self.cache_checkpoint,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RedactedThinkingBlock":
        return cls(data=data["data"], cache_checkpoint=data.get("cache_checkpoint", False))

    def to_anthropic(self) -> Dict[str, Any]:
        return {"type": "redacted_thinking", "data": self.data}

    def to_openai(self) -> Dict[str, Any]:
        return {"type": "text", "text": "[Redacted thinking]"}

    def to_google(self) -> genai_types.Part:
        return genai_types.Part.from_text(text="[Redacted thinking]")

    def to_markdown(self) -> str:
        return "*Redacted thinking*"


class ToolParameter:
    """Parameter definition for a tool"""

    def __init__(self, name: str, type: str, description: str, required: bool = False):
        self.name = name
        self.type = type
        self.description = description
        self.required = required


class ToolDefinition(ContentBlock):
    """Unified representation of a tool definition across providers"""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: List[ToolParameter],
        cache_checkpoint: bool = False,
        input_schema: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(type="tool_definition", cache_checkpoint=cache_checkpoint)
        self.name = name
        self.description = description
        self.parameters = parameters
        # When set, this explicit JSON schema (an "object" schema dict) is used
        # verbatim instead of the flat per-parameter schema. This lets a tool
        # declare nested shapes (arrays of objects, etc.) that the callable
        # introspection in tool_definition_from_callable cannot express.
        self.input_schema = input_schema

    def _object_schema(self) -> Dict[str, Any]:
        """The JSON-schema object describing this tool's input."""
        if self.input_schema is not None:
            return self.input_schema

        properties = {}
        required = []
        for param in self.parameters:
            properties[param.name] = {"type": param.type, "description": param.description}
            if param.required:
                required.append(param.name)
        return {"type": "object", "properties": properties, "required": required}

    @classmethod
    def _dict_to_google_schema(cls, schema: Dict[str, Any]) -> genai_types.Schema:
        """Recursively convert a JSON-schema dict to a google.genai Schema."""
        kwargs: Dict[str, Any] = {"type": str(schema.get("type", "string")).upper()}
        if schema.get("description"):
            kwargs["description"] = schema["description"]
        if "properties" in schema:
            kwargs["properties"] = {
                key: cls._dict_to_google_schema(value) for key, value in schema["properties"].items()
            }
        if "items" in schema:
            kwargs["items"] = cls._dict_to_google_schema(schema["items"])
        if schema.get("required"):
            kwargs["required"] = schema["required"]
        return genai_types.Schema(**kwargs)

    def to_anthropic(self) -> Dict[str, Any]:
        """
        Converts the tool definition into the Anthropic format.

        Returns:
            Dict[str, Any]: A dictionary with the structure expected by Anthropic API
        """
        result = {
            "name": self.name,
            "description": self.description,
            "input_schema": self._object_schema(),
        }

        if self.cache_checkpoint:
            result["cache_control"] = {"type": "ephemeral"}

        return result

    def to_openai(self) -> Dict[str, Any]:
        """
        Converts the tool definition into the OpenAI format.

        Returns:
            Dict[str, Any]: A dictionary with the structure expected by OpenAI API
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._object_schema(),
            },
        }

    def to_google(self) -> genai_types.Tool:
        if self.input_schema is not None:
            parameters = self._dict_to_google_schema(self.input_schema)
        else:
            properties = {}
            required = []
            for parameter in self.parameters:
                properties[parameter.name] = genai_types.Schema(
                    type=parameter.type.upper(), description=parameter.description
                )
                if parameter.required:
                    required.append(parameter.name)
            parameters = genai_types.Schema(type="OBJECT", properties=properties, required=required)

        function_declaration = genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=parameters,
        )

        return genai_types.Tool(function_declarations=[function_declaration])


@register_content_block
class ToolCall(ContentBlock):
    """Unified representation of a tool call across providers"""

    TYPE_NAME = "tool_call"  # Changed from 'tool_use' (Anthropic specific)

    def __init__(
        self,
        id: str,
        name: str,
        input: Dict[str, Any],
        cache_checkpoint: bool = False,
        execution_id: Optional[str] = None,
        thought_signature: Optional[bytes] = None,
    ):
        super().__init__(type=self.TYPE_NAME, cache_checkpoint=cache_checkpoint)
        self.id = id
        self.name = name
        self.input = input
        self.execution_id = execution_id or new_tool_execution_id()
        # Google (Gemini 3.x) returns an encrypted thought_signature on each function-call
        # part that must be echoed back verbatim when the history is resent, or the API 400s.
        self.thought_signature = thought_signature

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "type": self.type,
            "id": self.id,
            "name": self.name,
            "input": self.input,
            "cache_checkpoint": self.cache_checkpoint,
            "execution_id": self.execution_id,
        }
        if self.thought_signature is not None:
            # bytes aren't JSON-serializable; base64-encode for the session store.
            data["thought_signature"] = base64.b64encode(self.thought_signature).decode("ascii")
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        encoded_signature = data.get("thought_signature")
        return cls(
            id=data["id"],
            name=data["name"],
            input=data["input"],
            cache_checkpoint=data.get("cache_checkpoint", False),
            execution_id=data.get("execution_id"),
            thought_signature=base64.b64decode(encoded_signature) if encoded_signature else None,
        )

    def to_anthropic(self) -> Dict[str, Any]:
        """
        Converts the tool call into the Anthropic format.

        Returns:
            Dict[str, Any]: A dictionary with the structure expected by Anthropic API
        """
        result = {"type": "tool_use", "id": self.id, "name": self.name, "input": self.input}

        if self.cache_checkpoint:
            result["cache_control"] = {"type": "ephemeral"}

        return result

    def to_openai(self) -> Dict[str, Any]:
        """
        Converts the tool call into the OpenAI format.

        Returns:
            Dict[str, Any]: A dictionary with the structure expected by OpenAI API
        """
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": json.dumps(self.input)}}

    def to_google(self) -> genai_types.Part:
        return genai_types.Part(
            function_call=genai_types.FunctionCall(id=self.id, name=self.name, args=self.input),
            thought_signature=self.thought_signature,
        )

    def to_markdown(self) -> str:
        """
        Formats the tool call as a markdown string for conversation history display.

        Returns:
            str: A markdown formatted representation of the tool call
        """
        formatted_input = json.dumps(self.input, indent=2)
        return f"**{self.type.replace('_', ' ').capitalize()}**: `{self.name}`\n\n```json\n{formatted_input}\n```\n\n*Tool ID: {self.id}*"


@register_content_block
class ToolResult(ContentBlock):
    """Unified representation of a tool result across providers"""

    TYPE_NAME = "tool_result"

    def __init__(
        self,
        tool_use_id: str,
        content: Union[str, List[ContentBlock]],
        name: str,
        is_error: bool,
        cache_checkpoint: bool = False,
        execution_id: Optional[str] = None,
    ):
        super().__init__(type=self.TYPE_NAME, cache_checkpoint=cache_checkpoint)

        self.tool_use_id = tool_use_id
        self.content = content  # Can be str or list of ContentBlocks
        self.name = name
        self.is_error = bool(is_error)
        self.execution_id = execution_id

    def to_dict(self) -> Dict[str, Any]:
        serialized_content: Union[str, List[Dict[str, Any]]]
        if isinstance(self.content, str):
            serialized_content = self.content
        elif isinstance(self.content, list):
            serialized_content = [block.to_dict() for block in self.content]
        else:
            # Handle unexpected type, maybe log a warning or error
            serialized_content = str(self.content)

        result = {
            "type": self.type,
            "tool_use_id": self.tool_use_id,
            "content": serialized_content,
            "name": self.name,
            "is_error": self.is_error,
            "cache_checkpoint": self.cache_checkpoint,
        }
        if self.execution_id:
            result["execution_id"] = self.execution_id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolResult":
        deserialized_content: Union[str, List[ContentBlock]]
        raw_content = data["content"]

        if isinstance(raw_content, str):
            deserialized_content = raw_content
        elif isinstance(raw_content, list):
            # Recursively deserialize nested content blocks
            deserialized_content = [ContentBlock.from_dict(item) for item in raw_content]
        else:
            # Handle unexpected type
            raise ValueError(f"Unexpected content type during ToolResult deserialization: {type(raw_content)}")

        return cls(
            tool_use_id=data["tool_use_id"],
            content=deserialized_content,
            name=data["name"],
            is_error=data["is_error"],
            cache_checkpoint=data.get("cache_checkpoint", False),
            execution_id=data.get("execution_id"),
        )

    def to_anthropic(self) -> Dict[str, Any]:
        """
        Converts the tool result into the Anthropic format.

        Returns:
            Dict[str, Any]: A dictionary with the structure expected by Anthropic API
        """
        # Handle case where content is a list
        anthropic_content = self.content
        if isinstance(self.content, list):
            anthropic_content = [item.to_anthropic() for item in self.content]

        # Ensure error results have non-empty content - Anthropic API fails if content is empty
        if self.is_error and not anthropic_content:
            anthropic_content = "Tool execution error"

        result = {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": anthropic_content,
            "is_error": self.is_error,
        }

        if self.cache_checkpoint:
            result["cache_control"] = {"type": "ephemeral"}

        return result

    def to_openai(self) -> Dict[str, Any]:
        """
        Converts the tool result into the OpenAI format.

        OpenAI Chat Completions tool messages only accept string content or text
        parts. Image-bearing tool results are split by ``MessageHistory.to_openai``
        into a text-only tool message plus a follow-up user image message. This
        method returns only the valid text-only tool message for the result.

        Returns:
            Dict[str, Any]: A dictionary with the structure expected by OpenAI API
        """
        return _openai_tool_result_message(self)

    def to_google(self) -> genai_types.FunctionResponse:
        google_content = self.content
        if isinstance(self.content, list):
            google_content = [item.to_google() for item in self.content]

        response = {}

        if self.is_error:
            response["error"] = google_content
        else:
            response["output"] = google_content

        return genai_types.Part(
            function_response=genai_types.FunctionResponse(id=self.tool_use_id, name=self.name, response=response)
        )

    def to_markdown(self) -> str:
        """
        Formats the tool result as a markdown string for conversation history display.

        Returns:
            str: A markdown formatted representation of the tool result
        """
        status = "**Error**" if self.is_error else "**Result**"

        markdown_content = self.content
        if isinstance(self.content, list):
            markdown_content = "\n\n".join([item.to_markdown() for item in self.content])

        return f"{status} from tool call (ID: {self.tool_use_id}):\n\n```\n{markdown_content}\n```"


def _tool_result_blocks(tool_result: "ToolResult") -> List[ContentBlock]:
    if isinstance(tool_result.content, list):
        return tool_result.content
    return []


def _openai_tool_result_text(tool_result: "ToolResult") -> str:
    """Return text-only content for an OpenAI tool message.

    Chat Completions ``role=tool`` messages cannot carry image parts. If a tool
    returned images, acknowledge them in text; ``MessageHistory.to_openai`` emits
    the actual images in a following user message.
    """
    if isinstance(tool_result.content, str):
        text = tool_result.content
    else:
        text_parts: List[str] = []
        image_count = 0
        for block in tool_result.content or []:
            if isinstance(block, TextBlock):
                if block.text:
                    text_parts.append(block.text)
            elif isinstance(block, ImageBlock):
                image_count += 1
            else:
                block_text = getattr(block, "text", None)
                if block_text:
                    text_parts.append(str(block_text))
        text = "\n".join(text_parts).strip()
        if image_count:
            marker = (
                f"[{tool_result.name} returned {image_count} image"
                f"{'s' if image_count != 1 else ''}; attached in the following user message.]"
            )
            text = f"{text}\n{marker}".strip() if text else marker

    if tool_result.is_error and not text:
        return "Tool execution error"
    return text


def _openai_tool_result_images(tool_result: "ToolResult") -> List[ImageBlock]:
    return [block for block in _tool_result_blocks(tool_result) if isinstance(block, ImageBlock)]


def _openai_tool_result_message(tool_result: "ToolResult") -> Dict[str, Any]:
    return {
        "role": "tool",
        "content": _openai_tool_result_text(tool_result),
        "tool_call_id": tool_result.tool_use_id,
    }


def _openai_tool_result_image_followup(tool_result: "ToolResult") -> Optional[Dict[str, Any]]:
    images = _openai_tool_result_images(tool_result)
    if not images:
        return None
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": f"Image returned by tool {tool_result.name} for tool call {tool_result.tool_use_id}.",
        }
    ]
    content.extend(image.to_openai() for image in images)
    return {"role": "user", "content": content}


def _openai_messages_for_tool_results(tool_results: List["ToolResult"]) -> List[Dict[str, Any]]:
    """Serialize tool results preserving OpenAI's required tool-message ordering.

    All text-only ``role=tool`` messages are emitted first, followed by user image
    messages for any image-bearing tool results. This keeps all required tool
    responses contiguous immediately after the assistant tool-call message.
    """
    messages: List[Dict[str, Any]] = []
    followups: List[Dict[str, Any]] = []
    for tool_result in tool_results:
        messages.append(_openai_tool_result_message(tool_result))
        followup = _openai_tool_result_image_followup(tool_result)
        if followup is not None:
            followups.append(followup)
    messages.extend(followups)
    return messages


class MessageChunk:
    """Unified representation of a message chunk during streaming"""

    def __init__(
        self,
        type: str,  # "text", "tool_use", "thinking", "tool_use_start", "tool_use_delta", etc.
        text: Optional[str] = None,
        tool_call: Optional[ToolCall] = None,
        thinking: Optional[str] = None,
        tool_call_delta: Optional[Dict[str, Any]] = None,
    ):
        self.type = type
        self.text = text
        self.tool_call = tool_call
        self.thinking = thinking
        self.tool_call_delta = tool_call_delta

    @classmethod
    def from_anthropic(cls, chunk):
        """
        Converts an Anthropic message chunk to a MessageChunk instance.

        Args:
            chunk: The Anthropic message chunk from the streaming response

        Returns:
            MessageChunk: A unified message chunk representation
        """
        if chunk.type == "text":
            return cls(type="text", text=chunk.text)

        # Handle thinking chunks
        elif chunk.type == "thinking":
            return cls(type="thinking", thinking=chunk.thinking)

        # Handle tool use start events
        elif chunk.type == "content_block_start" and hasattr(chunk, "content_block"):
            if chunk.content_block.type == "tool_use":
                return cls(
                    type="tool_use_start",
                    tool_call_delta={"id": chunk.content_block.id, "name": chunk.content_block.name, "input": ""},
                )

        # Handle tool use delta events (streaming JSON input)
        elif chunk.type == "content_block_delta" and hasattr(chunk, "delta"):
            if chunk.delta.type == "input_json_delta":
                return cls(type="tool_use_delta", tool_call_delta={"input_delta": chunk.delta.partial_json})
            # The Anthropic SDK emits a synthetic `thinking` event for each
            # raw `thinking_delta`; handling both duplicates streamed thinking.

        # Handle tool use stop events
        elif chunk.type == "content_block_stop":
            return cls(type="tool_use_stop")

        # Also check for thinking attribute directly (some chunks may have it)
        elif hasattr(chunk, "thinking") and chunk.thinking:
            return cls(type="thinking", thinking=chunk.thinking)

        # Default empty chunk for other types
        return cls(type="ignore", text="")

    @classmethod
    def from_openai(cls, chunk):
        """
        Converts an OpenAI ChatCompletion chunk to a MessageChunk instance.

        Args:
            chunk: The OpenAI ChatCompletion chunk from the streaming response

        Returns:
            MessageChunk: A unified message chunk representation
        """
        delta = chunk.choices[0].delta

        # Fireworks and other OpenAI-compatible reasoning models stream their
        # reasoning separately from answer text.
        reasoning_content = getattr(delta, "reasoning_content", None)
        if reasoning_content:
            return cls(type="thinking", thinking=reasoning_content)

        # Handle text content
        if delta.content is not None:
            return cls(type="text", text=delta.content)

        # Default empty chunk if no content or tool calls
        return cls(type="ignore", text="")

    @classmethod
    def from_google(cls, chunk):
        if chunk.text:
            return cls(type="text", text=chunk.text)

        # Default empty chunk for other types
        return cls(type="ignore", text="")


class Message:
    """Unified representation of a full message"""

    def __init__(
        self,
        role: str,  # "system", "user", or "assistant"
        content: Union[str, List["ContentBlock"]],
        stop_reason: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        usage_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.role = role
        self.content = content
        self.stop_reason = stop_reason
        self.tool_calls = tool_calls or []
        self.usage_metadata = usage_metadata or {}

    def get_text_content(self) -> str:
        """
        Returns the concatenated text content from all content blocks.

        If content is a string, returns it directly.
        If content is a list of ContentBlock objects, extracts and concatenates their text values.

        Returns:
            str: The concatenated text content
        """
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            # Extract text from each content block and join them
            return "\n".join(block.text for block in self.content if hasattr(block, "text"))

        return ""

    def to_anthropic(self) -> Dict[str, Any]:
        """
        Converts the Message instance to an Anthropic-compatible dictionary format.

        Returns:
            Dict[str, Any]: A dictionary in Anthropic's expected format
        """
        if isinstance(self.content, str):
            # If content is a string, wrap it in a text block
            content = [{"type": "text", "text": self.content}]
        elif isinstance(self.content, list):
            # If content is a list, convert each item using its to_anthropic method
            content = [item.to_anthropic() for item in self.content]
        else:
            # Fallback for unexpected content type
            content = []

        return {"role": self.role, "content": content}

    def to_openai(self) -> Dict[str, Any]:
        """
        Converts the Message instance to an OpenAI-compatible dictionary format.

        Returns:
            Dict[str, Any]: A dictionary in OpenAI's expected format
        """
        if isinstance(self.content, str):
            content = self.content
        elif isinstance(self.content, list):
            # Exclude tool call and tool result blocks from assistant content; they are handled separately
            non_tool_blocks = [item for item in self.content if not isinstance(item, (ToolCall, ToolResult))]
            content = [item.to_openai() for item in non_tool_blocks]
        else:
            # Fallback for unexpected content type
            content = ""

        # Handle tool calls if present
        result = {"role": self.role, "content": content}

        # Extract tool calls from content if they exist
        tool_calls = (
            [item for item in self.content if isinstance(item, ToolCall)] if isinstance(self.content, list) else []
        )

        if tool_calls:
            result["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": (
                            json.dumps(tool_call.input) if not isinstance(tool_call.input, str) else tool_call.input
                        ),
                    },
                }
                for tool_call in tool_calls
            ]

        return result

    def to_google(self) -> genai_types.Content:
        return genai_types.Content(
            role=self.role if self.role == "user" else "model", parts=[c.to_google() for c in self.content]
        )

    @classmethod
    def from_anthropic(cls, message, tool_execution_ids: Optional[ToolExecutionIdRegistry] = None):
        """
        Converts an Anthropic message to a Message instance.

        Args:
            message: The Anthropic message from the response

        Returns:
            Message: A unified message representation
        """
        tool_execution_ids = tool_execution_ids or ToolExecutionIdRegistry()
        tool_use_blocks = []
        content_blocks = []

        if hasattr(message, "content"):
            if isinstance(message.content, str):
                # Handle string content by creating a TextBlock
                content_blocks.append(TextBlock(text=message.content))
            elif isinstance(message.content, list):
                # Process structured content
                for block in message.content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            content_blocks.append(TextBlock(text=block.text))
                        elif block.type == "tool_use":
                            tool_call = ToolCall(
                                id=block.id,
                                name=block.name,
                                input=block.input,
                                execution_id=tool_execution_ids.get_or_create(block.id),
                            )
                            tool_use_blocks.append(tool_call)
                            content_blocks.append(tool_call)
                        elif block.type == "thinking":
                            content_blocks.append(
                                ThinkingBlock(thinking=block.thinking, signature=getattr(block, "signature", None))
                            )
                        elif block.type == "redacted_thinking":
                            content_blocks.append(RedactedThinkingBlock(data=block.data))

        # Extract usage metadata
        usage_metadata = {}
        if hasattr(message, "usage"):
            usage = message.usage
            usage_metadata = {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
                "cache_write_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
                "provider": "anthropic",
            }

        # print(f"Stop reason: {message.stop_reason if hasattr(message, 'stop_reason') else ''}")

        return cls(
            role=message.role,
            content=content_blocks,
            tool_calls=tool_use_blocks if tool_use_blocks else None,
            stop_reason=message.stop_reason if hasattr(message, "stop_reason") else None,
            usage_metadata=usage_metadata,
        )

    @classmethod
    def from_openai(cls, message, tool_execution_ids: Optional[ToolExecutionIdRegistry] = None):
        """
        Converts an OpenAI message to a Message instance.

        Args:
            message: The OpenAI message from the response

        Returns:
            Message: A unified message representation
        """
        stop_reason_map = {
            "tool_calls": "tool_use",
            "function_call": "tool_use",
            "length": "max_tokens",
            "stop": "end_turn",
        }

        tool_execution_ids = tool_execution_ids or ToolExecutionIdRegistry()
        content_blocks = []
        tool_use_blocks = []

        # OpenAI-compatible reasoning models (including Fireworks via Chat
        # Completions) may return reasoning separately from answer text.
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            content_blocks.append(ThinkingBlock(thinking=reasoning_content))

        # Handle content
        if hasattr(message, "content") and message.content:
            content_blocks.append(TextBlock(text=message.content))

        # Handle tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                parsed_args = safe_parse_tool_arguments(tool_call.function.arguments)
                tool_call_obj = ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    input=parsed_args,
                    execution_id=tool_execution_ids.get_or_create(tool_call.id),
                )
                tool_use_blocks.append(tool_call_obj)
                content_blocks.append(tool_call_obj)

        # Extract usage metadata - OpenAI provides this on the response, not the message
        # This will need to be set separately after creation
        usage_metadata = {"provider": "openai"}

        return cls(
            role="assistant",
            content=content_blocks,
            tool_calls=tool_use_blocks if tool_use_blocks else None,
            stop_reason=stop_reason_map[message.finish_reason] if hasattr(message, "finish_reason") else None,
            usage_metadata=usage_metadata,
        )

    @classmethod
    def from_google(
        cls,
        message: genai_types.GenerateContentResponse,
        tool_execution_ids: Optional[ToolExecutionIdRegistry] = None,
    ):
        stop_reason_map = {
            "MAX_TOKENS": "max_tokens",
            "STOP": "end_turn",
        }

        tool_execution_ids = tool_execution_ids or ToolExecutionIdRegistry()
        content_blocks = []
        tool_use_blocks = []

        if message.candidates[0].content and message.candidates[0].content.parts:
            for part in message.candidates[0].content.parts:
                if part.function_call:
                    # Capture the part-level thought_signature (Gemini 3.x requires it echoed back).
                    tool_call = ToolCall(
                        id=part.function_call.id,
                        name=part.function_call.name,
                        input=part.function_call.args,
                        execution_id=tool_execution_ids.get_or_create(part.function_call.id),
                        thought_signature=part.thought_signature,
                    )
                    tool_use_blocks.append(tool_call)
                    content_blocks.append(tool_call)
                elif part.thought:
                    content_blocks.append(ThinkingBlock(thinking=part.text))
                elif part.text:
                    content_blocks.append(TextBlock(text=part.text))

        mapped_stop_reason = stop_reason_map[message.finish_reason] if hasattr(message, "finish_reason") else None
        if tool_use_blocks:
            mapped_stop_reason = "tool_use"

        # Extract usage metadata
        usage_metadata = {}
        if hasattr(message, "usage_metadata"):
            usage = message.usage_metadata
            usage_metadata = {
                "prompt_token_count": getattr(usage, "prompt_token_count", 0),
                "candidates_token_count": getattr(usage, "candidates_token_count", 0),
                "total_token_count": getattr(usage, "total_token_count", 0),
                "provider": "google",
            }

        return cls(
            role="assistant",
            content=content_blocks,
            tool_calls=tool_use_blocks if tool_use_blocks else None,
            stop_reason=mapped_stop_reason,
            usage_metadata=usage_metadata,
        )

    @classmethod
    def from_openai_stream(
        cls,
        role: str,
        content: str,
        reasoning_content: str = "",
        tool_calls: Optional[list] = None,
        stop_reason: Optional[str] = None,
        tool_execution_ids: Optional[ToolExecutionIdRegistry] = None,
    ):
        """
        Converts OpenAI message components to a Message instance.

        Args:
            content: The content text from the OpenAI message
            reasoning_content: Reasoning text from OpenAI-compatible reasoning models, if any
            tool_calls: List of tool calls from the OpenAI message, if any
            stop_reason: The reason why the generation stopped

        Returns:
            Message: A unified message representation
        """
        stop_reason_map = {
            "tool_calls": "tool_use",
            "function_call": "tool_use",
            "length": "max_tokens",
            "stop": "end_turn",
        }

        tool_execution_ids = tool_execution_ids or ToolExecutionIdRegistry()
        content_blocks = []
        tool_use_blocks = []

        # Handle reasoning content before answer text, matching Anthropic's
        # thinking-then-text block order.
        if reasoning_content:
            content_blocks.append(ThinkingBlock(thinking=reasoning_content))

        # Handle content
        if content:
            content_blocks.append(TextBlock(text=content))

        # Handle tool calls
        if tool_calls:
            for tool_call in tool_calls.values():
                parsed_args = safe_parse_tool_arguments(tool_call.function.arguments)
                tool_call_obj = ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    input=parsed_args,
                    execution_id=tool_execution_ids.get_or_create(tool_call.id),
                )
                tool_use_blocks.append(tool_call_obj)
                content_blocks.append(tool_call_obj)

        return cls(
            role=role,
            content=content_blocks,
            tool_calls=tool_use_blocks if tool_use_blocks else None,
            stop_reason=stop_reason_map[stop_reason] if stop_reason else None,
            usage_metadata={},
        )

    @classmethod
    def from_google_stream(
        cls,
        role: str,
        content: str,
        tool_calls: Optional[list] = None,
        stop_reason: Optional[str] = None,
        tool_execution_ids: Optional[ToolExecutionIdRegistry] = None,
    ):
        stop_reason_map = {
            "MAX_TOKENS": "max_tokens",
            "STOP": "end_turn",
        }

        tool_execution_ids = tool_execution_ids or ToolExecutionIdRegistry()
        content_blocks = []
        tool_use_blocks = []

        # Handle content
        if content:
            content_blocks.append(TextBlock(text=content))

        # Handle tool calls. Values are (FunctionCall, thought_signature) pairs collected by
        # GoogleStreamWrapper so the Gemini 3.x signature is preserved for the next turn.
        if tool_calls:
            for function_call, thought_signature in tool_calls.values():
                tool_call_obj = ToolCall(
                    id=function_call.id,
                    name=function_call.name,
                    input=function_call.args,
                    execution_id=tool_execution_ids.get_or_create(function_call.id),
                    thought_signature=thought_signature,
                )
                tool_use_blocks.append(tool_call_obj)
                content_blocks.append(tool_call_obj)

        mapped_stop_reason = stop_reason_map[stop_reason] if stop_reason else None
        if tool_use_blocks:
            mapped_stop_reason = "tool_use"

        return cls(
            role=role,
            content=content_blocks,
            tool_calls=tool_use_blocks if tool_use_blocks else None,
            stop_reason=mapped_stop_reason,
            usage_metadata={},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Message object to a dictionary."""
        serialized_content: Union[str, List[Dict[str, Any]]]
        if isinstance(self.content, str):
            serialized_content = self.content
        elif isinstance(self.content, list):
            # Use the ContentBlock's to_dict method
            serialized_content = [block.to_dict() for block in self.content]
        else:
            # Or handle error/unexpected type
            serialized_content = []

        # Note: Tool calls are part of content list now, no separate field needed for dump
        return {
            "role": self.role,
            "content": serialized_content,
            "stop_reason": self.stop_reason,
            "usage_metadata": self.usage_metadata,
            # 'tool_calls' is implicitly handled within the 'content' list
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Deserializes a Message object from a dictionary."""
        deserialized_content: Union[str, List[ContentBlock]]
        raw_content = data.get("content")
        tool_calls = []  # Initialize tool_calls

        if isinstance(raw_content, str):
            deserialized_content = raw_content
        elif isinstance(raw_content, list):
            # Use the base ContentBlock.from_dict to handle different block types
            deserialized_content = [ContentBlock.from_dict(item) for item in raw_content]
            # Extract tool calls specifically for the Message attribute
            tool_calls = [block for block in deserialized_content if isinstance(block, ToolCall)]
        else:
            # Handle missing or unexpected content type
            deserialized_content = []  # Or raise error

        return cls(
            role=data["role"],
            content=deserialized_content,
            stop_reason=data.get("stop_reason"),
            tool_calls=tool_calls,  # Populate from deserialized content
            usage_metadata=data.get("usage_metadata", {}),
        )

    def to_markdown(self) -> str:
        """
        Converts the message to a markdown representation for conversation history display.

        Returns:
            str: A markdown formatted representation of the message
        """
        # Start with the role as a header
        role_display = self.role.capitalize()
        markdown = f"## {role_display}:\n\n"

        # Process content blocks
        if isinstance(self.content, str):
            markdown += self.content + "\n\n"
        else:
            for block in self.content:
                if isinstance(block, ToolResult) and any([isinstance(c, ImageBlock) for c in block.content]):
                    markdown += "**image removed to reduce length**\n\n"
                else:
                    if hasattr(block, "to_markdown"):
                        markdown += block.to_markdown() + "\n\n"
                    elif hasattr(block, "text"):
                        markdown += block.text + "\n\n"
                    elif hasattr(block, "thinking"):
                        markdown += f"*Thinking:*\n\n```\n{block.thinking}\n```\n\n"

        # Add stop reason if present
        if self.stop_reason:
            markdown += f"*Stop reason: {self.stop_reason}*\n\n"

        return markdown.strip()


class MessageHistory(list):
    def __init__(self, initial_items=None):

        # Validate initial items if provided
        if initial_items:
            for item in initial_items:
                self._validate_item(item)
            super().__init__(initial_items)
        else:
            super().__init__()

    def _validate_item(self, item):
        if not isinstance(item, Message):
            raise TypeError(f"Item must be of type {Message.__name__}, got {type(item).__name__}")

    # Override methods that add or replace items
    def append(self, item):
        self._validate_item(item)
        super().append(item)

    def extend(self, iterable):
        for item in iterable:
            self._validate_item(item)
        super().extend(iterable)

    def insert(self, index, item):
        self._validate_item(item)
        super().insert(index, item)

    def __setitem__(self, index, item):
        self._validate_item(item)
        super().__setitem__(index, item)

    def to_anthropic(self) -> list:
        return [m.to_anthropic() for m in self]

    def to_openai(self) -> list:
        processed_messages = []
        consumed_tool_result_ids = set()
        messages = list(self)

        def payload_has_content(payload: Dict[str, Any]) -> bool:
            content = payload.get("content")
            return (isinstance(content, str) and bool(content)) or (isinstance(content, list) and bool(content))

        def collect_lookahead_tool_results(start_index: int, needed_ids: set) -> List[ToolResult]:
            found: List[ToolResult] = []
            if not needed_ids:
                return found
            found_ids = set()
            for look_ahead in messages[start_index + 1 :]:
                if not isinstance(look_ahead.content, list):
                    continue
                for item in look_ahead.content:
                    if (
                        isinstance(item, ToolResult)
                        and item.tool_use_id in needed_ids
                        and item.tool_use_id not in consumed_tool_result_ids
                        and item.tool_use_id not in found_ids
                    ):
                        found.append(item)
                        found_ids.add(item.tool_use_id)
                if needed_ids.issubset(found_ids):
                    break
            return found

        for index, message in enumerate(messages):
            # No list content: pass through
            if not isinstance(message.content, list):
                processed_messages.append(message.to_openai())
                continue

            # Partition ToolResult blocks so they become separate 'tool' messages
            non_tool_result_blocks = [item for item in message.content if not isinstance(item, ToolResult)]
            tool_result_blocks = [
                item
                for item in message.content
                if isinstance(item, ToolResult) and item.tool_use_id not in consumed_tool_result_ids
            ]

            if tool_result_blocks:
                # Emit the primary message without tool results
                temp_message = Message(
                    role=message.role,
                    content=non_tool_result_blocks,
                    stop_reason=message.stop_reason,
                    tool_calls=message.tool_calls,
                    usage_metadata=message.usage_metadata,
                )
                temp_payload = temp_message.to_openai()

                # Avoid emitting empty assistant/user messages with neither content nor tool_calls
                if payload_has_content(temp_payload) or temp_payload.get("tool_calls"):
                    processed_messages.append(temp_payload)

                # If this same message included tool_calls, pull any missing tool
                # results forward so all role=tool messages stay contiguous.
                tool_call_ids = [item.id for item in message.content if isinstance(item, ToolCall)]
                found_ids = {tr.tool_use_id for tr in tool_result_blocks}
                lookahead_results = collect_lookahead_tool_results(index, set(tool_call_ids) - found_ids)
                batch_tool_results = tool_result_blocks + lookahead_results

                processed_messages.extend(_openai_tool_result_message(tr) for tr in batch_tool_results)
                for tr in batch_tool_results:
                    consumed_tool_result_ids.add(tr.tool_use_id)

                # If still missing, emit placeholder tool messages to satisfy API requirements.
                # These must come before any image follow-up user messages so all required
                # role=tool responses stay contiguous after the assistant tool calls.
                added_ids = {tr.tool_use_id for tr in lookahead_results}
                remaining = set(tool_call_ids) - (found_ids | added_ids)
                for missing_id in remaining:
                    processed_messages.append({"role": "tool", "content": "", "tool_call_id": missing_id})
                for tr in batch_tool_results:
                    followup = _openai_tool_result_image_followup(tr)
                    if followup is not None:
                        processed_messages.append(followup)
            else:
                # If this message only contained tool results that were already
                # emitted next to their assistant tool calls, skip it.
                if not non_tool_result_blocks and any(isinstance(item, ToolResult) for item in message.content):
                    continue

                # No ToolResult in this message. If it has tool_calls, ensure immediate tool responses.
                temp_payload = message.to_openai()
                if payload_has_content(temp_payload) or temp_payload.get("tool_calls"):
                    processed_messages.append(temp_payload)

                tool_calls = temp_payload.get("tool_calls") or []
                if tool_calls:
                    tool_call_ids = [tc.get("id") for tc in tool_calls if tc.get("id")]
                    lookahead_results = collect_lookahead_tool_results(index, set(tool_call_ids))

                    processed_messages.extend(_openai_tool_result_message(tr) for tr in lookahead_results)
                    for tr in lookahead_results:
                        consumed_tool_result_ids.add(tr.tool_use_id)

                    # If any are still missing, emit placeholder tool messages to satisfy API ordering.
                    # Placeholders also come before image follow-up user messages to keep all role=tool
                    # responses contiguous after the assistant tool-call message.
                    added_ids = {tr.tool_use_id for tr in lookahead_results}
                    remaining = [tc_id for tc_id in tool_call_ids if tc_id not in added_ids]
                    for missing_id in remaining:
                        processed_messages.append({"role": "tool", "content": "", "tool_call_id": missing_id})
                    for tr in lookahead_results:
                        followup = _openai_tool_result_image_followup(tr)
                        if followup is not None:
                            processed_messages.append(followup)

        return processed_messages

    def to_google(self) -> list:
        processed_messages = []

        for message in self:
            # If the message content is not a list of ToolResult objects, add it directly
            if not isinstance(message.content, list) or not all(
                isinstance(item, ToolResult) for item in message.content
            ):
                processed_messages.append(message.to_google())
            else:
                tool_response_message = genai_types.Content(
                    role="tool", parts=[item.to_google() for item in message.content]
                )

                processed_messages.append(tool_response_message)

        return processed_messages

    def get_markdown_conversation(self) -> str:
        markdown_output = []
        markdown_output.append("# Conversation\n")

        for message in self:
            markdown_output.append(message.to_markdown())

        conversation = "\n".join(markdown_output)

        return conversation
