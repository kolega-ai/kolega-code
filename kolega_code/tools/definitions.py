"""Build provider-agnostic tool definitions from Python callables.

The callable's signature supplies the parameter schema and its docstring
supplies the descriptions, so the code documenting a tool for developers is
also what the model sees.

The docstring's ``Args:`` section is parsed into per-parameter schema
descriptions and is also removed from the wire description: keeping it would
document every parameter twice in the API payload. Source docstrings are
untouched; the deduplication happens here, where the ToolDefinition is built,
so every provider serialization (to_anthropic/to_openai/to_google) inherits
it.
"""

import inspect
import re
from typing import Any, Callable, Dict, Optional

from ..llm.models import ToolDefinition, ToolParameter

# A docstring ``Args:`` section: the header at column 0 (inspect.getdoc
# dedents) through the next column-0 section line or the end of the text.
# Parameter entries and continuation lines are indented, so any column-0 line
# after ``Args:`` starts a new section (``Returns:``, ``Yields:``, ...).
_ARGS_BLOCK_RE = re.compile(r"^Args:.*?(?=\n[^\s]|\Z)", re.DOTALL | re.MULTILINE)


def strip_args_section(description: str) -> str:
    """Remove the docstring ``Args:`` block from a wire description.

    Everything before the block is kept verbatim, and any section after it
    (``Returns:`` and friends) is kept with its natural separation. Text with
    no ``Args:`` block is returned unchanged.
    """
    match = _ARGS_BLOCK_RE.search(description)
    if match is None:
        return description
    start, end = match.span()
    before = description[:start]
    after = description[end:]
    if not after.strip():
        return before.rstrip()
    return f"{before.rstrip()}\n\n{after.lstrip()}".strip()


def schema_has_property_descriptions(schema: Dict[str, Any]) -> bool:
    """Whether an explicit input schema documents every one of its properties.

    Only top-level properties are checked: the docstring ``Args:`` block
    documents the callable's parameters, which map one-to-one onto the
    schema's top-level properties. If any property lacks a description, that
    parameter's only documentation is the ``Args:`` block, so it must stay in
    the description.
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return True
    return all(
        isinstance(value, dict) and bool(str(value.get("description", "")).strip()) for value in properties.values()
    )


def tool_definition_from_callable(
    name: str,
    method: Callable[..., Any],
    *,
    description_overrides: Optional[Dict[str, str]] = None,
    keep_args_in_description: bool = False,
) -> ToolDefinition:
    """Build a provider-agnostic tool definition from a Python callable."""
    signature = inspect.signature(method)
    docstring = inspect.getdoc(method) or ""

    description = docstring.split("Raises:")[0].strip() if "Raises:" in docstring else docstring.strip()

    if description_overrides and name in description_overrides:
        description = description_overrides[name]
    elif not keep_args_in_description:
        description = strip_args_section(description)

    properties = {}
    required = []

    for param_name, param in signature.parameters.items():
        if param_name == "self":
            continue

        if param.default == inspect.Parameter.empty:
            required.append(param_name)

        param_type = "string"
        param_description = ""

        param_doc_match = re.search(rf"{param_name}:\s*(.*?)(?:\n\s*\w+:|$)", docstring, re.DOTALL)
        if param_doc_match:
            # Collapse the whitespace runs that docstring continuation-line
            # indentation leaves inside the captured description.
            param_description = re.sub(r"\s+", " ", param_doc_match.group(1)).strip()

        if param.annotation != inspect.Parameter.empty:
            annotation = str(param.annotation)
            if "str" in annotation:
                param_type = "string"
            elif "int" in annotation:
                param_type = "integer"
            elif "float" in annotation:
                param_type = "number"
            elif "bool" in annotation:
                param_type = "boolean"
            elif "List" in annotation or "list" in annotation:
                param_type = "array"
            elif "Dict" in annotation or "dict" in annotation:
                param_type = "object"

        properties[param_name] = {
            "type": param_type,
            "description": param_description,
        }

    tool_parameters = [
        ToolParameter(
            name=param_name,
            type=param_info["type"],
            description=param_info["description"],
            required=param_name in required,
        )
        for param_name, param_info in properties.items()
    ]

    return ToolDefinition(
        name=name,
        description=description,
        parameters=tool_parameters,
    )


# Explicit input schema for the ask_user_choice tool. It declares a `questions` array of
# objects (header, question text, a multiSelect flag, and a list of {label, description}
# options) — a nested shape that cannot be introspected from the handler signature, so it
# is supplied verbatim via ToolExtension.tool_schemas.
ASK_USER_CHOICE_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "description": "The questions to ask the user (1-4 questions).",
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The complete question to ask the user.",
                    },
                    "header": {
                        "type": "string",
                        "description": "A very short label for the question (a few words).",
                    },
                    "multiSelect": {
                        "type": "boolean",
                        "description": "Whether multiple options may be selected. Currently answered single-select.",
                    },
                    "options": {
                        "type": "array",
                        "description": "Two to four distinct choices.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "description": "The concise choice text shown to the user.",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "A short explanation of what choosing this option means.",
                                },
                            },
                            "required": ["label", "description"],
                        },
                    },
                },
                "required": ["question", "header", "options", "multiSelect"],
            },
        }
    },
    "required": ["questions"],
}

# Canonical correct-shape example used in ask_user_choice validation error messages so the
# model can self-correct.
ASK_USER_CHOICE_SHAPE_HINT = (
    'Call it as ask_user_choice(questions=[{"question": "...", "header": "...", '
    '"multiSelect": false, "options": [{"label": "...", "description": "..."}, '
    '{"label": "...", "description": "..."}]}]) with at least two options per question.'
)
