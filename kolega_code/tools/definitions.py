"""Build provider-agnostic tool definitions from Python callables.

The callable's signature supplies the parameter schema and its docstring
supplies the descriptions, so the code documenting a tool for developers is
also what the model sees.
"""

import inspect
import re
from typing import Any, Callable, Dict, Optional

from ..llm.models import ToolDefinition, ToolParameter


def tool_definition_from_callable(
    name: str,
    method: Callable[..., Any],
    *,
    description_overrides: Optional[Dict[str, str]] = None,
) -> ToolDefinition:
    """Build a provider-agnostic tool definition from a Python callable."""
    signature = inspect.signature(method)
    docstring = inspect.getdoc(method) or ""

    description = docstring.split("Raises:")[0].strip() if "Raises:" in docstring else docstring.strip()

    if description_overrides and name in description_overrides:
        description = description_overrides[name]

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
            param_description = param_doc_match.group(1).strip()

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
