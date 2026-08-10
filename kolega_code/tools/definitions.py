"""Shared explicit tool-definition data.

Tool definitions are declared artifacts: built-in tools live in
``kolega_code.agent.tool_definitions`` and extension tools declare
``tool_descriptions``/``tool_schemas`` on their ``ToolExtension``. This module
holds definition data shared across hosts.
"""

from typing import Any, Dict

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
