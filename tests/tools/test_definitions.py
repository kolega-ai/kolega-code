"""Offline tests for tool-definition schema serialization.

Covers the explicit `input_schema` override on ToolDefinition (used for nested
shapes the callable introspector cannot express) across all three providers.
No network calls.
"""

from google.genai import types as genai_types

from kolega_code.llm.models import ToolDefinition, ToolParameter
from kolega_code.tools.definitions import tool_definition_from_callable


NESTED_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "description": "The questions to ask.",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question text."},
                    "header": {"type": "string", "description": "Short label."},
                    "multiSelect": {"type": "boolean", "description": "Allow multiple."},
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "description": "Choice text."},
                                "description": {"type": "string", "description": "Explanation."},
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


def _nested_definition() -> ToolDefinition:
    return ToolDefinition(
        name="ask_user_choice",
        description="Ask the user.",
        parameters=[ToolParameter(name="questions", type="array", description="", required=True)],
        input_schema=NESTED_SCHEMA,
    )


def test_explicit_schema_passed_through_for_anthropic():
    definition = _nested_definition()
    payload = definition.to_anthropic()
    assert payload["name"] == "ask_user_choice"
    assert payload["input_schema"] == NESTED_SCHEMA


def test_explicit_schema_passed_through_for_openai():
    definition = _nested_definition()
    payload = definition.to_openai()
    assert payload["type"] == "function"
    assert payload["function"]["parameters"] == NESTED_SCHEMA


def test_explicit_schema_converted_for_google():
    definition = _nested_definition()
    tool = definition.to_google()
    assert tool.function_declarations is not None
    params = tool.function_declarations[0].parameters
    assert params is not None

    assert params.type == genai_types.Type.OBJECT
    assert params.properties is not None
    questions = params.properties["questions"]
    assert questions.type == genai_types.Type.ARRAY

    item = questions.items
    assert item is not None
    assert item.type == genai_types.Type.OBJECT
    assert item.properties is not None
    assert set(item.properties) == {"question", "header", "multiSelect", "options"}
    assert "question" in (item.required or [])

    options = item.properties["options"]
    assert options.type == genai_types.Type.ARRAY
    assert options.items is not None
    assert options.items.type == genai_types.Type.OBJECT
    assert options.items.properties is not None
    assert "label" in options.items.properties
    assert options.items.properties["label"].type == genai_types.Type.STRING


def test_flat_definition_still_serializes_without_input_schema():
    """A definition built by introspection (no input_schema) keeps the flat object schema."""

    async def sample(query: str, limit: int) -> str:
        """Do a thing.

        Args:
            query: The search text.
            limit: Max results.
        """
        return ""

    definition = tool_definition_from_callable("sample", sample)
    assert definition.input_schema is None

    anthropic = definition.to_anthropic()["input_schema"]
    assert anthropic["type"] == "object"
    assert set(anthropic["properties"]) == {"query", "limit"}
    assert anthropic["properties"]["query"]["type"] == "string"
    assert anthropic["properties"]["limit"]["type"] == "integer"
    assert set(anthropic["required"]) == {"query", "limit"}

    tool = definition.to_google()
    assert tool.function_declarations is not None
    google_params = tool.function_declarations[0].parameters
    assert google_params is not None
    assert google_params.type == genai_types.Type.OBJECT
    assert google_params.properties is not None
    assert set(google_params.properties) == {"query", "limit"}
