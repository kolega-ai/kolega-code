"""Offline tests for tool-definition schema serialization.

Covers the explicit `input_schema` override on ToolDefinition (used for nested
shapes) across all three providers. No network calls.
"""

from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from google import genai
from google.genai import _transformers
from google.genai import types as genai_types

from kolega_code.llm.models import ToolDefinition, ToolParameter


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

DISPATCH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "model_override": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "provider": {"type": "string", "minLength": 1},
                "model": {"type": "string", "minLength": 1},
                "thinking_effort": {
                    "anyOf": [
                        {"type": "string", "enum": ["low", "medium", "high"]},
                        {"type": "null"},
                    ]
                },
            },
            "required": ["provider", "model", "thinking_effort"],
        }
    },
    "required": ["model_override"],
}


def _nested_definition() -> ToolDefinition:
    return ToolDefinition(
        name="ask_user_choice",
        description="Ask the user.",
        parameters=[ToolParameter(name="questions", type="array", description="", required=True)],
        input_schema=NESTED_SCHEMA,
    )


def _dispatch_definition() -> ToolDefinition:
    return ToolDefinition(
        name="dispatch_agent",
        description="Dispatch an agent with an optional model override.",
        parameters=[],
        input_schema=DISPATCH_SCHEMA,
    )


def test_builtin_tool_specs_serialize_for_google_without_additional_properties():
    """No builtin schema may put additionalProperties on the Google wire.

    Gemini's function-declaration parameters are an OpenAPI 3.0 subset with no
    such field; the API rejects it in either spelling with 400 INVALID_ARGUMENT
    (seen live on gemini-3.5/3.6/3.7-flash). Several builtin schemas legitimately
    declare ``additionalProperties: false`` for other providers, so every spec is
    swept to catch reintroductions at any nesting depth.
    """
    import json

    from kolega_code.agent.tool_definitions import BUILTIN_TOOL_SPECS, builtin_tool_definition

    assert BUILTIN_TOOL_SPECS, "builtin tool specs must not be empty"
    checked = 0
    for spec_key in BUILTIN_TOOL_SPECS:
        tool = builtin_tool_definition(spec_key).to_google()
        assert tool.function_declarations is not None
        for declaration in tool.function_declarations:
            payload = json.dumps(declaration.model_dump(by_alias=True, exclude_none=True))
            assert "additionalProperties" not in payload, spec_key
            assert "additional_properties" not in payload, spec_key
            checked += 1
    assert checked >= len(BUILTIN_TOOL_SPECS)


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


def test_strict_nested_google_schema_keeps_in_memory_schema_models() -> None:
    tool = _dispatch_definition().to_google()
    assert tool.function_declarations is not None
    params = tool.function_declarations[0].parameters
    assert isinstance(params, genai_types.Schema)
    # Gemini's function-declaration schema has no additionalProperties field;
    # forwarding it 400s at the API ("Unknown name ... Cannot find field").
    assert params.additional_properties is None
    assert params.properties is not None

    model_override = params.properties["model_override"]
    assert isinstance(model_override, genai_types.Schema)
    assert model_override.additional_properties is None
    assert model_override.properties is not None
    assert set(model_override.properties) == {"provider", "model", "thinking_effort"}
    assert model_override.properties["provider"].min_length == 1
    assert model_override.properties["model"].min_length == 1

    thinking_effort = model_override.properties["thinking_effort"]
    assert thinking_effort.nullable is True
    assert thinking_effort.any_of is not None
    assert thinking_effort.any_of[0].enum == ["low", "medium", "high"]


@pytest.mark.asyncio
async def test_google_mldev_serializes_nested_schema_aliases_without_renaming_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise google-genai 2.12.1's real tool normalization and mldev request path."""

    tool = _dispatch_definition().to_google()
    assert tool.function_declarations is not None
    declaration = tool.function_declarations[0]
    assert type(declaration) is not genai_types.FunctionDeclaration

    client = genai.Client(api_key="test-key")
    request = AsyncMock(return_value=genai_types.HttpResponse(headers={}, body="{}"))
    monkeypatch.setattr(client._api_client, "async_request", request)

    try:
        normalized = _transformers.t_tools(client._api_client, [tool])
        assert normalized[-1].function_declarations is not None
        normalized_declaration = normalized[-1].function_declarations[0]
        assert normalized_declaration is declaration
        assert type(normalized_declaration) is type(declaration)

        await client.aio.models.generate_content(
            model="gemini-test",
            contents="Use the dispatch tool.",
            config=genai_types.GenerateContentConfig(tools=[tool]),
        )
    finally:
        await client.aio.aclose()
        client.close()

    request_call = request.await_args
    assert request_call is not None
    request_payload = cast(dict[str, Any], request_call.args[2])
    parameters = request_payload["tools"][0]["functionDeclarations"][0]["parameters"]
    model_override = parameters["properties"]["model_override"]
    thinking_effort = model_override["properties"]["thinking_effort"]

    assert "additionalProperties" not in parameters
    assert "additionalProperties" not in model_override
    assert model_override["properties"]["provider"]["minLength"] == 1
    assert model_override["properties"]["model"]["minLength"] == 1
    assert thinking_effort["anyOf"][0]["enum"] == ["low", "medium", "high"]
    assert thinking_effort["nullable"] is True
    assert set(parameters["properties"]) == {"model_override"}
    assert set(model_override["properties"]) == {"provider", "model", "thinking_effort"}

    serialized = repr(request_payload)
    assert "additional_properties" not in serialized
    assert "min_length" not in serialized
    assert "any_of" not in serialized


def test_flat_definition_still_serializes_without_input_schema():
    """A hand-built flat definition (no input_schema) keeps the flat object schema."""

    definition = ToolDefinition(
        name="sample",
        description="Do a thing.",
        parameters=[
            ToolParameter(name="query", type="string", description="The search text.", required=True),
            ToolParameter(name="limit", type="integer", description="Max results.", required=True),
        ],
    )
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
