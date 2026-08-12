"""Tests for the first-class tool primitives (Tool, ToolRegistry, ToolPolicy)."""

import pytest

from kolega_code.config import EditProtocol
from kolega_code.llm.models import ToolDefinition
from kolega_code.tools import Tool, ToolError, ToolPolicy, ToolRegistry


def make_tool(name: str, *, parallel_safe: bool = False, result: str = "ok") -> Tool:
    async def handler(**inputs):
        return f"{result}:{inputs.get('arg', '')}"

    return Tool(
        name=name,
        definition=ToolDefinition(name=name, description=f"{name} tool", parameters=[]),
        handler=handler,
        parallel_safe=parallel_safe,
    )


class TestToolRegistry:
    def test_add_and_lookup(self):
        registry = ToolRegistry().add(make_tool("read"), make_tool("write"))
        assert "read" in registry
        assert "missing" not in registry
        assert registry.get("write").name == "write"
        assert registry.names() == ["read", "write"]

    def test_duplicate_registration_rejected(self):
        registry = ToolRegistry().add(make_tool("read"))
        with pytest.raises(ValueError, match="already registered"):
            registry.add(make_tool("read"))

    @pytest.mark.asyncio
    async def test_call_dispatches_by_name(self):
        registry = ToolRegistry().add(make_tool("read", result="contents"))
        assert await registry.call("read", arg="x") == "contents:x"

    @pytest.mark.asyncio
    async def test_call_allows_tool_input_named_name(self):
        async def handler(name: str):
            return f"activated:{name}"

        registry = ToolRegistry().add(
            Tool(
                name="sample_tool",
                definition=ToolDefinition(name="sample_tool", description="A sample tool", parameters=[]),
                handler=handler,
            )
        )

        assert await registry.call("sample_tool", name="some-skill") == "activated:some-skill"

    @pytest.mark.asyncio
    async def test_call_allows_tool_input_named_tool_name(self):
        async def handler(tool_name: str):
            return f"echo:{tool_name}"

        registry = ToolRegistry().add(
            Tool(
                name="echo",
                definition=ToolDefinition(name="echo", description="Echo a tool_name input", parameters=[]),
                handler=handler,
            )
        )

        assert await registry.call("echo", tool_name="payload") == "echo:payload"

    @pytest.mark.asyncio
    async def test_call_unknown_tool_raises(self):
        with pytest.raises(KeyError):
            await ToolRegistry().call("nope")

    def test_select_applies_policy(self):
        registry = ToolRegistry().add(make_tool("read"), make_tool("write"), make_tool("delete"))

        excluded = registry.select(ToolPolicy(exclude=frozenset({"delete"})))
        assert excluded.names() == ["read", "write"]

        allowlisted = registry.select(ToolPolicy(include=frozenset({"write"})))
        assert allowlisted.names() == ["write"]

    def test_definitions_put_cache_checkpoint_on_last_only(self):
        registry = ToolRegistry().add(make_tool("a"), make_tool("b"), make_tool("c"))

        definitions = registry.definitions()
        assert [d.cache_checkpoint for d in definitions] == [False, False, True]

        # A subset view moves the checkpoint to its own last definition
        subset = registry.select(ToolPolicy(exclude=frozenset({"c"})))
        subset_definitions = subset.definitions()
        assert [d.name for d in subset_definitions] == ["a", "b"]
        assert [d.cache_checkpoint for d in subset_definitions] == [False, True]


class TestMalformedArgumentErrors:
    """A wrong-arguments call becomes a clean, public ToolError — never a raw
    TypeError that leaks the handler's internal class/method name."""

    @staticmethod
    def _tool(handler, name="eval"):
        return Tool(
            name=name,
            definition=ToolDefinition(name=name, description=f"{name} tool", parameters=[]),
            handler=handler,
        )

    @pytest.mark.asyncio
    async def test_missing_required_argument_reports_usage(self):
        async def handler(code: str, language: str = "py"):
            return code

        tool = self._tool(handler)
        with pytest.raises(ToolError) as exc_info:
            await tool.call(language="py")  # required `code` omitted
        message = str(exc_info.value)
        assert "Invalid arguments for `eval`" in message
        assert "code" in message
        assert "Usage: eval(code, language='py')" in message

    @pytest.mark.asyncio
    async def test_unexpected_keyword_reports_usage(self):
        async def handler(code: str):
            return code

        tool = self._tool(handler)
        with pytest.raises(ToolError) as exc_info:
            await tool.call(code="x", bogus=1)
        message = str(exc_info.value)
        assert "unexpected keyword argument" in message
        assert "Usage: eval(code)" in message

    @pytest.mark.asyncio
    async def test_message_never_leaks_handler_qualname(self):
        # Public tool name `write` is backed by an internal method on a class the
        # model must never see the name of.
        class _SecretInternalCollection:
            async def claude_write(self, path: str, content: str):
                return content

        tool = self._tool(_SecretInternalCollection().claude_write, name="write")
        with pytest.raises(ToolError) as exc_info:
            await tool.call(path="a.txt")  # required `content` omitted
        message = str(exc_info.value)
        assert "_SecretInternalCollection" not in message
        assert "claude_write" not in message
        assert "`write`" in message
        assert "Usage: write(path, content)" in message

    @pytest.mark.asyncio
    async def test_varkw_handler_accepts_any_arguments(self):
        async def handler(**kwargs):
            return sorted(kwargs)

        tool = self._tool(handler, name="flex")
        assert await tool.call(a=1, b=2) == ["a", "b"]

    @pytest.mark.asyncio
    async def test_valid_call_passes_through_unchanged(self):
        async def handler(code: str, language: str = "py"):
            return f"{language}:{code}"

        tool = self._tool(handler)
        assert await tool.call(code="1+1") == "py:1+1"


class TestToolCollectionRegistry:
    def test_registry_marks_parallel_safety_from_groups(self, tmp_path):
        from unittest.mock import Mock

        from kolega_code.agent.tools import ToolCollection

        collection = ToolCollection(
            tmp_path,
            "ws",
            "thread",
            Mock(),
            Mock(),
            Mock(agent_name="test", edit_protocol=EditProtocol.SEARCH_REPLACE),
        )
        registry = collection.registry()

        assert registry.get("read").parallel_safe
        assert not registry.get("edit").parallel_safe
        assert not registry.get("lsp_edit").parallel_safe
        assert not registry.get("multi_edit").parallel_safe
        assert not registry.get("write").parallel_safe
        # Command tools have side effects, so they must never be parallel-safe —
        # even when exposed to read-only agents via the command_tools group.
        for command_tool in ToolCollection.command_tools:
            assert not registry.get(command_tool).parallel_safe

    def test_get_tool_list_matches_registry_definitions(self, tmp_path):
        from unittest.mock import Mock

        from kolega_code.agent.tools import ToolCollection

        collection = ToolCollection(
            tmp_path,
            "ws",
            "thread",
            Mock(),
            Mock(),
            Mock(agent_name="test"),
        )
        names_from_list = [d.name for d in collection.get_tool_list()]
        registry_names = [tool.name for tool in collection.registry()]
        assert names_from_list == registry_names

    @pytest.mark.asyncio
    async def test_collection_call_allows_conflicting_tool_input_names(self, tmp_path):
        from unittest.mock import Mock

        from kolega_code.agent.tools import ToolCollection, ToolExtension

        async def echo_conflicting_inputs(name: str, tool_name: str):
            return f"{name}:{tool_name}"

        collection = ToolCollection(
            tmp_path,
            "ws",
            "thread",
            Mock(),
            Mock(),
            Mock(agent_name="test"),
            tool_extensions=[
                ToolExtension(
                    name="test-extension",
                    tools={"echo_conflicting_inputs": echo_conflicting_inputs},
                    tool_descriptions={"echo_conflicting_inputs": "Echo two inputs joined by a colon."},
                    tool_schemas={
                        "echo_conflicting_inputs": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}, "tool_name": {"type": "string"}},
                            "required": ["name", "tool_name"],
                        }
                    },
                )
            ],
        )

        assert (
            await collection.call("echo_conflicting_inputs", name="some-skill", tool_name="payload")
            == "some-skill:payload"
        )
