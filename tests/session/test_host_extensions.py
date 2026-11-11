import asyncio
import json
from collections.abc import Callable
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.agent import PromptExtension
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.agent.prompts import (
    BUILD_QUESTION_PROMPT,
    PLANNING_QUESTION_PROMPT,
    SHARED_TASK_LIST_PROMPT,
    SHARED_TASK_LIST_READONLY_PROMPT,
)
from kolega_code.agent.tool_definitions import tool_description_asset
from kolega_code.session.host_extensions import (
    build_question_prompt_extension,
    planning_question_prompt_extension,
    planning_question_tool_extension,
    shared_task_list_prompt_extension,
    shared_task_list_readonly_prompt_extension,
    shared_task_list_readonly_tool_extension,
    shared_task_list_tool_extension,
)
from kolega_code.tools import ASK_USER_CHOICE_INPUT_SCHEMA, ToolError


@pytest.mark.parametrize(
    ("factory", "extension_id", "title", "markdown"),
    [
        (shared_task_list_prompt_extension, "cli-shared-task-list", "Shared Task List", SHARED_TASK_LIST_PROMPT),
        (
            shared_task_list_readonly_prompt_extension,
            "cli-shared-task-list-readonly",
            "Shared Task List (read-only)",
            SHARED_TASK_LIST_READONLY_PROMPT,
        ),
        (planning_question_prompt_extension, "cli-planning-questions", "Planning Questions", PLANNING_QUESTION_PROMPT),
        (build_question_prompt_extension, "cli-build-questions", "Asking the User", BUILD_QUESTION_PROMPT),
    ],
)
def test_prompt_metadata(factory: Callable[[], PromptExtension], extension_id: str, title: str, markdown: str) -> None:
    extension = factory()
    assert extension.id == extension_id
    assert extension.title == title
    assert extension.markdown == markdown
    assert extension.modes == [AgentMode.CLI]
    assert extension.agent_types is None
    assert extension.propagate_to_sub_agents is False


@pytest.mark.asyncio
async def test_task_list_tools_read_dynamically_and_strip_save_refresh_in_order() -> None:
    state = {"markdown": ""}
    events: list[tuple[str, str]] = []

    def set_markdown(markdown: str) -> None:
        state["markdown"] = markdown
        events.append(("set", markdown))

    async def save() -> None:
        events.append(("save", state["markdown"]))
        await asyncio.sleep(0)
        events.append(("saved", state["markdown"]))

    extension = shared_task_list_tool_extension(
        get_task_list_markdown=lambda: state["markdown"],
        set_task_list_markdown=set_markdown,
        save_session=save,
        refresh_task_list=lambda: events.append(("refresh", state["markdown"])),
    )
    readonly = shared_task_list_readonly_tool_extension(get_task_list_markdown=lambda: state["markdown"])
    assert extension.name == "cli-shared-task-list"
    assert set(extension.tools) == {"get_task_list", "update_task_list"}
    assert extension.tool_descriptions == {
        "get_task_list": tool_description_asset("get_task_list"),
        "update_task_list": tool_description_asset("update_task_list"),
    }
    assert extension.tool_schemas == {
        "get_task_list": {"type": "object", "properties": {}, "required": []},
        "update_task_list": {
            "type": "object",
            "properties": {
                "task_list_markdown": {
                    "type": "string",
                    "description": "The full current shared task list as Markdown.",
                }
            },
            "required": ["task_list_markdown"],
        },
    }
    assert extension.tool_groups == {
        "planning_tools": ["get_task_list", "update_task_list"],
        "cli_task_list_tools": ["get_task_list", "update_task_list"],
    }
    assert readonly.name == "cli-shared-task-list-readonly"
    assert set(readonly.tools) == {"get_task_list"}
    assert readonly.tool_descriptions == {"get_task_list": tool_description_asset("get_task_list_readonly")}
    assert readonly.tool_schemas == {"get_task_list": {"type": "object", "properties": {}, "required": []}}
    assert readonly.tool_groups == {"planning_tools": ["get_task_list"], "cli_task_list_tools": ["get_task_list"]}
    for item in (extension, readonly):
        assert item.propagate_to_sub_agents is False
        assert item.exclusive_tools == frozenset()
        assert await item.tools["get_task_list"]() == "No task list has been set."
    assert await extension.tools["update_task_list"](" \n- [ ] inspect\n- [x] plan \n") == "Task list updated."
    markdown = "- [ ] inspect\n- [x] plan"
    assert events == [("set", markdown), ("save", markdown), ("saved", markdown), ("refresh", markdown)]
    for item in (extension, readonly):
        assert await item.tools["get_task_list"]() == markdown
        state["markdown"] = "new session state"
        assert await item.tools["get_task_list"]() == "new session state"
        state["markdown"] = markdown
    assert await extension.tools["update_task_list"](" \n ") == "Task list updated."
    assert state["markdown"] == ""


@pytest.mark.asyncio
async def test_failed_task_list_save_does_not_refresh() -> None:
    setter = Mock()
    refresh = Mock()
    extension = shared_task_list_tool_extension(
        get_task_list_markdown=lambda: "",
        set_task_list_markdown=setter,
        save_session=AsyncMock(side_effect=RuntimeError("save failed")),
        refresh_task_list=refresh,
        empty_message="Empty host list",
    )
    assert await extension.tools["get_task_list"]() == "Empty host list"
    with pytest.raises(RuntimeError, match="save failed"):
        await extension.tools["update_task_list"](" task ")
    setter.assert_called_once_with("task")
    refresh.assert_not_called()


@pytest.mark.asyncio
async def test_task_list_publishes_the_written_snapshot_after_save_and_refresh() -> None:
    state = {"markdown": ""}
    events: list[tuple[str, str]] = []

    async def save() -> None:
        events.append(("save", state["markdown"]))
        state["markdown"] = "a later in-memory value"

    async def emit(markdown: str) -> None:
        events.append(("emit", markdown))

    extension = shared_task_list_tool_extension(
        get_task_list_markdown=lambda: state["markdown"],
        set_task_list_markdown=lambda markdown: state.update(markdown=markdown),
        save_session=save,
        refresh_task_list=lambda: events.append(("refresh", state["markdown"])),
        emit_task_list=emit,
    )

    assert await extension.tools["update_task_list"](" \n- [ ] Inspect\n ") == "Task list updated."
    assert events == [
        ("save", "- [ ] Inspect"),
        ("refresh", "a later in-memory value"),
        ("emit", "- [ ] Inspect"),
    ]


@pytest.mark.asyncio
async def test_failed_task_list_save_does_not_publish() -> None:
    emit = AsyncMock()
    extension = shared_task_list_tool_extension(
        get_task_list_markdown=lambda: "",
        set_task_list_markdown=Mock(),
        save_session=AsyncMock(side_effect=RuntimeError("save failed")),
        refresh_task_list=Mock(),
        emit_task_list=emit,
    )

    with pytest.raises(RuntimeError, match="save failed"):
        await extension.tools["update_task_list"]("task")
    emit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_stage", ["save", "emit"])
async def test_task_list_update_drains_repeated_cancellation_before_returning(blocked_stage: str) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    state = {"markdown": ""}
    events: list[str] = []
    owners: list[asyncio.Task | None] = []

    def mark(stage: str) -> None:
        events.append(stage)
        owners.append(asyncio.current_task())

    def set_markdown(markdown: str) -> None:
        state["markdown"] = markdown
        mark("set")

    async def save() -> None:
        mark("save")
        if blocked_stage == "save":
            entered.set()
            await release.wait()
        mark("saved")

    def refresh() -> None:
        mark("refresh")

    async def emit(markdown: str) -> None:
        assert markdown == "- [x] saved"
        mark("emit")
        if blocked_stage == "emit":
            entered.set()
            await release.wait()
        mark("emitted")

    extension = shared_task_list_tool_extension(
        get_task_list_markdown=lambda: state["markdown"],
        set_task_list_markdown=set_markdown,
        save_session=save,
        refresh_task_list=refresh,
        emit_task_list=emit,
    )
    caller = asyncio.create_task(extension.tools["update_task_list"](" \n- [x] saved\n "))
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        for _ in range(3):
            caller.cancel()
            await asyncio.sleep(0)
            assert not caller.done()
        assert state["markdown"] == "- [x] saved"
        assert events == (["set", "save"] if blocked_stage == "save" else ["set", "save", "saved", "refresh", "emit"])
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(caller, timeout=10)
        assert events == ["set", "save", "saved", "refresh", "emit", "emitted"]
        owner = owners[0]
        assert owner is not None and owner is not caller and owner.done()
        assert all(task is owner for task in owners)
        await asyncio.sleep(0)
        assert events.count("emitted") == 1
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)


def question(text: str, header: str = "") -> dict:
    return {
        "question": text,
        "header": header,
        "options": [
            {"label": " A ", "description": " first "},
            {"label": "  ", "description": "ignored"},
            {"label": " B "},
        ],
    }


@pytest.mark.asyncio
async def test_questions_normalize_and_await_sequentially() -> None:
    events: list[tuple[str, str]] = []

    async def ask(text: str, labels: list[str], descriptions: list[str]) -> str:
        assert labels == ["A", "B"]
        assert descriptions == ["first", ""]
        events.append(("ask", text))
        await asyncio.sleep(0)
        events.append(("answer", text))
        return f"custom answer {text}"

    extension = planning_question_tool_extension(ask_user_choice=ask, has_pending_question=lambda: False)
    assert extension.name == "cli-planning-questions"
    assert set(extension.tools) == {"ask_user_choice"}
    assert extension.tool_descriptions == {"ask_user_choice": tool_description_asset("ask_user_choice")}
    assert extension.tool_schemas == {"ask_user_choice": ASK_USER_CHOICE_INPUT_SCHEMA}
    assert extension.tool_groups == {
        "planning_tools": ["ask_user_choice"],
        "coder_agent_tools": ["ask_user_choice"],
    }
    assert extension.propagate_to_sub_agents is False
    assert extension.exclusive_tools == frozenset()
    result = await extension.tools["ask_user_choice"](
        questions=[question(" First? ", " Choice "), question(" Second? ")]
    )
    assert result == json.dumps({"Choice": "custom answer First?", "Second?": "custom answer Second?"})
    assert events == [("ask", "First?"), ("answer", "First?"), ("ask", "Second?"), ("answer", "Second?")]


@pytest.mark.asyncio
async def test_duplicate_question_keys_keep_last_answer() -> None:
    ask = AsyncMock(side_effect=["first", "second"])
    extension = planning_question_tool_extension(ask_user_choice=ask, has_pending_question=lambda: False)
    result = await extension.tools["ask_user_choice"](
        questions=[question("First?", "Same"), question("Second?", " Same ")]
    )
    assert result == '{"Same": "second"}'
    assert ask.await_count == 2


@pytest.mark.asyncio
async def test_question_pending_guard_reads_current_host_state() -> None:
    pending = False
    ask = AsyncMock(return_value="A")
    extension = planning_question_tool_extension(ask_user_choice=ask, has_pending_question=lambda: pending)
    tool = extension.tools["ask_user_choice"]
    assert json.loads(await tool(questions=[question("Q?")])) == {"Q?": "A"}
    pending = True
    with pytest.raises(ToolError, match="^A planning question is already waiting for an answer\\.$"):
        await tool(questions=[question("Q?")])
    assert ask.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "questions",
    [
        [],
        "Q?",
        [None],
        [question(" ")],
        [{"question": "Q?", "options": "A, B"}],
        [{"question": "Q?", "options": ["A", "B"]}],
        [{"question": "Q?", "options": [{"label": "A"}, {"label": " "}]}],
        [question("Valid first?"), {"question": "Invalid second?"}],
    ],
)
async def test_questions_validate_entire_batch_before_pending_check_or_asking(questions: object) -> None:
    ask = AsyncMock()
    pending = Mock(return_value=True)
    extension = planning_question_tool_extension(ask_user_choice=ask, has_pending_question=pending)
    with pytest.raises(ToolError) as error:
        await extension.tools["ask_user_choice"](questions=questions)
    assert "already waiting" not in str(error.value)
    ask.assert_not_awaited()
    pending.assert_not_called()


@pytest.mark.asyncio
async def test_question_failure_stops_before_next_question() -> None:
    ask = AsyncMock(side_effect=ToolError("No answer"))
    extension = planning_question_tool_extension(ask_user_choice=ask, has_pending_question=lambda: False)
    with pytest.raises(ToolError, match="No answer"):
        await extension.tools["ask_user_choice"](questions=[question("First?"), question("Second?")])
    ask.assert_awaited_once_with("First?", ["A", "B"], ["first", ""])
