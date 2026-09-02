"""Unit tests for handoff document generation (one-shot session summary)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from kolega_code.agent.handoff import (
    HANDOFF_EMPTY_ATTEMPTS,
    HANDOFF_MAX_TOKENS,
    HandoffCancelledError,
    HandoffResult,
    build_handoff_context_text,
    generate_handoff_document,
)
from kolega_code.agent.prompts import HANDOFF_SYSTEM_PROMPT, build_handoff_user_prompt
from kolega_code.llm.models import ImageBlock, Message, TextBlock

from .compaction_helpers import FakeLLM, text_msg, thinking_only_msg

HANDOFF_KW = {
    "model": "claude-haiku-4-5-20251001",
    "temperature": 1.0,
    "thinking": None,
}


def _history(size: int) -> list[Message]:
    messages: list[Message] = []
    for index in range(size):
        messages.append(text_msg("user", f"question {index}"))
        messages.append(text_msg("assistant", f"answer {index}"))
    return messages


class CancellableStream:
    """Stream that keeps yielding deltas until a cancel event fires."""

    def __init__(self, cancel_event: asyncio.Event) -> None:
        self._cancel = cancel_event
        self._yielded_once = False

    async def __aenter__(self) -> "CancellableStream":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def __aiter__(self) -> "CancellableStream":
        return self

    async def __anext__(self) -> dict[str, str]:
        if not self._yielded_once:
            self._yielded_once = True
            return {"type": "delta", "text": "x"}
        await asyncio.sleep(0.01)
        if self._cancel.is_set():
            raise StopAsyncIteration
        return {"type": "delta", "text": "x"}

    async def get_final_message(self) -> Message:
        raise AssertionError("must not be called after cancellation")


class CancellableFakeLLM(FakeLLM):
    def __init__(self, cancel_event: asyncio.Event) -> None:
        super().__init__()
        self.cancel_event = cancel_event

    async def _stream(self, *args: Any, **kwargs: Any) -> CancellableStream:
        return CancellableStream(self.cancel_event)


def test_handoff_prompt_includes_transcript_and_optional_focus() -> None:
    prompt = build_handoff_user_prompt("[user]\nfix the bug", focus="focus on the failing tests")
    assert "[user]\nfix the bug" in prompt
    assert "Additional focus: focus on the failing tests" in prompt
    assert "<conversation_history>" in prompt

    plain = build_handoff_user_prompt("[user]\nfix the bug")
    assert "Additional focus" not in plain


def test_handoff_system_prompt_defines_the_document_structure() -> None:
    for section in (
        "## Goal",
        "## Constraints & Preferences",
        "## Progress",
        "## Key Decisions",
        "## Critical Context",
        "## Next Steps",
    ):
        assert section in HANDOFF_SYSTEM_PROMPT
    assert "Output ONLY the handoff document" in HANDOFF_SYSTEM_PROMPT


def test_handoff_context_wrapper() -> None:
    wrapped = build_handoff_context_text("## Goal\nShip it")
    assert wrapped.startswith("<handoff-context>\n## Goal\nShip it\n</handoff-context>")
    assert "Use this context to continue the work seamlessly." in wrapped


@pytest.mark.asyncio
async def test_too_few_messages_does_not_dispatch() -> None:
    llm = FakeLLM()
    result = await generate_handoff_document(_history(0), llm=llm, **HANDOFF_KW)
    assert isinstance(result, HandoffResult)
    assert result.ok is False
    assert result.reason == "too_few"
    llm.stream.assert_not_called()


@pytest.mark.asyncio
async def test_happy_path_returns_the_document() -> None:
    llm = FakeLLM(summary_text="## Goal\nContinue the refactor")
    result = await generate_handoff_document(_history(2), llm=llm, focus="the refactor", **HANDOFF_KW)

    assert result.ok is True
    assert result.reason == "ok"
    assert result.document == "## Goal\nContinue the refactor"
    llm.stream.assert_awaited_once()
    assert llm.stream.await_args is not None
    kwargs = llm.stream.await_args.kwargs
    assert kwargs["model"] == HANDOFF_KW["model"]
    assert kwargs["max_completion_tokens"] == HANDOFF_MAX_TOKENS
    # The request is the rendered role-tagged transcript plus the focus line.
    prompt = kwargs["messages"][0].get_text_content()
    assert "[user]\nquestion 0" in prompt
    assert "Additional focus: the refactor" in prompt


@pytest.mark.asyncio
async def test_empty_document_retries_then_succeeds() -> None:
    llm = FakeLLM(
        message_script=[
            thinking_only_msg(),
            thinking_only_msg(),
            text_msg("assistant", "## Goal\nFinally produced a document"),
        ]
    )
    result = await generate_handoff_document(_history(2), llm=llm, **HANDOFF_KW)

    assert result.ok is True
    assert result.document == "## Goal\nFinally produced a document"
    assert llm.stream.await_count == 3


@pytest.mark.asyncio
async def test_empty_document_exhaustion_reports_empty_summary() -> None:
    llm = FakeLLM(message_script=[thinking_only_msg() for _ in range(HANDOFF_EMPTY_ATTEMPTS)])
    result = await generate_handoff_document(_history(2), llm=llm, **HANDOFF_KW)

    assert result.ok is False
    assert result.reason == "empty_summary"
    assert llm.stream.await_count == HANDOFF_EMPTY_ATTEMPTS


@pytest.mark.asyncio
async def test_cancel_before_dispatch_raises() -> None:
    llm = FakeLLM()
    cancel_event = asyncio.Event()
    cancel_event.set()
    with pytest.raises(HandoffCancelledError):
        await generate_handoff_document(_history(2), llm=llm, cancel_event=cancel_event, **HANDOFF_KW)
    llm.stream.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_mid_stream_closes_and_raises() -> None:
    cancel_event = asyncio.Event()
    llm = CancellableFakeLLM(cancel_event)

    async def cancel_soon() -> None:
        await asyncio.sleep(0.05)
        cancel_event.set()

    canceller = asyncio.create_task(cancel_soon())
    try:
        with pytest.raises(HandoffCancelledError):
            await generate_handoff_document(_history(2), llm=llm, cancel_event=cancel_event, **HANDOFF_KW)
    finally:
        await canceller
    assert llm.stream.await_count == 1


@pytest.mark.asyncio
async def test_llm_error_reports_llm_error() -> None:
    llm = FakeLLM()

    async def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("provider exploded")

    llm.stream.side_effect = fail
    result = await generate_handoff_document(_history(2), llm=llm, **HANDOFF_KW)

    assert result.ok is False
    assert result.reason == "llm_error"
    assert "provider exploded" in result.message


@pytest.mark.asyncio
async def test_images_are_replaced_with_placeholders_in_the_transcript() -> None:
    llm = FakeLLM(summary_text="## Goal\nDone")
    messages = [
        Message(
            role="user",
            content=[
                TextBlock(text="look at this screenshot"),
                ImageBlock(image_type="base64", media_type="image/png", data="QUJDREVGR0hJSg=="),
            ],
        ),
        text_msg("assistant", "ok"),
    ]
    result = await generate_handoff_document(messages, llm=llm, **HANDOFF_KW)

    assert result.ok is True
    assert llm.stream.await_args is not None
    prompt = llm.stream.await_args.kwargs["messages"][0].get_text_content()
    assert "look at this screenshot" in prompt
    assert "QUJDREVGR0hJSg==" not in prompt
