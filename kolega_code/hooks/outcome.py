"""HookOutcome: the normalized result every hook backend returns, plus merge()."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class HookOutcome:
    """What a hook decided. The same shape for command, python, and LLM backends.

    - ``blocked``: the hook denied/stopped the action; ``reason`` explains why.
    - ``updated_input``: replacement tool input (PreToolUse) applied before the call.
    - ``updated_output``: replacement tool output (PostToolUse) fed to the model.
    - ``additional_context``: extra text appended to the prompt / tool result.
    - ``end_turn``: end the current turn after this event (set implicitly when blocked).
    """

    blocked: bool = False
    reason: str = ""
    updated_input: Optional[dict[str, Any]] = None
    updated_output: Optional[str] = None
    additional_context: Optional[str] = None
    end_turn: bool = False

    @classmethod
    def empty(cls) -> "HookOutcome":
        return cls()

    @classmethod
    def deny(cls, reason: str) -> "HookOutcome":
        return cls(blocked=True, reason=reason or "Hook blocked the action.", end_turn=True)

    @property
    def is_empty(self) -> bool:
        return not (
            self.blocked
            or self.updated_input is not None
            or self.updated_output is not None
            or self.additional_context
            or self.end_turn
        )


def merge(outcomes: Iterable[HookOutcome]) -> HookOutcome:
    """Fold sequential hook outcomes into one.

    First block wins (and the dispatcher short-circuits remaining hooks). Input
    and output replacements take the last value (the dispatcher threads each
    hook's ``updated_input`` into the next, so last-writer-wins is the cumulative
    result). ``additional_context`` is concatenated in order.
    """
    blocked = False
    reason = ""
    updated_input: Optional[dict[str, Any]] = None
    updated_output: Optional[str] = None
    contexts: list[str] = []
    end_turn = False

    for outcome in outcomes:
        if outcome.updated_input is not None:
            updated_input = outcome.updated_input
        if outcome.updated_output is not None:
            updated_output = outcome.updated_output
        if outcome.additional_context:
            contexts.append(outcome.additional_context)
        if outcome.end_turn:
            end_turn = True
        if outcome.blocked and not blocked:
            blocked = True
            reason = outcome.reason

    return HookOutcome(
        blocked=blocked,
        reason=reason,
        updated_input=updated_input,
        updated_output=updated_output,
        additional_context="\n".join(contexts) if contexts else None,
        end_turn=end_turn or blocked,
    )
