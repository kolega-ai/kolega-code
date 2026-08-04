"""Machine-readable output for ``kolega-code ask --json``.

The JSON contract emits one ``{"kind": "message", ...}`` line per completed
assistant message instead of streaming chunks. Payloads are built from the
agent's in-memory history — which BaseAgent appends to before yielding a
message's final chunk — so each message is emitted complete, with its content
blocks (including tool calls), stop reason, and normalized usage. Opaque
provider state (thinking signatures, encrypted reasoning) is stripped: it is
meaningless to consumers and only meaningful when replayed to the provider.
Readable reasoning is kept — Anthropic thinking text, and DeepSeek flash's
plain-text ``responses_reasoning`` content — so ``--json`` consumers see the
same chain-of-thought the TUI displays. Note reasoning-heavy sessions can put
100k+ tokens of reasoning text into these payloads.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Optional

# Content-block fields that exist purely to be replayed to the provider.
_OPAQUE_BLOCK_TYPES = {"redacted_thinking"}
_OPAQUE_BLOCK_FIELDS = {
    "thinking": ("signature",),
    "tool_call": ("thought_signature",),
    "responses_reasoning": ("encrypted_content", "item_id"),
}


def serialize_ask_message(message: Any) -> dict[str, Any]:
    """Public ``message`` payload for one completed assistant message."""
    data = copy.deepcopy(message.to_dict())
    data.pop("usage_metadata", None)  # internal provider-shaped dict; `usage` is the contract
    content = data.get("content")
    if isinstance(content, list):
        cleaned: list[Any] = []
        for block in content:
            if not isinstance(block, dict):
                cleaned.append(block)
                continue
            block_type = block.get("type")
            if block_type in _OPAQUE_BLOCK_TYPES:
                continue
            if block_type == "responses_reasoning" and not block.get("content"):
                # Encrypted-only reasoning (OpenAI/ChatGPT backends): nothing
                # readable remains once the opaque blob is stripped, so the
                # whole block goes. DeepSeek flash blocks carry the raw
                # reasoning text in `content` and are kept — parity with the
                # Anthropic thinking blocks above.
                continue
            for field in _OPAQUE_BLOCK_FIELDS.get(block_type or "", ()):
                block.pop(field, None)
            block.pop("artifact_fields", None)
            cleaned.append(block)
        data["content"] = cleaned
    return data


def synthetic_text_message(text: str) -> dict[str, Any]:
    """Payload for informational agent output that never enters history
    (hook blocks, unsupported attachments, agent-command results, notices)."""
    return {"role": "assistant", "content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}


class AskMessageEmitter:
    """Emits each new assistant message from the agent's history exactly once.

    ``drain()`` is called on every completed response chunk: BaseAgent appends
    the assistant message to history before yielding its final chunk, so the
    message is complete when we read it. Tracks a plain index into the live
    history list; a shrinking history (``--loop-fresh`` clears) resets it.
    """

    def __init__(self, agent: Any) -> None:
        self._agent = agent
        self._seen = 0
        self.emitted_total = 0

    def drain(self, fallback_chunk: Optional[dict[str, Any]] = None) -> int:
        history = self._agent.history
        if len(history) < self._seen:
            self._seen = 0
        emitted = 0
        for message in list(history[self._seen :]):
            if getattr(message, "role", None) == "assistant":
                self._emit(serialize_ask_message(message))
                emitted += 1
        self._seen = len(history)

        if emitted == 0 and fallback_chunk is not None:
            text = fallback_chunk.get("content")
            if text and fallback_chunk.get("complete"):
                self._emit(synthetic_text_message(str(text)))
                emitted += 1
        return emitted

    def emit_text(self, text: str) -> None:
        """Emit a synthetic informational message (e.g. skill activation)."""
        self._emit(synthetic_text_message(text))

    def _emit(self, payload: dict[str, Any]) -> None:
        print(json.dumps({"kind": "message", "data": payload}, default=str))
        self.emitted_total += 1
