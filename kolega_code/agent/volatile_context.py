"""Prompt context that changes during a session, injected instead of rendered into the prompt.

Agent memory, repository guidance (``AGENTS.md``/``KOLEGA.md``), and today's date all change
while a session runs. Rendering them into the system prompt makes every change expensive out
of proportion to its size: providers cache on a prefix match and the system prompt is hashed
ahead of every message, so editing one line of memory invalidates the cached prefix for the
entire conversation. A single ``edit_memory`` call costs a full-history rewrite on the next
turn, and that is true of Anthropic's explicit breakpoints, OpenAI's automatic prefix cache,
and Gemini's implicit caching alike.

Keeping this content out of the system prompt and appending it to the conversation only when
it changes leaves the prefix frozen for the life of the session. An update becomes one block
at the end of the history, which every later turn then reads from cache.

The tracker is deliberately dumb: callers supply the current text for every key and it
reports what changed. That keeps it independent of ``BaseAgent`` and directly testable.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from kolega_code.llm.models import TextBlock

# A genuine reminder is a standalone user message, so structure already distinguishes it from a
# `<system-reminder>` that merely appears inside tool output or user text — and the prompt says
# so explicitly. Content read from elsewhere is therefore left exactly as it is; scrubbing it
# would corrupt legitimate reads (a session transcript, or this repository's own prompt
# templates) to no benefit.
#
# What does need scrubbing is the text this module wraps: a `</system-reminder>` inside MEMORY.md
# or AGENTS.md would close the envelope early and promote the remainder of the file to operator
# context, which structure alone cannot distinguish.
_REMINDER_TAG = re.compile(r"<(/?)system-reminder", re.IGNORECASE)


def scrub_reminder_markup(text: str) -> str:
    """Neutralize reminder tags so untrusted text cannot forge or close an envelope."""
    return _REMINDER_TAG.sub(r"&lt;\1system-reminder", text)


@dataclass(frozen=True, slots=True)
class VolatileSection:
    """One kind of volatile context. Empty ``text`` means "not currently present"."""

    key: str
    text: str
    path: str = ""

    def fingerprint(self) -> str:
        # The path is part of the identity: guidance moving from AGENTS.md to KOLEGA.md is a
        # change worth reporting even in the unlikely case the contents match.
        return f"{self.path}\x00{self.text}"


class VolatileContextTracker:
    """Reports volatile context that changed since it was last sent to the model."""

    def __init__(self) -> None:
        self._last_sent: Dict[str, VolatileSection] = {}

    def pending_block(self, sections: Sequence[VolatileSection]) -> Optional[TextBlock]:
        """Return one reminder block covering the changed sections, or ``None`` if nothing changed.

        Pass every key on every call. A key whose ``text`` is empty is treated as absent, and a
        transition from present to absent is itself reported so the model is not left acting on
        content that has since been deleted.
        """
        rendered: List[str] = []
        for section in sections:
            previous = self._last_sent.get(section.key)
            if previous is not None and previous.fingerprint() == section.fingerprint():
                continue
            self._last_sent[section.key] = section
            if section.text:
                rendered.append(_render(section, section.text))
            elif previous is not None and previous.text:
                rendered.append(_render(section, _removal_notice(previous)))
            # Never seen, or already absent: nothing worth saying.

        if not rendered:
            return None
        return TextBlock(text="\n".join(rendered))

    def forget(self) -> None:
        """Drop the sent-state so the next call re-sends everything.

        Used when the conversation history this tracker was mirroring is no longer the prefix
        being sent — after a compaction that discards the injected blocks, for example.
        """
        self._last_sent.clear()


def _render(section: VolatileSection, body: str) -> str:
    attributes = f' source="{section.key}"'
    if section.path:
        attributes += f' path="{section.path}"'
    return f"<system-reminder{attributes}>\n{scrub_reminder_markup(body)}\n</system-reminder>"


_REMOVAL_LABELS = {
    "memory": "Private project memory",
    "guidance": "Repository guidance",
}


def _removal_notice(section: VolatileSection) -> str:
    subject = section.path or _REMOVAL_LABELS.get(section.key, section.key)
    return f"{subject} is no longer present. Disregard its earlier contents."
