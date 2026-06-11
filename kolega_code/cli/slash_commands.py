"""Single source of truth for slash commands across the TUI, the agent, and skills.

This module must stay free of Textual imports so ``main.py`` and tests can
import it cheaply. Agent commands are declared on
``CommandProcessor.SPECS`` (the agent package cannot import the CLI package);
this module aggregates them with TUI commands and dynamically discovered skills.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List

from ..agent.utils.commands import CommandProcessor

if TYPE_CHECKING:
    from .skills import SkillCatalog


class CommandScope(str, Enum):
    TUI = "tui"      # handled by KolegaCodeApp
    AGENT = "agent"  # handled by CommandProcessor inside the agent
    SKILL = "skill"  # dynamic, from SkillCatalog


@dataclass(frozen=True)
class SlashCommandEntry:
    name: str  # without the leading "/", e.g. "clear"
    description: str
    scope: CommandScope

    @property
    def token(self) -> str:
        return f"/{self.name}"


THREAD_RESET_COMMANDS: frozenset[str] = frozenset({"/clear", "/reset"})
SKILLS_LIST_COMMAND = "/skills"

TUI_COMMAND_ENTRIES: tuple[SlashCommandEntry, ...] = (
    SlashCommandEntry("skills", "List available Agent Skills", CommandScope.TUI),
    SlashCommandEntry("plan", "Switch to plan mode", CommandScope.TUI),
    SlashCommandEntry("build", "Switch to build mode", CommandScope.TUI),
    SlashCommandEntry("model", "Show or switch the active model", CommandScope.TUI),
    SlashCommandEntry("copy", "Copy the last response to the clipboard", CommandScope.TUI),
    SlashCommandEntry("version", "Show the Kolega Code version", CommandScope.TUI),
    SlashCommandEntry("quit", "Save the session and exit", CommandScope.TUI),
)

TUI_COMMAND_NAMES: frozenset[str] = frozenset(entry.token for entry in TUI_COMMAND_ENTRIES)


def agent_command_entries() -> tuple[SlashCommandEntry, ...]:
    """Agent built-in commands, derived from the agent's own declarations."""
    return tuple(
        SlashCommandEntry(spec.name.removeprefix("/"), spec.description, CommandScope.AGENT)
        for spec in CommandProcessor.SPECS
    )


def agent_command_names() -> frozenset[str]:
    """Command tokens (with leading "/") handled by the agent's CommandProcessor."""
    return frozenset(spec.name for spec in CommandProcessor.SPECS)


def all_command_entries(skill_catalog: "SkillCatalog | None" = None) -> List[SlashCommandEntry]:
    """All known slash commands: agent built-ins, TUI commands, then skills.

    De-duplicated by name; agent and TUI commands take precedence over a
    skill that happens to share a name.
    """
    entries: List[SlashCommandEntry] = []
    seen: set[str] = set()
    skill_entries = (
        tuple(
            SlashCommandEntry(record.name, record.description, CommandScope.SKILL)
            for record in skill_catalog.skills.values()
        )
        if skill_catalog is not None
        else ()
    )
    for entry in (*agent_command_entries(), *TUI_COMMAND_ENTRIES, *skill_entries):
        if entry.name in seen:
            continue
        seen.add(entry.name)
        entries.append(entry)
    return entries


def search_commands(
    query: str, skill_catalog: "SkillCatalog | None" = None, limit: int = 8
) -> List[SlashCommandEntry]:
    """Filter commands by ``query``: prefix matches first, then substring matches.

    An empty query returns all commands (capped at ``limit``). Results are
    alphabetical within each tier.
    """
    needle = query.lower()
    entries = all_command_entries(skill_catalog)
    if not needle:
        return sorted(entries, key=lambda entry: entry.name)[:limit]
    prefix = [entry for entry in entries if entry.name.lower().startswith(needle)]
    substring = [
        entry for entry in entries if needle in entry.name.lower() and not entry.name.lower().startswith(needle)
    ]
    ranked = sorted(prefix, key=lambda entry: entry.name) + sorted(substring, key=lambda entry: entry.name)
    return ranked[:limit]
