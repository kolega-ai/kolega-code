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
    TUI = "tui"  # handled by KolegaCodeApp
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
TUI_AGENT_COMMAND_DESCRIPTION_OVERRIDES: dict[str, str] = {
    "/clear": "Clear message history, terminal output, and logs",
    "/reset": "Clear message history, terminal output, and logs (alias of /clear)",
}
SKILLS_LIST_COMMAND = "/skills"

TUI_COMMAND_ENTRIES: tuple[SlashCommandEntry, ...] = (
    SlashCommandEntry("skills", "List available Agent Skills", CommandScope.TUI),
    SlashCommandEntry("agents", "List and validate custom agents", CommandScope.TUI),
    SlashCommandEntry(
        "attach",
        "Attach an image (clipboard if no path, or a file path)",
        CommandScope.TUI,
    ),
    SlashCommandEntry("detach", "Remove pending image attachments", CommandScope.TUI),
    SlashCommandEntry("init", "Create or update AGENTS.md for this repository", CommandScope.TUI),
    SlashCommandEntry("plan", "Switch to plan mode", CommandScope.TUI),
    SlashCommandEntry("build", "Switch to build mode", CommandScope.TUI),
    SlashCommandEntry("sidebar", "Show or hide the side panel", CommandScope.TUI),
    SlashCommandEntry("settings", "Open Settings", CommandScope.TUI),
    SlashCommandEntry("permissions", "Show or switch shell/edit permission mode", CommandScope.TUI),
    SlashCommandEntry("model", "Show or switch the active model", CommandScope.TUI),
    SlashCommandEntry("effort", "Show or set the active thinking effort", CommandScope.TUI),
    SlashCommandEntry("lsp", "Show language server status and install instructions", CommandScope.TUI),
    SlashCommandEntry("login", "Sign in to a provider, e.g. /login chatgpt", CommandScope.TUI),
    SlashCommandEntry("logout", "Sign out of a provider, e.g. /logout chatgpt", CommandScope.TUI),
    SlashCommandEntry("goal", "Set, show, or clear an autonomous completion goal", CommandScope.TUI),
    SlashCommandEntry("gigacode", "Toggle gigacode workflow orchestration on or off", CommandScope.TUI),
    SlashCommandEntry("prompts", "Dump, list, or validate project prompt override files", CommandScope.TUI),
    SlashCommandEntry("queue-clear", "Clear queued follow-up messages", CommandScope.TUI),
    SlashCommandEntry("theme", "Show or switch the color theme", CommandScope.TUI),
    SlashCommandEntry("copy", "Copy the last response to the clipboard", CommandScope.TUI),
    SlashCommandEntry("diagnostics", "Show version, model/endpoint, and the diagnostics log path", CommandScope.TUI),
    SlashCommandEntry("bug", "Bundle local diagnostics into a shareable file for a bug report", CommandScope.TUI),
    SlashCommandEntry("version", "Show the Kolega Code version", CommandScope.TUI),
    SlashCommandEntry("update", "Update Kolega Code to the latest version", CommandScope.TUI),
    SlashCommandEntry("quit", "Save the session and exit", CommandScope.TUI),
    SlashCommandEntry("exit", "Save the session and exit", CommandScope.TUI),
)

TUI_COMMAND_NAMES: frozenset[str] = frozenset(entry.token for entry in TUI_COMMAND_ENTRIES)


def agent_command_entries() -> tuple[SlashCommandEntry, ...]:
    """Agent built-in commands, derived from the agent's own declarations."""
    return tuple(
        SlashCommandEntry(
            spec.name.removeprefix("/"),
            TUI_AGENT_COMMAND_DESCRIPTION_OVERRIDES.get(spec.name, spec.description),
            CommandScope.AGENT,
        )
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


def search_commands(query: str, skill_catalog: "SkillCatalog | None" = None, limit: int = 8) -> List[SlashCommandEntry]:
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
