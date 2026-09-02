"""Slash commands: gateway-level parsing shared by every adapter.

Command parsing happens before routing so a chat message that looks like a
command never reaches the model, and every future platform gets the same
command set for free. The trailing ``@BotName`` suffix Telegram appends in
groups is stripped.
"""

from __future__ import annotations

COMMAND_HELP = "help"
COMMAND_STATUS = "status"
COMMAND_NEW = "new"
COMMAND_RESET = "reset"
COMMAND_STOP = "stop"
COMMAND_MODEL = "model"
COMMAND_PERMISSIONS = "permissions"

#: Command menu registered with Telegram (setMyCommands), so the slash
#: menu shows up when typing in the chat. Single source for the adapter and
#: the help text.
BOT_COMMANDS: tuple[tuple[str, str], ...] = (
    ("status", "Show session and model info"),
    ("new", "Start a fresh session"),
    ("model", "Show or switch this chat's model"),
    ("permissions", "Show or change the permission mode"),
    ("stop", "Cancel the running turn"),
    ("help", "List commands"),
)

HELP_TEXT = (
    "Commands:\n"
    "/new — start a fresh session\n"
    "/status — show session and model info\n"
    "/model [model] — show or switch this chat's model\n"
    "/permissions [ask|auto] — show or change the permission mode\n"
    "/stop — cancel the running turn\n"
    "/help — this list"
)

UNKNOWN_COMMAND_REPLY = "Unknown command. Try /help."


def parse_command(text: str) -> tuple[str, str] | None:
    """Split a slash command into (name, args), or None for plain text."""
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    parts = stripped.split(None, 1)
    name = parts[0][1:].lower().split("@")[0]
    if not name:
        return None
    args = parts[1].strip() if len(parts) > 1 else ""
    return name, args
