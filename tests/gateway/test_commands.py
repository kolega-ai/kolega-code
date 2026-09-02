"""Slash-command parsing."""

from kolega_code.gateway.commands import BOT_COMMANDS, HELP_TEXT, parse_command


def test_plain_text_is_not_a_command() -> None:
    assert parse_command("hello there") is None
    assert parse_command("  about /status and stuff") is None
    assert parse_command("") is None
    assert parse_command("/") is None


def test_command_without_args() -> None:
    assert parse_command("/status") == ("status", "")
    assert parse_command("  /NEW ") == ("new", "")


def test_command_with_args() -> None:
    assert parse_command("/model claude-opus-4-6") == ("model", "claude-opus-4-6")


def test_group_chat_botname_suffix_is_stripped() -> None:
    assert parse_command("/status@kolega_bot") == ("status", "")


def test_help_text_lists_every_command() -> None:
    for name in ("new", "status", "model", "permissions", "stop", "help"):
        assert f"/{name}" in HELP_TEXT


def test_bot_commands_menu_matches_the_command_set() -> None:
    menu = {name for name, _description in BOT_COMMANDS}
    assert menu == {"status", "new", "model", "permissions", "stop", "help"}
    # Every menu entry has a non-empty description for Telegram.
    assert all(description for _name, description in BOT_COMMANDS)
