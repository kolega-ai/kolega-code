"""Entrypoint for the Kolega Code CLI."""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from kolega_code.agent import CoderAgent
from kolega_code.agent.prompt_dump import (
    dump_prompt_overrides,
    format_prompt_dump_result,
    format_prompt_list_result,
    format_prompt_validation_result,
    list_prompt_overrides,
    validate_prompt_overrides,
)
from kolega_code.hooks import HookDispatcher, HookEvent, load_hook_config
from kolega_code.llm.exceptions import LLMBillingError, billing_error_message
from kolega_code.llm.models import TextBlock
from kolega_code.mcp.config import (
    MCPConfigError,
    MCPServerConfig,
    global_mcp_config_path,
    load_mcp_config,
    project_mcp_config_path,
    remove_server_config,
    set_server_enabled,
    upsert_server_config,
)
from kolega_code.mcp.service import MCP_FAILURE_MESSAGE_GENERIC, MCPService
from kolega_code.mcp.tools import build_mcp_tool_extension
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.permissions import (
    PermissionDecision,
    PermissionMode,
    PermissionStoreError,
    ProjectPermissionStore,
    allow_rule_options,
    normalize_permission_mode,
)
from kolega_code.services.browser import PlaywrightBrowserManager
from kolega_code.utils.images import encode_image_file

from .diagnostics import write_crash_log
from .config import (
    DEPRECATED_THINKING_TOKENS_MESSAGE,
    CliConfigError,
    CliConfigOverrides,
    active_model_override_message,
    build_agent_config,
    config_summary,
)
from .connection import CliConnectionManager
from .mentions import build_file_attachments
from .session_store import SessionRecord, SessionStore, SessionStoreError
from .settings import CliSettings, SettingsStore, SettingsStoreError
from .slash_commands import SKILLS_LIST_COMMAND, agent_command_names
from .skills import (
    SkillCatalog,
    activated_skill_names,
    build_skill_prompt_extension,
    build_skill_tool_extension,
    discover_skills,
)
from .updater import check_for_update, run_self_update, update_status_message

SUBCOMMANDS = {"ask", "sessions", "doctor", "update", "prompts", "mcp", "tui"}
RESUME_LATEST = "__latest__"
CLI_AGENT_MODE = AgentMode.CLI.value
ASK_DEFAULT_PERMISSION_MODE = PermissionMode.AUTO.value
CLI_BILLING_ERROR_MESSAGE = (
    "The selected provider could not run this request because it reported insufficient balance. "
    "Add credits to the provider account or switch to another provider/model in Settings or with /model."
)
CLI_BILLING_ERROR_PAYLOAD = {
    "kind": "error",
    "data": {
        "type": "billing_error",
        "message": CLI_BILLING_ERROR_MESSAGE,
        "provider": "configured",
    },
}


def main(argv: Optional[Iterable[str]] = None) -> int:
    # Dump native stacks on a hard fault (segfault, etc.); idempotent, no overhead.
    try:
        faulthandler.enable()
    except (OSError, ValueError, RuntimeError):
        pass
    args = parse_args(list(argv) if argv is not None else sys.argv[1:])
    try:
        if getattr(args, "version", False):
            return _run_version()
        if args.command == "ask":
            return asyncio.run(_run_ask(args))
        if args.command == "sessions":
            return _run_sessions(args)
        if args.command == "doctor":
            return _run_doctor(args)
        if args.command == "prompts":
            return _run_prompts(args)
        if args.command == "mcp":
            return asyncio.run(_run_mcp(args))
        if args.command == "update":
            return _run_update()
        if args.command == "tui":
            return _run_tui(args)
        return _run_tui(args)
    except (CliConfigError, SessionStoreError, SettingsStoreError, ValueError) as exc:
        _print_styled(f"kolega-code: {exc}", style="error", stderr=True)
        return 2
    except KeyboardInterrupt:
        _print_styled("\nInterrupted.", style="warning", stderr=True)
        return 130


def _make_console(stderr: bool = False):
    """Build a themed rich Console, or None when rich is unavailable.

    rich is only a transitive dependency via textual, so plain installs
    without the [cli] extra fall back to unstyled print output.
    """
    try:
        from rich.console import Console

        from .theme import apply_theme, build_rich_theme
    except ImportError:
        return None
    # Apply the persisted theme so plain-CLI output matches the TUI palette.
    try:
        from .settings import SettingsStore

        apply_theme(SettingsStore().load().active_theme)
    except Exception:
        pass
    return Console(theme=build_rich_theme(), stderr=stderr)


def _print_styled(text: str, style: Optional[str] = None, stderr: bool = False) -> None:
    console = _make_console(stderr=stderr)
    if console is None:
        return
    console.print(text, style=style, highlight=False, markup=False, soft_wrap=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    if argv and argv[0] in {"--help", "-h"}:
        return _build_subcommand_parser().parse_args(argv)
    if argv and argv[0] in SUBCOMMANDS:
        return _build_subcommand_parser().parse_args(argv)
    return _build_tui_parser().parse_args(argv)


def _add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", help="Provider for the main coding model.")
    parser.add_argument("--model", help="Main coding model.")
    parser.add_argument("--fast-provider", help="Provider for fast utility calls.")
    parser.add_argument("--fast-model", help="Fast utility model.")
    parser.add_argument("--thinking-provider", help="Provider for think-hard operations.")
    parser.add_argument("--thinking-model", help="Model for think-hard operations.")
    parser.add_argument("--thinking-effort", help="Model-specific thinking effort for the active model.")
    parser.add_argument("--thinking-tokens", dest="deprecated_thinking_tokens", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--environment", help="Environment label for tracing/metadata.")


def _add_session_args(parser: argparse.ArgumentParser, session_help: str = "Session ID to resume or create.") -> None:
    parser.add_argument("--state-dir", type=Path, help="Directory for CLI session state.")
    parser.add_argument("--session", help=session_help)


def _add_tui_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--version", action="store_true", help="Show the Kolega Code version.")
    parser.add_argument("project_path", nargs="?", default=".", type=Path, help="Project directory to work in.")
    parser.add_argument(
        "--mode", choices=[mode.value for mode in AgentMode], default=CLI_AGENT_MODE, help=argparse.SUPPRESS
    )
    parser.add_argument("--new", action="store_true", help="Start a new session. This is now the default.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=RESUME_LATEST,
        metavar="THREAD_ID",
        help="Resume the latest saved thread, or resume the given thread/session ID.",
    )
    parser.add_argument("--browser-visible", action="store_true", help="Launch visible Playwright browser windows.")
    parser.add_argument("--show-logs", action="store_true", help="Show the diagnostic Logs sidebar tab.")
    parser.add_argument(
        "--permission-mode",
        choices=[mode.value for mode in PermissionMode],
        help="How to handle shell command and file edit permissions.",
    )
    parser.add_argument(
        "--trust-hooks",
        action="store_true",
        help="Trust and enable this project's .kolega/hooks.json (persisted for future runs).",
    )
    parser.add_argument(
        "--trust-mcp",
        action="store_true",
        help="Trust and enable this project's .kolega/mcp_servers.json (persisted for future runs).",
    )
    _add_session_args(parser, session_help="Legacy alias for --resume THREAD_ID.")
    _add_common_model_args(parser)


def _build_tui_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kolega-code", description="Run the Kolega Code Textual CLI.")
    parser.set_defaults(command="tui")
    _add_tui_args(parser)
    return parser


def _build_subcommand_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kolega-code",
        description="Kolega Code CLI. Run without a command to start the interactive TUI.",
    )
    parser.add_argument("--version", action="store_true", help="Show the Kolega Code version.")
    subparsers = parser.add_subparsers(dest="command", required=True, title="commands", metavar="command")

    tui = subparsers.add_parser("tui", help="Run the interactive Textual CLI.")
    tui.set_defaults(command="tui")
    _add_tui_args(tui)

    ask = subparsers.add_parser("ask", help="Run a single prompt and print the answer.")
    ask.add_argument("prompt", help="Prompt to send to Kolega Code.")
    ask.add_argument("--project", default=".", type=Path, help="Project directory to work in.")
    ask.add_argument(
        "--mode", choices=[mode.value for mode in AgentMode], default=CLI_AGENT_MODE, help=argparse.SUPPRESS
    )
    ask.add_argument("--save", action="store_true", help="Persist the session after the prompt completes.")
    ask.add_argument("--json", action="store_true", help="Emit response chunks and events as JSON.")
    ask.add_argument("--browser-visible", action="store_true", help="Launch visible Playwright browser windows.")
    ask.add_argument(
        "--permission-mode",
        choices=[mode.value for mode in PermissionMode],
        default=ASK_DEFAULT_PERMISSION_MODE,
        help="How to handle shell command and file edit permissions.",
    )
    ask.add_argument(
        "--trust-hooks",
        action="store_true",
        help="Trust and enable this project's .kolega/hooks.json (persisted for future runs).",
    )
    ask.add_argument(
        "--trust-mcp",
        action="store_true",
        help="Trust and enable this project's .kolega/mcp_servers.json (persisted for future runs).",
    )
    ask.add_argument(
        "--image",
        action="append",
        default=[],
        type=Path,
        help="Attach an image file to the prompt (repeatable).",
    )
    _add_session_args(ask)
    _add_common_model_args(ask)

    sessions = subparsers.add_parser("sessions", help="Manage local CLI sessions.")
    sessions_sub = sessions.add_subparsers(dest="sessions_command", required=True)
    sessions_list = sessions_sub.add_parser("list", help="List sessions.")
    sessions_list.add_argument("--project", type=Path, help="Filter by project path.")
    sessions_list.add_argument("--state-dir", type=Path, help="Directory for CLI session state.")
    sessions_delete = sessions_sub.add_parser("delete", help="Delete a session.")
    sessions_delete.add_argument("session_id")
    sessions_delete.add_argument("--state-dir", type=Path, help="Directory for CLI session state.")
    sessions_export = sessions_sub.add_parser("export", help="Print a session as JSON.")
    sessions_export.add_argument("session_id")
    sessions_export.add_argument("--output", type=Path, help="Write JSON to a file instead of stdout.")
    sessions_export.add_argument("--state-dir", type=Path, help="Directory for CLI session state.")

    doctor = subparsers.add_parser("doctor", help="Check local CLI configuration.")
    doctor.add_argument("--project", default=".", type=Path, help="Project directory to check.")
    doctor.add_argument("--state-dir", type=Path, help="Directory for CLI session state.")
    _add_common_model_args(doctor)

    prompts = subparsers.add_parser("prompts", help="Manage project prompt override files.")
    prompts_sub = prompts.add_subparsers(dest="prompts_command", required=True)
    prompts_dump = prompts_sub.add_parser("dump", help="Dump editable prompt override starter files.")
    prompts_dump.add_argument(
        "prompt_selectors",
        nargs="*",
        metavar="prompt",
        help="Prompts to dump (coder, planning, general, investigation, browser, compaction, or all).",
    )
    prompts_dump.add_argument("--project", default=".", type=Path, help="Project directory to write prompts into.")
    prompts_dump.add_argument("--force", action="store_true", help="Overwrite existing prompt override files.")
    prompts_list = prompts_sub.add_parser("list", help="List supported prompt override files.")
    prompts_list.add_argument("--project", default=".", type=Path, help="Project directory to inspect.")
    prompts_validate = prompts_sub.add_parser("validate", help="Validate existing prompt override files.")
    prompts_validate.add_argument("--project", default=".", type=Path, help="Project directory to inspect.")

    mcp = subparsers.add_parser("mcp", help="Manage MCP servers and verification state.")
    mcp.add_argument(
        "--project", default=".", type=Path, help="Project directory to use for trusted project MCP config."
    )
    mcp.add_argument("--state-dir", type=Path, help="Directory for CLI state and global MCP config.")
    mcp.add_argument(
        "--trust-mcp",
        action="store_true",
        help="Trust this project's .kolega/mcp_servers.json before running the command.",
    )
    mcp_sub = mcp.add_subparsers(dest="mcp_command", required=True)
    mcp_sub.add_parser("list", help="List configured MCP servers and verification status.")
    verify = mcp_sub.add_parser("verify", help="Verify one MCP server, or all enabled servers with --all.")
    verify.add_argument("server_id", nargs="?", help="MCP server id to verify.")
    verify.add_argument("--all", action="store_true", help="Verify all enabled MCP servers.")
    verify.add_argument("--yes", action="store_true", help="Confirm starting stdio MCP commands without prompting.")
    verify.add_argument("--no-browser", action="store_true", help="Print OAuth URL without opening a browser.")
    verify.add_argument("--json", action="store_true", help="Print verification results as JSON.")
    add = mcp_sub.add_parser("add", help="Add or update an MCP server in global config (or project config).")
    add.add_argument("server_id")
    add.add_argument("--project-config", action="store_true", help="Write to <project>/.kolega/mcp_servers.json.")
    add.add_argument("--name")
    add.add_argument("--transport", choices=["streamable_http", "sse", "stdio"], required=True)
    add.add_argument("--url", help="HTTP MCP endpoint for streamable_http or sse transports.")
    add.add_argument("--header", action="append", default=[], help="HTTP header as Name=Value (repeatable).")
    add.add_argument("--command", dest="stdio_command", help="Command for stdio transport.")
    add.add_argument("--arg", action="append", default=[], help="Argument for stdio command (repeatable).")
    add.add_argument("--env", action="append", default=[], help="Environment variable as NAME=VALUE (repeatable).")
    add.add_argument("--cwd", help="Working directory for stdio command; relative paths resolve under project.")
    add.add_argument("--oauth", action="store_true", help="Enable OAuth for this HTTP MCP server.")
    add.add_argument("--oauth-scope", help="OAuth scopes to request.")
    add.add_argument("--redirect-uri", help="OAuth redirect URI; defaults to an ephemeral localhost callback.")
    add.add_argument("--disabled", action="store_true", help="Add the server disabled.")
    remove = mcp_sub.add_parser("remove", help="Remove an MCP server from global or project config.")
    remove.add_argument("server_id")
    remove.add_argument("--project-config", action="store_true", help="Remove from <project>/.kolega/mcp_servers.json.")
    enable = mcp_sub.add_parser("enable", help="Enable an MCP server in global or project config.")
    enable.add_argument("server_id")
    enable.add_argument("--project-config", action="store_true", help="Update <project>/.kolega/mcp_servers.json.")
    disable = mcp_sub.add_parser("disable", help="Disable an MCP server in global or project config.")
    disable.add_argument("server_id")
    disable.add_argument("--project-config", action="store_true", help="Update <project>/.kolega/mcp_servers.json.")

    subparsers.add_parser("update", help="Update Kolega Code to the latest version.")

    return parser


def _overrides_from_args(args: argparse.Namespace) -> CliConfigOverrides:
    if getattr(args, "deprecated_thinking_tokens", None) is not None:
        raise CliConfigError(DEPRECATED_THINKING_TOKENS_MESSAGE)
    return CliConfigOverrides(
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        fast_provider=getattr(args, "fast_provider", None),
        fast_model=getattr(args, "fast_model", None),
        thinking_provider=getattr(args, "thinking_provider", None),
        thinking_model=getattr(args, "thinking_model", None),
        thinking_effort=getattr(args, "thinking_effort", None),
        environment=getattr(args, "environment", None),
    )


def _store_from_args(args: argparse.Namespace) -> SessionStore:
    return SessionStore(root=getattr(args, "state_dir", None))


def _settings_store_from_args(args: argparse.Namespace) -> SettingsStore:
    return SettingsStore(root=getattr(args, "state_dir", None))


def _validate_project(project_path: Path) -> Path:
    project_path = project_path.expanduser().resolve()
    if not project_path.exists():
        raise ValueError(f"Project path does not exist: {project_path}")
    if not project_path.is_dir():
        raise ValueError(f"Project path is not a directory: {project_path}")
    return project_path


def _get_or_create_session(
    store: SessionStore,
    project_path: Path,
    mode: str,
    summary: dict,
    session_id: Optional[str],
    force_new: bool = False,
) -> SessionRecord:
    if session_id and not force_new:
        try:
            return store.load(session_id)
        except SessionStoreError:
            return store.create(project_path, mode, summary, session_id=session_id)

    if not force_new:
        latest = store.latest_for_project(project_path)
        if latest:
            return latest

    return store.create(project_path, mode, summary, session_id=session_id)


def _validate_session_project(session: SessionRecord, project_path: Path) -> SessionRecord:
    resolved_project = str(project_path.resolve())
    if session.project_path != resolved_project:
        raise SessionStoreError(
            f"Session {session.session_id} belongs to project {session.project_path}, not {resolved_project}"
        )
    return session


def _normalize_cli_session_mode(store: SessionStore, session: SessionRecord, *, persist: bool) -> SessionRecord:
    if session.mode != CLI_AGENT_MODE:
        session.mode = CLI_AGENT_MODE
        if persist:
            store.save(session)
    return session


def _resolve_tui_session(
    store: SessionStore,
    project_path: Path,
    summary: dict,
    resume: Optional[str],
    legacy_session_id: Optional[str],
) -> SessionRecord:
    if resume is not None and legacy_session_id:
        raise ValueError("Use either --resume or --session, not both.")

    if legacy_session_id:
        session = _validate_session_project(store.load_session_or_thread(legacy_session_id), project_path)
        return _normalize_cli_session_mode(store, session, persist=True)

    if resume == RESUME_LATEST:
        latest = store.latest_for_project(project_path)
        if latest is None:
            raise SessionStoreError(f"No saved sessions found for project: {project_path}")
        return _normalize_cli_session_mode(store, latest, persist=True)

    if resume:
        session = _validate_session_project(store.load_session_or_thread(resume), project_path)
        return _normalize_cli_session_mode(store, session, persist=True)

    return store.create(project_path, CLI_AGENT_MODE, summary)


def _safe_permission_mode_value(value: Optional[str]) -> str:
    try:
        return normalize_permission_mode(value, default=PermissionMode.ASK).value
    except ValueError:
        return PermissionMode.ASK.value


def _resolve_tui_permission_mode(
    session: SessionRecord,
    settings: CliSettings,
    requested_permission_mode: Optional[str],
    *,
    resumed: bool,
) -> str:
    """Resolve the TUI permission mode for this launch.

    Precedence: explicit CLI flag, resumed session value, then global setting for
    new sessions. Invalid legacy values fall back to ask.
    """
    if requested_permission_mode:
        return normalize_permission_mode(requested_permission_mode, default=PermissionMode.ASK).value
    if resumed:
        return _safe_permission_mode_value(session.permission_mode)
    return _safe_permission_mode_value(settings.permission_mode)


def _run_version() -> int:
    result = check_for_update()
    print(f"kolega-code {result.current_version}")
    message = update_status_message(result)
    if message:
        print(message)
    return 0


def _run_update() -> int:
    result = run_self_update()
    if result.error:
        _print_styled(result.error, style="error", stderr=True)
    if result.returncode == 0:
        print("Kolega Code update completed. Run `kolega-code --version` to confirm.")
    elif not result.error:
        _print_styled("Kolega Code update failed.", style="error", stderr=True)
    return result.returncode


def _run_tui(args: argparse.Namespace) -> int:
    if importlib.util.find_spec("textual") is None:
        print("Textual is not installed. Reinstall the CLI with: uv tool install --force kolega-code", file=sys.stderr)
        return 2

    project_path = _validate_project(args.project_path)
    store = _store_from_args(args)
    settings_store = _settings_store_from_args(args)
    settings = settings_store.load()
    settings_changed = False
    if getattr(args, "trust_hooks", False):
        settings.trust_hook_project(project_path)
        settings_changed = True
    if getattr(args, "trust_mcp", False):
        settings.trust_mcp_project(project_path)
        settings_changed = True
    if settings_changed:
        settings_store.save(settings)
    summary = {}
    try:
        config = build_agent_config(
            project_path, _overrides_from_args(args), settings=settings, settings_store=settings_store
        )
        summary = config_summary(config)
    except CliConfigError as exc:
        if str(exc) == DEPRECATED_THINKING_TOKENS_MESSAGE:
            raise
        config = None
    session = _resolve_tui_session(
        store,
        project_path,
        summary,
        args.resume,
        args.session,
    )
    effective_permission_mode = _resolve_tui_permission_mode(
        session,
        settings,
        args.permission_mode,
        resumed=args.resume is not None or bool(args.session),
    )
    if session.permission_mode != effective_permission_mode:
        session.permission_mode = effective_permission_mode
        store.save(session)

    from .app import KolegaCodeApp

    app = KolegaCodeApp(
        project_path=project_path,
        config=config,
        mode=CLI_AGENT_MODE,
        store=store,
        settings_store=settings_store,
        overrides=_overrides_from_args(args),
        session=session,
        permission_mode=effective_permission_mode,
        browser_visible=args.browser_visible,
        check_for_updates=True,
        show_logs=args.show_logs,
    )
    try:
        app.run()
    except Exception as exc:  # noqa: BLE001 — last-resort crash capture before re-raising
        _secrets = [v for v in getattr(settings, "api_keys", {}).values() if v]
        try:
            from kolega_code.mcp.config import load_mcp_config, mcp_secret_values
            from kolega_code.mcp.state import MCPOAuthTokenStore

            mcp_config = getattr(config, "mcp_config", None) if config is not None else None
            if mcp_config is None:
                mcp_config = load_mcp_config(
                    project_path,
                    settings_store.root,
                    project_trusted=settings.is_mcp_project_trusted(project_path),
                )
            _secrets.extend(mcp_secret_values(mcp_config))
            _secrets.extend(MCPOAuthTokenStore(settings_store.root).secret_values())
        except Exception:
            pass
        path = write_crash_log(
            store.root, exc=exc, header=f"kolega-code crash | session {session.session_id}", secret_values=_secrets
        )
        if path is not None:
            _print_styled(
                f"\nKolega Code hit an unexpected error. Diagnostics (no API keys) saved to:\n  {path}\n"
                "Please share that file when reporting this.",
                style="error",
                stderr=True,
            )
        raise
    return 0


def _permission_callback_for_ask(project_path: Path):
    async def permission_callback(request) -> PermissionDecision:
        store = ProjectPermissionStore(project_path)
        try:
            matched_rule = store.first_match(request)
        except PermissionStoreError as exc:
            print(f"Warning: {exc}", file=sys.stderr)
            matched_rule = None

        if matched_rule is not None:
            return PermissionDecision(allowed=True, reason=f"Allowed by saved rule {matched_rule.id}.")

        if not sys.stdin.isatty():
            return PermissionDecision(
                allowed=False,
                reason="Permission required, but stdin is not interactive.",
            )

        rule_options = allow_rule_options(request)
        print("", file=sys.stderr)
        if request.kind.value == "command":
            print("Allow the agent to run this command?", file=sys.stderr)
            print(f"  {request.command}", file=sys.stderr)
        elif request.kind.value == "mcp":
            print("Allow the agent to call this MCP tool?", file=sys.stderr)
            print(f"  server: {request.mcp_server}", file=sys.stderr)
            print(f"  tool:   {request.mcp_tool}", file=sys.stderr)
        else:
            target = f" on {request.path}" if request.path else ""
            print(f"Allow the agent to run {request.tool_name}{target}?", file=sys.stderr)

        labels = ["Allow once", "Deny", *(option.label for option in rule_options)]
        for index, label in enumerate(labels, start=1):
            print(f"  {index}. {label}", file=sys.stderr)

        while True:
            print("Choose an option: ", end="", file=sys.stderr, flush=True)
            choice = (await asyncio.to_thread(sys.stdin.readline)).strip()
            if not choice:
                continue
            if not choice.isdigit():
                print("Enter a number from the list.", file=sys.stderr)
                continue
            option_index = int(choice) - 1
            if option_index < 0 or option_index >= len(labels):
                print("Enter a number from the list.", file=sys.stderr)
                continue
            break

        if option_index == 0:
            return PermissionDecision(allowed=True, reason="Allowed once by the user.")
        if option_index == 1:
            return PermissionDecision(allowed=False, reason="Denied by the user.")

        rule = rule_options[option_index - 2].rule
        try:
            store.add_rule(rule)
        except PermissionStoreError as exc:
            print(f"Warning: {exc}", file=sys.stderr)
            return PermissionDecision(allowed=True, reason="Allowed once because the rule could not be saved.")
        return PermissionDecision(allowed=True, reason="Allowed by a saved rule.", rule=rule)

    return permission_callback


async def _run_ask(args: argparse.Namespace) -> int:
    project_path = _validate_project(args.project)
    skill_catalog = discover_skills(project_path)
    skill_command = _parse_skill_prompt(args.prompt, skill_catalog)

    if skill_command and skill_command[0] == "skills":
        if args.json:
            print(json.dumps({"kind": "skills", "data": skill_catalog.format_catalog()}, default=str))
        else:
            print(skill_catalog.format_catalog())
        return 0

    if skill_command and skill_command[0] != "skills" and not skill_command[1] and not (args.save or args.session):
        activation_content = skill_catalog.activation_content(skill_command[0])
        if args.json:
            print(
                json.dumps(
                    {
                        "kind": "skill",
                        "data": {
                            "name": skill_command[0],
                            "content": activation_content,
                        },
                    },
                    default=str,
                )
            )
        else:
            print(activation_content)
        return 0

    store = _store_from_args(args)
    settings_store = _settings_store_from_args(args)
    settings = settings_store.load()
    settings_changed = False
    if getattr(args, "trust_hooks", False):
        settings.trust_hook_project(project_path)
        settings_changed = True
    if getattr(args, "trust_mcp", False):
        settings.trust_mcp_project(project_path)
        settings_changed = True
    if settings_changed:
        settings_store.save(settings)
    config = build_agent_config(
        project_path, _overrides_from_args(args), settings=settings, settings_store=settings_store
    )
    summary = config_summary(config)

    hook_config = load_hook_config(
        project_path, settings_store.root, project_trusted=settings.is_hook_project_trusted(project_path)
    )
    hook_dispatcher = HookDispatcher(hook_config)
    if not args.json:
        for diagnostic in hook_config.diagnostics:
            print(f"hooks: {diagnostic}", file=sys.stderr)

    if args.session:
        session = _get_or_create_session(store, project_path, CLI_AGENT_MODE, summary, args.session, force_new=False)
        session = _normalize_cli_session_mode(store, session, persist=True)
    elif args.save:
        session = store.create(project_path, CLI_AGENT_MODE, summary)
    else:
        session = SessionRecord.create(project_path, CLI_AGENT_MODE, summary)

    manager = CliConnectionManager()
    browser_manager = PlaywrightBrowserManager()
    browser_manager.headless = not args.browser_visible
    agent_ref: dict[str, CoderAgent] = {}
    prompt_extensions = []
    tool_extensions = []
    skill_prompt_extension = build_skill_prompt_extension(skill_catalog)
    skill_tool_extension = build_skill_tool_extension(
        skill_catalog,
        lambda: agent_ref["agent"].history if "agent" in agent_ref else [],
    )
    if skill_prompt_extension is not None:
        prompt_extensions.append(skill_prompt_extension)
    if skill_tool_extension is not None:
        tool_extensions.append(skill_tool_extension)
    mcp_config = getattr(config, "mcp_config", None)
    if not args.json and mcp_config is not None:
        for diagnostic in getattr(mcp_config, "diagnostics", []) or []:
            print(f"mcp: {diagnostic}", file=sys.stderr)
    mcp_extension = build_mcp_tool_extension(
        project_path,
        settings_store.root,
        project_trusted=settings.is_mcp_project_trusted(project_path),
        loaded_config=mcp_config,
    )
    if mcp_extension is not None:
        tool_extensions.append(mcp_extension)
    permission_mode = normalize_permission_mode(
        getattr(args, "permission_mode", ASK_DEFAULT_PERMISSION_MODE),
        default=PermissionMode.AUTO,
    )
    agent = CoderAgent(
        project_path=project_path,
        workspace_id=session.workspace_id,
        thread_id=session.thread_id,
        connection_manager=manager,
        config=config,
        browser_manager=browser_manager,
        agent_mode=AgentMode.CLI,
        prompt_extensions=prompt_extensions,
        tool_extensions=tool_extensions,
        permission_mode=permission_mode,
        permission_callback=_permission_callback_for_ask(project_path)
        if permission_mode == PermissionMode.ASK
        else None,
        hook_dispatcher=hook_dispatcher,
    )
    agent_ref["agent"] = agent
    if session.history:
        agent.restore_message_history(session.history)
        agent.restore_compaction_state(session.compaction)

    fire_hook = getattr(agent, "fire_hook", None)
    if fire_hook is not None:
        session_start = await fire_hook(HookEvent.SESSION_START, {"source": "startup"})
        if session_start.additional_context:
            agent.append_user_message([TextBlock(text=session_start.additional_context)])

    prompt = args.prompt
    if skill_command:
        skill_name, skill_prompt = skill_command
        active_names = activated_skill_names(agent.history)
        activation_content = skill_catalog.activation_content(skill_name, active_names=active_names)
        if skill_name not in active_names:
            agent.append_user_message([TextBlock(text=activation_content)])
        if args.json:
            print(
                json.dumps(
                    {
                        "kind": "skill",
                        "data": {
                            "name": skill_name,
                            "already_active": skill_name in active_names,
                        },
                    },
                    default=str,
                )
            )
        prompt = skill_prompt
        if not prompt:
            if args.json:
                print(json.dumps({"kind": "chunk", "data": {"type": "response", "content": activation_content}}))
            else:
                print(activation_content)
            if args.save or args.session:
                session.history = agent.dump_message_history()
                session.compaction = agent.dump_compaction_state()
                session.config = summary
                store.save(session)
            await agent.cleanup()
            return 0

    attachments, unresolved_mentions = build_file_attachments(prompt, project_path)
    for mention in unresolved_mentions:
        print(f"Note: @{mention} not found, sent as plain text", file=sys.stderr)
    for image_path in getattr(args, "image", None) or []:
        encoded = encode_image_file(image_path)
        if encoded is not None:
            attachments.append(encoded)
        else:
            print(
                f"Warning: --image {image_path} could not be attached (not a supported image, missing, or too large)",
                file=sys.stderr,
            )

    response_chunks: list[dict] = []
    exit_code = 0
    # Pump connection-manager events concurrently so sub-agent activity is
    # reported in real time instead of all at once after streaming finishes.
    pump_task = asyncio.create_task(_pump_ask_events(manager, args.json))
    try:
        stream = (
            agent.process_message_stream(prompt, attachments) if attachments else agent.process_message_stream(prompt)
        )
        async for chunk in stream:
            response_chunks.append(chunk)
            if args.json:
                print(json.dumps({"kind": "chunk", "data": chunk}, default=str))
            elif chunk.get("type") == "response" and chunk.get("content"):
                print(chunk["content"], end="" if not chunk.get("complete") else "\n")

        if args.save or args.session:
            session.history = agent.dump_message_history()
            session.compaction = agent.dump_compaction_state()
            session.config = summary
            store.save(session)
    except LLMBillingError as exc:
        exit_code = 1
        if args.json:
            json.dump(CLI_BILLING_ERROR_PAYLOAD, sys.stdout, default=str)
            print()
        else:
            message = billing_error_message(exc, model=config.long_context_config.model)
            _print_styled(message, style="error", stderr=True)
    finally:
        pump_task.cancel()
        try:
            await pump_task
        except asyncio.CancelledError:
            pass
        while not manager.events.empty():
            event = manager.events.get_nowait()
            _print_ask_event(event, args.json)
        end_fire_hook = getattr(agent, "fire_hook", None)
        if end_fire_hook is not None:
            try:
                await end_fire_hook(HookEvent.SESSION_END, {"reason": "ask_complete"})
            except Exception:
                pass
        await agent.cleanup()

    if exit_code:
        return exit_code

    if args.json:
        print(json.dumps({"kind": "summary", "chunks": len(response_chunks), "session_id": session.session_id}))
    return 0


async def _pump_ask_events(manager: CliConnectionManager, json_mode: bool) -> None:
    while True:
        event = await manager.next_event()
        _print_ask_event(event, json_mode)


def _print_ask_event(event, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps({"kind": "event", "data": event.model_dump()}, default=str))
        return

    # Plain mode: keep piped stdout as the pure answer; report concise
    # sub-agent lifecycle and tool activity on stderr.
    info = event.sub_agent_info
    if not info:
        return
    from . import theme
    from .theme import Glyph

    name = info.get("agent_name", event.sender)
    sep = theme.g(Glyph.BULLET_SEP)
    content = event.content
    status = content.get("status")
    message_type = content.get("message_type")
    if status:
        line = f"{theme.g(Glyph.SUB_AGENT)} {name} {sep} {str(status).lower()} {sep} {content.get('message', '')}"
        _print_styled(line.rstrip(f" {sep}"), style="muted", stderr=True)
    elif message_type in {"tool_call", "tool_error"}:
        tool = content.get("tool_description") or content.get("tool_name") or "tool"
        state = "failed" if message_type == "tool_error" else "running"
        _print_styled(f"{theme.g(Glyph.TOOL)} {tool} {sep} {state}", style="muted", stderr=True)
    # Streamed response chunks are suppressed in plain mode.


def _parse_skill_prompt(prompt: str, catalog: SkillCatalog) -> Optional[tuple[str, str]]:
    stripped = prompt.strip()
    if not stripped.startswith("/"):
        return None

    command_text, _, rest = stripped.partition(" ")
    command = command_text.lower()
    if command == SKILLS_LIST_COMMAND:
        return "skills", rest.strip()
    if command in agent_command_names():
        return None

    skill_name = command.removeprefix("/")
    if catalog.get(skill_name) is None:
        return None
    return skill_name, rest.strip()


def _run_sessions(args: argparse.Namespace) -> int:
    store = _store_from_args(args)
    if args.sessions_command == "list":
        project = args.project.expanduser().resolve() if args.project else None
        records = store.list(project_path=project)
        for record in records:
            print(
                f"{record.session_id}\t{record.thread_id}\t{record.updated_at}\t"
                f"{record.mode}\t{record.project_path}\t{record.title}"
            )
        return 0
    if args.sessions_command == "delete":
        store.delete(args.session_id)
        print(f"Deleted session {args.session_id}")
        return 0
    if args.sessions_command == "export":
        payload = store.export(args.session_id)
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
        return 0
    raise ValueError(f"Unknown sessions command: {args.sessions_command}")


def _run_prompts(args: argparse.Namespace) -> int:
    project_path = _validate_project(args.project)
    if args.prompts_command == "dump":
        result = dump_prompt_overrides(
            project_path,
            force=bool(args.force),
            selectors=getattr(args, "prompt_selectors", None),
        )
        print(format_prompt_dump_result(result))
        return 0 if result.ok else 1
    if args.prompts_command == "list":
        result = list_prompt_overrides(project_path)
        print(format_prompt_list_result(result))
        return 0
    if args.prompts_command == "validate":
        result = validate_prompt_overrides(project_path)
        print(format_prompt_validation_result(result))
        return 0 if result.ok else 1
    raise ValueError(f"Unknown prompts command: {args.prompts_command}")


async def _run_mcp(args: argparse.Namespace) -> int:
    project_path = _validate_project(args.project)
    settings_store = _settings_store_from_args(args)
    settings = settings_store.load()
    if getattr(args, "trust_mcp", False):
        settings.trust_mcp_project(project_path)
        settings_store.save(settings)

    config = load_mcp_config(
        project_path,
        settings_store.root,
        project_trusted=settings.is_mcp_project_trusted(project_path),
    )
    service = MCPService(config, state_dir=settings_store.root, project_path=project_path)

    if args.mcp_command == "list":
        _print_mcp_list(config, service)
        return 0

    if args.mcp_command == "verify":
        server_ids = _mcp_verify_server_ids(args, config)
        if not server_ids:
            raise ValueError("No MCP servers to verify.")
        if not _confirm_stdio_verification(args, config, server_ids):
            return 2
        results = []
        for server_id in server_ids:
            results.append(
                await service.verify_server(
                    server_id,
                    interactive_oauth=True,
                    open_browser=not getattr(args, "no_browser", False),
                    output=sys.stderr,
                )
            )
        if getattr(args, "json", False):
            print(json.dumps([result.__dict__ for result in results], default=str))
        else:
            for result in results:
                glyph = "✓" if result.ok else "✗"
                print(f"{glyph} {result.server_id}: {result.message}")
        return 0 if all(result.ok for result in results) else 1

    if args.mcp_command == "add":
        path, source = _mcp_mutation_target(args, project_path, settings_store.root)
        server = _server_config_from_add_args(args)
        try:
            upsert_server_config(path, server, source=source)
        except MCPConfigError as exc:
            raise ValueError(str(exc)) from exc
        print(f"Saved MCP server {server.id} to {path}")
        if source == "project" and not settings.is_mcp_project_trusted(project_path):
            print("Project MCP config is not trusted yet. Re-run with --trust-mcp to enable it.", file=sys.stderr)
        return 0

    if args.mcp_command == "remove":
        path, source = _mcp_mutation_target(args, project_path, settings_store.root)
        try:
            removed = remove_server_config(path, args.server_id, source=source)
        except MCPConfigError as exc:
            raise ValueError(str(exc)) from exc
        if not removed:
            raise ValueError(f"MCP server not found in {path}: {args.server_id}")
        service.status_store.clear(args.server_id)
        service.oauth_store.clear(args.server_id)
        print(f"Removed MCP server {args.server_id} from {path}")
        return 0

    if args.mcp_command in {"enable", "disable"}:
        path, source = _mcp_mutation_target(args, project_path, settings_store.root)
        enabled = args.mcp_command == "enable"
        try:
            changed = set_server_enabled(path, args.server_id, enabled, source=source)
        except MCPConfigError as exc:
            raise ValueError(str(exc)) from exc
        if not changed:
            raise ValueError(f"MCP server not found in {path}: {args.server_id}")
        print(f"{'Enabled' if enabled else 'Disabled'} MCP server {args.server_id} in {path}")
        return 0

    raise ValueError(f"Unknown mcp command: {args.mcp_command}")


def _print_mcp_list(config, service: MCPService) -> None:
    for diagnostic in config.diagnostics:
        print(f"mcp: {diagnostic}", file=sys.stderr)
    if not config.servers:
        print("No MCP servers configured.")
        print(f"Global config: {config.global_path}")
        if config.project_config_path:
            print(
                f"Project config: {config.project_config_path} ({'trusted' if config.project_trusted else 'untrusted'})"
            )
        return
    print("ID\tSOURCE\tTRANSPORT\tENABLED\tOAUTH\tSTATUS\tTOOLS\tMESSAGE")
    for row in service.list_status_rows():
        message = _mcp_cli_list_message(row)
        print(
            f"{row['id']}\t{row['source']}\t{row['transport']}\t{row['enabled']}\t{row['oauth']}\t"
            f"{row['status']}\t{row['tool_count']}\t{message}"
        )


def _mcp_cli_list_message(row: dict[str, object]) -> str:
    status = str(row.get("status", "unverified"))
    if status == "verified":
        return f"Verified {row.get('tool_count', 0)} tool(s)."
    if status == "stale":
        return "Configuration changed since last verification. Verify again."
    if status == "failed":
        return MCP_FAILURE_MESSAGE_GENERIC
    return "Not verified."


def _mcp_verify_server_ids(args: argparse.Namespace, config) -> list[str]:
    if args.all and args.server_id:
        raise ValueError("Use either `mcp verify SERVER_ID` or `mcp verify --all`, not both.")
    if args.all:
        return [server.id for server in config.enabled_servers]
    if not args.server_id:
        raise ValueError("Specify an MCP server id or --all.")
    return [args.server_id]


def _confirm_stdio_verification(args: argparse.Namespace, config, server_ids: list[str]) -> bool:
    stdio_servers = [
        config.servers[server_id]
        for server_id in server_ids
        if config.servers.get(server_id) and config.servers[server_id].transport == "stdio"
    ]
    if not stdio_servers:
        return True
    if getattr(args, "yes", False):
        return True
    if not sys.stdin.isatty():
        print("Refusing to start stdio MCP server command(s) without --yes in non-interactive mode.", file=sys.stderr)
        return False
    print("Verifying stdio MCP servers starts local commands:", file=sys.stderr)
    for server in stdio_servers:
        command = " ".join([server.command or "", *server.args]).strip()
        print(f"  {server.id}: {command}", file=sys.stderr)
    print("Continue? [y/N] ", end="", file=sys.stderr, flush=True)
    answer = sys.stdin.readline().strip().lower()
    return answer in {"y", "yes"}


def _mcp_mutation_target(args: argparse.Namespace, project_path: Path, state_dir: Path) -> tuple[Path, str]:
    if getattr(args, "project_config", False):
        return project_mcp_config_path(project_path), "project"
    return global_mcp_config_path(state_dir), "global"


def _server_config_from_add_args(args: argparse.Namespace) -> MCPServerConfig:
    headers = _parse_key_value_options(getattr(args, "header", []) or [], "--header")
    env = _parse_key_value_options(getattr(args, "env", []) or [], "--env")
    payload: dict[str, Any] = {
        "id": args.server_id,
        "name": args.name,
        "transport": args.transport,
        "enabled": not bool(args.disabled),
        "url": args.url,
        "headers": headers,
        "command": getattr(args, "stdio_command", None),
        "args": getattr(args, "arg", []) or [],
        "env": env,
        "cwd": args.cwd,
        "oauth": {
            "enabled": bool(args.oauth),
            "scope": args.oauth_scope,
            "redirect_uri": args.redirect_uri,
        },
    }
    return MCPServerConfig.model_validate(payload)


def _parse_key_value_options(values: list[str], flag_name: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"{flag_name} values must be NAME=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{flag_name} values must include a non-empty name")
        parsed[key] = value
    return parsed


def _run_doctor(args: argparse.Namespace) -> int:
    from . import theme
    from .theme import Glyph

    console = _make_console()

    def line(label: str, value: object, value_style: Optional[str] = None) -> None:
        if console is None:
            print(f"{label}: {value}")
            return
        from rich.text import Text

        text = Text()
        text.append(f"{label}: ", style="muted")
        text.append(str(value), style=value_style or "")
        console.print(text, highlight=False, soft_wrap=True)

    project_path = _validate_project(args.project)
    store = _store_from_args(args)
    settings_store = _settings_store_from_args(args)
    settings = settings_store.load()
    line("Project", project_path)
    line("State dir", store.root)
    textual_installed = importlib.util.find_spec("textual") is not None
    line("Textual installed", textual_installed, "success" if textual_installed else "warning")
    update_message = update_status_message(check_for_update(), include_up_to_date=True, include_errors=True)
    if update_message:
        line("Update", update_message)
    if settings.active_provider and settings.active_model:
        line("Stored active model", f"{settings.active_provider}/{settings.active_model}")
        line("Stored thinking effort", settings.active_thinking_effort or "model default")
    else:
        line("Stored active model", "not configured", "warning")

    try:
        config = build_agent_config(
            project_path, _overrides_from_args(args), settings=settings, settings_store=settings_store
        )
    except CliConfigError as exc:
        _print_styled(f"{theme.g(Glyph.CROSS)} Configuration: invalid ({exc})", style="error")
        return 2

    summary = config_summary(config)
    _print_styled(f"{theme.g(Glyph.CHECK)} Configuration: valid", style="success")
    override_message = active_model_override_message(config, project_path, _overrides_from_args(args), settings)
    if override_message:
        line("Override", override_message, "warning")
    line("Long model", f"{summary['long_provider']}/{summary['long_model']}")
    line("Fast model", f"{summary['fast_provider']}/{summary['fast_model']}")
    line("Thinking model", f"{summary['thinking_provider']}/{summary['thinking_model']}")
    line("Thinking effort", summary["thinking_effort"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
