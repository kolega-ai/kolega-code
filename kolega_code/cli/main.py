"""Entrypoint for the Kolega Code CLI."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import faulthandler
import importlib.util
import json
import logging
import sys
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Iterable, Optional

from kolega_code.agent import CoderAgent
from kolega_code.config import EditProtocol
from kolega_code.extensions import (
    ExtensionSelection,
    KolegaExtensionHost,
    KolegaExtensionLoadError,
    bind_extension_agent,
    cleanup_extension_bundle,
    create_extension_bundle,
    resolve_extension_selection,
)
from kolega_code.llm.ledger import UsageLedger
from kolega_code.llm.specs.custom_endpoints import CUSTOM_THINKING_PRESETS
from kolega_code.llm.specs.thinking import REASONING_REPLAY_VALUES

from .ask_output import InMemorySessionJournal, SemanticStdoutPrinter
from .session_journal import SessionRecorder
from .session_usage import SessionUsageSink
from kolega_code.agent.custom_agents import discover_custom_agents, validate_custom_agent_models
from kolega_code.agent.orchestration.guide import gigacode_prompt_extension
from kolega_code.agent.prompt_provider import PromptExtension
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
from kolega_code.llm.models import Message, TextBlock
from kolega_code.mcp.config import (
    _SERVER_ID_MAX_LENGTH,
    LoadedMCPConfig,
    MCPConfigError,
    MCPServerConfig,
    global_mcp_config_path,
    load_mcp_config,
    project_mcp_config_path,
    remove_server_config,
    sanitize_mcp_server_id,
    server_fingerprint,
    set_server_enabled,
    upsert_server_config,
)
from kolega_code.mcp.service import (
    MCP_FAILURE_MESSAGE_GENERIC,
    MCPService,
    mcp_tool_name_adjustment_note,
)
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
from kolega_code.browser_extension.installer import (
    BROWSER_EXTENSION_CHANNELS,
    DEFAULT_BROWSER_EXTENSION_CHANNEL,
    channel_web_store_url,
    default_state_dir,
    install_native_host,
    native_host_status,
    uninstall_native_host,
)
from kolega_code.utils.images import encode_image_file
from kolega_code.worktrees import WorktreeError, create_worktree, resolve_worktree

from . import messages
from .browser_backend import build_browser_manager
from .diagnostics import write_crash_log
from .config import (
    DEPRECATED_THINKING_TOKENS_MESSAGE,
    LSP_MODES,
    SKILL_MODES,
    SUBAGENT_MODES,
    WEB_SEARCH_MODES,
    CliConfigError,
    CliConfigOverrides,
    StrictContextBudgetError,
    _skills_enabled,
    active_model_override_message,
    build_agent_config,
    config_summary,
    load_cli_env,
    strict_context_budget_marker,
)
from .connection import CliConnectionManager
from .mentions import build_file_attachments
from .model_catalog import SORT_CHOICES, apply_all_catalog_overlays, run_models_list, run_models_refresh
from .session_store import SessionRecord, SessionStore, SessionStoreError, resolve_active_project
from .settings import CliSettings, SettingsStore, SettingsStoreError
from .goal import (
    DEFAULT_GOAL_MAX_TURNS,
    GoalState,
    build_goal_nudge,
    build_goal_prompt_extension_markdown,
    build_goal_task_prompt,
    now_iso,
)
from .loop import (
    DEFAULT_LOOP_EXPIRY_DAYS,
    DEFAULT_LOOP_MAX_ITERATIONS,
    PROMPT_SOURCE_INLINE,
    PROMPT_SOURCE_LOOP_MD,
    LoopError,
    LoopState,
    build_loop_iteration_prompt,
    build_loop_prompt_extension_markdown,
    format_duration_short,
    parse_duration,
    parse_schedule_text,
    read_loop_md,
)
from .slash_commands import SKILLS_LIST_COMMAND, agent_command_names
from .skills import (
    SkillCatalog,
    activated_skill_names,
    build_skill_prompt_extension,
    build_skill_tool_extension,
    context_window_tokens_for_skill_budget,
    discover_skills,
)
from .updater import check_for_update, run_self_update, update_status_message

SUBCOMMANDS = {
    "ask",
    "sessions",
    "doctor",
    "update",
    "prompts",
    "agents",
    "browser",
    "mcp",
    "models",
    "tui",
    "share",
}
RESUME_LATEST = "__latest__"
CLI_AGENT_MODE = AgentMode.CLI.value
ASK_DEFAULT_PERMISSION_MODE = PermissionMode.AUTO.value
CLI_BILLING_ERROR_MESSAGE = (
    "The selected provider could not run this request because it reported insufficient balance. "
    "Add credits to the provider account or switch to another provider/model in Settings or with /model."
)


def main(argv: Optional[Iterable[str]] = None) -> int:
    # Dump native stacks on a hard fault (segfault, etc.); idempotent, no overhead.
    try:
        faulthandler.enable()
    except (OSError, ValueError, RuntimeError):
        pass
    args = parse_args(list(argv) if argv is not None else sys.argv[1:])
    # Merge any cached provider catalogs before a command resolves a model.
    # This only reads local files; refreshing them is an explicit `models refresh`.
    apply_all_catalog_overlays(getattr(args, "state_dir", None))
    try:
        if getattr(args, "version", False):
            return _run_version()
        if args.command == "models":
            return _run_models(args)
        if args.command == "ask":
            return asyncio.run(_run_ask(args))
        if args.command == "sessions":
            return _run_sessions(args)
        if args.command == "share":
            return asyncio.run(_run_share(args))
        if args.command == "doctor":
            return _run_doctor(args)
        if args.command == "prompts":
            return _run_prompts(args)
        if args.command == "agents":
            return _run_agents(args)
        if args.command == "browser":
            return _run_browser(args)
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
        parser = _build_subcommand_parser()
    elif argv and argv[0] in SUBCOMMANDS:
        parser = _build_subcommand_parser()
    else:
        parser = _build_tui_parser()
    args = parser.parse_args(argv)
    _validate_worktree_args(parser, args)
    _validate_extension_args(parser, args)
    return args


def _add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", help="Provider for the main coding model.")
    parser.add_argument("--model", help="Main coding model.")
    parser.add_argument("--fast-provider", help="Provider for fast utility calls.")
    parser.add_argument("--fast-model", help="Fast utility model.")
    parser.add_argument("--thinking-effort", help="Model-specific thinking effort for the active model.")
    parser.add_argument("--thinking-tokens", dest="deprecated_thinking_tokens", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--custom-endpoints",
        metavar="JSON",
        help="Custom endpoint definitions as a JSON object, merged over settings.json per endpoint id. Not persisted.",
    )
    parser.add_argument(
        "--endpoint-url",
        metavar="URL",
        help="Base URL of an ephemeral custom endpoint (provider 'custom:cli'). When no --provider is set "
        "and a model is named, this endpoint becomes the active provider. Not persisted.",
    )
    parser.add_argument(
        "--endpoint-style",
        choices=["openai_chat", "openai_responses", "anthropic"],
        help="Wire style for --endpoint-url (default openai_chat).",
    )
    parser.add_argument("--endpoint-api-key", help="Optional credential for --endpoint-url.")
    parser.add_argument(
        "--endpoint-context", metavar="TOKENS", help="Context length for --endpoint-url (default 32768)."
    )
    parser.add_argument(
        "--endpoint-max-output", metavar="TOKENS", help="Max output tokens for --endpoint-url (default 8192)."
    )
    parser.add_argument("--endpoint-vision", action="store_true", help="Mark --endpoint-url models as vision-capable.")
    parser.add_argument(
        "--endpoint-thinking",
        choices=sorted(CUSTOM_THINKING_PRESETS),
        help="Thinking-effort mode for --endpoint-url.",
    )
    parser.add_argument(
        "--endpoint-reasoning",
        choices=list(REASONING_REPLAY_VALUES),
        help="Reasoning replay field for --endpoint-url (auto detects the emitted field).",
    )
    parser.add_argument(
        "--endpoint-temperature",
        metavar="T",
        help="Sampling temperature for --endpoint-url (0-2; default 1.0).",
    )
    parser.add_argument("--environment", help="Environment label for tracing/metadata.")
    parser.add_argument(
        "--compression-threshold",
        metavar="PERCENT",
        help="Input-budget usage percentage that triggers automatic history compression "
        "(10-100; default 95). Overrides settings.json for this session; not persisted. "
        "100 disables proactive compression; over-limit recovery and explicit context caps remain enforced.",
    )
    parser.add_argument(
        "--context-window-tokens",
        metavar="TOKENS",
        help="Total input-plus-output context-window limit for this process. "
        "Must be used with --max-output-tokens; may lower but not exceed the "
        "model's catalogued context length. Not persisted.",
    )
    parser.add_argument(
        "--max-output-tokens",
        metavar="TOKENS",
        help="Maximum generated tokens reserved for a primary model call in "
        "this process. Must be used with --context-window-tokens; may lower "
        "but not exceed the model's catalogued completion maximum.",
    )
    parser.add_argument(
        "--edit-protocol",
        choices=[protocol.value for protocol in EditProtocol],
        help="File edit protocol exposed to coding models.",
    )


def _add_session_args(parser: argparse.ArgumentParser, session_help: str = "Session ID to resume or create.") -> None:
    parser.add_argument("--state-dir", type=Path, help="Directory for CLI session state.")
    parser.add_argument("--session", help=session_help)


def _add_worktree_args(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--worktree",
        metavar="PATH_OR_BRANCH",
        help="Start in an existing registered worktree, selected by path or exact branch name.",
    )
    selection.add_argument(
        "--create-worktree",
        metavar="BRANCH",
        help="Create a worktree for BRANCH and start in it.",
    )
    parser.add_argument(
        "--from",
        dest="worktree_from",
        metavar="REF",
        help="Start point for a new --create-worktree branch.",
    )
    parser.add_argument(
        "--worktree-path",
        type=Path,
        metavar="PATH",
        help="Checkout destination for --create-worktree.",
    )


def _validate_worktree_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    create_branch = getattr(args, "create_worktree", None)
    creating = create_branch is not None
    if getattr(args, "worktree_from", None) is not None and not creating:
        parser.error("--from requires --create-worktree")
    if getattr(args, "worktree_path", None) is not None and not creating:
        parser.error("--worktree-path requires --create-worktree")
    if creating and (getattr(args, "resume", None) is not None or getattr(args, "session", None)):
        parser.error("--create-worktree cannot be combined with --resume or --session")


def _add_extension_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--extension",
        metavar="MODULE:FACTORY",
        help="Load one installed Python extension at launch. The factory runs as trusted "
        "arbitrary Python code with the same authority as Kolega Code itself; the module "
        "must already be importable in the active environment.",
    )
    parser.add_argument(
        "--extension-config",
        metavar="PATH",
        type=Path,
        help="Opaque configuration file path passed to the --extension factory. "
        "Kolega Code never reads or interprets its contents.",
    )


def _validate_extension_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if getattr(args, "extension_config", None) is not None and not getattr(args, "extension", None):
        parser.error("--extension-config requires --extension")


def _resolve_extension_selection_from_args(args: argparse.Namespace) -> Optional[ExtensionSelection]:
    """Import the --extension factory once at launch; None when no extension was requested."""
    spec = getattr(args, "extension", None)
    if not spec:
        return None
    return resolve_extension_selection(spec, getattr(args, "extension_config", None))


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
        metavar="SESSION_ID",
        help="Resume the latest saved session, or resume the given session ID (legacy thread IDs are also accepted).",
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
    parser.add_argument(
        "--trust-lsp",
        action="store_true",
        help="Trust and enable this project's .kolega/lsp.json (persisted for future runs).",
    )
    parser.add_argument(
        "--lsp",
        choices=list(LSP_MODES),
        default=None,
        help="Force LSP tools on or off for this session (overrides settings.json; not persisted).",
    )
    parser.add_argument(
        "--subagents",
        choices=list(SUBAGENT_MODES),
        default=None,
        help="Force sub-agent dispatch (dispatch_agent) on or off for this session "
        "(overrides settings.json; not persisted).",
    )
    parser.add_argument(
        "--skills",
        choices=list(SKILL_MODES),
        default=None,
        help="Force Agent Skills on or off for this session (overrides settings.json; not persisted).",
    )
    _add_session_args(parser, session_help="Legacy alias for --resume SESSION_ID.")
    _add_worktree_args(parser)
    _add_common_model_args(parser)
    _add_extension_args(parser)


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
    ask.add_argument("prompt", nargs="?", default=None, help="Prompt to send to Kolega Code.")
    ask.add_argument("--project", default=".", type=Path, help="Project directory to work in.")
    ask.add_argument(
        "--goal",
        default=None,
        help="Set an autonomous completion goal and loop until it is met or capped (no prompt required).",
    )
    ask.add_argument(
        "--goal-max-turns",
        type=int,
        default=None,
        help="Maximum evaluation turns before an unmet --goal gives up (default: 50).",
    )
    loop_schedule_group = ask.add_mutually_exclusive_group()
    loop_schedule_group.add_argument(
        "--loop",
        default=None,
        metavar="INTERVAL",
        help="Re-run the prompt on a fixed interval, e.g. 5m, 2h, 1d. Cannot be combined with --goal.",
    )
    loop_schedule_group.add_argument(
        "--loop-cron",
        default=None,
        metavar="EXPR",
        help='Re-run the prompt on a 5-field cron schedule, e.g. "0 9 * * 1-5".',
    )
    ask.add_argument(
        "--loop-max-iterations",
        type=int,
        default=None,
        help=f"Maximum loop iterations before stopping (default: {DEFAULT_LOOP_MAX_ITERATIONS}).",
    )
    ask.add_argument(
        "--loop-expires",
        default=None,
        metavar="DURATION",
        help=f"Stop the loop this long after it starts (default: {DEFAULT_LOOP_EXPIRY_DAYS}d).",
    )
    ask.add_argument(
        "--loop-fresh",
        action="store_true",
        help="Clear the conversation history before every loop iteration after the first.",
    )
    ask.add_argument(
        "--mode", choices=[mode.value for mode in AgentMode], default=CLI_AGENT_MODE, help=argparse.SUPPRESS
    )
    ask.add_argument(
        "--gigacode",
        action="store_true",
        help="Enable gigacode workflow orchestration (the TUI's /gigacode, for headless runs).",
    )
    ask.add_argument(
        "--no-memory-tools",
        action="store_true",
        help="Disable the persistent project-memory tools (read/list/write/edit/delete memory). "
        "Use for headless or benchmark runs where there is no durable memory store to read or write.",
    )
    ask.add_argument(
        "--web-search",
        choices=list(WEB_SEARCH_MODES),
        default=None,
        help="Web tool mode: auto (hosted server-side search when the model supports it, else the "
        "client web_search/web_fetch tools), hosted, client, or off (no web tools).",
    )
    ask.add_argument(
        "--lsp",
        choices=list(LSP_MODES),
        default=None,
        help="Force LSP tools on or off for this session (overrides settings.json; not persisted).",
    )
    ask.add_argument(
        "--subagents",
        choices=list(SUBAGENT_MODES),
        default=None,
        help="Force sub-agent dispatch (dispatch_agent) on or off for this session "
        "(overrides settings.json; not persisted).",
    )
    ask.add_argument(
        "--skills",
        choices=list(SKILL_MODES),
        default=None,
        help="Force Agent Skills on or off for this session (overrides settings.json; not persisted).",
    )
    ask.add_argument("--save", action="store_true", help="Persist the session after the prompt completes.")
    ask.add_argument("--json", action="store_true", help="Stream the semantic event protocol as JSON lines.")
    ask.add_argument(
        "--atif-output",
        type=Path,
        default=None,
        help=(
            "After the run ends (completed, failed, or cancelled), write a validated ATIF v1.7 "
            "trajectory to this file (image assets to <file-stem>.assets/). Works with or "
            "without --save; stdout is unchanged."
        ),
    )
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
        "--trust-lsp",
        action="store_true",
        help="Trust and enable this project's .kolega/lsp.json (persisted for future runs).",
    )
    ask.add_argument(
        "--image",
        action="append",
        default=[],
        type=Path,
        help="Attach an image file to the prompt (repeatable).",
    )
    _add_session_args(ask)
    _add_worktree_args(ask)
    _add_common_model_args(ask)
    _add_extension_args(ask)

    sessions = subparsers.add_parser("sessions", help="Manage local CLI sessions.")
    sessions_sub = sessions.add_subparsers(dest="sessions_command", required=True)
    sessions_list = sessions_sub.add_parser("list", help="List sessions.")
    sessions_list.add_argument("--project", type=Path, help="Filter by project path.")
    sessions_list.add_argument("--state-dir", type=Path, help="Directory for CLI session state.")
    sessions_delete = sessions_sub.add_parser("delete", help="Delete a session.")
    sessions_delete.add_argument("session_id")
    sessions_delete.add_argument("--state-dir", type=Path, help="Directory for CLI session state.")
    sessions_repair = sessions_sub.add_parser("repair", help="Renumber the events of a corrupted session journal.")
    sessions_repair.add_argument("session_id")
    sessions_repair.add_argument("--state-dir", type=Path, help="Directory for CLI session state.")
    sessions_export = sessions_sub.add_parser("export", help="Export a session (replay JSON or semantic events).")
    sessions_export.add_argument("session_id")
    sessions_export.add_argument(
        "--format",
        choices=("json", "events-jsonl", "atif"),
        default="json",
        help=(
            "json: effective-history replay snapshot (default, unchanged); "
            "events-jsonl: the canonical public semantic event log, one v2 envelope per line; "
            "atif: a validated ATIF v1.7 trajectory (image assets require --output and are "
            "written to <output-stem>.assets/)."
        ),
    )
    sessions_export.add_argument("--output", type=Path, help="Write the export to a file instead of stdout.")
    sessions_export.add_argument("--state-dir", type=Path, help="Directory for CLI session state.")

    share = subparsers.add_parser("share", help="Export a session as a shareable replay.")
    share_sub = share.add_subparsers(dest="share_command", required=True)
    share_export = share_sub.add_parser(
        "export",
        help="Write a self-contained replay file that plays in any browser.",
    )
    share_export.add_argument("session_id")
    share_export.add_argument(
        "--out",
        type=Path,
        help="Destination path (default: ./<session-id>-replay.html).",
    )
    share_export.add_argument(
        "--dir",
        action="store_true",
        help="Write a directory of separate files instead of one HTML file, for hosting on a static site.",
    )
    share_export.add_argument("--zip", action="store_true", help="Write a .zip of the directory form. Implies --dir.")
    share_export.add_argument("--title", help="Human-readable title shown in the player.")
    share_export.add_argument(
        "--theme",
        help="Theme slug to open the replay with (default: the active theme).",
    )
    share_export.add_argument("--state-dir", type=Path, help="Directory for CLI session state.")

    doctor = subparsers.add_parser("doctor", help="Check local CLI configuration.")
    doctor.add_argument("--project", default=".", type=Path, help="Project directory to check.")
    doctor.add_argument("--state-dir", type=Path, help="Directory for CLI session state.")
    _add_common_model_args(doctor)

    models = subparsers.add_parser("models", help="Inspect and refresh the model catalog.")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    models_list = models_sub.add_parser("list", help="List catalogued provider models.")
    models_list.add_argument("--provider", help="Restrict the listing to one provider.")
    models_list.add_argument(
        "--featured",
        action="store_true",
        help="Only models offered in the picker (gateway providers list their most-used models).",
    )
    models_list.add_argument(
        "--sort",
        choices=SORT_CHOICES,
        default="popularity",
        help="popularity keeps catalog order (usage rank for gateways); id sorts alphabetically.",
    )
    models_list.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    models_list.add_argument("--state-dir", type=Path, help="Directory for CLI session state.")
    models_refresh = models_sub.add_parser(
        "refresh",
        help="Fetch the provider's current model list into a local cache.",
    )
    models_refresh.add_argument("--provider", help="Provider to refresh (default: openrouter).")
    models_refresh.add_argument("--state-dir", type=Path, help="Directory for CLI session state.")

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

    agents = subparsers.add_parser("agents", help="Inspect user and project custom-agent definitions.")
    agents_sub = agents.add_subparsers(dest="agents_command", required=True)
    agents_list = agents_sub.add_parser("list", help="List effective custom agents and discovery diagnostics.")
    agents_list.add_argument("--project", default=".", type=Path, help="Project directory to inspect.")
    agents_list.add_argument("--state-dir", type=Path, help="Directory containing user-level custom agents.")
    agents_validate = agents_sub.add_parser("validate", help="Validate user and project custom-agent definitions.")
    agents_validate.add_argument("--project", default=".", type=Path, help="Project directory to inspect.")
    agents_validate.add_argument("--state-dir", type=Path, help="Directory containing user-level custom agents.")

    browser = subparsers.add_parser("browser", help="Manage the Chrome native-messaging integration.")
    browser_sub = browser.add_subparsers(dest="browser_command", required=True)

    def add_browser_registration_args(command: argparse.ArgumentParser, *, include_host_path: bool = False) -> None:
        command.add_argument(
            "--channel",
            choices=BROWSER_EXTENSION_CHANNELS,
            default=DEFAULT_BROWSER_EXTENSION_CHANNEL,
            help="Extension release channel.",
        )
        command.add_argument("--extension-id", help="Required explicit extension ID for beta and dev channels.")
        command.add_argument("--state-dir", type=Path, help="Directory for CLI and browser runtime state.")
        command.add_argument("--json", action="store_true", help="Print machine-readable status.")
        if include_host_path:
            command.add_argument("--host-path", type=Path, help="Explicit native-host executable path.")

    browser_install = browser_sub.add_parser("install", help="Install or refresh the Chrome native host.")
    add_browser_registration_args(browser_install, include_host_path=True)
    browser_status = browser_sub.add_parser("status", help="Check the Chrome native-host configuration.")
    add_browser_registration_args(browser_status)
    browser_doctor = browser_sub.add_parser("doctor", help="Diagnose the Chrome native-host configuration.")
    add_browser_registration_args(browser_doctor)
    browser_uninstall = browser_sub.add_parser("uninstall", help="Remove the Chrome native host.")
    add_browser_registration_args(browser_uninstall)

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
        thinking_effort=getattr(args, "thinking_effort", None),
        environment=getattr(args, "environment", None),
        edit_protocol=getattr(args, "edit_protocol", None),
        web_search_mode=getattr(args, "web_search", None),
        lsp_mode=getattr(args, "lsp", None),
        subagents_mode=getattr(args, "subagents", None),
        skills_mode=getattr(args, "skills", None),
        compression_threshold=getattr(args, "compression_threshold", None),
        context_window_tokens=getattr(args, "context_window_tokens", None),
        max_output_tokens=getattr(args, "max_output_tokens", None),
        custom_endpoints_json=getattr(args, "custom_endpoints", None),
        endpoint_url=getattr(args, "endpoint_url", None),
        endpoint_style=getattr(args, "endpoint_style", None),
        endpoint_api_key=getattr(args, "endpoint_api_key", None),
        endpoint_context=getattr(args, "endpoint_context", None),
        endpoint_max_output=getattr(args, "endpoint_max_output", None),
        endpoint_vision=bool(getattr(args, "endpoint_vision", False)),
        endpoint_thinking=getattr(args, "endpoint_thinking", None),
        endpoint_reasoning=getattr(args, "endpoint_reasoning", None),
        endpoint_temperature=getattr(args, "endpoint_temperature", None),
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


def _select_startup_project(project_path: Path, args: argparse.Namespace) -> Path:
    """Resolve the effective startup checkout before project-scoped services are built."""
    source_path = _validate_project(project_path)
    try:
        if getattr(args, "worktree", None) is not None:
            return resolve_worktree(source_path, args.worktree).path
        if getattr(args, "create_worktree", None) is not None:
            created = create_worktree(
                source_path,
                args.create_worktree,
                start_ref=getattr(args, "worktree_from", None),
                destination=getattr(args, "worktree_path", None),
            )
            print(
                f"Created worktree for {created.branch!r} at {created.path}; it will be retained.",
                file=sys.stderr,
            )
            return created.path
    except WorktreeError as exc:
        raise ValueError(str(exc)) from exc
    return source_path


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


def _active_project_for_resume(session: SessionRecord, store: SessionStore) -> Path:
    """Validate and return a resumed session's durable active workspace."""
    launch_path = Path(session.project_path).expanduser().resolve()
    active, warning = resolve_active_project(session, store, launch_path)
    if warning:
        print(f"Warning: {warning}", file=sys.stderr)
    return active


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


def _probe_chrome_extension(state_dir: Path | None, extension_id: str) -> dict[str, Any]:
    """Attempt a real attach so `doctor` cannot report green while unreachable."""
    from kolega_code.browser_extension.manager import (
        ChromeExtensionBrowserManager,
        ChromeExtensionUnavailableError,
    )

    resolved_state_dir = state_dir or default_state_dir()

    async def run() -> dict[str, Any]:
        manager = ChromeExtensionBrowserManager(
            state_dir=resolved_state_dir,
            kolega_session_id="browser-doctor",
            extension_origin=f"chrome-extension://{extension_id}/",
            connection_timeout=5.0,
        )
        try:
            return await manager.probe()
        finally:
            with contextlib.suppress(Exception):
                await manager.cleanup_all_browsers()

    try:
        return asyncio.run(run())
    except ChromeExtensionUnavailableError as exc:
        return {"state": "unreachable", "connected": False, "ready": False, "runtimes": [], "detail": str(exc)}


def _run_browser(args: argparse.Namespace) -> int:
    command = args.browser_command
    common = {
        "channel": args.channel,
        "extension_id": args.extension_id,
    }
    try:
        if command == "uninstall":
            removed = uninstall_native_host(**common)
            payload = {
                "command": command,
                "removed": removed,
                "detail": "Chrome native host removed." if removed else "Chrome native host was not installed.",
            }
            if args.json:
                print(json.dumps(payload, sort_keys=True))
            else:
                print(payload["detail"])
            return 0

        status_kwargs = {
            **common,
            "state_dir": args.state_dir,
        }
        if command == "install":
            status = install_native_host(host_path=args.host_path, **status_kwargs)
        else:
            status = native_host_status(**status_kwargs)

        store_url = channel_web_store_url(status.channel)
        payload = {
            "command": command,
            **status.to_dict(),
            "web_store_url": store_url,
        }
        if command == "doctor" and not status.valid:
            payload["remediation"] = "Run `kolega-code browser install` with the same channel and extension ID."

        # `status` is a cheap manifest check; only `doctor` attempts a real
        # attach, because a valid manifest says nothing about whether the
        # extension is installed, enabled, or paired with this session.
        probe: dict[str, Any] | None = None
        if command == "doctor" and status.valid:
            probe = _probe_chrome_extension(args.state_dir, status.extension_id)
            payload["extension"] = probe
            if probe["state"] != "paired":
                payload["remediation"] = probe["detail"]
            if probe["state"] == "awaiting_selection":
                # doctor publishes its own temporary runtime, so a choice is expected
                # whenever another Kolega session is *actively* claiming the browser.
                # Say so rather than implying a fault.
                payload["remediation"] = (
                    f"{probe['detail']} This check publishes its own temporary runtime, so a choice is "
                    "expected whenever another Kolega session is currently using the browser; that session "
                    "releases its claim when it detaches, or stop it to test pairing unattended."
                )

        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(status.detail)
            if command == "status":
                print("Checked the native-host manifest only; run `kolega-code browser doctor` to test the extension.")
            print(f"Manifest: {status.manifest_path}")
            if status.host_path is not None:
                print(f"Native host: {status.host_path}")
            if probe is not None:
                print(f"Extension: {probe['state']}")
                if probe.get("runtime_id"):
                    print(f"This session's runtime: {probe['runtime_id']}")
                for runtime in probe["runtimes"]:
                    marker = " (this session)" if runtime["current"] else ""
                    print(
                        f"  - {runtime['session_id']} — runtime {runtime['runtime_id']}, pid {runtime['pid']}{marker}"
                    )
            if payload.get("remediation"):
                print(payload["remediation"])
            if store_url:
                print(f"Chrome Web Store: {store_url}")

        if command == "install" and store_url:
            webbrowser.open(store_url)
        if probe is not None and probe["state"] != "paired":
            return 1
        return 0 if status.valid else 1
    except (RuntimeError, OSError) as exc:
        if getattr(args, "json", False):
            print(json.dumps({"command": command, "error": str(exc)}, sort_keys=True))
        else:
            _print_styled(f"kolega-code browser: {exc}", style="error", stderr=True)
        return 2


def _print_quit_resume_hint(session_id: str) -> None:
    """Print the post-quit resume command for this session."""
    _print_styled(messages.SESSION_RESUME_HINT.format(session_id=session_id), style="success")


def _run_tui(args: argparse.Namespace) -> int:
    if importlib.util.find_spec("textual") is None:
        print("Textual is not installed. Reinstall the CLI with: uv tool install --force kolega-code", file=sys.stderr)
        return 2

    project_path = _select_startup_project(args.project_path, args)
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
    if getattr(args, "trust_lsp", False):
        settings.trust_lsp_project(project_path)
        settings_changed = True
    if settings_changed:
        settings_store.save(settings)
    summary = {}
    startup_config_error = None
    try:
        config = build_agent_config(
            project_path, _overrides_from_args(args), settings=settings, settings_store=settings_store
        )
        summary = config_summary(config)
    except StrictContextBudgetError:
        # Strict caps are explicit per-run constraints, not recoverable missing
        # setup. Fail before creating a TUI session or emitting llm.run_started.
        raise
    except CliConfigError as exc:
        if str(exc) == DEPRECATED_THINKING_TOKENS_MESSAGE:
            raise
        config = None
        startup_config_error = str(exc)
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

    try:
        extension_selection = _resolve_extension_selection_from_args(args)
    except KolegaExtensionLoadError as exc:
        _print_styled(str(exc), style="error", stderr=True)
        return 1

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
        startup_config_error=startup_config_error,
        extension_selection=extension_selection,
    )

    async def _run_app() -> None:
        try:
            await app.run_async()
        finally:
            # Awaited teardown for exits that bypass action_quit (startup
            # errors, crashes); a no-op after a normal quit.
            await app._cleanup_agent_generation()
            usage_sink = getattr(app, "_usage_sink", None)
            if usage_sink is not None:
                try:
                    await usage_sink.aclose()
                except Exception as exc:  # noqa: BLE001 — reported, never masks the primary exception
                    print(f"Warning: usage sink close failed: {exc}", file=sys.stderr)
        # Normal quit only: action_quit set _quit_cleanly after saving the
        # session, so the terminal now shows how to resume exactly this one.
        if getattr(app, "_quit_cleanly", False):
            _print_quit_resume_hint(session.session_id)

    try:
        asyncio.run(_run_app())
    except Exception as exc:  # noqa: BLE001 — last-resort crash capture before re-raising
        _secrets = known_secret_values(
            settings, settings_store, project_path=project_path, mcp_config=getattr(config, "mcp_config", None)
        )
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


def known_secret_values(
    settings: CliSettings,
    settings_store: SettingsStore,
    *,
    project_path: Optional[Path] = None,
    mcp_config: Any = None,
) -> list[str]:
    """Every credential this installation knows it holds.

    Secret *detection* is pattern matching, and plenty of real keys match no
    pattern at all — a Fireworks ``fw_`` key, a Tavily ``tvly-`` key, and the
    bare-hex keys some providers issue all sail straight through. Anywhere
    session content leaves this machine, these values are handed over
    explicitly so the redactor does not have to guess.
    """
    values = [value for value in (settings.api_keys or {}).values() if value]
    for token in (settings.oauth_tokens or {}).values():
        if not isinstance(token, dict):
            continue
        for key in ("access_token", "refresh_token", "id_token"):
            value = token.get(key)
            if value:
                values.append(str(value))
    try:
        from kolega_code.mcp.config import load_mcp_config, mcp_secret_values
        from kolega_code.mcp.state import MCPOAuthTokenStore

        if mcp_config is None and project_path is not None:
            mcp_config = load_mcp_config(
                project_path,
                settings_store.root,
                project_trusted=settings.is_mcp_project_trusted(project_path),
            )
        if mcp_config is not None:
            values.extend(mcp_secret_values(mcp_config))
        values.extend(MCPOAuthTokenStore(settings_store.root).secret_values())
    except Exception:
        # Best effort: a broken MCP config must not stop a crash log or an
        # export from being written at all.
        pass
    return [value for value in values if value]


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


async def _sleep_until_loop_fire(state: LoopState, recorder: SessionRecorder, json_mode: bool) -> None:
    """Wait until the loop's next fire time, in slices so Ctrl-C stays responsive."""
    remaining = state.seconds_until()
    if remaining <= 0:
        return
    recorder.record_loop_sleeping({"seconds": round(remaining, 3), "next_fire_at": state.next_fire_at})
    if not json_mode:
        print(
            f"[loop] sleeping {format_duration_short(remaining)} until {state.next_fire_at}",
            file=sys.stderr,
        )
    while remaining > 0:
        slice_seconds = min(1.0, remaining)
        await asyncio.sleep(slice_seconds)
        remaining -= slice_seconds


def _emit_loop_iteration(state: LoopState, recorder: SessionRecorder, json_mode: bool) -> None:
    recorder.record_loop_iteration_started(
        {
            "iteration": state.iterations,
            "max_iterations": state.max_iterations,
            "scheduled_at": state.last_fired_at,
            "prompt_source": state.prompt_source,
        }
    )
    if not json_mode:
        print(
            f"[loop] iteration {state.iterations}/{state.max_iterations} ({state.schedule_label()})",
            file=sys.stderr,
        )


def _emit_loop_finished(state: LoopState, recorder: SessionRecorder, json_mode: bool) -> None:
    reason = "expired" if state.is_expired() else "reached the iteration cap"
    recorder.record_loop_completed({"iterations": state.iterations, "reason": reason})
    if not json_mode:
        print(f"[loop] finished after {state.iterations} iteration(s): {reason}", file=sys.stderr)


def _record_ask_run_terminal(
    recorder: SessionRecorder,
    *,
    status: str,
    exit_code: Optional[int],
    started_at: str,
    error: Optional[dict[str, Any]],
    usage_ledger: UsageLedger,
    provider: Optional[str],
    model: Optional[str],
) -> None:
    """Best-effort ``run.*`` terminal record; never raises (runs in a finally).

    ``exit_code`` is recorded only on paths that decide it locally; a run that
    ends by propagating an exception omits it (the process code is decided
    above this frame).
    """
    try:
        from kolega_code import __version__ as kolega_version

        totals = usage_ledger.snapshot()
        payload: dict[str, Any] = {
            "status": status,
            "started_at": started_at,
            "ended_at": now_iso(),
            "kolega_version": kolega_version,
            "provider": provider,
            "model": model,
            "totals": {
                "requests": totals.requests,
                "responses": totals.responses,
                "reported": totals.reported,
                "unreported": totals.unreported,
                "failed": totals.failed,
                "input_tokens": totals.input_tokens,
                "output_tokens": totals.output_tokens,
                "total_tokens": totals.total_tokens,
                "cache_read_input_tokens": totals.cache_read_input_tokens,
                "cache_write_input_tokens": totals.cache_write_input_tokens,
                "reasoning_output_tokens": totals.reasoning_output_tokens,
            },
        }
        if exit_code is not None:
            payload["exit_code"] = exit_code
        if error:
            payload["error"] = error
        recorder.record_run_terminal(status, payload)
    except Exception:
        logging.getLogger(__name__).exception("Failed to record the run terminal event")


async def _run_ask(args: argparse.Namespace) -> int:
    launch_project_path = _select_startup_project(args.project, args)
    project_path = launch_project_path
    store = _store_from_args(args)
    resumed_session: SessionRecord | None = None
    if args.session:
        try:
            resumed_session = store.load(args.session)
        except SessionStoreError:
            resumed_session = None
        if resumed_session is not None:
            _validate_session_project(resumed_session, launch_project_path)
            project_path = _active_project_for_resume(resumed_session, store)
    settings_store = _settings_store_from_args(args)
    settings = settings_store.load()
    overrides = _overrides_from_args(args)
    try:
        extension_selection = _resolve_extension_selection_from_args(args)
    except KolegaExtensionLoadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    skills_enabled = _skills_enabled(load_cli_env(launch_project_path), settings, overrides.skills_mode)
    skill_catalog = discover_skills(project_path) if skills_enabled else SkillCatalog()
    goal_condition = getattr(args, "goal", None)
    loop_interval = getattr(args, "loop", None)
    loop_cron = getattr(args, "loop_cron", None)
    looping = bool(loop_interval or loop_cron)
    raw_prompt = args.prompt

    if looping and goal_condition:
        print("Error: --loop cannot be combined with --goal.", file=sys.stderr)
        return 2

    # Validate the schedule up front so a typo fails before an agent is built.
    loop_schedule = None
    loop_expires_seconds = DEFAULT_LOOP_EXPIRY_DAYS * 86400
    if looping:
        try:
            loop_schedule = parse_schedule_text(loop_cron if loop_cron else str(loop_interval))
            if getattr(args, "loop_expires", None):
                loop_expires_seconds = parse_duration(args.loop_expires)
        except LoopError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    # A loop may take its prompt from .kolega/loop.md instead of the command line.
    loop_md_prompt: Optional[str] = None
    if looping and raw_prompt is None:
        try:
            loop_md = read_loop_md(project_path)
        except LoopError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        if loop_md is None:
            print(f"Error: {messages.LOOP_MD_MISSING}", file=sys.stderr)
            return 2
        loop_md_prompt = loop_md.prompt

    # --goal can run without a prompt (it synthesises one from the condition).
    if raw_prompt is None and not goal_condition and loop_md_prompt is None:
        print("Error: prompt is required (or use --goal <condition> or --loop).", file=sys.stderr)
        return 2
    skill_command = _parse_skill_prompt(raw_prompt, skill_catalog) if raw_prompt else None

    # Pre-session informational output: no journal exists yet, so these print
    # plain text regardless of --json rather than inventing envelope-less JSON.
    if skill_command and skill_command[0] == "skills":
        catalog_text = skill_catalog.format_catalog() if skills_enabled else messages.SKILLS_DISABLED
        print(catalog_text)
        return 0

    if skill_command and skill_command[0] != "skills" and not skill_command[1] and not (args.save or args.session):
        activation_content = skill_catalog.activation_content(skill_command[0])
        print(activation_content)
        return 0

    custom_agent_catalog = discover_custom_agents(project_path, settings_store.root)
    settings_changed = False
    if getattr(args, "trust_hooks", False):
        settings.trust_hook_project(launch_project_path)
        settings_changed = True
    if getattr(args, "trust_mcp", False):
        settings.trust_mcp_project(launch_project_path)
        settings_changed = True
    if getattr(args, "trust_lsp", False):
        settings.trust_lsp_project(launch_project_path)
        settings_changed = True
    if settings_changed:
        settings_store.save(settings)
    config = build_agent_config(launch_project_path, overrides, settings=settings, settings_store=settings_store)
    custom_agent_catalog = validate_custom_agent_models(custom_agent_catalog, config).for_mode("build")
    if not args.json:
        for diagnostic in custom_agent_catalog.diagnostics:
            print(f"agents: {diagnostic.format()}", file=sys.stderr)
    summary = config_summary(config)

    hook_config = load_hook_config(
        launch_project_path,
        settings_store.root,
        project_trusted=settings.is_hook_project_trusted(launch_project_path),
    )
    hook_dispatcher = HookDispatcher(hook_config)
    if not args.json:
        for diagnostic in hook_config.diagnostics:
            print(f"hooks: {diagnostic}", file=sys.stderr)

    if args.session:
        session = resumed_session or _get_or_create_session(
            store, project_path, CLI_AGENT_MODE, summary, args.session, force_new=False
        )
        session = _normalize_cli_session_mode(store, session, persist=True)
    elif args.save:
        session = store.create(project_path, CLI_AGENT_MODE, summary)
    else:
        session = SessionRecord.create(project_path, CLI_AGENT_MODE, summary)

    # Every run records the semantic journal; unsaved runs use an in-memory
    # sink with the same id/seq/schema rules and never touch the state dir.
    if args.save or args.session:
        session_recorder = store.recorder(session.session_id)
        session = store.load(session.session_id)
        journal = store.journal(session.session_id)
    else:
        journal = InMemorySessionJournal(session.session_id)
        session_recorder = None

    json_printer: Optional[SemanticStdoutPrinter] = None
    if args.json:
        json_printer = SemanticStdoutPrinter(
            secret_values=known_secret_values(settings, settings_store, project_path=launch_project_path)
        )
        journal.add_listener(json_printer)

    if session_recorder is None:
        from kolega_code import __version__ as _kolega_version

        bootstrap_epoch = str(uuid.uuid4())
        journal.append(
            "session.created",
            actor="system",
            payload={"metadata": session.to_metadata_dict(), "kolega_version": _kolega_version},
            epoch_id=bootstrap_epoch,
        )
        journal.append(
            "context.epoch_started",
            actor="system",
            payload={"reason": "session_created"},
            epoch_id=bootstrap_epoch,
        )
        session_recorder = SessionRecorder(journal, recover=False)
    elif json_printer is not None and args.save and not args.session:
        # A session created by this invocation journaled its bootstrap events
        # before the printer attached; a resumed session streams from the tail.
        json_printer.emit_backlog(journal.read_events())

    run_started_at = now_iso()

    manager = CliConnectionManager()
    browser_manager = build_browser_manager(
        store.root,
        session.session_id,
        browser_visible=args.browser_visible,
    )
    agent_ref: dict[str, CoderAgent] = {}
    prompt_extensions = []
    tool_extensions = []
    if skills_enabled:
        skill_prompt_extension = build_skill_prompt_extension(
            skill_catalog,
            context_window_tokens=context_window_tokens_for_skill_budget(config, CoderAgent.agent_name),
        )
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
        project_trusted=settings.is_mcp_project_trusted(launch_project_path),
        loaded_config=mcp_config if project_path == launch_project_path else None,
    )
    if mcp_extension is not None:
        tool_extensions.append(mcp_extension)
        if mcp_config is not None:
            _print_mcp_tool_name_warnings(mcp_config, settings_store.root, project_path)
    extension_bundle = None
    if extension_selection is not None:
        extension_host = KolegaExtensionHost(
            project_path=project_path,
            workspace_id=session.workspace_id,
            thread_id=session.thread_id,
            config=config,
            agent_mode=AgentMode.ASK,
        )
        try:
            extension_bundle = create_extension_bundle(extension_selection, extension_host)
        except KolegaExtensionLoadError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        prompt_extensions.extend(extension_bundle.prompt_extensions)
        tool_extensions.extend(extension_bundle.tool_extensions)

    # An extension bundle may exist from here on: every exit path below must
    # flow through the finally, which tears down agent/bundle/sink exactly once
    # and records the run terminal.
    usage_sink: Optional[SessionUsageSink] = None
    pump_task: Optional[asyncio.Task] = None
    usage_ledger = UsageLedger()
    response_chunks: list[dict] = []
    exit_code = 0
    run_status = "completed"
    run_error: Optional[dict[str, Any]] = None
    # Exit code stamped on the run terminal; stays None on paths that leave by
    # a propagating exception (the caller owns the process exit code there).
    terminal_exit_code: Optional[int] = None
    generation_closed = False

    async def _close_generation() -> None:
        # At most once; agent before bundle; each step guarded so a cleanup
        # failure never replaces the run's primary outcome.
        nonlocal generation_closed
        if generation_closed:
            return
        generation_closed = True
        built_agent = agent_ref.get("agent")
        if built_agent is not None:
            try:
                await built_agent.cleanup()
            except Exception as exc:  # noqa: BLE001 — reported, never masks the primary exception
                print(f"Warning: agent cleanup failed: {exc}", file=sys.stderr)
        if extension_bundle is not None:
            try:
                await cleanup_extension_bundle(extension_bundle)
            except Exception as exc:  # noqa: BLE001 — reported, never masks the primary exception
                print(f"Warning: extension cleanup failed: {exc}", file=sys.stderr)

    try:
        permission_mode = normalize_permission_mode(
            getattr(args, "permission_mode", ASK_DEFAULT_PERMISSION_MODE),
            default=PermissionMode.AUTO,
        )
        # Journal every settled non-history response and failure (in-memory journal
        # for unsaved runs). Attached before the SESSION_START hook below: hook
        # prompts are paid LLM calls and must be covered from the first request.
        sink = SessionUsageSink(
            journal,
            session_recorder,
            usage_ledger,
            mode="ask",
            run_metadata=(
                {"context_budget": budget_marker} if (budget_marker := strict_context_budget_marker(config)) else None
            ),
        )
        usage_sink = sink
        usage_ledger.observer = sink
        await sink.start()
        try:
            agent = CoderAgent(
                project_path=project_path,
                workspace_id=session.workspace_id,
                thread_id=session.thread_id,
                connection_manager=manager,
                config=config,
                browser_manager=browser_manager,
                agent_mode=AgentMode.ASK,
                prompt_extensions=prompt_extensions,
                tool_extensions=tool_extensions,
                permission_mode=permission_mode,
                permission_callback=_permission_callback_for_ask(launch_project_path)
                if permission_mode == PermissionMode.ASK
                else None,
                session_recorder=session_recorder,
                hook_dispatcher=hook_dispatcher,
                custom_agent_catalog=custom_agent_catalog,
                memory_project_path=launch_project_path,
                memory_enabled=not getattr(args, "no_memory_tools", False),
                usage_ledger=usage_ledger,
                llm_trace_sink=extension_bundle.llm_trace_sink if extension_bundle else None,
            )
        except ValueError as exc:
            if extension_bundle is None:
                raise
            # With an extension loaded, a tool-name conflict at construction
            # refuses to start rather than running without the extension.
            print(f"Error: {exc}", file=sys.stderr)
            run_status = "failed"
            run_error = {"code": "extension_error", "message": str(exc)[:500]}
            exit_code = 2
            terminal_exit_code = 2
            return exit_code
        agent_ref["agent"] = agent
        # --gigacode turns orchestration on for this run; a resumed session that had
        # it on keeps it on, exactly as the TUI's /gigacode toggle persists.
        gigacode_enabled = bool(getattr(args, "gigacode", False)) or bool(session.gigacode_enabled)
        if gigacode_enabled:
            agent.apply_gigacode(True, gigacode_prompt_extension())
        session.gigacode_enabled = gigacode_enabled
        # Web tool mode: an explicit --web-search flag already reached the agent via
        # AgentConfig; otherwise a resumed session keeps its persisted mode, exactly
        # as the TUI's /web-search toggle persists.
        flag_web_search = getattr(args, "web_search", None)
        if not flag_web_search and session.web_search_mode:
            agent.apply_web_search_mode(session.web_search_mode)
        session.web_search_mode = flag_web_search or session.web_search_mode
        lsp_messages = await agent.tool_collection.initialize()
        if not args.json:
            for msg in lsp_messages:
                print(msg, file=sys.stderr)
        if session.history:
            agent.restore_message_history(session.history)
            agent.restore_compaction_state(session.compaction)

        if extension_bundle is not None:
            await bind_extension_agent(extension_bundle, agent)

        fire_hook = getattr(agent, "fire_hook", None)
        if fire_hook is not None:
            session_start = await fire_hook(HookEvent.SESSION_START, {"source": "startup"})
            if session_start.additional_context:
                if session_recorder is not None:
                    session_recorder.record_context_message(
                        Message(role="user", content=[TextBlock(text=session_start.additional_context)])
                    )
                agent.append_user_message([TextBlock(text=session_start.additional_context)])

        prompt = raw_prompt
        if skill_command:
            skill_name, skill_prompt = skill_command
            active_names = activated_skill_names(agent.history)
            activation_content = skill_catalog.activation_content(skill_name, active_names=active_names)
            if skill_name not in active_names:
                session_recorder.record_context_message(
                    Message(role="user", content=[TextBlock(text=activation_content)])
                )
                agent.append_user_message([TextBlock(text=activation_content)])
            session_recorder.record_skill_activated(
                name=skill_name,
                source="prompt_command",
                already_active=skill_name in active_names,
            )
            prompt = skill_prompt
            if not prompt:
                session_recorder.record_synthetic_assistant(activation_content, notice_code="skill_activation")
                if not args.json:
                    print(activation_content)
                if args.save or args.session:
                    session.config = summary
                    store.save(session)
                terminal_exit_code = 0
                return exit_code

        # --goal: apply the goal-aware prompt extension and synthesise the first
        # work-turn message when no explicit prompt was given.
        goal_state: Optional[GoalState] = None
        if goal_condition:
            max_turns = getattr(args, "goal_max_turns", None) or DEFAULT_GOAL_MAX_TURNS
            goal_state = GoalState.create(goal_condition, max_turns=max_turns, run_to_completion=True)
            agent.apply_goal(
                goal_condition,
                PromptExtension(
                    id="cli-active-goal",
                    title="Active goal",
                    markdown=build_goal_prompt_extension_markdown(goal_condition),
                    modes=None,
                    propagate_to_sub_agents=True,
                ),
            )
            if not prompt:
                prompt = build_goal_task_prompt(goal_condition)

        # --loop / --loop-cron: arm the schedule and apply the loop-aware prompt
        # extension. The prompt itself is sent verbatim on every iteration.
        loop_state: Optional[LoopState] = None
        if loop_schedule is not None:
            loop_prompt = loop_md_prompt if loop_md_prompt is not None else (prompt or "")
            loop_state = LoopState.create(
                loop_schedule,
                loop_prompt,
                prompt_source=PROMPT_SOURCE_LOOP_MD if loop_md_prompt is not None else PROMPT_SOURCE_INLINE,
                fresh=bool(getattr(args, "loop_fresh", False)),
                max_iterations=getattr(args, "loop_max_iterations", None) or DEFAULT_LOOP_MAX_ITERATIONS,
                expires_seconds=loop_expires_seconds,
            )
            prompt = loop_prompt
            agent.apply_loop(
                True,
                PromptExtension(
                    id="cli-active-loop",
                    title="Scheduled loop",
                    markdown=build_loop_prompt_extension_markdown(loop_state),
                    modes=None,
                    propagate_to_sub_agents=True,
                ),
            )

        attachments, unresolved_mentions = build_file_attachments(prompt or "", project_path)
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

        # Pump connection-manager events concurrently so sub-agent activity is
        # reported in real time instead of all at once after streaming finishes.
        pump_task = asyncio.create_task(_pump_ask_events(manager, args.json))
        turn_prompt = prompt
        if loop_state is not None:
            # Interval schedules fire immediately; cron schedules wait for the
            # first matching wall-clock time.
            await _sleep_until_loop_fire(loop_state, session_recorder, args.json)
            loop_state.mark_fired()
            _emit_loop_iteration(loop_state, session_recorder, args.json)
            turn_prompt = build_loop_iteration_prompt(
                prompt or "",
                iteration=loop_state.iterations,
                max_iterations=loop_state.max_iterations,
                fresh=loop_state.fresh,
            )

        # Checkpoint for goal/loop token accounting: each drain adds everything
        # the ledger saw since the previous drain — the whole command tree
        # (verifier sub-agents, compression, web-fetch, hooks) —
        # to the state's counter. --goal and --loop are mutually exclusive.
        usage_mark = usage_ledger.snapshot()

        def _drain_tokens(state) -> None:
            nonlocal usage_mark
            current = usage_ledger.snapshot()
            delta = current.since(usage_mark)
            if delta.total_tokens > 0:
                state.tokens_spent += delta.total_tokens
            usage_mark = current

        # In --json mode the journal tee is the only stdout writer: assistant
        # messages, tool results, and terminal records print as semantic events
        # the moment they are journaled. Consuming the stream drives the turn
        # (and prints the answer text in plain mode).
        async def _consume_turn(stream) -> None:
            async for chunk in stream:
                response_chunks.append(chunk)
                if not args.json and chunk.get("type") == "response" and chunk.get("content"):
                    print(chunk["content"], end="" if not chunk.get("complete") else "\n")

        stream = (
            agent.process_message_stream(turn_prompt, attachments)
            if attachments
            else agent.process_message_stream(turn_prompt)
        )
        await _consume_turn(stream)

        # --loop: keep re-running the prompt until the cap or the expiry is hit.
        if loop_state is not None:
            await _run_ask_loop_iterations(
                agent=agent,
                loop_state=loop_state,
                prompt=prompt,
                attachments=attachments,
                session_recorder=session_recorder,
                json_mode=args.json,
                consume_turn=_consume_turn,
                drain_tokens=_drain_tokens,
            )

        # --goal: evaluate and auto-continue until the goal is met or the cap is hit.
        if goal_state is not None:
            exit_code = await _run_ask_goal_iterations(
                agent=agent,
                goal_state=goal_state,
                session_recorder=session_recorder,
                json_mode=args.json,
                consume_turn=_consume_turn,
                drain_tokens=_drain_tokens,
            )

        if args.save or args.session:
            session.config = summary
            store.save(session)
        terminal_exit_code = exit_code
    except KolegaExtensionLoadError as exc:
        # Only bind_extension_agent raises this here: refuse the run rather
        # than continue with the extension unbound.
        print(f"Error: {exc}", file=sys.stderr)
        run_status = "failed"
        run_error = {"code": "extension_error", "message": str(exc)[:500]}
        exit_code = 2
        terminal_exit_code = 2
        return exit_code
    except LLMBillingError as exc:
        exit_code = 1
        run_status = "failed"
        run_error = {"code": "billing_error", "message": CLI_BILLING_ERROR_MESSAGE}
        terminal_exit_code = 1
        message = billing_error_message(exc, model=config.long_context_config.model)
        _print_styled(message, style="error", stderr=True)
    except asyncio.CancelledError:
        # Ctrl-C: asyncio.run cancels this coroutine; main() still exits 130.
        run_status = "cancelled"
        run_error = {"code": "cancelled"}
        raise
    except BaseException as exc:
        run_status = "failed"
        run_error = {"code": type(exc).__name__, "message": str(exc)[:500]}
        raise
    finally:
        if pump_task is not None:
            pump_task.cancel()
            try:
                await pump_task
            except asyncio.CancelledError:
                pass
        while not manager.events.empty():
            event = manager.events.get_nowait()
            _print_ask_event(event, args.json)
        end_fire_hook = getattr(agent_ref.get("agent"), "fire_hook", None)
        if end_fire_hook is not None:
            try:
                await end_fire_hook(HookEvent.SESSION_END, {"reason": "ask_complete"})
            except Exception:
                pass
        await _close_generation()
        if usage_sink is not None:
            try:
                await usage_sink.aclose()
            except Exception as exc:  # noqa: BLE001 — reported, never masks the primary exception
                print(f"Warning: usage sink close failed: {exc}", file=sys.stderr)
        # The run terminal is recorded synchronously after the usage sink has
        # drained, on every exit path — an await here could be re-cancelled
        # while a CancelledError is already propagating.
        _record_ask_run_terminal(
            session_recorder,
            status=run_status,
            exit_code=terminal_exit_code,
            started_at=run_started_at,
            error=run_error,
            usage_ledger=usage_ledger,
            provider=str(getattr(config.long_context_config, "provider", None) or "") or None,
            model=config.long_context_config.model,
        )
        if getattr(args, "atif_output", None) is not None:
            # After the run terminal so the trajectory carries the outcome.
            # Synchronous and broadly guarded: this runs inside a finally and
            # must never mask billing errors or a propagating cancellation.
            try:
                from kolega_code import __version__ as _kolega_version_atif

                from .atif_export import export_atif_to_path

                path = export_atif_to_path(
                    journal,
                    output=args.atif_output,
                    session_metadata=session.to_metadata_dict(),
                    kolega_version=_kolega_version_atif,
                    secret_values=known_secret_values(settings, settings_store, project_path=launch_project_path),
                    state_dirs=(store.root,),
                )
                _print_styled(f"Wrote ATIF trajectory to {path}", style="success", stderr=True)
            except Exception as exc:  # noqa: BLE001 - reported; run outcome wins
                _print_styled(f"ATIF export failed: {exc}", style="error", stderr=True)
                if exit_code == 0:
                    # A conversion failure on an otherwise-clean run exits 1;
                    # a run that already failed keeps its own exit status.
                    exit_code = 1

        # The command's ownership of the session ends here: release the
        # cross-process lock so a sequential resume in this process (tests,
        # embedded hosts) is not refused. A concurrent process cannot run
        # while this command is alive, so there is no window to exploit.
        store.release_session_locks()

    return exit_code


async def _run_ask_loop_iterations(
    *,
    agent: "CoderAgent",
    loop_state: LoopState,
    prompt: Optional[str],
    attachments: list,
    session_recorder,
    json_mode: bool,
    consume_turn,
    drain_tokens,
) -> None:
    """Re-run the loop prompt until the iteration cap or the expiry is hit."""
    drain_tokens(loop_state)
    loop_state.advance_after_completion()
    while loop_state.is_active:
        await _sleep_until_loop_fire(loop_state, session_recorder, json_mode)
        if not loop_state.is_active:
            break
        loop_state.mark_fired()
        if loop_state.fresh:
            agent.clear_history()
        _emit_loop_iteration(loop_state, session_recorder, json_mode)
        turn_prompt = build_loop_iteration_prompt(
            prompt or "",
            iteration=loop_state.iterations,
            max_iterations=loop_state.max_iterations,
            fresh=loop_state.fresh,
        )
        stream = (
            agent.process_message_stream(turn_prompt, attachments)
            if attachments
            else agent.process_message_stream(turn_prompt)
        )
        await consume_turn(stream)
        drain_tokens(loop_state)
        loop_state.advance_after_completion()
    _emit_loop_finished(loop_state, session_recorder, json_mode)


async def _run_ask_goal_iterations(
    *,
    agent: "CoderAgent",
    goal_state: GoalState,
    session_recorder,
    json_mode: bool,
    consume_turn,
    drain_tokens,
) -> int:
    """Evaluate and auto-continue until the goal is met or the turn cap is hit.

    Returns the process exit code. An unmet goal exits 1 but the run itself
    completed: the process finished; the goal did not.
    """
    drain_tokens(goal_state)
    while not goal_state.met and goal_state.turns_evaluated < goal_state.max_turns:
        verdict = await agent.evaluate_goal_condition(goal_state.condition)
        goal_state.turns_evaluated += 1
        goal_state.last_reason = verdict.reason
        goal_state.last_evaluated_at = now_iso()
        session_recorder.record_goal_evaluated(
            {"met": verdict.met, "turns": goal_state.turns_evaluated, "reason": verdict.reason}
        )
        if not json_mode:
            tag = "MET" if verdict.met else f"not met — {verdict.reason}"
            print(f"[goal] turn {goal_state.turns_evaluated}: {tag}", file=sys.stderr)
        if verdict.met:
            goal_state.met = True
            break
        turns_remaining = goal_state.max_turns - goal_state.turns_evaluated
        nudge = build_goal_nudge(goal_state.condition, verdict, turns_remaining)
        await consume_turn(agent.process_message_stream(nudge))
        drain_tokens(goal_state)
    # The loop above can exit on a met verdict or the turn cap; count the
    # final verifier either way.
    drain_tokens(goal_state)
    session_recorder.record_goal_completed(
        {"met": goal_state.met, "turns": goal_state.turns_evaluated, "reason": goal_state.last_reason}
    )
    if not json_mode:
        status = "met" if goal_state.met else "not met (turn cap reached)"
        print(f"[goal] {status} after {goal_state.turns_evaluated} turn(s)", file=sys.stderr)
    return 0 if goal_state.met else 1


async def _pump_ask_events(manager: CliConnectionManager, json_mode: bool) -> None:
    while True:
        event = await manager.next_event()
        _print_ask_event(event, json_mode)


def _print_ask_event(event, json_mode: bool) -> None:
    if json_mode:
        # Presentation events are not part of the semantic protocol; the
        # journal tee owns stdout. Consuming them here keeps the queue drained.
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


async def _run_share(args: argparse.Namespace) -> int:
    """Export a session as a static replay bundle."""
    from kolega_code.cli.session_event_store import FileArtifactStore, FileSessionEventStore
    from kolega_code.web.bundle import export_bundle

    if args.share_command != "export":
        raise ValueError(f"Unknown share command: {args.share_command}")

    from .theme import textual_theme_name

    store = _store_from_args(args)
    record = store.load_session_or_thread(args.session_id)
    journal = store.journal(record.session_id)
    events = await FileSessionEventStore(journal).read(record.session_id)
    if not events:
        _print_styled(
            f"Session {record.session_id} has no recorded presentation events, so there is nothing to replay. "
            "Sessions recorded before this release only contain conversation history.",
            style="warning",
            stderr=True,
        )
        return 1

    # One HTML file unless a directory was asked for, because the common case is
    # sending a replay to someone and a folder cannot be opened by double-click.
    as_directory = args.dir or args.zip
    default_name = f"{record.session_id}-replay" + ("" if as_directory else ".html")
    destination = args.out or Path.cwd() / default_name
    settings_store = SettingsStore(root=getattr(args, "state_dir", None))
    settings = settings_store.load()
    result = await export_bundle(
        events,
        destination.expanduser(),
        session_id=record.session_id,
        title=args.title or getattr(record, "title", "") or "",
        theme_slug=args.theme or textual_theme_name(settings.active_theme),
        artifact_store=FileArtifactStore(journal),
        # An export leaves this machine, so hand the redactor every credential
        # we know about rather than trusting it to recognise their shapes.
        extra_secrets=known_secret_values(
            settings,
            settings_store,
            project_path=Path(record.project_path) if getattr(record, "project_path", None) else None,
        ),
        as_zip=args.zip,
        single_file=not as_directory,
    )

    _print_styled(f"Wrote replay to {result.path}", style="success")
    for line in result.report.summary_lines():
        print(f"  {line}")
    for warning in result.warnings:
        _print_styled(f"  {warning}", style="warning")
    if result.single_file:
        size_mb = result.path.stat().st_size / (1024 * 1024)
        print(f"\nOne file, {size_mb:.1f} MB. Open it in a browser, or send it to someone and they can too.")
    elif not args.zip:
        print(f"\nServe {result.path} over HTTP. Opening index.html directly will not work — browsers block")
        print("module scripts on file:// URLs. Drop --dir for a single file that does open by double-click.")
    return 0


def _run_sessions(args: argparse.Namespace) -> int:
    store = _store_from_args(args)
    if args.sessions_command == "list":
        project = args.project.expanduser().resolve() if args.project else None
        records = store.list(project_path=project)
        if not records:
            print("No saved sessions.")
            return 0
        entries = []
        for record in reversed(records):
            entries.append(
                "\n".join(
                    (
                        f"Updated:   {record.updated_at}",
                        f"Title:     {record.title}",
                        f"Mode:      {record.mode}",
                        f"Project:   {record.project_path}",
                        f"Resume ID: {record.session_id}",
                    )
                )
            )
        print("\n\n".join(entries))
        return 0
    if args.sessions_command == "delete":
        store.delete(args.session_id)
        print(f"Deleted session {args.session_id}")
        return 0
    if args.sessions_command == "repair":
        count = store.repair_sequence(args.session_id)
        print(f"Repaired session {args.session_id}: {count} events with a contiguous sequence.")
        return 0
    if args.sessions_command == "export":
        export_format = getattr(args, "format", "json")
        if export_format == "atif":
            return _run_sessions_export_atif(args, store)
        if export_format == "events-jsonl":
            settings_store = _settings_store_from_args(args)
            settings = settings_store.load()
            payload = store.export_events(
                args.session_id,
                secret_values=known_secret_values(settings, settings_store),
            )
        else:
            payload = store.export(args.session_id)
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
        return 0
    raise ValueError(f"Unknown sessions command: {args.sessions_command}")


def _run_sessions_export_atif(args: argparse.Namespace, store: SessionStore) -> int:
    # Local import: the converter (and the pinned atif package) load only when
    # an ATIF export is actually requested.
    from kolega_code import __version__ as kolega_version

    from .atif_export import AtifExportError, AtifImagesNeedOutputError, export_atif_to_path, export_atif_to_text

    settings_store = _settings_store_from_args(args)
    settings = settings_store.load()
    record = store.load(args.session_id)
    source = store.journal(record.session_id)
    common: dict[str, Any] = {
        "session_metadata": record.to_metadata_dict(),
        "kolega_version": kolega_version,
        "secret_values": known_secret_values(settings, settings_store, project_path=Path(record.project_path)),
        "state_dirs": (store.root,),
    }
    try:
        if args.output:
            path = export_atif_to_path(source, output=args.output, **common)
            print(f"Wrote ATIF trajectory to {path}", file=sys.stderr)
        else:
            sys.stdout.write(export_atif_to_text(source, **common))
        return 0
    except AtifImagesNeedOutputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except AtifExportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


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


def _run_agents(args: argparse.Namespace) -> int:
    project_path = _validate_project(args.project)
    state_dir = _settings_store_from_args(args).root
    catalog = discover_custom_agents(project_path, state_dir)
    print(catalog.format_catalog())
    if args.agents_command == "list":
        return 0
    if args.agents_command == "validate":
        return 1 if catalog.has_errors() else 0
    raise ValueError(f"Unknown agents command: {args.agents_command}")


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
        if getattr(args, "server_id", None):
            args.server_id = sanitize_mcp_server_id(args.server_id)
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
        if args.server_id.strip() != server.id:
            print(
                f"mcp: server id '{args.server_id}' exceeds {_SERVER_ID_MAX_LENGTH} characters; "
                f"saved as '{server.id}'.",
                file=sys.stderr,
            )
        if source == "project" and not settings.is_mcp_project_trusted(project_path):
            print("Project MCP config is not trusted yet. Re-run with --trust-mcp to enable it.", file=sys.stderr)
        return 0

    if args.mcp_command == "remove":
        path, source = _mcp_mutation_target(args, project_path, settings_store.root)
        server_id = sanitize_mcp_server_id(args.server_id)
        try:
            removed = remove_server_config(path, server_id, source=source)
        except MCPConfigError as exc:
            raise ValueError(str(exc)) from exc
        if not removed:
            raise ValueError(f"MCP server not found in {path}: {server_id}")
        service.status_store.clear(server_id)
        service.oauth_store.clear(server_id)
        print(f"Removed MCP server {server_id} from {path}")
        return 0

    if args.mcp_command in {"enable", "disable"}:
        path, source = _mcp_mutation_target(args, project_path, settings_store.root)
        enabled = args.mcp_command == "enable"
        server_id = sanitize_mcp_server_id(args.server_id)
        try:
            changed = set_server_enabled(path, server_id, enabled, source=source)
        except MCPConfigError as exc:
            raise ValueError(str(exc)) from exc
        if not changed:
            raise ValueError(f"MCP server not found in {path}: {server_id}")
        print(f"{'Enabled' if enabled else 'Disabled'} MCP server {server_id} in {path}")
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
    for server in config.servers.values():
        status_text, tool_count, message, _ = _mcp_cli_status_parts(server, service)
        print(
            f"{server.id}\t{server.source}\t{server.transport}\t{server.enabled}\t{server.oauth.enabled}\t"
            f"{status_text}\t{tool_count}\t{message}"
        )


def _print_mcp_tool_name_warnings(config: LoadedMCPConfig, state_dir: Path, project_path: Path) -> None:
    """Warn on stderr when verified MCP tools get provider-adjusted wire names.

    Mirrors the ``mcp: {diagnostic}`` startup output: the model-facing names are
    sanitized, and silently renaming them would be confusing.
    """
    service = MCPService(config, state_dir=state_dir, project_path=project_path)
    for server in config.enabled_servers:
        status = service.server_status(server)
        if not status or status.status != "verified":
            continue
        if status.fingerprint != server_fingerprint(server):
            continue
        note = mcp_tool_name_adjustment_note(server.id, status.tools)
        if note:
            print(f"mcp: warning: server '{server.id}': {note}", file=sys.stderr)


def _mcp_cli_status_parts(server: MCPServerConfig, service: MCPService) -> tuple[str, int, str, str]:
    status = service.server_status(server)
    current_fingerprint = server_fingerprint(server)
    verified = bool(status and status.status == "verified" and status.fingerprint == current_fingerprint)
    stale = bool(status and status.fingerprint and status.fingerprint != current_fingerprint)
    note = ""
    if verified and status:
        status_text = "verified"
        tool_count = status.tool_count
        note = mcp_tool_name_adjustment_note(server.id, status.tools)
    elif stale:
        status_text = "stale"
        tool_count = 0
    elif status and status.status == "failed":
        status_text = "failed"
        tool_count = 0
    else:
        status_text = "unverified"
        tool_count = 0
    return status_text, tool_count, _mcp_cli_list_message(status_text, tool_count, note), note


def _mcp_cli_list_message(status: str, tool_count: int, note: str = "") -> str:
    if status == "verified":
        message = f"Verified {tool_count} tool(s)."
        return f"{message} {note}" if note else message
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


def _run_models(args: argparse.Namespace) -> int:
    def _out(text: str) -> None:
        _print_styled(text)

    def _err(text: str) -> None:
        _print_styled(f"kolega-code: {text}", style="error", stderr=True)

    if args.models_command == "list":
        return run_models_list(args, _out)
    if args.models_command == "refresh":
        return run_models_refresh(args, _out, _err)
    raise ValueError(f"Unsupported models command: {args.models_command}")


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
    line("Thinking effort", summary["thinking_effort"])

    # MCP servers: offline status from config + local verification state (no network).
    mcp_config = load_mcp_config(
        project_path, settings_store.root, project_trusted=settings.is_mcp_project_trusted(project_path)
    )
    mcp_service = MCPService(mcp_config, state_dir=settings_store.root, project_path=project_path)
    for diagnostic in mcp_config.diagnostics:
        print(f"mcp: {diagnostic}", file=sys.stderr)
    if not mcp_config.servers:
        line("MCP servers", "none configured", "muted")
    for server in mcp_config.servers.values():
        status_text, tool_count, _, note = _mcp_cli_status_parts(server, mcp_service)
        status_style = "success" if status_text == "verified" else ("error" if status_text == "failed" else "muted")
        line(
            f"MCP {server.id}",
            f"{server.transport} · {'enabled' if server.enabled else 'disabled'} · {status_text} · "
            f"{tool_count} tool(s)",
            status_style,
        )
        if note:
            line(f"MCP {server.id} tool names", f"{note} See `kolega-code mcp list` for detail.", "warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
