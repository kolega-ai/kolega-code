"""Entrypoint for the Kolega Code CLI."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

from kolega_code.agent import CoderAgent
from kolega_code.agent.llm.models import TextBlock
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.agent.services.browser import PlaywrightBrowserManager

from .config import CliConfigError, CliConfigOverrides, build_agent_config, config_summary, key_status
from .connection import CliConnectionManager
from .session_store import SessionRecord, SessionStore, SessionStoreError
from .settings import SettingsStore, SettingsStoreError
from .skills import (
    SkillCatalog,
    activated_skill_names,
    build_skill_prompt_extension,
    build_skill_tool_extension,
    discover_skills,
)

SUBCOMMANDS = {"ask", "sessions", "doctor"}
RESUME_LATEST = "__latest__"
CLI_AGENT_MODE = AgentMode.CLI.value
AGENT_BUILTIN_COMMANDS = {"/help", "/compress", "/clear", "/reset", "/context"}
SKILLS_LIST_COMMAND = "/skills"


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(list(argv) if argv is not None else sys.argv[1:])
    try:
        if args.command == "ask":
            return asyncio.run(_run_ask(args))
        if args.command == "sessions":
            return _run_sessions(args)
        if args.command == "doctor":
            return _run_doctor(args)
        return _run_tui(args)
    except (CliConfigError, SessionStoreError, SettingsStoreError, ValueError) as exc:
        print(f"kolega-code: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


def parse_args(argv: list[str]) -> argparse.Namespace:
    if argv and argv[0] in SUBCOMMANDS:
        return _build_subcommand_parser().parse_args(argv)
    return _build_tui_parser().parse_args(argv)


def _add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", help="Provider for the main coding model.")
    parser.add_argument("--model", help="Main coding model.")
    parser.add_argument("--fast-provider", help="Provider for fast utility calls.")
    parser.add_argument("--fast-model", help="Fast utility model.")
    parser.add_argument("--edit-provider", help="Provider for edit-file operations.")
    parser.add_argument("--edit-model", help="Model for edit-file operations.")
    parser.add_argument("--thinking-provider", help="Provider for think-hard operations.")
    parser.add_argument("--thinking-model", help="Model for think-hard operations.")
    parser.add_argument("--thinking-tokens", type=int, help="Thinking token budget for think-hard operations.")
    parser.add_argument("--environment", help="Environment label for tracing/metadata.")


def _add_session_args(parser: argparse.ArgumentParser, session_help: str = "Session ID to resume or create.") -> None:
    parser.add_argument("--state-dir", type=Path, help="Directory for CLI session state.")
    parser.add_argument("--session", help=session_help)


def _build_tui_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kolega-code", description="Run the Kolega Code Textual CLI.")
    parser.set_defaults(command="tui")
    parser.add_argument("project_path", nargs="?", default=".", type=Path, help="Project directory to work in.")
    parser.add_argument("--mode", choices=[mode.value for mode in AgentMode], default=CLI_AGENT_MODE, help=argparse.SUPPRESS)
    parser.add_argument("--new", action="store_true", help="Start a new session. This is now the default.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=RESUME_LATEST,
        metavar="THREAD_ID",
        help="Resume the latest saved thread, or resume the given thread/session ID.",
    )
    parser.add_argument("--browser-visible", action="store_true", help="Launch visible Playwright browser windows.")
    _add_session_args(parser, session_help="Legacy alias for --resume THREAD_ID.")
    _add_common_model_args(parser)
    return parser


def _build_subcommand_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kolega-code", description="Kolega Code CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="Run a single prompt and print the answer.")
    ask.add_argument("prompt", help="Prompt to send to Kolega Code.")
    ask.add_argument("--project", default=".", type=Path, help="Project directory to work in.")
    ask.add_argument("--mode", choices=[mode.value for mode in AgentMode], default=CLI_AGENT_MODE, help=argparse.SUPPRESS)
    ask.add_argument("--save", action="store_true", help="Persist the session after the prompt completes.")
    ask.add_argument("--json", action="store_true", help="Emit response chunks and events as JSON.")
    ask.add_argument("--browser-visible", action="store_true", help="Launch visible Playwright browser windows.")
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

    return parser


def _overrides_from_args(args: argparse.Namespace) -> CliConfigOverrides:
    return CliConfigOverrides(
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        fast_provider=getattr(args, "fast_provider", None),
        fast_model=getattr(args, "fast_model", None),
        edit_provider=getattr(args, "edit_provider", None),
        edit_model=getattr(args, "edit_model", None),
        thinking_provider=getattr(args, "thinking_provider", None),
        thinking_model=getattr(args, "thinking_model", None),
        thinking_tokens=getattr(args, "thinking_tokens", None),
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


def _run_tui(args: argparse.Namespace) -> int:
    if importlib.util.find_spec("textual") is None:
        print("Textual is not installed. Install the CLI extra with: pip install 'kolega-code[cli]'", file=sys.stderr)
        return 2

    project_path = _validate_project(args.project_path)
    store = _store_from_args(args)
    settings_store = _settings_store_from_args(args)
    settings = settings_store.load()
    summary = {}
    try:
        config = build_agent_config(project_path, _overrides_from_args(args), settings=settings)
        summary = config_summary(config)
    except CliConfigError:
        config = None
    session = _resolve_tui_session(
        store,
        project_path,
        summary,
        args.resume,
        args.session,
    )

    from .app import KolegaCodeApp

    app = KolegaCodeApp(
        project_path=project_path,
        config=config,
        mode=CLI_AGENT_MODE,
        store=store,
        settings_store=settings_store,
        overrides=_overrides_from_args(args),
        session=session,
        browser_visible=args.browser_visible,
    )
    app.run()
    return 0


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
    config = build_agent_config(project_path, _overrides_from_args(args), settings=settings)
    summary = config_summary(config)

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
    )
    agent_ref["agent"] = agent
    if session.history:
        agent.restore_message_history(session.history)

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
                session.config = summary
                store.save(session)
            await agent.cleanup()
            return 0

    response_chunks: list[dict] = []
    try:
        async for chunk in agent.process_message_stream(prompt):
            response_chunks.append(chunk)
            if args.json:
                print(json.dumps({"kind": "chunk", "data": chunk}, default=str))
            elif chunk.get("type") == "response" and chunk.get("content"):
                print(chunk["content"], end="" if not chunk.get("complete") else "\n")

        while not manager.events.empty():
            event = manager.events.get_nowait()
            if args.json:
                print(json.dumps({"kind": "event", "data": event.model_dump()}, default=str))

        if args.save or args.session:
            session.history = agent.dump_message_history()
            session.config = summary
            store.save(session)
    finally:
        await agent.cleanup()

    if args.json:
        print(json.dumps({"kind": "summary", "chunks": len(response_chunks), "session_id": session.session_id}))
    return 0


def _parse_skill_prompt(prompt: str, catalog: SkillCatalog) -> Optional[tuple[str, str]]:
    stripped = prompt.strip()
    if not stripped.startswith("/"):
        return None

    command_text, _, rest = stripped.partition(" ")
    command = command_text.lower()
    if command == SKILLS_LIST_COMMAND:
        return "skills", rest.strip()
    if command in AGENT_BUILTIN_COMMANDS:
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


def _run_doctor(args: argparse.Namespace) -> int:
    project_path = _validate_project(args.project)
    store = _store_from_args(args)
    settings_store = _settings_store_from_args(args)
    settings = settings_store.load()
    print(f"Project: {project_path}")
    print(f"State dir: {store.root}")
    print(f"Textual installed: {importlib.util.find_spec('textual') is not None}")
    if settings.active_provider and settings.active_model:
        print(f"Stored active model: {settings.active_provider}/{settings.active_model}")
        print(f"Stored API key: {key_status(settings.active_provider, project_path, settings)}")
    else:
        print("Stored active model: not configured")

    try:
        config = build_agent_config(project_path, _overrides_from_args(args), settings=settings)
    except CliConfigError as exc:
        print(f"Configuration: invalid ({exc})")
        return 2

    summary = config_summary(config)
    print("Configuration: valid")
    print(f"Long model: {summary['long_provider']}/{summary['long_model']}")
    print(f"Fast model: {summary['fast_provider']}/{summary['fast_model']}")
    print(f"Edit model: {summary['edit_provider']}/{summary['edit_model']}")
    print(f"Thinking model: {summary['thinking_provider']}/{summary['thinking_model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
