"""``kolega-code gateway`` — run and inspect the messaging gateway daemon."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from kolega_code.gateway.access import AccessControlError, GatewayAccessControl
from kolega_code.gateway.adapters import adapter_names, build_adapter
from kolega_code.gateway.config import GatewayConfig, load_gateway_config
from kolega_code.gateway.daemon import (
    GatewayDaemon,
    GatewayDaemonError,
    LOCK_FILE_NAME,
    PID_FILE_NAME,
    STATUS_FILE_NAME,
    STATUS_STALE_SECONDS,
)
from kolega_code.gateway.handlers import EchoTurnHandler
from kolega_code.gateway.service import (
    LAUNCHD_LABEL,
    SERVICE_NAME,
    install_service,
    uninstall_service,
)


def add_gateway_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    gateway = subparsers.add_parser("gateway", help="Run the messaging gateway (chat platforms to agent sessions).")
    gateway_sub = gateway.add_subparsers(dest="gateway_command", required=True)

    run = gateway_sub.add_parser("run", help="Run the gateway in the foreground.")
    run.add_argument(
        "--adapter",
        choices=adapter_names(),
        default=None,
        help="Messaging adapter to run (default: the gateway.adapter setting, else echo).",
    )
    run.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Directory the gateway's agent sessions work in "
        "(default: the gateway.project setting, else ~/kolega-code-workspace).",
    )
    run.add_argument("--state-dir", type=Path, default=None, help="Override the state directory.")
    run.add_argument("--provider", help="Provider for the gateway's main coding model.")
    run.add_argument("--model", help="Main coding model for gateway sessions.")

    status = gateway_sub.add_parser("status", help="Show whether the gateway daemon is running.")
    status.add_argument("--state-dir", type=Path, default=None, help="Override the state directory.")

    pairing = gateway_sub.add_parser("pairing", help="Manage pending sender pairing requests.")
    pairing_sub = pairing.add_subparsers(dest="pairing_command", required=True)
    pairing_list = pairing_sub.add_parser("list", help="List pending pairing requests.")
    pairing_list.add_argument("--state-dir", type=Path, default=None, help="Override the state directory.")
    pairing_approve = pairing_sub.add_parser("approve", help="Approve a pending pairing code.")
    pairing_approve.add_argument("code", help="The pairing code a new sender was told to relay to you.")
    pairing_approve.add_argument("--state-dir", type=Path, default=None, help="Override the state directory.")

    install = gateway_sub.add_parser("install", help="Install the gateway as a user-level background service.")
    install.add_argument("--state-dir", type=Path, default=None, help="Override the state directory.")
    install.add_argument("--project", type=Path, default=None, help="Directory the service's sessions work in.")
    install.add_argument("--adapter", choices=adapter_names(), default=None, help="Messaging adapter the service runs.")

    uninstall = gateway_sub.add_parser("uninstall", help="Remove the gateway background service.")
    uninstall.add_argument("--state-dir", type=Path, default=None, help="Override the state directory.")

    telegram = gateway_sub.add_parser("telegram", help="Telegram adapter setup.")
    telegram_sub = telegram.add_subparsers(dest="telegram_command", required=True)
    setup = telegram_sub.add_parser("setup", help="Save the @BotFather bot token (stored in settings.json).")
    setup.add_argument("--token", default=None, help="Provide the token directly instead of prompting.")
    setup.add_argument(
        "--allow",
        default=None,
        metavar="USER_IDS",
        help="Comma-separated Telegram user ids for the allowlist (empty allowlist = open to anyone).",
    )
    setup.add_argument("--verify", action="store_true", help="Check the token against the Telegram API before saving.")
    setup.add_argument("--clear", action="store_true", help="Remove the saved token instead.")
    setup.add_argument("--state-dir", type=Path, default=None, help="Override the state directory.")


def run_gateway(args: argparse.Namespace) -> int:
    """Dispatch ``kolega-code gateway ...`` (sync entry point for cli/main.py)."""
    try:
        return asyncio.run(_run_gateway(args))
    except GatewayDaemonError as exc:
        print(f"gateway: {exc}", file=sys.stderr)
        return 1


async def _run_gateway(args: argparse.Namespace) -> int:
    config = load_gateway_config(
        state_dir=args.state_dir,
        project=getattr(args, "project", None),
        adapter=getattr(args, "adapter", None),
    )
    if args.gateway_command == "status":
        return _gateway_status(config)
    if args.gateway_command == "pairing":
        return _gateway_pairing(args, config)
    if args.gateway_command == "telegram":
        return await _gateway_telegram(args, config)
    if args.gateway_command in ("install", "uninstall"):
        return _gateway_service(args, config)
    return await _gateway_run(config, args)


async def _gateway_telegram(args: argparse.Namespace, config: GatewayConfig) -> int:
    import getpass

    from kolega_code.cli.settings import SettingsStore
    from kolega_code.gateway.adapters.telegram.adapter import validate_bot_token

    settings_store = SettingsStore(root=config.state_dir)
    if args.clear:
        settings = settings_store.load()
        settings.telegram_bot_token = None
        settings.gateway.pop("allowed_users", None)
        settings_store.save(settings)
        print("gateway: saved telegram bot token removed")
        return 0
    token = (args.token or "").strip()
    if not token and not sys.stdin.isatty():
        token = sys.stdin.readline().strip()
    if not token:
        try:
            token = getpass.getpass("Telegram bot token (from @BotFather): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("gateway: no token provided", file=sys.stderr)
            return 1
    if not token:
        print("gateway: no token provided", file=sys.stderr)
        return 1
    try:
        token = validate_bot_token(token)
    except ValueError as exc:
        print(f"gateway: {exc}", file=sys.stderr)
        return 1
    if args.verify:
        try:
            from aiogram import Bot

            bot = Bot(token=token)
            try:
                me = await bot.me()
            finally:
                await bot.session.close()
        except Exception as exc:  # noqa: BLE001 — verification failures must not save a bad token
            print(f"gateway: could not verify the token against Telegram: {exc}", file=sys.stderr)
            return 1
        print(f"gateway: token verified — @{me.username}")
    allowed = (args.allow or "").strip()
    if not allowed and sys.stdin.isatty() and not args.token:
        allowed = input("Your Telegram user id (for the allowlist; empty to skip): ").strip()
    settings = settings_store.load()
    settings.telegram_bot_token = token
    settings.gateway["adapter"] = "telegram"
    if allowed:
        settings.gateway["allowed_users"] = [part.strip() for part in allowed.split(",") if part.strip()]
    settings_store.save(settings)
    print(f"gateway: telegram bot token saved (state: {config.state_dir})")
    return 0


def _gateway_service(args: argparse.Namespace, config: GatewayConfig) -> int:
    if args.gateway_command == "install":
        if config.adapter != "telegram":
            print("gateway: only the telegram adapter can be installed as a service right now.", file=sys.stderr)
            return 1
        try:
            result = install_service(config)
        except RuntimeError as exc:
            print(f"gateway: {exc}", file=sys.stderr)
            return 1
        for note in result.notes:
            print(f"gateway: {note}")
        for note in _activate_service(args.gateway_command, result.unit_path):
            print(f"gateway: {note}")
        return 0
    for note in _activate_service("uninstall", None):
        print(f"gateway: {note}")
    for note in uninstall_service(config):
        print(f"gateway: {note}")
    return 0


def _activate_service(action: str, unit_path: Path | None) -> list[str]:
    """Best-effort service activation; failures are notes, never fatal."""
    import platform

    if action == "uninstall":
        if platform.system() == "Darwin":
            _run_service_command(["launchctl", "bootout", f"gui/{os.getuid()}", LAUNCHD_LABEL])
            return ["launchd agent unloaded"]
        _run_service_command(["systemctl", "--user", "disable", "--now", SERVICE_NAME])
        return ["systemd unit disabled and stopped"]
    if platform.system() == "Darwin":
        _run_service_command(["launchctl", "bootout", f"gui/{os.getuid()}", LAUNCHD_LABEL])
        _run_service_command(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(unit_path)])
        return ["launchd agent loaded and started"]
    _run_service_command(["systemctl", "--user", "daemon-reload"])
    _run_service_command(["systemctl", "--user", "enable", "--now", SERVICE_NAME])
    return ["systemd unit enabled and started"]


def _run_service_command(argv: list[str]) -> None:
    import subprocess

    try:
        subprocess.run(argv, capture_output=True, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"gateway: warning: {argv[0]} failed: {exc}", file=sys.stderr)


def _gateway_pairing(args: argparse.Namespace, config: GatewayConfig) -> int:
    access = GatewayAccessControl(
        state_dir=config.state_dir,
        allowed_users=config.allowed_users,
        pairing_enabled=config.pairing_enabled,
        code_ttl_seconds=config.pairing_code_ttl_seconds,
    )
    if args.pairing_command == "list":
        pending = access.pending()
        if not pending:
            print("gateway: no pending pairing requests")
            return 0
        for request in pending:
            who = request.sender_name or request.sender_id
            print(f"gateway: {request.code}  {who}  ({request.channel} chat {request.chat_id})")
        return 0
    try:
        sender_id = access.approve(args.code)
    except AccessControlError as exc:
        print(f"gateway: {exc}", file=sys.stderr)
        return 1
    print(f"gateway: approved sender {sender_id}; their next message will reach the gateway")
    return 0


def _gateway_status(config: GatewayConfig) -> int:
    status_path = config.state_dir / STATUS_FILE_NAME
    payload = _read_status_file(status_path)
    if payload is not None:
        adapter_state = payload.get("adapter_state") or {}
        adapter_state_text = adapter_state.get("state", "unknown")
        heartbeat_age = _heartbeat_age_seconds(payload)
        if heartbeat_age is not None and heartbeat_age <= STATUS_STALE_SECONDS:
            started = str(payload.get("started_at") or "")[:19]
            print(
                f"gateway: running (pid {payload.get('pid')})\n"
                f"  adapter: {payload.get('adapter')} ({adapter_state_text})\n"
                f"  sessions: {payload.get('active_sessions')}\n"
                f"  errors: {payload.get('recent_errors')}\n"
                f"  since: {started} (heartbeat {heartbeat_age:.0f}s ago)"
            )
            return 0
        print(f"gateway: not responding (status file is {heartbeat_age:.0f}s stale)")
        return 1
    lock = FileLock(str(config.state_dir / LOCK_FILE_NAME))
    try:
        lock.acquire(timeout=0)
    except FileLockTimeout:
        pid = ""
        pid_path = config.state_dir / PID_FILE_NAME
        if pid_path.exists():
            pid = f" (pid {pid_path.read_text(encoding='utf-8').strip()})"
        print(f"gateway: running{pid} (state: {config.state_dir})")
        return 0
    lock.release()
    print(f"gateway: not running (state: {config.state_dir})")
    return 0


def _read_status_file(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return raw if isinstance(raw, dict) else None


def _heartbeat_age_seconds(payload: dict[str, Any]) -> float | None:
    raw = payload.get("heartbeat_at")
    if not isinstance(raw, str):
        return None
    try:
        heartbeat = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - heartbeat).total_seconds())


def _build_turn_handler(config: GatewayConfig, adapter: Any, args: argparse.Namespace) -> Any:
    """The echo adapter stays the LLM-free transport harness; chat platforms
    run the real agent session host."""
    if config.adapter != "telegram":
        return EchoTurnHandler(adapter)
    from kolega_code.cli.config import CliConfigOverrides
    from kolega_code.cli.session_store import SessionStore
    from kolega_code.cli.settings import SettingsStore
    from kolega_code.gateway.sessions import AgentTurnHandler

    overrides = CliConfigOverrides(provider=getattr(args, "provider", None), model=getattr(args, "model", None))
    return AgentTurnHandler(
        config=config,
        adapter=adapter,
        store=SessionStore(root=config.state_dir),
        settings_store=SettingsStore(root=config.state_dir),
        overrides=overrides,
    )


async def _gateway_run(config: GatewayConfig, args: argparse.Namespace) -> int:
    adapter = build_adapter(config)
    turn_handler = _build_turn_handler(config, adapter, args)
    daemon = GatewayDaemon(config, adapter, turn_handler=turn_handler)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            # No signal handlers on this loop; Ctrl-C still interrupts the run.
            pass

    try:
        await daemon.start()
        print(f"gateway: {config.adapter} adapter running (project: {config.project_path}, state: {config.state_dir})")
        await stop.wait()
    finally:
        try:
            # A foreground daemon must always respond to Ctrl-C, even when a
            # channel teardown stalls; bound the graceful stop and force exit
            # rather than leaving the user with a wedged process.
            await asyncio.wait_for(daemon.stop(), timeout=15)
            print("gateway: stopped")
        except asyncio.TimeoutError:
            print("gateway: shutdown timed out; forcing exit", file=sys.stderr)
    return 0
