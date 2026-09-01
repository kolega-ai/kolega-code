"""``kolega-code gateway`` — run and inspect the messaging gateway daemon."""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from kolega_code.gateway.access import AccessControlError, GatewayAccessControl
from kolega_code.gateway.adapters import adapter_names, build_adapter
from kolega_code.gateway.config import GatewayConfig, load_gateway_config
from kolega_code.gateway.daemon import GatewayDaemon, GatewayDaemonError, LOCK_FILE_NAME, PID_FILE_NAME
from kolega_code.gateway.handlers import EchoTurnHandler


def add_gateway_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    gateway = subparsers.add_parser("gateway", help="Run the messaging gateway (chat platforms to agent sessions).")
    gateway_sub = gateway.add_subparsers(dest="gateway_command", required=True)

    run = gateway_sub.add_parser("run", help="Run the gateway in the foreground.")
    run.add_argument(
        "--adapter",
        choices=adapter_names(),
        default=None,
        help="Messaging adapter to run (default: $KOLEGA_GATEWAY_ADAPTER, else echo).",
    )
    run.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Directory the gateway's agent sessions work in "
        "(default: $KOLEGA_GATEWAY_PROJECT, else ~/kolega-code-workspace).",
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


def run_gateway(args: argparse.Namespace) -> int:
    """Dispatch ``kolega-code gateway ...`` (sync entry point for cli/main.py)."""
    try:
        return asyncio.run(_run_gateway(args))
    except GatewayDaemonError as exc:
        print(f"gateway: {exc}", file=sys.stderr)
        return 1


async def _run_gateway(args: argparse.Namespace) -> int:
    config = load_gateway_config(state_dir=args.state_dir, project=args.project)
    if args.adapter:
        config = replace(config, adapter=args.adapter)
    if args.gateway_command == "status":
        return _gateway_status(config)
    if args.gateway_command == "pairing":
        return _gateway_pairing(args, config)
    return await _gateway_run(config, args)


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
        await daemon.stop()
        print("gateway: stopped")
    return 0
