"""`kolega-code gateway` CLI: argument parsing and the status command."""

from pathlib import Path

import pytest

from kolega_code.cli.gateway import _gateway_status
from kolega_code.cli.main import parse_args
from kolega_code.gateway.adapters.base import InboundMessage
from kolega_code.gateway.config import GatewayConfig


def test_parse_gateway_run_defaults() -> None:
    args = parse_args(["gateway", "run"])
    assert args.command == "gateway"
    assert args.gateway_command == "run"
    assert args.adapter is None
    assert args.project is None


def test_parse_gateway_run_with_flags() -> None:
    args = parse_args(["gateway", "run", "--adapter", "echo", "--project", "/tmp/ws", "--state-dir", "/tmp/state"])
    assert args.adapter == "echo"
    assert args.project == Path("/tmp/ws")
    assert args.state_dir == Path("/tmp/state")


def test_parse_gateway_rejects_unknown_adapter() -> None:
    with pytest.raises(SystemExit):
        parse_args(["gateway", "run", "--adapter", "carrier-pigeon"])


def test_parse_gateway_status() -> None:
    args = parse_args(["gateway", "status"])
    assert args.gateway_command == "status"


def test_parse_gateway_pairing_subcommands() -> None:
    args = parse_args(["gateway", "pairing", "list"])
    assert args.pairing_command == "list"
    args = parse_args(["gateway", "pairing", "approve", "ABC123"])
    assert args.pairing_command == "approve"
    assert args.code == "ABC123"


def test_status_reports_not_running(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    config = GatewayConfig(adapter="echo", project_path=tmp_path / "ws", state_dir=tmp_path / "state")
    assert _gateway_status(config) == 0
    assert "not running" in capsys.readouterr().out


def test_status_via_the_full_cli_path(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    # Regression: subcommands without --project must not crash on args.project.
    from kolega_code.cli.gateway import run_gateway

    args = parse_args(["gateway", "status", "--state-dir", str(tmp_path / "state")])
    assert run_gateway(args) == 0
    assert "not running" in capsys.readouterr().out


def test_pairing_list_via_the_full_cli_path(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    from kolega_code.cli.gateway import run_gateway

    args = parse_args(["gateway", "pairing", "list", "--state-dir", str(tmp_path / "state")])
    assert run_gateway(args) == 0
    assert "no pending pairing requests" in capsys.readouterr().out


def test_status_reads_a_fresh_heartbeat_file(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    import json
    from datetime import datetime, timezone

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    payload = {
        "running": True,
        "adapter": "telegram",
        "adapter_state": {"state": "running"},
        "active_sessions": 3,
        "pid": 4242,
        "started_at": "2026-09-01T10:00:00+00:00",
        "recent_errors": 1,
        "heartbeat_at": datetime.now(timezone.utc).isoformat(),
    }
    (state_dir / "gateway.status.json").write_text(json.dumps(payload), encoding="utf-8")
    config = GatewayConfig(adapter="echo", project_path=tmp_path / "ws", state_dir=state_dir)
    assert _gateway_status(config) == 0
    out = capsys.readouterr().out
    assert "running (pid 4242)" in out
    assert "sessions: 3" in out
    assert "errors: 1" in out


def test_status_reports_a_stale_heartbeat_file(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    import json

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    payload = {
        "running": True,
        "adapter": "telegram",
        "adapter_state": {"state": "running"},
        "active_sessions": 0,
        "pid": 4242,
        "started_at": "2026-09-01T10:00:00+00:00",
        "recent_errors": 0,
        "heartbeat_at": "2026-09-01T10:00:00+00:00",
    }
    (state_dir / "gateway.status.json").write_text(json.dumps(payload), encoding="utf-8")
    config = GatewayConfig(adapter="echo", project_path=tmp_path / "ws", state_dir=state_dir)
    assert _gateway_status(config) == 1
    assert "not responding" in capsys.readouterr().out


def test_pairing_list_and_approve(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    from kolega_code.cli.gateway import _gateway_pairing
    from kolega_code.gateway.access import GatewayAccessControl

    config = GatewayConfig(
        adapter="echo",
        project_path=tmp_path / "ws",
        state_dir=tmp_path / "state",
        allowed_users=("123",),
        pairing_enabled=True,
    )
    access = GatewayAccessControl(state_dir=config.state_dir, allowed_users=config.allowed_users, pairing_enabled=True)
    reply = access.on_unknown_sender(
        InboundMessage(
            channel="recording", chat_id="42", sender_id="999", sender_name="New Person", message_id="m-1", text="hi"
        )
    )
    assert reply is not None
    code = reply.rsplit(" ", 1)[-1]

    list_args = parse_args(["gateway", "pairing", "list"])
    assert _gateway_pairing(list_args, config) == 0
    assert code in capsys.readouterr().out

    approve_args = parse_args(["gateway", "pairing", "approve", code])
    assert _gateway_pairing(approve_args, config) == 0
    assert "approved sender 999" in capsys.readouterr().out

    approve_args = parse_args(["gateway", "pairing", "approve", code])
    assert _gateway_pairing(approve_args, config) == 1
    assert "Unknown or expired" in capsys.readouterr().err
