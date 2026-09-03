"""`kolega-code gateway` CLI: argument parsing and the status command."""

import sys
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


def test_parse_gateway_restart() -> None:
    args = parse_args(["gateway", "restart"])
    assert args.gateway_command == "restart"


def test_gateway_restart_reports_error_when_not_installed(capsys, monkeypatch) -> None:
    from kolega_code.cli.gateway import run_gateway

    monkeypatch.setattr("kolega_code.cli.gateway.is_service_installed", lambda: False)
    args = parse_args(["gateway", "restart"])
    exit_code = run_gateway(args)
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "not installed" in err


def test_gateway_restart_succeeds_when_installed(capsys, monkeypatch) -> None:
    from kolega_code.cli.gateway import run_gateway

    monkeypatch.setattr("kolega_code.cli.gateway.is_service_installed", lambda: True)
    monkeypatch.setattr(
        "kolega_code.cli.gateway.restart_service",
        lambda: (True, ["launchd agent restarted"]),
    )
    args = parse_args(["gateway", "restart"])
    exit_code = run_gateway(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "launchd agent restarted" in out


def test_parse_gateway_pairing_subcommands() -> None:
    args = parse_args(["gateway", "pairing", "list"])
    assert args.pairing_command == "list"
    args = parse_args(["gateway", "pairing", "approve", "ABC123"])
    assert args.pairing_command == "approve"
    assert args.code == "ABC123"


def test_parse_gateway_telegram_setup() -> None:
    args = parse_args(["gateway", "telegram", "setup"])
    assert args.telegram_command == "setup"
    args = parse_args(["gateway", "telegram", "setup", "--token", "x", "--verify"])
    assert args.token == "x"
    assert args.verify is True


TEST_BOT_TOKEN = "123:fake-bot-token-for-tests-only"


def test_telegram_setup_saves_the_token_to_settings(tmp_path: Path) -> None:
    from kolega_code.cli.gateway import run_gateway
    from kolega_code.cli.settings import SettingsStore

    state_dir = tmp_path / "state"
    args = parse_args(
        [
            "gateway",
            "telegram",
            "setup",
            "--token",
            TEST_BOT_TOKEN,
            "--allow",
            "111, 222",
            "--state-dir",
            str(state_dir),
        ]
    )
    assert run_gateway(args) == 0
    settings = SettingsStore(root=state_dir).load()
    assert settings.telegram_bot_token == TEST_BOT_TOKEN
    # Setup also flips the gateway to the telegram adapter and saves the allowlist.
    assert settings.gateway["adapter"] == "telegram"
    assert settings.gateway["allowed_users"] == ["111", "222"]


def test_telegram_setup_reads_piped_stdin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import io

    from kolega_code.cli.gateway import run_gateway
    from kolega_code.cli.settings import SettingsStore

    monkeypatch.setattr("sys.stdin", io.StringIO(f"{TEST_BOT_TOKEN}\n"))
    state_dir = tmp_path / "state"
    args = parse_args(["gateway", "telegram", "setup", "--state-dir", str(state_dir)])
    assert run_gateway(args) == 0
    assert SettingsStore(root=state_dir).load().telegram_bot_token == TEST_BOT_TOKEN


def test_telegram_setup_prompts_interactively(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Regression: the interactive prompt path (no --token, tty stdin) must
    # prompt for the token and the allowlist instead of crashing.
    from kolega_code.cli.gateway import run_gateway
    from kolega_code.cli.settings import SettingsStore

    monkeypatch.setattr("getpass.getpass", lambda prompt: TEST_BOT_TOKEN)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "111")

    state_dir = tmp_path / "state"
    args = parse_args(["gateway", "telegram", "setup", "--state-dir", str(state_dir)])
    assert run_gateway(args) == 0
    settings = SettingsStore(root=state_dir).load()
    assert settings.telegram_bot_token == TEST_BOT_TOKEN
    assert settings.gateway["allowed_users"] == ["111"]
    out = capsys.readouterr().out
    assert "token saved" in out


def test_telegram_setup_rejects_malformed_tokens(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    from kolega_code.cli.gateway import run_gateway

    state_dir = tmp_path / "state"
    args = parse_args(["gateway", "telegram", "setup", "--token", "garbage", "--state-dir", str(state_dir)])
    assert run_gateway(args) == 1
    assert "BotFather" in capsys.readouterr().err


def test_telegram_setup_clear_removes_the_token(tmp_path: Path) -> None:
    from kolega_code.cli.gateway import run_gateway
    from kolega_code.cli.settings import SettingsStore

    state_dir = tmp_path / "state"
    run_gateway(
        parse_args(
            [
                "gateway",
                "telegram",
                "setup",
                "--token",
                TEST_BOT_TOKEN,
                "--allow",
                "111",
                "--state-dir",
                str(state_dir),
            ]
        )
    )
    assert run_gateway(parse_args(["gateway", "telegram", "setup", "--clear", "--state-dir", str(state_dir)])) == 0
    settings = SettingsStore(root=state_dir).load()
    assert settings.telegram_bot_token is None
    assert "allowed_users" not in settings.gateway


def test_telegram_setup_verify_checks_the_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from unittest.mock import AsyncMock

    from kolega_code.cli.gateway import run_gateway
    from kolega_code.cli.settings import SettingsStore

    class FakeMe:
        username = "test_bot"

    class FakeBot:
        def __init__(self, token: str) -> None:
            self.token = token
            self.session = AsyncMock()

        async def me(self) -> FakeMe:
            return FakeMe()

    monkeypatch.setattr("aiogram.Bot", FakeBot)
    state_dir = tmp_path / "state"
    args = parse_args(
        ["gateway", "telegram", "setup", "--token", TEST_BOT_TOKEN, "--verify", "--state-dir", str(state_dir)]
    )
    assert run_gateway(args) == 0
    assert "token verified — @test_bot" in capsys.readouterr().out
    assert SettingsStore(root=state_dir).load().telegram_bot_token == TEST_BOT_TOKEN


def test_telegram_setup_verify_failure_does_not_save(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from unittest.mock import AsyncMock

    from kolega_code.cli.gateway import run_gateway
    from kolega_code.cli.settings import SettingsStore

    class FailingBot:
        def __init__(self, token: str) -> None:
            self.token = token
            self.session = AsyncMock()

        async def me(self) -> None:
            raise RuntimeError("unauthorized")

    monkeypatch.setattr("aiogram.Bot", FailingBot)
    state_dir = tmp_path / "state"
    args = parse_args(
        ["gateway", "telegram", "setup", "--token", TEST_BOT_TOKEN, "--verify", "--state-dir", str(state_dir)]
    )
    assert run_gateway(args) == 1
    assert "could not verify" in capsys.readouterr().err
    assert SettingsStore(root=state_dir).load().telegram_bot_token is None


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
    import os
    from datetime import datetime, timezone

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    payload = {
        "running": True,
        "adapter": "telegram",
        "adapter_state": {"state": "running"},
        "active_sessions": 3,
        "pid": os.getpid(),  # the test process itself: definitely alive
        "started_at": "2026-09-01T10:00:00+00:00",
        "recent_errors": 1,
        "heartbeat_at": datetime.now(timezone.utc).isoformat(),
    }
    (state_dir / "gateway.status.json").write_text(json.dumps(payload), encoding="utf-8")
    config = GatewayConfig(adapter="echo", project_path=tmp_path / "ws", state_dir=state_dir)
    assert _gateway_status(config) == 0
    out = capsys.readouterr().out
    assert f"running (pid {os.getpid()})" in out
    assert "sessions: 3" in out
    assert "errors: 1" in out


def test_status_reports_a_stale_heartbeat_file(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    import json
    import os

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    payload = {
        "running": True,
        "adapter": "telegram",
        "adapter_state": {"state": "running"},
        "active_sessions": 0,
        "pid": os.getpid(),  # alive, so the stale heartbeat means "not responding"
        "started_at": "2026-09-01T10:00:00+00:00",
        "recent_errors": 0,
        "heartbeat_at": "2026-09-01T10:00:00+00:00",
    }
    (state_dir / "gateway.status.json").write_text(json.dumps(payload), encoding="utf-8")
    config = GatewayConfig(adapter="echo", project_path=tmp_path / "ws", state_dir=state_dir)
    assert _gateway_status(config) == 1
    assert "not responding" in capsys.readouterr().out


def test_status_reports_a_dead_pid_as_not_running(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    import json
    import subprocess
    from datetime import datetime, timezone

    process = subprocess.Popen(["true"])
    dead_pid = process.pid
    process.wait()  # reaped: the pid is guaranteed dead for the check below
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    payload = {
        "running": True,
        "adapter": "telegram",
        "adapter_state": {"state": "running"},
        "active_sessions": 0,
        "pid": dead_pid,
        "started_at": "2026-09-01T10:00:00+00:00",
        "recent_errors": 0,
        "heartbeat_at": datetime.now(timezone.utc).isoformat(),
    }
    (state_dir / "gateway.status.json").write_text(json.dumps(payload), encoding="utf-8")
    config = GatewayConfig(adapter="echo", project_path=tmp_path / "ws", state_dir=state_dir)
    assert _gateway_status(config) == 0
    assert f"not running (stale heartbeat from pid {dead_pid})" in capsys.readouterr().out


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
