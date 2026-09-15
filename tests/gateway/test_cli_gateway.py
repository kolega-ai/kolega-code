"""`kolega-code gateway` CLI: argument parsing and the status command."""

import sys
from pathlib import Path

import pytest

from kolega_code.cli.gateway import _gateway_status
from kolega_code.cli.main import parse_args
from kolega_code.gateway.adapters.base import InboundMessage
from kolega_code.gateway.config import GatewayConfig

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


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
    assert settings.gateway["allowed_users"] == ["111"]


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
            raise RuntimeError(f"unauthorized: {TEST_BOT_TOKEN}")

    monkeypatch.setattr("aiogram.Bot", FailingBot)
    state_dir = tmp_path / "state"
    args = parse_args(
        ["gateway", "telegram", "setup", "--token", TEST_BOT_TOKEN, "--verify", "--state-dir", str(state_dir)]
    )
    assert run_gateway(args) == 1
    error = capsys.readouterr().err
    assert "could not verify" in error
    assert TEST_BOT_TOKEN not in error
    assert SettingsStore(root=state_dir).load().telegram_bot_token is None


def test_status_reports_not_running(capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("kolega_code.cli.gateway.service_state_summary", lambda: None)
    config = GatewayConfig(adapter="echo", project_path=tmp_path / "ws", state_dir=tmp_path / "state")
    assert _gateway_status(config) == 0
    assert "not running" in capsys.readouterr().out


def test_status_reports_an_installed_service_that_is_failing(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "kolega_code.cli.gateway.service_state_summary",
        lambda: "launchd agent state spawn scheduled, 34318 starts, last exit code 2",
    )
    config = GatewayConfig(adapter="echo", project_path=tmp_path / "ws", state_dir=tmp_path / "state")
    assert _gateway_status(config) == 0
    out = capsys.readouterr().out
    assert "not running" in out
    assert "34318 starts" in out
    assert "last exit code 2" in out


def test_status_via_the_full_cli_path(capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch) -> None:
    # Regression: subcommands without --project must not crash on args.project.
    from kolega_code.cli.gateway import run_gateway

    monkeypatch.setattr("kolega_code.cli.gateway.service_state_summary", lambda: None)
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
    assert "access policy unknown; restart/update to verify" in out


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
    output = capsys.readouterr().out
    assert code in output
    assert "sender ID 999" in output
    assert "New Person" in output
    assert "display names are not proof of identity" in output

    approve_args = parse_args(["gateway", "pairing", "approve", code])
    assert _gateway_pairing(approve_args, config) == 0
    assert "approved sender 999" in capsys.readouterr().out

    approve_args = parse_args(["gateway", "pairing", "approve", code])
    assert _gateway_pairing(approve_args, config) == 1
    assert "Unknown or expired" in capsys.readouterr().err


@pytest.mark.parametrize("pairing_enabled", [False, True])
def test_token_only_setup_succeeds_with_no_authorized_users_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], pairing_enabled: bool
) -> None:
    from kolega_code.cli.gateway import run_gateway
    from kolega_code.cli.settings import CliSettings, SettingsStore

    store = SettingsStore(tmp_path)
    store.save(CliSettings(gateway={"pairing_enabled": pairing_enabled}))
    args = parse_args(["gateway", "telegram", "setup", "--token", TEST_BOT_TOKEN, "--state-dir", str(tmp_path)])
    assert run_gateway(args) == 0
    saved = store.load()
    assert saved.telegram_bot_token == TEST_BOT_TOKEN
    assert saved.gateway["pairing_enabled"] is pairing_enabled
    output = capsys.readouterr()
    assert "no operators are authorized" in output.err
    assert ("pairing-only" if pairing_enabled else "locked") in output.err
    assert "--allow" in output.err
    assert "Settings → Gateway" in output.err
    assert "restart" in output.out


@pytest.mark.parametrize("allow", [None, "", "  "])
def test_setup_omitted_allow_preserves_and_explicit_empty_clears_only_configured_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], allow: str | None
) -> None:
    from kolega_code.cli.gateway import run_gateway
    from kolega_code.cli.settings import CliSettings, SettingsStore
    from kolega_code.gateway.access import GatewayAccessControl

    store = SettingsStore(tmp_path)
    store.save(CliSettings(gateway={"adapter": "telegram", "allowed_users": ["111"]}))
    access = GatewayAccessControl(state_dir=tmp_path, allowed_users=("111",), pairing_enabled=True)
    message = InboundMessage(channel="telegram", chat_id="222", sender_id="222", message_id="1", text="hi")
    assert access.on_unknown_sender(message) is not None
    access.approve(access.pending()[0].code)
    approval_file = tmp_path / "gateway_allowlist.json"
    original_approvals = approval_file.read_bytes()
    argv = ["gateway", "telegram", "setup", "--token", TEST_BOT_TOKEN, "--state-dir", str(tmp_path)]
    if allow is not None:
        argv += ["--allow", allow]
    assert run_gateway(parse_args(argv)) == 0
    assert store.load().gateway["allowed_users"] == (["111"] if allow is None else [])
    assert approval_file.read_bytes() == original_approvals
    output = capsys.readouterr()
    assert "no operators are authorized" not in output.err
    if allow is not None:
        assert "persisted pairing approvals were not revoked" in output.out


@pytest.mark.parametrize("allow", ["@owner", "*", "111, ,222", "0", "-123", "١٢٣", "123,"])
def test_invalid_allow_does_not_replace_saved_token_or_call_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], allow: str
) -> None:
    from kolega_code.cli.gateway import run_gateway
    from kolega_code.cli.settings import CliSettings, SettingsStore

    store = SettingsStore(tmp_path)
    store.save(
        CliSettings(
            telegram_bot_token="456:fake-original-token", gateway={"adapter": "telegram", "allowed_users": ["111"]}
        )
    )
    before = store.path.read_bytes()

    def unexpected_bot(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid access settings must be rejected before Telegram verification")

    monkeypatch.setattr("aiogram.Bot", unexpected_bot)
    args = parse_args(
        [
            "gateway",
            "telegram",
            "setup",
            "--token",
            TEST_BOT_TOKEN,
            "--allow",
            allow,
            "--verify",
            "--state-dir",
            str(tmp_path),
        ]
    )
    assert run_gateway(args) == 1
    assert store.path.read_bytes() == before
    error = capsys.readouterr().err
    assert "gateway.allowed_users" in error
    assert TEST_BOT_TOKEN not in error


def test_gateway_cli_reports_malformed_security_settings_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from kolega_code.cli.gateway import run_gateway

    (tmp_path / "settings.json").write_text(
        '{"schema_version": 3, "gateway": {"allowed_users": [111]}, "telegram_bot_token": "fake-secret"}',
        encoding="utf-8",
    )
    assert run_gateway(parse_args(["gateway", "run", "--state-dir", str(tmp_path)])) == 1
    error = capsys.readouterr().err
    assert "gateway.allowed_users" in error
    assert "Traceback" not in error
    assert "fake-secret" not in error


@pytest.mark.parametrize(
    ("policy", "configured", "paired", "pairing"),
    [("locked", 0, 0, False), ("pairing-only", 0, 0, True), ("restricted", 0, 2, False), ("local-echo", 0, 0, False)],
)
def test_status_displays_heartbeat_policy_not_current_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], policy: str, configured: int, paired: int, pairing: bool
) -> None:
    import json
    import os
    from datetime import datetime, timezone

    payload = {
        "pid": os.getpid(),
        "heartbeat_at": datetime.now(timezone.utc).isoformat(),
        "adapter": "telegram",
        "access": {
            "policy": policy,
            "configured_users": configured,
            "paired_users": paired,
            "pairing_enabled": pairing,
        },
    }
    (tmp_path / "gateway.status.json").write_text(json.dumps(payload), encoding="utf-8")
    # Deliberately different: a running daemon has not applied these settings.
    config = GatewayConfig(
        adapter="telegram",
        state_dir=tmp_path,
        project_path=tmp_path,
        allowed_users=("999",),
        pairing_enabled=not pairing,
    )
    assert _gateway_status(config) == 0
    output = capsys.readouterr().out
    assert f"access: {policy}" in output
    assert f"configured users: {configured}" in output
    assert f"paired users: {paired}" in output
    assert f"pairing: {'enabled' if pairing else 'disabled'}" in output


def test_status_without_snapshot_reports_unknown_when_lock_is_held(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from filelock import FileLock

    config = GatewayConfig(adapter="telegram", state_dir=tmp_path, project_path=tmp_path, allowed_users=("111",))
    with FileLock(str(tmp_path / "gateway.lock")):
        assert _gateway_status(config) == 0
    assert "access policy unknown; restart/update to verify" in capsys.readouterr().out


@pytest.mark.parametrize(
    "access",
    [
        None,
        {},
        "locked",
        {"policy": "locked"},
        {"policy": "locked", "configured_users": True, "paired_users": 0, "pairing_enabled": False},
        {"policy": "locked", "configured_users": 0, "paired_users": -1, "pairing_enabled": False},
        {"policy": "locked", "configured_users": 0, "paired_users": 0, "pairing_enabled": "false"},
    ],
)
def test_incomplete_access_snapshot_reports_unknown(access: object, capsys: pytest.CaptureFixture[str]) -> None:
    from kolega_code.cli.gateway import _print_access_summary

    _print_access_summary(access)
    assert "access policy unknown; restart/update to verify" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_foreground_startup_prints_daemon_access_after_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from kolega_code.cli.gateway import _gateway_run

    start = AsyncMock()
    stop = AsyncMock()

    def status() -> SimpleNamespace:
        assert start.await_count == 1
        return SimpleNamespace(
            access={"policy": "restricted", "configured_users": 0, "paired_users": 2, "pairing_enabled": False}
        )

    daemon = SimpleNamespace(start=start, stop=stop, status=status)
    monkeypatch.setattr("kolega_code.cli.gateway.build_adapter", lambda config: object())
    monkeypatch.setattr("kolega_code.cli.gateway._build_turn_handler", lambda *args: object())
    monkeypatch.setattr("kolega_code.cli.gateway.GatewayDaemon", lambda *args, **kwargs: daemon)
    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", lambda sig, callback: callback())
    config = GatewayConfig(adapter="telegram", project_path=tmp_path, state_dir=tmp_path)
    assert await _gateway_run(config, parse_args(["gateway", "run"])) == 0
    output = capsys.readouterr().out
    assert "access: restricted" in output
    assert "paired users: 2" in output
    assert "configured users: 0" in output
    stop.assert_awaited_once()
