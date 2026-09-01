"""`kolega-code gateway` CLI: argument parsing and the status command."""

from pathlib import Path

import pytest

from kolega_code.cli.gateway import _gateway_status
from kolega_code.cli.main import parse_args
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


def test_status_reports_not_running(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    config = GatewayConfig(adapter="echo", project_path=tmp_path / "ws", state_dir=tmp_path / "state")
    assert _gateway_status(config) == 0
    assert "not running" in capsys.readouterr().out
