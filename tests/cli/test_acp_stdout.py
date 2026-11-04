from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import textwrap
from collections.abc import Coroutine
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import acp
import acp.stdio
import pytest

from kolega_code.cli import acp as cli_acp


@pytest.mark.parametrize("failure", [None, "construct", "request", "shutdown", "interrupt", "streams"])
def test_stdout_restored_after_entire_runner(
    monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, failure: str | None
) -> None:
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(cli_acp.platform, "system", lambda: "POSIX")
    reader, writer = object(), Mock()
    stages: list[str] = []

    def noise(stage: str) -> None:
        assert sys.stdout is stderr
        print(stage, flush=True)
        sys.stdout.buffer.write(f"{stage}-binary\n".encode())
        sys.stdout.buffer.flush()
        stages.append(stage)

    async def streams(*, limit: int) -> tuple[Any, Any]:
        assert sys.stdout is stdout
        assert limit == 50 * 1024 * 1024
        if failure == "streams":
            raise RuntimeError("streams")
        return reader, writer

    class Factory:
        def __init__(self, **kwargs: Any) -> None:
            noise("construct")
            if failure == "construct":
                raise RuntimeError("construct")

    class Agent:
        def __init__(self, **kwargs: Any) -> None:
            noise("agent")

        async def on_transport_message(self, *args: Any) -> None:
            pass

        async def shutdown(self) -> None:
            noise("shutdown")
            await asyncio.sleep(0)
            noise("shutdown-awaited")
            if failure == "shutdown":
                raise RuntimeError("shutdown")

    async def run_agent(agent: Agent, **kwargs: Any) -> None:
        assert kwargs == {
            "input_stream": writer,
            "output_stream": reader,
            "observers": [agent.on_transport_message],
        }
        noise("request")
        await asyncio.sleep(0)
        noise("request-awaited")
        if failure == "request":
            raise RuntimeError("request")
        if failure == "interrupt":
            raise KeyboardInterrupt

    real_run = asyncio.run

    def run(coroutine: Coroutine[Any, Any, None]) -> None:
        try:
            real_run(coroutine)
        finally:
            if failure != "streams":
                noise("after-asyncio-run")

    factory_module = ModuleType("kolega_code.acp.agent_factory")
    setattr(factory_module, "AgentFactory", Factory)
    server_module = ModuleType("kolega_code.acp.server")
    setattr(server_module, "AcpAgent", Agent)
    monkeypatch.setitem(sys.modules, factory_module.__name__, factory_module)
    monkeypatch.setitem(sys.modules, server_module.__name__, server_module)
    monkeypatch.setattr(acp.stdio, "stdio_streams", streams)
    monkeypatch.setattr(acp, "run_agent", run_agent)
    monkeypatch.setattr(cli_acp.asyncio, "run", run)

    if failure in {"construct", "request", "shutdown", "streams"}:
        with pytest.raises(RuntimeError, match=failure):
            cli_acp.run_acp(SimpleNamespace())
    else:
        assert cli_acp.run_acp(SimpleNamespace()) == (130 if failure == "interrupt" else 0)

    assert sys.stdout is stdout
    assert stdout.buffer.getvalue() == b""
    writer.close.assert_not_called()
    if failure != "streams":
        assert stages[-1] == "after-asyncio-run"
    if failure not in {"construct", "streams"}:
        assert "shutdown-awaited" in stages


@pytest.mark.asyncio
async def test_windows_writer_captures_stdout_and_preserves_borrowed_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = io.BytesIO()
    stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    writer = cli_acp._windows_stdout_writer(stdout)
    monkeypatch.setattr(sys, "stdout", stderr)
    try:
        assert isinstance(writer, asyncio.StreamWriter)
        assert not writer.can_write_eof()
        assert writer.get_extra_info("missing", "default") == "default"
        writer.write(b'{"jsonrpc":"2.0","id":1,"result":{}}\n')
        await writer.drain()
        print("hook-text", flush=True)
        sys.stdout.buffer.write(b"hook-binary\n")
        sys.stdout.buffer.flush()
        writer.writelines([b"first", b"-second\n"])
        await writer.drain()
        writer.close()
        writer.close()
        await writer.wait_closed()
        assert writer.is_closing()
        with pytest.raises(ConnectionResetError):
            await writer.drain()
        with pytest.raises(ConnectionResetError):
            writer.write(b"after-close")
    finally:
        writer.close()
        await writer.wait_closed()
    assert stdout.getvalue() == b'{"jsonrpc":"2.0","id":1,"result":{}}\nfirst-second\n'
    assert stderr.buffer.getvalue() == b"hook-text\nhook-binary\n"
    assert not stdout.closed
    assert not stderr.closed


@pytest.mark.asyncio
async def test_windows_writer_flushes_each_write_and_close() -> None:
    stdout = Mock()
    writer = cli_acp._windows_stdout_writer(stdout)
    writer.write(b"one")
    writer.write(b"two")
    await writer.drain()
    writer.transport.abort()
    await writer.wait_closed()
    assert stdout.method_calls == [
        ("write", (b"one",), {}),
        ("flush", (), {}),
        ("write", (b"two",), {}),
        ("flush", (), {}),
        ("flush", (), {}),
    ]
    stdout.close.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["write", "flush", "close"])
async def test_windows_writer_propagates_stream_errors(operation: str) -> None:
    stdout = Mock()
    writer = cli_acp._windows_stdout_writer(stdout)
    error = BrokenPipeError("client disconnected")
    if operation == "close":
        stdout.flush.side_effect = error
        writer.close()
        with pytest.raises(BrokenPipeError, match="client disconnected"):
            await writer.wait_closed()
    else:
        getattr(stdout, operation).side_effect = error
        with pytest.raises(BrokenPipeError, match="client disconnected"):
            writer.write(b"frame\n")
        stdout.flush.side_effect = None
        writer.close()
        await writer.wait_closed()
    stdout.close.assert_not_called()


def test_windows_launcher_replaces_dynamic_sdk_writer(monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None) -> None:
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(cli_acp.platform, "system", lambda: "Windows")
    reader, sdk_writer = object(), Mock()
    monkeypatch.setattr(acp.stdio, "stdio_streams", AsyncMock(return_value=(reader, sdk_writer)))
    factory_module = ModuleType("kolega_code.acp.agent_factory")
    setattr(factory_module, "AgentFactory", Mock())
    server_module = ModuleType("kolega_code.acp.server")
    agent = SimpleNamespace(shutdown=AsyncMock(), on_transport_message=Mock())
    setattr(server_module, "AcpAgent", Mock(return_value=agent))
    monkeypatch.setitem(sys.modules, factory_module.__name__, factory_module)
    monkeypatch.setitem(sys.modules, server_module.__name__, server_module)
    writers: list[asyncio.StreamWriter] = []

    async def run_agent(agent: Any, *, input_stream: asyncio.StreamWriter, **kwargs: Any) -> None:
        assert sys.stdout is stderr
        assert kwargs["output_stream"] is reader
        assert kwargs["observers"] == [agent.on_transport_message]
        assert input_stream is not sdk_writer
        writers.append(input_stream)
        input_stream.write(b"protocol\n")
        await input_stream.drain()
        print("hook", flush=True)

    monkeypatch.setattr(acp, "run_agent", run_agent)
    assert cli_acp.run_acp(SimpleNamespace()) == 0
    assert sys.stdout is stdout
    assert stdout.buffer.getvalue() == b"protocol\n"
    assert stderr.buffer.getvalue() == b"hook\n"
    assert writers[0].is_closing()
    sdk_writer.close.assert_called_once()
    agent.shutdown.assert_awaited_once()
    assert not stdout.closed
    assert not stderr.closed


_SUBPROCESS_BOOTSTRAP = textwrap.dedent(
    r"""
    import asyncio
    import builtins
    import sys
    import acp.stdio

    if sys.argv[1] == "windows":
        acp.stdio.platform.system = lambda: "Windows"

    def noise(stage):
        print("noise-" + stage, flush=True)
        sys.stdout.buffer.write(("binary-" + stage + "\n").encode())
        sys.stdout.buffer.flush()

    original_import = builtins.__import__
    patched = set()
    pending = []
    generators = []

    async def background():
        try:
            await asyncio.Event().wait()
        finally:
            noise("task-shutdown")

    async def generator():
        try:
            yield None
        finally:
            await asyncio.sleep(0)
            noise("generator-shutdown")

    def noisy_import(name, *args, **kwargs):
        module = original_import(name, *args, **kwargs)
        if name not in patched and name in {
            "kolega_code.acp.agent_factory", "kolega_code.acp.server"
        }:
            patched.add(name)
            noise("import-" + name.rsplit(".", 1)[-1])
            if name.endswith("agent_factory"):
                original_factory = module.AgentFactory.__init__
                def factory_init(self, *args, **kwargs):
                    noise("factory")
                    original_factory(self, *args, **kwargs)
                module.AgentFactory.__init__ = factory_init
            else:
                original_init = module.AcpAgent.__init__
                original_initialize = module.AcpAgent.initialize
                original_shutdown = module.AcpAgent.shutdown
                def agent_init(self, *args, **kwargs):
                    noise("agent")
                    original_init(self, *args, **kwargs)
                async def initialize(self, *args, **kwargs):
                    noise("request")
                    pending.append(asyncio.create_task(background()))
                    gen = generator()
                    generators.append(gen)
                    await anext(gen)
                    await asyncio.sleep(0)
                    noise("request-awaited")
                    return await original_initialize(self, *args, **kwargs)
                async def shutdown(self):
                    noise("shutdown")
                    await original_shutdown(self)
                    await asyncio.sleep(0)
                    noise("shutdown-awaited")
                module.AcpAgent.__init__ = agent_init
                module.AcpAgent.initialize = initialize
                module.AcpAgent.shutdown = shutdown
        return module

    builtins.__import__ = noisy_import
    original_run = asyncio.run
    def run(*args, **kwargs):
        try:
            return original_run(*args, **kwargs)
        finally:
            noise("after-asyncio-run")
    asyncio.run = run

    from kolega_code.cli.main import main
    sys.argv = ["kolega-code", "acp"]
    raise SystemExit(main())
    """
)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["native", "windows"])
async def test_subprocess_stdout_contains_only_raw_protocol_frames(isolated_cli_env: None, transport: str) -> None:
    env = dict(os.environ, PYTHON_DOTENV_DISABLED="1")
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _SUBPROCESS_BOOTSTRAP,
        transport,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    stderr_task = asyncio.create_task(process.stderr.read())
    received = bytearray()
    try:
        async with asyncio.timeout(60):
            for request_id in (1, 2):
                request = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "initialize",
                    "params": {"protocolVersion": acp.PROTOCOL_VERSION},
                }
                process.stdin.write(json.dumps(request).encode() + b"\n")
                await process.stdin.drain()
                frame = await process.stdout.readline()
                received.extend(frame)
                assert frame.endswith(b"\n"), frame
                message = json.loads(frame)
                assert set(message) == {"jsonrpc", "id", "result"}
                assert message["jsonrpc"] == "2.0"
                assert message["id"] == request_id
                assert message["result"]["protocolVersion"] == acp.PROTOCOL_VERSION
            process.stdin.close()
            received.extend(await process.stdout.read())
            assert await process.wait() == 0
            stderr = await stderr_task
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        await stderr_task

    frames = bytes(received).splitlines(keepends=True)
    assert len(frames) == 2, received
    assert all(frame.endswith(b"\n") for frame in frames)
    assert [json.loads(frame)["id"] for frame in frames] == [1, 2]
    for stage in (
        "import-agent_factory",
        "import-server",
        "factory",
        "agent",
        "request",
        "request-awaited",
        "shutdown",
        "shutdown-awaited",
        "task-shutdown",
        "generator-shutdown",
        "after-asyncio-run",
    ):
        assert f"noise-{stage}\n".encode() in stderr, stderr
        assert f"binary-{stage}\n".encode() in stderr, stderr
