from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from decimal import Decimal
import json
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
from typing import Any

from acp import PROTOCOL_VERSION, spawn_agent_process
from acp.helpers import SessionUpdate
from acp.interfaces import Client
from acp.schema import ToolCallUpdate
import pytest


pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]

ROOT = Path(__file__).resolve().parents[2]
META = "dev.kolega/usage"
HOSTS = {"openrouter.ai", "api.x.ai", "api.deepseek.com"}
MODELS = {
    "openrouter": "openai/gpt-4.1-mini",
    "xai": "grok-4.7",
    "deepseek": "deepseek-flash",
}


@dataclass
class Reply:
    input_tokens: int = 100
    output_tokens: int = 10
    text: str = "The fixture has completed the requested task."
    cost: str | None = "0.0123"
    tools: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    duplicate_usage: bool = False
    incomplete: bool = False
    failed: bool = False
    http_status: int = 200
    gate: asyncio.Event | None = None
    usage_before_gate: bool = False
    match: str | None = None
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    evidence_sent: asyncio.Event = field(default_factory=asyncio.Event)
    completed: asyncio.Event = field(default_factory=asyncio.Event)


class AccountingClient(Client):
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self.changed = asyncio.Event()

    async def session_update(self, session_id: str, update: SessionUpdate, **kwargs: Any) -> None:
        self.updates.append({"sessionId": session_id, **update.model_dump(by_alias=True, exclude_none=True)})
        self.changed.set()

    async def request_permission(
        self, session_id: str, tool_call: ToolCallUpdate, options: list[Any], **kwargs: Any
    ) -> Any:
        option = next(option for option in options if option.kind in ("allow_once", "allow_always"))
        return {"outcome": {"outcome": "selected", "optionId": option.option_id}}

    async def wait_total(self, total: int, *, open_requests: int | None = None) -> dict[str, Any]:
        async with asyncio.timeout(60):
            while True:
                self.changed.clear()
                for update in reversed(self.updates):
                    extension = update.get("_meta", {}).get(META)
                    if (
                        extension
                        and extension["session"]["totalTokens"] == total
                        and (open_requests is None or extension["session"]["requests"]["open"] == open_requests)
                    ):
                        return extension
                await self.changed.wait()

    async def wait_text(self, text: str) -> None:
        async with asyncio.timeout(60):
            while True:
                self.changed.clear()
                if any(text in json.dumps(update) for update in self.updates):
                    return
                await self.changed.wait()


class NativeProviderWire:
    """Terminate SDK HTTPS_PROXY CONNECT locally, retaining the native billed route."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.requests: list[dict[str, Any]] = []
        self.replies: list[Reply] = []
        self.errors: list[str] = []
        self.tasks: set[asyncio.Task[None]] = set()
        self.server: asyncio.Server | None = None
        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.proxy = ""
        self.document_requests = 0

    async def __aenter__(self) -> NativeProviderWire:
        executable = shutil.which("openssl")
        assert executable is not None, "The hermetic native-route fixture requires openssl to issue a test certificate"
        cert = self.directory / "provider-cert.pem"
        key = self.directory / "provider-key.pem"
        subprocess.run(
            [
                executable,
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-days",
                "2",
                "-subj",
                "/CN=ACP hermetic provider fixture",
                "-addext",
                "subjectAltName=DNS:openrouter.ai,DNS:api.x.ai,DNS:api.deepseek.com",
                "-keyout",
                str(key),
                "-out",
                str(cert),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.context.load_cert_chain(cert, key)
        self.server = await asyncio.start_server(self._connected, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        self.proxy = f"http://127.0.0.1:{port}"
        return self

    async def __aexit__(self, *args: Any) -> None:
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()
        for task in tuple(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        assert not self.errors, self.errors

    def _connected(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.create_task(self._serve(reader, writer))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def add(self, **kwargs: Any) -> Reply:
        reply = Reply(**kwargs)
        self.replies.append(reply)
        return reply

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            async with asyncio.timeout(120):
                connect = await reader.readuntil(b"\r\n\r\n")
                method, target, _ = connect.split(b"\r\n", 1)[0].decode().split()
                if method == "GET" and target == "http://fixture.invalid/accounting.txt":
                    self.document_requests += 1
                    content = b"The accounting fixture has exactly seven blue widgets."
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nConnection: close\r\nContent-Length: "
                        + str(len(content)).encode()
                        + b"\r\n\r\n"
                        + content
                    )
                    await writer.drain()
                    return
                host, port = target.rsplit(":", 1)
                assert method == "CONNECT" and host in HOSTS and port == "443", connect
                writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
                await writer.drain()
                await writer.start_tls(self.context)
                headers_raw = await reader.readuntil(b"\r\n\r\n")
                first, *lines = headers_raw.decode().split("\r\n")
                method, path, _ = first.split()
                headers = dict(line.split(": ", 1) for line in lines if ": " in line)
                length = int(next(value for key, value in headers.items() if key.lower() == "content-length"))
                body = json.loads(await reader.readexactly(length))
                self.requests.append({"host": host, "path": path, "body": body})
                assert method == "POST", first
                assert self.replies, f"Unscripted provider request: {host}{path}: {str(body)[:500]}"
                messages = body.get("messages", body.get("input", []))
                users = [message for message in messages if message.get("role") == "user"]
                last_user = json.dumps(users)
                selected = next(
                    (
                        index
                        for index, candidate in enumerate(self.replies)
                        if candidate.match is None or candidate.match in last_user
                    ),
                    None,
                )
                assert selected is not None, f"No scripted response for {last_user[:500]}"
                reply = self.replies.pop(selected)
                reply.entered.set()
                assert path.endswith("/responses" if host == "api.deepseek.com" else "/chat/completions"), path
                if reply.http_status != 200:
                    encoded = json.dumps(
                        {"error": {"message": "Hermetic retry fixture", "type": "rate_limit_error"}}
                    ).encode()
                    writer.write(
                        f"HTTP/1.1 {reply.http_status} Fixture retry\r\nContent-Type: application/json\r\n"
                        f"Retry-After: 0\r\nConnection: close\r\nContent-Length: {len(encoded)}\r\n\r\n".encode()
                        + encoded
                    )
                    await writer.drain()
                    reply.completed.set()
                    return
                if not body.get("stream"):
                    assert host != "api.deepseek.com", body
                    payload = {
                        "id": f"chatcmpl-helper-{len(self.requests)}",
                        "object": "chat.completion",
                        "created": 1,
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": reply.text},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": reply.input_tokens,
                            "completion_tokens": reply.output_tokens,
                            "total_tokens": reply.input_tokens + reply.output_tokens,
                            "cost": float(reply.cost) if reply.cost is not None else None,
                        },
                    }
                    encoded = json.dumps(payload).encode()
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
                        + str(len(encoded)).encode()
                        + b"\r\n\r\n"
                        + encoded
                    )
                    await writer.drain()
                    reply.evidence_sent.set()
                    reply.completed.set()
                    return
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    b"Cache-Control: no-cache\r\nConnection: close\r\n\r\n"
                )
                await writer.drain()
                if host == "api.deepseek.com":
                    await self._responses(writer, body, reply)
                else:
                    await self._chat(writer, body, reply, host)
                reply.completed.set()
        except (ConnectionError, asyncio.IncompleteReadError, ssl.SSLError):
            pass
        except Exception as exc:
            self.errors.append(repr(exc))
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, ssl.SSLError):
                pass

    @staticmethod
    async def _event(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
        event = f"event: {payload['type']}\n" if "type" in payload else ""
        writer.write(f"{event}data: {json.dumps(payload)}\n\n".encode())
        await writer.drain()

    async def _chat(self, writer: asyncio.StreamWriter, body: dict[str, Any], reply: Reply, host: str) -> None:
        base = {
            "id": f"chatcmpl-{len(self.requests)}",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": body["model"],
        }
        delta: dict[str, Any] = {"role": "assistant", "content": reply.text}
        if reply.tools:
            delta["tool_calls"] = [
                {
                    "index": index,
                    "id": f"call_{len(self.requests)}_{index}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
                for index, (name, arguments) in enumerate(reply.tools)
            ]
        await self._event(writer, {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
        usage: dict[str, Any] = {
            "prompt_tokens": reply.input_tokens,
            "completion_tokens": reply.output_tokens,
            "total_tokens": reply.input_tokens + reply.output_tokens,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
        if reply.cost is not None:
            if host == "api.x.ai":
                usage["cost_in_usd_ticks"] = int(Decimal(reply.cost) * 10**10)
            else:
                usage["cost"] = float(reply.cost)
                usage["cost_details"] = {"upstream_inference_cost": 0.008}
        usage_event = {**base, "choices": [], "usage": usage}
        if reply.usage_before_gate:
            await self._event(writer, usage_event)
            await self._event(
                writer,
                {
                    **base,
                    "choices": [{"index": 0, "delta": {"content": "USAGE_OBSERVED_BARRIER"}, "finish_reason": None}],
                },
            )
            reply.evidence_sent.set()
        if reply.gate is not None:
            await reply.gate.wait()
        await self._event(
            writer,
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if reply.tools else "stop"}]},
        )
        if reply.duplicate_usage:
            await self._event(
                writer,
                {
                    **base,
                    "choices": [],
                    "usage": {**usage, "prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55},
                },
            )
        await self._event(writer, usage_event)
        if reply.duplicate_usage:
            await self._event(writer, usage_event)
        reply.evidence_sent.set()
        writer.write(b"data: [DONE]\n\n")
        await writer.drain()

    async def _responses(self, writer: asyncio.StreamWriter, body: dict[str, Any], reply: Reply) -> None:
        response = {
            "id": f"resp_{len(self.requests)}",
            "object": "response",
            "created_at": 1,
            "status": "in_progress",
            "model": body["model"],
            "output": [],
        }
        await self._event(writer, {"type": "response.created", "sequence_number": 0, "response": response})
        await self._event(
            writer,
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "item_id": "msg_fixture",
                "output_index": 0,
                "content_index": 0,
                "delta": reply.text,
            },
        )
        if reply.gate is not None:
            await reply.gate.wait()
        status = "failed" if reply.failed else "incomplete" if reply.incomplete else "completed"
        response.update(
            status=status,
            usage={
                "input_tokens": reply.input_tokens,
                "output_tokens": reply.output_tokens,
                "total_tokens": reply.input_tokens + reply.output_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        )
        if reply.incomplete:
            response["incomplete_details"] = {"reason": "max_output_tokens"}
        if reply.failed:
            response["error"] = {"code": "server_error", "message": "Hermetic stream failure after authoritative usage"}
        event = {"type": f"response.{status}", "sequence_number": 2, "response": response}
        await self._event(writer, event)
        if reply.duplicate_usage:
            await self._event(writer, {**event, "sequence_number": 3})
        reply.evidence_sent.set()

    def assert_clean(self) -> None:
        assert self.errors == []
        assert self.replies == [], "Some scripted requests never reached the real provider client"


def environment(tmp_path: Path, wire: NativeProviderWire, provider: str) -> dict[str, str]:
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__).with_name("_accounting_subprocess_bootstrap.py"), bootstrap / "sitecustomize.py")
    settings = {
        "schema_version": 3,
        "active_provider": provider,
        "active_model": MODELS[provider],
        "api_keys": {name: "sk-hermetic-not-a-real-api-key" for name in MODELS},
        "lsp_enabled": False,
    }
    (state / "settings.json").write_text(json.dumps(settings))
    env = {key: value for key, value in os.environ.items() if key in ("PATH", "SYSTEMROOT", "LANG")}
    env.update(
        HOME=str(home),
        TMPDIR=str(tmp_path),
        XDG_CONFIG_HOME=str(home / ".config"),
        XDG_CACHE_HOME=str(home / ".cache"),
        KOLEGA_CODE_STATE_DIR=str(state),
        KOLEGA_CODE_LSP="off",
        PYTHONPATH=os.pathsep.join((str(bootstrap), str(ROOT))),
        CI="1",
        PYTHONUNBUFFERED="1",
        HTTPS_PROXY=wire.proxy,
        HTTP_PROXY=wire.proxy,
        ALL_PROXY=wire.proxy,
        NO_PROXY="",
        SSL_CERT_FILE=str(tmp_path / "provider-cert.pem"),
    )
    return env


@asynccontextmanager
async def process(client: AccountingClient, env: dict[str, str], tmp_path: Path) -> AsyncIterator[Any]:
    async with asyncio.timeout(120):
        async with spawn_agent_process(
            client,
            sys.executable,
            "-m",
            "kolega_code.cli.main",
            "acp",
            env=env,
            cwd=str(tmp_path / "project"),
            transport_kwargs={"shutdown_timeout": 30},
        ) as (connection, child):
            stderr: list[str] = []

            async def drain_stderr() -> None:
                assert child.stderr is not None
                while line := await child.stderr.readline():
                    stderr.append(line.decode(errors="replace"))

            drainer = asyncio.create_task(drain_stderr())
            try:
                await connection.initialize(protocol_version=PROTOCOL_VERSION)
                assert child.returncode is None
                yield connection
            except Exception as exc:
                raise AssertionError(
                    f"ACP subprocess failed: {exc!r}; data={getattr(exc, 'data', None)!r}\n{''.join(stderr)}"
                ) from exc
            finally:
                drainer.cancel()
                await asyncio.gather(drainer, return_exceptions=True)


def extension(response: Any) -> dict[str, Any]:
    payload = response.model_dump(by_alias=True, exclude_none=True)
    assert META in payload.get("_meta", {}), payload
    result = payload["_meta"][META]
    assert result["schemaVersion"] == 1
    return result


async def prompt(connection: Any, session_id: str, text: str) -> Any:
    return await connection.prompt(session_id=session_id, prompt=[{"type": "text", "text": text}])


def settled_facts(tmp_path: Path, session_id: str) -> list[dict[str, Any]]:
    paths = list((tmp_path / "state").rglob("events.jsonl"))
    assert paths
    events = [json.loads(line) for path in paths if session_id in str(path) for line in path.read_text().splitlines()]
    starts = [event["payload"] for event in events if event["type"] == "llm.request_started"]
    facts = [event["payload"] for event in events if event["type"] == "llm.request_settled"]
    identities = [(fact["run_id"], fact["request_id"]) for fact in facts]
    assert len(identities) == len(set(identities)), "A request must have only one terminal fact"
    assert set(identities) <= {(fact["run_id"], fact["request_id"]) for fact in starts}
    return facts


def assert_totals(scope: dict[str, Any], input_tokens: int, output_tokens: int, requests: int) -> None:
    assert scope["inputTokens"] == input_tokens, scope
    assert scope["outputTokens"] == output_tokens, scope
    assert scope["totalTokens"] == input_tokens + output_tokens, scope
    assert scope["requests"]["requests"] == requests, scope
    assert scope["requests"]["open"] == 0, scope
    assert scope["persistence"]["healthy"] is True, scope


@pytest.mark.parametrize("provider", ["openrouter", "xai", "deepseek"])
async def test_native_provider_two_prompts_restart_and_model_switch(tmp_path: Path, provider: str) -> None:
    async with NativeProviderWire(tmp_path) as wire:
        wire.add(duplicate_usage=True, cost=None if provider == "deepseek" else "0.0123")
        wire.add(input_tokens=140, output_tokens=20, cost=None if provider == "deepseek" else "0.0234")
        env = environment(tmp_path, wire, provider)
        client = AccountingClient()  # pyright: ignore[reportAbstractUsage]
        async with process(client, env, tmp_path) as connection:
            session = await connection.new_session(cwd=str(tmp_path / "project"), mcp_servers=[])
            session_id = session.session_id
            first = extension(await prompt(connection, session_id, "FIRST_ACCOUNTING_PROMPT"))
            assert_totals(first["session"], 100, 10, 1)
            assert_totals(first["prompt"], 100, 10, 1)
            second = extension(await prompt(connection, session_id, "SECOND_ACCOUNTING_PROMPT"))
            assert_totals(second["session"], 240, 30, 2)
            assert_totals(second["prompt"], 140, 20, 1)
            assert first["prompt"]["promptId"] != second["prompt"]["promptId"]
            assert first["session"]["runId"] == second["session"]["runId"]
            assert second["session"]["revision"] > first["session"]["revision"]
            assert second["session"]["context"]["used"] != 270
            await client.wait_total(270)
            facts = settled_facts(tmp_path, session_id)
            assert len(facts) == 2
            assert {fact["provider"] for fact in facts} == {provider}
            assert [fact["prompt_id"] for fact in facts] == [
                first["prompt"]["promptId"],
                second["prompt"]["promptId"],
            ]
            assert sum(fact["usage"]["total_tokens"] for fact in facts) == 270
            costs = [fact["cost"] for fact in facts]
            if provider == "deepseek":
                assert all(cost["source"] == "estimated" and cost["pricing_source"] == "catalogue" for cost in costs)
                for fact, (wire_input, wire_output) in zip(facts, [(100, 10), (140, 20)], strict=True):
                    card = fact["pinned_pricing"]["pricing"]
                    assert card["unit"] == "per_million_tokens"
                    rates = dict(card["rates"])
                    for schedule in card.get("schedules", []):
                        if schedule["id"] in fact["cost"]["applied_schedules"]:
                            rates.update(schedule["rates"])
                    expected = (
                        wire_input * Decimal(rates["input"]) + wire_output * Decimal(rates["output"])
                    ) / 1_000_000
                    assert fact["cost"]["complete"] is True
                    assert Decimal(fact["cost"]["amount"]) == expected
            else:
                assert [cost["source"] for cost in costs] == ["provider_reported", "provider_reported"]
                assert [Decimal(cost["amount"]) for cost in costs] == [Decimal("0.0123"), Decimal("0.0234")]
            monetary = second["session"]["cost"]["monetary"]
            assert len(monetary) == 1 and monetary[0]["currency"] == "USD"
            assert Decimal(monetary[0]["amount"]) == sum(Decimal(cost["amount"]) for cost in costs)
            cost_updates = [
                update
                for update in client.updates
                if update.get("sessionUpdate") == "usage_update"
                and update.get("_meta", {}).get(META, {}).get("session", {}).get("totalTokens") == 270
            ]
            assert cost_updates, client.updates
            assert Decimal(str(cost_updates[-1]["cost"]["amount"])) == Decimal(monetary[0]["amount"])
            original_run = second["session"]["runId"]
            original_cost = second["session"]["cost"]
        wire.assert_clean()
        resumed_client = AccountingClient()  # pyright: ignore[reportAbstractUsage]
        async with process(resumed_client, env, tmp_path) as connection:
            await connection.load_session(cwd=str(tmp_path / "project"), session_id=session_id, mcp_servers=[])
            restored = await resumed_client.wait_total(270)
            assert_totals(restored["session"], 240, 30, 2)
            assert restored["session"]["runId"] != original_run
            assert restored["session"]["cost"] == original_cost
            switched = "deepseek" if provider != "deepseek" else "xai"
            await connection.set_config_option(
                session_id=session_id, config_id="model", value=f"{switched}/{MODELS[switched]}"
            )
            wire.add(input_tokens=7, output_tokens=3, cost=None if switched == "deepseek" else "0.0005")
            third = extension(await prompt(connection, session_id, "AFTER_MODEL_SWITCH"))
            assert_totals(third["session"], 247, 33, 3)
            assert_totals(third["prompt"], 7, 3, 1)
            assert third["session"]["runId"] == restored["session"]["runId"]
            assert [row["provider"] for row in third["session"]["breakdown"]] == [provider, switched]
            assert wire.requests[-1]["body"]["model"] == MODELS[switched]
            all_facts = settled_facts(tmp_path, session_id)
            assert len(all_facts) == 3
            assert Decimal(third["session"]["cost"]["monetary"][0]["amount"]) == sum(
                Decimal(fact["cost"]["amount"]) for fact in all_facts
            )
        wire.assert_clean()


async def test_scripted_tools_account_for_real_helper_and_delegate(tmp_path: Path) -> None:
    async with NativeProviderWire(tmp_path) as wire:
        wire.add(
            input_tokens=10,
            output_tokens=1,
            tools=[
                (
                    "web_fetch",
                    {"url": "http://fixture.invalid/accounting.txt", "instruction": "How many blue widgets?"},
                ),
            ],
        )
        wire.add(
            input_tokens=20,
            output_tokens=2,
            text=json.dumps(
                {
                    "answer": "Seven blue widgets.",
                    "evidence": ["exactly seven blue widgets"],
                    "insufficient": False,
                }
            ),
        )
        wire.add(
            input_tokens=30,
            output_tokens=3,
            tools=[
                (
                    "dispatch_agent",
                    {"agent_type": "general", "task": "DELEGATE_ONLY: report that the delegated task is complete."},
                ),
            ],
        )
        wire.add(input_tokens=40, output_tokens=4, match="DELEGATE_ONLY")
        wire.add(input_tokens=50, output_tokens=5, match="TOOLS_AND_HELPERS")
        env = environment(tmp_path, wire, "openrouter")
        client = AccountingClient()  # pyright: ignore[reportAbstractUsage]
        async with process(client, env, tmp_path) as connection:
            session = await connection.new_session(cwd=str(tmp_path / "project"), mcp_servers=[])
            result = extension(await prompt(connection, session.session_id, "TOOLS_AND_HELPERS"))
            assert_totals(result["session"], 150, 15, 5)
            assert_totals(result["prompt"], 150, 15, 5)
            facts = settled_facts(tmp_path, session.session_id)
            assert len(facts) == 5
            assert sum(fact["usage"]["total_tokens"] for fact in facts) == 165
            assert {fact["prompt_id"] for fact in facts} == {result["prompt"]["promptId"]}
            assert sum(fact["origin"]["kind"] == "history" and not fact["origin"].get("depth") for fact in facts) == 3
            assert any(fact["origin"] == {"kind": "helper", "helper": "web_fetch"} for fact in facts)
            child = next(fact for fact in facts if fact["origin"].get("depth") == 1)
            assert child["origin"]["parent_tool_call_id"]
            assert child["origin"]["depth"] == 1
            assert wire.document_requests == 1
            assert len(wire.requests) == 5
            assert sum(request["body"].get("stream") is not True for request in wire.requests) == 1
        wire.assert_clean()


async def test_truncated_native_responses_continue_as_distinct_requests(tmp_path: Path) -> None:
    async with NativeProviderWire(tmp_path) as wire:
        wire.add(input_tokens=9, output_tokens=4, incomplete=True, cost=None)
        wire.add(input_tokens=11, output_tokens=6, cost=None)
        env = environment(tmp_path, wire, "deepseek")
        client = AccountingClient()  # pyright: ignore[reportAbstractUsage]
        async with process(client, env, tmp_path) as connection:
            session = await connection.new_session(cwd=str(tmp_path / "project"), mcp_servers=[])
            result = extension(await prompt(connection, session.session_id, "CONTINUE_AFTER_TRUNCATION"))
            assert_totals(result["session"], 20, 10, 2)
            assert_totals(result["prompt"], 20, 10, 2)
            facts = settled_facts(tmp_path, session.session_id)
            assert len(facts) == 2
            assert {fact["prompt_id"] for fact in facts} == {result["prompt"]["promptId"]}
            assert len(wire.requests) == 2
            assert "continue" in json.dumps(wire.requests[1]["body"]).lower()
        wire.assert_clean()


async def test_sdk_http_retry_does_not_invent_an_extra_accounted_request(tmp_path: Path) -> None:
    async with NativeProviderWire(tmp_path) as wire:
        wire.add(http_status=429)
        wire.add(input_tokens=12, output_tokens=3, cost="0.001")
        env = environment(tmp_path, wire, "openrouter")
        client = AccountingClient()  # pyright: ignore[reportAbstractUsage]
        async with process(client, env, tmp_path) as connection:
            session = await connection.new_session(cwd=str(tmp_path / "project"), mcp_servers=[])
            result = extension(await prompt(connection, session.session_id, "SDK_RETRY"))
            assert_totals(result["session"], 12, 3, 1)
            assert_totals(result["prompt"], 12, 3, 1)
            assert len(wire.requests) == 2
            assert len(settled_facts(tmp_path, session.session_id)) == 1
            assert any("SDK-internal retry" in item for item in result["session"]["limitations"])
        wire.assert_clean()


async def test_failed_stream_evidence_survives_real_agent_loop_retry(tmp_path: Path) -> None:
    async with NativeProviderWire(tmp_path) as wire:
        wire.add(input_tokens=9, output_tokens=4, failed=True, cost=None)
        wire.add(input_tokens=11, output_tokens=6, cost=None)
        env = environment(tmp_path, wire, "deepseek")
        client = AccountingClient()  # pyright: ignore[reportAbstractUsage]
        async with process(client, env, tmp_path) as connection:
            session = await connection.new_session(cwd=str(tmp_path / "project"), mcp_servers=[])
            result = extension(await prompt(connection, session.session_id, "RETRY_FAILED_STREAM"))
            assert_totals(result["session"], 20, 10, 2)
            assert_totals(result["prompt"], 20, 10, 2)
            assert result["session"]["requests"]["failed"] == 1
            facts = settled_facts(tmp_path, session.session_id)
            assert len(facts) == 2
            assert [fact["state"] for fact in facts] == ["failed", "responded"]
            assert [fact["usage"]["total_tokens"] for fact in facts] == [13, 17]
            assert {fact["prompt_id"] for fact in facts} == {result["prompt"]["promptId"]}
            assert len(wire.requests) == 2
        wire.assert_clean()


async def test_cancel_preserves_consumed_wire_tokens_and_reported_charge(tmp_path: Path) -> None:
    async with NativeProviderWire(tmp_path) as wire:
        release = asyncio.Event()
        reply = wire.add(input_tokens=80, output_tokens=8, cost="0.0042", gate=release, usage_before_gate=True)
        env = environment(tmp_path, wire, "openrouter")
        client = AccountingClient()  # pyright: ignore[reportAbstractUsage]
        async with process(client, env, tmp_path) as connection:
            session = await connection.new_session(cwd=str(tmp_path / "project"), mcp_servers=[])
            running = asyncio.create_task(prompt(connection, session.session_id, "CANCEL_AFTER_USAGE"))
            try:
                await reply.evidence_sent.wait()
                await client.wait_text("USAGE_OBSERVED_BARRIER")
                await connection.cancel(session_id=session.session_id)
                response = await running
                assert response.stop_reason == "cancelled"
                result = extension(response)
                assert_totals(result["session"], 80, 8, 1)
                assert_totals(result["prompt"], 80, 8, 1)
                facts = settled_facts(tmp_path, session.session_id)
                assert len(facts) == 1
                assert facts[0]["state"] in ("failed", "cancelled")
                assert facts[0]["usage"]["total_tokens"] == 88
                assert facts[0]["cost"]["source"] == "provider_reported"
                assert Decimal(facts[0]["cost"]["amount"]) == Decimal("0.0042")
                assert len(wire.requests) == 1, "Cancelling must never cause another inference request"
            finally:
                release.set()
                if not running.done():
                    running.cancel()
                await asyncio.gather(running, return_exceptions=True)
        wire.assert_clean()


async def test_background_launched_by_a_settles_during_b_without_charging_b(tmp_path: Path) -> None:
    async with NativeProviderWire(tmp_path) as wire:
        child_release = asyncio.Event()
        b_release = asyncio.Event()
        wire.add(
            input_tokens=10,
            output_tokens=1,
            tools=[
                (
                    "dispatch_agent",
                    {
                        "agent_type": "general",
                        "task": "CHILD_A_TASK: finish the independent report.",
                        "background": True,
                    },
                ),
            ],
        )
        child = wire.add(input_tokens=30, output_tokens=3, match="CHILD_A_TASK", gate=child_release)
        wire.add(input_tokens=20, output_tokens=2, match="LAUNCH_A", gate=child.entered)
        b_reply = wire.add(input_tokens=40, output_tokens=4, match="PROMPT_B", gate=b_release)
        env = environment(tmp_path, wire, "openrouter")
        client = AccountingClient()  # pyright: ignore[reportAbstractUsage]
        async with process(client, env, tmp_path) as connection:
            session = await connection.new_session(cwd=str(tmp_path / "project"), mcp_servers=[])
            first = extension(await prompt(connection, session.session_id, "LAUNCH_A"))
            a_id = first["prompt"]["promptId"]
            assert first["prompt"]["totalTokens"] == 33
            assert first["prompt"]["requests"]["open"] == 1
            assert first["session"]["requests"]["requests"] == 3
            original_run = first["session"]["runId"]
            running_b = asyncio.create_task(prompt(connection, session.session_id, "PROMPT_B"))
            try:
                await b_reply.entered.wait()
                child_release.set()
                during_b = await client.wait_total(66, open_requests=1)
                assert during_b["session"]["requests"]["open"] == 1
                assert during_b["session"]["runId"] == original_run
                b_release.set()
                second = extension(await running_b)
                assert_totals(second["session"], 100, 10, 4)
                assert_totals(second["prompt"], 40, 4, 1)
                assert second["prompt"]["promptId"] != a_id
                facts = settled_facts(tmp_path, session.session_id)
                assert len(facts) == 4
                child_fact = next(fact for fact in facts if fact["origin"].get("depth") == 1)
                assert child_fact["prompt_id"] == a_id
                assert child_fact["usage"]["total_tokens"] == 33
                a_facts = [fact for fact in facts if fact["prompt_id"] == a_id]
                assert sum(fact["usage"]["total_tokens"] for fact in a_facts) == 66
                b_facts = [fact for fact in facts if fact["prompt_id"] == second["prompt"]["promptId"]]
                assert sum(fact["usage"]["total_tokens"] for fact in b_facts) == 44
                assert len(wire.requests) == 4
            finally:
                child_release.set()
                b_release.set()
                if not running_b.done():
                    running_b.cancel()
                await asyncio.gather(running_b, return_exceptions=True)
        wire.assert_clean()
