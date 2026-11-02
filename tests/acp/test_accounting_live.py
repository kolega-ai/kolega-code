from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
from typing import Any
from zoneinfo import ZoneInfo

from acp import PROTOCOL_VERSION, spawn_agent_process
from acp.helpers import SessionUpdate
from acp.interfaces import Client
from acp.schema import ToolCallUpdate
from dotenv import dotenv_values
import httpx
import pytest

from kolega_code.cli.config import API_KEY_ENV
from kolega_code.config import ModelProvider
from kolega_code.llm.specs.accessors import get_model_pricing, get_model_specs


pytestmark = [pytest.mark.integration, pytest.mark.live_api, pytest.mark.e2e, pytest.mark.asyncio]

ROOT = Path(__file__).resolve().parents[2]
META = "dev.kolega/usage"
ROUTES = {
    "openrouter": ("openrouter.ai", "deepseek/deepseek-v4.1-flash-1017", "none"),
    "deepseek": ("api.deepseek.com", "deepseek-flash", "none"),
    "zai": ("api.z.ai", "glm-5.3-flash", None),
}
MAX_REQUESTS = 18
BUDGET = Decimal("1")


class LiveClient(Client):
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

    async def snapshot(self, total: int) -> dict[str, Any]:
        async with asyncio.timeout(60):
            while True:
                self.changed.clear()
                for update in reversed(self.updates):
                    extension = update.get("_meta", {}).get(META)
                    if extension and extension["session"]["totalTokens"] == total:
                        return extension
                await self.changed.wait()


class LiveWire:
    def __init__(self, directory: Path, *, prior_requests: int = 0, prior_spend: Decimal = Decimal(0)) -> None:
        self.directory = directory
        self.requests: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.blocked: list[str] = []
        self.tasks: set[asyncio.Task[None]] = set()
        self.failed_hosts: set[str] = set()
        self.fingerprints: set[str] = set()
        self.reserved = Decimal(0)
        self.spent = prior_spend
        self.prior_requests = prior_requests
        self.cancel_next = False
        self.cancel_evidence = asyncio.Event()
        self.cancel_release = asyncio.Event()
        self.server: asyncio.Server | None = None
        self.proxy = ""
        self.client: httpx.AsyncClient | None = None
        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    async def __aenter__(self) -> LiveWire:
        cert, key = self.directory / "cert.pem", self.directory / "key.pem"
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-days",
                "1",
                "-subj",
                "/CN=bounded ACP live accounting probe",
                "-addext",
                "subjectAltName=" + ",".join(f"DNS:{host}" for host, _, _ in ROUTES.values()),
                "-keyout",
                str(key),
                "-out",
                str(cert),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.context.load_cert_chain(cert, key)
        self.client = httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(120), follow_redirects=False)
        self.server = await asyncio.start_server(self._connected, "127.0.0.1", 0)
        self.proxy = f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.cancel_release.set()
        assert self.server is not None and self.client is not None
        self.server.close()
        await self.server.wait_closed()
        for task in tuple(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.client.aclose()

    def _connected(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.create_task(self._serve(reader, writer))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    @staticmethod
    def _json_frame(frame: bytes) -> dict[str, Any]:
        for line in frame.splitlines():
            if line.startswith(b"data: ") and line[6:] != b"[DONE]":
                try:
                    result = json.loads(line[6:])
                    return result if isinstance(result, dict) else {}
                except ValueError:
                    return {}
        return {}

    def _reserve(self, host: str, path: str, body: dict[str, Any], raw: bytes) -> dict[str, Any]:
        provider = next(provider for provider, route in ROUTES.items() if route[0] == host)
        model = ROUTES[provider][1]
        assert body.get("model") == model
        assert path.endswith(("/chat/completions", "/responses", "/messages"))
        fingerprint = hashlib.sha256(raw).hexdigest()
        assert fingerprint not in self.fingerprints, "No identical-payload SDK retry is allowed"
        assert host not in self.failed_hosts, "No additional inference after a provider failure"
        assert self.prior_requests + len(self.requests) < MAX_REQUESTS, "18 actual inference requests exhausted"
        self.fingerprints.add(fingerprint)
        specs = get_model_specs(provider, model)
        cap = body.get("max_output_tokens", body.get("max_completion_tokens", body.get("max_tokens")))
        assert type(cap) is int and 0 < cap <= specs["max_completion_tokens"], "Unbounded output refused"
        # UTF-8 byte count overbounds text input tokens; double list rates cover gateway routing variation.
        # Z.AI uses double the separate API list as a safety proxy, never as a credit/USD conversion.
        rates = {
            "openrouter": ("0.0000006", "0.0000024"),
            "deepseek": ("0.00000030", "0.00000120"),
            "zai": ("0.00000030", "0.00000100"),
        }
        reserve = Decimal(0)
        if provider in rates:
            input_rate, output_rate = map(Decimal, rates[provider])
            reserve = Decimal(len(raw)) * input_rate + Decimal(cap) * output_rate
        assert self.spent + self.reserved + reserve <= BUDGET, "Next maximum list charge would exceed USD 1"
        row: dict[str, Any] = {
            "index": len(self.requests) + 1,
            "provider": provider,
            "host": host,
            "path": path,
            "model": model,
            "stream": bool(body.get("stream")),
            "output_bound": cap,
            "request_bytes": len(raw),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "reserved_usd": str(reserve),
            "usage": [],
            "returned_models": [],
            "tools": [tool.get("function", tool).get("name") for tool in body.get("tools", [])],
        }
        self.reserved += reserve
        self.requests.append(row)
        return row

    @staticmethod
    def _observe(row: dict[str, Any], payload: dict[str, Any]) -> bool:
        response = payload.get("response", payload.get("message", payload))
        model = response.get("model")
        if isinstance(model, str) and model not in row["returned_models"]:
            row["returned_models"].append(model)
        usage = response.get("usage", payload.get("usage"))
        if isinstance(usage, dict) and usage:
            row["usage"].append(usage)
            return True
        return False

    def _finish(self, row: dict[str, Any]) -> None:
        reserve = Decimal(row["reserved_usd"])
        actual = reserve
        if row["usage"] and row["provider"] == "openrouter":
            usage = row["usage"][-1]
            if usage.get("cost") is not None:
                actual = max(
                    Decimal(str(usage["cost"])),
                    Decimal(str((usage.get("cost_details") or {}).get("upstream_inference_cost") or 0)),
                    Decimal(usage["prompt_tokens"]) * Decimal("0.0000006")
                    + Decimal(usage["completion_tokens"]) * Decimal("0.0000024"),
                )
        elif row["usage"] and row["provider"] == "deepseek":
            usage = row["usage"][-1]
            if "input_tokens" in usage and "output_tokens" in usage:
                actual = Decimal(usage["input_tokens"]) * Decimal("0.00000030") + Decimal(
                    usage["output_tokens"]
                ) * Decimal("0.00000120")
        elif row["usage"] and row["provider"] == "zai":
            usage = row["usage"][-1]
            if "input_tokens" in usage and "output_tokens" in usage:
                inclusive = sum(
                    usage.get(key, 0)
                    for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
                )
                actual = Decimal(inclusive) * Decimal("0.00000030") + Decimal(usage["output_tokens"]) * Decimal(
                    "0.00000100"
                )
        self.reserved -= reserve
        self.spent += actual
        row["budget_debit_usd"] = str(actual)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        row: dict[str, Any] | None = None
        try:
            async with asyncio.timeout(180):
                connect = await reader.readuntil(b"\r\n\r\n")
                method, target, _ = connect.split(b"\r\n", 1)[0].decode().split()
                if method == "GET" and target == "http://fixture.invalid/accounting.txt":
                    content = b"The accounting probe has exactly seven blue widgets."
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nConnection: close\r\nContent-Length: "
                        + str(len(content)).encode()
                        + b"\r\n\r\n"
                        + content
                    )
                    await writer.drain()
                    return
                host, port = target.rsplit(":", 1)
                assert method == "CONNECT" and port == "443" and host in {route[0] for route in ROUTES.values()}
                writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
                await writer.drain()
                await writer.start_tls(self.context)
                headers_raw = await reader.readuntil(b"\r\n\r\n")
                first, *lines = headers_raw.decode().split("\r\n")
                method, path, _ = first.split()
                headers = dict(line.split(": ", 1) for line in lines if ": " in line)
                lower = {name.lower(): value for name, value in headers.items()}
                raw = await reader.readexactly(int(lower["content-length"]))
                body = json.loads(raw)
                if path.endswith("/count_tokens"):
                    raise AssertionError("Native token-count preflight needs separate non-inference forwarding")
                row = self._reserve(host, path, body, raw)
                cancel = self.cancel_next
                self.cancel_next = False
                assert self.client is not None
                async with self.client.stream(
                    method,
                    f"https://{host}{path}",
                    content=raw,
                    headers={
                        name: value
                        for name, value in headers.items()
                        if name.lower() not in {"host", "connection", "content-length", "accept-encoding"}
                    },
                ) as response:
                    row["http_status"] = response.status_code
                    if response.status_code >= 400:
                        self.failed_hosts.add(host)
                    content_type = response.headers.get("content-type", "application/json")
                    writer.write(
                        f"HTTP/1.1 {response.status_code} {response.reason_phrase}\r\n"
                        f"Content-Type: {content_type}\r\nConnection: close\r\n\r\n".encode()
                    )
                    await writer.drain()
                    pending = b""
                    async for chunk in response.aiter_bytes():
                        pending += chunk
                        if "text/event-stream" not in content_type:
                            continue
                        while b"\n\n" in pending:
                            frame, pending = pending.split(b"\n\n", 1)
                            observed = self._observe(row, self._json_frame(frame))
                            writer.write(frame + b"\n\n")
                            await writer.drain()
                            if cancel and observed:
                                self.cancel_evidence.set()
                                await self.cancel_release.wait()
                    if pending:
                        if "text/event-stream" not in content_type:
                            try:
                                self._observe(row, json.loads(pending))
                            except ValueError:
                                pass
                        writer.write(pending)
                        await writer.drain()
        except AssertionError as exc:
            self.blocked.append(str(exc))
            writer.write(b"HTTP/1.1 400 Probe guard refused request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        except (ConnectionError, asyncio.IncompleteReadError, ssl.SSLError):
            pass
        except Exception as exc:
            self.errors.append(type(exc).__name__)
        finally:
            if row is not None:
                self._finish(row)
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, ssl.SSLError):
                pass


def environment(directory: Path, wire: LiveWire, provider: str, secrets: dict[str, str]) -> dict[str, str]:
    for name in ("state", "home", "project", "token-cache"):
        (directory / name).mkdir(exist_ok=True)
    for cached in Path(os.environ.get("TIKTOKEN_CACHE_DIR", "/tmp/data-gym-cache")).glob("*"):
        if cached.is_file():
            shutil.copyfile(cached, directory / "token-cache" / cached.name)
    _, model, effort = ROUTES[provider]
    settings = {
        "schema_version": 3,
        "active_provider": provider,
        "active_model": model,
        "active_thinking_effort": effort,
        "lsp_enabled": False,
        "usage_pricing": {},
        "skills_enabled": False,
        "agent_models": {},
        "model_slots": {},
    }
    (directory / "state" / "settings.json").write_text(json.dumps(settings))
    env = {name: value for name, value in os.environ.items() if name in ("PATH", "LANG", "SYSTEMROOT")}
    env.update(
        HOME=str(directory / "home"),
        TMPDIR=str(directory),
        XDG_CONFIG_HOME=str(directory / "home/.config"),
        XDG_CACHE_HOME=str(directory / "home/.cache"),
        KOLEGA_CODE_STATE_DIR=str(directory / "state"),
        PYTHONPATH=str(ROOT),
        PYTHONUNBUFFERED="1",
        KOLEGA_CODE_LSP="off",
        KOLEGA_CODE_SKILLS="off",
        HTTPS_PROXY=wire.proxy,
        HTTP_PROXY=wire.proxy,
        ALL_PROXY=wire.proxy,
        NO_PROXY="",
        SSL_CERT_FILE=str(wire.directory / "cert.pem"),
        TIKTOKEN_CACHE_DIR=str(directory / "token-cache"),
    )
    credential_name = API_KEY_ENV[ModelProvider(provider)]
    env[credential_name] = secrets[credential_name]
    return env


@asynccontextmanager
async def process(
    client: LiveClient, env: dict[str, str], directory: Path, diagnostics: list[str]
) -> AsyncIterator[Any]:
    async with asyncio.timeout(240):
        async with spawn_agent_process(
            client,
            sys.executable,
            "-m",
            "kolega_code.cli.main",
            "acp",
            env=env,
            cwd=str(directory / "project"),
            transport_kwargs={"shutdown_timeout": 30},
        ) as (connection, child):

            async def drain() -> None:
                assert child.stderr is not None
                while line := await child.stderr.readline():
                    text = line.decode(errors="replace")
                    if "Error" in text or "Exception" in text:
                        diagnostics.append(text.split(":", 1)[0][-160:])

            drainer = asyncio.create_task(drain())
            try:
                await connection.initialize(protocol_version=PROTOCOL_VERSION)
                yield connection
            finally:
                drainer.cancel()
                await asyncio.gather(drainer, return_exceptions=True)


def extension(response: Any) -> dict[str, Any]:
    payload = response.model_dump(by_alias=True, exclude_none=True)
    result = payload.get("_meta", {}).get(META)
    assert result and result["schemaVersion"] == 1, "Missing authoritative accounting metadata"
    return result


async def prompt(connection: Any, session_id: str, text: str) -> dict[str, Any]:
    return extension(await connection.prompt(session_id=session_id, prompt=[{"type": "text", "text": text}]))


def facts(directory: Path, session_id: str) -> list[dict[str, Any]]:
    paths = list((directory / "state").rglob("events.jsonl"))
    events = [json.loads(line) for path in paths if session_id in str(path) for line in path.read_text().splitlines()]
    settled = [event["payload"] for event in events if event["type"] == "llm.request_settled"]
    starts = [event["payload"] for event in events if event["type"] == "llm.request_started"]
    keys = {(fact["run_id"], fact["request_id"]) for fact in settled}
    assert len(keys) == len(settled), "Duplicate request settlement"
    assert keys <= {(fact["run_id"], fact["request_id"]) for fact in starts}
    for event in events:
        if event["type"] == "llm.usage_observed":
            refined = event["payload"]
            settled = [
                refined if (fact["run_id"], fact["request_id"]) == (refined["run_id"], refined["request_id"]) else fact
                for fact in settled
            ]
    return settled


def reconcile(snapshot: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    session = snapshot["session"]
    assert session["requests"]["requests"] == len(rows), "Journal / ACP request count disagreement"
    assert session["requests"]["open"] == 0
    assert session["persistence"]["healthy"]
    for public, normalized in (
        ("inputTokens", "input_tokens"),
        ("outputTokens", "output_tokens"),
        ("totalTokens", "total_tokens"),
    ):
        known = [row["usage"][normalized] for row in rows if row.get("usage")]
        if known:
            assert session["knownSubtotals"][public] == sum(known), public
    prompt_rows = [row for row in rows if row.get("prompt_id") == snapshot["prompt"]["promptId"]]
    assert snapshot["prompt"]["requests"]["requests"] == len(prompt_rows)
    amounts: dict[str, Decimal] = {}
    for row in rows:
        cost = row.get("cost", {})
        if cost.get("amount") is not None and cost.get("currency"):
            currency = cost["currency"]
            amounts[currency] = amounts.get(currency, Decimal(0)) + Decimal(cost["amount"])
    assert {row["currency"]: Decimal(row["amount"]) for row in session["cost"]["monetary"]} == amounts


def reconcile_wire(provider: str, wire: LiveWire, rows: list[dict[str, Any]]) -> None:
    requests = [row for row in wire.requests if row["provider"] == provider]
    assert len(requests) == len(rows), "Actual provider POSTs differ from persisted request facts"
    observed: list[tuple[int, int]] = []
    charges: list[Decimal] = []
    for request in requests:
        assert request["usage"], "Provider did not report usage"
        usage = request["usage"][-1]
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        if provider == "zai":
            input_tokens += usage.get("cache_read_input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
        observed.append((input_tokens, usage.get("output_tokens", usage.get("completion_tokens"))))
        if provider == "openrouter":
            assert usage.get("cost") is not None, "Gateway monetary field absent"
            charges.append(Decimal(str(usage["cost"])))
    assert sorted(observed) == sorted((row["usage"]["input_tokens"], row["usage"]["output_tokens"]) for row in rows)
    if provider == "openrouter":
        assert sorted(charges) == sorted(Decimal(row["cost"]["amount"]) for row in rows)


def deepseek_arithmetic(row: dict[str, Any]) -> str:
    card = row["pinned_pricing"]["pricing"]
    stamp = datetime.fromisoformat(row["pinned_pricing"]["context"]["requested_at"]).astimezone(
        ZoneInfo("Asia/Shanghai")
    )
    rates = card["rates"]
    holidays = {day for schedule in card["schedules"] for day in schedule["when"].get("holidays", [])}
    if stamp.weekday() < 5 and stamp.date().isoformat() not in holidays:
        for schedule in card["schedules"]:
            when = schedule["when"]
            if when["start_time"] <= stamp.strftime("%H:%M") < when["end_time"]:
                rates = schedule["rates"]
    usage = row["usage"]
    cached = usage["cache_read_input_tokens"]
    assert cached is not None, "Missing cache split prevents independent exact pricing"
    result = (
        Decimal(usage["input_tokens"] - cached) * Decimal(rates["input"])
        + Decimal(cached) * Decimal(rates["cache_read"])
        + Decimal(usage["output_tokens"]) * Decimal(rates["output"])
    ) / Decimal(1_000_000)
    assert row["cost"]["source"] == "estimated" and row["cost"]["pricing_source"] == "catalogue"
    assert Decimal(row["cost"]["amount"]) == result, "Catalogue arithmetic differs from request cost"
    return str(result)


@pytest.mark.skipif(os.environ.get("KOLEGA_RUN_LIVE_ACP_ACCOUNTING") != "1", reason="Explicit paid opt-in required")
async def test_bounded_live_acp_accounting(tmp_path: Path) -> None:
    selected = os.environ.get("KOLEGA_ACP_LIVE_ROUTES", ",".join(ROUTES)).split(",")
    assert selected and all(provider in ROUTES for provider in selected)
    assert len(selected) == len(set(selected)), "Do not repeat a route within a campaign"
    prior_requests, prior_spend = 0, Decimal(0)
    previous = os.environ.get("KOLEGA_ACP_LIVE_PREVIOUS_EVIDENCE")
    if previous:
        prior = json.loads(Path(previous).read_text())
        prior_requests = prior.get("campaign_inference_requests", prior["actual_inference_requests"])
        prior_spend = Decimal(prior["conservative_list_debit_usd"])
        assert type(prior_requests) is int and 0 <= prior_requests < MAX_REQUESTS
        assert prior_spend.is_finite() and 0 <= prior_spend < BUDGET
    dotenv_path = Path(os.environ.get("KOLEGA_ACP_LIVE_DOTENV", str(ROOT / ".env")))
    values = {**dotenv_values(dotenv_path), **os.environ}
    secrets = {
        API_KEY_ENV[ModelProvider(provider)]: str(values[API_KEY_ENV[ModelProvider(provider)]])
        for provider in selected
        if values.get(API_KEY_ENV[ModelProvider(provider)])
    }
    assert len(secrets) == len(selected), "Required provider credentials missing; live paths are not verified"
    report: dict[str, Any] = {
        "date": datetime.now(timezone.utc).isoformat(),
        "maximum_requests": MAX_REQUESTS,
        "budget_usd": str(BUDGET),
        "prior_inference_requests": prior_requests,
        "prior_conservative_list_debit_usd": str(prior_spend),
        "sdk": {name: importlib.metadata.version(name) for name in ("agent-client-protocol", "openai", "anthropic")},
        "routes": {},
        "failures": [],
        "limitations": [
            "Published list-value estimates are not invoices or account debits.",
            "Coding Plan credits have no universal USD conversion.",
            "Cancellation gates a real OpenRouter usage frame before forwarding stream EOF.",
            "SDK retries cannot reach a provider twice with the identical payload.",
        ],
    }
    diagnostics: list[str] = []
    async with LiveWire(tmp_path, prior_requests=prior_requests, prior_spend=prior_spend) as wire:
        try:
            for provider in selected:
                _, model, _ = ROUTES[provider]
                directory = tmp_path / provider
                directory.mkdir()
                env = environment(directory, wire, provider, secrets)
                route: dict[str, Any] = {"model": model, "pricing": get_model_pricing(provider, model), "snapshots": []}
                report["routes"][provider] = route
                try:
                    client = LiveClient()  # pyright: ignore[reportAbstractUsage]
                    async with process(client, env, directory, diagnostics) as connection:
                        session = await connection.new_session(cwd=str(directory / "project"), mcp_servers=[])
                        session_id = session.session_id
                        first = await prompt(connection, session_id, "Reply with only READY. Do not use tools.")
                        route["snapshots"].append(first)
                        reconcile(first, facts(directory, session_id))
                        if provider == "openrouter":
                            second = await prompt(
                                connection,
                                session_id,
                                "Accounting E2E: invoke dispatch_agent exactly once (foreground, general agent) "
                                "with task 'Reply only BLUE. No tools.' Also invoke web_fetch exactly once for "
                                "http://fixture.invalid/accounting.txt with instruction 'How many blue widgets? "
                                "Answer in one short sentence.' Then reply only DONE. No other tools, no retries, "
                                "no model overrides. This is an explicit request for both tools.",
                            )
                        else:
                            second = await prompt(connection, session_id, "Reply with only SECOND. Do not use tools.")
                        route["snapshots"].append(second)
                        rows = facts(directory, session_id)
                        reconcile(second, rows)
                        assert first["prompt"]["promptId"] != second["prompt"]["promptId"]
                        assert second["session"]["totalTokens"] > first["session"]["totalTokens"]
                        route["facts"] = rows
                        if provider == "openrouter":
                            origins = [row.get("origin", {}) for row in rows]
                            if not any(
                                origin.get("kind") == "sub_agent"
                                or (origin.get("depth", 0) > 0 and origin.get("parent_tool_call_id"))
                                for origin in origins
                            ):
                                report["failures"].append("openrouter: requested delegate inference not exercised")
                            if not any(
                                origin.get("kind") == "helper" and origin.get("helper") == "web_fetch"
                                for origin in origins
                            ):
                                report["failures"].append("openrouter: requested helper inference not exercised")
                            assert all(row["cost"]["source"] == "provider_reported" for row in rows)
                            wire.cancel_next = True
                            task = asyncio.create_task(
                                prompt(connection, session_id, "Reply only CANCEL_EVIDENCE. No tools.")
                            )
                            try:
                                async with asyncio.timeout(90):
                                    await wire.cancel_evidence.wait()
                                await connection.cancel(session_id=session_id)
                                cancelled = await task
                            finally:
                                wire.cancel_release.set()
                                if not task.done():
                                    task.cancel()
                                    await asyncio.gather(task, return_exceptions=True)
                            route["snapshots"].append(cancelled)
                            reconcile(cancelled, facts(directory, session_id))
                            route["cancelled_after_provider_usage"] = True
                        elif provider == "deepseek":
                            route["independent_amounts_usd"] = [deepseek_arithmetic(row) for row in rows]
                        else:
                            assert not second["session"]["cost"]["monetary"], "Coding credits are not USD"
                            assert all(row["cost"].get("currency") is None for row in rows)
                            route["monetary_gap_verified"] = True
                        before = route["snapshots"][-1]
                    if provider == "openrouter":
                        resumed_client = LiveClient()  # pyright: ignore[reportAbstractUsage]
                        async with process(resumed_client, env, directory, diagnostics) as connection:
                            await connection.load_session(
                                cwd=str(directory / "project"), session_id=session_id, mcp_servers=[]
                            )
                            restored = await resumed_client.snapshot(before["session"]["totalTokens"])
                            assert restored["session"]["runId"] != before["session"]["runId"]
                            assert restored["session"]["cost"] == before["session"]["cost"]
                            continued = await prompt(connection, session_id, "Reply only RESUMED. No tools.")
                            route["snapshots"].append(continued)
                            reconcile(continued, facts(directory, session_id))
                            route["restart_load_continue_verified"] = True
                    route["facts"] = facts(directory, session_id)
                    reconcile_wire(provider, wire, route["facts"])
                    updates = client.updates
                    if provider == "openrouter":
                        updates = [*updates, *resumed_client.updates]
                    route["native_usage_updates"] = [
                        {
                            **{key: update[key] for key in ("sessionUpdate", "used", "size", "cost") if key in update},
                            "totalTokens": update["_meta"][META]["session"]["totalTokens"],
                            "runId": update["_meta"][META]["session"]["runId"],
                            "revision": update["_meta"][META]["session"]["revision"],
                        }
                        for update in updates
                        if update.get("sessionUpdate") == "usage_update" and META in update.get("_meta", {})
                    ]
                    if provider == "zai":
                        assert route["native_usage_updates"], "No native Coding Plan UsageUpdate observed"
                        assert all("cost" not in update for update in route["native_usage_updates"])
                    else:
                        expected = route["snapshots"][-1]["session"]
                        assert any(
                            update["totalTokens"] == expected["totalTokens"]
                            and "cost" in update
                            and Decimal(str(update["cost"]["amount"]))
                            == Decimal(expected["cost"]["monetary"][0]["amount"])
                            for update in route["native_usage_updates"]
                        ), "No matching native cumulative monetary UsageUpdate"
                    route["wire_journal_reconciliation_verified"] = True
                    route["completed"] = True
                except Exception as exc:
                    route["error_type"] = type(exc).__name__
                    report["failures"].append(f"{provider}: {type(exc).__name__}: {str(exc)[:200]}")
        finally:
            report.update(
                actual_inference_requests=len(wire.requests),
                campaign_inference_requests=prior_requests + len(wire.requests),
                wire=wire.requests,
                guard_refusals=wire.blocked,
                wire_errors=wire.errors,
                conservative_list_debit_usd=str(wire.spent),
                outstanding_maximum_reservation_usd=str(wire.reserved),
                diagnostics=diagnostics,
            )
            evidence = Path(os.environ.get("KOLEGA_ACP_LIVE_EVIDENCE", str(tmp_path / "accounting-live.json")))
            serialized = json.dumps(report, indent=2, default=str)
            for secret in secrets.values():
                assert secret not in serialized, "Secret must not enter saved evidence"
            evidence.write_text(serialized + "\n")
    assert not report["failures"], report["failures"]
    assert not wire.errors and not wire.blocked
    assert prior_requests + len(wire.requests) <= MAX_REQUESTS
    assert wire.spent + wire.reserved <= BUDGET
