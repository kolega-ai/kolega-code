from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"


def save(directory: Path, name: str, value: Any) -> None:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


async def relay(directory: Path) -> None:
    config = json.loads((directory / "agent-env.json").read_text())
    child = await asyncio.create_subprocess_exec(
        str(PYTHON),
        "-m",
        "kolega_code.cli.main",
        "acp",
        env=config,
        cwd=directory / "project",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdin = asyncio.StreamReader()
    loop = asyncio.get_running_loop()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(stdin), sys.stdin.buffer)

    async def forward(reader: asyncio.StreamReader, direction: str) -> None:
        while line := await reader.readline():
            with (directory / "traffic.jsonl").open("a") as log:
                try:
                    data = json.loads(line)
                except ValueError:
                    data = line.decode(errors="replace").rstrip()
                log.write(json.dumps({"direction": direction, "data": data}) + "\n")
            if direction == "client":
                assert child.stdin is not None
                child.stdin.write(line)
                await child.stdin.drain()
            elif direction == "agent":
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()

    assert child.stdout is not None and child.stderr is not None
    tasks = [
        asyncio.create_task(forward(stdin, "client")),
        asyncio.create_task(forward(child.stdout, "agent")),
        asyncio.create_task(forward(child.stderr, "stderr")),
    ]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if child.returncode is None:
            child.terminate()
        try:
            await asyncio.wait_for(child.wait(), 10)
        except TimeoutError:
            child.kill()
            await child.wait()


async def serve(directory: Path) -> None:
    sys.path.insert(0, str(ROOT / "tests/acp"))
    fixtures = importlib.import_module("test_accounting_subprocess")
    directory.mkdir(parents=True, exist_ok=False)
    async with fixtures.NativeProviderWire(directory) as wire:
        wire.add(
            input_tokens=100,
            output_tokens=10,
            cost=None,
            text="EDITOR_SMOKE_FIRST: catalogue-backed accounting fixture completed.",
        )
        wire.add(
            input_tokens=140,
            output_tokens=20,
            cost=None,
            text="EDITOR_SMOKE_SECOND: cumulative accounting fixture completed.",
        )
        save(directory, "agent-env.json", fixtures.environment(directory, wire, "deepseek"))
        command = {
            "command": str(PYTHON),
            "args": [str(Path(__file__).resolve()), "--directory", str(directory), "--agent"],
            "env": {},
        }
        save(
            directory,
            "vscode-profile/User/settings.json",
            {
                "acp.agents": {"Kolega accounting smoke": command},
                "acp.defaultWorkingDirectory": str(directory / "project"),
                "acp.logTraffic": True,
                "telemetry.telemetryLevel": "off",
                "update.mode": "none",
                "extensions.autoCheckUpdates": False,
                "extensions.autoUpdate": False,
                "window.restoreWindows": "none",
                "chat.disableAIFeatures": True,
            },
        )
        save(
            directory,
            "zed-data/config/settings.json",
            {
                "telemetry": {"diagnostics": False, "metrics": False},
                "auto_update": False,
                "agent_servers": {"Kolega accounting smoke": {"type": "custom", **command}},
            },
        )
        print(json.dumps({"ready": True, "directory": str(directory)}), flush=True)
        try:
            while True:
                save(
                    directory,
                    "provider-summary.json",
                    {
                        "requests": [
                            {"host": item["host"], "path": item["path"], "model": item["body"]["model"]}
                            for item in wire.requests
                        ],
                        "errors": wire.errors,
                        "remainingReplies": len(wire.replies),
                    },
                )
                await asyncio.sleep(0.5)
        finally:
            print(json.dumps({"requests": len(wire.requests), "errors": wire.errors}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve two offline ACP provider replies for an isolated real editor.")
    parser.add_argument("--directory", type=Path, required=True, help="New scratch directory; never a user profile")
    parser.add_argument("--agent", action="store_true", help="Editor-only stdio relay to the real CLI")
    args = parser.parse_args()
    try:
        asyncio.run(relay(args.directory.resolve()) if args.agent else serve(args.directory.resolve()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
