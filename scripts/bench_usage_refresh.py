#!/usr/bin/env python3
"""Measure the session-usage publisher's refresh cost on long journals.

``SessionUsageSink`` publishes a usage snapshot after each accounting event, and
``UsagePresentation`` derives it from the whole event log. The log is append-only,
so a read only needs to parse the bytes appended since the previous read. Run this
before and after changes to the read path:

    uv run python scripts/bench_usage_refresh.py
    uv run python scripts/bench_usage_refresh.py --requests 3000 --rounds 30

All content is synthetic; the journal lives in a temporary directory that is
removed afterwards.
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

from kolega_code.cli.session_journal import SessionJournal
from kolega_code.cli.session_usage import derive_session_usage

RELEVANT_EVENTS = frozenset(
    {
        "llm.run_started",
        "llm.request_started",
        "llm.request_settled",
        "llm.usage_observed",
        "llm.run_finished",
        "llm.message",
        "llm.request_failed",
        "assistant.message",
        "turn.started",
        "context.epoch_started",
    }
)


def build_journal(root: Path, requests: int) -> SessionJournal:
    """A long session: tool results and messages around one accounted request per turn."""
    root.mkdir(parents=True, exist_ok=True)
    journal = SessionJournal("bench-session", root)
    epoch = journal.start_epoch("start")
    for index in range(requests):
        turn_id = f"t{index}"
        journal.append("turn.started", actor="user", payload={"text": "do work " * 20}, turn_id=turn_id, epoch_id=epoch)
        for sub in range(6):
            journal.append(
                "tool.result",
                actor="tool",
                payload={
                    "tool_name": "read_file",
                    "content": "line of file content\n" * (40 + sub * 5),
                    "tool_call_id": f"c{index}-{sub}",
                },
                turn_id=turn_id,
                epoch_id=epoch,
            )
        journal.append(
            "llm.run_started",
            actor="system",
            payload={"run_id": f"run{index}", "sink_id": "sink", "mode": "tui", "accounting_version": 2},
            turn_id=turn_id,
            epoch_id=epoch,
        )
        journal.append(
            "llm.request_started",
            actor="system",
            payload={
                "request_id": f"r{index}",
                "run_id": f"run{index}",
                "sink_id": "sink",
                "provider": "anthropic",
                "model": "model-x",
                "revision": 0,
                "usage": None,
                "origin": {"kind": "turn"},
            },
            turn_id=turn_id,
            epoch_id=epoch,
        )
        journal.append(
            "llm.message",
            actor="assistant",
            payload={
                "request_id": f"r{index}",
                "run_id": f"run{index}",
                "sink_id": "sink",
                "provider": "anthropic",
                "model": "model-x",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "assistant answer " * 60}],
                    "usage": {"input_tokens": 1000, "output_tokens": 200},
                },
            },
            turn_id=turn_id,
            epoch_id=epoch,
        )
        journal.append(
            "llm.request_settled",
            actor="system",
            payload={
                "request_id": f"r{index}",
                "run_id": f"run{index}",
                "sink_id": "sink",
                "provider": "anthropic",
                "model": "model-x",
                "revision": 1,
                "state": "responded",
                "usage": {"input_tokens": 1000, "output_tokens": 200},
                "observation": {
                    "usage": {"input_tokens": 1000, "output_tokens": 200},
                    "cost": {"source": "provider_reported", "amount": "0.01", "currency": "USD", "complete": True},
                },
            },
            turn_id=turn_id,
            epoch_id=epoch,
        )
        journal.append(
            "llm.run_finished",
            actor="system",
            payload={"run_id": f"run{index}", "sink_id": "sink", "counts": {"requests": 1}, "persist_failures": 0},
            turn_id=turn_id,
            epoch_id=epoch,
        )
    return journal


def refresh(journal: SessionJournal) -> tuple[int, dict]:
    """The derivation ``UsagePresentation._read_snapshot`` performs."""
    events = [event for event in journal.read_events(repair_tail=False) if event.event_type in RELEVANT_EVENTS]
    return max((event.seq for event in events), default=0), derive_session_usage(events)


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:8.2f} ms"


def measure(requests: int, rounds: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        journal = build_journal(Path(tmp) / "session", requests)
        size = journal.events_path.stat().st_size
        refresh(journal)  # cold read: parses the whole log
        cold = []
        for index in range(rounds):
            journal.append(
                "turn.started",
                actor="user",
                payload={"text": "more work " * 20},
                turn_id=f"extra-{index}",
            )
            started = time.perf_counter()
            refresh(journal)
            cold.append(time.perf_counter() - started)
        print(
            f"requests={requests:5d} journal={size / 1e6:7.2f}MB rounds={rounds:3d} "
            f"median={_ms(statistics.median(cold))} p95={_ms(sorted(cold)[max(0, int(len(cold) * 0.95) - 1)])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=500, help="accounted requests already in the journal")
    parser.add_argument("--rounds", type=int, default=20, help="append-then-refresh rounds to time")
    parser.add_argument("--cases", type=int, nargs="*", default=None, help="request counts to sweep")
    args = parser.parse_args()
    print(f"session-usage refresh benchmark — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    for requests in args.cases if args.cases is not None else (args.requests,):
        measure(requests, args.rounds)


if __name__ == "__main__":
    main()
