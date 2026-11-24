#!/usr/bin/env python3
"""Measure ``FileSessionEventStore.head`` on session-heavy state directories.

The web session list summarizes every session in the store, and a long session
journal can contain hundreds of thousands of records. ``head`` used to parse and
Pydantic-validate each one; it now reads only the metadata a summary needs and
reuses the result while the file is unchanged. Run this before and after a
change to the head path (the numbers are medians over fresh journals):

    uv run python scripts/bench_session_head.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import tempfile
import time
from pathlib import Path

from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.session_journal import SessionJournal
from kolega_code.events import AgentEvent

# Per turn: three provider records interleaved with the presentation events, the
# mix a real journal has. A UI delta dominates the presentation record count.
PROVIDER_RECORDS_PER_TURN = 3
UI_DELTAS_PER_TURN = 20
LISTED_SESSIONS = 20
LISTED_TURNS = 230


def _record(seq: int, session_id: str, record_type: str, event: dict) -> str:
    return (
        json.dumps(
            {
                "version": 2,
                "id": f"id-{seq}",
                "session_id": session_id,
                "seq": seq,
                "epoch_id": "epoch",
                "turn_id": "turn",
                "timestamp": "2026-11-24T07:00:00+00:00",
                "actor": "agent",
                "type": record_type,
                "payload": {"event": event},
                "artifacts": [],
            },
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    )


def _event(event_type: str, *, seq: int, elapsed_ms: int, timestamp: str) -> dict:
    return AgentEvent(
        sender="agent",
        event_type=event_type,
        content={"text": "x" * 40, "complete": event_type == "turn_ended"},
        seq=seq,
        elapsed_ms=elapsed_ms,
        timestamp=timestamp,
        session_id="bench",
    ).model_dump(mode="json")


def build_journal(root: Path, session_id: str, turns: int) -> Path:
    session_dir = root / session_id
    session_dir.mkdir(parents=True)
    path = session_dir / "events.jsonl"
    seq = 0
    with path.open("w", encoding="utf-8") as handle:
        for turn in range(turns):
            for _ in range(PROVIDER_RECORDS_PER_TURN):
                seq += 1
                handle.write(_record(seq, session_id, "llm.message", {"message": {}}))
            seq += 1
            handle.write(
                _record(
                    seq,
                    session_id,
                    "ui.turn_started",
                    _event("turn_started", seq=seq, elapsed_ms=turn * 1000, timestamp="2026-11-24T07:00:00+00:00"),
                )
            )
            for index in range(UI_DELTAS_PER_TURN):
                seq += 1
                handle.write(
                    _record(
                        seq,
                        session_id,
                        "ui.assistant_delta",
                        _event(
                            "assistant_delta",
                            seq=seq,
                            elapsed_ms=turn * 1000 + index,
                            timestamp="2026-11-24T07:00:00+00:00",
                        ),
                    )
                )
            seq += 1
            handle.write(
                _record(
                    seq,
                    session_id,
                    "ui.turn_ended",
                    _event("turn_ended", seq=seq, elapsed_ms=turn * 1000 + 500, timestamp="2026-11-24T07:00:01+00:00"),
                )
            )
    return path


async def _head_ms(session_dir: Path, session_id: str) -> float:
    store = FileSessionEventStore(SessionJournal(session_id, session_dir))
    start = time.perf_counter()
    await store.head(session_id)
    return (time.perf_counter() - start) * 1000


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        print("cold head of one session (median of five fresh journals)")
        print(f"{'turns':>7} {'ui events':>10} {'MiB':>7} {'head ms':>10}")
        for turns in (200, 500, 1000):
            samples = []
            sizes = []
            for attempt in range(5):
                session_id = f"single-{turns}-{attempt}"
                path = build_journal(root, session_id, turns)
                sizes.append(path.stat().st_size)
                samples.append(await _head_ms(path.parent, session_id))
            print(
                f"{turns:>7} {turns * (UI_DELTAS_PER_TURN + 2):>10} "
                f"{statistics.median(sizes) / 1048576:>7.2f} {statistics.median(samples):>10.2f}"
            )

        print(f"\nlisting {LISTED_SESSIONS} sessions (~{LISTED_TURNS * (UI_DELTAS_PER_TURN + 2):,} ui events each)")
        listing_root = root / "listing"
        listing_root.mkdir()
        ids = []
        for index in range(LISTED_SESSIONS):
            session_id = f"listed-{index}"
            build_journal(listing_root, session_id, LISTED_TURNS)
            ids.append(session_id)
        for label in ("cold", "warm"):
            start = time.perf_counter()
            for session_id in ids:
                await _head_ms(listing_root / session_id, session_id)
            print(f"  {label:>4} {(time.perf_counter() - start) * 1000:9.2f} ms")

        print("\ndone.")


if __name__ == "__main__":
    asyncio.run(main())
