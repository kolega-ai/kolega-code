"""Both first-party session event stores must satisfy the shipped contract.

Running the same checks against the in-memory and filesystem implementations is
what keeps the contract real: a guarantee only one backend honours is not a
contract, and host backends are validated against this identical suite.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

import pytest

from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.session_journal import SessionJournal
from kolega_code.session.inmemory import InMemorySessionEventStore
from kolega_code.session.store import SessionEventStore
from kolega_code.testing.store_conformance import CONFORMANCE_CHECKS, StoreFactory


def _in_memory_factory() -> StoreFactory:
    async def factory() -> tuple[SessionEventStore, str]:
        return InMemorySessionEventStore(), f"session-{uuid.uuid4()}"

    return factory


def _file_factory(root: Path) -> StoreFactory:
    async def factory() -> tuple[SessionEventStore, str]:
        session_id = f"session-{uuid.uuid4()}"
        session_dir = root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        journal = SessionJournal(session_id, session_dir, lock=threading.RLock())
        # Every journal record needs a context epoch; sessions created through
        # SessionStore get one at creation time.
        journal.start_epoch("test")
        return FileSessionEventStore(journal), session_id

    return factory


@pytest.mark.parametrize("check", CONFORMANCE_CHECKS, ids=lambda check: check.__name__)
@pytest.mark.asyncio
async def test_in_memory_store_conformance(check) -> None:
    await check(_in_memory_factory())


@pytest.mark.parametrize("check", CONFORMANCE_CHECKS, ids=lambda check: check.__name__)
@pytest.mark.asyncio
async def test_file_store_conformance(check, tmp_path: Path) -> None:
    await check(_file_factory(tmp_path))
