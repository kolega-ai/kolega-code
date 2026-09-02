"""SessionRegistry: LRU ordering, idle-TTL reaping, and eviction callbacks."""

import pytest

from kolega_code.gateway.adapters.base import ChatRef
from kolega_code.gateway.router import SessionRegistry


def ref(chat_id: str) -> ChatRef:
    return ChatRef("telegram", chat_id)


@pytest.mark.asyncio
async def test_put_get_and_touch_move_to_end() -> None:
    registry = SessionRegistry(max_sessions=3, idle_ttl_seconds=None)
    registry.put(ref("a"), "payload-a")
    registry.put(ref("b"), "payload-b")
    # Getting "a" makes it most-recently-active.
    entry_a = registry.get(ref("a"))
    assert entry_a is not None
    assert entry_a.payload == "payload-a"
    # Unknown chat returns None.
    assert registry.get(ref("c")) is None
    assert registry.active_count() == 2


@pytest.mark.asyncio
async def test_prune_evicts_least_recently_active_over_capacity() -> None:
    evicted: list[str] = []

    async def on_evict(payload: str) -> None:
        evicted.append(payload)

    registry = SessionRegistry(max_sessions=2, idle_ttl_seconds=None, on_evict=on_evict)
    registry.put(ref("a"), "payload-a")
    registry.put(ref("b"), "payload-b")
    registry.get(ref("a"))  # a is now most recent; b is LRU
    registry.put(ref("c"), "payload-c")

    result = await registry.prune()
    assert [entry.payload for entry in result] == ["payload-b"]
    assert evicted == ["payload-b"]
    assert registry.active_count() == 2
    assert registry.get(ref("b")) is None


@pytest.mark.asyncio
async def test_prune_evicts_idle_sessions() -> None:
    evicted: list[str] = []

    def on_evict(payload: str) -> None:
        evicted.append(payload)

    registry = SessionRegistry(max_sessions=10, idle_ttl_seconds=60.0, on_evict=on_evict)
    registry.put(ref("a"), "payload-a")
    registry.put(ref("b"), "payload-b")
    # Age "b" past the TTL without waiting.
    entry_b = registry.get(ref("b"))
    assert entry_b is not None
    entry_b.last_active -= 120.0

    result = await registry.prune()
    assert [entry.payload for entry in result] == ["payload-b"]
    assert evicted == ["payload-b"]
    entry_a = registry.get(ref("a"))
    assert entry_a is not None
    assert entry_a.payload == "payload-a"


@pytest.mark.asyncio
async def test_remove_does_not_invoke_evict_callback() -> None:
    evicted: list[str] = []

    async def on_evict(payload: str) -> None:
        evicted.append(payload)

    registry = SessionRegistry(max_sessions=2, idle_ttl_seconds=None, on_evict=on_evict)
    registry.put(ref("a"), "payload-a")
    removed = registry.remove(ref("a"))
    assert removed is not None
    assert removed.payload == "payload-a"
    assert registry.remove(ref("a")) is None
    await registry.prune()
    assert evicted == []


@pytest.mark.asyncio
async def test_clear_evicts_everything() -> None:
    evicted: list[str] = []

    def on_evict(payload: str) -> None:
        evicted.append(payload)

    registry = SessionRegistry(max_sessions=10, idle_ttl_seconds=None, on_evict=on_evict)
    registry.put(ref("a"), "payload-a")
    registry.put(ref("b"), "payload-b")
    result = await registry.clear()
    assert sorted(entry.payload for entry in result) == ["payload-a", "payload-b"]
    assert sorted(evicted) == ["payload-a", "payload-b"]
    assert registry.active_count() == 0
