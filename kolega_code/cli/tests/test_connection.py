import pytest

from kolega_code.agent.models.public import AgentEvent
from kolega_code.cli.connection import CliConnectionManager


@pytest.mark.asyncio
async def test_cli_connection_manager_queues_broadcast_events() -> None:
    manager = CliConnectionManager()
    event = AgentEvent(event_type="log_message", sender="test", content={"text": "hello"})

    await manager.broadcast_event(event, "workspace", "thread")

    assert await manager.next_event() == event


@pytest.mark.asyncio
async def test_cli_connection_manager_counts_connections() -> None:
    manager = CliConnectionManager()

    await manager.connect(object(), "workspace", "thread", "chat")
    await manager.connect(object(), "workspace", "thread", "chat")
    await manager.connect(object(), "workspace", "thread", "logs")
    manager.disconnect(object(), "workspace", "thread", "chat")

    assert manager.get_connection_count("workspace", "thread") == {"chat": 1, "logs": 1}
