"""Loopback adapter: a terminal window as a chat client.

Reads lines from stdin as inbound messages and prints outbound operations to
stdout. It exists for two reasons: developers can drive the full gateway
pipeline without a Telegram account, and tests can exercise streaming and
edits through the real adapter contract with an injected ``StringIO`` sink.
"""

from __future__ import annotations

import asyncio
import sys
from typing import IO, Any, Optional

from kolega_code.gateway.adapters.base import (
    STREAMING_EDIT_IN_PLACE,
    AdapterCapabilities,
    GatewayAdapter,
    InboundMessage,
)

DEFAULT_CHAT_ID = "console"
DEFAULT_SENDER_ID = "owner"


class EchoAdapter(GatewayAdapter):
    """Stdin/stdout transport implementing the gateway adapter contract."""

    name = "echo"
    capabilities = AdapterCapabilities(
        supports_edits=True,
        supports_delete=True,
        streaming_mode=STREAMING_EDIT_IN_PLACE,
        text_chunk_limit=4096,
    )

    def __init__(
        self,
        *,
        stdin: Optional[IO[str]] = None,
        stdout: Optional[IO[str]] = None,
        chat_id: str = DEFAULT_CHAT_ID,
        sender_id: str = DEFAULT_SENDER_ID,
    ) -> None:
        super().__init__()
        self._stdin = stdin if stdin is not None else sys.stdin
        self._stdout = stdout if stdout is not None else sys.stdout
        self._chat_id = chat_id
        self._sender_id = sender_id
        self._seq = 0
        self._running = False
        self._reader_task: Optional[asyncio.Task[None]] = None

    # -- Lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self._reader_task = asyncio.create_task(self._read_loop(), name="echo-adapter-reader")

    async def stop(self) -> None:
        self._running = False
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

    def health(self) -> dict[str, Any]:
        return {"state": "running" if self._running else "stopped", "chat_id": self._chat_id}

    async def _read_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            # A blocking readline off the event loop; stdin in a dev tool is
            # fine to occupy one executor thread per line.
            line = await loop.run_in_executor(None, self._stdin.readline)
            if line == "":  # EOF
                break
            text = line.rstrip("\n")
            if not text:
                continue
            self.publish(text)

    def publish(self, text: str) -> None:
        """Turn one line of console text into an inbound message (tests use
        this to drive the adapter without a real stdin)."""
        self._seq += 1
        self.inbound.put_nowait(
            InboundMessage(
                channel=self.name,
                chat_id=self._chat_id,
                sender_id=self._sender_id,
                message_id=f"echo-{self._seq}",
                text=text,
            )
        )

    # -- Outbound ----------------------------------------------------------

    def _emit(self, kind: str, message_id: str, text: str) -> str:
        self._seq += 1
        new_id = message_id or f"echo-{self._seq}"
        self._stdout.write(f"[gateway:{kind}] {new_id} {text}\n")
        self._stdout.flush()
        return new_id

    async def send_text(self, chat_id: str, text: str, *, reply_to_message_id: Optional[str] = None) -> str:
        return self._emit("reply", "", text)

    async def edit_text(self, chat_id: str, message_id: str, text: str) -> None:
        self._emit("edit", message_id, text)

    async def delete_message(self, chat_id: str, message_id: str) -> None:
        self._emit("delete", message_id, "")
