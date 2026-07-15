"""Model-safe façade over private project memory."""

from __future__ import annotations

import asyncio
from typing import Any

from kolega_code.memory import (
    MISSING_REVISION,
    MemoryAccessError,
    MemoryEntry,
    MemorySafetyError,
    MemoryToolBinding,
    MemoryUnavailableError,
    MemoryWriteResult,
)
from kolega_code.tools import ToolError


class MemoryTool:
    def __init__(self, manager: Any, caller: Any) -> None:
        self.manager = manager
        self.caller = caller

    def bindings(self) -> tuple[MemoryToolBinding, ...]:
        return self.manager.tool_bindings() if self.manager is not None else ()

    async def read_memory(self, path: str = "MEMORY.md") -> str:
        """Read private project memory through the active backend."""
        return await self._invoke_named("read_memory", path=path)

    async def write_memory(
        self,
        memory_content: str,
        path: str = "MEMORY.md",
        mode: str = "append",
        expected_sha256: str | None = None,
    ) -> str:
        """Append or compare-and-swap replace private project memory."""
        return await self._invoke_named(
            "write_memory",
            memory_content=memory_content,
            path=path,
            mode=mode,
            expected_sha256=expected_sha256,
        )

    async def delete_memory(self, path: str, expected_sha256: str) -> str:
        """Compare-and-swap delete a private project-memory entry."""
        return await self._invoke_named(
            "delete_memory",
            path=path,
            expected_sha256=expected_sha256,
        )

    async def _invoke_named(self, name: str, **inputs: Any) -> str:
        binding = next((item for item in self.bindings() if item.name == name), None)
        if binding is None:
            raise ToolError(f"Project memory tool `{name}` is disabled or unavailable.")
        return await self.invoke(binding, **inputs)

    async def invoke(self, binding: MemoryToolBinding, **inputs: Any) -> str:
        try:
            result = await asyncio.to_thread(binding.handler, **inputs)
        except (MemoryAccessError, MemorySafetyError, MemoryUnavailableError) as exc:
            raise ToolError(f"Project memory error: {exc}") from exc
        except Exception as exc:
            raise ToolError(
                "Project memory operation failed without exposing private storage details. "
                "Use `/memory status` to inspect the local diagnostic."
            ) from exc

        if isinstance(result, MemoryEntry):
            if result.withheld:
                raise ToolError("Memory content was withheld because it contains a probable secret.")
            if not result.present:
                return (
                    f"Memory `{result.reference}` is missing (revision: {result.revision}, bytes: {result.byte_count})."
                )
            return (
                f"Memory `{result.reference}` (sha256: {result.revision}, "
                f"bytes: {result.byte_count}):\n\n{result.content or ''}"
            )

        if isinstance(result, MemoryWriteResult):
            if not result.ok:
                detail = result.error or "mutation failed"
                if result.current_revision:
                    detail += f"; current revision: {result.current_revision}"
                raise ToolError(f"Project memory mutation failed for `{result.reference}`: {detail}.")
            action = "updated" if result.revision != MISSING_REVISION else "deleted"
            output = (
                f"Project memory {action}: `{result.reference}` "
                f"(revision: {result.revision}, bytes: {result.byte_count or 0})."
            )
        else:
            output = str(result)

        if binding.mutating:
            initialize = getattr(self.caller, "_initialize_system_prompt", None)
            if callable(initialize):
                try:
                    initialize()
                except Exception:
                    output += (
                        "\n\nWarning: the memory mutation was committed, but the active prompt "
                        "could not be refreshed. Memory will be reloaded before the next top-level turn."
                    )
        return output
