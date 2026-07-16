"""Model-safe façade over private project memory."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from kolega_code.memory import (
    MISSING_REVISION,
    MemoryAccessError,
    MemoryEntry,
    MemoryEntrySummary,
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

    async def list_memory(self, query: str | None = None) -> str:
        """List private project-memory entries through the active backend."""
        return await self._invoke_named("list_memory", query=query)

    async def write_memory(
        self,
        memory_content: str,
        path: str = "MEMORY.md",
        mode: str = "append",
        expected_sha256: str | None = None,
    ) -> str:
        """Append to or replace private project memory using its current revision."""
        return await self._invoke_named(
            "write_memory",
            memory_content=memory_content,
            path=path,
            mode=mode,
            expected_sha256=expected_sha256,
        )

    async def delete_memory(self, path: str, expected_sha256: str) -> str:
        """Delete a private project-memory entry after reading its current revision."""
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
            if not result.present:
                return (
                    f"Memory `{result.reference}` is missing (revision: {result.revision}, bytes: {result.byte_count})."
                )
            return (
                f"Memory `{result.reference}` (sha256: {result.revision}, "
                f"bytes: {result.byte_count}):\n\n{result.content or ''}"
            )

        if isinstance(result, list) and all(isinstance(item, MemoryEntrySummary) for item in result):
            query = inputs.get("query")
            match = f" matching '{query}'" if query is not None else ""
            if not result:
                return f"No memory entries found{match}."
            lines = [f"{len(result)} memory entries{match}:"]
            for item in result:
                modified = (
                    datetime.fromtimestamp(item.modified_ns / 1_000_000_000, tz=UTC).strftime("%Y-%m-%d")
                    if item.modified_ns is not None
                    else "unknown"
                )
                title = f" — {item.display_name}" if item.display_name and item.display_name != item.reference else ""
                lines.append(f"- {item.reference}{title} ({item.byte_count:,} bytes, modified {modified})")
            return "\n".join(lines)

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
            for warning in result.warnings:
                output += f"\nWarning: {warning}"
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
