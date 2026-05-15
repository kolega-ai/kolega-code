import uuid


def new_tool_execution_id() -> str:
    """Create an app-level identifier for one tool execution."""
    return f"tool_exec_{uuid.uuid4().hex}"


class ToolExecutionIdRegistry:
    """Response-scoped mapping from provider tool call IDs to app execution IDs."""

    def __init__(self) -> None:
        self._by_provider_tool_call_id: dict[str, str] = {}

    def get_or_create(self, provider_tool_call_id: str) -> str:
        if provider_tool_call_id not in self._by_provider_tool_call_id:
            self._by_provider_tool_call_id[provider_tool_call_id] = new_tool_execution_id()
        return self._by_provider_tool_call_id[provider_tool_call_id]
