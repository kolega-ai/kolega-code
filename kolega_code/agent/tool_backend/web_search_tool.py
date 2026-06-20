from pathlib import Path
from typing import Union

from kolega_code.config import AgentConfig

from .search_backends import (
    DEFAULT_BACKEND,
    DEFAULT_RESULTS,
    SearchBackendError,
    SearchBackendNotConfigured,
    SearchBackendRateLimited,
    SearchResponse,
    build_search_backend,
    clamp_results,
)
from .streaming_tool import StreamingTool


class WebSearchTool(StreamingTool):
    """Tool for searching the web via a configurable backend (DuckDuckGo, Firecrawl,
    Tavily, or a self-hosted SearXNG instance)."""

    SEARCH_TIMEOUT_SECONDS = 15.0
    MAX_SNIPPET_CHARS = 500

    def __init__(
        self,
        project_path: Union[str, Path],
        workspace_id: str,
        thread_id: str,
        connection_manager,
        config: AgentConfig,
        caller,
        filesystem=None,
    ):
        super().__init__(project_path, workspace_id, thread_id, connection_manager, config, caller, filesystem)

    async def web_search(self, query: str, max_results: int = DEFAULT_RESULTS) -> str:
        """Search the web with the configured backend and return formatted results.

        See the ToolCollection ``web_search`` wrapper for the model-facing description.
        Never raises across the tool boundary: failures are returned as ``Error: ...``
        strings, mirroring WebFetchTool.
        """
        if not query or not query.strip():
            return "Error: Provide a non-empty search query."
        query = query.strip()
        count = clamp_results(max_results)
        backend_name = self.config.web_search_backend or DEFAULT_BACKEND

        tool_call_id = getattr(self.caller, "current_tool_call_id", None)
        if tool_call_id:
            await self.send_streaming_update(
                f"Searching the web for “{query}” via {backend_name}...",
                tool_call_id,
                "web_search",
                is_complete=False,
            )

        try:
            backend = build_search_backend(
                backend_name,
                api_key=self.config.web_search_api_key,
                base_url=self.config.web_search_base_url,
                timeout=self.SEARCH_TIMEOUT_SECONDS,
            )
            response = await backend.search(query, max_results=count)
        except SearchBackendNotConfigured as exc:
            return await self._error(f"{exc} Configure it in Settings > Web Search.", tool_call_id)
        except SearchBackendRateLimited as exc:
            return await self._error(
                f"{exc} The search backend is rate-limited; retry shortly or switch backends "
                "in Settings > Web Search.",
                tool_call_id,
            )
        except SearchBackendError as exc:
            return await self._error(f"Web search failed ({backend_name}): {exc}", tool_call_id)
        except Exception as exc:  # pragma: no cover - defensive
            return await self._error(f"web_search failed: {exc}", tool_call_id)

        text = self._format_response(response)
        if tool_call_id:
            summary = (
                f"Found {len(response.results)} result(s) for “{query}”."
                if response.results
                else f"No results for “{query}”."
            )
            await self.send_streaming_update(summary, tool_call_id, "web_search", is_complete=True)
        return text

    async def _error(self, message: str, tool_call_id) -> str:
        text = f"Error: {message}"
        if tool_call_id:
            await self.send_streaming_update(text, tool_call_id, "web_search", is_complete=True)
        return text

    def _format_response(self, response: SearchResponse) -> str:
        if not response.results:
            return f"No results found for: {response.query}"

        lines = [f"Web search results for “{response.query}” (via {response.backend}):", ""]
        if response.answer:
            lines.append(f"**Answer:** {response.answer.strip()}")
            lines.append("")
        for index, result in enumerate(response.results, start=1):
            snippet = result.snippet.strip()
            if len(snippet) > self.MAX_SNIPPET_CHARS:
                snippet = snippet[: self.MAX_SNIPPET_CHARS].rstrip() + "…"
            lines.append(f"{index}. {result.title or result.url}")
            lines.append(f"   {result.url}")
            if snippet:
                lines.append(f"   {snippet}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"
