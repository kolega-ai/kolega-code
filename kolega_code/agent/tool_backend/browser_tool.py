from pathlib import Path
from typing import Union

from ..config import AgentConfig
from ..models.public import AgentEvent
from ..services.browser import PlaywrightBrowserManager
from .base_tool import BaseTool


class BrowserTool(BaseTool):
    def __init__(
        self,
        project_path: Union[str, Path],
        workspace_id: str,
        thread_id: str,
        connection_manager,
        config: AgentConfig,
        caller,
        filesystem=None,
        browser_manager=None,
    ):
        super().__init__(
            project_path,
            workspace_id,
            thread_id,
            connection_manager,
            config,
            caller,
            filesystem,
            browser_manager=browser_manager,
        )

        # Use injected browser_manager if provided, otherwise create local one
        if self.browser_manager is None:
            self.browser_manager = PlaywrightBrowserManager()

    async def launch_browser(self, url: str) -> str:
        browser_result = await self.browser_manager.launch_browser(url)

        # Check if we got an error dict instead of a browser ID
        if isinstance(browser_result, dict) and "error" in browser_result:
            return f"Failed to launch browser. Error: {browser_result['error']}"

        if browser_result:
            browser_launched_event = AgentEvent(
                event_type="browser_launched", sender=self.caller.agent_name, content={"browser_id": browser_result}
            )
            await self.connection_manager.broadcast_event(browser_launched_event, self.workspace_id, self.thread_id)

            return f"Launched new browser with browser_id {browser_result}"
        else:
            return f"Failed to launch browser."

    async def list_browsers(self):
        results = await self.browser_manager.list_browsers()

        if not results:
            return "No browsers are currently running."

        markdown_output = "# Running Browsers\n\n"
        markdown_output += "| Browser ID | URL | Launched At |\n"
        markdown_output += "|------------|-----|------------|\n"

        for browser_id, browser_info in results.items():
            url = browser_info.get("url", "N/A")
            launched_at = browser_info.get("launched_at", "N/A")
            markdown_output += f"| {browser_id} | {url} | {launched_at} |\n"

        return markdown_output

    async def get_browser_console_logs(
        self,
        browser_id: str,
        max_logs: int = 50,
        log_types: list = None,
        minutes_back: int = None,
        max_chars: int = 8000,
    ) -> str:
        logs = await self.browser_manager.get_browser_console_logs(
            browser_id, max_logs=max_logs, log_types=log_types, minutes_back=minutes_back, max_chars=max_chars
        )

        if not logs["console_logs"]:
            return "## Console Logs\n\nNo console logs found."

        # Add metadata about filtering
        markdown_output = "## Console Logs\n\n"
        markdown_output += f"**Showing {logs['returned_count']} of {logs['total_logs_count']} total logs**\n\n"

        filters_applied = logs["filters_applied"]
        if filters_applied["log_types"]:
            markdown_output += f"**Filtered by types:** {', '.join(filters_applied['log_types'])}\n"
        if filters_applied["minutes_back"]:
            markdown_output += f"**Time window:** Last {filters_applied['minutes_back']} minutes\n"
        if filters_applied["max_chars"]:
            markdown_output += f"**Character limit:** {filters_applied['max_chars']}\n"
        markdown_output += f"**Max logs:** {filters_applied['max_logs']}\n\n"

        markdown_output += "| Type | Timestamp | Message | Location |\n"
        markdown_output += "|------|-----------|---------|----------|\n"

        for log in logs["console_logs"]:
            log_type = log.get("type", "unknown")
            timestamp = log.get("timestamp", "N/A")
            text = log.get("text", "").replace("|", "\\|")  # Escape pipe characters for markdown tables
            location = log.get("location", "N/A")
            if location and location != "N/A":
                location_str = f"{location.get('url', 'unknown')}:{location.get('lineNumber', '?')}:{location.get('columnNumber', '?')}"
            else:
                location_str = "N/A"
            markdown_output += f"| {log_type} | {timestamp} | {text} | {location_str} |\n"

        return markdown_output

    async def get_browser_interactive_elements(self, browser_id: str) -> str:
        result = await self.browser_manager.get_browser_interactive_elements(browser_id)

        # Format the interactive elements as markdown
        markdown_output = f"# Interactive Elements: {result['title']}\n\n"
        markdown_output += f"**Current URL:** {result['current_url']}\n\n"

        if result["interactive_elements"]:
            markdown_output += "## Elements\n\n"
            markdown_output += "| Type | Text | Selector | Attributes |\n"
            markdown_output += "|------|------|----------|------------|\n"

            for element in result["interactive_elements"]:
                element_type = element.get("element_type", "unknown")
                text = element.get("text", "").replace("|", "\\|")  # Escape pipe characters for markdown tables
                selector = element.get("selector", "").replace("|", "\\|")
                attributes = str(element.get("attributes", {})).replace("|", "\\|")

                markdown_output += f"| {element_type} | {text} | `{selector}` | {attributes} |\n"
        else:
            markdown_output += "No interactive elements found on the page."

        return markdown_output

    async def get_browser_content(
        self,
        browser_id: str,
        max_logs: int = 50,
        log_types: list = None,
        minutes_back: int = None,
        max_chars: int = 8000,
    ) -> str:
        content = await self.browser_manager.get_browser_content(
            browser_id, max_logs=max_logs, log_types=log_types, minutes_back=minutes_back, max_chars=max_chars
        )

        # Format the browser content as markdown
        markdown_output = f"# Browser Content: {content['title']}\n\n"
        markdown_output += f"**Current URL:** {content['current_url']}\n\n"

        # Add console logs section if there are any
        if content["console_logs"]:
            markdown_output += "## Console Logs\n\n"

            # Add metadata about console log filtering
            if "console_log_metadata" in content:
                metadata = content["console_log_metadata"]
                markdown_output += (
                    f"**Showing {metadata['returned_count']} of {metadata['total_logs_count']} total logs**\n\n"
                )

                filters_applied = metadata["filters_applied"]
                if filters_applied["log_types"]:
                    markdown_output += f"**Filtered by types:** {', '.join(filters_applied['log_types'])}\n"
                if filters_applied["minutes_back"]:
                    markdown_output += f"**Time window:** Last {filters_applied['minutes_back']} minutes\n"
                if filters_applied["max_chars"]:
                    markdown_output += f"**Character limit:** {filters_applied['max_chars']}\n"
                markdown_output += f"**Max logs:** {filters_applied['max_logs']}\n\n"

            markdown_output += "| Type | Timestamp | Message | Location |\n"
            markdown_output += "|------|-----------|---------|----------|\n"

            for log in content["console_logs"]:
                log_type = log.get("type", "unknown")
                timestamp = log.get("timestamp", "N/A")
                text = log.get("text", "").replace("|", "\\|")  # Escape pipe characters for markdown tables
                location = log.get("location", "N/A")
                if location and location != "N/A":
                    location_str = f"{location.get('url', 'unknown')}:{location.get('lineNumber', '?')}:{location.get('columnNumber', '?')}"
                else:
                    location_str = "N/A"
                markdown_output += f"| {log_type} | {timestamp} | {text} | {location_str} |\n"

        # Add HTML content in a code block
        markdown_output += "\n## Page HTML\n\n"
        markdown_output += "```html\n"
        markdown_output += content["html"]
        markdown_output += "\n```\n"

        return markdown_output

    async def take_browser_screenshot(self, browser_id: str) -> dict:
        return await self.browser_manager.take_browser_screenshot(browser_id)

    async def interact_with_browser(
        self, browser_id: str, action: str, selector: str, text: str, scroll_px: int
    ) -> str:
        result = await self.browser_manager.interact_with_browser(browser_id, action, selector, text, scroll_px)

        # Format the interaction result as markdown
        markdown_output = f"# Browser Interaction Result\n\n"
        markdown_output += f"**Status:** {result['status']}\n"
        markdown_output += f"**Current URL:** {result['current_url']}\n\n"
        markdown_output += f"**Action Performed:** {result['action']}\n"

        if result["selector"]:
            markdown_output += f"**Selector:** `{result['selector']}`\n"

        if result["text"]:
            markdown_output += f"**Text/URL:** {result['text']}\n"

        return markdown_output

    async def set_browser_select_value(self, browser_id: str, selector: str, value: str) -> str:
        result = await self.browser_manager.set_select_value(browser_id, selector, value)

        # Format the select value result as markdown
        markdown_output = f"# Select Value Update Result\n\n"
        markdown_output += f"**Status:** {result['status']}\n"
        markdown_output += f"**Current URL:** {result['current_url']}\n"
        markdown_output += f"**Selector:** `{result['selector']}`\n\n"

        if result["status"] == "success":
            markdown_output += "## ✅ Success\n\n"
            markdown_output += f"**Requested Value:** `{result['requested_value']}`\n"
            markdown_output += f"**Selected Value:** `{result['selected_value']}`\n"

            if result["requested_value"] == result["selected_value"]:
                markdown_output += "\n✓ The select box value was successfully updated."
            else:
                markdown_output += "\n⚠️ Warning: The selected value differs from the requested value."
        else:
            markdown_output += "## ❌ Error\n\n"
            markdown_output += f"**Error Message:** {result.get('error', 'Unknown error')}\n"

        return markdown_output

    async def close_browser(self, browser_id) -> str:
        await self.browser_manager.close_browser(browser_id)

        browser_closed_event = AgentEvent(
            event_type="browser_closed", sender=self.caller.agent_name, content={"browser_id": browser_id}
        )
        await self.connection_manager.broadcast_event(browser_closed_event, self.workspace_id, self.thread_id)

        return f"Browser with ID {browser_id} closed."

    async def cleanup(self) -> None:
        """
        Clean up all browser resources.
        This should be called when the tool is being destroyed.
        """
        await self.browser_manager.cleanup_all_browsers()
