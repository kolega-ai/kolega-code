import base64
import datetime
import json
import os
import urllib.parse
import uuid
from typing import List, Optional
import asyncio

from playwright.async_api import async_playwright

from .html import extract_interactive_elements_from_html
from .base import BrowserManager


class PlaywrightBrowserManager(BrowserManager):
    def __init__(self, use_browserstack: bool = False, browser_backend: str = "local"):
        self.browsers = {}
        self.viewport = {"width": 1280, "height": 720}
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        self.headless = False
        self.interaction_timeout = 5000
        self.max_console_logs_per_browser = 200  # Maximum logs to keep per browser (circular buffer)

        # Browser backend configuration
        self.browser_backend = browser_backend
        if use_browserstack:  # Legacy support
            self.browser_backend = "browserstack"

        # BrowserStack configuration
        if self.browser_backend == "browserstack":
            self.browserstack_username = os.environ.get("BROWSERSTACK_USERNAME")
            self.browserstack_access_key = os.environ.get("BROWSERSTACK_ACCESS_KEY")

            if not self.browserstack_username or not self.browserstack_access_key:
                raise ValueError(
                    "BrowserStack credentials not found. Please set BROWSERSTACK_USERNAME and "
                    "BROWSERSTACK_ACCESS_KEY environment variables."
                )

        # Browserless configuration
        elif self.browser_backend == "browserless":
            self.browserless_api_key = os.environ.get("BROWSERLESS_API_KEY")
            if not self.browserless_api_key:
                raise ValueError("Browserless API key not found. Please set BROWSERLESS_API_KEY environment variable.")

        # Keepalive configuration
        self.keepalive_interval = 30  # Send keepalive every 30 seconds
        self.keepalive_tasks = {}  # Store keepalive tasks per browser

    async def _keepalive_loop(self, browser_id: str):
        """Background task to keep the browser connection alive."""
        while browser_id in self.browsers:
            try:
                browser_info = self.browsers.get(browser_id)
                if not browser_info:
                    break

                page = browser_info.get("page")
                if page:
                    # Send a simple keepalive command
                    await page.evaluate("() => { /* keepalive */ }")

                await asyncio.sleep(self.keepalive_interval)
            except Exception as e:
                print(f"[Keepalive] Error for browser {browser_id}: {e}")
                # Sleep shorter on error but continue trying
                await asyncio.sleep(5)

    def _get_browserstack_cdp_url(self) -> str:
        """Generate the BrowserStack CDP URL with capabilities."""
        capability = {
            "browser": "chrome",
            "browser_version": "latest",
            "os": "Windows",
            "os_version": "11",
            "name": "Kolega Browser Agent",
            "build": "kolega-platform",
            "browserstack.username": self.browserstack_username,
            "browserstack.accessKey": self.browserstack_access_key,
            "browserstack.console": "verbose",
            "browserstack.networkLogs": "true",
        }

        # According to BrowserStack docs, we need to encode the JSON capabilities
        caps_json = json.dumps(capability)
        caps_string = urllib.parse.quote(caps_json)
        cdp_url = f"wss://cdp.browserstack.com/playwright?caps={caps_string}"

        return cdp_url

    def _get_browserless_cdp_url(self) -> str:
        """Generate the Browserless CDP URL."""
        # Browserless cloud CDP URL format for connectOverCDP
        # Format: wss://production-sfo.browserless.io?token=YOUR_TOKEN
        # Alternative regions available: production-lon, production-ams, etc.

        # Use timeout to control maximum session duration
        # Maximum allowed timeout is 60,000 (units unclear but this value works)
        # This provides either 60 seconds or 60,000ms depending on interpretation
        timeout = 1800000  # Maximum allowed timeout value

        return f"wss://production-sfo.browserless.io?token={self.browserless_api_key}&timeout={timeout}"

    async def launch_browser(self, url: str) -> str:
        try:
            playwright = await async_playwright().start()

            # Connect based on backend
            if self.browser_backend == "browserstack":
                cdp_url = self._get_browserstack_cdp_url()
                browser = await playwright.chromium.connect(ws_endpoint=cdp_url)
            elif self.browser_backend == "browserless":
                cdp_url = self._get_browserless_cdp_url()
                # Use connectOverCDP for browserless as recommended in their docs
                browser = await playwright.chromium.connect_over_cdp(cdp_url)
            else:  # local
                browser_factory = playwright.chromium
                browser = await browser_factory.launch(headless=self.headless)

            context_options = {"viewport": self.viewport, "user_agent": self.user_agent}
            context = await browser.new_context(**context_options)
            page = await context.new_page()

            console_logs = []
            network_requests = []

            # Register console log listener BEFORE any navigation
            def console_log_handler(msg):
                log_entry = {
                    "type": msg.type,
                    "text": msg.text,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "location": msg.location if hasattr(msg, "location") else None,
                }
                console_logs.append(log_entry)

                # Implement circular buffer to prevent unlimited growth
                if len(console_logs) > self.max_console_logs_per_browser:
                    console_logs.pop(0)  # Remove oldest log

            page.on("console", console_log_handler)

            # Also capture page errors and unhandled exceptions
            def page_error_handler(error):
                log_entry = {
                    "type": "error",
                    "text": f"Page Error: {str(error)}",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "location": None,
                }
                console_logs.append(log_entry)

                # Implement circular buffer to prevent unlimited growth
                if len(console_logs) > self.max_console_logs_per_browser:
                    console_logs.pop(0)  # Remove oldest log

            page.on("pageerror", page_error_handler)

            # Wait for the listener to be fully registered
            await page.evaluate("() => console.log('Console listener ready')")

            # Now navigate to the URL with better wait strategy
            await page.goto(url, wait_until="domcontentloaded")

            browser_id = str(uuid.uuid4())

            browser_info = {
                "type": "chromium",
                "url": url,
                "playwright": playwright,
                "browser": browser,
                "context": context,
                "page": page,
                "console_logs": console_logs,
                "network_requests": network_requests,
                "launched_at": datetime.datetime.now().isoformat(),
                "browserstack": self.browser_backend == "browserstack",
                "backend": self.browser_backend,
            }

            self.browsers[browser_id] = browser_info

            # Start keepalive task for this browser
            keepalive_task = asyncio.create_task(self._keepalive_loop(browser_id))
            self.keepalive_tasks[browser_id] = keepalive_task

            return browser_id
        except Exception as ex:
            error_msg = f"[Browser Launch Error] {type(ex).__name__}: {str(ex)}"
            print(error_msg)
            # Clean up resources in case of error
            if "playwright" in locals():
                await playwright.stop()

            # Return error details instead of None
            return {"error": error_msg}

    async def list_browsers(self) -> dict:
        result = {}
        for browser_id, browser in self.browsers.items():
            result[browser_id] = {
                "url": browser["url"],
                "launched_at": browser["launched_at"],
                "browserstack": browser.get("browserstack", False),
                "backend": browser.get("backend", "unknown"),
            }

        return result

    async def get_browser_console_logs(
        self,
        browser_id: str,
        max_logs: int = 50,
        log_types: Optional[List[str]] = None,
        minutes_back: Optional[int] = None,
        max_chars: Optional[int] = 8000,
    ) -> dict:
        """
        Get console logs from a browser with configurable filtering to prevent context window overflow.

        Args:
            browser_id: The unique identifier of the browser instance
            max_logs: Maximum number of logs to return (most recent)
            log_types: List of log types to include (e.g., ['error', 'warning', 'assert'])
                      If None, defaults to important types: ['error', 'warning', 'assert']
            minutes_back: Only return logs from the last N minutes
            max_chars: Maximum total character count for all log messages combined

        Returns:
            Dictionary containing filtered console logs and metadata
        """
        if browser_id not in self.browsers:
            raise KeyError(f"Browser with ID {browser_id} not found.")

        browser_info = self.browsers[browser_id]
        console_logs = browser_info["console_logs"].copy()  # Work with a copy
        original_count = len(console_logs)

        # Apply time filter first if specified
        if minutes_back is not None:
            cutoff_time = datetime.datetime.now() - datetime.timedelta(minutes=minutes_back)
            console_logs = [
                log for log in console_logs if datetime.datetime.fromisoformat(log["timestamp"]) > cutoff_time
            ]

        # Apply type filter - default to important types if not specified
        if log_types is None:
            log_types = ["error", "warning", "assert"]

        if log_types:  # Only filter if log_types is not empty
            console_logs = [log for log in console_logs if log["type"] in log_types]

        # Apply count limit (most recent)
        if len(console_logs) > max_logs:
            console_logs = console_logs[-max_logs:]

        # Apply character limit if specified
        if max_chars is not None:
            limited_logs = []
            char_count = 0
            for log in reversed(console_logs):
                log_text = f"{log['type']}: {log['text']}"
                if char_count + len(log_text) > max_chars and limited_logs:
                    break
                limited_logs.insert(0, log)
                char_count += len(log_text)
            console_logs = limited_logs

        return {
            "console_logs": console_logs,
            "total_logs_count": original_count,
            "returned_count": len(console_logs),
            "filters_applied": {
                "max_logs": max_logs,
                "log_types": log_types,
                "minutes_back": minutes_back,
                "max_chars": max_chars,
            },
        }

    async def get_browser_interactive_elements(self, browser_id: str) -> list:
        if browser_id not in self.browsers:
            raise KeyError(f"Browser with ID {browser_id} not found.")

        browser_info = self.browsers[browser_id]
        page = browser_info["page"]
        current_url = page.url
        title = await page.title()
        html = await page.content()

        interactive_elements = extract_interactive_elements_from_html(html)

        return {"current_url": current_url, "title": title, "interactive_elements": interactive_elements}

    async def get_browser_content(self, browser_id: str, **console_log_filters) -> dict:
        """
        Get the current content of a browser page including HTML and filtered console logs.

        Args:
            browser_id: The unique identifier of the browser instance
            **console_log_filters: Optional filters for console logs (max_logs, log_types, minutes_back, max_chars)

        Returns:
            Dictionary containing current URL, title, HTML content, and filtered console logs
        """
        if browser_id not in self.browsers:
            raise KeyError(f"Browser with ID {browser_id} not found.")

        browser_info = self.browsers[browser_id]
        page = browser_info["page"]
        current_url = page.url
        title = await page.title()
        html = await page.content()

        # Get filtered console logs using the new method
        console_log_result = await self.get_browser_console_logs(browser_id, **console_log_filters)
        console_logs = console_log_result["console_logs"]

        return {
            "current_url": current_url,
            "title": title,
            "html": html,
            "console_logs": console_logs,
            "console_log_metadata": {
                "total_logs_count": console_log_result["total_logs_count"],
                "returned_count": console_log_result["returned_count"],
                "filters_applied": console_log_result["filters_applied"],
            },
        }

    async def take_browser_screenshot(self, browser_id: str) -> dict:
        if browser_id not in self.browsers:
            raise KeyError(f"Browser with ID {browser_id} not found.")

        browser_info = self.browsers[browser_id]
        page = browser_info["page"]
        current_url = page.url
        title = await page.title()

        screenshot = base64.b64encode(await page.screenshot()).decode("utf-8")

        return {"current_url": current_url, "title": title, "screenshot": screenshot}

    async def interact_with_browser(self, browser_id: str, action: str, selector: str, text: str, scroll_px) -> dict:
        if browser_id not in self.browsers:
            raise KeyError(f"Browser with ID {browser_id} not found.")

        browser_info = self.browsers[browser_id]
        page = browser_info["page"]

        if action == "click":
            await page.click(selector, timeout=self.interaction_timeout)
        elif action == "type":
            await page.fill(selector, text)
        elif action == "scroll":
            await page.evaluate(f"window.scrollBy(0, {scroll_px})")
        elif action == "navigate":
            await page.goto(text, wait_until="domcontentloaded")
        else:
            raise ValueError(f"Unknown action: {action}")

        await page.wait_for_load_state("networkidle")

        return {"status": "success", "current_url": page.url, "action": action, "selector": selector, "text": text}

    async def set_select_value(self, browser_id: str, selector: str, value: str) -> dict:
        if browser_id not in self.browsers:
            raise KeyError(f"Browser with ID {browser_id} not found.")

        browser_info = self.browsers[browser_id]
        page = browser_info["page"]

        try:
            # First, check if the element exists and is a select element
            element = await page.query_selector(selector)
            if not element:
                raise ValueError(f"Element with selector '{selector}' not found.")

            # Check if it's actually a select element
            tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
            if tag_name != "select":
                raise ValueError(f"Element with selector '{selector}' is not a select box (found: {tag_name}).")

            # Get all option values to validate the value exists
            option_values = await element.evaluate(
                """
                el => Array.from(el.options).map(option => option.value)
            """
            )

            if value not in option_values:
                raise ValueError(f"Value '{value}' not found in select options. Available values: {option_values}")

            # Set the value using Playwright's select_option method
            await page.select_option(selector, value, timeout=self.interaction_timeout)

            # Get the currently selected value to confirm
            selected_value = await element.evaluate("el => el.value")

            return {
                "status": "success",
                "current_url": page.url,
                "selector": selector,
                "selected_value": selected_value,
                "requested_value": value,
            }

        except Exception as e:
            return {"status": "error", "current_url": page.url, "selector": selector, "error": str(e)}

    async def close_browser(self, browser_id: str) -> None:
        if browser_id not in self.browsers:
            raise KeyError(f"Browser with ID {browser_id} not found.")

        # Cancel the keepalive task for this browser
        if browser_id in self.keepalive_tasks:
            self.keepalive_tasks[browser_id].cancel()
            del self.keepalive_tasks[browser_id]

        await self.browsers[browser_id]["browser"].close()
        await self.browsers[browser_id]["playwright"].stop()

        del self.browsers[browser_id]

    async def cleanup_all_browsers(self) -> None:
        """
        Close all open browser instances and clean up resources.
        This should be called when the browser manager is being destroyed.
        """
        browser_ids = list(self.browsers.keys())  # Create a copy of keys to avoid modification during iteration

        for browser_id in browser_ids:
            try:
                await self.close_browser(browser_id)
            except Exception as e:
                # Log error but continue closing other browsers
                print(f"Error closing browser {browser_id}: {e}")

        # Cancel any remaining keepalive tasks (cleanup safety)
        for task_id, task in list(self.keepalive_tasks.items()):
            task.cancel()
        self.keepalive_tasks.clear()
