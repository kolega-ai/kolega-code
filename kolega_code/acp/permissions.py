"""Agent permission requests answered through the ACP client.

Saved allow rules are matched first (same semantics as the TUI's
``SessionRuntime``); otherwise the request is sent to the client via
``session/request_permission`` and the outcome mapped back to a
``PermissionDecision``. allow-always persists the chosen rule before
answering.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

from acp.helpers import update_tool_call
from acp.interfaces import Client
from acp.schema import PermissionOption

from kolega_code.acp.bridge import TOOL_KINDS
from kolega_code.permissions import (
    PermissionDecision,
    PermissionRequest,
    PermissionRule,
    ProjectPermissionStore,
    allow_rule_options,
)
from kolega_code.session.control import DEFAULT_REQUEST_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

OPTION_ALLOW_ONCE = "allow-once"
OPTION_REJECT_ONCE = "reject-once"


class AcpPermissionBroker:
    """Prompt and rule handling for one ACP session's permission requests."""

    def __init__(
        self,
        conn: Client,
        session_id: str,
        project_path: Path,
        agent_provider: Callable[[], Any],
        *,
        timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._conn = conn
        self._session_id = session_id
        self._store = ProjectPermissionStore(project_path)
        self._agent_provider = agent_provider
        self._timeout = timeout

    async def __call__(self, request: PermissionRequest) -> PermissionDecision:
        saved = self._store.first_match(request)
        if saved is not None:
            return PermissionDecision(allowed=True, reason=f"Allowed by saved rule {saved.id}.", rule=saved)
        return await self._prompt(request)

    async def _prompt(self, request: PermissionRequest) -> PermissionDecision:
        rules_by_option: dict[str, PermissionRule] = {}
        options: list[PermissionOption] = [
            PermissionOption(option_id=OPTION_ALLOW_ONCE, name="Allow once", kind="allow_once"),
        ]
        for index, candidate in enumerate(allow_rule_options(request)):
            option_id = f"allow-always-{index}"
            rules_by_option[option_id] = candidate.rule
            options.append(PermissionOption(option_id=option_id, name=candidate.label, kind="allow_always"))
        options.append(PermissionOption(option_id=OPTION_REJECT_ONCE, name="Reject", kind="reject_once"))

        tool_call = update_tool_call(
            self._current_tool_call_id(),
            title=request.summary or request.tool_name,
            kind=TOOL_KINDS.get(request.tool_name, "other"),
            status="pending",
            raw_input=request.inputs,
        )
        try:
            response = await asyncio.wait_for(
                self._conn.request_permission(self._session_id, tool_call, options),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            return PermissionDecision(allowed=False, reason="Timed out waiting for approval.")
        except Exception:
            logger.exception("acp permission request failed")
            return PermissionDecision(allowed=False, reason="The permission prompt could not be shown.")

        outcome = getattr(response, "outcome", None)
        if outcome is None and isinstance(response, dict):
            outcome = response.get("outcome")
        if isinstance(outcome, dict):
            selected = outcome.get("outcome") == "selected"
            option_id = str(outcome.get("optionId") or "")
        else:
            selected = getattr(outcome, "outcome", None) == "selected"
            option_id = str(getattr(outcome, "option_id", "") or "")
        if not selected:
            return PermissionDecision(allowed=False, reason="Denied by the user.")
        if option_id == OPTION_ALLOW_ONCE:
            return PermissionDecision(allowed=True)
        if option_id == OPTION_REJECT_ONCE:
            return PermissionDecision(allowed=False, reason="Denied by the user.")
        rule = rules_by_option.get(option_id)
        if rule is None:
            return PermissionDecision(allowed=False, reason="Unknown approval option.")
        try:
            self._store.add_rule(rule)
        except Exception:
            logger.exception("acp could not persist allow-always rule")
        return PermissionDecision(allowed=True, reason="Allowed and remembered.", rule=rule)

    def _current_tool_call_id(self) -> str:
        agent = self._agent_provider()
        return str(getattr(agent, "current_tool_call_id", "") or "") or "pending"
