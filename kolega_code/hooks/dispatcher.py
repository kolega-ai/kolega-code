"""HookDispatcher: select matching hooks for an event, run them, merge outcomes.

The dispatcher is stateless: every call is a pure function of its arguments and
returns a fresh ``HookOutcome``. This is required for correctness — tool events
fire concurrently for parallel-safe batches and recursively inside sub-agents, so
the dispatcher must never stash per-call state on itself.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from .backends import HookCapabilities, run_hook
from .config import HookConfig
from .events import TOOL_EVENTS, LifecycleEvent, enter_hook, exit_hook, in_hook
from .outcome import HookOutcome, merge


class HookDispatcher:
    """Runs the hooks configured for a lifecycle event and folds their outcomes."""

    def __init__(self, config: HookConfig) -> None:
        self.config = config

    @property
    def is_active(self) -> bool:
        return not self.config.is_empty

    async def dispatch(
        self,
        event: LifecycleEvent,
        *,
        target: str = "",
        caps: Optional[HookCapabilities] = None,
    ) -> HookOutcome:
        # Suppress re-entrant dispatch: a hook (e.g. an ``agent`` hook) may itself
        # drive the agent / spawn a sub-agent, whose fire points must not recurse.
        if in_hook():
            return HookOutcome.empty()

        specs = self.config.specs_for(event.name, target)
        if not specs:
            return HookOutcome.empty()

        caps = caps or HookCapabilities()
        token = enter_hook()
        try:
            current_input = event.payload.get("tool_input")
            outcomes: list[HookOutcome] = []
            for spec in specs:
                fired = event
                if event.name in TOOL_EVENTS and current_input is not None:
                    fired = replace(event, payload={**event.payload, "tool_input": current_input})
                try:
                    outcome = await run_hook(fired, spec, caps)
                except Exception as exc:  # noqa: BLE001 - hooks must never crash the turn
                    await self._log(caps, f"hook error [{event.name.value}/{spec.type}]: {exc}")
                    outcome = HookOutcome.empty()

                outcomes.append(outcome)
                if outcome.updated_input is not None:
                    current_input = outcome.updated_input
                if outcome.blocked:
                    break  # first block wins; skip remaining hooks

            return merge(outcomes)
        finally:
            exit_hook(token)

    @staticmethod
    async def _log(caps: HookCapabilities, message: str) -> None:
        if caps.log is not None:
            try:
                await caps.log(message)
            except Exception:  # noqa: BLE001 - logging must never crash the turn
                pass


# Default dispatcher used everywhere a host has not configured hooks. Its empty
# config makes dispatch() return immediately, so existing callers/tests are unaffected.
NO_OP_DISPATCHER = HookDispatcher(HookConfig())
