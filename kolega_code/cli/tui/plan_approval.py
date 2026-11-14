from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from kolega_code.agent import BaseAgent
from kolega_code.session.plans import PlanApprovalAction
from kolega_code.session.runtime import SessionRuntime


@dataclass(eq=False)
class PlanApprovalOwner:
    session_id: str
    runtime: SessionRuntime
    agent: BaseAgent
    revision: int
    plan: str
    origin_task: asyncio.Task | None = None
    task: asyncio.Task[None] | None = None
    request_id: str | None = None
    action: PlanApprovalAction | None = None
    valid: bool = True
    transitioning: bool = False
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    released: asyncio.Event = field(default_factory=asyncio.Event)
    invalidated: asyncio.Event = field(default_factory=asyncio.Event)
