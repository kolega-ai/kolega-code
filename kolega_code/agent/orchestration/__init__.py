"""gigacode — dynamic multi-agent workflow orchestration.

A workflow is a short authored Python script whose primitives (``agent``,
``parallel``, ``pipeline``, ``phase``, ``log``) fan out sub-agents with real
control flow. The pieces:

- :mod:`executor`  — validate ``meta``, wrap the body, run it in a curated namespace
- :mod:`runtime`   — the primitives, concurrency caps, and resume
- :mod:`budget`    — token ceiling shared across the run
- :mod:`journal`   — state-dir artifacts and the resume journal
- :mod:`types`     — the dispatch/emit interface the production adapter implements
"""

from .budget import Budget
from .errors import (
    WorkflowAgentCapExceeded,
    WorkflowBudgetExceeded,
    WorkflowError,
    WorkflowScriptError,
)
from .executor import extract_meta, run_script, safe_builtins
from .journal import RunJournal, saved_workflows_dir, workflows_root
from .runtime import DEFAULT_AGENT_CAP, MAX_FANOUT, WorkflowRuntime, default_concurrency
from .types import AgentRunResult, AgentRunSpec, DispatchFn, EmitFn

__all__ = [
    "Budget",
    "WorkflowError",
    "WorkflowScriptError",
    "WorkflowBudgetExceeded",
    "WorkflowAgentCapExceeded",
    "extract_meta",
    "run_script",
    "safe_builtins",
    "RunJournal",
    "workflows_root",
    "saved_workflows_dir",
    "WorkflowRuntime",
    "default_concurrency",
    "MAX_FANOUT",
    "DEFAULT_AGENT_CAP",
    "AgentRunSpec",
    "AgentRunResult",
    "DispatchFn",
    "EmitFn",
]
