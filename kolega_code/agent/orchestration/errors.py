"""Exceptions raised by the gigacode workflow runtime."""


class WorkflowError(RuntimeError):
    """Base class for all workflow orchestration failures."""


class WorkflowScriptError(WorkflowError):
    """The authored script is malformed or used a primitive incorrectly.

    Raised for problems the model can fix by rewriting the script: a missing or
    non-literal ``meta`` block, a non-string prompt, an over-large fan-out, or an
    illegally nested ``workflow()`` call.
    """


class WorkflowBudgetExceeded(WorkflowError):
    """The run reached its token budget ceiling; further ``agent()`` calls abort."""


class WorkflowAgentCapExceeded(WorkflowError):
    """The run exceeded the lifetime agent-count backstop (runaway-loop guard)."""
