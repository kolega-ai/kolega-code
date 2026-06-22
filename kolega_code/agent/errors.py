"""Exceptions raised by agent runtime loops."""


class AgentError(RuntimeError):
    """Base class for catchable agent runtime errors."""


class MaxAgentIterationsExceeded(AgentError):
    """Raised when an opt-in agent loop iteration cap is exceeded."""
