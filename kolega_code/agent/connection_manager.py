"""Deprecated location. Import from kolega_code.events instead."""

import warnings

warnings.warn(
    "kolega_code.agent.connection_manager is deprecated; import from kolega_code.events instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kolega_code.events import *  # noqa: F401,F403,E402
from kolega_code.events import AgentConnectionManager, AgentEvent  # noqa: F401,E402
