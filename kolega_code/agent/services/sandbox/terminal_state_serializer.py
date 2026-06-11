"""Deprecated location. Import from kolega_code.sandbox.serializer instead."""

import warnings

warnings.warn(
    "kolega_code.agent.services.sandbox.terminal_state_serializer is deprecated; import from kolega_code.sandbox.serializer instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kolega_code.sandbox.serializer import *  # noqa: F401,F403,E402
from kolega_code.sandbox.serializer import TerminalStateSerializer  # noqa: F401,E402
