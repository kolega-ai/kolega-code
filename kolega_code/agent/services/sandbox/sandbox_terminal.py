"""Deprecated location. Import from kolega_code.sandbox.terminal instead."""

import warnings

warnings.warn(
    "kolega_code.agent.services.sandbox.sandbox_terminal is deprecated; import from kolega_code.sandbox.terminal instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kolega_code.sandbox.terminal import *  # noqa: F401,F403,E402
from kolega_code.sandbox.terminal import SandboxTerminalManager  # noqa: F401,E402
