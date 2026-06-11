"""Deprecated location. Import from kolega_code.services.terminal instead."""

import warnings

warnings.warn(
    "kolega_code.agent.services.terminal is deprecated; import from kolega_code.services.terminal instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kolega_code.services.terminal import *  # noqa: F401,F403,E402
from kolega_code.services.terminal import LocalTerminalManager, AsyncPersistentTerminal  # noqa: F401,E402
