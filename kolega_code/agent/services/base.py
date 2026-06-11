"""Deprecated location. Import from kolega_code.services.base instead."""

import warnings

warnings.warn(
    "kolega_code.agent.services.base is deprecated; import from kolega_code.services.base instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kolega_code.services.base import *  # noqa: F401,F403,E402
from kolega_code.services.base import TerminalManager, BrowserManager  # noqa: F401,E402
