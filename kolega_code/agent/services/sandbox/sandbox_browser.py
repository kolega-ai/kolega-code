"""Deprecated location. Import from kolega_code.sandbox.browser instead."""

import warnings

warnings.warn(
    "kolega_code.agent.services.sandbox.sandbox_browser is deprecated; import from kolega_code.sandbox.browser instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kolega_code.sandbox.browser import *  # noqa: F401,F403,E402
from kolega_code.sandbox.browser import SandboxBrowserManager  # noqa: F401,E402
