"""Deprecated location. Import from kolega_code.services.browser instead."""

import warnings

warnings.warn(
    "kolega_code.agent.services.browser is deprecated; import from kolega_code.services.browser instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kolega_code.services.browser import *  # noqa: F401,F403,E402
from kolega_code.services.browser import PlaywrightBrowserManager, BrowserManager  # noqa: F401,E402
