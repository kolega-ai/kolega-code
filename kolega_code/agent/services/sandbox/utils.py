"""Deprecated location. Import from kolega_code.sandbox.utils instead."""

import warnings

warnings.warn(
    "kolega_code.agent.services.sandbox.utils is deprecated; import from kolega_code.sandbox.utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kolega_code.sandbox.utils import *  # noqa: F401,F403,E402
