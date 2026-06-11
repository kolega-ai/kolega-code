"""Deprecated location. Import from kolega_code.sandbox instead."""

import warnings

warnings.warn(
    "kolega_code.agent.services.sandbox is deprecated; import from kolega_code.sandbox instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kolega_code.sandbox import *  # noqa: F401,F403,E402
