"""Deprecated location. Import from kolega_code.services instead."""

import warnings

warnings.warn(
    "kolega_code.agent.services is deprecated; import from kolega_code.services instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kolega_code.services import *  # noqa: F401,F403,E402
