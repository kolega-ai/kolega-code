"""Deprecated location. Import from kolega_code.llm instead."""

import warnings

warnings.warn(
    "kolega_code.agent.llm is deprecated; import from kolega_code.llm instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kolega_code.llm import *  # noqa: F401,F403,E402
