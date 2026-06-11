"""Deprecated location. Import from kolega_code.llm.models instead."""

import warnings

warnings.warn(
    "kolega_code.agent.llm.models is deprecated; import from kolega_code.llm.models instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kolega_code.llm.models import *  # noqa: F401,F403,E402
