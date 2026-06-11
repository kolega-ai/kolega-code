"""Deprecated location. Import from kolega_code.llm.client instead."""

import warnings

warnings.warn(
    "kolega_code.agent.llm.client is deprecated; import from kolega_code.llm.client instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kolega_code.llm.client import *  # noqa: F401,F403,E402
from kolega_code.llm.client import LLMClient  # noqa: F401,E402
