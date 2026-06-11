"""Deprecated location. Import from kolega_code.llm.instrumented_client instead."""

import warnings

warnings.warn(
    "kolega_code.agent.llm.instrumented_client is deprecated; import from kolega_code.llm.instrumented_client instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kolega_code.llm.instrumented_client import *  # noqa: F401,F403,E402
from kolega_code.llm.instrumented_client import InstrumentedLLMClient  # noqa: F401,E402
