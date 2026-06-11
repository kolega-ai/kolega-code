"""Deprecated location. Import from kolega_code.config instead."""

import warnings

warnings.warn(
    "kolega_code.agent.config is deprecated; import from kolega_code.config instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kolega_code.config import *  # noqa: F401,F403,E402
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig  # noqa: F401,E402
