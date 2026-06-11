"""Deprecated location. Import from kolega_code.sandbox.async_filesystem instead."""

import warnings

warnings.warn(
    "kolega_code.agent.services.sandbox.async_sandbox_filesystem is deprecated; import from kolega_code.sandbox.async_filesystem instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kolega_code.sandbox.async_filesystem import *  # noqa: F401,F403,E402
from kolega_code.sandbox.async_filesystem import AsyncSandboxFileSystem  # noqa: F401,E402
