"""Deprecated location. Import from kolega_code.sandbox.filesystem instead."""

import warnings

warnings.warn(
    "kolega_code.agent.services.sandbox.sandbox_filesystem is deprecated; import from kolega_code.sandbox.filesystem instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kolega_code.sandbox.filesystem import *  # noqa: F401,F403,E402
from kolega_code.sandbox.filesystem import SandboxFileSystem  # noqa: F401,E402
