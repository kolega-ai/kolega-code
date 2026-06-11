"""Deprecated location. Import from kolega_code.sandbox.local instead."""

import warnings

warnings.warn(
    "kolega_code.agent.services.sandbox.local_sandbox is deprecated; import from kolega_code.sandbox.local instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kolega_code.sandbox.local import *  # noqa: F401,F403,E402
from kolega_code.sandbox.local import LocalSandboxManager  # noqa: F401,E402
