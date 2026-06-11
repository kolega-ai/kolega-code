"""Deprecated location. Import from kolega_code.sandbox.base instead."""

import warnings

warnings.warn(
    "kolega_code.agent.services.sandbox.base is deprecated; import from kolega_code.sandbox.base instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kolega_code.sandbox.base import *  # noqa: F401,F403,E402
from kolega_code.sandbox.base import SandboxManager, SandboxConfig, ProjectManifest  # noqa: F401,E402
