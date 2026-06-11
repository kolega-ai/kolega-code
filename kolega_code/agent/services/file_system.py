"""Deprecated location. Import from kolega_code.services.file_system instead."""

import warnings

warnings.warn(
    "kolega_code.agent.services.file_system is deprecated; import from kolega_code.services.file_system instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kolega_code.services.file_system import *  # noqa: F401,F403,E402
from kolega_code.services.file_system import FileSystem, LocalFileSystem, FileSystemPath  # noqa: F401,E402
