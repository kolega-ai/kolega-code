"""Sanitized environment for Git subprocesses.

Git honours a handful of environment variables that retarget it at a different
repository, working tree, or index regardless of the process working directory.
When Kolega Code runs inside a linked worktree and one of them is exported (by a
Git hook, a wrapper script, or a shell the agent itself used), commands such as
``git rev-parse --show-toplevel`` silently resolve to the *main* checkout, so
worktree-scoped features would report another worktree's state.

Every Git subprocess that must resolve the repository from its working directory
should pass ``env=git_env()``.
"""

from __future__ import annotations

import os
from typing import Mapping

# Variables that make Git ignore the working directory when locating the
# repository, the work tree, the index, or the object store.
GIT_ENV_OVERRIDES = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES",
    "GIT_NAMESPACE",
    "GIT_PREFIX",
)


def git_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return ``base`` (default ``os.environ``) without repository-retargeting Git variables."""
    source = os.environ if base is None else base
    return {key: value for key, value in source.items() if key not in GIT_ENV_OVERRIDES}
