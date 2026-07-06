"""Shared fixtures for LSP integration tests.

Provides a fake LSP server fixture that launches ``_fake_server.py`` as a
subprocess and returns a configured ``LspManager`` connected to it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator

import pytest_asyncio

if TYPE_CHECKING:
    from kolega_code.services.lsp import LspManager

_FAKE_SERVER = Path(__file__).parent / "_fake_server.py"


@pytest_asyncio.fixture
async def fake_lsp_manager(tmp_path: Path) -> AsyncGenerator["LspManager", None]:
    """Return an ``LspManager`` backed by the fake LSP server.

    Creates a temporary project with a Python file so detection resolves the
    fake server.  The manager is initialized and ready to query.
    """
    from kolega_code.services.lsp import LspConfig
    from kolega_code.services.lsp.manager import LspManager

    # Create a minimal project so Python is detected
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    src = project / "src.py"
    src.write_text("def hello():\n    pass\n", encoding="utf-8")

    config = LspConfig(
        enabled=True,
        prompt_on_missing=False,
        custom_servers={
            "fake-lsp": {
                "bin": sys.executable,
                "args": [str(_FAKE_SERVER)],
                "languages": ["python"],
            },
        },
        preferences={"python": "fake-lsp"},
    )

    manager = LspManager(project, config=config)
    await manager.initialize()

    yield manager

    await manager.shutdown()


@pytest_asyncio.fixture
async def fake_lsp_manager_no_pull(tmp_path: Path) -> AsyncGenerator["LspManager", None]:
    """Like ``fake_lsp_manager`` but the fake server doesn't support pull diagnostics."""
    from kolega_code.services.lsp import LspConfig
    from kolega_code.services.lsp.manager import LspManager

    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    src = project / "src.py"
    src.write_text("def hello():\n    pass\n", encoding="utf-8")

    config = LspConfig(
        enabled=True,
        prompt_on_missing=False,
        custom_servers={
            "fake-lsp": {
                "bin": sys.executable,
                "args": [str(_FAKE_SERVER), "dummy"],  # args ignored by fake server
                "languages": ["python"],
                "env": {"FAKE_LSP_PULL_DIAGS": "0"},
            },
        },
        preferences={"python": "fake-lsp"},
    )

    manager = LspManager(project, config=config)
    await manager.initialize()

    yield manager

    await manager.shutdown()


@pytest_asyncio.fixture
async def fake_lsp_manager_with_extra_strict(tmp_path: Path) -> AsyncGenerator["LspManager", None]:
    """Manager with primary + extra fake servers that reject didChange before didOpen."""
    from kolega_code.services.lsp import LspConfig
    from kolega_code.services.lsp.manager import LspManager

    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    (project / "src.py").write_text("def hello():\n    pass\n", encoding="utf-8")

    config = LspConfig(
        enabled=True,
        prompt_on_missing=False,
        custom_servers={
            "fake-primary": {
                "bin": sys.executable,
                "args": [str(_FAKE_SERVER)],
                "languages": ["python"],
                "env": {"FAKE_LSP_STRICT_OPEN": "1", "FAKE_LSP_SOURCE": "primary"},
            },
            "fake-extra": {
                "bin": sys.executable,
                "args": [str(_FAKE_SERVER)],
                "languages": ["python"],
                "env": {"FAKE_LSP_STRICT_OPEN": "1", "FAKE_LSP_SOURCE": "extra"},
            },
        },
        preferences={"python": "fake-primary"},
        diagnostic_servers=["fake-extra"],
    )

    manager = LspManager(project, config=config)
    await manager.initialize()

    yield manager

    await manager.shutdown()
