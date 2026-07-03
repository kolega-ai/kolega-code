from __future__ import annotations

from typing import Optional

import yaml

from .base_tool import BaseTool


class BuildTool(BaseTool):
    """
    Provides backend and frontend build operations driven by .kolega-manifest.yaml.

    Resolves the appropriate build command and executes it via the injected
    TerminalManager so it works in both local and sandbox environments.
    """

    def _read_manifest(self) -> dict:
        """
        Read the project manifest from the repository root.

        Returns an empty dict when the file is missing or invalid.
        """
        manifest_path = ".kolega-manifest.yaml"
        try:
            if not self.filesystem.exists(manifest_path):
                return {}
            content = self.filesystem.read_text(manifest_path)
            data = yaml.safe_load(content) or {}
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _get_build_command(manifest: dict, kind: str) -> Optional[str]:
        """
        Resolve the build command from the manifest.

        kind: 'backend' | 'frontend'
        """
        if not isinstance(manifest, dict):
            return None
        if kind == "backend":
            return manifest.get("backend_build_command") or manifest.get("build_command")
        if kind == "frontend":
            return manifest.get("frontend_build_command") or manifest.get("build_command")
        return None

    async def _run_build(self, kind: str) -> str:
        """
        Execute the resolved build command and return markdown-formatted output.
        """
        manifest = self._read_manifest()
        command = self._get_build_command(manifest, kind)

        if not command:
            return f"Error: No {kind}_build_command or build_command found in .kolega-manifest.yaml"

        try:
            assert self.terminal_manager is not None, "terminal_manager is required to run build commands"
            output = await self.terminal_manager.run_command(
                command=command,
                cwd=str(self.project_path),
                timeout=1800,
            )
        except Exception as exc:
            return f"Build failed to start: {str(exc)}"

        return f"""Ran {kind} build command:

```
{command}
```

Output:
```
{output}
```"""

    async def build_backend(self) -> str:
        """
        Build the backend using the manifest command (backend_build_command → build_command).

        Returns the combined stdout/stderr output as markdown.
        """
        return await self._run_build("backend")

    async def build_frontend(self) -> str:
        """
        Build the frontend using the manifest command (frontend_build_command → build_command).

        Returns the combined stdout/stderr output as markdown.
        """
        return await self._run_build("frontend")
