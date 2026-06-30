import asyncio
import json
import os
import re
import shlex
import shutil
from base64 import b64decode
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

from .base_tool import BaseTool

# Marker appended to sandbox shell commands so the shell always exits 0 (the E2B
# command runner may raise on a non-zero exit). We recover the real exit code by
# parsing this line out of stdout.
_RC_MARKER = "__KOLEGA_RG_RC"

# Display caps (shared by every engine via the single formatter).
_MAX_FILES = 128
_MAX_LINES_PER_FILE = 5
_MAX_LINE_LENGTH = 200
_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


class _EngineUnavailable(Exception):
    """Raised by an engine runner when it cannot run (binary missing, unknown
    flag, non-regex error) so the caller falls back to the next engine tier."""


# One normalized match line, shared currency between every engine and the formatter.
# (relative_path, line_number, raw_line_text)
Record = Tuple[str, int, str]


class SearchCodebaseTool(BaseTool):
    # Define binary extensions to skip without checking content
    BINARY_EXTENSIONS = {
        ".pyc",
        ".so",
        ".dll",
        ".exe",
        ".bin",
        ".jar",
        ".war",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".ico",
        ".svg",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".tgz",
        ".rar",
        ".7z",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wav",
        ".o",
        ".obj",
        ".class",
        ".binary",
        ".wasm",
        ".node",
    }

    # Define directories to exclude
    EXCLUDE_DIRS = {
        ".git",
        ".svn",
        ".hg",
        ".idea",
        ".vscode",
        "__pycache__",
        "node_modules",
        "venv",
        "env",
        ".env",
        "dist",
        "build",
        "target",
        "bin",
        "obj",
        ".next",
        ".nuxt",
        "coverage",
    }

    @property
    def _fs_root(self) -> str:
        """Filesystem root as a string (LocalFileSystem and SandboxFileSystem both
        set ``root_path``; it is not on the base ``FileSystem`` type)."""
        return str(cast(Any, self.filesystem).root_path)

    @property
    def _fs_sandbox(self) -> Any:
        """The sandbox handle (only present on sandbox filesystems)."""
        return cast(Any, self.filesystem).sandbox

    async def search_codebase(
        self, pattern: str, file_pattern: str = "*", case_sensitive: bool = False, literal: bool = False
    ) -> str:
        """
        Search the codebase for lines matching a regular expression (grep/ripgrep).

        The pattern is treated as a regular expression by default, so `|` is
        alternation: search for `TODO|FIXME|HACK` to match any of the three. Use
        ripgrep/POSIX-ERE syntax (alternation, character classes `[...]`, anchors
        `^ $`, quantifiers `* + ? {n,m}`, groups `(...)`). Set `literal=True` to match
        the pattern as plain text instead (e.g. to find `arr[0]` or `a||b` verbatim).

        Args:
            pattern: The regular expression to search for (use `literal=True` to match it as plain text)
            file_pattern: Optional glob to filter which files to search (default: all files)
            case_sensitive: Whether the search is case-sensitive (default: False)
            literal: Treat the pattern as plain text instead of a regular expression (default: False)

        Returns:
            Markdown formatted list of files and matches, limited to 128 results

        Raises:
            Exception: If any error occurs during the search operation
        """
        try:
            await self.log_info(f"Searching codebase for pattern: '{pattern}'", sender=self.caller.agent_name)

            # A NUL byte can never appear in searchable text and would break the
            # subprocess argv / shell command, so short-circuit to no-match.
            if "\x00" in pattern:
                return f"No matches found for pattern '{pattern}'"

            # If literal search, escape special regex characters for the validator
            # and the in-process Python fallback engine.
            search_pattern = re.escape(pattern) if literal else pattern

            # Compile the pattern to validate it (and to drive the Python fallback).
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(search_pattern, flags)
            except re.error as e:
                error_msg = f"Invalid regular expression: {str(e)}"
                await self.log_error(error_msg, sender=self.caller.agent_name)
                return f"Error: {error_msg}"

            # Run the first available engine; fall back down the tiers on failure.
            for runner in await self._engine_chain():
                try:
                    records = await runner(pattern, file_pattern, case_sensitive, literal, regex)
                except _EngineUnavailable:
                    continue
                return self._format_results(records, pattern)

            # The Python tier never raises _EngineUnavailable, so this is unreachable.
            return f"No matches found for pattern '{pattern}'"

        except Exception as e:
            error_msg = f"Error searching codebase: {str(e)}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            return f"Error: {error_msg}"

    # ------------------------------------------------------------------
    # Engine selection
    # ------------------------------------------------------------------

    async def _engine_chain(self):
        """Ordered list of engine runners to try: ripgrep (preferred) -> grep
        -> in-process Python `re` (always succeeds)."""
        if hasattr(self.filesystem, "sandbox"):
            chain = []
            if await self._sandbox_has_rg():
                chain.append(self._run_rg_sandbox)
            chain.append(self._run_grep_sandbox)
            chain.append(self._run_python)
            return chain

        chain = []
        if shutil.which("rg"):
            chain.append(self._run_rg_local)
        if shutil.which("grep"):
            chain.append(self._run_grep_local)
        chain.append(self._run_python)
        return chain

    async def _sandbox_has_rg(self) -> bool:
        """Probe (once per sandbox) whether ripgrep is installed in the sandbox."""
        sandbox = self._fs_sandbox
        if getattr(self, "_rg_probe_sandbox", None) is sandbox:
            return self._rg_probe_result
        available = False
        try:
            result = await sandbox.commands.run(f"command -v rg >/dev/null 2>&1 ; echo {_RC_MARKER}=$?")
            rc, _ = self._split_rc(result.stdout or "")
            available = rc == 0
        except Exception:
            available = False
        self._rg_probe_sandbox = sandbox
        self._rg_probe_result = available
        return available

    # ------------------------------------------------------------------
    # ripgrep engine
    # ------------------------------------------------------------------

    def _rg_args(self, pattern: str, file_pattern: str, case_sensitive: bool, literal: bool) -> List[str]:
        args = [
            "--json",
            "--hidden",
            "--no-config",
            "--no-ignore-parent",
            "--no-ignore-global",
            "--glob-case-insensitive",
            "--max-filesize",
            "10M",
        ]
        if not case_sensitive:
            args.append("-i")
        if literal:
            args.append("-F")
        # --hidden re-includes .git etc., so explicitly re-exclude the dirs.
        for exclude_dir in sorted(self.EXCLUDE_DIRS):
            args += ["-g", f"!{exclude_dir}/"]
        # Also drop a *file* literally named .env (the glob above only excludes dirs).
        args += ["-g", "!.env"]
        # ripgrep does not skip by extension, so reproduce the binary-extension skip.
        for ext in sorted(self.BINARY_EXTENSIONS):
            args += ["-g", f"!*{ext}"]
        if file_pattern != "*":
            glob = file_pattern if "/" not in file_pattern else f"**/{file_pattern}"
            args += ["-g", glob]
        # Pattern as its own argv element (after -e) and an explicit search path so
        # ripgrep never reads stdin in a headless context.
        args += ["-e", pattern, "."]
        return args

    async def _run_rg_local(
        self, pattern: str, file_pattern: str, case_sensitive: bool, literal: bool, regex
    ) -> List[Record]:
        if shutil.which("rg") is None:
            raise _EngineUnavailable()
        args = self._rg_args(pattern, file_pattern, case_sensitive, literal)
        env = {**os.environ, "RIPGREP_CONFIG_FILE": ""}
        try:
            proc = await asyncio.create_subprocess_exec(
                "rg",
                *args,
                cwd=self._fs_root,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, _stderr = await proc.communicate()
        except (OSError, ValueError):
            raise _EngineUnavailable()
        # 0 = matches, 1 = no matches; anything else (2 = usage/regex error on an
        # old rg, etc.) falls back to the next tier.
        if proc.returncode not in (0, 1):
            raise _EngineUnavailable()
        return self._parse_rg_json(stdout.decode("utf-8", "replace"))

    async def _run_rg_sandbox(
        self, pattern: str, file_pattern: str, case_sensitive: bool, literal: bool, regex
    ) -> List[Record]:
        args = self._rg_args(pattern, file_pattern, case_sensitive, literal)
        cmd = "RIPGREP_CONFIG_FILE= rg " + " ".join(shlex.quote(a) for a in args)
        full_cmd = f"cd {shlex.quote(self._fs_root)} && {cmd} ; echo {_RC_MARKER}=$?"
        try:
            result = await self._fs_sandbox.commands.run(full_cmd)
        except Exception:
            raise _EngineUnavailable()
        rc, out = self._split_rc(result.stdout or "")
        if rc not in (0, 1):
            raise _EngineUnavailable()
        return self._parse_rg_json(out)

    def _parse_rg_json(self, stdout_text: str) -> List[Record]:
        records: List[Record] = []
        for line in stdout_text.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue  # skip the RC marker / any shell noise
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if obj.get("type") != "match":
                continue
            data = obj.get("data", {})
            path = self._json_text(data.get("path"))
            raw = self._json_text(data.get("lines"))
            line_num = data.get("line_number")
            if path is None or line_num is None:
                continue
            records.append((self._normalize_path(path), int(line_num), raw or ""))
        return records

    @staticmethod
    def _json_text(node) -> Optional[str]:
        """ripgrep --json fields are {"text": ...} or, for non-UTF8 data,
        {"bytes": "<base64>"}. Decode either defensively."""
        if not isinstance(node, dict):
            return None
        if "text" in node:
            return node["text"]
        if "bytes" in node:
            try:
                return b64decode(node["bytes"]).decode("utf-8", "replace")
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    # grep engine
    # ------------------------------------------------------------------

    def _grep_argv(self, pattern: str, file_pattern: str, case_sensitive: bool, literal: bool) -> List[str]:
        argv = ["grep", "-r", "-n", "--binary-files=without-match"]
        if not case_sensitive:
            argv.append("-i")
        argv.append("-F" if literal else "-E")
        if file_pattern != "*":
            argv.append(f"--include={file_pattern}")
        for exclude_dir in sorted(self.EXCLUDE_DIRS):
            argv.append(f"--exclude-dir={exclude_dir}")
        for ext in sorted(self.BINARY_EXTENSIONS):
            argv.append(f"--exclude=*{ext}")
        # -e guards a pattern that starts with '-'; '.' is the search root.
        argv += ["-e", pattern, "."]
        return argv

    async def _run_grep_local(
        self, pattern: str, file_pattern: str, case_sensitive: bool, literal: bool, regex
    ) -> List[Record]:
        if shutil.which("grep") is None:
            raise _EngineUnavailable()
        argv = self._grep_argv(pattern, file_pattern, case_sensitive, literal)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=self._fs_root,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _stderr = await proc.communicate()
        except (OSError, ValueError):
            raise _EngineUnavailable()
        if proc.returncode not in (0, 1):
            raise _EngineUnavailable()
        records = self._parse_grep(stdout.decode("utf-8", "replace"))
        # grep has no --max-filesize; reproduce the 10MB skip via stat (local only).
        return self._drop_oversize_local(records)

    async def _run_grep_sandbox(
        self, pattern: str, file_pattern: str, case_sensitive: bool, literal: bool, regex
    ) -> List[Record]:
        argv = self._grep_argv(pattern, file_pattern, case_sensitive, literal)
        cmd = " ".join(shlex.quote(a) for a in argv)
        full_cmd = f"cd {shlex.quote(self._fs_root)} && {cmd} ; echo {_RC_MARKER}=$?"
        try:
            result = await self._fs_sandbox.commands.run(full_cmd)
        except Exception:
            raise _EngineUnavailable()
        rc, out = self._split_rc(result.stdout or "")
        if rc not in (0, 1):
            raise _EngineUnavailable()
        return self._parse_grep(out)

    def _parse_grep(self, stdout_text: str) -> List[Record]:
        """Parse `grep -rn` output lines of the form `<path>:<lineno>:<content>`.

        Splits on the first two colons (line numbers are always digits) — the same
        approach the previous awk formatter used, and identical across GNU grep (the
        e2b sandbox) and BSD grep (macOS), neither of which emits a NUL separator for
        matching lines. Paths containing a ':' are rare and skipped."""
        records: List[Record] = []
        for line in stdout_text.split("\n"):
            colon1 = line.find(":")
            if colon1 < 0:
                continue
            colon2 = line.find(":", colon1 + 1)
            if colon2 < 0:
                continue
            num_str = line[colon1 + 1 : colon2]
            if not num_str.isdigit():
                continue
            path = line[:colon1]
            content = line[colon2 + 1 :]
            records.append((self._normalize_path(path), int(num_str), content))
        return records

    def _drop_oversize_local(self, records: List[Record]) -> List[Record]:
        sizes: dict = {}
        kept: List[Record] = []
        for path, line_num, raw in records:
            if path not in sizes:
                try:
                    sizes[path] = self.filesystem.get_path(path).stat().st_size
                except Exception:
                    sizes[path] = 0
            if sizes[path] <= _MAX_FILE_SIZE:
                kept.append((path, line_num, raw))
        return kept

    # ------------------------------------------------------------------
    # Python (in-process) engine — the no-binary fallback (e.g. Windows)
    # ------------------------------------------------------------------

    async def _run_python(
        self, pattern: str, file_pattern: str, case_sensitive: bool, literal: bool, regex
    ) -> List[Record]:
        files_with_info = await self._get_files_batch_local(file_pattern)
        records: List[Record] = []
        for file_path, file_size in files_with_info:
            if file_size > _MAX_FILE_SIZE:
                continue
            if self._is_likely_binary_by_extension(file_path):
                continue
            try:
                content = self.filesystem.read_text(file_path)
                if "\x00" in content[:1024]:
                    continue
            except Exception:
                continue
            norm = self._normalize_path(file_path)
            for line_num, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    records.append((norm, line_num, line))
        return records

    # ------------------------------------------------------------------
    # Shared output formatting
    # ------------------------------------------------------------------

    def _format_results(self, records: List[Record], original_pattern: str) -> str:
        """Group records by file (first-seen order), apply the display caps, and
        render the markdown output. `(N matches)` is the matching-line count."""
        files: dict = {}
        limit_hit = False
        for path, line_num, raw in records:
            if path in files:
                files[path].append((line_num, raw))
            elif len(files) < _MAX_FILES:
                files[path] = [(line_num, raw)]
            else:
                limit_hit = True

        if not files:
            return f"No matches found for pattern '{original_pattern}'"

        blocks = []
        for path, hits in files.items():
            count = len(hits)
            shown = []
            for line_num, raw in hits[:_MAX_LINES_PER_FILE]:
                content = raw.strip()
                if len(content) > _MAX_LINE_LENGTH:
                    content = content[:_MAX_LINE_LENGTH] + "..."
                shown.append(f"  Line {line_num}: {content}")
            block = f"- **{path}** ({count} matches)\n" + "\n".join(shown)
            if count > _MAX_LINES_PER_FILE:
                block += f"\n  ... and {count - _MAX_LINES_PER_FILE} more matches"
            blocks.append(block)

        output = f"# Search Results for '{original_pattern}'\n\n"
        if limit_hit:
            output += (
                f"⚠️ **Note:** Showing only the first {_MAX_FILES} results. There are more matches in the codebase.\n\n"
            )
        output += "\n\n".join(blocks)
        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_rc(stdout_text: str) -> Tuple[Optional[int], str]:
        """Split the trailing `__KOLEGA_RG_RC=N` marker line off sandbox output."""
        marker = _RC_MARKER + "="
        rc: Optional[int] = None
        kept = []
        for line in stdout_text.split("\n"):
            if line.startswith(marker):
                try:
                    rc = int(line[len(marker) :].strip())
                except ValueError:
                    rc = None
            else:
                kept.append(line)
        return rc, "\n".join(kept)

    @staticmethod
    def _normalize_path(path: str) -> str:
        path = path.replace("\\", "/")
        if path.startswith("./"):
            path = path[2:]
        return path

    async def _get_files_batch_local(self, file_pattern: str) -> List[Tuple[str, int]]:
        """
        Get files for local filesystem with minimal stat calls.
        """
        # Get files using glob
        if file_pattern == "*":
            files = self.filesystem.glob("**/*")
        else:
            files = self.filesystem.glob(f"**/{file_pattern}")

        files_with_info = []
        for file_path in files:
            # Quick path-based exclusions
            if self._should_exclude_by_path(file_path):
                continue

            # Check if it's a file and get size in one stat call
            try:
                if self.filesystem.is_file(file_path):
                    path_obj = self.filesystem.get_path(file_path)
                    # Use the existing _should_exclude_file logic from base class
                    if self._should_exclude_file(path_obj):
                        continue

                    # Get size from stat
                    try:
                        stat_info = path_obj.stat()
                        size = stat_info.st_size
                    except Exception:
                        size = 0

                    files_with_info.append((file_path, size))
            except Exception:
                continue

        return files_with_info

    def _should_exclude_by_path(self, file_path: str) -> bool:
        """
        Check if file should be excluded based on its path alone.
        """
        path_parts = Path(file_path).parts
        return any(part in self.EXCLUDE_DIRS for part in path_parts)

    def _is_likely_binary_by_extension(self, file_path: str) -> bool:
        """
        Quick binary check based on file extension.
        """
        return Path(file_path).suffix.lower() in self.BINARY_EXTENSIONS
