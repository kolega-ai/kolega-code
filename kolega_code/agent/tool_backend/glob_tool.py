import asyncio
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import List, Tuple, Literal
import shlex

from kolega_code.services.workspace_scan import ScanLimits, compile_workspace_glob, scan_workspace

from .base_tool import BaseTool

FileType = Literal["f", "d"]
FileRow = Tuple[str, FileType, int, int]  # (path, type, size_bytes, mtime_epoch)

_BROAD_ROOT_EXCLUDE_DIRS = {
    ".Trash",
    ".cache",
    ".cargo",
    ".conda",
    ".docker",
    ".dropbox",
    ".gem",
    ".local",
    ".npm",
    ".nuget",
    ".nvm",
    ".pyenv",
    ".rustup",
    ".wine",
    "Applications",
    "Applications (Parallels)",
    "Caches",
    "CloudStorage",
    "Containers",
    "Developer",
    "Group Containers",
    "Metadata",
    "Mobile Documents",
    "Movies",
    "Music",
    "Parallels",
    "Pictures",
    "calibre_library",
    "models",
}


@dataclass
class GlobSearchResult:
    rows: List[FileRow]
    observed_items: int
    complete: bool
    stop_reason: str | None = None
    visited_entries: int = 0
    elapsed_seconds: float = 0.0


class GlobTool(BaseTool):
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

    MAX_RESULTS = 128
    MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

    async def find_files_by_pattern(
        self, pattern: str, include_directories: bool = True, show_details: bool = True
    ) -> str:
        """
        Find files matching a glob pattern in the project directory.

        Args:
            pattern: Glob pattern to match files (e.g., "*.py", "src/**/*.js")
            include_directories: Whether to include directories in results (default: False)
            show_details: Whether to show file details like size and modification time (default: True)

        Returns:
            Markdown formatted list of files matching the pattern, limited to MAX_RESULTS

        Raises:
            Exception: If any error occurs during the search operation
        """
        try:
            await self.log_info(f"Searching for files matching pattern: '{pattern}'", sender=self.caller.agent_name)

            normalized_pattern = self._normalize_pattern(pattern)

            if hasattr(self.filesystem, "sandbox"):
                search = await self._search_files_sandbox(normalized_pattern, include_directories, self.MAX_RESULTS)
            else:
                search = await self._search_files_local(normalized_pattern, include_directories, self.MAX_RESULTS)

            if not search.complete:
                await self.log_warning(
                    "File scan stopped before completion "
                    f"(reason={search.stop_reason}, visited={search.visited_entries}, "
                    f"elapsed={search.elapsed_seconds:.2f}s)",
                    sender=self.caller.agent_name,
                )

            if search.observed_items == 0 and search.complete:
                return f"No files found matching pattern: '{normalized_pattern}'"

            return self._format_results(search, show_details, normalized_pattern)

        except Exception as e:
            error_msg = f"Error finding files: {str(e)}"
            await self.log_error(error_msg, sender=self.caller.agent_name)
            return f"Error: {error_msg}"

    def _normalize_pattern(self, pattern: str) -> str:
        p = (pattern or "").strip()
        if p.startswith("/"):
            p = p[1:]
        # Bare filename → recursive filename search
        if all(ch not in p for ch in ("*", "?", "[")) and "/" not in p:
            p = f"**/{p}"
        return p

    @staticmethod
    def _is_broad_local_root(root: Path) -> bool:
        resolved = root.expanduser().resolve()
        return resolved == Path.home().resolve() or resolved == Path(resolved.anchor)

    def _search_exclude_dirs(self, root: Path) -> set[str]:
        excluded = set(self.EXCLUDE_DIRS)
        if self._is_broad_local_root(root):
            excluded.update(_BROAD_ROOT_EXCLUDE_DIRS)
        return excluded

    @staticmethod
    def _find_scope(pattern: str) -> tuple[str, bool]:
        """Return the fixed start directory and whether traversal is recursive."""
        parts = PurePosixPath(pattern.replace("\\", "/").lstrip("/")).parts
        fixed: list[str] = []
        for part in parts:
            if any(character in part for character in ("*", "?", "[")):
                break
            fixed.append(part)
        if len(fixed) == len(parts) and fixed:
            fixed.pop()
        start = "./" + "/".join(fixed) if fixed else "."
        return start, "**" in parts

    async def _search_files_local(self, pattern: str, include_directories: bool, limit: int) -> GlobSearchResult:
        if os.name != "nt" and shutil.which("find") is not None:
            return await self._search_files_local_process(pattern, include_directories, limit)

        # Windows/no-find fallback: still off-loop and cancellable, with generous
        # emergency bounds rather than an ordinary-result correctness cutoff.
        root = Path(getattr(self.filesystem, "root_path", self.project_path))
        outcome = await scan_workspace(
            root,
            pattern=pattern,
            include_files=True,
            include_directories=include_directories,
            exclude_directories=frozenset(self.EXCLUDE_DIRS),
            binary_extensions=frozenset(self.BINARY_EXTENSIONS),
            max_file_size=self.MAX_FILE_SIZE_BYTES,
            limits=ScanLimits(
                timeout_seconds=300.0,
                max_entries=5_000_000,
                max_results=limit + 1,
            ),
        )
        rows: List[FileRow] = [
            (path.path, "d" if path.is_dir else "f", path.size, path.modified_time) for path in outcome.paths[:limit]
        ]
        return GlobSearchResult(
            rows=rows,
            observed_items=len(outcome.paths),
            complete=outcome.complete,
            stop_reason=outcome.stop_reason,
            visited_entries=outcome.visited_entries,
            elapsed_seconds=outcome.elapsed_seconds,
        )

    async def _search_files_local_process(
        self, pattern: str, include_directories: bool, limit: int
    ) -> GlobSearchResult:
        """Search with a cancellable subprocess so traversal never blocks Textual."""
        root = Path(getattr(self.filesystem, "root_path", self.project_path))
        excluded = sorted(self._search_exclude_dirs(root))
        start_path, recursive = self._find_scope(pattern)
        if not (root / start_path).exists():
            return GlobSearchResult([], 0, True)
        args = ["find", start_path]
        if not recursive:
            args.extend(["-maxdepth", "1"])
        if excluded:
            args.append("(")
            for index, directory in enumerate(excluded):
                if index:
                    args.append("-o")
                args.extend(["-name", directory])
            args.extend([")", "-type", "d", "-prune", "-o"])
        basename_pattern = pattern.replace("\\", "/").rsplit("/", 1)[-1] or "*"
        args.extend(["-name", basename_pattern, "-print0"])

        matcher = compile_workspace_glob(pattern)
        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=root,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        assert proc.stdout is not None
        rows: List[FileRow] = []
        observed = 0
        try:
            while True:
                try:
                    raw_path = await proc.stdout.readuntil(b"\0")
                except asyncio.IncompleteReadError:
                    break
                relative = raw_path[:-1].decode("utf-8", "surrogateescape")
                if relative.startswith("./"):
                    relative = relative[2:]
                if not relative:
                    continue

                row = self._local_file_row(root, relative, include_directories, matcher)
                if row is None:
                    continue

                observed += 1
                if len(rows) < limit:
                    rows.append(row)
                if observed > limit:
                    await self._stop_local_find_process(proc)
                    return GlobSearchResult(
                        rows,
                        observed,
                        False,
                        "result_limit",
                        elapsed_seconds=time.monotonic() - started,
                    )

            return_code = await proc.wait()
        except asyncio.CancelledError:
            await self._stop_local_find_process(proc)
            raise

        return GlobSearchResult(
            rows,
            observed,
            return_code == 0,
            None if return_code == 0 else "command_error",
            elapsed_seconds=time.monotonic() - started,
        )

    def _local_file_row(self, root: Path, relative: str, include_directories: bool, matcher) -> FileRow | None:
        full_path = root / relative
        try:
            is_directory = full_path.is_dir()
            is_file = full_path.is_file()
        except OSError:
            return None
        candidate = relative + "/" if is_directory else relative
        if not matcher.match_file(candidate):
            return None
        if is_directory:
            if not include_directories:
                return None
            try:
                modified_time = int(full_path.stat().st_mtime)
            except OSError:
                modified_time = 0
            return (relative, "d", 0, modified_time)
        if is_file:
            if full_path.suffix.lower() in self.BINARY_EXTENSIONS:
                return None
            try:
                stat_result = full_path.stat()
            except OSError:
                return None
            if stat_result.st_size > self.MAX_FILE_SIZE_BYTES:
                return None
            return (relative, "f", int(stat_result.st_size), int(stat_result.st_mtime))
        return None

    @staticmethod
    async def _stop_local_find_process(proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except (ProcessLookupError, asyncio.TimeoutError):
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    return
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass

    async def _search_files_sandbox(self, pattern: str, include_directories: bool, limit: int) -> GlobSearchResult:
        root = shlex.quote(getattr(self.filesystem, "root_path", "."))
        include_flag = "1" if include_directories else "0"

        # Build prune expression for find
        exclude_list = " -o ".join([f"-name {shlex.quote(d)}" for d in sorted(self.EXCLUDE_DIRS)])
        prune = f"\\( {exclude_list} \\) -type d -prune -o"

        script = f"""bash -O globstar -c '
set -euo pipefail
cd {root}

pattern={shlex.quote(pattern)}
max_results={limit}
scan_limit=$((max_results + 1))
include_dirs={include_flag}

run_find() {{
  case "$pattern" in
    **"**/"**)
      name_pat="${{pattern##*/}}"
      find . {prune} -name "$name_pat" -print
      ;;
    *"/"*)
      dir_part="${{pattern%/*}}"
      name_pat="${{pattern##*/}}"
      base_dir="${{dir_part##*/}}"
      case " {" ".join(sorted(self.EXCLUDE_DIRS))} " in *" $base_dir "*) exit 0;; esac
      find "$dir_part" -maxdepth 1 -name "$name_pat" -print
      ;;
    *)
      find . {prune} -name "$pattern" -print
      ;;
  esac
}}

# Emit TSV: path \t type(f|d) \t size(bytes) \t mtime(epoch)
total_items=0
while IFS= read -r p; do
  [[ -z "$p" ]] && continue
  if [[ -f "$p" ]]; then
    sz=$(stat -c %s "$p" 2>/dev/null || echo 0)
    [[ "$sz" -gt {self.MAX_FILE_SIZE_BYTES} ]] && continue
    case "${{p##*.}}" in
      pyc|so|dll|exe|bin|jar|war|jpg|jpeg|png|gif|bmp|ico|svg|pdf|zip|tar|gz|tgz|rar|7z|mp3|mp4|avi|mov|mkv|wav|o|obj|class|binary|wasm|node) continue;;
    esac
    mt=$(stat -c %Y "$p" 2>/dev/null || echo 0)
    printf "%s\tf\t%s\t%s\n" "$p" "$sz" "$mt"
  elif [[ -d "$p" ]] && [[ "$include_dirs" == "1" ]]; then
    mt=$(stat -c %Y "$p" 2>/dev/null || echo 0)
    printf "%s\td\t0\t%s\n" "$p" "$mt"
  else
    continue
  fi
  total_items=$((total_items + 1))
  [[ "$total_items" -ge "$scan_limit" ]] && break
done < <(run_find | sed "s#^\\./##")

echo "__TOTAL__ $total_items"
if [[ "$total_items" -ge "$scan_limit" ]]; then
  echo "__COMPLETE__ 0 result_limit"
else
  echo "__COMPLETE__ 1"
fi
'"""

        sandbox = getattr(self.filesystem, "sandbox", None)
        assert sandbox is not None, "sandbox filesystem required for _search_files_sandbox"
        result = await sandbox.commands.run(script, timeout=0)
        if result.exit_code != 0:
            return GlobSearchResult([], 0, False, "command_error")

        lines = [ln for ln in (result.stdout or "").splitlines() if ln.strip()]
        rows: List[FileRow] = []
        total_items = 0
        complete = True
        stop_reason = None

        for ln in lines:
            if ln.startswith("__TOTAL__ "):
                try:
                    total_items = int(ln.split()[-1])
                except Exception:
                    total_items = len(rows)
                continue
            if ln.startswith("__COMPLETE__ "):
                fields = ln.split()
                complete = len(fields) >= 2 and fields[1] == "1"
                stop_reason = fields[2] if len(fields) >= 3 else None
                continue
            parts = ln.split("\t")
            if len(parts) != 4:
                continue
            path_str, type_str, size_str, mtime_str = parts
            # Additional filtering matching local rules
            if type_str == "f":
                if Path(path_str).suffix.lower() in self.BINARY_EXTENSIONS:
                    continue
                try:
                    size_val = int(size_str or "0")
                    if size_val > self.MAX_FILE_SIZE_BYTES:
                        continue
                except Exception:
                    continue
            rows.append((path_str, "f" if type_str == "f" else "d", int(size_str or "0"), int(float(mtime_str or "0"))))

        return GlobSearchResult(
            rows=rows[:limit],
            observed_items=total_items,
            complete=complete,
            stop_reason=stop_reason,
            elapsed_seconds=0.0,
        )

    def _format_results(self, search: GlobSearchResult, show_details: bool, pattern: str) -> str:
        results: List[str] = [f"# Files Matching '{pattern}'"]
        if search.stop_reason == "result_limit":
            results.append(
                f"\nFound at least {search.observed_items} matching items (showing first {self.MAX_RESULTS})\n"
            )
            results.append(f"⚠️ **Note:** Search stopped after collecting the first {self.MAX_RESULTS} results.\n")
        elif not search.complete:
            results.append(f"\nFound {search.observed_items} matching items before the search stopped\n")
            results.append(
                f"⚠️ **Incomplete search:** stopped because of {search.stop_reason or 'a scan limit'}. "
                "Narrow the pattern or project root for exhaustive results.\n"
            )
        else:
            results.append(f"\nFound {search.observed_items} matching items\n")

        by_directory: dict[str, List[FileRow]] = {}
        for path_str, ftype, size_bytes, mtime_epoch in search.rows:
            parent = self.filesystem.get_parent(path_str) or ""
            by_directory.setdefault(parent, []).append((path_str, ftype, size_bytes, mtime_epoch))

        for directory in sorted(by_directory.keys()):
            # Match original behavior: only empty string maps to Root Directory; '.' prints as './'
            if directory:
                results.append(f"## {directory}/")
            else:
                results.append("## Root Directory")

            for path_str, ftype, size_bytes, mtime_epoch in sorted(by_directory[directory]):
                filename = self.filesystem.get_name(path_str)
                if ftype == "d":
                    item_type = "📁 Directory"
                    size_text = "unknown items"
                else:
                    item_type = "📄 File"
                    try:
                        if size_bytes < 1024:
                            size_text = f"{size_bytes} bytes"
                        elif size_bytes < 1024 * 1024:
                            size_text = f"{size_bytes / 1024:.1f} KB"
                        else:
                            size_text = f"{size_bytes / (1024 * 1024):.1f} MB"
                    except Exception:
                        size_text = "unknown size"

                line = f"- **{filename}** ({item_type})"
                if show_details:
                    try:
                        mod_time = datetime.fromtimestamp(mtime_epoch)
                        mod_time_str = mod_time.strftime("%Y-%m-%d %H:%M:%S")
                        line += f"\n  - Size: {size_text}"
                        line += f"\n  - Modified: {mod_time_str}"
                        if ftype == "f":
                            ext = Path(filename).suffix
                            if ext:
                                line += f"\n  - Type: {ext} file"
                    except Exception:
                        line += f"\n  - Size: {size_text}"
                results.append(line)
            results.append("")

        return "\n".join(results)
