from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Literal
import shlex

from .base_tool import BaseTool

FileType = Literal["f", "d"]
FileRow = Tuple[str, FileType, int, int]  # (path, type, size_bytes, mtime_epoch)


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
                rows, total_items, reached_limit = await self._search_files_sandbox(
                    normalized_pattern, include_directories, self.MAX_RESULTS
                )
            else:
                rows, total_items, reached_limit = await self._search_files_local(
                    normalized_pattern, include_directories, self.MAX_RESULTS
                )

            if total_items == 0:
                return f"No files found matching pattern: '{normalized_pattern}'"

            return self._format_results(rows, total_items, reached_limit, show_details, normalized_pattern)

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

    async def _search_files_local(
        self, pattern: str, include_directories: bool, limit: int
    ) -> Tuple[List[FileRow], int, bool]:
        matched_paths = self.filesystem.glob(pattern)

        filtered: List[FileRow] = []
        total_items = 0

        for rel_path in sorted(matched_paths):
            is_file = self.filesystem.is_file(rel_path)
            is_dir = self.filesystem.is_directory(rel_path)

            if not include_directories and not is_file:
                continue

            # Exclude by directory names
            parts = Path(rel_path).parts
            if any(part in self.EXCLUDE_DIRS for part in parts):
                continue

            if is_file:
                # Exclude by extension
                if Path(rel_path).suffix.lower() in self.BINARY_EXTENSIONS:
                    continue

                # Exclude by size and collect size/mtime
                try:
                    stat_info = self.filesystem.stat(rel_path)
                    size = int(stat_info.get("size", 0))
                    if size > self.MAX_FILE_SIZE_BYTES:
                        continue
                    mtime = int(stat_info.get("modified_time", 0))
                except Exception:
                    # If stat fails, skip file
                    continue

                row: FileRow = (rel_path, "f", size, mtime)
            elif is_dir and include_directories:
                try:
                    stat_info = self.filesystem.stat(rel_path)
                    mtime = int(stat_info.get("modified_time", 0))
                except Exception:
                    mtime = 0
                row = (rel_path, "d", 0, mtime)
            else:
                continue

            total_items += 1
            if len(filtered) < limit:
                filtered.append(row)

        reached_limit = total_items > limit
        return filtered, total_items, reached_limit

    async def _search_files_sandbox(
        self, pattern: str, include_directories: bool, limit: int
    ) -> Tuple[List[FileRow], int, bool]:
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

matches=$(run_find | sed "s#^\\./##" | sort)
total_items=$(printf "%s\\n" "$matches" | sed "/^$/d" | wc -l | tr -d " ")
limited=$(printf "%s\\n" "$matches" | sed "/^$/d" | head -n "$max_results")

# Emit TSV: path \t type(f|d) \t size(bytes) \t mtime(epoch)
while IFS= read -r p; do
  [[ -z "$p" ]] && continue
  if [[ -f "$p" ]]; then
    sz=$(stat -c %s "$p" 2>/dev/null || echo 0)
    mt=$(stat -c %Y "$p" 2>/dev/null || echo 0)
    printf "%s\tf\t%s\t%s\n" "$p" "$sz" "$mt"
  elif [[ -d "$p" ]] && [[ "$include_dirs" == "1" ]]; then
    mt=$(stat -c %Y "$p" 2>/dev/null || echo 0)
    printf "%s\td\t0\t%s\n" "$p" "$mt"
  fi
done <<< "$limited"

echo "__TOTAL__ $total_items"
'"""

        result = await self.filesystem.sandbox.commands.run(script)
        if result.exit_code != 0:
            return [], 0, False

        lines = [ln for ln in (result.stdout or "").splitlines() if ln.strip()]
        rows: List[FileRow] = []
        total_items = 0

        for ln in lines:
            if ln.startswith("__TOTAL__ "):
                try:
                    total_items = int(ln.split()[-1])
                except Exception:
                    total_items = len(rows)
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

        reached_limit = total_items > limit
        return rows, total_items, reached_limit

    def _format_results(
        self, rows: List[FileRow], total_items: int, reached_limit: bool, show_details: bool, pattern: str
    ) -> str:
        results: List[str] = [f"# Files Matching '{pattern}'"]
        if reached_limit:
            results.append(f"\nFound {total_items} matching items (showing first {self.MAX_RESULTS})\n")
            results.append(f"⚠️ **Note:** Displaying only the first {self.MAX_RESULTS} of {total_items} results.\n")
        else:
            results.append(f"\nFound {total_items} matching items\n")

        by_directory: dict[str, List[FileRow]] = {}
        for path_str, ftype, size_bytes, mtime_epoch in rows:
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
