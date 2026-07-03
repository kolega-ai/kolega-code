from pathlib import Path

from .base_tool import BaseTool


class ListDirectoryTool(BaseTool):
    async def list_directory(self, path: str = "") -> str:
        """
        List files and directories at the specified path.

        Args:
            path: Path to list. Relative to the project root is preferred; an absolute path is also accepted.

        Returns:
            Markdown formatted list of files and directories with details

        Raises:
            NotADirectoryError: If the path is not a directory
        """
        if not self.filesystem.exists(path):
            raise FileNotFoundError(f"Directory not found: {path}")

        if not self.filesystem.is_directory(path):
            raise NotADirectoryError(f"Not a directory: {path}")

        # Use sandbox-specific implementation if available
        if hasattr(self.filesystem, "sandbox"):
            return await self._list_directory_sandbox(path)
        else:
            # Use the original implementation for local filesystem
            return await self._list_directory_local(path)

    async def _list_directory_sandbox(self, path: str) -> str:
        """
        Sandbox-specific implementation using a single command for efficiency.
        """
        # ``_resolve_path``/``root_path``/``sandbox`` are provided by the concrete
        # sandbox filesystem subclasses, not the ``FileSystem`` base, so access them
        # via ``getattr``. This method is only reached when ``hasattr(filesystem,
        # "sandbox")`` is true (see ``list_directory``).
        resolve = getattr(self.filesystem, "_resolve_path", None)
        full_path = resolve(path) if path and callable(resolve) else getattr(self.filesystem, "root_path", ".")

        # Use ls with detailed format to get all info in one command
        # -la: list all files with details
        # --time-style=long-iso: consistent datetime format
        # --group-directories-first: directories first
        ls_cmd = f"cd {full_path} && ls -la --time-style=long-iso --group-directories-first 2>/dev/null"

        # Also get directory item counts in one go
        # Use a more robust command that handles cases with no directories
        count_cmd = f'cd {full_path} && find . -maxdepth 1 -type d ! -name . -exec sh -c \'echo "$(basename "{{}}"):$(ls -1 "{{}}" 2>/dev/null | wc -l)"\' \\; 2>/dev/null || true'

        # Run both commands - always await since sandbox commands are always async
        sandbox = getattr(self.filesystem, "sandbox", None)
        assert sandbox is not None, "sandbox filesystem required for _list_directory_sandbox"
        ls_result = await sandbox.commands.run(ls_cmd)
        count_result = await sandbox.commands.run(count_cmd)

        if ls_result.exit_code != 0:
            raise OSError(f"Failed to list directory: {ls_result.stderr}")

        # Parse directory counts
        dir_counts = {}
        if count_result.exit_code == 0 and count_result.stdout.strip():
            for line in count_result.stdout.strip().split("\n"):
                if ":" in line:
                    dir_name, count = line.rsplit(":", 1)
                    dir_counts[dir_name.rstrip("/")] = count.strip()

        # Parse ls output
        lines_data = []
        for line in ls_result.stdout.strip().split("\n")[1:]:  # Skip "total" line
            parts = line.split(None, 8)  # Split into max 9 parts
            if len(parts) < 8:  # With long-iso format, we need at least 8 parts
                continue

            permissions = parts[0]
            # parts[1] = number of links (skip)
            # parts[2] = owner (skip)
            # parts[3] = group (skip)
            size = parts[4]
            date = parts[5]  # Date in YYYY-MM-DD format
            time = parts[6]  # Time in HH:MM format
            name = parts[7] if len(parts) == 8 else " ".join(parts[7:])  # Handle filenames with spaces

            # Skip . and ..
            if name in [".", ".."]:
                continue

            # Skip .git directory
            if name == ".git":
                continue

            # Determine if it's a directory
            is_dir = permissions.startswith("d")

            # Clean up name (remove trailing / for directories)
            clean_name = name.rstrip("/")

            # Build relative path
            if path:
                item_path = f"{path}/{clean_name}"
            else:
                item_path = clean_name

            lines_data.append(
                {
                    "name": clean_name,
                    "path": item_path,
                    "is_dir": is_dir,
                    "size": int(size) if not is_dir else dir_counts.get(clean_name, "0"),
                    "date": f"{date} {time}",
                    "permissions": permissions,
                }
            )

        # Sort: directories first, then alphabetically
        lines_data.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))

        # Build the markdown output
        return self._format_directory_listing(path, lines_data)

    async def _list_directory_local(self, path: str) -> str:
        """
        Original implementation for local filesystem.
        """
        # Get all items in the directory
        items = self.filesystem.list_directory(path)

        # Sort items: directories first, then files, alphabetically within each group
        items.sort(key=lambda x: (not self.filesystem.is_directory(x), self.filesystem.get_name(x).lower()))

        # Build data for formatting
        lines_data = []
        for item in items:
            # Skip .git directory
            if self.filesystem.get_name(item) == ".git":
                continue

            try:
                is_dir = self.filesystem.is_directory(item)

                if is_dir:
                    # For directories, count items
                    try:
                        dir_items = self.filesystem.list_directory(item)
                        size = len(dir_items)
                    except Exception:
                        size = 0
                else:
                    # For files, get size
                    try:
                        size = self.filesystem.get_size(item)
                    except Exception:
                        size = 0

                # Get modification time
                try:
                    mod_time = self.filesystem.get_modification_time(item).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    mod_time = "Unknown"

                lines_data.append(
                    {
                        "name": self.filesystem.get_name(item),
                        "path": item,
                        "is_dir": is_dir,
                        "size": size,
                        "date": mod_time,
                        "permissions": None,  # Not available in local
                    }
                )
            except Exception as e:
                # Log error but continue
                await self.log_error(f"Error processing item {item}: {e}", sender=self.caller.agent_name)
                continue

        return self._format_directory_listing(path, lines_data)

    def _format_directory_listing(self, path: str, items_data: list) -> str:
        """
        Format the directory listing data into markdown.
        """
        # Prepare the header
        if path:
            title = f"# Directory: {path}"
            parent_dir = str(Path(path).parent)
            if parent_dir and parent_dir != ".":
                navigation = f"📁 Parent Directory: {parent_dir}"
            else:
                navigation = "📁 Root Directory"
        else:
            title = "# Root Directory"
            navigation = ""

        lines = [title, ""]
        if navigation:
            lines.append(navigation)
            lines.append("")

        lines.append("| Type | Name | Size | Modified | Description |")
        lines.append("|------|------|------|----------|-------------|")

        # Process each item
        total_size = 0
        dir_count = 0
        file_count = 0

        for item_data in items_data:
            name = item_data["name"]
            is_dir = item_data["is_dir"]
            size = item_data["size"]
            date = item_data["date"]

            if is_dir:
                icon = "📁"
                size_str = f"{size} items"
                description = "Directory"
                dir_count += 1
            else:
                icon = "📄"
                size_str = self._format_size(size)
                description = self._get_file_description(name)
                file_count += 1
                total_size += size

            # Clean up and escape the name
            escaped_name = name.replace("|", "\\|")

            # Format the line
            lines.append(f"| {icon} | {escaped_name} | {size_str} | {date} | {description} |")

        # Add summary
        lines.append("")
        lines.append(f"**Summary:** {dir_count} directories, {file_count} files, {self._format_size(total_size)} total")

        return "\n".join(lines)

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

    def _get_file_description(self, filename: str) -> str:
        """Get a description based on file extension."""
        extension_map = {
            ".py": "Python Source",
            ".js": "JavaScript Source",
            ".jsx": "React JSX Source",
            ".ts": "TypeScript Source",
            ".tsx": "React TSX Source",
            ".html": "HTML Document",
            ".css": "CSS Stylesheet",
            ".json": "JSON Data",
            ".md": "Markdown Document",
            ".txt": "Text File",
            ".csv": "CSV Data",
            ".yml": "YAML Configuration",
            ".yaml": "YAML Configuration",
            ".xml": "XML Document",
            ".sql": "SQL Script",
            ".sh": "Shell Script",
            ".bat": "Batch Script",
            ".ps1": "PowerShell Script",
            ".jpg": "JPEG Image",
            ".jpeg": "JPEG Image",
            ".png": "PNG Image",
            ".gif": "GIF Image",
            ".svg": "SVG Image",
            ".pdf": "PDF Document",
            ".zip": "ZIP Archive",
            ".tar": "TAR Archive",
            ".gz": "GZIP Archive",
            ".env": "Environment Variables",
            ".dockerfile": "Docker Definition",
        }

        # Handle special files by exact filename match
        lower_name = filename.lower()
        if lower_name == "dockerfile":
            return "Docker Definition"
        elif lower_name == ".gitignore":
            return "Git Ignore Rules"
        elif lower_name == "readme.md":
            return "Project Documentation"
        elif lower_name == "license":
            return "License Information"
        elif lower_name == "requirements.txt":
            return "Python Dependencies"
        elif lower_name == "package.json":
            return "Node.js Package"
        elif lower_name in ["makefile", "makefile.in"]:
            return "Make Build Rules"

        # Get extension
        ext = Path(filename).suffix.lower()
        return extension_map.get(ext, f"{ext[1:].upper() if ext else 'Unknown'} File")
