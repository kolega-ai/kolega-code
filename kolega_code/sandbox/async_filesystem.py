"""Async FileSystem implementation for sandbox environments."""

import os
import base64
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import asyncio
import nest_asyncio

from ..services.file_system import FileSystem


class AsyncSandboxFileSystem(FileSystem):
    """FileSystem implementation that operates within an async sandbox.

    This class provides a synchronous interface to an async sandbox by using
    nest_asyncio to allow running async operations in sync contexts.
    """

    def __init__(self, sandbox: Any, root_path: str = "/home/user/workspace"):
        """
        Initialize async sandbox filesystem.

        Args:
            sandbox: The async sandbox instance (e.g., AsyncE2B Sandbox)
            root_path: Root path within the sandbox
        """
        self.sandbox = sandbox
        self.root_path = root_path
        self._nest_asyncio_applied = False

    def _run_async(self, coro):
        """Run an async coroutine in a sync context using nest_asyncio."""
        try:
            # Try to get the current running loop
            loop = asyncio.get_running_loop()

            # Apply nest_asyncio only once and only when needed
            if not self._nest_asyncio_applied:
                try:
                    # Only apply if we're not in a uvloop context
                    if not hasattr(loop, "__module__") or "uvloop" not in loop.__module__:
                        nest_asyncio.apply()
                        self._nest_asyncio_applied = True
                except:
                    # If patching fails, we'll try alternative approaches
                    pass

        except RuntimeError:
            # No running loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # Apply nest_asyncio for the new loop
            if not self._nest_asyncio_applied:
                nest_asyncio.apply()
                self._nest_asyncio_applied = True

        # Use nest_asyncio to run the coroutine if it was applied
        if self._nest_asyncio_applied:
            return loop.run_until_complete(coro)
        else:
            # If we couldn't apply nest_asyncio (e.g., uvloop),
            # we need to use a different approach
            import concurrent.futures

            # Create a new event loop in a separate thread
            def run_in_new_loop():
                # Lazy import to avoid circular dependency
                from kolega_code.sandbox.event_loop import cleanup_event_loop

                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    # Proper cleanup following asyncio.Runner pattern
                    cleanup_event_loop(new_loop)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_new_loop)
                return future.result()

    def _resolve_path(self, path: str) -> str:
        """Resolve path relative to root."""
        if os.path.isabs(path):
            return path
        if path == ".":
            return self.root_path
        return os.path.join(self.root_path, path)

    # Synchronous methods wrapping async E2B API
    def open(self, path: str, mode: str = "r", encoding: Optional[str] = None) -> Any:
        """Open is not directly supported in sandbox - use read/write methods instead."""
        raise NotImplementedError("Direct file handles not supported in sandbox. Use read_text/write_text instead.")

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read text from file."""
        full_path = self._resolve_path(path)
        try:

            async def _read():
                return await self.sandbox.files.read(full_path)

            content = self._run_async(_read())
            if isinstance(content, bytes):
                return content.decode(encoding)
            return content
        except Exception as e:
            raise FileNotFoundError(f"Could not read file {path}: {e}")

    def read_bytes(self, path: str) -> bytes:
        """Read bytes from file."""
        full_path = self._resolve_path(path)
        try:
            # For binary data, use base64 encoding via shell commands
            # to avoid E2B files API text encoding issues
            async def _read_bytes():
                result = await self.sandbox.commands.run(f"base64 {full_path}")
                if result.exit_code != 0:
                    raise FileNotFoundError(f"Could not read file {path}")
                # Decode the base64 content
                return base64.b64decode(result.stdout.strip())

            return self._run_async(_read_bytes())

        except Exception as e:
            raise FileNotFoundError(f"Could not read file {path}: {e}")

    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
        """Write text to file."""
        full_path = self._resolve_path(path)
        try:

            async def _write():
                # Ensure parent directory exists
                parent_dir = os.path.dirname(full_path)
                if parent_dir != self.root_path:
                    await self.sandbox.commands.run(f"mkdir -p {parent_dir}")

                await self.sandbox.files.write(full_path, content)

            self._run_async(_write())
        except Exception as e:
            raise OSError(f"Could not write file {path}: {e}")

    def write_bytes(self, path: str, content: bytes) -> None:
        """Write bytes to file."""
        full_path = self._resolve_path(path)
        try:

            async def _write_bytes():
                # Ensure parent directory exists
                parent_dir = os.path.dirname(full_path)
                if parent_dir != self.root_path:
                    await self.sandbox.commands.run(f"mkdir -p {parent_dir}")

                # For binary data, use base64 encoding and write via shell commands
                # to avoid E2B files API text encoding issues
                encoded_content = base64.b64encode(content).decode("ascii")

                # Write the base64 encoded content and decode it
                result = await self.sandbox.commands.run(f"echo '{encoded_content}' | base64 -d > {full_path}")

                if result.exit_code != 0:
                    raise OSError(f"Failed to write binary file {path}: {result.stderr}")

            self._run_async(_write_bytes())

        except Exception as e:
            raise OSError(f"Could not write file {path}: {e}")

    def exists(self, path: str) -> bool:
        """Check if path exists."""
        full_path = self._resolve_path(path)
        try:

            async def _exists():
                result = await self.sandbox.commands.run(f"test -e {full_path}")
                return result.exit_code == 0

            return self._run_async(_exists())
        except:
            return False

    def is_file(self, path: str) -> bool:
        """Check if path is a file."""
        full_path = self._resolve_path(path)
        try:

            async def _is_file():
                result = await self.sandbox.commands.run(f"test -f {full_path}")
                return result.exit_code == 0

            return self._run_async(_is_file())
        except:
            return False

    def is_dir(self, path: str) -> bool:
        """Check if path is a directory."""
        full_path = self._resolve_path(path)
        try:

            async def _is_dir():
                result = await self.sandbox.commands.run(f"test -d {full_path}")
                return result.exit_code == 0

            return self._run_async(_is_dir())
        except:
            return False

    def stat(self, path: str) -> Dict[str, Any]:
        """Get file statistics."""
        full_path = self._resolve_path(path)
        try:

            async def _stat():
                # Use stat command to get file info
                result = await self.sandbox.commands.run(f"stat -c '%s %Y %Z' {full_path}")
                if result.exit_code != 0:
                    raise FileNotFoundError(f"File not found: {path}")

                size, mtime, ctime = result.stdout.strip().split()

                return {
                    "size": int(size),
                    "modified_time": int(mtime),
                    "created_time": int(ctime),
                    "is_file": await self._async_is_file(full_path),
                    "is_directory": await self._async_is_dir(full_path),
                }

            return self._run_async(_stat())
        except Exception as e:
            raise OSError(f"Could not stat {path}: {e}")

    async def _async_is_file(self, full_path: str) -> bool:
        """Async helper to check if path is a file."""
        try:
            result = await self.sandbox.commands.run(f"test -f {full_path}")
            return result.exit_code == 0
        except Exception:
            # E2B throws exception on non-zero exit codes
            return False

    async def _async_is_dir(self, full_path: str) -> bool:
        """Async helper to check if path is a directory."""
        try:
            result = await self.sandbox.commands.run(f"test -d {full_path}")
            return result.exit_code == 0
        except Exception:
            # E2B throws exception on non-zero exit codes
            return False

    def mkdir(self, path: str, parents: bool = False, exist_ok: bool = False) -> None:
        """Create directory."""
        full_path = self._resolve_path(path)

        # Check if directory already exists
        if self.exists(path):
            if not exist_ok:
                raise FileExistsError(f"Directory already exists: {path}")
            return

        try:

            async def _mkdir():
                if parents:
                    result = await self.sandbox.commands.run(f"mkdir -p {full_path}")
                else:
                    result = await self.sandbox.commands.run(f"mkdir {full_path}")

                if result.exit_code != 0:
                    raise OSError(f"Failed to create directory {path}: {result.stderr}")

            self._run_async(_mkdir())
        except Exception as e:
            raise OSError(f"Could not create directory {path}: {e}")

    def remove(self, path: str, missing_ok: bool = False) -> None:
        """Remove file."""
        full_path = self._resolve_path(path)

        if not self.exists(path):
            if not missing_ok:
                raise FileNotFoundError(f"File not found: {path}")
            return

        try:

            async def _remove():
                result = await self.sandbox.commands.run(f"rm -f {full_path}")
                if result.exit_code != 0:
                    raise OSError(f"Failed to remove file {path}: {result.stderr}")

            self._run_async(_remove())
        except Exception as e:
            raise OSError(f"Could not remove file {path}: {e}")

    def rmdir(self, path: str) -> None:
        """Remove empty directory."""
        full_path = self._resolve_path(path)
        try:

            async def _rmdir():
                result = await self.sandbox.commands.run(f"rmdir {full_path}")
                if result.exit_code != 0:
                    raise OSError(f"Failed to remove directory {path}: {result.stderr}")

            self._run_async(_rmdir())
        except Exception as e:
            raise OSError(f"Could not remove directory {path}: {e}")

    def rmtree(self, path: str) -> None:
        """Remove directory tree."""
        full_path = self._resolve_path(path)
        try:

            async def _rmtree():
                result = await self.sandbox.commands.run(f"rm -rf {full_path}")
                if result.exit_code != 0:
                    raise OSError(f"Failed to remove directory tree {path}: {result.stderr}")

            self._run_async(_rmtree())
        except Exception as e:
            raise OSError(f"Could not remove directory tree {path}: {e}")

    def listdir(self, path: str) -> List[str]:
        """List directory contents."""
        full_path = self._resolve_path(path)

        if not self.is_dir(path):
            raise NotADirectoryError(f"Not a directory: {path}")

        try:

            async def _listdir():
                result = await self.sandbox.commands.run(f"ls -1 {full_path}")
                if result.exit_code != 0:
                    raise FileNotFoundError(f"Directory not found: {path}")

                if not result.stdout.strip():
                    return []

                return [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]

            return self._run_async(_listdir())
        except Exception as e:
            raise OSError(f"Could not list directory {path}: {e}")

    def iterdir(self, path: str) -> Iterator[str]:
        """Iterate directory contents."""
        return iter(self.listdir(path))

    def glob(self, pattern: str) -> List[str]:
        """Find paths matching pattern."""
        try:

            async def _glob():
                # Handle different types of glob patterns
                if pattern.startswith("**/"):
                    # Recursive pattern like **/*.py or **/*
                    remaining_pattern = pattern[3:]  # Remove '**/'

                    if remaining_pattern == "*":
                        # Pattern is **/* - find all files and directories recursively
                        result = await self.sandbox.commands.run(
                            f"cd {self.root_path} && find . -mindepth 1 2>/dev/null | sed 's|^./||' | sort"
                        )
                    else:
                        # Pattern like **/*.py - find files matching pattern recursively
                        if "*" in remaining_pattern or "?" in remaining_pattern:
                            # Use find with -name for pattern matching
                            result = await self.sandbox.commands.run(
                                f"cd {self.root_path} && find . -name '{remaining_pattern}' 2>/dev/null | sed 's|^./||' | sort"
                            )
                        else:
                            # Exact filename search recursively
                            result = await self.sandbox.commands.run(
                                f"cd {self.root_path} && find . -name '{remaining_pattern}' 2>/dev/null | sed 's|^./||' | sort"
                            )

                elif "*" in pattern or "?" in pattern:
                    # Simple glob pattern like *.py or test*.txt
                    if "/" in pattern:
                        # Pattern has directory component like subdir/*.txt
                        parent_dir = os.path.dirname(pattern)
                        filename_pattern = os.path.basename(pattern)
                        result = await self.sandbox.commands.run(
                            f"cd {self.root_path} && find {parent_dir} -maxdepth 1 -name '{filename_pattern}' 2>/dev/null | sort"
                        )
                    else:
                        # Simple pattern like *.py in current directory
                        result = await self.sandbox.commands.run(
                            f"cd {self.root_path} && find . -maxdepth 1 -name '{pattern}' 2>/dev/null | sed 's|^./||' | sort"
                        )

                else:
                    # No wildcards - check if exact path exists
                    if self.exists(pattern):
                        return [pattern]
                    else:
                        return []

                # Process the result
                if result.exit_code == 0 and result.stdout.strip():
                    paths = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
                    # Filter out empty strings and current directory
                    return [path for path in paths if path and path != "."]
                else:
                    return []

            return self._run_async(_glob())

        except Exception as e:
            # If any error occurs, return empty list to match expected behavior
            return []

    def is_binary_file(self, path: str) -> bool:
        """Check if file is binary."""
        full_path = self._resolve_path(path)
        try:

            async def _is_binary():
                # Use file command to detect binary
                result = await self.sandbox.commands.run(f"file -b --mime {full_path}")
                return result.exit_code == 0 and "charset=binary" in result.stdout

            return self._run_async(_is_binary())
        except:
            return False

    def get_name(self, path: str) -> str:
        """Get basename of path."""
        return os.path.basename(path)

    def get_suffix(self, path: str) -> str:
        """Get file extension."""
        return os.path.splitext(path)[1]

    def get_parent(self, path: str) -> str:
        """Get parent directory."""
        parent = os.path.dirname(path)
        # Return "." for root-level files to match LocalFileSystem behavior
        return parent if parent else "."

    def get_parents(self, path: str) -> List[str]:
        """Get all parent directories."""
        parents = []
        current = os.path.dirname(path)
        while current and current != "/":
            parents.append(current)
            current = os.path.dirname(current)
        return parents

    def relative_to(self, path: str, other: str) -> str:
        """Get relative path."""
        return os.path.relpath(path, other)

    def join_path(self, *parts: str) -> str:
        """Join path components."""
        return os.path.join(*parts)

    def is_absolute(self, path: str) -> bool:
        """Check if path is absolute."""
        return os.path.isabs(path)

    def get_path(self, path: str) -> Path:
        """Get a Path object for the given path."""
        resolved = self._resolve_path(path)
        # Since _resolve_path returns a string for sandbox filesystem,
        # we need to create a Path object from it
        return Path(resolved)
