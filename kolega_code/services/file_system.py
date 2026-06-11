import shutil
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union, Iterator, ContextManager
from datetime import datetime


class FileSystemPath:
    """
    A path-like object that provides filesystem operations through the FileSystem interface.
    This allows for more natural path operations while still going through the abstraction layer.
    """

    def __init__(self, filesystem: "FileSystem", path: str):
        self.filesystem = filesystem
        self.path = path

    def __str__(self) -> str:
        return self.path

    def __truediv__(self, other: str) -> "FileSystemPath":
        """Support path / 'subpath' syntax"""
        if self.path == ".":
            new_path = other
        else:
            new_path = f"{self.path}/{other}"
        return FileSystemPath(self.filesystem, new_path)

    @property
    def name(self) -> str:
        """Get the final component of the path"""
        return self.filesystem.get_name(self.path)

    @property
    def suffix(self) -> str:
        """Get the file extension"""
        return self.filesystem.get_suffix(self.path)

    @property
    def parent(self) -> "FileSystemPath":
        """Get the parent directory"""
        parent_path = self.filesystem.get_parent(self.path)
        return FileSystemPath(self.filesystem, parent_path)

    @property
    def parents(self) -> List["FileSystemPath"]:
        """Get all parent directories"""
        parent_paths = self.filesystem.get_parents(self.path)
        return [FileSystemPath(self.filesystem, p) for p in parent_paths]

    def exists(self) -> bool:
        return self.filesystem.exists(self.path)

    def is_file(self) -> bool:
        return self.filesystem.is_file(self.path)

    def is_dir(self) -> bool:
        return self.filesystem.is_dir(self.path)

    def stat(self) -> Dict[str, Any]:
        return self.filesystem.stat(self.path)

    def mkdir(self, parents: bool = False, exist_ok: bool = False) -> None:
        self.filesystem.mkdir(self.path, parents=parents, exist_ok=exist_ok)

    def open(self, mode: str = "r", encoding: Optional[str] = None) -> ContextManager:
        return self.filesystem.open(self.path, mode=mode, encoding=encoding)

    def read_text(self, encoding: str = "utf-8") -> str:
        return self.filesystem.read_text(self.path, encoding=encoding)

    def write_text(self, content: str, encoding: str = "utf-8") -> None:
        self.filesystem.write_text(self.path, content, encoding=encoding)

    def read_bytes(self) -> bytes:
        return self.filesystem.read_bytes(self.path)

    def write_bytes(self, content: bytes) -> None:
        self.filesystem.write_bytes(self.path, content)

    def unlink(self, missing_ok: bool = False) -> None:
        self.filesystem.remove(self.path, missing_ok=missing_ok)

    def rmdir(self) -> None:
        self.filesystem.rmdir(self.path)

    def iterdir(self) -> Iterator["FileSystemPath"]:
        """Iterate over directory contents"""
        for item in self.filesystem.iterdir(self.path):
            yield FileSystemPath(self.filesystem, item)

    def glob(self, pattern: str) -> Iterator["FileSystemPath"]:
        """Find paths matching a glob pattern relative to this path"""
        full_pattern = f"{self.path}/{pattern}" if self.path != "." else pattern
        for match in self.filesystem.glob(full_pattern):
            yield FileSystemPath(self.filesystem, match)

    def relative_to(self, other: Union[str, "FileSystemPath"]) -> str:
        """Get path relative to another path"""
        other_path = str(other) if isinstance(other, FileSystemPath) else other
        return self.filesystem.relative_to(self.path, other_path)


class FileSystem(ABC):
    """
    Abstract base class defining the interface for filesystem operations.
    Implementations of this class provide access to different storage backends.
    """

    def validate_root(self) -> None:
        """
        Validate that the filesystem root is usable.

        Default: no-op. Remote filesystems (e.g. cloud sandboxes) are
        provisioned by their manager and may not exist at construction time;
        LocalFileSystem overrides this to check the directory eagerly.

        Raises:
            ValueError: If the root is known to be invalid.
        """
        return None

    @abstractmethod
    def open(self, path: str, mode: str = "r", encoding: Optional[str] = None) -> Union[BinaryIO, Any]:
        """
        Open a file and return a file-like object.

        Args:
            path: Path to the file
            mode: Mode to open the file in ('r', 'w', 'rb', etc.)
            encoding: Text encoding to use (for text modes)

        Returns:
            A file-like object

        Raises:
            FileNotFoundError: If the file doesn't exist in read mode
            PermissionError: If the file cannot be accessed
        """

    @abstractmethod
    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """
        Read the entire contents of a file as text.

        Args:
            path: Path to the file
            encoding: Text encoding to use

        Returns:
            The file contents as a string

        Raises:
            FileNotFoundError: If the file doesn't exist
            PermissionError: If the file cannot be accessed
        """

    @abstractmethod
    def read_bytes(self, path: str) -> bytes:
        """
        Read the entire contents of a file as bytes.

        Args:
            path: Path to the file

        Returns:
            The file contents as bytes

        Raises:
            FileNotFoundError: If the file doesn't exist
            PermissionError: If the file cannot be accessed
        """

    @abstractmethod
    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
        """
        Write text content to a file, creating the file if it doesn't exist.

        Args:
            path: Path to the file
            content: Text content to write
            encoding: Text encoding to use

        Raises:
            PermissionError: If the file cannot be accessed or created
        """

    @abstractmethod
    def write_bytes(self, path: str, content: bytes) -> None:
        """
        Write binary content to a file, creating the file if it doesn't exist.

        Args:
            path: Path to the file
            content: Binary content to write

        Raises:
            PermissionError: If the file cannot be accessed or created
        """

    @abstractmethod
    def exists(self, path: str) -> bool:
        """
        Check if a path exists.

        Args:
            path: Path to check

        Returns:
            True if the path exists, False otherwise
        """

    @abstractmethod
    def is_file(self, path: str) -> bool:
        """
        Check if a path is a file.

        Args:
            path: Path to check

        Returns:
            True if the path is a file, False otherwise
        """

    @abstractmethod
    def is_dir(self, path: str) -> bool:
        """
        Check if a path is a directory.

        Args:
            path: Path to check

        Returns:
            True if the path is a directory, False otherwise
        """

    @abstractmethod
    def stat(self, path: str) -> Dict[str, Any]:
        """
        Get file or directory metadata.

        Args:
            path: Path to get metadata for

        Returns:
            Dictionary with metadata (size, modified_time, etc.)

        Raises:
            FileNotFoundError: If the path doesn't exist
        """

    @abstractmethod
    def mkdir(self, path: str, parents: bool = False, exist_ok: bool = False) -> None:
        """
        Create a directory.

        Args:
            path: Path to create
            parents: If True, create parent directories as needed
            exist_ok: If True, don't raise an error if directory exists

        Raises:
            FileExistsError: If the directory exists and exist_ok is False
            PermissionError: If the directory cannot be created
        """

    @abstractmethod
    def remove(self, path: str, missing_ok: bool = False) -> None:
        """
        Remove a file.

        Args:
            path: Path to remove
            missing_ok: If True, don't raise an error if file doesn't exist

        Raises:
            FileNotFoundError: If the file doesn't exist and missing_ok is False
            PermissionError: If the file cannot be removed
            IsADirectoryError: If the path is a directory
        """

    @abstractmethod
    def rmdir(self, path: str) -> None:
        """
        Remove an empty directory.

        Args:
            path: Path to remove

        Raises:
            FileNotFoundError: If the directory doesn't exist
            PermissionError: If the directory cannot be removed
            OSError: If the directory is not empty
        """

    @abstractmethod
    def rmtree(self, path: str) -> None:
        """
        Remove a directory and all its contents.

        Args:
            path: Path to remove

        Raises:
            FileNotFoundError: If the directory doesn't exist
            PermissionError: If the directory cannot be removed
        """

    @abstractmethod
    def listdir(self, path: str) -> List[str]:
        """
        List the contents of a directory.

        Args:
            path: Path to list

        Returns:
            List of filenames in the directory

        Raises:
            FileNotFoundError: If the directory doesn't exist
            NotADirectoryError: If the path is not a directory
        """

    @abstractmethod
    def iterdir(self, path: str) -> Iterator[str]:
        """
        Iterate over the contents of a directory.

        Args:
            path: Path to iterate over

        Returns:
            Iterator of paths in the directory

        Raises:
            FileNotFoundError: If the directory doesn't exist
            NotADirectoryError: If the path is not a directory
        """

    @abstractmethod
    def glob(self, pattern: str) -> List[str]:
        """
        Find paths matching a glob pattern.

        Args:
            pattern: Glob pattern to match

        Returns:
            List of paths matching the pattern
        """

    @abstractmethod
    def is_binary_file(self, path: str) -> bool:
        """
        Determine if a file is binary.

        Args:
            path: Path to check

        Returns:
            True if the file is binary, False otherwise

        Raises:
            FileNotFoundError: If the file doesn't exist
        """

    @abstractmethod
    def get_name(self, path: str) -> str:
        """
        Get the final component of the path.

        Args:
            path: Path to get name from

        Returns:
            The final component of the path
        """

    @abstractmethod
    def get_suffix(self, path: str) -> str:
        """
        Get the file extension.

        Args:
            path: Path to get suffix from

        Returns:
            The file extension (including the dot)
        """

    @abstractmethod
    def get_parent(self, path: str) -> str:
        """
        Get the parent directory path.

        Args:
            path: Path to get parent from

        Returns:
            The parent directory path
        """

    @abstractmethod
    def get_parents(self, path: str) -> List[str]:
        """
        Get all parent directories.

        Args:
            path: Path to get parents from

        Returns:
            List of parent directory paths
        """

    @abstractmethod
    def relative_to(self, path: str, other: str) -> str:
        """
        Get path relative to another path.

        Args:
            path: The path to make relative
            other: The base path

        Returns:
            The relative path

        Raises:
            ValueError: If path is not relative to other
        """

    @abstractmethod
    def join_path(self, *parts: str) -> str:
        """
        Join path components.

        Args:
            parts: Path components to join

        Returns:
            The joined path
        """

    @abstractmethod
    def is_absolute(self, path: str) -> bool:
        """
        Check if a path is absolute.

        Args:
            path: Path to check

        Returns:
            True if the path is absolute, False otherwise
        """

    def path(self, path: str) -> FileSystemPath:
        """
        Create a FileSystemPath object for more natural path operations.

        Args:
            path: The path string

        Returns:
            A FileSystemPath object
        """
        return FileSystemPath(self, path)

    # Convenience methods for backward compatibility
    def get_extension(self, path: str) -> str:
        """Alias for get_suffix for backward compatibility."""
        return self.get_suffix(path)

    def is_directory(self, path: str) -> bool:
        """Alias for is_dir for backward compatibility."""
        return self.is_dir(path)

    def get_path(self, path: str) -> Path:
        """Get a Path object for the given path."""
        return self._resolve_path(path) if hasattr(self, "_resolve_path") else Path(path)

    def list_directory(self, path: str) -> List[str]:
        """Alias for iterdir that returns a list."""
        return list(self.iterdir(path))

    def create_directory(self, path: str, parents: bool = True, exist_ok: bool = True) -> None:
        """Create a directory with sensible defaults."""
        self.mkdir(path, parents=parents, exist_ok=exist_ok)

    def delete(self, path: str) -> None:
        """Delete a file or directory."""
        if self.is_dir(path):
            self.rmtree(path)
        else:
            self.remove(path, missing_ok=True)

    def get_size(self, path: str) -> int:
        """Get the size of a file."""
        stat_info = self.stat(path)
        return stat_info.get("size", 0)

    def get_modification_time(self, path: str) -> datetime:
        """Get the modification time of a file."""
        stat_info = self.stat(path)
        return datetime.fromtimestamp(stat_info.get("modified_time", 0))


class LocalFileSystem(FileSystem):
    """
    Implementation of FileSystem that uses the local filesystem.
    """

    def __init__(self, root_path: Optional[Union[str, Path]] = None):
        """
        Initialize with an optional root path to use as a base for all operations.

        Args:
            root_path: Optional root path to use as base for relative paths
        """
        self.root_path = Path(root_path) if root_path else None

    def validate_root(self) -> None:
        if not self.exists("."):
            raise ValueError(f"Project path does not exist: {self.root_path}")
        if not self.is_dir("."):
            raise ValueError(f"Project path is not a directory: {self.root_path}")

    def _resolve_path(self, path: str) -> Path:
        """
        Resolve a potentially relative path against the root path.

        Args:
            path: Path to resolve

        Returns:
            Resolved path as a Path object
        """
        if self.root_path:
            return self.root_path / path
        return Path(path)

    def open(self, path: str, mode: str = "r", encoding: Optional[str] = None) -> Union[BinaryIO, Any]:
        resolved_path = self._resolve_path(path)
        return resolved_path.open(mode=mode, encoding=encoding)

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        resolved_path = self._resolve_path(path)
        return resolved_path.read_text(encoding=encoding)

    def read_bytes(self, path: str) -> bytes:
        resolved_path = self._resolve_path(path)
        return resolved_path.read_bytes()

    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
        resolved_path = self._resolve_path(path)
        resolved_path.write_text(content, encoding=encoding)

    def write_bytes(self, path: str, content: bytes) -> None:
        resolved_path = self._resolve_path(path)
        resolved_path.write_bytes(content)

    def exists(self, path: str) -> bool:
        resolved_path = self._resolve_path(path)
        return resolved_path.exists()

    def is_file(self, path: str) -> bool:
        resolved_path = self._resolve_path(path)
        return resolved_path.is_file()

    def is_dir(self, path: str) -> bool:
        resolved_path = self._resolve_path(path)
        return resolved_path.is_dir()

    def stat(self, path: str) -> Dict[str, Any]:
        resolved_path = self._resolve_path(path)
        stat_result = resolved_path.stat()
        return {
            "size": stat_result.st_size,
            "modified_time": stat_result.st_mtime,
            "created_time": stat_result.st_ctime,
            "accessed_time": stat_result.st_atime,
            "is_directory": resolved_path.is_dir(),
            "is_file": resolved_path.is_file(),
            "stat_result": stat_result,  # Include the full stat result for advanced operations
        }

    def mkdir(self, path: str, parents: bool = False, exist_ok: bool = False) -> None:
        resolved_path = self._resolve_path(path)
        resolved_path.mkdir(parents=parents, exist_ok=exist_ok)

    def remove(self, path: str, missing_ok: bool = False) -> None:
        resolved_path = self._resolve_path(path)
        resolved_path.unlink(missing_ok=missing_ok)

    def rmdir(self, path: str) -> None:
        resolved_path = self._resolve_path(path)
        resolved_path.rmdir()

    def rmtree(self, path: str) -> None:
        resolved_path = self._resolve_path(path)
        shutil.rmtree(resolved_path)

    def listdir(self, path: str) -> List[str]:
        resolved_path = self._resolve_path(path)
        return [str(p.name) for p in resolved_path.iterdir()]

    def iterdir(self, path: str) -> Iterator[str]:
        resolved_path = self._resolve_path(path)
        for item in resolved_path.iterdir():
            if self.root_path:
                yield str(item.relative_to(self.root_path))
            else:
                yield str(item)

    def glob(self, pattern: str) -> List[str]:
        # If we have a root path, we need to make the pattern relative to it
        if self.root_path:
            paths = list(self.root_path.glob(pattern))
            # Return paths relative to the root path
            return [str(p.relative_to(self.root_path)) for p in paths]
        else:
            return [str(p) for p in Path().glob(pattern)]

    def is_binary_file(self, path: str) -> bool:
        resolved_path = self._resolve_path(path)

        # Check extension for common binary formats
        binary_extensions = {
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
        }

        if resolved_path.suffix.lower() in binary_extensions:
            return True

        # Sample file content to check for null bytes
        try:
            with resolved_path.open("rb") as f:
                sample = f.read(1024)
                if b"\x00" in sample:  # If null byte is present, likely binary
                    return True
        except Exception:
            # If there's an error reading the file, consider it binary to be safe
            return True

        return False

    def get_name(self, path: str) -> str:
        resolved_path = self._resolve_path(path)
        return resolved_path.name

    def get_suffix(self, path: str) -> str:
        resolved_path = self._resolve_path(path)
        return resolved_path.suffix

    def get_parent(self, path: str) -> str:
        resolved_path = self._resolve_path(path)
        parent = resolved_path.parent
        if self.root_path:
            try:
                return str(parent.relative_to(self.root_path))
            except ValueError:
                # If parent is outside root_path, return the absolute parent
                return str(parent)
        return str(parent)

    def get_parents(self, path: str) -> List[str]:
        resolved_path = self._resolve_path(path)
        parents = []
        for parent in resolved_path.parents:
            if self.root_path:
                try:
                    parents.append(str(parent.relative_to(self.root_path)))
                except ValueError:
                    # If parent is outside root_path, return the absolute parent
                    parents.append(str(parent))
            else:
                parents.append(str(parent))
        return parents

    def relative_to(self, path: str, other: str) -> str:
        resolved_path = self._resolve_path(path)
        other_resolved = self._resolve_path(other)
        return str(resolved_path.relative_to(other_resolved))

    def join_path(self, *parts: str) -> str:
        if self.root_path:
            result = self.root_path
            for part in parts:
                result = result / part
            return str(result.relative_to(self.root_path))
        else:
            result = Path(parts[0]) if parts else Path()
            for part in parts[1:]:
                result = result / part
            return str(result)

    def is_absolute(self, path: str) -> bool:
        return Path(path).is_absolute()

    # Additional utility methods for compatibility with os.path operations
    def path_join(self, *parts: str) -> str:
        """Equivalent to os.path.join"""
        return os.path.join(*parts)

    def path_exists(self, path: str) -> bool:
        """Equivalent to os.path.exists"""
        return os.path.exists(path)

    def path_isdir(self, path: str) -> bool:
        """Equivalent to os.path.isdir"""
        return os.path.isdir(path)

    def path_isfile(self, path: str) -> bool:
        """Equivalent to os.path.isfile"""
        return os.path.isfile(path)

    def format_datetime(self, timestamp: float) -> str:
        """Format a timestamp as a datetime string"""
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
