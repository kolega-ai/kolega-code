import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
from datetime import datetime

from ...services.file_system import LocalFileSystem, FileSystemPath


class TestLocalFileSystem:
    """Comprehensive tests for LocalFileSystem implementation."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def filesystem(self, temp_dir):
        """Create a LocalFileSystem instance with a temporary root."""
        return LocalFileSystem(root_path=temp_dir)

    @pytest.fixture
    def filesystem_no_root(self):
        """Create a LocalFileSystem instance without a root path."""
        return LocalFileSystem()

    def test_init_with_root_path_string(self, temp_dir):
        """Test initialization with root path as string."""
        fs = LocalFileSystem(root_path=str(temp_dir))
        assert fs.root_path == temp_dir

    def test_init_with_root_path_pathlib(self, temp_dir):
        """Test initialization with root path as Path object."""
        fs = LocalFileSystem(root_path=temp_dir)
        assert fs.root_path == temp_dir

    def test_init_without_root_path(self):
        """Test initialization without root path."""
        fs = LocalFileSystem()
        assert fs.root_path is None

    def test_resolve_path_with_root(self, filesystem, temp_dir):
        """Test path resolution with root path."""
        resolved = filesystem._resolve_path("test.txt")
        expected = temp_dir / "test.txt"
        assert resolved == expected

    def test_resolve_path_without_root(self, filesystem_no_root):
        """Test path resolution without root path."""
        resolved = filesystem_no_root._resolve_path("test.txt")
        expected = Path("test.txt")
        assert resolved == expected

    def test_write_and_read_text(self, filesystem):
        """Test writing and reading text files."""
        content = "Hello, World!\nThis is a test file."
        filesystem.write_text("test.txt", content)

        read_content = filesystem.read_text("test.txt")
        assert read_content == content

    def test_write_and_read_text_with_encoding(self, filesystem):
        """Test writing and reading text files with specific encoding."""
        content = "Hello, 世界! 🌍"
        filesystem.write_text("test_utf8.txt", content, encoding="utf-8")

        read_content = filesystem.read_text("test_utf8.txt", encoding="utf-8")
        assert read_content == content

    def test_write_and_read_bytes(self, filesystem):
        """Test writing and reading binary files."""
        content = b"\x00\x01\x02\x03\xff\xfe\xfd"
        filesystem.write_bytes("test.bin", content)

        read_content = filesystem.read_bytes("test.bin")
        assert read_content == content

    def test_exists(self, filesystem):
        """Test checking if files and directories exist."""
        # File doesn't exist initially
        assert not filesystem.exists("nonexistent.txt")

        # Create file and check it exists
        filesystem.write_text("exists_test.txt", "content")
        assert filesystem.exists("exists_test.txt")

        # Create directory and check it exists
        filesystem.mkdir("test_dir")
        assert filesystem.exists("test_dir")

    def test_is_file(self, filesystem):
        """Test checking if path is a file."""
        # Create file
        filesystem.write_text("test_file.txt", "content")
        assert filesystem.is_file("test_file.txt")

        # Create directory
        filesystem.mkdir("test_dir")
        assert not filesystem.is_file("test_dir")

        # Non-existent path
        assert not filesystem.is_file("nonexistent.txt")

    def test_is_dir(self, filesystem):
        """Test checking if path is a directory."""
        # Create directory
        filesystem.mkdir("test_dir")
        assert filesystem.is_dir("test_dir")

        # Create file
        filesystem.write_text("test_file.txt", "content")
        assert not filesystem.is_dir("test_file.txt")

        # Non-existent path
        assert not filesystem.is_dir("nonexistent_dir")

    def test_stat(self, filesystem):
        """Test getting file statistics."""
        content = "Test content for stat"
        filesystem.write_text("stat_test.txt", content)

        stat_info = filesystem.stat("stat_test.txt")

        assert "size" in stat_info
        assert "modified_time" in stat_info
        assert "created_time" in stat_info
        assert "accessed_time" in stat_info
        assert "is_directory" in stat_info
        assert "is_file" in stat_info
        assert "stat_result" in stat_info

        assert stat_info["size"] == len(content.encode())
        assert stat_info["is_file"] is True
        assert stat_info["is_directory"] is False

    def test_mkdir(self, filesystem):
        """Test creating directories."""
        # Simple directory creation
        filesystem.mkdir("simple_dir")
        assert filesystem.is_dir("simple_dir")

        # Directory creation with parents
        filesystem.mkdir("parent/child/grandchild", parents=True)
        assert filesystem.is_dir("parent/child/grandchild")

        # Directory creation with exist_ok
        filesystem.mkdir("simple_dir", exist_ok=True)  # Should not raise

        # Directory creation without exist_ok should raise
        with pytest.raises(FileExistsError):
            filesystem.mkdir("simple_dir", exist_ok=False)

    def test_remove(self, filesystem):
        """Test removing files."""
        # Create and remove file
        filesystem.write_text("remove_test.txt", "content")
        assert filesystem.exists("remove_test.txt")

        filesystem.remove("remove_test.txt")
        assert not filesystem.exists("remove_test.txt")

        # Test missing_ok=True
        filesystem.remove("nonexistent.txt", missing_ok=True)  # Should not raise

        # Test missing_ok=False (default)
        with pytest.raises(FileNotFoundError):
            filesystem.remove("nonexistent.txt", missing_ok=False)

    def test_rmdir(self, filesystem):
        """Test removing empty directories."""
        # Create and remove empty directory
        filesystem.mkdir("empty_dir")
        assert filesystem.is_dir("empty_dir")

        filesystem.rmdir("empty_dir")
        assert not filesystem.exists("empty_dir")

        # Test removing non-empty directory should raise
        filesystem.mkdir("non_empty_dir")
        filesystem.write_text("non_empty_dir/file.txt", "content")

        with pytest.raises(OSError):
            filesystem.rmdir("non_empty_dir")

    def test_rmtree(self, filesystem):
        """Test removing directories recursively."""
        # Create directory structure
        filesystem.mkdir("tree/branch1/leaf1", parents=True)
        filesystem.mkdir("tree/branch2/leaf2", parents=True)
        filesystem.write_text("tree/file.txt", "content")
        filesystem.write_text("tree/branch1/file1.txt", "content1")

        assert filesystem.is_dir("tree")

        # Remove entire tree
        filesystem.rmtree("tree")
        assert not filesystem.exists("tree")

    def test_listdir(self, filesystem):
        """Test listing directory contents."""
        # Create test structure
        filesystem.mkdir("list_test")
        filesystem.write_text("list_test/file1.txt", "content1")
        filesystem.write_text("list_test/file2.txt", "content2")
        filesystem.mkdir("list_test/subdir")

        contents = filesystem.listdir("list_test")

        assert len(contents) == 3
        assert "file1.txt" in contents
        assert "file2.txt" in contents
        assert "subdir" in contents

    def test_iterdir(self, filesystem):
        """Test iterating over directory contents."""
        # Create test structure
        filesystem.mkdir("iter_test")
        filesystem.write_text("iter_test/file1.txt", "content1")
        filesystem.write_text("iter_test/file2.txt", "content2")
        filesystem.mkdir("iter_test/subdir")

        contents = list(filesystem.iterdir("iter_test"))

        assert len(contents) == 3
        assert "iter_test/file1.txt" in contents
        assert "iter_test/file2.txt" in contents
        assert "iter_test/subdir" in contents

    def test_glob(self, filesystem):
        """Test glob pattern matching."""
        # Create test files
        filesystem.write_text("test1.txt", "content")
        filesystem.write_text("test2.txt", "content")
        filesystem.write_text("test.py", "content")
        filesystem.mkdir("subdir")
        filesystem.write_text("subdir/test3.txt", "content")

        # Test simple pattern
        txt_files = filesystem.glob("*.txt")
        assert "test1.txt" in txt_files
        assert "test2.txt" in txt_files
        assert "test.py" not in txt_files

        # Test recursive pattern
        all_txt = filesystem.glob("**/*.txt")
        assert "test1.txt" in all_txt
        assert "test2.txt" in all_txt
        assert "subdir/test3.txt" in all_txt

    def test_is_binary_file(self, filesystem):
        """Test binary file detection."""
        # Text file
        filesystem.write_text("text.txt", "This is text content")
        assert not filesystem.is_binary_file("text.txt")

        # Binary file by extension
        filesystem.write_bytes("binary.jpg", b"fake image data")
        assert filesystem.is_binary_file("binary.jpg")

        # Binary file by content (contains null bytes)
        filesystem.write_bytes("binary_content.dat", b"text\x00with\x00nulls")
        assert filesystem.is_binary_file("binary_content.dat")

    def test_get_name(self, filesystem):
        """Test getting filename from path."""
        assert filesystem.get_name("test.txt") == "test.txt"
        assert filesystem.get_name("dir/subdir/file.py") == "file.py"
        assert filesystem.get_name("dir/") == "dir"
        assert filesystem.get_name("dir") == "dir"

    def test_get_suffix(self, filesystem):
        """Test getting file extension."""
        assert filesystem.get_suffix("test.txt") == ".txt"
        assert filesystem.get_suffix("file.tar.gz") == ".gz"
        assert filesystem.get_suffix("no_extension") == ""
        assert filesystem.get_suffix("dir/file.py") == ".py"

    def test_get_parent(self, filesystem):
        """Test getting parent directory."""
        # With root path, should return relative parent
        parent = filesystem.get_parent("dir/subdir/file.txt")
        assert parent == "dir/subdir"

        parent = filesystem.get_parent("file.txt")
        assert parent == "."

    def test_get_parents(self, filesystem):
        """Test getting all parent directories."""
        parents = filesystem.get_parents("dir/subdir/file.txt")
        assert "dir/subdir" in parents
        assert "dir" in parents
        assert "." in parents

    def test_relative_to(self, filesystem):
        """Test getting relative path."""
        # Create some structure
        filesystem.mkdir("base/sub", parents=True)

        relative = filesystem.relative_to("base/sub/file.txt", "base")
        assert relative == "sub/file.txt"

    def test_join_path(self, filesystem):
        """Test joining path components."""
        joined = filesystem.join_path("dir", "subdir", "file.txt")
        assert joined == "dir/subdir/file.txt"

    def test_is_absolute(self, filesystem):
        """Test checking if path is absolute."""
        assert not filesystem.is_absolute("relative/path")
        assert filesystem.is_absolute("/absolute/path")

        # Windows absolute path
        if os.name == "nt":
            assert filesystem.is_absolute("C:\\absolute\\path")

    def test_path_join(self, filesystem):
        """Test os.path.join compatibility method."""
        joined = filesystem.path_join("dir", "subdir", "file.txt")
        expected = os.path.join("dir", "subdir", "file.txt")
        assert joined == expected

    def test_path_exists(self, filesystem):
        """Test os.path.exists compatibility method."""
        # This tests the raw os.path.exists, not the filesystem abstraction
        with tempfile.NamedTemporaryFile() as tmp:
            assert filesystem.path_exists(tmp.name)
        assert not filesystem.path_exists("/nonexistent/path")

    def test_path_isdir(self, filesystem):
        """Test os.path.isdir compatibility method."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            assert filesystem.path_isdir(tmp_dir)
        assert not filesystem.path_isdir("/nonexistent/dir")

    def test_path_isfile(self, filesystem):
        """Test os.path.isfile compatibility method."""
        with tempfile.NamedTemporaryFile() as tmp:
            assert filesystem.path_isfile(tmp.name)
        assert not filesystem.path_isfile("/nonexistent/file")

    def test_format_datetime(self, filesystem):
        """Test datetime formatting utility."""
        timestamp = 1640995200.0  # 2022-01-01 00:00:00 UTC
        formatted = filesystem.format_datetime(timestamp)
        # The exact format depends on timezone, but should contain date components
        assert "2022" in formatted or "2021" in formatted  # Account for timezone differences
        assert "-" in formatted
        assert ":" in formatted

    def test_open_context_manager(self, filesystem):
        """Test file opening as context manager."""
        content = "Test content for context manager"

        # Write using context manager
        with filesystem.open("context_test.txt", "w") as f:
            f.write(content)

        # Read using context manager
        with filesystem.open("context_test.txt", "r") as f:
            read_content = f.read()

        assert read_content == content

    def test_error_handling(self, filesystem):
        """Test various error conditions."""
        # FileNotFoundError for reading non-existent file
        with pytest.raises(FileNotFoundError):
            filesystem.read_text("nonexistent.txt")

        # FileNotFoundError for stat on non-existent file
        with pytest.raises(FileNotFoundError):
            filesystem.stat("nonexistent.txt")

        # NotADirectoryError for listing non-directory
        filesystem.write_text("not_a_dir.txt", "content")
        with pytest.raises(NotADirectoryError):
            filesystem.listdir("not_a_dir.txt")


class TestFileSystemPath:
    """Tests for the FileSystemPath wrapper class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def filesystem(self, temp_dir):
        """Create a LocalFileSystem instance with a temporary root."""
        return LocalFileSystem(root_path=temp_dir)

    @pytest.fixture
    def fs_path(self, filesystem):
        """Create a FileSystemPath instance."""
        return filesystem.path("test_path")

    def test_init_and_str(self, filesystem):
        """Test FileSystemPath initialization and string representation."""
        path = FileSystemPath(filesystem, "test/path")
        assert str(path) == "test/path"
        assert path.path == "test/path"

    def test_truediv_operator(self, filesystem):
        """Test path / 'subpath' syntax."""
        path = FileSystemPath(filesystem, "base")
        new_path = path / "subdir" / "file.txt"

        assert str(new_path) == "base/subdir/file.txt"
        assert isinstance(new_path, FileSystemPath)

    def test_truediv_with_root_path(self, filesystem):
        """Test path / 'subpath' syntax with root path."""
        path = FileSystemPath(filesystem, ".")
        new_path = path / "file.txt"

        assert str(new_path) == "file.txt"

    def test_name_property(self, filesystem):
        """Test name property."""
        path = FileSystemPath(filesystem, "dir/file.txt")
        assert path.name == "file.txt"

    def test_suffix_property(self, filesystem):
        """Test suffix property."""
        path = FileSystemPath(filesystem, "dir/file.txt")
        assert path.suffix == ".txt"

    def test_parent_property(self, filesystem):
        """Test parent property."""
        path = FileSystemPath(filesystem, "dir/subdir/file.txt")
        parent = path.parent

        assert isinstance(parent, FileSystemPath)
        assert str(parent) == "dir/subdir"

    def test_parents_property(self, filesystem):
        """Test parents property."""
        path = FileSystemPath(filesystem, "dir/subdir/file.txt")
        parents = path.parents

        assert all(isinstance(p, FileSystemPath) for p in parents)
        parent_strs = [str(p) for p in parents]
        assert "dir/subdir" in parent_strs
        assert "dir" in parent_strs

    def test_file_operations(self, filesystem):
        """Test file operations through FileSystemPath."""
        path = filesystem.path("test_file.txt")

        # Write and read
        path.write_text("Hello, World!")
        content = path.read_text()
        assert content == "Hello, World!"

        # Check existence and type
        assert path.exists()
        assert path.is_file()
        assert not path.is_dir()

        # Get stats
        stat_info = path.stat()
        assert stat_info["is_file"] is True

    def test_directory_operations(self, filesystem):
        """Test directory operations through FileSystemPath."""
        path = filesystem.path("test_dir")

        # Create directory
        path.mkdir()
        assert path.exists()
        assert path.is_dir()
        assert not path.is_file()

        # Create subdirectory with parents
        subpath = path / "subdir" / "deep"
        subpath.mkdir(parents=True)
        assert subpath.exists()

    def test_iterdir(self, filesystem):
        """Test iterating over directory contents."""
        # Create test structure
        base_path = filesystem.path("iter_base")
        base_path.mkdir()

        (base_path / "file1.txt").write_text("content1")
        (base_path / "file2.txt").write_text("content2")
        (base_path / "subdir").mkdir()

        # Iterate and collect
        contents = list(base_path.iterdir())
        content_names = [item.name for item in contents]

        assert len(contents) == 3
        assert "file1.txt" in content_names
        assert "file2.txt" in content_names
        assert "subdir" in content_names
        assert all(isinstance(item, FileSystemPath) for item in contents)

    def test_glob(self, filesystem):
        """Test glob pattern matching through FileSystemPath."""
        # Create test structure
        base_path = filesystem.path("glob_base")
        base_path.mkdir()

        (base_path / "test1.txt").write_text("content")
        (base_path / "test2.txt").write_text("content")
        (base_path / "test.py").write_text("content")

        # Test glob
        txt_files = list(base_path.glob("*.txt"))
        txt_names = [f.name for f in txt_files]

        assert len(txt_files) == 2
        assert "test1.txt" in txt_names
        assert "test2.txt" in txt_names
        assert all(isinstance(f, FileSystemPath) for f in txt_files)

    def test_relative_to(self, filesystem):
        """Test relative_to method."""
        base_path = filesystem.path("base")
        file_path = filesystem.path("base/sub/file.txt")

        relative = file_path.relative_to(base_path)
        assert relative == "sub/file.txt"

        # Test with string
        relative_str = file_path.relative_to("base")
        assert relative_str == "sub/file.txt"

    def test_unlink(self, filesystem):
        """Test file removal through FileSystemPath."""
        path = filesystem.path("remove_me.txt")
        path.write_text("content")
        assert path.exists()

        path.unlink()
        assert not path.exists()

        # Test missing_ok
        path.unlink(missing_ok=True)  # Should not raise

    def test_rmdir(self, filesystem):
        """Test directory removal through FileSystemPath."""
        path = filesystem.path("remove_dir")
        path.mkdir()
        assert path.exists()

        path.rmdir()
        assert not path.exists()

    def test_binary_operations(self, filesystem):
        """Test binary file operations through FileSystemPath."""
        path = filesystem.path("binary_test.bin")
        binary_data = b"\x00\x01\x02\x03\xff"

        path.write_bytes(binary_data)
        read_data = path.read_bytes()

        assert read_data == binary_data

    def test_open_context_manager(self, filesystem):
        """Test opening files as context manager through FileSystemPath."""
        path = filesystem.path("context_test.txt")
        content = "Context manager test"

        # Write using context manager
        with path.open("w") as f:
            f.write(content)

        # Read using context manager
        with path.open("r") as f:
            read_content = f.read()

        assert read_content == content


class TestFileSystemIntegration:
    """Integration tests for filesystem operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def filesystem(self, temp_dir):
        """Create a LocalFileSystem instance with a temporary root."""
        return LocalFileSystem(root_path=temp_dir)

    def test_complex_directory_structure(self, filesystem):
        """Test creating and manipulating complex directory structures."""
        # Create complex structure
        structure = [
            "project/src/main.py",
            "project/src/utils/helper.py",
            "project/tests/test_main.py",
            "project/docs/readme.md",
            "project/config.json",
        ]

        for file_path in structure:
            path = filesystem.path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"Content of {file_path}")

        # Verify structure
        assert filesystem.exists("project")
        assert filesystem.is_dir("project")
        assert filesystem.exists("project/src/main.py")
        assert filesystem.is_file("project/src/main.py")

        # Test glob patterns
        py_files = filesystem.glob("**/*.py")
        assert len(py_files) == 3
        assert "project/src/main.py" in py_files
        assert "project/src/utils/helper.py" in py_files
        assert "project/tests/test_main.py" in py_files

        # Test directory listing
        src_contents = list(filesystem.iterdir("project/src"))
        assert "project/src/main.py" in src_contents
        assert "project/src/utils" in src_contents

    def test_file_operations_chain(self, filesystem):
        """Test chaining multiple file operations."""
        # Create initial file
        path = filesystem.path("chain_test.txt")
        path.write_text("Initial content")

        # Read, modify, write back
        content = path.read_text()
        modified_content = content + "\nAdditional line"
        path.write_text(modified_content)

        # Verify modification
        final_content = path.read_text()
        assert "Initial content" in final_content
        assert "Additional line" in final_content

        # Get file stats
        stats = path.stat()
        assert stats["size"] > len("Initial content")

    def test_error_recovery(self, filesystem):
        """Test error handling and recovery scenarios."""
        # Try to read non-existent file
        with pytest.raises(FileNotFoundError):
            filesystem.read_text("nonexistent.txt")

        # Create file after error
        filesystem.write_text("recovery_test.txt", "Recovered!")
        content = filesystem.read_text("recovery_test.txt")
        assert content == "Recovered!"

        # Try to create directory that already exists
        filesystem.mkdir("existing_dir")
        filesystem.mkdir("existing_dir", exist_ok=True)  # Should not raise

        with pytest.raises(FileExistsError):
            filesystem.mkdir("existing_dir", exist_ok=False)

    def test_path_object_consistency(self, filesystem):
        """Test that FileSystemPath objects maintain consistency."""
        # Create path objects for same path
        path1 = filesystem.path("consistency_test.txt")
        path2 = filesystem.path("consistency_test.txt")

        # Operations on one should be visible through the other
        path1.write_text("Consistent content")
        assert path2.exists()
        assert path2.read_text() == "Consistent content"

        # Path operations should work consistently
        assert path1.name == path2.name
        assert path1.suffix == path2.suffix
        assert str(path1.parent) == str(path2.parent)

    def test_mixed_operations(self, filesystem):
        """Test mixing direct filesystem calls with FileSystemPath operations."""
        # Create file using direct filesystem
        filesystem.write_text("mixed_test.txt", "Direct content")

        # Access using FileSystemPath
        path = filesystem.path("mixed_test.txt")
        assert path.exists()
        assert path.read_text() == "Direct content"

        # Modify using FileSystemPath
        path.write_text("Modified via path")

        # Read using direct filesystem
        content = filesystem.read_text("mixed_test.txt")
        assert content == "Modified via path"
