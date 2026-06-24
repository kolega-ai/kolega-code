# ruff: noqa: F401,F811,E402
import os
import tempfile
import pytest
from pathlib import Path

from kolega_code.services.file_system import LocalFileSystem, FileSystemPath

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

    def test_iterdir_outside_root(self, filesystem):
        """Listing a directory outside root_path yields absolute child paths, not a crash."""
        # A sibling directory that is NOT under the filesystem's root_path.
        with tempfile.TemporaryDirectory() as outside:
            outside_dir = Path(outside)
            (outside_dir / "child.txt").write_text("hello")
            (outside_dir / "subdir").mkdir()

            # Pre-fix this raised ValueError ("is not in the subpath of") while
            # relativizing the out-of-root children to root_path.
            contents = list(filesystem.iterdir(str(outside_dir)))

            assert str(outside_dir / "child.txt") in contents
            assert str(outside_dir / "subdir") in contents

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
