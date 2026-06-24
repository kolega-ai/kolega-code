# ruff: noqa: F401,F811,E402
import os
import tempfile
import pytest
from pathlib import Path

from kolega_code.services.file_system import LocalFileSystem, FileSystemPath


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
