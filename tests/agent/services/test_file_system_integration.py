# ruff: noqa: F401,F811,E402
import os
import tempfile
import pytest
from pathlib import Path

from kolega_code.services.file_system import LocalFileSystem, FileSystemPath


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
