import asyncio
import base64
from typing import List

from .base_tool import BaseTool
from kolega_code.llm.models import ContentBlock, ImageBlock
from kolega_code.utils.images import IMAGE_MIME_TYPES, MAX_IMAGE_BYTES


class ReadImageTool(BaseTool):
    async def read_image(self, path: str) -> List[ContentBlock]:
        """
        Read an image file from the project directory so you can see it. Use when the user references a screenshot, diagram, mockup, or other visual asset, or when visual inspection of a file in the workspace is needed. The image is returned for you to view directly.

        When to use: the user asks you to look at an image/screenshot/mockup; you need to inspect a visual asset in the project; text-based reading is insufficient to understand a visual file.

        Args:
            path: Path to the image file. Relative to the project root is preferred; an absolute path is also accepted.

        Returns:
            The image, viewable directly.

        Supported formats: PNG, JPEG, GIF, WebP, BMP.
        """
        if not self.filesystem.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        if not self.filesystem.is_file(path):
            raise ValueError(f"Not a file: {path}")
        suffix = self.filesystem.get_suffix(path).lower()
        media_type = IMAGE_MIME_TYPES.get(suffix)
        if media_type is None:
            raise ValueError(f"Unsupported image format '{suffix}'. Supported: " + ", ".join(sorted(IMAGE_MIME_TYPES)))
        # Off the event loop: the read, the base64 encode, and the ASCII decode
        # are each a full pass over the file, and this runs once per screenshot
        # in a review loop. The size gate lives in the same hop so an oversized
        # file is rejected before it is read rather than after.
        data = await asyncio.to_thread(self._read_encoded, path)
        return [ImageBlock(image_type="base64", media_type=media_type, data=data)]

    def _read_encoded(self, path: str) -> str:
        size = self.filesystem.get_size(path)
        if size > MAX_IMAGE_BYTES:
            raise ValueError(
                f"Image is too large to read: {size / (1024 * 1024):.1f} MB exceeds the "
                f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB limit. Resize or crop it first."
            )
        return base64.b64encode(self.filesystem.read_bytes(path)).decode("ascii")
