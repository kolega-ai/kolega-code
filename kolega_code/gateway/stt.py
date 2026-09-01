"""Local voice-note transcription via faster-whisper (optional dependency).

faster-whisper ships in the ``stt`` extra so the core gateway stays lean. The
import happens lazily inside :meth:`WhisperTranscriber.transcribe`, and
callers degrade with a helpful message when it is absent — an uninstalled
optional dependency must never crash a turn.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)

DEFAULT_MODEL_SIZE = "base"


class Transcriber(Protocol):
    """Anything that turns an audio file into text."""

    async def transcribe(self, path: Path) -> str: ...


class WhisperTranscriber:
    """Lazy faster-whisper adapter; construction never imports the dependency.

    The model loads on first use and transcribes in a worker thread — model
    load and inference are CPU-heavy and must stay off the event loop.
    """

    def __init__(self, *, model_size: str = DEFAULT_MODEL_SIZE) -> None:
        self._model_size = model_size
        self._model: Optional[Any] = None

    @staticmethod
    def available() -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    async def transcribe(self, path: Path) -> str:
        def _transcribe() -> str:
            if self._model is None:
                from faster_whisper import WhisperModel

                logger.info("gateway stt: loading faster-whisper model %r", self._model_size)
                self._model = WhisperModel(self._model_size, device="auto", compute_type="default")
            segments, _info = self._model.transcribe(str(path))
            return " ".join(segment.text.strip() for segment in segments if segment.text and segment.text.strip())

        return await asyncio.to_thread(_transcribe)
