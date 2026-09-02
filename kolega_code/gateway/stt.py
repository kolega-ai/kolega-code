"""Remote speech-to-text providers for gateway voice notes.

Remote-only by design: voice notes upload to a hosted transcription API —
no local model downloads, no heavyweight optional extras. The registry
mirrors the web-search backend pattern: named provider classes derive from
:class:`SttProvider`, and the TUI Select options plus gateway config derive
from the registry, so adding a provider is a class plus a registry entry.

``groq`` is the built-in default: Groq's hosted Whisper
(``whisper-large-v3-turbo`` by default), the same provider/model family
Hermes uses; it authenticates with the Groq API key stored for the ``groq``
LLM provider on the Providers page (or ``GROQ_API_KEY`` in the environment).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)

DEFAULT_STT_PROVIDER = "groq"
GROQ_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MAX_AUDIO_BYTES = 25 * 1024 * 1024

_AUDIO_CONTENT_TYPES = {
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".mpga": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
}


class SttProviderError(RuntimeError):
    """Raised when a speech-to-text provider cannot be used or built."""


class SttProviderNotConfigured(SttProviderError):
    """Raised when a provider is selected but lacks required configuration."""


class SttProviderUnavailable(SttProviderError):
    """Raised when a provider request cannot be completed."""


class SttProvider:
    """Base class for pluggable speech-to-text providers."""

    name: ClassVar[str]
    label: ClassVar[str]
    default_model: ClassVar[str]
    key_env_var: ClassVar[Optional[str]] = None

    def __init__(self, *, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.model = model or self.default_model
        self.api_key = api_key

    async def transcribe(self, path: Path) -> str:
        raise NotImplementedError


def _bounded_api_error(detail: str) -> str:
    """A short, safe API failure reason; response bodies can be large and
    opaque, and nothing from them is useful to the operator at full length."""
    cleaned = " ".join(str(detail).split())
    return cleaned[:200] or "unknown API error"


class GroqWhisperTranscriber(SttProvider):
    """Groq's hosted Whisper API (OpenAI-compatible transcription endpoint).

    Authenticates with the Groq API key already stored for the ``groq`` LLM
    provider — the same single key Hermes uses. Files are uploaded as-is;
    Groq accepts the common voice-note containers (ogg, m4a, mp3, wav, …).
    """

    name = "groq"
    label = "Groq (whisper-large-v3-turbo)"
    default_model = "whisper-large-v3-turbo"
    key_env_var = "GROQ_API_KEY"

    def __init__(self, *, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        if not api_key:
            raise SttProviderNotConfigured(
                "Groq transcription needs a Groq API key: add it on the Providers page or set GROQ_API_KEY.",
            )
        super().__init__(model=model, api_key=api_key)

    async def transcribe(self, path: Path) -> str:
        data = await asyncio.to_thread(path.read_bytes)
        if not data:
            raise SttProviderUnavailable("The voice note is empty.")
        if len(data) > GROQ_MAX_AUDIO_BYTES:
            raise SttProviderError(f"The voice note is {len(data) // (1024 * 1024)} MB; Groq accepts at most 25 MB.")
        import httpx

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
                response = await client.post(
                    GROQ_TRANSCRIPTIONS_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={
                        "file": (
                            path.name or "voice.ogg",
                            data,
                            _AUDIO_CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream"),
                        )
                    },
                    data={"model": self.model},
                )
        except httpx.HTTPError as exc:
            raise SttProviderError(f"Groq transcription request failed: {_bounded_api_error(str(exc))}") from exc
        if response.status_code != 200:
            raise SttProviderError(
                f"Groq transcription failed ({response.status_code}): {_bounded_api_error(response.text)}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise SttProviderError("Groq returned a non-JSON transcription response.") from exc
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise SttProviderError("Groq returned an empty transcription.")
        return text.strip()


_STT_PROVIDERS: Dict[str, Type[SttProvider]] = {provider.name: provider for provider in (GroqWhisperTranscriber,)}


def stt_provider_names() -> List[str]:
    """All registered provider names."""
    return list(_STT_PROVIDERS)


def available_stt_providers() -> List[Tuple[str, str]]:
    """``(label, name)`` pairs for the TUI Select, with the default provider first."""
    ordered = [_STT_PROVIDERS[DEFAULT_STT_PROVIDER]] + [
        provider for name, provider in _STT_PROVIDERS.items() if name != DEFAULT_STT_PROVIDER
    ]
    return [(provider.label, provider.name) for provider in ordered]


def get_stt_provider_class(name: str) -> Type[SttProvider]:
    """Look up a provider class by name, or raise SttProviderError for an unknown name."""
    try:
        return _STT_PROVIDERS[name]
    except KeyError as exc:
        valid = ", ".join(_STT_PROVIDERS)
        raise SttProviderError(f"Unknown speech-to-text provider '{name}'. Valid: {valid}.") from exc


def build_transcriber(
    name: Optional[str],
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> SttProvider:
    """Construct the configured provider instance.

    ``model`` falls back to the provider's built-in default, so an unset
    ``stt_model`` means "provider default", not a cross-provider constant.
    """
    provider_cls = get_stt_provider_class(name or DEFAULT_STT_PROVIDER)
    return provider_cls(model=model, api_key=api_key)


__all__ = [
    "DEFAULT_STT_PROVIDER",
    "GROQ_TRANSCRIPTIONS_URL",
    "GroqWhisperTranscriber",
    "SttProvider",
    "SttProviderError",
    "SttProviderNotConfigured",
    "SttProviderUnavailable",
    "available_stt_providers",
    "build_transcriber",
    "get_stt_provider_class",
    "stt_provider_names",
]
