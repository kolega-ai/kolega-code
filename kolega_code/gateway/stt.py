"""Reusable remote speech-to-text providers.

Remote-only by design: audio uploads to a hosted transcription API — no local
model downloads, no heavyweight optional extras. Providers accept either a
file path or in-memory WebM bytes so gateway voice notes and other clients can
share the same implementation. The registry mirrors the web-search backend
pattern: named provider classes derive from :class:`SttProvider`, and settings
surfaces plus consumers derive from the registry.

``groq`` is the built-in default: Groq's hosted Whisper
(``whisper-large-v3-turbo`` by default), the same provider/model family
Hermes uses; it authenticates with the Groq API key stored for the ``groq``
LLM provider on the Providers page (or ``GROQ_API_KEY`` in the environment).
``openai`` uses OpenAI's transcription endpoint with ``gpt-transcribe`` by
default and reuses the configured OpenAI API key.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple, Type

from kolega_code.cli.settings import CliSettings

logger = logging.getLogger(__name__)

DEFAULT_STT_PROVIDER = "groq"
GROQ_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
OPENAI_TRANSCRIPTIONS_URL = "https://api.openai.com/v1/audio/transcriptions"
STT_MAX_AUDIO_BYTES = 25 * 1024 * 1024
# Backward-compatible name for callers that imported the Groq-specific limit.
GROQ_MAX_AUDIO_BYTES = STT_MAX_AUDIO_BYTES

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

    async def transcribe(self, audio: Path | bytes) -> str:
        raise NotImplementedError


def _bounded_api_error(detail: str) -> str:
    """A short, safe API failure reason; response bodies can be large and
    opaque, and nothing from them is useful to the operator at full length."""
    cleaned = " ".join(str(detail).split())
    return cleaned[:200] or "unknown API error"


async def _audio_upload(audio: Path | bytes) -> tuple[str, bytes, str]:
    """Normalize a file path or in-memory WebM recording for multipart upload."""
    if isinstance(audio, Path):
        data = await asyncio.to_thread(audio.read_bytes)
        filename = audio.name or "voice.ogg"
    elif isinstance(audio, bytes):
        data = audio
        filename = "voice.webm"
    else:
        raise TypeError("audio must be pathlib.Path or bytes")
    content_type = _AUDIO_CONTENT_TYPES.get(Path(filename).suffix.lower(), "application/octet-stream")
    return filename, data, content_type


class _OpenAICompatibleTranscriber(SttProvider):
    """Shared multipart implementation for OpenAI-compatible transcription APIs."""

    service_name: ClassVar[str]
    api_key_label: ClassVar[str]
    transcriptions_url: ClassVar[str]
    max_audio_bytes: ClassVar[int] = STT_MAX_AUDIO_BYTES

    def __init__(self, *, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        if not api_key:
            raise SttProviderNotConfigured(
                f"{self.service_name} transcription needs a configured {self.api_key_label}: "
                f"add it on the Providers page or set {self.key_env_var}.",
            )
        super().__init__(model=model, api_key=api_key)

    async def transcribe(self, audio: Path | bytes) -> str:
        filename, data, content_type = await _audio_upload(audio)
        if not data:
            raise SttProviderUnavailable("The audio input is empty.")
        if len(data) > self.max_audio_bytes:
            max_mb = self.max_audio_bytes // (1024 * 1024)
            raise SttProviderError(
                f"The audio input is {len(data) // (1024 * 1024)} MB; {self.service_name} accepts at most {max_mb} MB."
            )
        import httpx

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
                response = await client.post(
                    self.transcriptions_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"file": (filename, data, content_type)},
                    data={"model": self.model},
                )
        except httpx.HTTPError as exc:
            raise SttProviderError(
                f"{self.service_name} transcription request failed: {_bounded_api_error(str(exc))}"
            ) from exc
        if response.status_code != 200:
            raise SttProviderError(
                f"{self.service_name} transcription failed "
                f"({response.status_code}): {_bounded_api_error(response.text)}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise SttProviderError(f"{self.service_name} returned a non-JSON transcription response.") from exc
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise SttProviderError(f"{self.service_name} returned an empty transcription.")
        return text.strip()


class GroqWhisperTranscriber(_OpenAICompatibleTranscriber):
    """Groq's hosted Whisper API.

    Authenticates with the Groq API key already stored for the ``groq`` LLM
    provider — the same single key Hermes uses. Files are uploaded as-is;
    Groq accepts the common voice-note containers (ogg, m4a, mp3, wav, …).
    """

    name = "groq"
    label = "Groq (whisper-large-v3-turbo)"
    default_model = "whisper-large-v3-turbo"
    key_env_var = "GROQ_API_KEY"
    service_name = "Groq"
    api_key_label = "Groq API key"
    transcriptions_url = GROQ_TRANSCRIPTIONS_URL


class OpenAITranscriber(_OpenAICompatibleTranscriber):
    """OpenAI's hosted transcription API."""

    name = "openai"
    label = "OpenAI (gpt-transcribe)"
    default_model = "gpt-transcribe"
    key_env_var = "OPENAI_API_KEY"
    service_name = "OpenAI"
    api_key_label = "OpenAI API key"
    transcriptions_url = OPENAI_TRANSCRIPTIONS_URL


_STT_PROVIDERS: Dict[str, Type[SttProvider]] = {
    provider.name: provider for provider in (GroqWhisperTranscriber, OpenAITranscriber)
}


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


def resolve_stt(settings: Optional[CliSettings]) -> tuple[bool, str, Optional[str], Optional[str]]:
    """Resolve ``(enabled, provider, model, api_key)`` from persisted settings.

    The selected provider's environment variable takes precedence over its
    stored provider key. Unknown provider names fall back to the compatibility
    default so older or manually edited settings remain loadable.
    """
    enabled = bool(settings.stt_enabled) if settings is not None else False
    provider = (settings.stt_provider if settings is not None else None) or DEFAULT_STT_PROVIDER
    if provider not in _STT_PROVIDERS:
        logger.warning("stt: unknown provider %r; using %r", provider, DEFAULT_STT_PROVIDER)
        provider = DEFAULT_STT_PROVIDER
    model = settings.stt_model if settings is not None else None
    provider_cls = get_stt_provider_class(provider)
    env_name = provider_cls.key_env_var
    api_key = (os.environ.get(env_name) if env_name else None) or (
        settings.api_keys.get(provider) if settings is not None else None
    )
    return enabled, provider, model, api_key


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
    "GROQ_MAX_AUDIO_BYTES",
    "GROQ_TRANSCRIPTIONS_URL",
    "GroqWhisperTranscriber",
    "OPENAI_TRANSCRIPTIONS_URL",
    "OpenAITranscriber",
    "STT_MAX_AUDIO_BYTES",
    "SttProvider",
    "SttProviderError",
    "SttProviderNotConfigured",
    "SttProviderUnavailable",
    "available_stt_providers",
    "build_transcriber",
    "get_stt_provider_class",
    "resolve_stt",
    "stt_provider_names",
]
