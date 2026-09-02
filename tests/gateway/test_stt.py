"""Remote STT provider registry: registration, building, and Groq uploads."""

from pathlib import Path

import httpx
import pytest

from kolega_code.gateway.stt import (
    DEFAULT_STT_PROVIDER,
    GROQ_MAX_AUDIO_BYTES,
    GROQ_TRANSCRIPTIONS_URL,
    GroqWhisperTranscriber,
    SttProviderError,
    SttProviderNotConfigured,
    SttProviderUnavailable,
    available_stt_providers,
    build_transcriber,
    get_stt_provider_class,
    stt_provider_names,
)


def test_registry_lists_only_the_groq_provider() -> None:
    assert stt_provider_names() == ["groq"]
    assert available_stt_providers() == [("Groq (whisper-large-v3-turbo)", "groq")]
    assert get_stt_provider_class("groq") is GroqWhisperTranscriber
    with pytest.raises(SttProviderError, match="Unknown speech-to-text provider"):
        get_stt_provider_class("bogus")


def test_build_transcriber_defaults_to_groq() -> None:
    assert DEFAULT_STT_PROVIDER == "groq"
    with pytest.raises(SttProviderNotConfigured, match="Groq API key"):
        build_transcriber(None)  # groq requires its key


def test_build_transcriber_applies_the_provider_default_model() -> None:
    provider = build_transcriber("groq", api_key="gsk_test")
    assert isinstance(provider, GroqWhisperTranscriber)
    assert provider.model == "whisper-large-v3-turbo"
    provider = build_transcriber("groq", api_key="gsk_test", model="whisper-large-v3")
    assert provider.model == "whisper-large-v3"


def test_groq_requires_an_api_key() -> None:
    with pytest.raises(SttProviderNotConfigured, match="Groq API key"):
        build_transcriber("groq")


@pytest.mark.asyncio
async def test_groq_transcribes_via_the_hosted_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio = tmp_path / "note.ogg"
    audio.write_bytes(b"fake ogg audio")

    calls: dict = {}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> dict:
            return {"text": "hello from groq"}

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            calls["kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc) -> bool:
            return False

        async def post(self, url: str, **kwargs) -> FakeResponse:
            calls["url"] = url
            calls["post"] = kwargs
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    transcriber = build_transcriber("groq", api_key="gsk_test")
    text = await transcriber.transcribe(audio)
    assert text == "hello from groq"
    assert calls["url"] == GROQ_TRANSCRIPTIONS_URL
    assert calls["post"]["headers"] == {"Authorization": "Bearer gsk_test"}
    assert calls["post"]["data"] == {"model": "whisper-large-v3-turbo"}
    file_name, file_bytes, content_type = calls["post"]["files"]["file"]
    assert file_name == "note.ogg"
    assert file_bytes == b"fake ogg audio"
    assert content_type == "audio/ogg"


@pytest.mark.asyncio
async def test_groq_rejects_oversized_notes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio = tmp_path / "note.ogg"
    audio.write_bytes(b"x" * (GROQ_MAX_AUDIO_BYTES + 1))
    transcriber = build_transcriber("groq", api_key="gsk_test")
    with pytest.raises(SttProviderError, match="25 MB"):
        await transcriber.transcribe(audio)


@pytest.mark.asyncio
async def test_groq_surfaces_http_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio = tmp_path / "note.ogg"
    audio.write_bytes(b"fake ogg audio")

    class FailingClient:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc) -> bool:
            return False

        async def post(self, url: str, **kwargs) -> None:
            raise httpx.ConnectError("network down")

    monkeypatch.setattr(httpx, "AsyncClient", FailingClient)
    transcriber = build_transcriber("groq", api_key="gsk_test")
    with pytest.raises(SttProviderError, match="request failed"):
        await transcriber.transcribe(audio)


@pytest.mark.asyncio
async def test_groq_surfaces_empty_notes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio = tmp_path / "note.ogg"
    audio.write_bytes(b"")
    transcriber = build_transcriber("groq", api_key="gsk_test")
    with pytest.raises(SttProviderUnavailable, match="empty"):
        await transcriber.transcribe(audio)
