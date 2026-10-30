"""Remote STT provider registry, resolution, and multipart uploads."""

from pathlib import Path
from typing import Any

import httpx
import pytest

from kolega_code.cli.settings import CliSettings
from kolega_code.gateway.stt import (
    DEFAULT_STT_PROVIDER,
    GROQ_MAX_AUDIO_BYTES,
    GROQ_TRANSCRIPTIONS_URL,
    OPENAI_TRANSCRIPTIONS_URL,
    STT_MAX_AUDIO_BYTES,
    GroqWhisperTranscriber,
    OpenAITranscriber,
    SttProviderError,
    SttProviderNotConfigured,
    SttProviderUnavailable,
    available_stt_providers,
    build_transcriber,
    get_stt_provider_class,
    resolve_stt,
    stt_provider_names,
)


def test_registry_lists_groq_first_and_openai_second() -> None:
    assert stt_provider_names() == ["groq", "openai"]
    assert available_stt_providers() == [
        ("Groq (whisper-large-v3-turbo)", "groq"),
        ("OpenAI (gpt-transcribe)", "openai"),
    ]
    assert get_stt_provider_class("groq") is GroqWhisperTranscriber
    assert get_stt_provider_class("openai") is OpenAITranscriber
    with pytest.raises(SttProviderError, match="Unknown speech-to-text provider"):
        get_stt_provider_class("bogus")


def test_build_transcriber_keeps_groq_as_the_default() -> None:
    assert DEFAULT_STT_PROVIDER == "groq"
    with pytest.raises(SttProviderNotConfigured, match="Groq API key"):
        build_transcriber(None)


@pytest.mark.parametrize(
    ("name", "api_key", "provider_type", "default_model"),
    [
        ("groq", "gsk_test", GroqWhisperTranscriber, "whisper-large-v3-turbo"),
        ("openai", "sk-test", OpenAITranscriber, "gpt-transcribe"),
    ],
)
def test_build_transcriber_applies_provider_defaults_and_model_overrides(
    name: str,
    api_key: str,
    provider_type: type,
    default_model: str,
) -> None:
    provider = build_transcriber(name, api_key=api_key)
    assert isinstance(provider, provider_type)
    assert provider.model == default_model
    provider = build_transcriber(name, api_key=api_key, model="custom-transcription-model")
    assert provider.model == "custom-transcription-model"


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("groq", "Groq API key"),
        ("openai", "OpenAI API key"),
    ],
)
def test_remote_providers_require_their_api_keys(name: str, message: str) -> None:
    with pytest.raises(SttProviderNotConfigured, match=message):
        build_transcriber(name)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_name", "api_key", "expected_url", "expected_model"),
    [
        ("groq", "gsk_test", GROQ_TRANSCRIPTIONS_URL, "whisper-large-v3-turbo"),
        ("openai", "sk-test", OPENAI_TRANSCRIPTIONS_URL, "gpt-transcribe"),
    ],
)
@pytest.mark.parametrize(
    ("input_kind", "expected_filename", "expected_content_type"),
    [
        ("path", "note.ogg", "audio/ogg"),
        ("bytes", "voice.webm", "audio/webm"),
    ],
)
async def test_providers_share_path_and_webm_bytes_multipart_uploads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider_name: str,
    api_key: str,
    expected_url: str,
    expected_model: str,
    input_kind: str,
    expected_filename: str,
    expected_content_type: str,
) -> None:
    audio_bytes = b"fake audio"
    if input_kind == "path":
        audio: Path | bytes = tmp_path / "note.ogg"
        audio.write_bytes(audio_bytes)
    else:
        audio = audio_bytes
    calls: dict[str, Any] = {}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> dict[str, str]:
            return {"text": " shared transcription "}

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            calls["kwargs"] = kwargs

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def post(self, url: str, **kwargs: Any) -> FakeResponse:
            calls["url"] = url
            calls["post"] = kwargs
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    text = await build_transcriber(provider_name, api_key=api_key).transcribe(audio)

    assert text == "shared transcription"
    assert calls["url"] == expected_url
    assert calls["post"]["headers"] == {"Authorization": f"Bearer {api_key}"}
    assert calls["post"]["data"] == {"model": expected_model}
    filename, uploaded_bytes, content_type = calls["post"]["files"]["file"]
    assert filename == expected_filename
    assert uploaded_bytes == audio_bytes
    assert content_type == expected_content_type


@pytest.mark.asyncio
async def test_openai_rejects_audio_over_the_shared_25_mb_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    assert STT_MAX_AUDIO_BYTES == 25 * 1024 * 1024
    assert GROQ_MAX_AUDIO_BYTES == STT_MAX_AUDIO_BYTES
    transcriber = build_transcriber("openai", api_key="sk-test")
    monkeypatch.setattr(transcriber, "max_audio_bytes", 1024 * 1024)

    with pytest.raises(SttProviderError, match="OpenAI accepts at most 1 MB"):
        await transcriber.transcribe(b"x" * (1024 * 1024 + 1))


@pytest.mark.asyncio
@pytest.mark.parametrize("input_kind", ["path", "bytes"])
async def test_empty_path_and_bytes_inputs_are_rejected(
    tmp_path: Path,
    input_kind: str,
) -> None:
    if input_kind == "path":
        audio: Path | bytes = tmp_path / "empty.ogg"
        audio.write_bytes(b"")
    else:
        audio = b""

    with pytest.raises(SttProviderUnavailable, match="empty"):
        await build_transcriber("groq", api_key="gsk_test").transcribe(audio)


@pytest.mark.asyncio
async def test_openai_surfaces_http_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "FailingClient":
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def post(self, url: str, **kwargs: Any) -> None:
            raise httpx.ConnectError("network down")

    monkeypatch.setattr(httpx, "AsyncClient", FailingClient)
    with pytest.raises(SttProviderError, match="OpenAI transcription request failed: network down"):
        await build_transcriber("openai", api_key="sk-test").transcribe(b"audio")


@pytest.mark.asyncio
async def test_openai_surfaces_bounded_api_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status_code = 400
        text = "bad request " * 100

    class FailingClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "FailingClient":
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def post(self, url: str, **kwargs: Any) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FailingClient)
    with pytest.raises(SttProviderError) as exc_info:
        await build_transcriber("openai", api_key="sk-test").transcribe(b"audio")

    message = str(exc_info.value)
    assert message.startswith("OpenAI transcription failed (400): bad request")
    assert len(message) <= 240


def test_resolve_stt_defaults_to_disabled_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert resolve_stt(None) == (False, "groq", None, None)


def test_resolve_stt_uses_the_selected_providers_stored_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = CliSettings(
        stt_enabled=True,
        stt_provider="openai",
        stt_model="gpt-4o-mini-transcribe",
        api_keys={"openai": "sk-stored"},
    )

    assert resolve_stt(settings) == (True, "openai", "gpt-4o-mini-transcribe", "sk-stored")


def test_resolve_stt_environment_key_beats_stored_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    settings = CliSettings(stt_provider="openai", api_keys={"openai": "sk-stored"})

    assert resolve_stt(settings) == (False, "openai", None, "sk-env")


def test_resolve_stt_unknown_provider_falls_back_to_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    settings = CliSettings(
        stt_enabled=True,
        stt_provider="missing",
        stt_model="custom-model",
        api_keys={"groq": "gsk-stored", "missing": "ignored"},
    )

    assert resolve_stt(settings) == (True, "groq", "custom-model", "gsk-stored")
