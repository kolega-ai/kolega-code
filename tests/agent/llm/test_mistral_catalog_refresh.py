"""Mistral catalog-overlay conversion and cache, without network access.

The payload fixture mirrors the shape of ``GET /v1/models``: several rows share
one underlying model through ``name``/``billing_model_name``, non-chat endpoints
are published in the same list, and one entry is organization-gated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kolega_code.llm.specs import mistral_catalog
from kolega_code.llm.specs.types import ThinkingEffortSpec

_CAPABILITY_NAMES = (
    "completion_chat",
    "function_calling",
    "reasoning",
    "completion_fim",
    "fine_tuning",
    "vision",
    "ocr",
    "classification",
    "moderation",
    "audio",
    "audio_transcription",
    "audio_transcription_realtime",
    "audio_speech",
    "unified_resources",
)


def _capabilities(**enabled: bool) -> dict[str, bool]:
    capabilities = {name: False for name in _CAPABILITY_NAMES}
    capabilities.update(enabled)
    return capabilities


def _entry(
    identifier: str,
    *,
    name: str | None = None,
    billing: str | None = None,
    context: int | None = 262144,
    temperature: float = 1.0,
    capabilities: dict[str, bool] | None = None,
    deprecation: dict | None = None,
) -> dict:
    return {
        "id": identifier,
        "object": "model",
        "owned_by": "mistralai",
        "name": name or identifier,
        "billing_model_name": billing or identifier,
        "max_context_length": context,
        "default_model_temperature": temperature,
        "aliases": [],
        "deprecation": deprecation,
        "capabilities": (
            capabilities if capabilities is not None else _capabilities(completion_chat=True, function_calling=True)
        ),
    }


_CHAT_VISION_REASONING = _capabilities(
    completion_chat=True,
    function_calling=True,
    reasoning=True,
    vision=True,
)


def _payload() -> dict:
    return {
        "object": "list",
        "data": [
            # One model, three spellings, all sharing the same billing identity.
            _entry(
                "mistral-medium-3-5",
                name="mistral-medium-latest",
                billing="mistral-medium-3-5",
                capabilities=_CHAT_VISION_REASONING,
            ),
            _entry(
                "mistral-medium-latest",
                name="mistral-medium-latest",
                billing="mistral-medium-3-5",
                capabilities=_CHAT_VISION_REASONING,
            ),
            _entry(
                "mistral-vibe-cli-latest",
                name="mistral-medium-latest",
                billing="mistral-medium-3-5",
                capabilities=_CHAT_VISION_REASONING,
            ),
            # Billing id differs from the marketing id; both are listed.
            _entry(
                "mistral-large-4",
                name="mistral-large-4",
                billing="mistral-large-4-0",
                context=1048576,
                capabilities=_CHAT_VISION_REASONING,
            ),
            _entry(
                "mistral-large-4-0",
                name="mistral-large-4",
                billing="mistral-large-4-0",
                context=1048576,
                capabilities=_CHAT_VISION_REASONING,
            ),
            # Coding model: text only, low default temperature, FIM variant.
            _entry(
                "codestral-2508",
                context=256000,
                temperature=0.3,
                capabilities=_capabilities(
                    completion_chat=True,
                    function_calling=True,
                    completion_fim=True,
                ),
            ),
            _entry(
                "mistral-code-latest",
                name="codestral-2508",
                billing="codestral-2508",
                context=256000,
                temperature=0.3,
                capabilities=_capabilities(
                    completion_chat=True,
                    function_calling=True,
                    completion_fim=True,
                ),
            ),
            # Small vision model with no reasoning control.
            _entry(
                "ministral-3b-2512",
                context=131072,
                temperature=0.3,
                capabilities=_capabilities(
                    completion_chat=True,
                    function_calling=True,
                    vision=True,
                ),
            ),
            # Third-party model hosted by Mistral: reasoning is reported, but the
            # native effort control rejects "none", so it must not be attached.
            _entry(
                "zai-glm-5-3",
                context=1048576,
                capabilities=_capabilities(
                    completion_chat=True,
                    function_calling=True,
                    reasoning=True,
                ),
            ),
            # Audio-first chat endpoint: excluded even though completion_chat is true.
            _entry(
                "voxtral-small-2507",
                name="Voxtral Small Transcribe",
                context=32768,
                capabilities=_capabilities(
                    completion_chat=True,
                    function_calling=True,
                    audio=True,
                ),
            ),
            # Non-chat endpoints published in the same catalog.
            _entry(
                "mistral-ocr-4",
                context=16384,
                capabilities=_capabilities(ocr=True, vision=True, function_calling=True),
            ),
            _entry("codestral-embed", context=8192, capabilities=_capabilities()),
            # Organization-gated research model.
            _entry("labs-leanstral-1-5", capabilities=_CHAT_VISION_REASONING),
            # Retired model.
            _entry(
                "mistral-small-3-2",
                capabilities=_CHAT_VISION_REASONING,
                deprecation={"replacement": "mistral-small-2603"},
            ),
        ],
    }


def _by_id(entries: list[tuple[str, dict]]) -> dict[str, dict]:
    return {identifier: spec for identifier, spec in entries}


# --- canonical grouping ---------------------------------------------------


def test_aliases_collapse_onto_one_canonical_entry() -> None:
    entries = mistral_catalog.catalog_entries(_payload())

    assert [identifier for identifier, _spec in entries] == [
        "mistral-medium-3-5",
        "mistral-large-4-0",
        "codestral-2508",
        "ministral-3b-2512",
        "zai-glm-5-3",
    ]


def test_alias_ids_are_not_emitted() -> None:
    catalogued = _by_id(mistral_catalog.catalog_entries(_payload()))

    for alias in ("mistral-medium-latest", "mistral-vibe-cli-latest", "mistral-code-latest", "mistral-large-4"):
        assert alias not in catalogued


# --- exclusions -----------------------------------------------------------


def test_audio_ocr_and_embedding_endpoints_are_excluded() -> None:
    catalogued = _by_id(mistral_catalog.catalog_entries(_payload()))

    assert "voxtral-small-2507" not in catalogued
    assert "mistral-ocr-4" not in catalogued
    assert "codestral-embed" not in catalogued


def test_labs_and_retired_models_are_excluded() -> None:
    catalogued = _by_id(mistral_catalog.catalog_entries(_payload()))

    assert "labs-leanstral-1-5" not in catalogued
    assert "mistral-small-3-2" not in catalogued


def test_entries_without_a_context_window_are_skipped() -> None:
    payload = {"data": [_entry("mistral-medium-3-5", context=None)]}

    assert mistral_catalog.catalog_entries(payload) == []


def test_payload_must_be_a_model_list() -> None:
    with pytest.raises(mistral_catalog.MistralCatalogError):
        mistral_catalog.catalog_entries("not-json")

    with pytest.raises(mistral_catalog.MistralCatalogError):
        mistral_catalog.catalog_entries({"object": "list"})


# --- spec mapping ---------------------------------------------------------


def test_spec_fields_follow_model_metadata() -> None:
    catalogued = _by_id(mistral_catalog.catalog_entries(_payload()))

    medium = catalogued["mistral-medium-3-5"]
    assert medium["context_length"] == 262144
    assert medium["max_completion_tokens"] == mistral_catalog.MAX_COMPLETION_TOKENS
    assert medium["input_budget"] == "window_minus_output"
    assert medium["default_temperature"] == 1.0
    assert medium["supports_vision"] is True
    assert medium["supports_function_calling"] is True
    assert medium["supports_hosted_web_search"] is False
    assert "featured" not in medium

    large = catalogued["mistral-large-4-0"]
    assert large["context_length"] == 1048576

    codestral = catalogued["codestral-2508"]
    assert codestral["context_length"] == 256000
    assert codestral["default_temperature"] == 0.3
    assert codestral["supports_vision"] is False


def test_first_party_reasoning_models_get_the_none_high_control() -> None:
    catalogued = _by_id(mistral_catalog.catalog_entries(_payload()))

    effort = catalogued["mistral-medium-3-5"]["thinking_effort"]
    assert isinstance(effort, ThinkingEffortSpec)
    assert effort.options == ("none", "high")
    assert effort.default == "none"
    assert effort.mode == "mistral_effort"


def test_third_party_reasoning_models_get_no_effort_control() -> None:
    """Mistral-hosted Z.ai models reject ``reasoning_effort: none``."""

    catalogued = _by_id(mistral_catalog.catalog_entries(_payload()))

    assert "thinking_effort" not in catalogued["zai-glm-5-3"]


def test_non_reasoning_models_get_no_effort_control() -> None:
    catalogued = _by_id(mistral_catalog.catalog_entries(_payload()))

    assert "thinking_effort" not in catalogued["ministral-3b-2512"]
    assert "thinking_effort" not in catalogued["codestral-2508"]


# --- cache ----------------------------------------------------------------


def test_cache_round_trip_preserves_the_thinking_spec(tmp_path: Path) -> None:
    entries = mistral_catalog.catalog_entries(_payload())
    path = tmp_path / mistral_catalog.CACHE_FILENAME

    mistral_catalog.save_cache(path, entries, fetched_at="2026-10-21T00:00:00+00:00")
    loaded = mistral_catalog.load_cache(path)

    assert [identifier for identifier, _spec in loaded] == [identifier for identifier, _spec in entries]
    effort = _by_id(loaded)["mistral-medium-3-5"]["thinking_effort"]
    assert isinstance(effort, ThinkingEffortSpec)
    assert effort.options == ("none", "high")
    assert effort.mode == "mistral_effort"


def test_cache_rejects_a_different_schema_version(tmp_path: Path) -> None:
    path = tmp_path / mistral_catalog.CACHE_FILENAME
    path.write_text(
        json.dumps({"schema_version": mistral_catalog.CACHE_SCHEMA_VERSION + 1, "models": []}),
        encoding="utf-8",
    )

    assert mistral_catalog.load_cache(path) == []


def test_cache_tolerates_malformed_files(tmp_path: Path) -> None:
    path = tmp_path / mistral_catalog.CACHE_FILENAME
    path.write_text("{not json", encoding="utf-8")

    assert mistral_catalog.load_cache(path) == []
    assert mistral_catalog.load_cache(tmp_path / "missing.json") == []


def test_cache_skips_invalid_individual_entries(tmp_path: Path) -> None:
    path = tmp_path / mistral_catalog.CACHE_FILENAME
    path.write_text(
        json.dumps(
            {
                "schema_version": mistral_catalog.CACHE_SCHEMA_VERSION,
                "models": [
                    {
                        "id": "valid",
                        "spec": {
                            "context_length": 1000,
                            "max_completion_tokens": 100,
                            "input_budget": "window_minus_output",
                        },
                    },
                    {
                        "id": "invalid",
                        "spec": {
                            "context_length": 0,
                            "max_completion_tokens": 100,
                            "input_budget": "window_minus_output",
                        },
                    },
                    {"id": "not-a-spec", "spec": "nope"},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert [identifier for identifier, _spec in mistral_catalog.load_cache(path)] == ["valid"]
