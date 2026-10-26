"""Cloudflare Workers AI ``models refresh`` overlay: conversion, fetch, cache, and CLI wiring (no network)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from kolega_code.cli import model_catalog
from kolega_code.llm.specs import MODEL_SPECS
from kolega_code.llm.specs import cloudflare_workers_ai_catalog as catalog
from kolega_code.llm.specs.types import ThinkingEffortSpec

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "cloudflare_workers_ai" / "models_search.json"
ACCOUNT_ID = "0123456789abcdef0123456789abcdef"


def _payload() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _row(name: str, *, task: str = "Text Generation", deprecated: Any = False, **properties: Any) -> dict[str, Any]:
    return {
        "name": name,
        "task": {"name": task},
        "deprecated": deprecated,
        "properties": [{"property_id": key, "value": value} for key, value in properties.items()],
    }


def _entries(*rows: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return dict(catalog.catalog_entries({"success": True, "result": list(rows)}))


@pytest.fixture
def restore_catalog():
    before = dict(MODEL_SPECS)
    yield
    MODEL_SPECS.clear()
    MODEL_SPECS.update(before)
    model_catalog.rebuild_ui_model_options()


# --- conversion ------------------------------------------------------------


def test_only_agent_capable_text_generation_models_survive_the_published_catalog() -> None:
    entries = dict(catalog.catalog_entries(_payload()))

    assert "@cf/zai-org/glm-5.3" in entries
    assert "@cf/openai/gpt-oss-20b" in entries
    # 24K/32K windows, no tool calling, embeddings, and LoRA bases are dropped.
    assert "@cf/meta/llama-3.3-70b-instruct-fp8-fast" not in entries
    assert "@cf/qwen/qwen3-30b-a3b-fp8" not in entries
    assert "@cf/cloudflare/clef" not in entries
    assert "@cf/baai/bge-base-en-v1.5" not in entries
    assert not any(identifier.endswith("-lora") for identifier in entries)


@pytest.mark.parametrize(
    "row",
    [
        _row("@cf/x/embed", task="Text Embeddings", context_window="262144", function_calling="true"),
        _row("@cf/x/no-tools", context_window="262144"),
        _row("@cf/x/small", context_window="32768", function_calling="true"),
        _row("@cf/x/retired", deprecated=True, context_window="262144", function_calling="true"),
        _row("@cf/x/adapter-lora", context_window="262144", function_calling="true"),
        _row("@cf/x/no-window", function_calling="true"),
    ],
)
def test_filtered_rows(row: dict[str, Any]) -> None:
    assert _entries(row) == {}


def test_spec_fields_follow_the_metadata() -> None:
    spec = _entries(
        _row(
            "@cf/x/model",
            context_window="131072",
            function_calling=True,
            vision="true",
            reasoning_effort={"supported_efforts": ["max", "unknown", "low", "none"], "default_effort": "low"},
        )
    )["@cf/x/model"]

    assert spec["context_length"] == 131072
    assert spec["max_completion_tokens"] == 32768
    assert spec["input_budget"] == "window_minus_output"
    assert spec["supports_vision"] is True
    assert spec["thinking_effort"] == ThinkingEffortSpec(
        options=("none", "low", "max"), default="low", mode="cloudflare_reasoning_effort"
    )
    assert "preferred_edit_protocol" not in spec


def test_small_windows_reserve_a_quarter_for_output() -> None:
    spec = _entries(_row("@cf/x/model", context_window="65536", function_calling="true"))["@cf/x/model"]
    assert spec["max_completion_tokens"] == 16384


def test_invalid_default_effort_falls_back_to_the_highest_option() -> None:
    spec = _entries(
        _row(
            "@cf/x/model",
            context_window="262144",
            function_calling="true",
            reasoning_effort={"supported_efforts": ["low", "high"], "default_effort": "max"},
        )
    )["@cf/x/model"]
    assert spec["thinking_effort"].default == "high"


def test_mandatory_reasoning_without_a_ladder_gets_no_effort_control() -> None:
    spec = _entries(
        _row(
            "@cf/x/model",
            context_window="262144",
            function_calling="true",
            reasoning_effort={"mandatory": True, "default_enabled": True},
        )
    )["@cf/x/model"]
    assert "thinking_effort" not in spec


def test_openai_open_models_prefer_codex_apply_patch() -> None:
    spec = _entries(_row("@cf/openai/gpt-oss-next", context_window="131072", function_calling="true"))
    assert spec["@cf/openai/gpt-oss-next"]["preferred_edit_protocol"] == "codex_apply_patch"


def test_payload_must_carry_a_result_list() -> None:
    with pytest.raises(catalog.CloudflareWorkersAICatalogError):
        catalog.catalog_entries({"success": True, "result": "nope"})


# --- fetch -----------------------------------------------------------------


def test_fetch_follows_pagination_until_a_short_page() -> None:
    rows = [_row(f"@cf/x/m{index}", context_window="131072", function_calling="true") for index in range(150)]
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        page = int(request.url.params["page"])
        per_page = int(request.url.params["per_page"])
        chunk = rows[(page - 1) * per_page : page * per_page]
        return httpx.Response(200, json={"success": True, "errors": [], "result": chunk})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        payload = catalog.fetch_models(api_key="cf-test-token", account_id=ACCOUNT_ID, client=client)

    assert len(payload["result"]) == 150
    assert [request.url.params["page"] for request in seen] == ["1", "2"]
    assert all(request.url.params["task"] == "Text Generation" for request in seen)
    assert seen[0].url.path == f"/client/v4/accounts/{ACCOUNT_ID}/ai/models/search"
    assert seen[0].headers["authorization"] == "Bearer cf-test-token"


def test_fetch_raises_on_an_unsuccessful_envelope() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"success": False, "errors": [{"code": 10000, "message": "auth"}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(catalog.CloudflareWorkersAICatalogError, match="auth"):
            catalog.fetch_models(api_key="cf-test-token", account_id=ACCOUNT_ID, client=client)


def test_fetch_raises_on_http_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"success": False})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            catalog.fetch_models(api_key="cf-test-token", account_id=ACCOUNT_ID, client=client)


# --- cache -----------------------------------------------------------------


def test_cache_round_trip_preserves_thinking_specs(tmp_path: Path) -> None:
    entries = catalog.catalog_entries(_payload())
    path = tmp_path / catalog.CACHE_FILENAME
    catalog.save_cache(path, entries, fetched_at="2026-10-23T00:00:00+00:00")

    assert catalog.load_cache(path) == entries


@pytest.mark.parametrize("content", ["", "{", "[]", '{"schema_version": 99, "models": []}'])
def test_unusable_cache_files_load_as_empty(tmp_path: Path, content: str) -> None:
    path = tmp_path / catalog.CACHE_FILENAME
    path.write_text(content, encoding="utf-8")
    assert catalog.load_cache(path) == []


def test_cache_skips_invalid_entries(tmp_path: Path) -> None:
    path = tmp_path / catalog.CACHE_FILENAME
    good = catalog.spec_to_jsonable(
        _entries(_row("@cf/x/ok", context_window="131072", function_calling="true"))["@cf/x/ok"]
    )
    bad_effort = dict(
        good, thinking_effort={"options": ["low"], "default": "max", "mode": "cloudflare_reasoning_effort"}
    )
    path.write_text(
        json.dumps(
            {
                "schema_version": catalog.CACHE_SCHEMA_VERSION,
                "models": [
                    {"id": "@cf/x/ok", "spec": good},
                    {"id": "@cf/x/bad-effort", "spec": bad_effort},
                    {"id": "@cf/x/bad-budget", "spec": dict(good, input_budget="nope")},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert [identifier for identifier, _ in catalog.load_cache(path)] == ["@cf/x/ok"]


def test_overlay_adds_new_models_without_overriding_curated_specs(tmp_path: Path, restore_catalog) -> None:
    entries = catalog.catalog_entries(_payload())
    catalog.save_cache(tmp_path / catalog.CACHE_FILENAME, entries, fetched_at="2026-10-23T00:00:00+00:00")

    added = model_catalog.apply_catalog_overlay(tmp_path, env={}, catalog=catalog.PROVIDER)

    assert added > 0
    assert (catalog.PROVIDER, "@cf/meta/llama-4-scout-17b-16e-instruct") in MODEL_SPECS
    assert MODEL_SPECS[(catalog.PROVIDER, "@cf/zai-org/glm-5.3")]["max_completion_tokens"] == 131072
    assert (
        model_catalog.apply_catalog_overlay(
            tmp_path, env={"KOLEGA_CODE_DISABLE_CLOUDFLARE_WORKERS_AI_CATALOG": "1"}, catalog=catalog.PROVIDER
        )
        == 0
    )


# --- ``models refresh`` ------------------------------------------------------


def _run_refresh(tmp_path: Path) -> tuple[int, list[str], list[str]]:
    lines: list[str] = []
    errors: list[str] = []
    exit_code = model_catalog.run_models_refresh(
        argparse.Namespace(provider=catalog.PROVIDER, state_dir=tmp_path), lines.append, errors.append
    )
    return exit_code, lines, errors


def _stub_fetch(monkeypatch: pytest.MonkeyPatch, seen: list[dict[str, Any]]) -> None:
    def fake_fetch(**kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs)
        return _payload()

    monkeypatch.setattr(catalog, "fetch_models", fake_fetch)


def test_refresh_requires_the_token(isolated_cli_env: None, tmp_path: Path) -> None:
    exit_code, lines, errors = _run_refresh(tmp_path)

    assert exit_code == 2
    assert "CLOUDFLARE_API_TOKEN" in "\n".join(errors)
    assert lines == []


def test_refresh_requires_the_account_id(isolated_cli_env: None, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test-token")

    exit_code, _lines, errors = _run_refresh(tmp_path)

    assert exit_code == 2
    assert "CLOUDFLARE_ACCOUNT_ID" in "\n".join(errors)


def test_refresh_prefers_the_environment_over_settings(isolated_cli_env: None, tmp_path: Path, monkeypatch) -> None:
    from kolega_code.cli.settings import SettingsStore

    store = SettingsStore()
    settings = store.load()
    settings.set_api_key(catalog.PROVIDER, "saved-token")
    settings.cloudflare_account_id = "fedcba9876543210fedcba9876543210"
    store.save(settings)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "exported-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT_ID.upper())
    seen: list[dict[str, Any]] = []
    _stub_fetch(monkeypatch, seen)

    exit_code, _lines, errors = _run_refresh(tmp_path)

    assert exit_code == 0, errors
    assert seen == [{"api_key": "exported-token", "account_id": ACCOUNT_ID}]


def test_refresh_falls_back_to_settings_and_writes_its_cache(
    isolated_cli_env: None, tmp_path: Path, monkeypatch
) -> None:
    from kolega_code.cli.settings import SettingsStore

    store = SettingsStore()
    settings = store.load()
    settings.set_api_key(catalog.PROVIDER, "saved-token")
    settings.cloudflare_account_id = ACCOUNT_ID
    store.save(settings)
    seen: list[dict[str, Any]] = []
    _stub_fetch(monkeypatch, seen)

    exit_code, lines, errors = _run_refresh(tmp_path)

    assert exit_code == 0, errors
    assert seen == [{"api_key": "saved-token", "account_id": ACCOUNT_ID}]
    cached = dict(catalog.load_cache(tmp_path / catalog.CACHE_FILENAME))
    assert "@cf/meta/llama-4-scout-17b-16e-instruct" in cached
    assert "not in the bundled catalog" in "\n".join(lines)
    assert "{account_id}" in "\n".join(lines)
    assert ACCOUNT_ID not in "\n".join(lines)
