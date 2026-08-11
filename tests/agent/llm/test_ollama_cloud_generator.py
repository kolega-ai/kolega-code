"""Offline tests for the Ollama Cloud catalog transform and generator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from kolega_code.llm.specs import ollama_cloud_catalog as oc
from kolega_code.llm.specs.types import ThinkingEffortSpec

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "ollama_cloud" / "catalog_2026-08-10.json"
SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "refresh_ollama_cloud_catalog.py"


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def entries(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return oc.catalog_entries(payload)


def _combined(*models: tuple[str, list[Any], dict[str, Any]]) -> dict[str, Any]:
    return {
        "models": {"data": [{"id": identifier} for identifier, _capabilities, _model_info in models]},
        "details": {
            identifier: {"capabilities": capabilities, "model_info": model_info}
            for identifier, capabilities, model_info in models
        },
    }


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("test_refresh_ollama_cloud_catalog", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transform_parses_context_vision_and_thinking() -> None:
    specs = dict(
        oc.catalog_entries(
            _combined(
                (
                    "gpt-oss:test",
                    ["completion", "tools", "thinking"],
                    {"architecture.context_length": 524288},
                ),
                (
                    "vision-model",
                    ["completion", "tools", "vision"],
                    {"architecture.context_length": 262144},
                ),
            )
        )
    )

    assert specs["gpt-oss:test"]["context_length"] == 524288
    assert specs["vision-model"]["context_length"] == 262144
    assert specs["vision-model"]["supports_vision"] is True
    assert specs["gpt-oss:test"]["supports_vision"] is False
    assert "thinking_effort" not in specs["vision-model"]
    gpt_oss_effort = specs["gpt-oss:test"]["thinking_effort"]
    assert isinstance(gpt_oss_effort, ThinkingEffortSpec)
    assert gpt_oss_effort.options == ("low", "medium", "high")


def test_thinking_effort_is_exactly_the_reviewed_common_subset(entries: list[tuple[str, dict[str, Any]]]) -> None:
    efforts = [spec["thinking_effort"] for _identifier, spec in entries if "thinking_effort" in spec]
    assert efforts
    for effort in efforts:
        assert effort.options == ("low", "medium", "high")
        assert effort.default == "medium"
        assert effort.mode == "openai_reasoning_effort"


def test_reviewed_output_overrides_and_new_model_fallback() -> None:
    payload = _combined(
        *[
            (identifier, ["completion", "tools"], {f"{identifier}.context_length": 131072})
            for identifier in (*oc.MAX_COMPLETION_TOKEN_OVERRIDES, "new-model")
        ]
    )
    specs = dict(oc.catalog_entries(payload))
    for identifier in oc.MAX_COMPLETION_TOKEN_OVERRIDES:
        assert specs[identifier]["max_completion_tokens"] == 65536
    assert specs["new-model"]["max_completion_tokens"] == 32768


def test_entries_are_sorted_alphabetically(entries: list[tuple[str, dict[str, Any]]]) -> None:
    identifiers = [identifier for identifier, _spec in entries]
    assert identifiers == sorted(identifiers)
    assert oc.catalog_entries(json.loads(FIXTURE.read_text(encoding="utf-8"))) == entries


def test_models_missing_completion_or_tools_are_filtered() -> None:
    result = oc.transform_catalog(
        _combined(
            ("tools-only", ["tools"], {"x.context_length": 1000}),
            ("completion-only", ["completion"], {"x.context_length": 1000}),
            ("eligible", ["completion", "tools"], {"x.context_length": 1000}),
        )
    )
    assert [identifier for identifier, _spec in result.entries] == ["eligible"]
    assert result.filtered_ids == ("completion-only", "tools-only")


def test_duplicate_ids_and_missing_details_are_rejected() -> None:
    duplicate = _combined(("same", ["completion", "tools"], {"x.context_length": 1000}))
    duplicate["models"]["data"].append({"id": "same"})
    with pytest.raises(oc.OllamaCloudCatalogError, match="Duplicate"):
        oc.transform_catalog(duplicate)

    missing = _combined(("missing", ["completion", "tools"], {"x.context_length": 1000}))
    del missing["details"]["missing"]
    with pytest.raises(oc.OllamaCloudCatalogError, match="Missing .* details"):
        oc.transform_catalog(missing)


@pytest.mark.parametrize("capabilities", [None, "completion,tools", ["completion", 7], [""]])
def test_invalid_capabilities_are_rejected(capabilities: Any) -> None:
    payload = _combined(("bad", capabilities, {"x.context_length": 1000}))
    with pytest.raises(oc.OllamaCloudCatalogError, match="capabilit"):
        oc.transform_catalog(payload)


@pytest.mark.parametrize(
    "model_info",
    [
        {},
        {"x.context_length": 0},
        {"x.context_length": 1000, "y.context_length": 2000},
    ],
)
def test_zero_or_multiple_context_fields_are_rejected(model_info: dict[str, Any]) -> None:
    payload = _combined(("bad", ["completion", "tools"], model_info))
    with pytest.raises(oc.OllamaCloudCatalogError, match="context_length"):
        oc.transform_catalog(payload)


def test_rendered_source_compiles_and_imports(entries: list[tuple[str, dict[str, Any]]], tmp_path: Path) -> None:
    source = oc.render_catalog_module(entries, header_lines=["Generated by a test."])
    compile(source, "generated_ollama_cloud.py", "exec")
    path = tmp_path / "generated_ollama_cloud.py"
    path.write_text(source, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("generated_ollama_cloud", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.OLLAMA_CLOUD_SPECS == {("ollama_cloud", identifier): model_spec for identifier, model_spec in entries}


def test_cache_round_trip(entries: list[tuple[str, dict[str, Any]]], tmp_path: Path) -> None:
    path = tmp_path / "ollama.json"
    oc.save_cache(path, entries, fetched_at="2026-08-10T00:00:00Z")
    assert oc.load_cache(path) == entries


@pytest.mark.parametrize(
    "contents",
    [
        "{",
        json.dumps({"schema_version": 999, "provider": "ollama_cloud", "models": []}),
        json.dumps({"schema_version": oc.CACHE_SCHEMA_VERSION, "provider": "other", "models": []}),
        json.dumps({"schema_version": oc.CACHE_SCHEMA_VERSION, "provider": "ollama_cloud", "models": "bad"}),
        json.dumps(
            {
                "schema_version": oc.CACHE_SCHEMA_VERSION,
                "provider": "ollama_cloud",
                "models": [{"id": "x", "spec": {}}],
            }
        ),
    ],
)
def test_invalid_cache_is_ignored(contents: str, tmp_path: Path) -> None:
    path = tmp_path / "ollama.json"
    path.write_text(contents, encoding="utf-8")
    assert oc.load_cache(path) == []


@pytest.mark.parametrize("api_key, expected_headers", [(None, {}), ("secret", {"Authorization": "Bearer secret"})])
def test_fetch_models_uses_all_ids_and_optional_authorization(
    monkeypatch: pytest.MonkeyPatch,
    api_key: str | None,
    expected_headers: dict[str, str],
) -> None:
    calls: list[tuple[str, Any]] = []

    class Response:
        def __init__(self, data: Any) -> None:
            self.data = data

        def raise_for_status(self) -> None:
            return None

        def json(self) -> Any:
            return self.data

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["headers"] == expected_headers

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def get(self, url: str) -> Response:
            calls.append(("get", url))
            return Response({"data": [{"id": "z-model"}, {"id": "a-model"}]})

        def post(self, url: str, *, json: Any) -> Response:
            calls.append(("post", (url, json)))
            return Response({"capabilities": ["completion", "tools"], "model_info": {"x.context_length": 1000}})

    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setattr(oc.httpx, "Client", Client)
    result = oc.fetch_models(api_key=api_key)
    assert list(result["details"]) == ["z-model", "a-model"]
    assert calls == [
        ("get", oc.MODELS_URL),
        ("post", (oc.SHOW_URL, {"model": "z-model"})),
        ("post", (oc.SHOW_URL, {"model": "a-model"})),
    ]


def test_fetch_models_reports_the_model_whose_detail_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __init__(self, data: Any, *, fails: bool = False) -> None:
            self.data = data
            self.fails = fails

        def raise_for_status(self) -> None:
            if self.fails:
                raise RuntimeError("detail unavailable")

        def json(self) -> Any:
            return self.data

    class Client:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def get(self, _url: str) -> Response:
            return Response({"data": [{"id": "good"}, {"id": "broken"}]})

        def post(self, _url: str, *, json: dict[str, str]) -> Response:
            return Response({}, fails=json["model"] == "broken")

    monkeypatch.setattr(oc.httpx, "Client", Client)
    with pytest.raises(oc.OllamaCloudCatalogError, match="'broken'.*detail unavailable"):
        oc.fetch_models()


def test_generator_offline_check_matches_semantics_and_detects_drift(
    payload: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _load_generator()
    output = tmp_path / "ollama_cloud.py"
    entries = oc.catalog_entries(payload)
    output.write_text(oc.render_catalog_module(entries, header_lines=["Old date: 1999-01-01"]), encoding="utf-8")
    monkeypatch.setattr(
        generator.catalog, "fetch_models", lambda: pytest.fail("offline check attempted network access")
    )

    args = ["--snapshot-file", str(FIXTURE), "--output", str(output), "--check"]
    original = output.read_text(encoding="utf-8")
    assert generator.main(args) == 0
    assert "up to date" in capsys.readouterr().out
    assert output.read_text(encoding="utf-8") == original

    drifted = [(identifier, dict(spec)) for identifier, spec in entries]
    drifted[0][1]["context_length"] += 1
    output.write_text(oc.render_catalog_module(drifted), encoding="utf-8")
    drifted_source = output.read_text(encoding="utf-8")
    assert generator.main(args) == 1
    assert "drift detected" in capsys.readouterr().out
    assert output.read_text(encoding="utf-8") == drifted_source
