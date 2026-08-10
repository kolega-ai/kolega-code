"""``kolega-code models`` and the optional OpenRouter catalog overlay.

No test here performs network I/O: the fetch is always stubbed.
"""

import argparse
import json
from pathlib import Path

import pytest

from kolega_code.cli import model_catalog
from kolega_code.llm.specs import MODEL_SPECS
from kolega_code.llm.specs import ollama_cloud_catalog as ollama_cloud
from kolega_code.llm.specs import openrouter_catalog as openrouter
from kolega_code.llm.specs.types import ThinkingEffortSpec

PROVIDER = openrouter.PROVIDER


@pytest.fixture
def restore_catalog():
    """Undo any overlay a test merges into the process-wide catalog."""
    before = dict(MODEL_SPECS)
    yield
    MODEL_SPECS.clear()
    MODEL_SPECS.update(before)
    model_catalog.rebuild_ui_model_options()


def _args(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def _collect():
    lines: list[str] = []
    return lines, lines.append


# --- models list ----------------------------------------------------------


def test_list_defaults_to_catalog_order_and_ranks_per_provider() -> None:
    rows = model_catalog.catalog_rows(provider=PROVIDER)

    catalogued = [model for provider, model in MODEL_SPECS if provider == PROVIDER]
    assert [row["model"] for row in rows] == catalogued
    # Catalog order is the gateway's usage ranking, so rank 1 is the most used.
    assert [row["rank"] for row in rows] == list(range(1, len(catalogued) + 1))


def test_list_can_sort_alphabetically_without_losing_rank() -> None:
    rows = model_catalog.catalog_rows(provider=PROVIDER, sort="id")

    assert [row["model"] for row in rows] == sorted(row["model"] for row in rows)
    # Rank still reports the model's position in the popularity ordering.
    assert min(row["rank"] for row in rows) == 1


def test_list_featured_matches_the_picker() -> None:
    from kolega_code.cli.provider_registry import ui_model_options

    rows = model_catalog.catalog_rows(provider=PROVIDER, featured_only=True)

    assert [row["model"] for row in rows] == [model for _label, model in ui_model_options(PROVIDER)]
    assert all(row["featured"] for row in rows)


def test_list_rejects_an_unknown_provider() -> None:
    # A typo must fail loudly instead of printing an empty table.
    _lines, printer = _collect()
    with pytest.raises(ValueError, match="Unsupported provider"):
        model_catalog.run_models_list(_args(provider="nope", featured=False, sort="popularity", json=False), printer)


def test_list_rejects_an_unknown_sort() -> None:
    with pytest.raises(ValueError, match="Unsupported sort"):
        model_catalog.catalog_rows(sort="whatever")


def test_list_json_output_is_machine_readable() -> None:
    lines, printer = _collect()

    exit_code = model_catalog.run_models_list(
        _args(provider=PROVIDER, featured=True, sort="popularity", json=True), printer
    )

    assert exit_code == 0
    payload = json.loads("\n".join(lines))
    assert payload
    assert {"rank", "provider", "model", "context_length", "featured"} <= set(payload[0])
    assert all(row["provider"] == PROVIDER for row in payload)


def test_list_table_output_marks_featured_models() -> None:
    lines, printer = _collect()

    model_catalog.run_models_list(_args(provider=PROVIDER, featured=False, sort="popularity", json=False), printer)

    table = "\n".join(lines)
    assert "RANK" in table and "PROVIDER" in table
    assert "listed in the model picker" in table


# --- overlay --------------------------------------------------------------


def _cache_entries() -> list[tuple[str, dict]]:
    return [
        (
            "vendor/brand-new",
            {
                "context_length": 200000,
                "max_completion_tokens": 8192,
                "default_temperature": 1.0,
                "supports_vision": False,
                "preferred_edit_protocol": "claude_code",
                "thinking_effort": ThinkingEffortSpec(
                    options=("none", "medium"), default="medium", mode="openrouter_reasoning"
                ),
            },
        )
    ]


def test_overlay_adds_new_models_without_featuring_them(tmp_path: Path, restore_catalog) -> None:
    path = tmp_path / openrouter.CACHE_FILENAME
    openrouter.save_cache(path, _cache_entries(), fetched_at="2026-07-31T00:00:00+00:00")

    added = model_catalog.apply_catalog_overlay(tmp_path, env={})

    assert added == 1
    spec = MODEL_SPECS[(PROVIDER, "vendor/brand-new")]
    assert spec["context_length"] == 200000
    assert isinstance(spec["thinking_effort"], ThinkingEffortSpec)
    # The picker keeps showing the release's reviewed most-used set.
    assert "featured" not in spec

    from kolega_code.cli.provider_registry import get_ui_model, ui_model_options

    assert get_ui_model(PROVIDER, "vendor/brand-new") is not None
    assert "vendor/brand-new" not in [model for _label, model in ui_model_options(PROVIDER)]


def test_overlay_never_overrides_a_bundled_model(tmp_path: Path, restore_catalog) -> None:
    bundled = next(model for provider, model in MODEL_SPECS if provider == PROVIDER)
    original = dict(MODEL_SPECS[(PROVIDER, bundled)])
    path = tmp_path / openrouter.CACHE_FILENAME
    openrouter.save_cache(
        path,
        [(bundled, {"context_length": 1, "max_completion_tokens": 1, "supports_vision": False})],
        fetched_at="2026-07-31T00:00:00+00:00",
    )

    added = model_catalog.apply_catalog_overlay(tmp_path, env={})

    assert added == 0
    assert MODEL_SPECS[(PROVIDER, bundled)] == original


def test_overlay_is_skipped_when_disabled(tmp_path: Path, restore_catalog) -> None:
    path = tmp_path / openrouter.CACHE_FILENAME
    openrouter.save_cache(path, _cache_entries(), fetched_at="2026-07-31T00:00:00+00:00")

    added = model_catalog.apply_catalog_overlay(tmp_path, env={model_catalog.CATALOG_DISABLE_ENV: "1"})

    assert added == 0
    assert (PROVIDER, "vendor/brand-new") not in MODEL_SPECS


def test_overlay_path_can_be_pinned_by_env(tmp_path: Path, restore_catalog) -> None:
    pinned = tmp_path / "pinned.json"
    openrouter.save_cache(pinned, _cache_entries(), fetched_at="2026-07-31T00:00:00+00:00")

    env = {model_catalog.CATALOG_PATH_ENV: str(pinned)}
    assert model_catalog.overlay_path(tmp_path / "elsewhere", env) == pinned
    assert model_catalog.apply_catalog_overlay(tmp_path / "elsewhere", env=env) == 1


@pytest.mark.parametrize(
    "content",
    [
        "not json at all",
        json.dumps({"schema_version": 999, "models": []}),
        json.dumps({"schema_version": openrouter.CACHE_SCHEMA_VERSION, "models": "nope"}),
        json.dumps({"schema_version": openrouter.CACHE_SCHEMA_VERSION, "models": [{"id": "x"}]}),
    ],
)
def test_unusable_cache_files_are_ignored_not_fatal(tmp_path: Path, content: str, restore_catalog) -> None:
    path = tmp_path / openrouter.CACHE_FILENAME
    path.write_text(content, encoding="utf-8")

    # The bundled snapshot is a complete catalog on its own; a bad cache must
    # never take the CLI down.
    assert openrouter.load_cache(path) == []
    assert model_catalog.apply_catalog_overlay(tmp_path, env={}) == 0


def test_missing_cache_is_a_no_op(tmp_path: Path, restore_catalog) -> None:
    assert model_catalog.apply_catalog_overlay(tmp_path, env={}) == 0


# --- models refresh -------------------------------------------------------


def test_refresh_writes_a_cache_from_the_documented_endpoint(tmp_path: Path, monkeypatch) -> None:
    payload = {
        "data": [
            {
                "id": "vendor/fresh",
                "canonical_slug": "vendor/fresh",
                "context_length": 128000,
                "architecture": {"input_modalities": ["text"]},
                "pricing": {"prompt": "0.000001"},
                "top_provider": {"max_completion_tokens": 4096},
                "supported_parameters": ["tools", "temperature"],
            }
        ]
    }
    monkeypatch.setattr(openrouter, "fetch_models", lambda **_: payload)
    lines, printer = _collect()
    errors, error_printer = _collect()

    exit_code = model_catalog.run_models_refresh(_args(provider=None, state_dir=tmp_path), printer, error_printer)

    assert exit_code == 0, errors
    cached = openrouter.load_cache(tmp_path / openrouter.CACHE_FILENAME)
    assert [identifier for identifier, _spec in cached] == ["vendor/fresh"]
    assert "1 not in the bundled catalog" in "\n".join(lines)


def test_refresh_reports_a_fetch_failure_without_touching_the_cache(tmp_path: Path, monkeypatch) -> None:
    def _boom(**_):
        raise RuntimeError("network down")

    monkeypatch.setattr(openrouter, "fetch_models", _boom)
    lines, printer = _collect()
    errors, error_printer = _collect()

    exit_code = model_catalog.run_models_refresh(_args(provider=None, state_dir=tmp_path), printer, error_printer)

    assert exit_code == 1
    assert "network down" in "\n".join(errors)
    assert not (tmp_path / openrouter.CACHE_FILENAME).exists()


def test_refresh_rejects_providers_without_a_refreshable_catalog(tmp_path: Path) -> None:
    lines, printer = _collect()
    errors, error_printer = _collect()

    exit_code = model_catalog.run_models_refresh(
        _args(provider="anthropic", state_dir=tmp_path), printer, error_printer
    )

    assert exit_code == 2
    assert "refreshable catalog" in "\n".join(errors)


# --- tinker overlay (same machinery, second provider) ---------------------


def test_tinker_overlay_adds_new_models_without_featuring_them(tmp_path: Path, restore_catalog) -> None:
    from kolega_code.llm.specs import tinker_catalog as tinker

    entries = [
        (
            "Qwen/Qwen3.7-8B",
            {
                "context_length": 65536,
                "max_completion_tokens": 32768,
                "default_temperature": 1.0,
                "supports_vision": False,
            },
        )
    ]
    path = tmp_path / tinker.CACHE_FILENAME
    tinker.save_cache(path, entries, fetched_at="2026-08-07T00:00:00+00:00")

    added = model_catalog.apply_catalog_overlay(tmp_path, env={}, catalog=tinker.PROVIDER)

    assert added == 1
    spec = MODEL_SPECS[(tinker.PROVIDER, "Qwen/Qwen3.7-8B")]
    assert spec["context_length"] == 65536
    assert "featured" not in spec
    # The tinker overlay has its own disable env var, derived from the provider.
    disabled = model_catalog.apply_catalog_overlay(
        tmp_path, env={"KOLEGA_CODE_DISABLE_TINKER_CATALOG": "1"}, catalog=tinker.PROVIDER
    )
    assert disabled == 0


def test_tinker_refresh_writes_its_own_cache(tmp_path: Path, monkeypatch) -> None:
    from kolega_code.llm.specs import tinker_catalog as tinker

    payload = [
        {
            "name": "Qwen3.7-8B",
            "tinker_id": "Qwen/Qwen3.7-8B",
            "context": "64K",
            "type": "Hybrid + Vision",
            "arch": "Dense",
        },
        {
            "name": "Some-Base-Model",
            "tinker_id": "vendor/Something-Base",
            "context": "32K",
            "type": "Base",
            "arch": "Dense",
        },
    ]
    monkeypatch.setattr(tinker, "fetch_models", lambda **_: payload)
    lines, printer = _collect()
    errors, error_printer = _collect()

    exit_code = model_catalog.run_models_refresh(
        _args(provider=tinker.PROVIDER, state_dir=tmp_path), printer, error_printer
    )

    assert exit_code == 0, errors
    cached = tinker.load_cache(tmp_path / tinker.CACHE_FILENAME)
    assert [identifier for identifier, _spec in cached] == ["Qwen/Qwen3.7-8B"]
    assert "1 not in the bundled catalog" in "\n".join(lines)


# --- Ollama Cloud overlay -------------------------------------------------


def _ollama_payload(*identifiers: str) -> dict:
    return {
        "models": {"data": [{"id": identifier} for identifier in identifiers]},
        "details": {
            identifier: {
                "capabilities": ["completion", "tools", "thinking"],
                "model_info": {f"{identifier}.context_length": 131072},
            }
            for identifier in identifiers
        },
    }


def test_ollama_cloud_overlay_adds_new_ids_without_overwriting_bundled_specs(tmp_path: Path, restore_catalog) -> None:
    bundled = next(model for provider, model in MODEL_SPECS if provider == ollama_cloud.PROVIDER)
    original = dict(MODEL_SPECS[(ollama_cloud.PROVIDER, bundled)])
    entries = [
        (
            bundled,
            {
                "context_length": 1,
                "max_completion_tokens": 1,
                "default_temperature": 1.0,
                "supports_vision": False,
            },
        ),
        (
            "future-agent-model",
            {
                "context_length": 131072,
                "max_completion_tokens": 32768,
                "default_temperature": 1.0,
                "supports_vision": True,
            },
        ),
    ]
    path = tmp_path / ollama_cloud.CACHE_FILENAME
    ollama_cloud.save_cache(path, entries, fetched_at="2026-08-10T00:00:00+00:00")

    added = model_catalog.apply_catalog_overlay(tmp_path, env={}, catalog=ollama_cloud.PROVIDER)

    assert added == 1
    assert MODEL_SPECS[(ollama_cloud.PROVIDER, bundled)] == original
    assert MODEL_SPECS[(ollama_cloud.PROVIDER, "future-agent-model")]["supports_vision"] is True


def test_ollama_cloud_overlay_path_and_disable_env_are_provider_specific(tmp_path: Path, restore_catalog) -> None:
    pinned = tmp_path / "pinned-ollama.json"
    entries = [
        (
            "future-disabled-model",
            {
                "context_length": 131072,
                "max_completion_tokens": 32768,
                "default_temperature": 1.0,
                "supports_vision": False,
            },
        )
    ]
    ollama_cloud.save_cache(pinned, entries, fetched_at="2026-08-10T00:00:00+00:00")
    env = {
        "KOLEGA_CODE_OLLAMA_CLOUD_CATALOG": str(pinned),
        "KOLEGA_CODE_DISABLE_OLLAMA_CLOUD_CATALOG": "1",
    }

    assert model_catalog.overlay_path(tmp_path / "elsewhere", env, catalog=ollama_cloud.PROVIDER) == pinned
    assert model_catalog.apply_catalog_overlay(tmp_path / "elsewhere", env=env, catalog=ollama_cloud.PROVIDER) == 0
    assert (ollama_cloud.PROVIDER, "future-disabled-model") not in MODEL_SPECS

    env.pop("KOLEGA_CODE_DISABLE_OLLAMA_CLOUD_CATALOG")
    assert model_catalog.apply_catalog_overlay(tmp_path / "elsewhere", env=env, catalog=ollama_cloud.PROVIDER) == 1


def test_ollama_cloud_refresh_writes_its_own_cache_and_reports_new_ids(tmp_path: Path, monkeypatch) -> None:
    bundled = next(model for provider, model in MODEL_SPECS if provider == ollama_cloud.PROVIDER)
    payload = _ollama_payload(bundled, "future-live-model")
    monkeypatch.setattr(ollama_cloud, "fetch_models", lambda **_: payload)
    lines, printer = _collect()
    errors, error_printer = _collect()

    exit_code = model_catalog.run_models_refresh(
        _args(provider=ollama_cloud.PROVIDER, state_dir=tmp_path),
        printer,
        error_printer,
    )

    assert exit_code == 0, errors
    cached = ollama_cloud.load_cache(tmp_path / ollama_cloud.CACHE_FILENAME)
    assert [entry[0] for entry in cached] == sorted([bundled, "future-live-model"])
    output = "\n".join(lines)
    assert "1 not in the bundled catalog" in output
    assert "Newly available: future-live-model" in output


def test_ollama_cloud_refresh_preserves_prior_overlay_ids_and_bundled_accounting(
    tmp_path: Path,
    monkeypatch,
    restore_catalog,
) -> None:
    bundled = next(model for provider, model in MODEL_SPECS if provider == ollama_cloud.PROVIDER)
    path = tmp_path / ollama_cloud.CACHE_FILENAME
    previous_entries = [
        (
            "previous-overlay-model",
            {
                "context_length": 65536,
                "max_completion_tokens": 16384,
                "default_temperature": 1.0,
                "supports_vision": False,
            },
        )
    ]
    ollama_cloud.save_cache(path, previous_entries, fetched_at="2026-08-09T00:00:00+00:00")
    assert model_catalog.apply_catalog_overlay(tmp_path, env={}, catalog=ollama_cloud.PROVIDER) == 1

    monkeypatch.setattr(
        ollama_cloud,
        "fetch_models",
        lambda **_: _ollama_payload(bundled, "new-overlay-model"),
    )
    lines, printer = _collect()
    errors, error_printer = _collect()

    exit_code = model_catalog.run_models_refresh(
        _args(provider=ollama_cloud.PROVIDER, state_dir=tmp_path),
        printer,
        error_printer,
    )

    assert exit_code == 0, errors
    cached = ollama_cloud.load_cache(path)
    assert [identifier for identifier, _spec in cached] == sorted(
        [bundled, "new-overlay-model", "previous-overlay-model"]
    )
    output = "\n".join(lines)
    assert "2 not in the bundled catalog (1 new since the last refresh)" in output
    assert "Newly available: new-overlay-model" in output


def test_ollama_cloud_refresh_failure_leaves_existing_cache_untouched(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / ollama_cloud.CACHE_FILENAME
    ollama_cloud.save_cache(
        path,
        [
            (
                "previously-cached",
                {
                    "context_length": 131072,
                    "max_completion_tokens": 32768,
                    "default_temperature": 1.0,
                    "supports_vision": False,
                },
            )
        ],
        fetched_at="2026-08-10T00:00:00+00:00",
    )
    original = path.read_bytes()
    monkeypatch.setattr(ollama_cloud, "fetch_models", lambda **_: {"models": {"data": [{"id": "missing"}]}})
    _lines, printer = _collect()
    errors, error_printer = _collect()

    exit_code = model_catalog.run_models_refresh(
        _args(provider=ollama_cloud.PROVIDER, state_dir=tmp_path),
        printer,
        error_printer,
    )

    assert exit_code == 1
    assert "payload.details" in "\n".join(errors)
    assert path.read_bytes() == original


# --- /model resolution ----------------------------------------------------


def _match_model(provider: str, options: list[tuple[str, str]], value: str):
    """Drive CommandHandlersMixin._match_model_value without booting the TUI."""
    from types import SimpleNamespace

    from kolega_code.cli.tui.command_handlers import CommandHandlersMixin

    app = SimpleNamespace(settings=SimpleNamespace(active_provider=provider))
    # The method only reads settings.active_provider, so a stub avoids booting
    # the whole Textual app just to resolve a model name.
    return CommandHandlersMixin._match_model_value(app, options, value)  # pyright: ignore[reportArgumentType]


def test_model_command_accepts_an_offered_option() -> None:
    options = [("Kimi K3", "moonshotai/kimi-k3")]
    assert _match_model(PROVIDER, options, "moonshotai/kimi-k3") == "moonshotai/kimi-k3"
    assert _match_model(PROVIDER, options, "  MOONSHOTAI/KIMI-K3  ") == "moonshotai/kimi-k3"


def test_model_command_accepts_a_catalogued_model_outside_the_offered_list() -> None:
    # `/model` lists a gateway's most-used models, but typing any catalogued id works.
    unlisted = next(
        model
        for provider, model in MODEL_SPECS
        if provider == PROVIDER and not model_catalog.is_featured_model(PROVIDER, model)
    )
    assert _match_model(PROVIDER, [("Kimi K3", "moonshotai/kimi-k3")], unlisted) == unlisted


def test_model_command_still_rejects_unknown_models() -> None:
    options = [("Kimi K3", "moonshotai/kimi-k3")]
    assert _match_model(PROVIDER, options, "vendor/not-a-real-model") is None
    assert _match_model(PROVIDER, options, "") is None
    # A model that exists on a different provider is not silently accepted.
    assert _match_model("anthropic", options, "z-ai/glm-5.2") is None
