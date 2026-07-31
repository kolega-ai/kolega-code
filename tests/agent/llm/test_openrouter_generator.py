"""Offline tests for the OpenRouter catalog transform.

The fixtures under ``tests/fixtures/openrouter/`` are trimmed captures of the two
upstream endpoints taken on 2026-07-31. No test here performs network I/O.
"""

import json
from pathlib import Path

import pytest

from kolega_code.llm.specs import openrouter_catalog as oc
from kolega_code.llm.specs.types import ThinkingEffortSpec

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "openrouter"

# OpenRouter's public "This Week / All models" leaderboard as it stood when the
# fixtures were captured. Reproducing it is the contract of the ranking code: if
# the aggregation ever changes shape, catalog order and the featured set silently
# stop matching what users see on openrouter.ai.
EXPECTED_LEADERBOARD = [
    ("xiaomi/mimo-v2.5", 8.90),
    ("deepseek/deepseek-v4-flash", 7.57),
    ("tencent/hy3", 4.79),
    ("deepseek/deepseek-v4-pro", 3.59),
    ("z-ai/glm-5.2", 3.24),
    ("nvidia/nemotron-3-ultra-550b-a55b:free", 2.69),
    ("minimax/minimax-m3", 2.02),
    ("stepfun/step-3.7-flash", 1.74),
    ("moonshotai/kimi-k3", 1.34),
    ("inclusionai/ling-3.0-flash:free", 1.30),
    ("anthropic/claude-sonnet-5", 1.02),
    ("google/gemini-3-flash-preview", 0.981),
    ("anthropic/claude-sonnet-4.6", 0.936),
    ("anthropic/claude-opus-5", 0.783),
    ("google/gemini-2.5-flash-lite", 0.703),
    ("anthropic/claude-opus-4.8", 0.676),
    ("xiaomi/mimo-v2.5-pro", 0.644),
    ("google/gemini-2.5-flash", 0.596),
    ("google/gemini-3.1-flash-lite", 0.572),
    ("openai/gpt-oss-120b", 0.505),
]


@pytest.fixture(scope="module")
def models_payload() -> dict:
    return json.loads((FIXTURES / "models.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rankings_payload() -> dict:
    return json.loads((FIXTURES / "rankings_week.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scores(rankings_payload: dict) -> dict:
    return oc.ranking_scores(rankings_payload)


def _spec_for(models_payload: dict, model_id: str) -> dict:
    entry = next(item for item in models_payload["data"] if item["id"] == model_id)
    return oc.model_spec(entry)


def test_ranking_reproduces_the_public_leaderboard(models_payload: dict, scores: dict) -> None:
    entries = oc.catalog_entries(models_payload, scores)
    by_id = {oc.model_id(item): item for item in models_payload["data"]}

    ranked = [identifier for identifier, _spec in entries][: len(EXPECTED_LEADERBOARD)]
    assert ranked == [identifier for identifier, _tokens in EXPECTED_LEADERBOARD]

    for identifier, expected_trillions in EXPECTED_LEADERBOARD:
        actual = scores[oc.ranking_key(by_id[identifier])] / 1e12
        assert actual == pytest.approx(expected_trillions, abs=0.005), identifier


def test_ranking_joins_on_canonical_slug_and_variant(models_payload: dict, scores: dict) -> None:
    by_id = {oc.model_id(item): item for item in models_payload["data"]}

    # Dated canonical slug, standard variant.
    assert oc.ranking_key(by_id["z-ai/glm-5.2"])[1] == "standard"
    assert oc.ranking_key(by_id["z-ai/glm-5.2"])[0].startswith("z-ai/glm-5.2")

    # A ":free" id keeps the variant out of the permaslug.
    permaslug, variant = oc.ranking_key(by_id["inclusionai/ling-3.0-flash:free"])
    assert variant == "free"
    assert ":" not in permaslug
    assert scores[(permaslug, variant)] > 0


def test_unranked_models_sort_last_and_ties_break_alphabetically(models_payload: dict, scores: dict) -> None:
    entries = oc.catalog_entries(models_payload, scores)
    identifiers = [identifier for identifier, _spec in entries]
    by_id = {oc.model_id(item): item for item in models_payload["data"]}

    unranked = [identifier for identifier in identifiers if scores.get(oc.ranking_key(by_id[identifier]), 0) == 0]
    assert unranked, "fixture should contain a model with no traffic"
    # They occupy the tail, alphabetically ordered among themselves.
    assert identifiers[-len(unranked) :] == unranked
    assert unranked == sorted(unranked)


def test_ordering_is_deterministic(models_payload: dict, scores: dict) -> None:
    first = oc.catalog_entries(models_payload, scores)
    second = oc.catalog_entries(models_payload, scores)
    assert [identifier for identifier, _ in first] == [identifier for identifier, _ in second]


def test_without_rankings_the_catalog_degrades_to_alphabetical(models_payload: dict, scores: dict) -> None:
    ranked = oc.catalog_entries(models_payload, scores)
    unranked = oc.catalog_entries(models_payload, None)

    assert [identifier for identifier, _ in unranked] == sorted(identifier for identifier, _ in unranked)
    # Losing an undocumented endpoint must only change order, never coverage.
    assert {identifier for identifier, _ in unranked} == {identifier for identifier, _ in ranked}


def test_malformed_rankings_payloads_are_rejected_not_guessed() -> None:
    for payload in ({"data": []}, {"nope": 1}, [], {"data": [{"variant": "standard"}]}):
        with pytest.raises(oc.OpenRouterCatalogError):
            oc.ranking_scores(payload)


def test_featured_is_the_top_n_plus_pinned_ids(models_payload: dict, scores: dict) -> None:
    entries = oc.catalog_entries(models_payload, scores)
    identifiers = [identifier for identifier, _ in entries]

    featured = oc.featured_ids(entries, always_include=[identifiers[0]])
    assert featured == identifiers[: oc.FEATURED_COUNT]

    # A pinned model outside the cut is appended, never reordered in.
    outsider = identifiers[oc.FEATURED_COUNT + 2]
    featured = oc.featured_ids(entries, always_include=[outsider])
    assert featured == identifiers[: oc.FEATURED_COUNT] + [outsider]
    assert len(featured) == oc.FEATURED_COUNT + 1

    # Unknown pins are ignored rather than inventing a catalog entry.
    assert oc.featured_ids(entries, always_include=["nope/not-a-model"]) == identifiers[: oc.FEATURED_COUNT]


def test_selection_filter_excludes_unusable_models(models_payload: dict, scores: dict) -> None:
    identifiers = {identifier for identifier, _ in oc.catalog_entries(models_payload, scores)}

    # Present in the fixture, deliberately absent from the catalog.
    assert "openrouter/auto" not in identifiers  # meta-router, pricing "-1"
    assert "~openai/gpt-latest" not in identifiers  # floating alias
    assert "google/gemini-3.6-flash:batch" not in identifiers  # async batch endpoint
    assert all(not identifier.startswith("~") for identifier in identifiers)
    assert all(not identifier.endswith(":batch") for identifier in identifiers)


def test_non_tool_capable_models_are_excluded_and_do_not_take_a_featured_slot() -> None:
    payload = {
        "data": [
            {
                "id": "vendor/chatty",
                "canonical_slug": "vendor/chatty",
                "context_length": 100000,
                "architecture": {"input_modalities": ["text"]},
                "pricing": {"prompt": "0.000001"},
                "top_provider": {"max_completion_tokens": 4096},
                "supported_parameters": ["temperature"],  # no "tools"
            },
            {
                "id": "vendor/agentic",
                "canonical_slug": "vendor/agentic",
                "context_length": 100000,
                "architecture": {"input_modalities": ["text"]},
                "pricing": {"prompt": "0.000001"},
                "top_provider": {"max_completion_tokens": 4096},
                "supported_parameters": ["tools", "temperature"],
            },
        ]
    }
    # The tool-less model outranks the other but cannot back an agent, so it is
    # absent and the next usable model takes its place.
    scores = {("vendor/chatty", "standard"): 10**12, ("vendor/agentic", "standard"): 1}
    entries = oc.catalog_entries(payload, scores)
    assert [identifier for identifier, _ in entries] == ["vendor/agentic"]
    assert oc.featured_ids(entries) == ["vendor/agentic"]


def test_temperature_and_vision_flags_follow_the_payload(models_payload: dict) -> None:
    # gpt-5.x reject an explicit temperature; the flag is what stops a 400.
    assert _spec_for(models_payload, "openai/gpt-5.6-sol")["supports_temperature"] is False
    # A model that accepts temperature omits the key entirely.
    assert "supports_temperature" not in _spec_for(models_payload, "z-ai/glm-5.2")

    assert _spec_for(models_payload, "anthropic/claude-opus-5")["supports_vision"] is True
    assert _spec_for(models_payload, "z-ai/glm-5.2")["supports_vision"] is False


def test_effort_options_are_canonically_ordered_with_none_only_when_optional(models_payload: dict) -> None:
    spec = _spec_for(models_payload, "anthropic/claude-opus-5")["thinking_effort"]
    assert isinstance(spec, ThinkingEffortSpec)
    assert spec.options == ("none", "low", "medium", "high", "xhigh", "max")
    assert spec.default == "medium"
    assert spec.mode == "openrouter_reasoning"

    # reasoning.mandatory models must never be offered a disable option.
    mandatory = next(
        item
        for item in models_payload["data"]
        if isinstance(item.get("reasoning"), dict)
        and item["reasoning"].get("mandatory")
        and item["reasoning"].get("supported_efforts")
    )
    assert "none" not in oc.model_spec(mandatory)["thinking_effort"].options


def test_effort_default_prefers_medium_over_an_expensive_upstream_default(models_payload: dict) -> None:
    # kimi-k3 defaults to "max" upstream; Kolega Code's convention is "medium"
    # when the model offers it, and the upstream default otherwise.
    kimi = _spec_for(models_payload, "moonshotai/kimi-k3")["thinking_effort"]
    assert "medium" not in kimi.options
    assert kimi.default == "max"

    opus = _spec_for(models_payload, "anthropic/claude-opus-5")["thinking_effort"]
    assert opus.default == "medium"


def test_models_without_supported_efforts_get_no_effort_control(models_payload: dict) -> None:
    assert "thinking_effort" not in _spec_for(models_payload, "minimax/minimax-m3")


def test_missing_output_cap_falls_back_to_a_bounded_budget(models_payload: dict) -> None:
    entry = next(item for item in models_payload["data"] if item["id"] == "moonshotai/kimi-k3")
    assert entry["top_provider"]["max_completion_tokens"] is None
    spec = oc.model_spec(entry)
    assert spec["max_completion_tokens"] == min(entry["context_length"], oc.FALLBACK_MAX_COMPLETION_TOKENS)


def test_openai_models_get_apply_patch_and_everything_else_claude_code(models_payload: dict) -> None:
    # OpenAI models are trained on the Codex apply_patch freeform format, matching
    # what the direct `openai` / `openai_chatgpt` catalogs already select.
    assert _spec_for(models_payload, "openai/gpt-5.6-sol")["preferred_edit_protocol"] == "codex_apply_patch"
    assert _spec_for(models_payload, "openai/gpt-5.4-mini")["preferred_edit_protocol"] == "codex_apply_patch"
    assert _spec_for(models_payload, "openai/gpt-oss-120b")["preferred_edit_protocol"] == "codex_apply_patch"

    # Everything else on the gateway defaults to the Claude Code-style edit tool.
    for model in ("deepseek/deepseek-v4-pro", "z-ai/glm-5.2", "anthropic/claude-opus-5", "xiaomi/mimo-v2.5"):
        assert _spec_for(models_payload, model)["preferred_edit_protocol"] == "claude_code", model


def test_every_catalogued_model_declares_an_edit_protocol(models_payload: dict, scores: dict) -> None:
    for identifier, spec in oc.catalog_entries(models_payload, scores):
        assert spec["preferred_edit_protocol"] in ("claude_code", "codex_apply_patch"), identifier


def test_anthropic_models_are_marked_unreplayable(models_payload: dict) -> None:
    assert _spec_for(models_payload, "anthropic/claude-opus-5")["drop_prior_reasoning"] is True
    assert "drop_prior_reasoning" not in _spec_for(models_payload, "z-ai/glm-5.2")


def test_specs_round_trip_through_the_cache_encoding(models_payload: dict) -> None:
    original = _spec_for(models_payload, "anthropic/claude-opus-5")
    encoded = oc.spec_to_jsonable(original)
    # Must be plain JSON for the on-disk overlay cache.
    assert json.loads(json.dumps(encoded)) == encoded
    assert oc.spec_from_jsonable(encoded) == original


def test_rendered_module_is_importable_and_stable(models_payload: dict, scores: dict, tmp_path: Path) -> None:
    entries = oc.catalog_entries(models_payload, scores)
    featured = set(oc.featured_ids(entries))
    rendered = [
        (identifier, {**spec, "featured": True} if identifier in featured else spec) for identifier, spec in entries
    ]
    source = oc.render_catalog_module(rendered, header_lines=["Generated by a test.", "", "Second line."])
    assert source == oc.render_catalog_module(rendered, header_lines=["Generated by a test.", "", "Second line."])
    assert source.startswith("# Generated by a test.")

    module_path = tmp_path / "generated_openrouter.py"
    module_path.write_text(source, encoding="utf-8")
    namespace: dict = {}
    exec(compile(source, str(module_path), "exec"), namespace)  # noqa: S102 - generated source under test
    specs = namespace["OPENROUTER_SPECS"]
    assert list(specs) == [("openrouter", identifier) for identifier, _ in rendered]
    assert specs[("openrouter", "anthropic/claude-opus-5")]["thinking_effort"].default == "medium"
    assert sum(1 for spec in specs.values() if spec.get("featured")) == len(featured)
