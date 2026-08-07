from typing import Any, Dict, Optional

from .catalog import MODEL_SPECS, WILDCARD_MODEL_SPECS
from .types import ThinkingEffortSpec


def _provider_value(provider: Any) -> str:
    return provider.value if hasattr(provider, "value") else provider


def _resolve_specs(provider: Any, model_name: str) -> Optional[Dict[str, Any]]:
    """Resolve exact catalog entries first, then provider wildcard templates.

    A wildcard template (e.g. ``("tinker", "tinker://*")``) supplies the spec
    for a model addressed by an uncatalogued, per-user identifier — currently
    Tinker sampler checkpoint paths. Returning the template for any prefix
    match keeps every accessor (vision, effort, context, budgets) working for
    those models instead of raising.
    """
    provider_str = _provider_value(provider)
    key = (provider_str, model_name)
    if key in MODEL_SPECS:
        return MODEL_SPECS[key]
    for (wildcard_provider, prefix), spec in WILDCARD_MODEL_SPECS.items():
        # A trailing ``*`` is a glob marker, not part of the prefix.
        match_prefix = prefix[:-1] if prefix.endswith("*") else prefix
        if wildcard_provider == provider_str and model_name.startswith(match_prefix):
            return spec
    return None


def get_model_specs(provider: str, model_name: str) -> Dict[str, Any]:
    """
    Get the specifications for a given model.

    Args:
        provider: The LLM provider (e.g., 'anthropic', 'openai') - can be string or enum
        model_name: The name of the model

    Returns:
        Dictionary containing context_length, max_completion_tokens, and default_temperature
    """
    # Handle both string and enum provider types
    specs = _resolve_specs(provider, model_name)
    if specs is None:
        provider_str = _provider_value(provider)
        raise ValueError(f"Model {model_name} from provider {provider_str} is not supported.")
    return specs


def model_is_known(provider: Any, model_name: str) -> bool:
    """Whether ``model_name`` resolves to a catalog entry or wildcard template.

    Mirrors the strict ``get_model_specs`` acceptance boundary without raising,
    so config/UI layers can accept wildcard-addressed models (Tinker sampler
    checkpoint paths) alongside catalogued ones.
    """
    return _resolve_specs(provider, model_name) is not None


def provider_has_wildcard_models(provider: Any) -> bool:
    """Whether ``provider`` declares a wildcard template, i.e. accepts
    uncatalogued model identifiers (used to enable typed-id entry in pickers)."""
    provider_str = _provider_value(provider)
    return any(wildcard_provider == provider_str for wildcard_provider, _ in WILDCARD_MODEL_SPECS)


def supports_vision(provider: str, model_name: str) -> bool:
    """Whether a model accepts image input.

    Defaults to ``False`` for any entry that omits the flag, so a missing key
    is safely treated as non-vision (a clear guard message beats a mid-flight
    provider error).
    """
    return get_model_specs(provider, model_name).get("supports_vision", False)


def supports_hosted_web_search(provider: str, model_name: str) -> bool:
    """Whether the model's API surface executes a hosted ``web_search`` tool.

    True only for Responses-API models whose backend accepts
    ``tools: [{"type": "web_search"}]`` and runs it server-side (verified per
    provider by findings/probes/hosted_web_search_probe.py). Defaults to
    ``False`` — including for unknown models — so the hosted tool is never
    requested from a backend that would reject or ignore it.
    """
    specs = MODEL_SPECS.get((_provider_value(provider), model_name))
    return bool(specs and specs.get("supports_hosted_web_search", False))


def get_thinking_effort_spec(provider: str, model_name: str) -> Optional[ThinkingEffortSpec]:
    """Return the thinking effort spec for a model, if it supports a public control."""
    return get_model_specs(provider, model_name).get("thinking_effort")


def thinking_effort_options(provider: str, model_name: str) -> tuple[str, ...]:
    """Return supported effort values for a model."""
    spec = get_thinking_effort_spec(provider, model_name)
    return spec.options if spec else ()


def default_thinking_effort(provider: str, model_name: str) -> Optional[str]:
    """Return Kolega's default thinking effort for a model."""
    spec = get_thinking_effort_spec(provider, model_name)
    return spec.default if spec else None


def is_featured_model(provider: str, model_name: str) -> bool:
    """Whether a model is one of a gateway provider's listed (featured) models.

    Only gateway catalogs large enough to overwhelm a picker mark entries
    featured; for every other provider this is uniformly ``False`` and callers
    fall back to listing the whole provider.
    """
    specs = MODEL_SPECS.get((_provider_value(provider), model_name))
    return bool(specs and specs.get("featured", False))


def provider_has_featured_models(provider: str) -> bool:
    """Whether ``provider`` marks any model featured, i.e. wants a short list."""
    provider_str = _provider_value(provider)
    return any(key[0] == provider_str and specs.get("featured", False) for key, specs in MODEL_SPECS.items())


def prior_reasoning_is_replayable(provider: str, model_name: str) -> bool:
    """Whether prior reasoning may be sent back to this model.

    Defaults to ``True`` — including for unknown models — so only a catalog
    entry that explicitly opts out changes behavior. Anthropic models reached
    through a gateway opt out: their reasoning is carried by signed thinking
    blocks that no plain-text replay field can reconstruct.
    """
    specs = MODEL_SPECS.get((_provider_value(provider), model_name))
    if specs is None:
        return True
    return not specs.get("drop_prior_reasoning", False)


def preferred_edit_protocol(provider: str, model_name: str) -> Optional[str]:
    """Return the catalogue-preferred edit protocol, when one is configured.

    Unknown models and entries without a preference deliberately return ``None``
    so callers can retain the stable search/replace fallback.
    """

    provider_str = _provider_value(provider)
    specs = MODEL_SPECS.get((provider_str, model_name))
    if specs is None:
        return None
    value = specs.get("preferred_edit_protocol")
    return str(value) if value is not None else None


# DeepSeek's published max_completion_tokens (384000) is fiction. Probed
# 2026-08-03, tests/agent/llm/test_deepseek_output_cap_live.py:
#   - first-party chat, uncapped: hard server cut at exactly 65536 output
#     tokens (reported honestly as finish_reason="length");
#   - first-party Responses, uncapped: the model SELF-CENSORS ~7.4k tokens in
#     ("response would exceed the maximum output length...") with a clean
#     status="completed" — silent underdelivery; with an explicit cap it
#     streams to the cap and reports the hit honestly
#     (status="incomplete"/reason="max_output_tokens");
#   - fireworks, uncapped: cut at its 32768 DEFAULT max_tokens (honest
#     "length"); an explicit 64000 is honored and honestly reported;
#   - openrouter deepseek slugs: accept max_tokens=64000 and report honest
#     "length" truncations through the gateway;
#   - over-ceiling caps (384000) are accepted silently everywhere — nothing
#     rejects or visibly clamps, so the client must send the cap it wants.
# An explicit client cap below the ceiling is therefore both the only reliable
# honest-truncation signal AND what unlocks the model's full output (uncapped,
# flash delivers ~7k; capped, 64000). 64000 leaves margin under the 65536 ceiling
# measured on the chat path, and applies only there: flash was separately measured
# running to 112990 output tokens in a single Responses call (2026-08-03), so it
# takes its catalog value instead.
DEEPSEEK_WIRE_OUTPUT_CAP = 64000

# Routes whose catalog max_completion_tokens is itself the real ceiling.
_UNCLAMPED_DEEPSEEK_ROUTES = {("deepseek", "deepseek-v4-flash")}


def is_deepseek_model(model_name: str) -> bool:
    """True for DeepSeek-family models under any provider's naming scheme.

    Covers "deepseek-v4-pro" (first-party), "accounts/fireworks/models/deepseek-v4-flash"
    (fireworks), "deepseek/deepseek-v4-pro" (openrouter), "deepseek-v3.2" (ollama_cloud).
    """
    return "deepseek" in model_name.lower()


def deepseek_output_token_cap(provider: str, model_name: str, requested: Optional[int]) -> int:
    """The output-token cap to put on the wire for a DeepSeek-model request.

    The min over the non-None of (requested cap, catalog max_completion_tokens,
    ``DEEPSEEK_WIRE_OUTPUT_CAP``). Always returns a value — DeepSeek requests
    always carry an explicit cap, because the server truncates silently without
    one (see the block comment above). The min keeps smaller models honest: a
    catalog entry with a cap below 64000 stays at its own cap. Routes in
    ``_UNCLAMPED_DEEPSEEK_ROUTES`` skip the clamp and pass their catalog value
    through.
    """
    provider_value = _provider_value(provider)
    specs = MODEL_SPECS.get((provider_value, model_name))
    ceiling = specs.get("max_completion_tokens") if specs else None
    clamp = None if (provider_value, model_name) in _UNCLAMPED_DEEPSEEK_ROUTES else DEEPSEEK_WIRE_OUTPUT_CAP
    return min(value for value in (requested, ceiling, clamp) if value is not None)
