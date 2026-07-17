from typing import Any, Dict, Optional

from .accessors import (
    _provider_value,
    default_thinking_effort,
    get_thinking_effort_spec,
    thinking_effort_options,
)


def validate_thinking_effort(provider: str, model_name: str, effort: Optional[Any]) -> Optional[str]:
    """Validate and normalize a model-specific thinking effort value."""
    if effort is None:
        return None

    effort_value = effort.value if hasattr(effort, "value") else effort
    normalized = str(effort_value).strip().lower()
    if not normalized:
        return None

    options = thinking_effort_options(provider, model_name)
    provider_str = _provider_value(provider)
    if not options:
        raise ValueError(f"Model {model_name} from provider {provider_str} does not support thinking effort.")
    if normalized not in options:
        valid = ", ".join(options)
        raise ValueError(
            f"Unsupported thinking effort '{effort}' for {provider_str}/{model_name}. Valid values: {valid}"
        )
    return normalized


def normalize_thinking_effort(provider: str, model_name: str, effort: Optional[Any]) -> Optional[str]:
    """Validate an explicit effort or return the model default when no effort is set."""
    if effort is None or not str(effort).strip():
        return default_thinking_effort(provider, model_name)
    return validate_thinking_effort(provider, model_name, effort)


def build_thinking_request_params(provider: str, model_name: str, effort: Optional[Any]) -> Dict[str, Any]:
    """Convert a model-specific effort value into provider request parameters."""
    normalized = validate_thinking_effort(provider, model_name, effort)
    if normalized is None:
        return {}

    spec = get_thinking_effort_spec(provider, model_name)
    if spec is None:
        return {}

    if spec.mode == "anthropic_adaptive_effort":
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": normalized},
        }

    if spec.mode == "deepseek_effort":
        # DeepSeek's OpenAI-compatible /v1 endpoint: reasoning is on by default and graded
        # via the standard reasoning_effort param (high/max; it also accepts low/medium/xhigh,
        # collapsing low/medium->high and xhigh->max). There is no reasoning_effort=none, so
        # "none" disables thinking via the documented extra_body toggle instead.
        if normalized == "none":
            return {"extra_body": {"thinking": {"type": "disabled"}}}
        return {"reasoning_effort": normalized}

    if spec.mode == "zai_effort":
        # Z.AI GLM toggles thinking via {"thinking": {"type": "enabled"|"disabled"}}.
        # GLM-5.2 adds two named effort levels (High/Max) carried in output_config.effort.
        if normalized == "none":
            return {"thinking": {"type": "disabled"}}
        params: Dict[str, Any] = {"thinking": {"type": "enabled"}}
        if normalized in ("high", "max"):
            params["output_config"] = {"effort": normalized}
        return params

    if spec.mode == "moonshot_toggle":
        if normalized == "none":
            return {"thinking": {"type": "disabled"}}
        return {"thinking": {"type": "enabled"}}

    if spec.mode == "moonshot_reasoning_effort":
        # Moonshot's standard API expects this as a top-level request field.
        # Kolega currently reaches Moonshot through its Anthropic-compatible
        # endpoint, so use the Anthropic SDK's passthrough body for the field.
        return {"extra_body": {"reasoning_effort": normalized}}

    if spec.mode == "kimi_coding_effort":
        # Kimi Coding Plan exposes K3 through its Anthropic-compatible endpoint
        # and accepts Claude's output_config effort shape. Deep thinking is
        # always enabled for K3, so no separate thinking toggle is needed.
        return {"output_config": {"effort": normalized}}

    if spec.mode == "google_thinking_budget":
        return {"thinking_config": {"thinking_budget": spec.budgets[normalized]}}

    if spec.mode == "google_thinking_level":
        return {"thinking_config": {"thinking_level": normalized}}

    if spec.mode == "openai_reasoning_effort":
        # OpenAI-compatible reasoning APIs use a flat reasoning_effort field.
        # Fireworks additionally accepts "none" to disable reasoning.
        return {"reasoning_effort": normalized}

    if spec.mode == "openai_responses_reasoning":
        # The Responses API nests reasoning effort under a "reasoning" object,
        # unlike Chat Completions' flat "reasoning_effort". We request
        # summary="auto" so the backend streams a human-readable reasoning summary
        # for the TUI thinking display. (Codex defaults these models to summary
        # "none" and shows no reasoning text; kolega surfaces it, so we keep it
        # on.) This is independent of reasoning continuity, which is carried by
        # reasoning.encrypted_content (see the ChatGPT provider), not the summary.
        return {"reasoning": {"effort": normalized, "summary": "auto"}}

    raise ValueError(f"Unknown thinking effort mode '{spec.mode}' for {_provider_value(provider)}/{model_name}.")


# Provider-native assistant-message field for replaying prior reasoning on the
# Chat Completions path. Streaming capture reads delta.reasoning_content /
# delta.reasoning (see OpenAIStreamWrapper); replay must echo back the same field
# the provider emits so the model continues its chain-of-thought across tool-call
# rounds instead of re-deriving it. Providers absent here fall back to the
# visible-text ThinkingBlock.to_openai() placeholder. The field is per provider,
# not per mode: Fireworks and Ollama Cloud share mode "openai_reasoning_effort"
# but use different field names.
_REASONING_REPLAY_FIELDS: Dict[str, str] = {
    "deepseek": "reasoning_content",
    "fireworks": "reasoning_content",
    "ollama_cloud": "reasoning",
    # xAI's Chat Completions endpoint returns and accepts reasoning_content even
    # though its public docs only describe the Responses API (verified live
    # against grok-4.3: streaming emits reasoning_content deltas and replaying it
    # on a multi-turn tool-call request is accepted).
    "xai": "reasoning_content",
}

# Chat-Completions reasoning modes whose reasoning text is replayable via a flat
# top-level field. Responses/Anthropic/Google modes carry reasoning differently
# and are excluded.
_REASONING_REPLAY_MODES = frozenset({"openai_reasoning_effort", "deepseek_effort"})


def reasoning_replay_field(provider: str, model_name: str) -> Optional[str]:
    """Return the assistant-message field for replaying prior reasoning, if any.

    Returns the provider-native field name (e.g. ``"reasoning_content"``) only
    when ``model_name`` is a Chat-Completions reasoning model for ``provider`` and
    that provider has a configured replay field. Returns ``None`` for
    non-reasoning models, unknown provider/model pairs, and providers whose
    reasoning is not replayable via a flat field.
    """
    field = _REASONING_REPLAY_FIELDS.get(_provider_value(provider))
    if field is None:
        return None
    try:
        spec = get_thinking_effort_spec(provider, model_name)
    except ValueError:
        # get_model_specs raises for unknown provider/model; treat as non-reasoning.
        return None
    if spec is None or spec.mode not in _REASONING_REPLAY_MODES:
        return None
    return field
