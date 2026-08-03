"""Provider-independent usage normalization for completed LLM messages.

Every completed provider ``Message`` carries two usage views:

- ``Message.usage_metadata`` — the raw provider-shaped dict (Anthropic
  ``input_tokens``/…, OpenAI ``prompt_tokens``/…, Google ``prompt_token_count``/…),
  preserved verbatim for callers that know the provider vocabulary.
- ``Message.usage`` — the :class:`NormalizedUsage` produced here: one schema
  across all providers with fixed semantics:

  * ``input_tokens`` is inclusive of cached input.
  * ``output_tokens`` includes reasoning the provider bills as output.
  * ``total_tokens == input_tokens + output_tokens`` whenever reported.
  * Cache/reasoning fields are informational subsets, never re-added to totals.
  * Details a provider does not expose are ``None``, never fabricated zeros.

Field semantics were live-verified against every keyed provider
(findings/provider-usage-field-semantics.md): Anthropic-shaped providers report
``input_tokens`` exclusive of cache reads/writes; OpenAI-shaped providers report
``prompt_tokens`` inclusive of cached tokens; xAI alone bills reasoning outside
``completion_tokens``; Google reports ``candidates_token_count`` exclusive of
thoughts.

This module is internal and has no runtime dependency on the rest of the
package, so providers and ``models.py`` may import it freely.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:
    from .models import Message

logger = logging.getLogger(__name__)

REASON_NOT_REPORTED = "provider_did_not_report_usage"
REASON_MALFORMED = "provider_reported_malformed_usage"

ANTHROPIC_USAGE_PROVIDERS = frozenset({"anthropic", "moonshot", "zai", "kimi_coding"})
OPENAI_USAGE_PROVIDERS = frozenset(
    {
        "openai",
        "openai_chatgpt",
        "together",
        "groq",
        "fireworks",
        "llama",
        "xai",
        "dashscope",
        "deepseek",
        "ollama_cloud",
        "openrouter",
    }
)
GOOGLE_USAGE_PROVIDERS = frozenset({"google"})

# Live-verified exception: xAI's completion_tokens EXCLUDES reasoning_tokens and
# its own total is prompt + completion + reasoning. Every other OpenAI-shaped
# provider (including OpenRouter fronting Grok models) bills reasoning inside
# completion_tokens.
_OUTPUT_EXCLUDES_REASONING_PROVIDERS = frozenset({"xai"})

_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_input_tokens",
    "cache_write_input_tokens",
    "reasoning_output_tokens",
)


def usage_token_fields(provider: object) -> Optional[tuple[str, str]]:
    """Return the provider-shaped input/output field names for a known provider."""
    if not isinstance(provider, str):
        return None
    if provider in ANTHROPIC_USAGE_PROVIDERS:
        return "input_tokens", "output_tokens"
    if provider in OPENAI_USAGE_PROVIDERS:
        return "prompt_tokens", "completion_tokens"
    if provider in GOOGLE_USAGE_PROVIDERS:
        return "prompt_token_count", "candidates_token_count"
    return None


def _strict_nonneg_int(value: Any) -> Optional[int]:
    # type(value) is int rejects bools, floats, and numeric strings — a malformed
    # count must never be coerced into a billed number.
    return value if type(value) is int and value >= 0 else None


def read_detail_int(details: Any, name: str) -> Optional[int]:
    """Read an int field from an SDK usage-details object or its dict form."""
    value = getattr(details, name, None)
    if value is None and isinstance(details, dict):
        value = details.get(name)
    return _strict_nonneg_int(value)


@dataclass(frozen=True, slots=True)
class NormalizedUsage:
    """One provider-independent usage record for a completed message."""

    provider: str
    model: Optional[str]
    reported: bool
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    cache_read_input_tokens: Optional[int]
    cache_write_input_tokens: Optional[int]
    reasoning_output_tokens: Optional[int]
    unavailable_reason: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> Optional["NormalizedUsage"]:
        """Rebuild from a serialized dict; anything malformed deserializes to None."""
        if not isinstance(data, dict):
            return None
        provider = data.get("provider")
        model = data.get("model")
        reported = data.get("reported")
        reason = data.get("unavailable_reason")
        if not isinstance(provider, str) or not provider:
            return None
        if type(reported) is not bool:
            return None
        if model is not None and not isinstance(model, str):
            return None
        if reason is not None and not isinstance(reason, str):
            return None
        tokens: dict[str, Optional[int]] = {}
        for field in _TOKEN_FIELDS:
            value = data.get(field)
            if value is not None and _strict_nonneg_int(value) is None:
                return None
            tokens[field] = value
        return cls(provider=provider, model=model, reported=reported, unavailable_reason=reason, **tokens)


def _unreported(provider: str, model: Optional[str], reason: str) -> NormalizedUsage:
    return NormalizedUsage(
        provider=provider,
        model=model,
        reported=False,
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        cache_read_input_tokens=None,
        cache_write_input_tokens=None,
        reasoning_output_tokens=None,
        unavailable_reason=reason,
    )


_ABSENT = object()


def _classify(metadata: Mapping[str, Any], key: str) -> tuple[Optional[int], bool]:
    """Return (value, malformed) for a raw count: absent/None -> (None, False)."""
    raw = metadata.get(key, _ABSENT)
    if raw is _ABSENT or raw is None:
        return None, False
    value = _strict_nonneg_int(raw)
    return value, value is None


def _read_counts(
    metadata: Mapping[str, Any],
    core_keys: tuple[str, str],
    arithmetic_keys: tuple[str, ...],
    informational_keys: tuple[str, ...],
    provider: str,
) -> tuple[Optional[dict[str, Optional[int]]], Optional[str]]:
    """Validate raw counts into a dict, or classify why usage is unavailable.

    Returns (values, None) when core usage is reported, (_, reason) otherwise.
    A malformed core or arithmetic participant poisons the whole record: nulling
    a component would silently shift tokens out of the inclusive totals. A
    malformed informational subset only nulls itself — it never enters a total.
    """
    values: dict[str, Optional[int]] = {}
    malformed = False
    for key in core_keys + arithmetic_keys:
        value, bad = _classify(metadata, key)
        values[key] = value
        malformed = malformed or bad
    if malformed:
        return None, REASON_MALFORMED
    if any(values[key] is None for key in core_keys):
        return None, REASON_NOT_REPORTED
    for key in informational_keys:
        value, bad = _classify(metadata, key)
        if bad:
            logger.debug("%s usage: discarding malformed %s=%r", provider, key, metadata.get(key))
        values[key] = value
    return values, None


def _check_subset(values: dict[str, Optional[int]], subset_key: str, bound: int, provider: str) -> Optional[int]:
    subset = values.get(subset_key)
    if subset is not None and subset > bound:
        logger.warning("%s usage: %s=%d exceeds its superset (%d); discarding", provider, subset_key, subset, bound)
        return None
    return subset


def _log_total_mismatch(metadata: Mapping[str, Any], total_key: str, computed: int, provider: str) -> None:
    raw_total = _strict_nonneg_int(metadata.get(total_key))
    if raw_total is not None and raw_total != computed:
        logger.debug("%s usage: provider total %d != computed %d", provider, raw_total, computed)


def normalize_usage(
    usage_metadata: Optional[Mapping[str, Any]], provider_name: str, model: Optional[str]
) -> NormalizedUsage:
    """Compute the NormalizedUsage for one completed message. Pure function.

    ``provider_name`` is the client-resolved provider key and is authoritative
    for family selection; ``usage_metadata["provider"]`` is not consulted (it is
    absent on some no-usage paths). Unknown providers are a programming error —
    the guard test enumerates every ModelProvider value against the family map.
    """
    metadata = usage_metadata or {}
    if provider_name in ANTHROPIC_USAGE_PROVIDERS:
        return _normalize_anthropic(metadata, provider_name, model)
    if provider_name in OPENAI_USAGE_PROVIDERS:
        return _normalize_openai(metadata, provider_name, model)
    if provider_name in GOOGLE_USAGE_PROVIDERS:
        return _normalize_google(metadata, provider_name, model)
    raise ValueError(f"No usage-normalization family for provider {provider_name!r}")


def _normalize_anthropic(metadata: Mapping[str, Any], provider: str, model: Optional[str]) -> NormalizedUsage:
    # Raw input_tokens excludes cache reads and writes (live-verified for all
    # four providers); reconstruct the inclusive value. Thinking is billed
    # inside output_tokens; moonshot additionally reports it as a subset.
    values, reason = _read_counts(
        metadata,
        core_keys=("input_tokens", "output_tokens"),
        arithmetic_keys=("cache_read_input_tokens", "cache_write_input_tokens"),
        informational_keys=("reasoning_output_tokens",),
        provider=provider,
    )
    if reason is not None:
        return _unreported(provider, model, reason)
    assert values is not None
    cache_read = values["cache_read_input_tokens"]
    cache_write = values["cache_write_input_tokens"]
    input_tokens = values["input_tokens"] + (cache_read or 0) + (cache_write or 0)  # type: ignore[operator]
    output_tokens: int = values["output_tokens"]  # type: ignore[assignment]
    return NormalizedUsage(
        provider=provider,
        model=model,
        reported=True,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cache_read_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
        reasoning_output_tokens=_check_subset(values, "reasoning_output_tokens", output_tokens, provider),
        unavailable_reason=None,
    )


def _normalize_openai(metadata: Mapping[str, Any], provider: str, model: Optional[str]) -> NormalizedUsage:
    # Capture invariant: when cache_write_input_tokens is present it has already
    # been subtracted from the stored prompt_tokens (the OpenRouter adjustment in
    # providers/openai.py); adding it back restores the provider-inclusive input.
    # For xAI, reasoning is billed outside completion_tokens, so it participates
    # in output arithmetic instead of being a plain subset.
    reasoning_is_arithmetic = provider in _OUTPUT_EXCLUDES_REASONING_PROVIDERS
    arithmetic: tuple[str, ...] = ("cache_write_input_tokens",)
    informational: tuple[str, ...] = ("cache_read_input_tokens",)
    if reasoning_is_arithmetic:
        arithmetic += ("reasoning_output_tokens",)
    else:
        informational += ("reasoning_output_tokens",)
    values, reason = _read_counts(
        metadata,
        core_keys=("prompt_tokens", "completion_tokens"),
        arithmetic_keys=arithmetic,
        informational_keys=informational,
        provider=provider,
    )
    if reason is not None:
        return _unreported(provider, model, reason)
    assert values is not None
    cache_write = values["cache_write_input_tokens"]
    reasoning = values["reasoning_output_tokens"]
    input_tokens = values["prompt_tokens"] + (cache_write or 0)  # type: ignore[operator]
    output_tokens: int = values["completion_tokens"]  # type: ignore[assignment]
    if reasoning_is_arithmetic:
        output_tokens += reasoning or 0
    else:
        reasoning = _check_subset(values, "reasoning_output_tokens", output_tokens, provider)
    total = input_tokens + output_tokens
    _log_total_mismatch(metadata, "total_tokens", total, provider)
    return NormalizedUsage(
        provider=provider,
        model=model,
        reported=True,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
        cache_read_input_tokens=_check_subset(values, "cache_read_input_tokens", input_tokens, provider),
        cache_write_input_tokens=cache_write,
        reasoning_output_tokens=reasoning,
        unavailable_reason=None,
    )


def _normalize_google(metadata: Mapping[str, Any], provider: str, model: Optional[str]) -> NormalizedUsage:
    # prompt_token_count includes cached content; candidates_token_count excludes
    # thoughts; the SDK's own total is prompt + candidates + thoughts +
    # tool_use_prompt (live-verified), which matches the computed total here.
    values, reason = _read_counts(
        metadata,
        core_keys=("prompt_token_count", "candidates_token_count"),
        arithmetic_keys=("thoughts_token_count", "tool_use_prompt_token_count"),
        informational_keys=("cached_content_token_count",),
        provider=provider,
    )
    if reason is not None:
        return _unreported(provider, model, reason)
    assert values is not None
    thoughts = values["thoughts_token_count"]
    input_tokens = values["prompt_token_count"] + (values["tool_use_prompt_token_count"] or 0)  # type: ignore[operator]
    output_tokens = values["candidates_token_count"] + (thoughts or 0)  # type: ignore[operator]
    total = input_tokens + output_tokens
    _log_total_mismatch(metadata, "total_token_count", total, provider)
    return NormalizedUsage(
        provider=provider,
        model=model,
        reported=True,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
        cache_read_input_tokens=_check_subset(values, "cached_content_token_count", input_tokens, provider),
        cache_write_input_tokens=None,
        reasoning_output_tokens=thoughts,
        unavailable_reason=None,
    )


def attach_normalized_usage(message: "Message", provider_name: str, model: Optional[str]) -> "Message":
    """Attach the normalized usage to a completed message. Overwrite-safe:
    recomputing from the same raw metadata yields an equal value, so repeated
    stream finalization and double-attach paths stay idempotent."""
    message.usage = normalize_usage(message.usage_metadata, provider_name, model)
    return message
