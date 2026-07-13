"""Provider-neutral token aggregation for benchmark transcripts."""

from __future__ import annotations

from typing import Any

from .models import UsageTotals


ANTHROPIC_SHAPED = {"anthropic", "moonshot", "zai", "kimi_coding"}
OPENAI_SHAPED = {
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
}


def add_usage(total: UsageTotals, metadata: dict[str, Any], provider: str) -> None:
    source = str(metadata.get("provider") or provider)
    if source in ANTHROPIC_SHAPED:
        total.input_tokens += int(metadata.get("input_tokens") or 0)
        total.output_tokens += int(metadata.get("output_tokens") or 0)
    elif source in OPENAI_SHAPED:
        total.input_tokens += int(metadata.get("prompt_tokens") or metadata.get("input_tokens") or 0)
        total.output_tokens += int(metadata.get("completion_tokens") or metadata.get("output_tokens") or 0)
    elif source == "google":
        total.input_tokens += int(metadata.get("prompt_token_count") or 0)
        total.output_tokens += int(metadata.get("candidates_token_count") or 0)
    total.cache_read_input_tokens += int(metadata.get("cache_read_input_tokens") or 0)
    total.cache_write_input_tokens += int(metadata.get("cache_write_input_tokens") or 0)
    total.requests += 1
