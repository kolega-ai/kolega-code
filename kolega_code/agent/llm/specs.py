from typing import Dict, Tuple

# Dictionary mapping (provider, model_name) to model specifications
# Each entry contains context_length (maximum input tokens), max_completion_tokens, and default_temperature
MODEL_SPECS: Dict[Tuple[str, str], Dict[str, int | float]] = {
    # Anthropic models
    ("anthropic", "claude-opus-4-7"): {"context_length": 1000000, "max_completion_tokens": 128000, "default_temperature": 1.0},
    ("anthropic", "claude-sonnet-4-6"): {"context_length": 1000000, "max_completion_tokens": 64000, "default_temperature": 1.0},
    ("anthropic", "claude-3-7-sonnet-20250219"): {"context_length": 200000, "max_completion_tokens": 16384, "default_temperature": 1.0},
    ("anthropic", "claude-3-haiku-20240307"): {"context_length": 200000, "max_completion_tokens": 4096, "default_temperature": 1.0},
    ("anthropic", "claude-3-5-sonnet-20241022"): {"context_length": 200000, "max_completion_tokens": 8192, "default_temperature": 1.0},
    ("anthropic", "claude-opus-4-20250514"): {"context_length": 200000, "max_completion_tokens": 16384, "default_temperature": 1.0},
    ("anthropic", "claude-sonnet-4-20250514"): {"context_length": 200000, "max_completion_tokens": 16384, "default_temperature": 1.0},
    ("anthropic", "claude-sonnet-4-5-20250929"): {"context_length": 200000, "max_completion_tokens": 16384, "default_temperature": 1.0},
    ("anthropic", "claude-opus-4-5-20251101"): {"context_length": 200000, "max_completion_tokens": 16384, "default_temperature": 1.0},
    ("anthropic", "claude-haiku-4-5-20251001"): {"context_length": 200000, "max_completion_tokens": 16384, "default_temperature": 1.0},
    # Moonshot models
    ("moonshot", "kimi-k2.6"): {"context_length": 262144, "max_completion_tokens": 32768, "default_temperature": 1.0},
    # DeepSeek models
    ("deepseek", "deepseek-v4-pro"): {"context_length": 1000000, "max_completion_tokens": 384000, "default_temperature": 1.0},
    # OpenAI models
    ("openai", "gpt-4o"): {"context_length": 128000, "max_completion_tokens": 4096, "default_temperature": 1.0},
    ("openai", "o3-mini"): {"context_length": 200000, "max_completion_tokens": 16384, "default_temperature": 1.0},
    ("openai", "gpt-4.1-2025-04-14"): {"context_length": 1000000, "max_completion_tokens": 32768, "default_temperature": 1.0},
    ("openai", "gpt-4.1-mini"): {"context_length": 1000000, "max_completion_tokens": 32768, "default_temperature": 1.0},
    ("openai", "o3-2025-04-16"): {"context_length": 200000, "max_completion_tokens": 100000, "default_temperature": 1.0},
    ("openai", "o3"): {"context_length": 200000, "max_completion_tokens": 100000, "default_temperature": 1.0},
    ("openai", "o4-mini"): {"context_length": 200000, "max_completion_tokens": 100000, "default_temperature": 1.0},
    # Together.ai models
    ("together", "deepseek-ai/DeepSeek-R1"): {"context_length": 64000, "max_completion_tokens": 8000, "default_temperature": 1.0},
    # Google models
    ("google", "gemini-2.0-flash"): {"context_length": 1000000, "max_completion_tokens": 8192, "default_temperature": 1.0},
    ("google", "gemini-2.5-pro-exp-03-25"): {"context_length": 1000000, "max_completion_tokens": 65536, "default_temperature": 1.0},
    ("google", "gemini-2.5-pro"): {"context_length": 1000000, "max_completion_tokens": 65536, "default_temperature": 1.0},
    # X.ai models
    ("xai", "grok-3-beta"): {"context_length": 128000, "max_completion_tokens": 16384, "default_temperature": 1.0},
    # Fireworks models
    ("fireworks", "accounts/fireworks/models/glm-4p5"): {"context_length": 128000, "max_completion_tokens": 16384, "default_temperature": 0.6},
    ("dashscope", "qwen3-coder-plus"): {"context_length": 1000000, "max_completion_tokens": 16384, "default_temperature": 0.7},
}


def get_model_specs(provider: str, model_name: str) -> Dict[str, int | float]:
    """
    Get the specifications for a given model.

    Args:
        provider: The LLM provider (e.g., 'anthropic', 'openai') - can be string or enum
        model_name: The name of the model

    Returns:
        Dictionary containing context_length, max_completion_tokens, and default_temperature
    """
    # Handle both string and enum provider types
    provider_str = provider.value if hasattr(provider, "value") else provider
    key = (provider_str, model_name)

    if key not in MODEL_SPECS:
        raise ValueError(f"Model {model_name} from provider {provider_str} is not supported.")

    return MODEL_SPECS.get(key)
