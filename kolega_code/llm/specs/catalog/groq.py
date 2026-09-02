"""Groq catalog specs: fast OpenAI-compatible open-weight inference.

Groq serves open-weight models (gpt-oss, Qwen, Llama 4) through an
OpenAI-compatible chat-completions endpoint at 131K context. The client
already routes ``groq`` to https://api.groq.com/openai/v1; these specs make
the models catalogued, validatable, and visible in the provider picker.
The deprecated llama-3.1-8b-instant / llama-3.3-70b-versatile entries are
deliberately absent (Groq retired them 2026-08-16).
"""

GROQ_SPECS = {
    ("groq", "openai/gpt-oss-120b"): {
        "context_length": 131072,
        "max_completion_tokens": 16384,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        "supports_vision": False,
    },
    ("groq", "openai/gpt-oss-20b"): {
        "context_length": 131072,
        "max_completion_tokens": 16384,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        "supports_vision": False,
    },
    ("groq", "qwen/qwen3.6-27b"): {
        "context_length": 131072,
        "max_completion_tokens": 16384,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        "supports_vision": True,
    },
    ("groq", "qwen/qwen3-32b"): {
        "context_length": 131072,
        "max_completion_tokens": 16384,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        "supports_vision": False,
    },
    ("groq", "meta-llama/llama-4-scout-17b-16e-instruct"): {
        "context_length": 131072,
        "max_completion_tokens": 16384,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        "supports_vision": True,
    },
    ("groq", "meta-llama/llama-4-maverick-17b-128e-instruct"): {
        "context_length": 131072,
        "max_completion_tokens": 16384,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        "supports_vision": True,
    },
}
