# DashScope / Qwen models
DASHSCOPE_SPECS = {
    ("dashscope", "qwen3-coder-plus"): {
        "context_length": 1000000,
        "max_completion_tokens": 65536,
        "input_budget": "window_minus_output",
        "default_temperature": 0.7,
        "supports_vision": False,
    },
    ("dashscope", "qwen3-coder-flash"): {
        "context_length": 1000000,
        "max_completion_tokens": 65536,
        "input_budget": "window_minus_output",
        "default_temperature": 0.7,
        "supports_vision": False,
    },
}
