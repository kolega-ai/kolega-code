# Together.ai models
TOGETHER_SPECS = {
    ("together", "moonshotai/Kimi-K2.7-Code"): {
        "context_length": 262144,
        "max_completion_tokens": 32768,
        "default_temperature": 1.0,
        "supports_vision": True,
    },
    ("together", "zai-org/GLM-5.1"): {
        "context_length": 202752,
        "max_completion_tokens": 16384,
        "default_temperature": 1.0,
        "supports_vision": False,
    },
}
