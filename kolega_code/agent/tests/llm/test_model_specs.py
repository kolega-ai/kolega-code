from kolega_code.agent.llm.specs import get_model_specs


def test_kimi_k26_model_specs():
    specs = get_model_specs("moonshot", "kimi-k2.6")

    assert specs["context_length"] == 262144
    assert specs["max_completion_tokens"] == 32768
    assert specs["default_temperature"] == 1.0
