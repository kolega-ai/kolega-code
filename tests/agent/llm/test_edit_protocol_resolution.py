from kolega_code.config import AgentConfig, EditProtocol, ModelConfig, ModelProvider
from kolega_code.llm.specs import MODEL_SPECS, preferred_edit_protocol


def config(*, edit_protocol: EditProtocol | None = None) -> AgentConfig:
    model = ModelConfig(provider=ModelProvider.ANTHROPIC, model="claude-opus-4-8")
    return AgentConfig(
        anthropic_api_key="test",
        long_context_config=model,
        fast_config=model,
        thinking_config=model,
        edit_protocol=edit_protocol,
    )


def test_protocol_defaults_to_search_replace_without_catalog_preference() -> None:
    value = config()

    assert value.edit_protocol is None
    assert value.resolve_edit_protocol_with_source() == (EditProtocol.SEARCH_REPLACE, "default")


def test_catalog_preference_is_provider_and_model_specific(monkeypatch) -> None:
    key = ("anthropic", "claude-opus-4-8")
    monkeypatch.setitem(MODEL_SPECS[key], "preferred_edit_protocol", "claude_code")

    value = config()

    assert preferred_edit_protocol(*key) == "claude_code"
    assert value.resolve_edit_protocol_with_source() == (EditProtocol.CLAUDE_CODE, "model_catalog")
    assert preferred_edit_protocol("openai", "claude-opus-4-8") is None


def test_session_override_wins_over_catalog_preference(monkeypatch) -> None:
    monkeypatch.setitem(MODEL_SPECS[("anthropic", "claude-opus-4-8")], "preferred_edit_protocol", "claude_code")

    value = config(edit_protocol=EditProtocol.CODEX_APPLY_PATCH)

    assert value.resolve_edit_protocol_with_source() == (
        EditProtocol.CODEX_APPLY_PATCH,
        "session_override",
    )


def test_role_model_resolves_its_own_catalog_preference(monkeypatch) -> None:
    override_model = ModelConfig(provider=ModelProvider.ANTHROPIC, model="claude-haiku-4-5-20251001")
    monkeypatch.setitem(
        MODEL_SPECS[("anthropic", "claude-haiku-4-5-20251001")],
        "preferred_edit_protocol",
        "claude_code",
    )
    value = config().model_copy(update={"agent_models": {"building": override_model}})

    effective_model = value.model_config_for_agent("coder")

    assert effective_model == override_model
    assert value.resolve_edit_protocol(effective_model) == EditProtocol.CLAUDE_CODE


def test_every_catalog_protocol_preference_is_a_production_protocol() -> None:
    invalid = {
        key: specs.get("preferred_edit_protocol")
        for key, specs in MODEL_SPECS.items()
        if specs.get("preferred_edit_protocol") is not None
        and specs.get("preferred_edit_protocol") not in {protocol.value for protocol in EditProtocol}
    }

    assert invalid == {}


def test_all_openai_catalog_models_prefer_codex_apply_patch() -> None:
    openai_models = {
        key: specs.get("preferred_edit_protocol")
        for key, specs in MODEL_SPECS.items()
        if key[0] in {"openai", "openai_chatgpt"}
    }

    assert openai_models
    assert set(openai_models.values()) == {EditProtocol.CODEX_APPLY_PATCH.value}


def test_direct_deepseek_models_prefer_claude_code() -> None:
    deepseek_models = {
        key: specs.get("preferred_edit_protocol") for key, specs in MODEL_SPECS.items() if key[0] == "deepseek"
    }

    assert deepseek_models == {
        ("deepseek", "deepseek-v4-pro"): EditProtocol.CLAUDE_CODE.value,
        ("deepseek", "deepseek-v4-flash"): EditProtocol.CLAUDE_CODE.value,
    }
