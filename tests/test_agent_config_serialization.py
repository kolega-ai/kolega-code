"""T5: ``AgentConfig`` serialization excludes secret-bearing config fields.

``lsp`` (which carries ``custom_servers.env`` and ``workspace_configuration``,
both of which can hold secrets) and ``mcp_config`` must be excluded from
``model_dump()`` — parity with the existing ``mcp_config`` exclusion.
"""

from __future__ import annotations

from kolega_code.config import AgentConfig, LspConfig, ModelConfig, ModelProvider, RateLimitConfig


def _minimal_config(**overrides) -> AgentConfig:
    return AgentConfig(
        anthropic_api_key="test_key",
        long_context_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig()
        ),
        fast_config=ModelConfig(provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig()),
        thinking_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="test-model",
            rate_limits=RateLimitConfig(),
            thinking_effort="medium",
        ),
        **overrides,
    )


def test_model_dump_excludes_lsp():
    """F11/T5: the ``lsp`` field (with custom_servers env) is excluded from dumps."""
    config = _minimal_config(
        lsp=LspConfig(
            enabled=True,
            custom_servers={"my-server": {"bin": "my-server", "env": {"SECRET": "sk-leaked"}}},
            workspace_configuration={"pyright": {"token": "sk-leaked"}},
        ),
    )

    dumped = config.model_dump()

    assert "lsp" not in dumped
    # The secret must not leak through the serialized form.
    serialized = repr(dumped)
    assert "sk-leaked" not in serialized


def test_model_dump_excludes_mcp_config():
    """T5: ``mcp_config`` remains excluded (parity check)."""
    config = _minimal_config(mcp_config={"servers": {"x": {"env": {"SECRET": "sk-leaked"}}}})

    dumped = config.model_dump()

    assert "mcp_config" not in dumped
    assert "sk-leaked" not in repr(dumped)


def test_model_dump_excludes_lsp_project_trusted():
    """T5: the runtime trust flag is excluded from dumps."""
    config = _minimal_config(lsp_project_trusted=True)

    dumped = config.model_dump()

    assert "lsp_project_trusted" not in dumped


def test_lsp_field_round_trip_via_attribute():
    """T5: although excluded from dumps, the field is still accessible on the model."""
    lsp = LspConfig(enabled=False, max_diagnostics=5)
    config = _minimal_config(lsp=lsp)

    assert config.lsp is lsp
    assert config.lsp.enabled is False
    assert config.lsp.max_diagnostics == 5
