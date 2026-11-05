from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kolega_code.agent import baseagent
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


@pytest.mark.parametrize(
    "model",
    [
        "deepseek-v4.1-flash-abliterated",
        "deepseek-v4.1-flash-fast",
    ],
)
def test_refuseless_does_not_inherit_deepseek_truncation_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mock_connection_manager: AsyncMock, model: str
) -> None:
    def synthetic_specs(_provider: Any, _model: str) -> dict[str, Any]:
        return {"context_length": 4096, "max_completion_tokens": 1024, "input_budget": "window_minus_output"}

    monkeypatch.setattr(baseagent, "get_model_specs", synthetic_specs)
    config = AgentConfig(
        anthropic_api_key="fake-anthropic-key",
        refuseless_api_key="abl_FAKE_refuseless_key",
        long_context_config=ModelConfig(provider=ModelProvider.REFUSELESS, model=model),
    )
    agent = baseagent.BaseAgent(
        project_path=tmp_path,
        workspace_id="test-refuseless",
        thread_id="test-refuseless-truncation",
        connection_manager=mock_connection_manager,
        config=config,
    )
    assert agent.deepseek_wire_output_cap is None
