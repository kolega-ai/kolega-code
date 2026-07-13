from pathlib import Path
import tomllib

import pytest

from benchmarks.edit_tools.__main__ import _catalog_smoke_matrix
from benchmarks.edit_tools.protocols import create_tool_collection, get_protocol
from kolega_code.config import AgentConfig, EditProtocol, ModelConfig, ModelProvider
from kolega_code.llm.providers.models import GenerationParams
from kolega_code.llm.providers.responses_common import responses_tools
from kolega_code.llm.specs import MODEL_SPECS
from kolega_code.services.lsp.config import LspConfig


def config() -> AgentConfig:
    model = ModelConfig(provider=ModelProvider.ANTHROPIC, model="claude-haiku-4-5-20251001")
    return AgentConfig(
        anthropic_api_key="test",
        long_context_config=model,
        fast_config=model,
        thinking_config=model,
        edit_protocol=EditProtocol.CODEX_APPLY_PATCH,
        lsp=LspConfig(enabled=False),
    )


def catalog_providers() -> list[str]:
    return list(dict.fromkeys(provider for provider, _ in MODEL_SPECS))


def test_provider_smoke_matrix_covers_every_catalog_provider() -> None:
    matrix = _catalog_smoke_matrix()

    assert [item.provider for item in matrix.models] == catalog_providers()
    assert len(matrix.models) == 13
    assert all(item.protocols == ["search_replace", "codex_apply_patch"] for item in matrix.models)


@pytest.mark.parametrize("provider", catalog_providers())
@pytest.mark.asyncio
async def test_every_catalog_provider_has_a_freeform_transport_contract(provider: str, tmp_path: Path) -> None:
    workspace = tmp_path / provider
    workspace.mkdir()
    adapter = get_protocol("codex_apply_patch")
    collection, _, _ = create_tool_collection(workspace, config(), adapter, tmp_path / "artifacts")
    try:
        definition = next(item for item in adapter.definitions(collection) if item.name == "apply_patch")

        if provider in {"openai", "openai_chatgpt"}:
            tools = responses_tools(GenerationParams(tools=[definition]))
            assert tools is not None
            assert tools[0]["type"] == "custom"
            assert tools[0]["format"]["syntax"] == "lark"
        elif provider == "google":
            declarations = definition.to_google().function_declarations
            assert declarations is not None
            declaration = declarations[0]
            assert declaration.parameters is not None
            assert declaration.parameters.required == ["input"]
        elif provider in {"anthropic", "moonshot", "zai", "kimi_coding"}:
            assert definition.to_anthropic()["input_schema"]["required"] == ["input"]
        else:
            assert definition.to_openai()["function"]["parameters"]["required"] == ["input"]
    finally:
        await collection.cleanup()


def test_benchmark_is_not_part_of_installed_package_or_console_scripts() -> None:
    root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["kolega_code"]
    assert set(pyproject["project"]["scripts"]) == {"kolega-code"}
