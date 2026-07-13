from pathlib import Path

import pytest

from benchmarks.edit_tools.models import (
    FileContent,
    MatrixSpec,
    ModelRunSpec,
    SuiteSpec,
    TaskSpec,
)
from benchmarks.edit_tools.runner import _controlled_execution, plan_trials, run_benchmark
from benchmarks.edit_tools.workspace import materialize_task, verify_task
from kolega_code.config import AgentConfig, EditProtocol, ModelConfig, ModelProvider
from kolega_code.llm.models import Message, TextBlock, ToolCall
from kolega_code.llm.specs import default_thinking_effort
from kolega_code.services.lsp.config import LspConfig


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    async def generate(self, **kwargs):
        return self.responses.pop(0)


def benchmark_config(protocol: EditProtocol = EditProtocol.SEARCH_REPLACE) -> AgentConfig:
    model = ModelConfig(provider=ModelProvider.ANTHROPIC, model="claude-haiku-4-5-20251001")
    return AgentConfig(
        anthropic_api_key="test",
        long_context_config=model,
        fast_config=model,
        thinking_config=model,
        edit_protocol=protocol,
        lsp=LspConfig(enabled=False),
    )


def task(required_capabilities: set[str] | None = None) -> TaskSpec:
    return TaskSpec(
        id="simple",
        prompt="Change a.txt from before to after.",
        before_files={"a.txt": FileContent(text="before\n")},
        expected_files={"a.txt": FileContent(text="after\n")},
        required_capabilities=required_capabilities or {"update"},
    )


def matrix(protocol: str = "search_replace") -> MatrixSpec:
    return MatrixSpec(
        id="test",
        models=[
            ModelRunSpec(
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                protocols=[protocol],
            )
        ],
        repetitions=1,
    )


def test_plan_uses_catalog_defaults_and_stable_trial_ids() -> None:
    suite = SuiteSpec(id="test", curated_tasks=[task()])
    first = plan_trials(suite, [task()], matrix())
    second = plan_trials(suite, [task()], matrix())

    assert first[0].model_parameters["thinking_effort"] == default_thinking_effort(
        "anthropic", "claude-haiku-4-5-20251001"
    )
    assert first[0].trial_id == second[0].trial_id


@pytest.mark.asyncio
async def test_controlled_loop_executes_real_edit_tool(monkeypatch, tmp_path: Path) -> None:
    suite = SuiteSpec(id="test", curated_tasks=[task()])
    trial = plan_trials(suite, [task()], matrix())[0]
    edit_call = ToolCall(
        id="call-1",
        name="edit",
        input={
            "path": "a.txt",
            "block": "<<<<<<< SEARCH\nbefore\n=======\nafter\n>>>>>>> REPLACE",
        },
    )
    fake = FakeClient(
        [
            Message(
                role="assistant",
                content=[edit_call],
                tool_calls=[edit_call],
                usage_metadata={"provider": "anthropic", "input_tokens": 10, "output_tokens": 5},
            ),
            Message(role="assistant", content=[TextBlock(text="done")], usage_metadata={"provider": "anthropic"}),
        ]
    )
    monkeypatch.setattr("benchmarks.edit_tools.runner._client", lambda config, provider: fake)
    workspace = tmp_path / "workspace"
    materialize_task(workspace, task())

    result = await _controlled_execution(workspace, tmp_path / "artifacts", benchmark_config(), trial)

    assert result.terminal_stop
    assert result.attempts[0].apply_ok
    assert result.usage.input_tokens == 10
    assert (await verify_task(workspace, task())).success


@pytest.mark.asyncio
async def test_resume_does_not_duplicate_completed_records(tmp_path: Path) -> None:
    unsupported = task({"move"})
    suite = SuiteSpec(id="resume", curated_tasks=[unsupported])
    run_dir, first = await run_benchmark(
        repo_root=Path(__file__).resolve().parents[2],
        suite=suite,
        tasks=[unsupported],
        matrix=matrix("search_replace"),
        output_root=tmp_path,
    )
    resumed_dir, second = await run_benchmark(
        repo_root=Path(__file__).resolve().parents[2],
        suite=suite,
        tasks=[unsupported],
        matrix=matrix("search_replace"),
        output_root=tmp_path,
        resume_dir=run_dir,
    )

    assert resumed_dir == run_dir
    assert len(first) == len(second) == 1
    assert second[0].status == "unsupported"
    assert len((run_dir / "trials.jsonl").read_text().splitlines()) == 1
