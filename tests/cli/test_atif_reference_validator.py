"""Representative ATIF exports validate against the reference implementation
in a clean, isolated environment.

The `atif` PyPI package is the ATIF RFC's reference implementation; validating
in a throwaway `uvx` environment proves our documents stand on their own,
independent of this repo's installed dependency tree. Harbor itself (0.21.0)
ships no separate importable/CLI trajectory validator and does not depend on
`atif`; if a future Harbor version grows a validator surface, extend this test
to invoke it as well.

Slow-marked: provisions an environment at test time and skips cleanly when
that is unavailable (offline CI).
"""

import json
import shutil
import subprocess

import pytest

from kolega_code.cli.atif_export import export_atif_to_text
from kolega_code.llm.models import Message, TextBlock, ToolCall, ToolResult
from kolega_code.llm.usage import normalize_usage

from .test_atif_export import bootstrap, make_journal, settled

ATIF_REFERENCE_PIN = "atif==1.7.0"

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _reference_validate(document_text: str) -> subprocess.CompletedProcess:
    uvx = shutil.which("uvx")
    if uvx is None:
        pytest.skip("uvx is not available to provision the reference validator")
    return subprocess.run(
        [
            uvx,
            "--with",
            ATIF_REFERENCE_PIN,
            "python",
            "-c",
            "import atif, json, sys; atif.Trajectory.model_validate(json.load(sys.stdin)); print('VALID')",
        ],
        input=document_text,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_representative_exports_pass_the_reference_validator():
    from kolega_code.llm.ledger import UsageLedger

    journal = make_journal(session_id="ref-validate")
    recorder = bootstrap(journal)
    ledger = UsageLedger()
    recorder.start_turn(Message(role="user", content=[TextBlock("multi tool")]))
    first = settled(
        ledger,
        Message(
            role="assistant",
            content=[
                TextBlock("working"),
                ToolCall(id="a", name="read_file", input={"path": "x"}, execution_id="tool_exec_a"),
                ToolCall(id="b", name="bash", input={"cmd": "ls"}, execution_id="tool_exec_b"),
            ],
            stop_reason="tool_use",
            usage=normalize_usage({"input_tokens": 10, "output_tokens": 4}, "anthropic", "m"),
        ),
    )
    recorder.record_assistant(first)
    recorder.record_tool_results(
        [
            ToolResult(tool_use_id="b", content="dir", name="bash", is_error=False, execution_id="tool_exec_b"),
            ToolResult(tool_use_id="a", content="text", name="read_file", is_error=False, execution_id="tool_exec_a"),
        ]
    )
    child = recorder.scoped_child(agent_id="sub-ref", agent_name="worker", parent_tool_call_id="tool_exec_b", depth=1)
    child.record_agent_started({"agent_name": "worker", "task": "t"}, turn_id=recorder.current_turn_id)
    child.start_turn(Message(role="user", content=[TextBlock("t")]))
    child.record_assistant(
        settled(
            ledger,
            Message(
                role="assistant",
                content=[TextBlock("done")],
                stop_reason="end_turn",
                usage=normalize_usage({"input_tokens": 3, "output_tokens": 1}, "anthropic", "m"),
            ),
        )
    )
    child.finish_turn("completed")
    child.record_agent_terminal("completed", {"summary": "done"})
    recorder.record_synthetic_assistant("notice", notice_code="skill_activation")
    recorder.finish_turn("completed")
    recorder.record_run_terminal("completed", {"status": "completed", "exit_code": 0})

    text = export_atif_to_text(journal, kolega_version="0.0.0-test", secret_values=(), state_dirs=())
    try:
        result = _reference_validate(text)
    except (OSError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"Could not provision the reference validator: {exc}")
    if result.returncode != 0 and "VALID" not in result.stdout:
        if "No solution found" in result.stderr or "network" in result.stderr.lower():
            pytest.skip(f"Reference environment unavailable: {result.stderr[-200:]}")
        pytest.fail(f"Reference validator rejected the document: {result.stderr[-2000:]}")
    assert "VALID" in result.stdout
    # Sanity: the document round-trips as JSON with embedded subagents.
    document = json.loads(text)
    assert document["subagent_trajectories"][0]["trajectory_id"] == "sub-ref"
