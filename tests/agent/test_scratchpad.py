# ruff: noqa: F401,F811,E402
"""Tests for the per-session scratchpad directory (kolega_code.scratchpad)."""

import os
import stat
import subprocess
import uuid
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from kolega_code.agent.prompts import build_scratchpad_prompt
from kolega_code.agent.tool_backend.terminal_tool import TerminalTool
from kolega_code.scratchpad import (
    SCRATCHPAD_PROMPT_EXTENSION_ID,
    _user_suffix,
    ensure_scratchpad_dir,
    scratchpad_dir_for,
    scratchpad_root,
)


def _git_init(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)


class TestScratchpadPaths:
    def test_root_is_per_user_under_temp_dir(self, tmp_path: Path) -> None:
        root = scratchpad_root()

        # tests/conftest.py redirects tempfile.tempdir to the test tmp_path.
        assert root.parent == tmp_path
        assert root.name == f"kolega-code-{_user_suffix()}"
        assert root.name.startswith("kolega-code-")
        assert root.name != "kolega-code-"

    def test_dir_shape_uses_project_key_session_and_scratchpad(self, tmp_path: Path) -> None:
        project = tmp_path / "my-project"
        project.mkdir()

        path = scratchpad_dir_for(project, "session-abc")

        assert path.name == "scratchpad"
        assert path.parent.name == "session-abc"
        assert path.parent.parent.parent == scratchpad_root()
        # The project component is the project-memory directory key: name slug + digest.
        assert path.parent.parent.name.startswith("my-project-")

    def test_same_project_and_session_is_stable(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()

        assert scratchpad_dir_for(project, "s1") == scratchpad_dir_for(project, "s1")
        assert scratchpad_dir_for(str(project), "s1") == scratchpad_dir_for(project, "s1")

    def test_different_sessions_are_isolated(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()

        first = scratchpad_dir_for(project, "s1")
        second = scratchpad_dir_for(project, "s2")

        assert first != second
        assert first.parent.parent == second.parent.parent  # same project namespace

    def test_directories_in_one_git_repo_share_project_namespace(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _git_init(repo)
        sub_a = repo / "a"
        sub_b = repo / "b"
        sub_a.mkdir()
        sub_b.mkdir()

        path_a = scratchpad_dir_for(sub_a, "s1")
        path_b = scratchpad_dir_for(sub_b, "s1")

        assert path_a.parent.parent == path_b.parent.parent

    def test_separate_git_clones_do_not_share_project_namespace(self, tmp_path: Path) -> None:
        repo_one = tmp_path / "one"
        repo_two = tmp_path / "two"
        _git_init(repo_one)
        _git_init(repo_two)

        assert scratchpad_dir_for(repo_one, "s1").parent.parent != scratchpad_dir_for(repo_two, "s1").parent.parent

    @pytest.mark.parametrize("bad_id", ["", "   ", ".", "..", "a/b", "a\\b", "a\0b"])
    def test_invalid_session_id_rejected(self, tmp_path: Path, bad_id: str) -> None:
        with pytest.raises(ValueError):
            scratchpad_dir_for(tmp_path, bad_id)

    def test_session_id_is_stripped(self, tmp_path: Path) -> None:
        assert scratchpad_dir_for(tmp_path, "  s1  ") == scratchpad_dir_for(tmp_path, "s1")


class TestEnsureScratchpadDir:
    def test_creates_owner_only_chain(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()

        old_umask = os.umask(0)
        try:
            path = ensure_scratchpad_dir(project, "session-xyz")
        finally:
            os.umask(old_umask)

        assert path == scratchpad_dir_for(project, "session-xyz")
        assert path.is_dir()
        if os.name != "nt":
            assert stat.S_IMODE(path.stat().st_mode) == 0o700
            assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
            assert stat.S_IMODE(scratchpad_root().stat().st_mode) == 0o700

    def test_second_call_is_noop(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()

        first = ensure_scratchpad_dir(project, "s1")
        marker = first / "keep.txt"
        marker.write_text("still here", encoding="utf-8")
        second = ensure_scratchpad_dir(project, "s1")

        assert first == second
        assert marker.read_text(encoding="utf-8") == "still here"


class TestScratchpadPrompt:
    def test_renders_path_and_core_rules(self, tmp_path: Path) -> None:
        rendered = build_scratchpad_prompt(tmp_path / "scratchpad")

        assert str(tmp_path / "scratchpad") in rendered
        # Throwaway-only: no deliverables, OS may delete at any time.
        assert "deliverables" in rendered
        assert "throwaway" in rendered
        # Shared-directory etiquette for sub-agents.
        assert "uniquely named files" in rendered
        assert "did not create" in rendered
        # Sanctioned out-of-worktree location.
        assert "outside the working directory" in rendered

    def test_template_is_fully_rendered(self, tmp_path: Path) -> None:
        rendered = build_scratchpad_prompt(tmp_path / "scratchpad")

        assert "{{" not in rendered
        assert "}}" not in rendered


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self._text = text

    def get_text_content(self) -> str:
        return self._text


@pytest.fixture
def terminal_tool(tmp_path, mock_connection_manager, agent_config):
    caller = Mock()
    caller.agent_name = "test_agent"
    caller.scratchpad_dir = None
    tool = TerminalTool(tmp_path, "test_workspace", str(uuid.uuid4()), mock_connection_manager, agent_config, caller)
    tool.initialized = True
    return tool


class TestTerminalSafetyCheck:
    @staticmethod
    def _patch_security_llm(monkeypatch: pytest.MonkeyPatch, captured: dict, verdict: str = "safe") -> None:
        class _FakeClient:
            def __init__(self, **kwargs) -> None:
                pass

            async def generate(self, **kwargs):
                captured.update(kwargs)
                return _FakeResponse(verdict)

        monkeypatch.setattr("kolega_code.agent.tool_backend.terminal_tool.LLMClient", _FakeClient)
        monkeypatch.setattr(
            "kolega_code.agent.tool_backend.terminal_tool.get_model_specs",
            lambda *args, **kwargs: {"max_completion_tokens": 16},
        )

    @staticmethod
    def _user_text(captured: dict) -> str:
        message = captured["messages"][0]
        return "\n".join(block.text for block in message.content)

    @pytest.mark.asyncio
    async def test_security_check_includes_scratchpad_directory(self, terminal_tool, monkeypatch) -> None:
        scratchpad = Path("/tmp/kolega-code-test/key/session/scratchpad")
        terminal_tool.caller.scratchpad_dir = scratchpad
        captured: dict = {}
        self._patch_security_llm(monkeypatch, captured)

        ok, _ = await terminal_tool._run_command_security_check("echo hi > scratchpad/script.py")

        assert ok is True
        text = self._user_text(captured)
        assert "Project directory:" in text
        assert f"Scratchpad directory (session-writable):\n{scratchpad}" in text
        assert "echo hi > scratchpad/script.py" in text

    @pytest.mark.asyncio
    async def test_security_check_omits_scratchpad_when_unavailable(self, terminal_tool, monkeypatch) -> None:
        terminal_tool.caller.scratchpad_dir = None
        captured: dict = {}
        self._patch_security_llm(monkeypatch, captured)

        ok, _ = await terminal_tool._run_command_security_check("echo hi")

        assert ok is True
        assert "Scratchpad directory" not in self._user_text(captured)

    def test_safety_prompt_whitelists_scratchpad(self) -> None:
        from kolega_code.agent import prompts

        assert "scratchpad" in prompts.SHELL_SAFETY_SYSTEM_PROMPT
        assert "session scratchpad directory" in prompts.SHELL_SAFETY_SYSTEM_PROMPT


class TestBaseAgentFallback:
    """The BaseAgent fallback gives non-TUI local hosts a scratchpad for free."""

    @staticmethod
    def _make_agent(tmp_path: Path, agent_config, mock_connection_manager, **overrides):
        from kolega_code.agent.baseagent import BaseAgent

        kwargs = {
            "project_path": tmp_path,
            "workspace_id": "test_workspace",
            "thread_id": "thread-abc123",
            "connection_manager": mock_connection_manager,
            "config": agent_config,
        }
        kwargs.update(overrides)
        return BaseAgent(**kwargs)

    @staticmethod
    def _scratchpad_extensions(agent) -> list:
        return [ext for ext in agent.prompt_extensions if getattr(ext, "id", None) == SCRATCHPAD_PROMPT_EXTENSION_ID]

    def test_top_level_local_agent_gets_scratchpad(self, tmp_path, agent_config, mock_connection_manager) -> None:
        agent = self._make_agent(tmp_path, agent_config, mock_connection_manager)

        extensions = self._scratchpad_extensions(agent)
        assert len(extensions) == 1
        extension = extensions[0]
        expected = scratchpad_dir_for(tmp_path, "thread-abc123")
        assert str(expected) in extension.markdown
        assert extension.propagate_to_sub_agents is True
        assert extension.modes is None  # fallback serves any host mode
        assert agent.scratchpad_dir == expected
        assert expected.is_dir()

    def test_sub_agent_gets_no_fallback(self, tmp_path, agent_config, mock_connection_manager) -> None:
        agent = self._make_agent(tmp_path, agent_config, mock_connection_manager, sub_agent=True)

        assert self._scratchpad_extensions(agent) == []
        assert agent.scratchpad_dir is None

    def test_non_local_filesystem_gets_no_fallback(self, tmp_path, agent_config, mock_connection_manager) -> None:
        from kolega_code.services.file_system import FileSystem

        sandbox_fs = MagicMock(spec=FileSystem)
        agent = self._make_agent(tmp_path, agent_config, mock_connection_manager, filesystem=sandbox_fs)

        assert self._scratchpad_extensions(agent) == []
        assert agent.scratchpad_dir is None

    def test_host_injected_extension_is_not_duplicated(self, tmp_path, agent_config, mock_connection_manager) -> None:
        from kolega_code.agent.prompt_provider import PromptExtension

        injected = PromptExtension(
            id=SCRATCHPAD_PROMPT_EXTENSION_ID,
            title="Session Scratchpad",
            markdown="host-managed scratchpad",
        )
        agent = self._make_agent(tmp_path, agent_config, mock_connection_manager, prompt_extensions=[injected])

        extensions = self._scratchpad_extensions(agent)
        assert len(extensions) == 1
        assert extensions[0].markdown == "host-managed scratchpad"
        # The host owns the extension; the fallback does not resolve a directory.
        assert agent.scratchpad_dir is None

    def test_scratchpad_creation_failure_is_non_fatal(
        self, tmp_path, agent_config, mock_connection_manager, monkeypatch
    ) -> None:
        def _boom(*args, **kwargs):
            raise OSError("temp dir unavailable")

        monkeypatch.setattr("kolega_code.scratchpad.ensure_scratchpad_dir", _boom)
        # baseagent imports ensure_scratchpad_dir lazily inside __init__, so patch
        # the symbol it looks up on the module.
        agent = self._make_agent(tmp_path, agent_config, mock_connection_manager)

        assert self._scratchpad_extensions(agent) == []
        assert agent.scratchpad_dir is None
