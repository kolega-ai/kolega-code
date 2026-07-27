from __future__ import annotations

from kolega_code.git_env import GIT_ENV_OVERRIDES, git_env


def test_strips_every_repository_retargeting_variable() -> None:
    base = {name: "/somewhere" for name in GIT_ENV_OVERRIDES}
    base["PATH"] = "/usr/bin"

    assert git_env(base) == {"PATH": "/usr/bin"}


def test_preserves_unrelated_and_other_git_variables() -> None:
    base = {
        "GIT_AUTHOR_NAME": "Test User",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/home/test",
        "GIT_DIR": "/elsewhere/.git",
    }

    assert git_env(base) == {
        "GIT_AUTHOR_NAME": "Test User",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/home/test",
    }


def test_defaults_to_the_process_environment(monkeypatch) -> None:
    monkeypatch.setenv("GIT_WORK_TREE", "/elsewhere")
    monkeypatch.setenv("KOLEGA_GIT_ENV_MARKER", "kept")

    result = git_env()

    assert "GIT_WORK_TREE" not in result
    assert result["KOLEGA_GIT_ENV_MARKER"] == "kept"


def test_result_is_an_independent_copy(monkeypatch) -> None:
    monkeypatch.setenv("KOLEGA_GIT_ENV_MARKER", "original")
    result = git_env()
    result["KOLEGA_GIT_ENV_MARKER"] = "mutated"

    import os

    assert os.environ["KOLEGA_GIT_ENV_MARKER"] == "original"
