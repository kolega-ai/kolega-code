from __future__ import annotations

import pytest

from kolega_code.cli.config import CliConfigError, CliConfigOverrides, _skills_enabled, build_agent_config
from kolega_code.cli.settings import CliSettings


def test_skills_resolution_precedence() -> None:
    settings = CliSettings(skills_enabled=True)
    env = {"KOLEGA_CODE_SKILLS": "off"}

    assert _skills_enabled(env, settings, "on") is True
    assert _skills_enabled(env, settings, None) is False
    assert _skills_enabled({}, settings, None) is True
    assert _skills_enabled({}, CliSettings(skills_enabled=False), None) is False
    assert _skills_enabled({}, None, None) is True


def test_skills_resolution_rejects_unknown_value() -> None:
    with pytest.raises(CliConfigError, match="Unsupported skills mode"):
        _skills_enabled({"KOLEGA_CODE_SKILLS": "sideways"}, None, None)


def test_build_agent_config_carries_resolved_skills_switch(tmp_path) -> None:
    settings = CliSettings(
        active_provider="anthropic",
        active_model="claude-opus-5",
        skills_enabled=False,
    )
    config = build_agent_config(tmp_path, settings=settings, env={"ANTHROPIC_API_KEY": "test-key"})
    assert config.skills_enabled is False

    forced = build_agent_config(
        tmp_path,
        CliConfigOverrides(skills_mode="on"),
        settings=settings,
        env={"ANTHROPIC_API_KEY": "test-key", "KOLEGA_CODE_SKILLS": "off"},
    )
    assert forced.skills_enabled is True
