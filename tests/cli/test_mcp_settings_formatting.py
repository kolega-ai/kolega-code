import pytest


def _rows(*rows):
    return list(rows)


def test_mcp_status_rendering_all_verified_has_single_aggregate_tick() -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui import settings_panel

    rows = _rows(
        {
            "id": "context7",
            "name": "Context7",
            "source": "global",
            "transport": "streamable_http",
            "enabled": True,
            "oauth": False,
            "status": "verified",
            "tool_count": 2,
            "message": "OK",
        },
        {
            "id": "repo-tools",
            "name": "Repo Tools",
            "source": "project",
            "transport": "stdio",
            "enabled": True,
            "oauth": False,
            "status": "verified",
            "tool_count": 1,
            "message": "OK",
        },
    )

    content, tone = settings_panel._render_mcp_status_text([], rows)
    fake_panel = _FakeMCPStatusPanel()
    settings_panel.SettingsPanelMixin._set_mcp_status(fake_panel, content, tone=tone)

    plain = fake_panel.updated.plain
    check = settings_panel.theme.g(settings_panel.Glyph.CHECK)
    sep = f" {settings_panel.theme.g(settings_panel.Glyph.BULLET_SEP)} "

    assert tone == "ok"
    assert plain.startswith(f"{check} 2 MCP servers configured{sep}all enabled verified")
    assert plain.count(check) == 1
    assert "\n  ✓" not in plain
    assert "\n  !" not in plain
    assert "\n  •" not in plain
    assert "Context7 (context7)" not in plain
    assert "Repo Tools (repo-tools)" not in plain
    assert f"verified{sep}2 tools{sep}global{sep}HTTP" in plain
    assert f"verified{sep}1 tool{sep}project{sep}stdio" in plain


def test_mcp_status_rendering_mixed_attention_states_use_text_labels() -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui import settings_panel

    rows = _rows(
        {
            "id": "docs",
            "name": "Docs",
            "source": "global",
            "transport": "streamable_http",
            "enabled": True,
            "oauth": True,
            "status": "failed",
            "tool_count": 0,
            "message": "Unauthorized. Please check your API key.",
        },
        {
            "id": "repo-tools",
            "name": "Repo Tools",
            "source": "project",
            "transport": "stdio",
            "enabled": True,
            "oauth": False,
            "status": "stale",
            "tool_count": 0,
            "message": "Previously verified.",
        },
    )

    content, tone = settings_panel._render_mcp_status_text(["Project MCP config is untrusted."], rows)
    plain = content.plain
    sep = f" {settings_panel.theme.g(settings_panel.Glyph.BULLET_SEP)} "

    assert tone == "warning"
    assert f"2 MCP servers configured{sep}2 need verification" in plain
    assert "Project MCP config is untrusted." in plain
    assert f"verify failed{sep}global{sep}HTTP{sep}oauth — Unauthorized. Please check your API key." in plain
    assert f"needs re-verify{sep}project{sep}stdio — Config changed; verify again." in plain
    assert "\n  !" not in plain
    assert "\n  ✓" not in plain


def test_mcp_status_rendering_disabled_servers_do_not_force_warning() -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui import settings_panel

    rows = _rows(
        {
            "id": "disabled-docs",
            "name": "Disabled Docs",
            "source": "global",
            "transport": "streamable_http",
            "enabled": False,
            "oauth": False,
            "status": "failed",
            "tool_count": 0,
            "message": "Connection refused.",
        }
    )

    content, tone = settings_panel._render_mcp_status_text([], rows)
    plain = content.plain
    sep = f" {settings_panel.theme.g(settings_panel.Glyph.BULLET_SEP)} "

    assert tone == "info"
    assert f"1 MCP server configured{sep}all disabled" in plain
    assert f"disabled{sep}global{sep}HTTP" in plain
    assert "verify failed" not in plain
    assert "Connection refused" not in plain
    assert "tools" not in plain


def test_mcp_status_rendering_no_servers_has_add_and_verify_hint() -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui import settings_panel

    content, tone = settings_panel._render_mcp_status_text([], [])

    assert tone == "info"
    assert content.plain == "No MCP servers configured. Add one below, then Verify."


def test_mcp_server_selector_labels_are_readable() -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui import settings_panel
    from kolega_code.mcp.config import MCPServerConfig

    http_server = MCPServerConfig(
        id="context7",
        name="Context7",
        transport="streamable_http",
        url="https://mcp.context7.com/mcp",
        enabled=True,
        source="global",
    )
    stdio_server = MCPServerConfig(
        id="repo-tools",
        name="Repo Tools",
        transport="stdio",
        command="npx",
        args=["-y", "@vendor/mcp-server"],
        enabled=False,
        source="project",
    )

    sep = settings_panel._mcp_separator()

    assert settings_panel._mcp_server_select_label(http_server) == f"Context7 — global{sep}HTTP{sep}enabled"
    assert settings_panel._mcp_server_select_label(stdio_server) == f"Repo Tools — project{sep}stdio{sep}disabled"
    assert (
        settings_panel.SettingsPanelMixin._mcp_server_option_label(object(), http_server)
        == f"Context7 — global{sep}HTTP{sep}enabled"
    )


class _FakeMCPStatusPanel:
    def __init__(self) -> None:
        self.updated = None

    def query_one(self, *_args, **_kwargs):
        return self

    def update(self, content) -> None:
        self.updated = content
