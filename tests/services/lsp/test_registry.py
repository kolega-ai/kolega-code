"""Unit tests for LSP registry loading, extension lookup, server resolution, and user config merging."""

from kolega_code.services.lsp import LspConfig, LspRegistry


# ---------------------------------------------------------------------------
# registry loading
# ---------------------------------------------------------------------------


def test_registry_loads_presets():
    """Bundled presets are loaded and indexed."""
    registry = LspRegistry()
    langs = registry.languages
    assert len(langs) >= 20
    assert "python" in langs
    assert "javascript" in langs
    assert "typescript" in langs
    assert "rust" in langs


def test_registry_extension_lookup():
    """Extensions resolve to correct language IDs."""
    registry = LspRegistry()
    assert registry.language_for_extension(".py") == "python"
    assert registry.language_for_extension(".PY") == "python"  # case-insensitive
    assert registry.language_for_extension(".rs") == "rust"
    assert registry.language_for_extension(".ts") == "typescript"
    assert registry.language_for_extension(".tsx") == "typescript"
    assert registry.language_for_extension(".go") == "go"


def test_registry_filename_lookup():
    """Exact filenames resolve correctly."""
    registry = LspRegistry()
    assert registry.language_for_filename("Dockerfile") == "docker"


def test_registry_unknown_extension_returns_none():
    """Unknown extensions return None."""
    registry = LspRegistry()
    assert registry.language_for_extension(".xyzzy") is None


# ---------------------------------------------------------------------------
# server resolution
# ---------------------------------------------------------------------------


def test_resolve_server_finds_available_binary():
    """Python language server resolves when pyright is on PATH."""
    registry = LspRegistry()
    bin_path, spec, candidates = registry.resolve_server("python")
    # pyright may or may not be installed; the function should not crash
    assert isinstance(candidates, list)
    assert len(candidates) > 0
    # If pyright is installed, it should be found
    if bin_path:
        assert spec is not None
        assert "pyright" in spec.name or "basedpyright" in spec.name
        assert "pyright-langserver" in bin_path or "basedpyright" in bin_path


def test_resolve_server_unknown_language():
    """Unknown language returns empty results."""
    registry = LspRegistry()
    bin_path, spec, candidates = registry.resolve_server("nonexistent")
    assert bin_path is None
    assert spec is None
    assert candidates == []


# ---------------------------------------------------------------------------
# user config merging
# ---------------------------------------------------------------------------


def test_user_preferences_reorder_servers():
    """User preference moves preferred server to front of candidates list."""
    config = LspConfig(
        preferences={"python": "basedpyright"},
    )
    registry = LspRegistry(config=config)
    _, _, candidates = registry.resolve_server("python")
    # basedpyright should be first
    assert candidates[0].name == "basedpyright"


def test_user_config_disabled_language():
    """Disabled language is pruned from the registry."""
    config = LspConfig(disabled_languages=["python"])
    registry = LspRegistry(config=config)
    assert "python" not in registry.languages
    assert registry.language_for_extension(".py") is None


def test_lsp_config_defaults():
    """LspConfig has sensible defaults."""
    cfg = LspConfig()
    assert cfg.enabled is True
    assert cfg.auto_diagnostics_on_edit is True
    assert cfg.auto_fallback is True
    assert cfg.prompt_on_missing is True
    assert cfg.max_diagnostics == 20
    assert cfg.disabled_languages == []
    assert cfg.preferences == {}
    assert cfg.custom_servers == {}
