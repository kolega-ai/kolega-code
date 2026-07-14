"""Production edit-protocol registry.

The registry separates model-facing tool names from their internal handlers so
multiple protocols can expose the same name with different schemas.  In
particular, both the legacy search/replace surface and the Claude-style surface
expose lowercase ``edit`` and ``write`` tools.
"""

from __future__ import annotations

from dataclasses import dataclass

from kolega_code.config import EditProtocol


@dataclass(frozen=True)
class EditToolBinding:
    """One model-facing edit tool and the ToolCollection method behind it."""

    name: str
    handler_name: str


@dataclass(frozen=True)
class EditProtocolSpec:
    """The complete edit surface exposed for one protocol."""

    protocol: EditProtocol
    tools: tuple[EditToolBinding, ...]

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self.tools)


EDIT_PROTOCOL_SPECS: dict[EditProtocol, EditProtocolSpec] = {
    EditProtocol.SEARCH_REPLACE: EditProtocolSpec(
        protocol=EditProtocol.SEARCH_REPLACE,
        tools=(
            EditToolBinding("edit", "edit"),
            EditToolBinding("multi_edit", "multi_edit"),
            EditToolBinding("write", "write"),
        ),
    ),
    EditProtocol.CODEX_APPLY_PATCH: EditProtocolSpec(
        protocol=EditProtocol.CODEX_APPLY_PATCH,
        tools=(EditToolBinding("apply_patch", "apply_patch"),),
    ),
    EditProtocol.CLAUDE_CODE: EditProtocolSpec(
        protocol=EditProtocol.CLAUDE_CODE,
        tools=(
            EditToolBinding("edit", "claude_edit"),
            EditToolBinding("write", "claude_write"),
        ),
    ),
    EditProtocol.HASHLINE_V2: EditProtocolSpec(
        protocol=EditProtocol.HASHLINE_V2,
        tools=(
            EditToolBinding("edit", "hashline_edit"),
            EditToolBinding("write", "hashline_write"),
        ),
    ),
}

EDIT_HANDLER_NAMES = frozenset(binding.handler_name for spec in EDIT_PROTOCOL_SPECS.values() for binding in spec.tools)


def edit_protocol_spec(protocol: EditProtocol) -> EditProtocolSpec:
    """Return the registered production surface for ``protocol``."""

    return EDIT_PROTOCOL_SPECS[protocol]


def production_edit_protocols() -> tuple[EditProtocol, ...]:
    """Return production protocols in their stable display/benchmark order."""

    return tuple(EDIT_PROTOCOL_SPECS)
