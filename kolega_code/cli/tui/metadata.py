"""Literal, width-aware session context with permission risk kept visible."""

from __future__ import annotations

from pathlib import Path
from unicodedata import category

from rich.cells import cell_len, split_graphemes
from rich.text import Text
from textual import events
from textual.widgets import Static

from ..theme import Color
from .startup import display_project_path


def _single_line(value: str) -> str:
    return "".join(" " if category(char) == "Cc" or char in "\u2028\u2029" else char for char in value)


def _prefix(value: str, width: int) -> str:
    """Take whole graphemes only, without padding or a zero-width ellipsis."""
    if width <= 0:
        return ""
    spans, _ = split_graphemes(value)
    end = 0
    used = 0
    for _, stop, size in spans:
        if used + size > width:
            break
        end = stop
        used += size
    return value[:end]


def _project_for_width(project: str, width: int) -> str:
    if width <= 0:
        return ""
    if cell_len(project) <= width:
        return project
    # Discard leading directories, not the repository name.
    parts = project.split("/")
    for start in range(1, len(parts)):
        shortened = "…/" + "/".join(parts[start:])
        if cell_len(shortened) <= width:
            return shortened
    basename = parts[-1] or project
    if cell_len(basename) <= width:
        return basename
    return _prefix(basename, width - 1) + "…"


def _mode_style(mode: str) -> str:
    """Semantic color for the interaction-mode indicator."""
    return Color.SUCCESS if mode == "plan" else Color.ACCENT


class MetadataStrip(Static):
    """Session metadata supplied by the app; never performs repository lookups.

    Call ``update_context`` on state changes. ``content_for_width(None)`` gives
    the unabridged *display* line as Rich Text (use ``.plain`` for string callers).
    ``full_context`` and the literal tooltip retain full paths and session IDs,
    including fields omitted from the display. Rendering uses current theme roles.
    """

    DEFAULT_CSS = """
    MetadataStrip {
        height: 1;
        overflow: hidden hidden;
    }
    """

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(markup=False, id=id, classes=classes)
        self.project_path: Path = Path(".")
        self.branch: str = ""
        self.session_id: str = ""
        self.interaction_mode: str = "build"
        self.permission_mode: str = "ask"
        self.gigacode_enabled: bool = False
        self._metadata_context_snapshot: tuple[object, ...] | None = None

    def update_context(
        self,
        *,
        project_path: Path,
        branch: str,
        session_id: str,
        interaction_mode: str,
        permission_mode: str,
        gigacode_enabled: bool = False,
    ) -> None:
        """Replace cached context, without filesystem or subprocess discovery."""
        context = (project_path, branch, session_id, interaction_mode, permission_mode, gigacode_enabled)
        if context == self._metadata_context_snapshot:
            return
        self._metadata_context_snapshot = context
        self.project_path = project_path
        self.branch = branch
        self.session_id = session_id
        self.interaction_mode = interaction_mode
        self.permission_mode = permission_mode
        self.gigacode_enabled = gigacode_enabled
        self.tooltip = Text(self.full_context)
        self.refresh()

    @property
    def full_context(self) -> str:
        """Plain accessibility/context text, independent of available columns."""
        fields = [
            f"Project: {self.project_path}",
            f"Branch: {self.branch or '(none)'}",
            f"Session: {self.session_id or '(none)'}",
            f"Mode: {self.interaction_mode}",
            f"Permissions: {self.permission_mode}",
        ]
        if self.gigacode_enabled:
            fields.append("Gigacode: enabled")
        return "\n".join(fields)

    def content_for_width(self, width: int | None = None) -> Text:
        """Return a literal single line fitting ``width`` terminal cells.

        Session is omitted before branch. Mode and labeled permissions outrank
        every optional field; tiny widths degrade to permission state alone.
        ``None`` means unrestricted, and nonpositive widths return empty Text.
        """
        result = Text(no_wrap=True, overflow="crop")
        if width is not None and width <= 0:
            return result
        project = _single_line(display_project_path(self.project_path))
        branch = _single_line(self.branch)
        session = _prefix(_single_line(self.session_id), 8)
        mode = _single_line(self.interaction_mode)
        mode_style = _mode_style(self.interaction_mode)
        permission = _single_line(self.permission_mode)
        permission_style = Color.WARNING if self.permission_mode == "auto" else Color.SUCCESS
        separator = " · "

        def join(fields: list[tuple[str, str]]) -> Text:
            text = Text(no_wrap=True, overflow="crop")
            for value, style in fields:
                if not value:
                    continue
                if text.plain:
                    text.append(separator, Color.MUTED)
                text.append(value, style)
            return text

        protected = [(mode, mode_style), (f"permissions {permission}", permission_style)]
        if width is not None and join(protected).cell_len > width:
            for label in ("perm ", ""):
                protected = [(mode, mode_style), (label + permission, permission_style)]
                if join(protected).cell_len <= width:
                    break
            else:
                # At very small widths spend cells on words, not decoration.
                compact = Text(mode, mode_style)
                compact.append(" ")
                compact.append(permission, permission_style)
                if compact.cell_len <= width:
                    compact.no_wrap = True
                    return compact
                return Text(_prefix(permission, width), permission_style, no_wrap=True, overflow="crop")

        if width is not None:
            project_budget = width - join(protected).cell_len - cell_len(separator)
            minimum_project = _project_for_width(
                project, min(max(0, project_budget), cell_len("…/" + project.split("/")[-1]), cell_len(project))
            )
            # Preserve at least a recognizable project before optional context.
            for omit in ("session", "branch"):
                candidates = [
                    (minimum_project, "bold"),
                    (branch, Color.MUTED),
                    *protected,
                    (session, Color.MUTED),
                ]
                if join(candidates).cell_len <= width:
                    break
                if omit == "session":
                    session = ""
                else:
                    branch = ""
            other = join([(branch, Color.MUTED), *protected, (session, Color.MUTED)])
            project = _project_for_width(project, width - other.cell_len - cell_len(separator))

        result = join([(project, "bold"), (branch, Color.MUTED), *protected, (session, Color.MUTED)])
        if self.gigacode_enabled:
            extra = join([("gigacode", Color.ACCENT)])
            if width is None or result.cell_len + cell_len(separator) + extra.cell_len <= width:
                result.append(separator, Color.MUTED)
                result.append_text(extra)
        return result

    def render(self) -> Text:
        return self.content_for_width(self.content_size.width)

    def on_resize(self, event: events.Resize) -> None:
        self.refresh()
