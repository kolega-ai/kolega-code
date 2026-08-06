"""Declarative param-alias registry for tool dispatch.

Models regularly call tools with wrong-but-reasonable argument names, usually
cross-contaminated from a sibling tool's signature (``exec_command(timeout=…)``
borrowed from ``eval``; ``eval(command=…)`` borrowed from ``exec_command``).
Each such call would be rejected and cost a recovery round-trip. This registry
maps the known aliases onto the canonical parameter at the dispatch choke point
(``Tool.call``), before argument validation, so the call succeeds first try.

Measured basis: findings/tool-arg-cross-contamination-friction.md (~100
rejections across ~820 benchmark trials; per-alias counts on each entry).

Resolution is signature-aware: an argument name is only treated as an alias
when the bound callable does NOT itself accept that name. ``Tool`` computes
the accepted parameter names once at construction (via ``inspect``), and
``resolve_param_aliases`` skips any registered alias the binding accepts — a
name the tool itself accepts is a real parameter there, never a misspelling.
This lets one entry serve a tool name bound to different callables per edit
protocol (``write``: ``path`` is canonical under search_replace and an alias
under claude_code) without ever rewriting a legitimate call.

Adding an alias requires exactly one entry here plus one case row in
tests/agent/test_param_aliases.py. Unregistered unknown parameters still
error exactly as before.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple

# Sentinel a transform may return when the value is unusable: the alias is then
# dropped instead of erroring, because an alias hit must never fail the call.
DROP = object()


@dataclass(frozen=True)
class ParamAlias:
    """One accepted wrong-name spelling of a tool parameter.

    ``canonical=None`` means accept-and-discard. ``transform`` (renames only)
    adapts the alias value; returning ``DROP`` discards the alias instead.
    """

    canonical: Optional[str] = None
    transform: Optional[Callable[[Any], Any]] = None


@dataclass(frozen=True)
class AliasHit:
    """A resolved alias occurrence, surfaced for debug logging / trajectory mining."""

    tool: str
    alias: str
    action: str


def _timeout_to_yield_ms(value: Any) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return DROP
    # Unit heuristic: models mix two conventions — seconds (OpenAI-style bash
    # tools: 5, 30, 120…) and milliseconds (Claude-Code-style: 5000, 120000…).
    # Sub-second shell timeouts don't occur in practice, so values < 1000 read
    # as seconds; anything larger is taken as milliseconds already. Seconds
    # values that would exceed 1000 land at the 30000 ms clamp under either
    # reading, so nothing meaningful is lost at the boundary.
    return int(number * 1000) if number < 1000 else int(number)


# tool name -> alias name -> action. Every entry is justified by observed model
# behavior, with the measured count at the time it was added.
PARAM_ALIASES: Dict[str, Dict[str, ParamAlias]] = {
    "exec_command": {
        # Seen 48×. Models mean "kill after N" (every other harness's bash
        # tool has a timeout), but exec_command has no kill deadline —
        # yield_time_ms means "yield back after N ms; the command keeps
        # running". Mapping onto the yield window is the closest honest
        # interpretation: the model regains control once its deadline passes
        # (the terminal backend clamps to 250–30000 ms) and can still poll or
        # kill the session. Units are guessed by magnitude (see the
        # transform); non-numeric values are dropped so the call succeeds
        # regardless.
        "timeout": ParamAlias("yield_time_ms", transform=_timeout_to_yield_ms),
        # Seen 9×, borrowed from eval's title. Purely cosmetic: discard.
        "title": ParamAlias(),
    },
    "eval": {
        # Seen 36× as eval "missing required argument: code" — exec-style
        # cross-over of exec_command's payload name.
        "command": ParamAlias("code"),
    },
    # ``read`` merged ``read_entire_file``/``read_file_section`` into one tool
    # with a canonical ``file_path`` parameter (like claude_edit/claude_write),
    # but every sibling file tool (edit, lsp, snapshot, …) spells it ``path``,
    # so models will cross-contaminate. Map-and-accept: ``file_path`` stays
    # canonical in the schema.
    "read": {
        "path": ParamAlias("file_path"),
    },
    # The tool NAME ``write`` binds to different callables per edit protocol:
    # under claude_code it is claude_write(file_path, content) — where the
    # misses happened — and under search_replace it is write(path, content),
    # where ``path`` is the CANONICAL parameter. Seen 15×: models borrowed
    # ``path`` from every sibling file tool (edit, lsp, snapshot, …); record:
    # findings/harness-tool-surface-census.md ("still open" item). The
    # signature guard above skips an alias on any binding that itself accepts
    # the name, so this entry is a rename under claude_code and a no-op under
    # search_replace.
    "write": {
        "path": ParamAlias("file_path"),
    },
    # ``search_codebase`` (path=…, max_results=…) and ``find_files_by_pattern``
    # (path→pattern) aliases were removed with the file-discovery tools
    # themselves — both were merged into ``glob`` and then deleted in favor of
    # terminal-based discovery.
}


def resolve_param_aliases(
    tool_name: str,
    inputs: Mapping[str, Any],
    *,
    accepted_params: FrozenSet[str],
) -> Tuple[Dict[str, Any], List[AliasHit]]:
    """Map registered wrong-name arguments for ``tool_name`` onto canonical ones.

    ``accepted_params`` is the set of parameter names the bound callable
    accepts (computed once per Tool at construction, via ``inspect``). An
    argument name is only treated as an alias when the binding does NOT itself
    accept that name: a registered alias the binding accepts is a real
    parameter there, so it passes through untouched — this is what lets a
    single entry serve a tool name bound to different callables per edit
    protocol without ever rewriting a legitimate call.

    Returns the resolved inputs plus one ``AliasHit`` per alias consumed. When
    both an alias and its canonical parameter are present, the canonical value
    wins and the alias is dropped. Unregistered names pass through untouched
    (and fail validation downstream exactly as before).
    """
    aliases = PARAM_ALIASES.get(tool_name)
    if not aliases or not aliases.keys() & inputs.keys():
        return dict(inputs), []
    resolved: Dict[str, Any] = {}
    hits: List[AliasHit] = []
    for name, value in inputs.items():
        alias = aliases.get(name)
        if alias is None:
            resolved[name] = value
            continue
        if name in accepted_params:
            # The binding itself accepts this name — it is a real parameter
            # here, not a wrong-name spelling. Pass through untouched.
            resolved[name] = value
            continue
        if alias.canonical is None:
            hits.append(AliasHit(tool_name, name, "discarded"))
            continue
        if alias.canonical in inputs:
            hits.append(AliasHit(tool_name, name, f"dropped (canonical `{alias.canonical}` also present)"))
            continue
        if alias.transform is not None:
            value = alias.transform(value)
            if value is DROP:
                hits.append(AliasHit(tool_name, name, f"discarded (value unusable as `{alias.canonical}`)"))
                continue
        resolved[alias.canonical] = value
        hits.append(AliasHit(tool_name, name, f"renamed to `{alias.canonical}`"))
    return resolved, hits
