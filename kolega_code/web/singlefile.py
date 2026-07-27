"""Fold a replay bundle into one HTML file that opens with a double-click.

A directory of ten files is not something you can send someone. Browsers refuse
to load ES modules or run ``fetch`` from a ``file://`` origin, so a recipient who
double-clicks ``index.html`` gets a blank page — and not even an error, because
the script that would report the failure is the one that was blocked. Serving the
directory over HTTP works, but "install a web server" is not a sharing story.

So the export embeds everything: stylesheets, both scripts, the manifest, the
event log, and any images. The result is a single document that needs no server,
no unpacking and no network, and is small enough to email — the event log is the
bulk of it and gzips about eight-fold before base64.

The player is not duplicated for this. It reads ``globalThis.__KC_REPLAY__`` when
present and fetches when it is absent, so the served player, the directory bundle
and this file are all the same code.
"""

from __future__ import annotations

import base64
import gzip
import json
import re
from pathlib import Path
from typing import Mapping, Optional

_ASSET_DIR = Path(__file__).parent / "assets"

#: Exact fragments of ``player.html`` this module rewrites. They are asserted
#: rather than pattern-matched: if someone edits the template, the export must
#: fail loudly in tests instead of quietly producing a blank page.
_THEME_LINK = '<link rel="stylesheet" href="theme.css" />'
_PLAYER_LINK = '<link rel="stylesheet" href="player.css" />'
_MODULE_TAG = '<script type="module" src="player.js"></script>'

_EXPORT_KEYWORD = re.compile(r"^export\s+", re.MULTILINE)
_FOLD_IMPORT = 'import { emptyState, fold } from "./fold.js";'


class SingleFileError(RuntimeError):
    """Raised when the player assets no longer match what the inliner expects."""


def _read_asset(name: str) -> str:
    path = _ASSET_DIR / name
    if not path.exists():  # pragma: no cover - packaging accident
        raise SingleFileError(f"player asset {name} is missing from {_ASSET_DIR}")
    return path.read_text(encoding="utf-8")


def _replace_once(document: str, marker: str, replacement: str, *, what: str) -> str:
    count = document.count(marker)
    if count != 1:
        raise SingleFileError(f"expected exactly one {what} in player.html, found {count}. Update singlefile.py.")
    return document.replace(marker, replacement)


def _inline_script_payload(payload: object) -> str:
    """Serialize ``payload`` so it can sit safely inside a classic script tag.

    ``ensure_ascii`` escapes U+2028/U+2029, which are newlines to a JavaScript
    parser but not to JSON, and escaping ``<`` means a string in the data can
    never close the tag early with ``</script>``.
    """
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")


def _module_source() -> str:
    """fold.js and player.js concatenated into one module.

    fold.js comes first because player.js calls ``main()`` at the top level and
    fold's module constants must already be initialised when it does.
    """
    fold = _read_asset("fold.js")
    player = _read_asset("player.js")

    stripped, replaced = _EXPORT_KEYWORD.subn("", fold)
    if not replaced:
        raise SingleFileError("fold.js no longer uses `export`; the inliner needs updating.")
    if _FOLD_IMPORT not in player:
        raise SingleFileError("player.js no longer imports from fold.js; the inliner needs updating.")
    player = player.replace(_FOLD_IMPORT, "// fold.js is inlined above.")

    for name, source in (("fold.js", stripped), ("player.js", player)):
        if "</script" in source:
            raise SingleFileError(f"{name} contains '</script', which cannot be inlined safely.")
    return f"{stripped}\n{player}"


def build_single_file(
    *,
    manifest: Mapping[str, object],
    events_jsonl: bytes,
    theme_css: str,
    artifacts: Optional[Mapping[str, tuple[str, bytes]]] = None,
) -> str:
    """Return a complete HTML document holding an entire replay.

    ``artifacts`` maps a sha256 digest to ``(media_type, data)``. Only images are
    worth embedding: the player renders them, whereas a text artifact is already
    described by the preview text its event carries.
    """
    document = _read_asset("player.html")

    inlined_artifacts = {
        digest: f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"
        for digest, (media_type, data) in (artifacts or {}).items()
    }
    payload = {
        "manifest": dict(manifest),
        # mtime=0 keeps the output byte-identical across runs of the same session.
        "events": base64.b64encode(gzip.compress(events_jsonl, compresslevel=9, mtime=0)).decode("ascii"),
        "artifacts": inlined_artifacts,
    }

    document = _replace_once(
        document,
        _THEME_LINK,
        f"<style>\n{theme_css}\n</style>",
        what="theme.css link",
    )
    document = _replace_once(
        document,
        _PLAYER_LINK,
        f"<style>\n{_read_asset('player.css')}\n</style>",
        what="player.css link",
    )
    scripts = (
        f"<script>globalThis.__KC_REPLAY__ = {_inline_script_payload(payload)};</script>\n"
        f'    <script type="module">\n{_module_source()}\n</script>'
    )
    return _replace_once(document, _MODULE_TAG, scripts, what="player.js script tag")
