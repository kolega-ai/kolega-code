#!/usr/bin/env python3
"""Regenerate the Perplexity Agent API model catalog from its ``/models`` endpoint.

Dev-time release chore, never imported by the CLI. Run it from the repo root:

    uv run python scripts/refresh_perplexity_catalog.py

Rewrites ``kolega_code/llm/specs/catalog/perplexity_agent.py``. (Perplexity's
inference Gateway is preview-gated and is not shipped as a provider.)

The endpoint requires a valid ``PERPLEXITY_API_KEY`` (the docs' "no
authentication required" claim for the Agent API's ``/models`` does not hold —
probed 2026-08-14, 401 with and without the header). The payload carries only
``id``/``owned_by``; specs are uniform conservative values plus the probed
Agent API effort vocabulary (see ``kolega_code/llm/specs/perplexity_catalog.py``)
— nothing is derived from native-provider catalogs.

Because a working key was not available when this generator was written, the
bundled snapshots were bootstrapped from the official docs model lists with
``--from-docs-ids``; regenerate from the live endpoints as soon as a valid key
exists so the files carry the authoritative membership.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GENERATOR_REL_PATH = "scripts/refresh_perplexity_catalog.py"

AGENT_OUTPUT = REPO_ROOT / "kolega_code" / "llm" / "specs" / "catalog" / "perplexity_agent.py"

# Seed importing the shared transform (which reads the catalog package) when a
# generated file has been deleted: a placeholder keeps the import alive.
for output, variable in ((AGENT_OUTPUT, "PERPLEXITY_AGENT_SPECS"),):
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            f"# Placeholder written by {GENERATOR_REL_PATH}; rerun it to populate.\n{variable}: dict = {{}}\n",
            encoding="utf-8",
        )

from kolega_code.llm.specs import perplexity_catalog as catalog  # noqa: E402

# Docs-verified membership (docs.perplexity.ai, snapshot 2026-08-14) used to
# bootstrap the bundled catalogs while no valid API key was available.
DOCS_AGENT_IDS = (
    "anthropic/claude-opus-5",
    "anthropic/claude-opus-4-8",
    "anthropic/claude-opus-4-7",
    "anthropic/claude-opus-4-6",
    "anthropic/claude-opus-4-5",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-fable-5",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-sonnet-4-5",
    "anthropic/claude-haiku-4-5",
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-terra",
    "openai/gpt-5.6-luna",
    "openai/gpt-5.5",
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
    "openai/gpt-5.2",
    "openai/gpt-5.1",
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.1-flash-lite",
    "google/gemini-3.5-flash",
    "google/gemini-3.5-flash-lite",
    "google/gemini-3.6-flash",
    "google/gemini-3-flash-preview",
    "xai/grok-4.6",
    "xai/grok-4.5",
    "xai/grok-4.3",
    "xai/grok-4.20-reasoning",
    "xai/grok-4.20-non-reasoning",
    "xai/grok-4.20-multi-agent",
    "perplexity/deepseek-v4-flash-0731",
    "perplexity/glm-5.2",
    "perplexity/kimi-k3",
    "perplexity/kimi-k2.7-code",
    "perplexity/nemotron-3.5-lightning-30b-a3b",
    "perplexity/nemotron-3-ultra-550b-a55b",
    "perplexity/sonar",
)


def _payload_from_ids(identifiers: Sequence[str]) -> dict[str, Any]:
    return {"data": [{"id": identifier, "object": "model", "created": 0, "owned_by": ""} for identifier in identifiers]}


def _load_payload(
    source: catalog.PerplexityCatalogSource,
    payload: Optional[Any],
    docs_ids: Optional[Sequence[str]],
) -> Any:
    if payload is not None:
        return payload
    if docs_ids is not None:
        return _payload_from_ids(docs_ids)
    return source.fetch_models()


def _load_existing_entries(path: Path, provider: str, variable: str) -> list[tuple[str, dict[str, Any]]]:
    if not path.exists():
        return []
    namespace = runpy.run_path(str(path))
    specs = namespace.get(variable)
    if not isinstance(specs, Mapping):
        raise catalog.PerplexityCatalogError(f"{path} does not define {variable}.")
    entries: list[tuple[str, dict[str, Any]]] = []
    for key, spec in specs.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or key[0] != provider
            or not isinstance(key[1], str)
            or not isinstance(spec, Mapping)
        ):
            raise catalog.PerplexityCatalogError(f"{path} contains an invalid catalog entry.")
        entries.append((key[1], dict(spec)))
    return entries


def _diff_entries(
    previous: Sequence[tuple[str, Mapping[str, Any]]],
    current: Sequence[tuple[str, Mapping[str, Any]]],
) -> tuple[list[str], list[str], list[str]]:
    previous_by_id = {identifier: dict(spec) for identifier, spec in previous}
    current_by_id = {identifier: dict(spec) for identifier, spec in current}
    added = sorted(current_by_id.keys() - previous_by_id.keys())
    removed = sorted(previous_by_id.keys() - current_by_id.keys())
    changed = sorted(
        identifier
        for identifier in previous_by_id.keys() & current_by_id.keys()
        if previous_by_id[identifier] != current_by_id[identifier]
    )
    return added, removed, changed


def _print_ids(label: str, identifiers: Sequence[str]) -> None:
    print(f"{label}: {', '.join(identifiers) if identifiers else '(none)'}")


def _print_report(
    *,
    label: str,
    entries: Sequence[tuple[str, Mapping[str, Any]]],
    added: Sequence[str],
    removed: Sequence[str],
    changed: Sequence[str],
) -> None:
    print(f"\n{label}: {len(entries)} models")
    _print_ids("Added", added)
    _print_ids("Removed/retired", removed)
    _print_ids("Spec changed", changed)


def _render_value(value: Any) -> str:
    if isinstance(value, catalog.ThinkingEffortSpec):
        options = ", ".join(repr(option) for option in value.options)
        trailing = "," if len(value.options) == 1 else ""
        return (
            "ThinkingEffortSpec(\n"
            f"            options=({options}{trailing}),\n"
            f"            default={value.default!r},\n"
            f"            mode={value.mode!r},\n"
            "        )"
        )
    return repr(value)


def _render_catalog(
    *,
    provider: str,
    variable: str,
    entries: Sequence[tuple[str, Mapping[str, Any]]],
    header_lines: Sequence[str],
) -> str:
    lines = [f"# {line}" if line else "#" for line in header_lines]
    lines.append("")
    if any("thinking_effort" in spec for _, spec in entries):
        lines.append("from kolega_code.llm.specs.types import ThinkingEffortSpec")
        lines.append("")
    lines.append(f"{variable} = {{")
    for identifier, spec in entries:
        lines.append(f"    ({provider!r}, {identifier!r}): {{")
        for key, value in spec.items():
            lines.append(f"        {key!r}: {_render_value(value)},")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _header_lines(*, entry_count: int, source_url: str, snapshot_date: str, source_note: str) -> list[str]:
    return [
        f"Generated by {GENERATOR_REL_PATH} — do not edit by hand.",
        "",
        f"Membership: {source_url} ({source_note} {snapshot_date})",
        "",
        f"{entry_count} models. Specs are uniform conservative values plus the probed",
        "Agent API effort vocabulary; nothing is derived from native-provider catalogs",
        "(see specs/perplexity_catalog.py).",
        "",
        f"Regenerate with: uv run python {GENERATOR_REL_PATH}",
    ]


def _write_generated(output: Path, source: str, *, skip_format: bool) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    compile(source, str(output), "exec")

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.",
        suffix=".py",
        dir=output.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as temporary_file:
            temporary_file.write(source)

        if not skip_format:
            result = subprocess.run(
                ["uv", "run", "ruff", "format", str(temporary_path)],
                cwd=REPO_ROOT,
                check=False,
            )
            if result.returncode != 0:
                print("ERROR: ruff format failed; the existing catalog was not changed.", file=sys.stderr)
                return False

        formatted_source = temporary_path.read_text(encoding="utf-8")
        compile(formatted_source, str(output), "exec")
        temporary_path.chmod(0o644)
        os.replace(temporary_path, output)
        return True
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _refresh_one(
    *,
    label: str,
    source: catalog.PerplexityCatalogSource,
    output: Path,
    variable: str,
    payload: Optional[Any],
    docs_ids: Optional[Sequence[str]],
    snapshot_date: str,
    skip_format: bool,
) -> bool:
    payload = _load_payload(source, payload, docs_ids)
    entries = source.catalog_entries(payload)
    previous = _load_existing_entries(output, source.PROVIDER, variable)
    added, removed, changed = _diff_entries(previous, entries)
    _print_report(label=label, entries=entries, added=added, removed=removed, changed=changed)

    source_note = "docs model list, bootstrapped" if docs_ids is not None else "snapshot"
    header = _header_lines(
        entry_count=len(entries), source_url=source.MODELS_URL, snapshot_date=snapshot_date, source_note=source_note
    )
    rendered = _render_catalog(provider=source.PROVIDER, variable=variable, entries=entries, header_lines=header)
    return _write_generated(output, rendered, skip_format=skip_format)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--snapshot-file",
        help="Read the payload from a JSON file: {'perplexity_agent': {...}} or the /models response directly",
    )
    parser.add_argument(
        "--from-docs-ids",
        action="store_true",
        help=(
            "Bootstrap from the docs-verified model list embedded in this script "
            "(used when no valid API key is available); live fetch is authoritative."
        ),
    )
    parser.add_argument("--snapshot-date", help="Override the snapshot date stamped into the headers (YYYY-MM-DD)")
    parser.add_argument("--skip-format", action="store_true", help="Do not run ruff format on the output")
    args = parser.parse_args(argv)

    snapshot_date = args.snapshot_date or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

    payload: Optional[Any] = None
    if args.snapshot_file:
        raw = json.loads(Path(args.snapshot_file).read_text(encoding="utf-8"))
        payload = raw.get("perplexity_agent") if isinstance(raw, dict) and "perplexity_agent" in raw else raw

    try:
        ok = _refresh_one(
            label="Perplexity Agent API",
            source=catalog.AGENT_CATALOG,
            output=AGENT_OUTPUT,
            variable="PERPLEXITY_AGENT_SPECS",
            payload=payload,
            docs_ids=DOCS_AGENT_IDS if args.from_docs_ids else None,
            snapshot_date=snapshot_date,
            skip_format=args.skip_format,
        )
    except catalog.PerplexityCatalogError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not ok:
        return 1
    if args.from_docs_ids:
        print(
            "\nNOTE: bootstrapped from the embedded docs lists; regenerate from the live "
            "endpoints with a valid PERPLEXITY_API_KEY when available."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
