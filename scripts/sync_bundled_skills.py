#!/usr/bin/env python3
"""Vendor an explicit kolega-skills Git tag into Kolega Code package data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path

from kolega_code.cli.skills import _iter_skill_files, _load_skill


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = REPOSITORY_ROOT / "kolega_code" / "_bundled_skills"
UPSTREAM_REPOSITORY = "https://github.com/kolega-ai/kolega-skills"
MANIFEST_NAME = "manifest.json"


class SyncError(Exception):
    """Raised when a tagged skill snapshot cannot be generated safely."""


def _git(repository: Path, *args: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=False,
        capture_output=True,
        text=text,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() if text else result.stderr.decode(errors="replace").strip()
        raise SyncError(f"git {' '.join(args)} failed: {stderr}")
    return result.stdout


def _tag_commit(repository: Path, tag: str) -> str:
    if not tag.strip() or tag != tag.strip():
        raise SyncError("tag must be a non-empty explicit Git tag")
    _git(repository, "rev-parse", "--verify", f"refs/tags/{tag}")
    commit = _git(repository, "rev-parse", f"refs/tags/{tag}^{{commit}}")
    assert isinstance(commit, str)
    return commit.strip()


def _extract_archive(repository: Path, commit: str, destination: Path) -> None:
    archive_bytes = _git(repository, "archive", "--format=tar", commit, "LICENSE", "skills", text=False)
    assert isinstance(archive_bytes, bytes)

    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        for member in archive.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise SyncError(f"unsafe path in Git archive: {member.name}")
            target = destination / relative

            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise SyncError(f"unsupported non-file entry in Git archive: {member.name}")

            source = archive.extractfile(member)
            if source is None:
                raise SyncError(f"could not read Git archive entry: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())
            mode = 0o755 if member.mode & stat.S_IXUSR else 0o644
            target.chmod(mode)


def _validate_skills(snapshot: Path) -> list[str]:
    skills_root = snapshot / "skills"
    skill_files = list(_iter_skill_files(skills_root))
    if not skill_files:
        raise SyncError("tagged snapshot contains no skills")

    names: list[str] = []
    errors: list[str] = []
    for skill_file in skill_files:
        record, diagnostics = _load_skill(skill_file, "bundled")
        errors.extend(diagnostic.format() for diagnostic in diagnostics if diagnostic.severity == "error")
        if record is not None:
            names.append(record.name)

    if errors:
        raise SyncError("tagged snapshot contains invalid skills:\n" + "\n".join(f"- {error}" for error in errors))
    if len(names) != len(set(names)):
        raise SyncError("tagged snapshot contains duplicate skill names")
    return sorted(names)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(snapshot: Path, *, tag: str, commit: str, skills: list[str]) -> None:
    files = [
        {
            "path": path.relative_to(snapshot).as_posix(),
            "sha256": _sha256(path),
        }
        for path in sorted(snapshot.rglob("*"))
        if path.is_file() and path.name != MANIFEST_NAME
    ]
    manifest = {
        "schema_version": 1,
        "source": {
            "repository": UPSTREAM_REPOSITORY,
            "tag": tag,
            "commit": commit,
        },
        "skills": skills,
        "files": files,
    }
    (snapshot / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _replace_snapshot(staged_snapshot: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.parent / f".{destination.name}.backup"
    if backup.exists():
        raise SyncError(f"stale bundle backup exists: {backup}")

    had_destination = destination.exists()
    if had_destination:
        destination.rename(backup)
    try:
        staged_snapshot.rename(destination)
    except Exception:
        if had_destination:
            backup.rename(destination)
        raise
    else:
        if had_destination:
            shutil.rmtree(backup)


def sync_bundled_skills(repository: Path, tag: str, destination: Path = DEFAULT_DESTINATION) -> tuple[str, list[str]]:
    """Generate and atomically install a bundled-skills snapshot."""
    repository = repository.resolve()
    destination = destination.resolve()
    if destination == destination.parent:
        raise SyncError("refusing to use a filesystem root as the destination")

    destination.parent.mkdir(parents=True, exist_ok=True)
    _git(repository, "rev-parse", "--is-inside-work-tree")
    commit = _tag_commit(repository, tag)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    staged_snapshot = temporary_root / destination.name
    staged_snapshot.mkdir()
    try:
        _extract_archive(repository, commit, staged_snapshot)
        skills = _validate_skills(staged_snapshot)
        _write_manifest(staged_snapshot, tag=tag, commit=commit, skills=skills)
        _replace_snapshot(staged_snapshot, destination)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    return commit, skills


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path, help="local kolega-skills Git repository")
    parser.add_argument("--tag", required=True, help="explicit kolega-skills tag to vendor")
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"bundle destination (default: {DEFAULT_DESTINATION})",
    )
    args = parser.parse_args()

    try:
        commit, skills = sync_bundled_skills(args.repository, args.tag, args.destination)
    except (OSError, SyncError) as exc:
        parser.exit(1, f"Bundled skill sync failed: {exc}\n")

    print(f"Bundled {len(skills)} skills from {args.tag} ({commit}): {', '.join(skills)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
