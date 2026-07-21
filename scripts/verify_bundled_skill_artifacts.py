#!/usr/bin/env python3
"""Verify bundled skill files and hashes in built wheel and sdist archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Callable


MANIFEST_SUFFIX = "kolega_code/_bundled_skills/manifest.json"


class VerificationError(Exception):
    """Raised when a distribution is missing or corrupts bundled skill data."""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _verify_members(archive: Path, names: list[str], read: Callable[[str], bytes]) -> None:
    manifest_names = [name for name in names if name.endswith(MANIFEST_SUFFIX)]
    if len(manifest_names) != 1:
        raise VerificationError(f"{archive}: expected one bundled skill manifest, found {len(manifest_names)}")

    manifest_name = manifest_names[0]
    prefix = manifest_name[: -len("manifest.json")]
    manifest = json.loads(read(manifest_name))
    for item in manifest["files"]:
        member_name = f"{prefix}{item['path']}"
        if member_name not in names:
            raise VerificationError(f"{archive}: missing bundled file {member_name}")
        actual_hash = _sha256(read(member_name))
        if actual_hash != item["sha256"]:
            raise VerificationError(f"{archive}: hash mismatch for bundled file {member_name}")


def verify_archive(archive: Path) -> None:
    """Verify one wheel or source distribution."""
    if archive.suffix == ".whl":
        with zipfile.ZipFile(archive) as package:
            names = package.namelist()
            _verify_members(archive, names, package.read)
        return

    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, mode="r:gz") as package:
            members = {member.name: member for member in package.getmembers() if member.isfile()}

            def read(name: str) -> bytes:
                source = package.extractfile(members[name])
                if source is None:
                    raise VerificationError(f"{archive}: could not read {name}")
                return source.read()

            _verify_members(archive, list(members), read)
        return

    raise VerificationError(f"unsupported distribution format: {archive}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path, help="wheel and sdist archives to verify")
    args = parser.parse_args()

    try:
        for archive in args.archives:
            verify_archive(archive)
            print(f"Verified bundled skills in {archive}")
    except (KeyError, OSError, VerificationError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Bundled skill artifact verification failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
