"""Strict access-setting validation without settings/config dependencies."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


class AccessSettingsError(ValueError):
    """An access setting is malformed; its restrictions must not be discarded."""


def validate_access_settings(raw: object, *, adapter: str | None = None) -> dict[str, Any]:
    """Return a copy with normalized access fields, preserving other settings.

    The caller supplies ``{}`` for a missing section; an explicit null is invalid.
    Telegram identities are decimal strings, never usernames or coerced numbers.
    Other adapters retain their own string identities (for example echo's owner).
    """
    if not isinstance(raw, Mapping):
        raise AccessSettingsError("gateway: expected an object in settings.json; use {} for safe defaults.")
    result = dict(raw)
    selected_adapter = adapter if adapter is not None else result.get("adapter")
    telegram = isinstance(selected_adapter, str) and selected_adapter.strip() == "telegram"
    for key in ("allowed_users", "group_ids"):
        if key not in result:
            continue
        value = result[key]
        expected = "a list of nonempty strings"
        if telegram:
            expected = (
                "a list of positive ASCII-decimal user ID strings (not handles or wildcards)"
                if key == "allowed_users"
                else "a list of signed nonzero ASCII-decimal group ID strings"
            )
        error = f"gateway.{key}: expected {expected}; edit this field in settings.json or use [] to clear it."
        if not isinstance(value, list):
            raise AccessSettingsError(error)
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise AccessSettingsError(error)
            item = item.strip()
            if telegram:
                pattern = r"[0-9]+" if key == "allowed_users" else r"[+-]?[0-9]+"
                if re.fullmatch(pattern, item) is None or not item.lstrip("+-").strip("0"):
                    raise AccessSettingsError(error)
                # Telegram serializes integer identities canonically. Preserve
                # their meaning without accepting numeric JSON values or relying
                # on int()'s size limits for hand-edited strings.
                digits = item.lstrip("+-").lstrip("0")
                item = f"-{digits}" if item.startswith("-") else digits
            if item not in normalized:
                normalized.append(item)
        result[key] = normalized
    if "pairing_enabled" in result and not isinstance(result["pairing_enabled"], bool):
        raise AccessSettingsError(
            "gateway.pairing_enabled: expected a boolean; set true or false in settings.json "
            "(false disables onboarding)."
        )
    return result


def parse_allowed_users(value: str) -> list[str]:
    """Split form/CLI input without silently dropping malformed empty entries."""
    return [] if not value.strip() else [part.strip() for part in value.split(",")]
