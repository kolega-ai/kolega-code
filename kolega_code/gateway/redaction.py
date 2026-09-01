"""Outbound redaction: keep credentials out of chat text and transcripts.

The model can regurgitate API keys it has seen in config or tool output; a
chat app turns any leak into a screenshot on a stranger's phone. Every reply
and status line passes through :func:`scrub` with the secret list the CLI
already maintains for exports (``known_secret_values``), plus the gateway's
own Telegram token.
"""

from __future__ import annotations

from typing import Sequence

REDACTED_PLACEHOLDER = "[redacted]"
#: Shorter values are far too likely to collide with ordinary words ("ask",
#: "new"); real credentials are never that short.
MIN_SECRET_LENGTH = 6


def scrub(text: str, secret_values: Sequence[str]) -> str:
    """Replace every known secret in ``text`` with a placeholder.

    Longest-first so overlapping secrets redact fully, and each distinct
    value is replaced at most once per pass for determinism.
    """
    if not text or not secret_values:
        return text
    secrets = sorted(
        {str(secret) for secret in secret_values if secret and len(str(secret)) >= MIN_SECRET_LENGTH},
        key=len,
        reverse=True,
    )
    for secret in secrets:
        text = text.replace(secret, REDACTED_PLACEHOLDER)
    return text
