"""Minimal, dependency-free JWT *payload* reader.

We only need to read claims (account id, plan type, email) from tokens that
arrive over TLS from OpenAI's token endpoint — TLS is the trust boundary, so we
do not verify the signature here. Do not use this to make trust decisions about
tokens received from untrusted transports.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from .constants import AUTH_CLAIMS_NAMESPACE


def decode_jwt_payload(token: str) -> dict[str, Any]:
    """Return the decoded JSON payload (middle segment) of a JWT.

    Returns an empty dict if the token is malformed or not a JWT.
    """
    try:
        payload_segment = token.split(".")[1]
    except (AttributeError, IndexError):
        return {}
    # Re-pad to a multiple of 4 for urlsafe base64.
    padding = "=" * (-len(payload_segment) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload_segment + padding)
        decoded = json.loads(raw)
    except (ValueError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def extract_chatgpt_claims(id_token: str) -> dict[str, Any]:
    """Pull the ChatGPT-subscription claims out of an id_token.

    Returns a dict with ``account_id``, ``plan_type``, ``email`` and ``user_id``
    (values may be ``None`` when absent). The subscription claims live under the
    ``https://api.openai.com/auth`` namespace; ``email`` is a top-level claim.
    """
    payload = decode_jwt_payload(id_token)
    auth_claims = payload.get(AUTH_CLAIMS_NAMESPACE)
    if not isinstance(auth_claims, dict):
        auth_claims = {}
    return {
        "account_id": auth_claims.get("chatgpt_account_id"),
        "plan_type": auth_claims.get("chatgpt_plan_type"),
        "user_id": auth_claims.get("chatgpt_user_id"),
        "email": payload.get("email") or auth_claims.get("email"),
    }
