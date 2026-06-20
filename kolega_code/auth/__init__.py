"""ChatGPT-subscription ("Sign in with ChatGPT") OAuth support.

This package replicates the OpenAI Codex CLI login flow so a user's ChatGPT
subscription (Plus/Pro/Business) can drive OpenAI models inside kolega-code.

IMPORTANT / unofficial: this reuses Codex's first-party OAuth ``client_id`` and
the ``chatgpt.com`` backend. That is undocumented and against the spirit of
OpenAI's terms of service, with attendant account-ban risk. All Codex-specific
constants live in :mod:`kolega_code.auth.constants` so a forced change (e.g.
OpenAI rotating the client id) is a single-file edit.
"""

from __future__ import annotations

from .tokens import ChatGPTTokenManager, OAuthError, OAuthTokens, TokenRefreshError

__all__ = ["ChatGPTTokenManager", "OAuthError", "OAuthTokens", "TokenRefreshError"]
