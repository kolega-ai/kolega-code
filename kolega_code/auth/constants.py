"""Codex "Sign in with ChatGPT" OAuth constants.

Single source of truth for every Codex-specific value. These are reverse-
engineered from the open-source ``openai/codex`` CLI (``codex-rs/login`` and
``codex-rs/model-provider-info``). If OpenAI rotates the client id or moves the
backend, this is the only file that needs to change.
"""

from __future__ import annotations

# OAuth provider id used to key stored tokens in settings (a distinct provider
# from the static-API-key "openai").
PROVIDER_KEY = "openai_chatgpt"

# Public PKCE client; no secret. Codex allows overriding via the
# CODEX_APP_SERVER_LOGIN_CLIENT_ID env var — we mirror that for parity/testing.
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CLIENT_ID_ENV = "KOLEGA_CODE_CHATGPT_CLIENT_ID"

# OAuth 2.0 Authorization Code + PKCE (S256) endpoints.
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"

# Loopback redirect for the browser flow.
DEFAULT_REDIRECT_PORT = 1455
FALLBACK_REDIRECT_PORT = 1457
REDIRECT_PATH = "/auth/callback"

# Scopes + extra authorize params Codex requests.
SCOPES = "openid profile email offline_access api.connectors.read api.connectors.invoke"
EXTRA_AUTHORIZE_PARAMS = {
    "id_token_add_organizations": "true",
    "codex_cli_simplified_flow": "true",
}

# Inference backend (Responses API only).
INFERENCE_BASE_URL = "https://chatgpt.com/backend-api/codex"
# Default model selected after a successful ChatGPT sign-in.
DEFAULT_MODEL = "gpt-5.5"
# Codex identifies itself with these on the backend; the originator + a
# codex_cli_rs User-Agent are what the backend expects from the Codex client.
ORIGINATOR = "codex_cli_rs"
CODEX_VERSION = "0.50.0"
USER_AGENT = f"codex_cli_rs/{CODEX_VERSION}"
ACCOUNT_ID_HEADER = "ChatGPT-Account-ID"

# Custom claims namespace inside the id_token / access_token JWT.
AUTH_CLAIMS_NAMESPACE = "https://api.openai.com/auth"

# Refresh the access token this many seconds before it expires.
REFRESH_LEEWAY_SECONDS = 300

# Plan types that may NOT use subscription inference.
INELIGIBLE_PLAN_TYPES = frozenset({"free"})


def redirect_uri(port: int) -> str:
    """Loopback redirect URI for the given port."""
    return f"http://localhost:{port}{REDIRECT_PATH}"


def client_id(env: dict[str, str] | None = None) -> str:
    """Resolve the OAuth client id, honoring the override env var."""
    import os

    source = env if env is not None else os.environ
    return source.get(CLIENT_ID_ENV) or CLIENT_ID
