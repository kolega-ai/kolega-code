"""Fixed, credential-safe status messages for MCP failures.

A stored failure message is shown again only when it is exactly one of these strings,
so an exception that happens to carry a token can never resurface from a status file.
"""

from __future__ import annotations

MCP_FAILURE_MESSAGE_GENERIC = "Verification failed. Check the MCP server configuration and try again."
MCP_FAILURE_MESSAGE_GITHUB_COPILOT_OAUTH = (
    "GitHub remote MCP OAuth failed: this endpoint does not support generic MCP dynamic client registration. "
    "Use a PAT Authorization header for this endpoint, or a client with GitHub-provided OAuth support."
)
MCP_FAILURE_MESSAGE_OAUTH_REGISTRATION = (
    "OAuth dynamic client registration failed. This server may require a pre-registered OAuth client or "
    "bearer-token header."
)
MCP_FAILURE_MESSAGE_OAUTH_REGISTRATION_NON_DCR = (
    "OAuth dynamic client registration failed. Check the server's registration requirements; "
    "if it requires pre-registered credentials, configure an OAuth client ID."
)
MCP_FAILURE_MESSAGE_OAUTH_SECRET_ENV = (
    "The OAuth client secret environment variable is missing or empty. "
    "Set the configured client_secret_env variable and verify again."
)
MCP_FAILURE_MESSAGE_OAUTH_CALLBACK = (
    "Could not listen on the configured OAuth callback address. "
    "Check that its port is available and the redirect URI matches the registered application."
)
MCP_FAILURE_MESSAGE_OAUTH_UNAUTHORIZED = (
    "MCP OAuth authorization failed. Check the server credentials or bearer-token header."
)
MCP_FAILURE_MESSAGE_OAUTH_SIGNIN_INCOMPLETE = (
    "MCP sign-in was not completed. Run `kolega-code mcp login` for this server to sign in."
)
MCP_FAILURE_MESSAGE_OAUTH_SIGNIN_REQUIRED = (
    "MCP sign-in is required. Run `kolega-code mcp login` for this server to sign in."
)
MCP_FAILURE_MESSAGE_UNAUTHORIZED = (
    "MCP authentication failed. Check the configured credentials or Authorization header."
)
MCP_FAILURE_MESSAGE_ENV_MISSING = "An environment variable this server references is not set. Set it and verify again."
MCP_FAILURE_MESSAGE_NOT_FOUND = "MCP server endpoint was not found. Check the configured URL."
MCP_FAILURE_MESSAGE_TIMEOUT = "MCP server verification timed out. Check the server URL and network connectivity."
MCP_FAILURE_MESSAGE_CONNECTION = (
    "Could not connect to the MCP server. Check the command, URL, and network connectivity."
)
MCP_FAILURE_MESSAGE_COMMAND = "Could not start the MCP server command. Check the configured command and arguments."
MCP_FAILURE_MESSAGE_PERMISSION = "Permission denied while starting or contacting the MCP server."
MCP_TOOL_FAILURE_MESSAGE_GENERIC = "MCP tool call failed."
MCP_TOOL_FAILURE_MESSAGE_TIMEOUT = "MCP tool call timed out. Check the server URL and network connectivity."

SAFE_FAILED_STATUS_MESSAGES = frozenset(
    {
        MCP_FAILURE_MESSAGE_GENERIC,
        MCP_FAILURE_MESSAGE_GITHUB_COPILOT_OAUTH,
        MCP_FAILURE_MESSAGE_OAUTH_REGISTRATION,
        MCP_FAILURE_MESSAGE_OAUTH_REGISTRATION_NON_DCR,
        MCP_FAILURE_MESSAGE_OAUTH_SECRET_ENV,
        MCP_FAILURE_MESSAGE_OAUTH_CALLBACK,
        MCP_FAILURE_MESSAGE_OAUTH_UNAUTHORIZED,
        MCP_FAILURE_MESSAGE_OAUTH_SIGNIN_INCOMPLETE,
        MCP_FAILURE_MESSAGE_OAUTH_SIGNIN_REQUIRED,
        MCP_FAILURE_MESSAGE_UNAUTHORIZED,
        MCP_FAILURE_MESSAGE_ENV_MISSING,
        MCP_FAILURE_MESSAGE_NOT_FOUND,
        MCP_FAILURE_MESSAGE_TIMEOUT,
        MCP_FAILURE_MESSAGE_CONNECTION,
        MCP_FAILURE_MESSAGE_COMMAND,
        MCP_FAILURE_MESSAGE_PERMISSION,
    }
)


def safe_failed_status_message(message: str) -> str:
    """The stored message when it is a known safe one, otherwise the generic failure."""
    return message if message in SAFE_FAILED_STATUS_MESSAGES else MCP_FAILURE_MESSAGE_GENERIC
