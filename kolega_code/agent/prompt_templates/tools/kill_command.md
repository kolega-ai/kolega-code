Terminate a running session and its process group.

Sends SIGTERM (then SIGKILL after a short grace period). Use
signal="INT" to send Ctrl-C (SIGINT) instead.

Returns:
    A JSON object describing the final state of the session.
