List currently running exec sessions.

Includes sessions started with exec_command background=true, annotated
with "background": true.

Returns:
    A JSON object mapping each running session id to its command,
    working directory, runtime in seconds, and background flag.
