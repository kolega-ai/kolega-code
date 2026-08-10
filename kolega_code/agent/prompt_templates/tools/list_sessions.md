List currently running exec sessions.

Includes sessions started with exec_command background=true, annotated
with "background": true.

Returns:
    Structured text listing each running session id, command, working
    directory, runtime in seconds, and background flag. Returns `No
    running sessions.` when the registry is empty.
