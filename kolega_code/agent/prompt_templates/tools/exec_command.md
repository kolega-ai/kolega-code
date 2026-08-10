Run a shell command as a fresh process and return its output.

The command runs under a pseudo-terminal so interactive programs behave
normally. Output is collected for up to yield_time_ms; if the process
exits within that window the full result with its real exit code is
returned, otherwise a session_id comes back to drive with write_stdin
(send input or poll) and stop with kill_command.

The working directory does NOT persist between calls: pass `workdir`,
or chain `cd path && ...` in one call. Defaults to the project root.

Use background=true for long-lived processes (dev servers, watchers,
builds) when you want to keep working while they run. Never use shell
`&` — those processes are killed when the command that started them
ends. Verify a server answers (e.g. curl) before handing its URL to the
browser agent — its log may be buffered.

Returns:
    A JSON object: {"status": "exited"|"running", "exit_code",
    "session_id", "output", "truncated", "original_token_count",
    "duration_ms"}. Background launches that are still running also
    include "background": true and a "note" with management hints.
