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
    Structured text with status, session id, exit code, duration, and
    model-visible output. `max_output_tokens` is a requested budget and
    is always clamped to the global 10,000-token terminal-result limit.
    Oversized streams are represented by a bounded head/tail preview;
    the result reports truncation statistics and an ordinary `Full
    output:` filesystem path containing the complete normalized stream.
    Individual preview lines are capped so one huge line cannot consume
    the result. Background launches that are still running also report
    `Background: true` and management guidance.
