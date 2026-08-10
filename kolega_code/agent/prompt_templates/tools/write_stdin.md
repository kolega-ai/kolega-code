Write input to a running session's stdin and read recent output.

Pass chars="" to poll (read new output without writing). Use this to
answer prompts (e.g. send "y\n"), drive a REPL, or send control
characters (e.g. "\x03" for Ctrl-C). The text is sent raw — include a
trailing "\n" to submit a line. Waits up to yield_time_ms for more
output or for the process to exit.

Works for background sessions too: input is delivered to their stdin
but not echoed, so verify the effect from the command's output; their
stdin never reaches EOF, so stop stdin-reading commands with
kill_command.

Returns:
    Structured text with the same bounded output and spill-file metadata
    as exec_command. `max_output_tokens` is a requested budget and is
    always clamped to the global 10,000-token terminal-result limit.
