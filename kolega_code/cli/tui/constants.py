"""Constants used by the Textual UI."""

#: Identifies this client to the session control channel. Exactly one client may
#: answer prompts at a time, and this is how the terminal UI claims that right.
TUI_CLIENT_ID = "kolega-code-tui"

BUILD_INTERACTION_MODE = "build"
PLAN_INTERACTION_MODE = "plan"
QUESTION_TOOL_NAME = "ask_user_choice"
QUESTION_OPTION_ID_PREFIX = "question_option_"
APPROVAL_OPTION_ID_PREFIX = "approval_option_"
MODEL_OPTION_ID_PREFIX = "model_option_"
EFFORT_OPTION_ID_PREFIX = "effort_option_"
THEME_OPTION_ID_PREFIX = "theme_option_"

#: How long a stop request waits for the turn worker to unwind before the UI
#: detaches it. Normal cancels settle in milliseconds; this only fires when the
#: worker is parked in a wait that swallows cancellation (a journal write on a
#: worker thread, whose finalizer loops with ``asyncio.shield``), which otherwise
#: leaves the dashboard on "Stopping…" forever.
STOP_FORCE_AFTER_SECONDS = 5.0

#: How long the post-turn history save may take before the turn finalizes without
#: it. Generous (a large history dump is normal), but finite: a journal write
#: stalled behind the session lock must not hide the turn's result (the plan card).
TURN_SAVE_TIMEOUT_SECONDS = 30.0

TAB_BASE_LABELS = {
    "logs_pane": "Logs",
    "terminal_pane": "Terminal",
}

STARTUP_WORDMARK = (
    " _  __     _                   ____          _",
    "| |/ /___ | | ___  __ _  __ _ / ___|___   __| | ___",
    "| ' // _ \\| |/ _ \\/ _` |/ _` | |   / _ \\ / _` |/ _ \\",
    "| . \\ (_) | |  __/ (_| | (_| | |__| (_) | (_| |  __/",
    "|_|\\_\\___/|_|\\___|\\__, |\\__,_|\\____\\___/ \\__,_|\\___|",
    "                  |___/",
)
