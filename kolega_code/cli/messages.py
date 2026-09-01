"""User-facing microcopy for the Kolega Code CLI.

Single voice for every string the user reads: sentence case, full sentences
end with a period, in-progress states end with a single ellipsis character,
no exclamation marks.
"""

from __future__ import annotations

# Composer placeholders and modal prompts
COMPOSER_PLACEHOLDER = "Ask Kolega Code..."
DISCONNECTED_COMPOSER_PLACEHOLDER = "Finish setup or connect a model in Settings before chatting."
PLAN_READY_PLACEHOLDER = "Plan ready. Choose Implement plan or Discuss further."
QUESTION_PLACEHOLDER = "Choose an option below or type a custom answer..."
APPROVAL_PLACEHOLDER = "Choose whether to allow this action..."
MODEL_PLACEHOLDER = "Choose a model below or type a supported model name..."
EFFORT_PLACEHOLDER = "Choose a thinking effort below or type a supported value..."
THEME_PLACEHOLDER = "Choose a theme below or type a theme name..."

# Durable transcript messages
THREAD_RESET_MESSAGE = "Thread reset. Previous messages were cleared."
DISCONNECTED_HEADLINE = "Not connected."
DISCONNECTED_STARTUP_GUIDANCE = "Complete onboarding, or open Settings, to choose a provider and connect a credential."
DISCONNECTED_SIDEBAR_GUIDANCE = "Press Ctrl+O, select Settings, then choose Continue Setup."
DISCONNECTED_ACTIVITY = "Complete setup or open Settings to connect a provider."
DISCONNECTED_MODEL = "not connected"
TASK_LIST_EMPTY_MESSAGE = "No task list has been set."
TASK_LIST_HEADER = "## Task List"
PLAN_EMPTY_MESSAGE = "No plan captured yet."
# Shown once in the startup block when running inside tmux/screen, where Shift
# chords often never reach the app.
TMUX_SHORTCUT_HINT = (
    "tmux/screen: Shift shortcuts may not reach the app. "
    "Use Ctrl+J for newline, /plan or /build for mode, Alt+V or /attach for images. "
    "See docs: Terminal & tmux shortcuts."
)
ATTACH_CLIPBOARD_EMPTY = (
    "No image on the clipboard, or no clipboard tool is available. "
    "Copy an image first, or use /attach <path> or @image.png."
)

# Turn progress
WORKING = "Working…"
PREPARING_CHECKPOINT = "Preparing checkpoint…"
GENERATING = "Generating…"
THINKING = "Thinking…"
THINKING_TITLE = "Thinking"
THINKING_DONE_TITLE = "Thought"
READING_RESPONSE = "Reading model response…"
STOP_REQUESTED = "Stopping…"
FINISHED = "Finished."
STOPPED_BY_USER = "Stopped by user."
STOPPED_WITH_ERROR = "Stopped due to an error: {error}"
CANCEL_REQUESTED = "Cancellation requested."
WAITING_FOR_ANSWER = "Waiting for your answer…"
WAITING_FOR_PERMISSION = "Waiting for permission…"
QUEUED_MESSAGE = "Queued. It will be delivered to the agent at the next tool boundary, or when this turn finishes."
QUEUE_PLACEHOLDER = "Queue a follow-up…"
QUEUE_EMPTY = "No queued messages."
QUEUE_CLEARED = "Cleared {count} queued message(s)."
QUEUE_LIST_TITLE = "Queued messages:"
QUEUE_DELIVERED_MID_TURN = "Delivered {count} queued message(s) to the current turn."
# Peer messages (cross-session): queue-preview attribution and lifecycle notices.
PEER_QUEUE_FROM = "from {sender}"
PEER_MESSAGE_QUEUED = (
    "Peer message from {sender} queued. It will be delivered at the next tool boundary, or when this turn finishes."
)
PEER_MESSAGES_DROPPED_ON_RESTORE = "{count} peer message(s) dropped — they are not restored into the composer."
MESSAGING_SOCKET_UNAVAILABLE = "Peer messaging is in-process only: {reason}"
PEER_RENAMED = "Renamed '{previous}' → '{name}'. Peers can address this session by its new name."
PEER_REPEAT_DROPPED = "Dropped an identical repeat from {sender} (loop protection)."
PEER_QUEUE_FULL = "Dropped an inbound peer message — the queue cap ({limit}) is full."

# Turn status strip finals
DONE_IN = "Done in {duration}"
STOPPED_AFTER = "Stopped after {duration}"
ERRORED_AFTER = "Errored after {duration}"

# Tool and sub-agent activity
RUNNING_TOOL = "Running {tool}…"
TOOL_DONE = "{tool} finished."
TOOL_FAILED = "{tool} failed."
RUNNING_TERMINAL_COMMAND = "Running terminal command…"
RUNNING_SUB_AGENT = "Running sub-agent {name} #{index}…"
RUNNING_SUB_AGENTS = "Running {count} sub-agents…"
SUB_AGENT_INSPECT_HINT = "Ctrl+G to inspect"
SUB_AGENT_INSPECTOR_EMPTY = "No sub-agents have run in this turn yet."
SUB_AGENT_INSPECTOR_NO_SELECTION = "No sub-agent selected."
SUB_AGENT_INSPECTOR_NO_STEPS = "No trajectory captured yet…"
SUB_AGENT_TRAJECTORY_COPIED = "Copied the sub-agent trajectory to the clipboard."
CHANGES_INSPECTOR_EMPTY = "No file changes since this session started."
CHANGES_INSPECTOR_EMPTY_VS = "No file changes since {label}."
CHANGES_INSPECTOR_NO_SELECTION = "No file selected."
CHANGES_COPIED = "Copied file changes to the clipboard."
CHANGES_BASELINE_SESSION_START = "Session start"
CHANGES_SCOPE_WORKTREE = "Worktree: {name}"
CHANGES_SCOPE_BRANCH = "Branch: {branch}"
CHANGES_HISTORY_MOVED = "Commits merged, pulled, or rebased in since the baseline are not shown."
CHANGES_HISTORY_UNTRACKED = "Committed changes are not tracked in this repository."

# Agent-initiated worktree switch. The agent may only switch when the user asked
# for it, so every switch is confirmed here before anything durable is written.
WORKTREE_SWITCH_CONFIRM_QUESTION = "Move this session's workspace from {old_root} to {new_root}{branch_note}?"
WORKTREE_SWITCH_CONFIRM_APPROVE = "Switch workspace"
WORKTREE_SWITCH_CONFIRM_APPROVE_DESCRIPTION = (
    "Rebuild the session in the new checkout at the end of this turn and start a fresh Changes/Rewind baseline."
)
WORKTREE_SWITCH_CONFIRM_DECLINE = "Stay here"
WORKTREE_SWITCH_CONFIRM_DECLINE_DESCRIPTION = "Keep working in {old_root}. Nothing is changed."
WORKTREE_SWITCH_DECLINED = (
    "The user declined the workspace switch, so the active workspace is unchanged. Continue working in "
    "`{old_root}` and do not request the switch again unless the user raises it."
)
WORKTREE_SWITCH_DECLINED_ANSWER = ' The user answered: "{answer}".'

# Rewind
REWIND_BLOCKED_TURN = "Stop the current turn (Esc) before rewinding."
REWIND_NOTHING = "Nothing to rewind — no file changes since {label}."
REWIND_NO_CHECKPOINTS = "No turns to rewind yet."
REWIND_UNAVAILABLE = "Rewind is unavailable until the agent connects."
REWIND_USAGE = "Usage: /rewind [turns-back], e.g. /rewind or /rewind 2"
REWIND_CONFIRM_TITLE = "Rewind session?"
REWIND_CONFIRM_COPY = (
    "Restore {count} file(s) to their state at {label}, discarding +{adds} -{dels}. "
    "Files created since then are deleted; deleted files are restored. Git commits are kept."
)
REWIND_CONFIRM_CONVERSATION = (
    " The conversation also rewinds to that point, and the rewound request is loaded into the composer."
)
REWIND_CONFIRM_FILES_ONLY = " The conversation is not changed."
REWIND_HISTORY_MOVED_WARNING = (
    " History moved since this baseline (merge, rebase, or pull). Restoring writes session-start content "
    "and can revert commits that came from outside this session."
)
REWIND_FILE_CONFIRM_TITLE = "Restore file?"
REWIND_FILE_CONFIRM_COPY = (
    "Restore {path} to its state at {label}, discarding +{adds} -{dels}. The conversation is not changed."
)
REWIND_CONFIRM_LABEL = "Rewind"
REWIND_FILE_CONFIRM_LABEL = "Restore File"
REWIND_DONE = "Rewound {count} file(s) to before {label}."
REWIND_DONE_CONVERSATION = "Rewound the conversation and {count} file(s) to before {label}."
REWIND_DRIFT = "Files changed while confirming. The diff was refreshed — review it and rewind again."
REWIND_PARTIAL = "Rewind stopped after file errors; the conversation was not changed: {detail}"
REWIND_SKIPPED_NOTE = " Skipped: {detail}."
REWIND_SAFETY_NOTE = " Undo with the snapshot tool (id {snapshot_id})."
REWOUND_MARKER = "⤺ Rewound to before: {excerpt}"
REWIND_GOAL_PAUSED = "Goal paused by rewind. Send a message to resume it."
REWIND_LOOP_STOPPED = "Loop stopped by rewind. Start a new one with /loop <interval> <prompt>."

# Confirmations
SWITCHED_MODE = "Switched to {mode} mode."
SWITCHED_PERMISSION_MODE = "Switched permissions to {mode} mode."
SIDEBAR_HIDDEN = "Sidebar hidden."
SIDEBAR_SHOWN = "Sidebar shown."
PLAN_CAPTURED = "Plan captured. Choose Implement plan or Discuss further."
PLAN_REOFFERED = "No new plan captured. Reusing the last captured plan."
PLAN_DISCUSSION_RESUMED = "Planning discussion resumed."
SKILL_ACTIVATED = "Activated skill {name}."
SKILLS_LISTED = "Listed agent skills."
SKILLS_DISABLED = "Agent Skills are disabled for this session."

# Mentions
MENTIONS_NOT_FOUND = "Not found, sent as plain text: {mentions}"

# Slash commands
MODEL_SWITCHED = "Switched model to {provider}/{model} with thinking effort {effort}."
MODEL_UNKNOWN = "Unknown model {model} for provider {provider}."
MODEL_CUSTOM_EMPTY = "Type a model id (vendor/model) in the custom model field, or pick a listed model."
MODEL_CUSTOM_UNKNOWN = (
    '"{model}" is not a known {provider} model. See "kolega-code models list --provider {provider}" '
    'for valid ids, or run "kolega-code models refresh" to pick up newer models.'
)
MODEL_SWITCH_HINT = "Choose below, or switch with /model <name>."
MODEL_PARTIAL_LISTING_HINT = (
    "{provider} offers many more models. Any of them works with /model <id> — "
    "list them with `kolega-code models list --provider {provider}`."
)
MODEL_NON_VISION_IMAGE_HISTORY = (
    "This thread contains images from earlier turns. The current model does not support "
    "vision, so those images will be replaced with text placeholders — switch back to a "
    "vision-capable model with /model to see them again."
)
MODEL_NON_VISION_IMAGE_ATTACHED = "This model can't see images — use /detach to remove or /model to switch."
MODEL_NON_VISION_IMAGE_BLOCKED = "Not sent — this model can't see images. Use /detach to remove or /model to switch."
EFFORT_SWITCHED = "Switched thinking effort to {effort} for {provider}/{model}."
EFFORT_UNKNOWN = "Unknown thinking effort {effort} for {provider}/{model}."
EFFORT_UNSUPPORTED = "{provider}/{model} does not support a thinking effort setting."
EFFORT_SWITCH_HINT = "Choose below, or switch with /effort <level>."
THEME_SWITCHED = "Switched theme to {theme}."
THEME_UNKNOWN = "Unknown theme {theme}."
THEME_SWITCH_HINT = "Choose below, or switch with /theme <name>."
PERMISSIONS_STATUS = "Permissions are in {mode} mode."
PERMISSIONS_SWITCH_HINT = "Switch with /permissions auto, /permissions ask, /permissions toggle, or Ctrl+P."
COPY_LAST_RESPONSE = "Copied the last response to the clipboard."
COPY_NOTHING = "No response to copy yet."

# Live sharing
SHARE_STARTED = "Sharing this session. The link is on your clipboard."
SHARE_ALREADY = "Already sharing. The link is on your clipboard."
SHARE_STOPPED = "Stopped sharing this session."
SHARE_NOT_RUNNING = "This session is not being shared."
SHARE_FAILED = "Could not start sharing: {error}"
SHARE_USAGE = "Usage: /share [lan] [port], or /share stop. For example: /share, /share lan, /share 9000."
SHARE_PORT_TAKEN = "Port {port} was busy, so this share took another one. Pass a port to insist on a specific one."
SHARE_LINK_HEADING = "Live session link"
SHARE_LOOPBACK_NOTE = (
    "Reachable from this machine only. Use /share lan for your local network, "
    "or forward the port through a tunnel you control."
)
SHARE_LAN_WARNING = (
    "Bound to this machine's address on your local network: anyone who can reach it there and "
    "has this link can read this whole session, including file contents and command output. The "
    "link's token is the only thing gating it. Run /share stop when you are done."
)
SHARE_READ_ONLY_NOTE = "Viewers watch live and can read; they cannot type, approve, or interrupt."
SHARE_UNREDACTED_NOTE = (
    "A live link is not redacted: whatever the agent printed, including secrets, is visible to "
    "anyone holding it. Use kolega-code share export for a copy with secrets scrubbed."
)
VERSION_INFO = "Kolega Code version {version}."
UPDATE_STARTED = "Updating Kolega Code…"
UPDATE_COMPLETED = "Kolega Code update completed. Restart this TUI to use the updated version."
UPDATE_FAILED = "Kolega Code update failed with exit code {code}."

# Blockers
BLOCK_STOP_BEFORE_RESET = "Stop the current turn before resetting the thread."
BLOCK_STOP_BEFORE_INIT = "Stop the current turn before running /init."
BLOCK_STOP_BEFORE_MODE_SWITCH = "Stop the current turn before switching modes."
BLOCK_STOP_BEFORE_SKILL = "Stop the current turn before activating a skill."
BLOCK_STOP_BEFORE_MODEL_SWITCH = "Stop the current turn before switching models."
BLOCK_STOP_BEFORE_EFFORT_SWITCH = "Stop the current turn before switching thinking effort."
BLOCK_STOP_BEFORE_UPDATE = "Stop the current turn before updating Kolega Code."
BLOCK_STOP_BEFORE_HANDOFF = "Stop the current turn before handing off the session."
BLOCK_PLAN_DECISION = "Choose Implement plan or Discuss further before sending another message."
BLOCK_PLAN_DECISION_INIT = "Choose Implement plan or Discuss further before running /init."
BLOCK_PLAN_DECISION_MODE_SWITCH = "Choose Implement plan or Discuss further before switching modes."
BLOCK_PLAN_DECISION_SKILL = "Choose Implement plan or Discuss further before activating a skill."
BLOCK_PENDING_QUESTION_INIT = "Answer the pending planning question before running /init."
BLOCK_PENDING_QUESTION_SKILL = "Answer the pending planning question before activating a skill."
BLOCK_PENDING_APPROVAL = "Choose whether to allow the pending action before continuing."
BLOCK_PENDING_APPROVAL_MODE_SWITCH = "Choose whether to allow the pending action before switching permission modes."
SETTINGS_REQUIRED = "Configure a provider/model and API key before chatting."
SETTINGS_REQUIRED_SKILL = "Configure a provider/model and API key before activating a skill."

# Handoff
HANDOFF_NOTHING_TO_SUMMARIZE = "Nothing to hand off (no messages yet)."
HANDOFF_GENERATING = "Generating handoff… (Esc to cancel)"
HANDOFF_CANCELLED = "Handoff cancelled."
HANDOFF_FAILED = "Handoff failed: {error}"
HANDOFF_SUCCESS = "New session started with handoff context: {title}"
HANDOFF_OLD_SESSION_NOTICE = "Handed off to new session {title}."

# Settings tab
SETTINGS_SAVED = "Settings saved."
SETTINGS_INCOMPLETE = "Configuration incomplete: {error}"
SETTINGS_ACTIVE_MODEL = "Active model: {provider}/{model}"
SETTINGS_ACTIVE_MODEL_UNCONFIGURED = "Active model: not configured"
SETTINGS_API_KEY_LINE = "API key: {status}"
SETTINGS_THINKING_EFFORT_LINE = "Thinking effort: {effort}"
SETTINGS_ACTIVE_THEME = "Active theme: {theme}"
BROWSER_MODEL_VISION_READY = "Browser agent model supports vision."
BROWSER_MODEL_INHERIT_VISION_READY = "Browser agent inherits a vision-capable model."
BROWSER_MODEL_INHERIT_NO_VISION = (
    "Browser agent is unavailable: inherited model {provider}/{model} does not support vision. "
    "Choose a vision-capable Browser model."
)
BROWSER_MODEL_EXPLICIT_NO_VISION = (
    "Browser model {provider}/{model} does not support vision. Choose a vision-capable model or Default (inherit)."
)
BROWSER_MODEL_PROVIDER_NO_VISION = (
    "Provider {provider} has no vision-capable Browser models. Choose another provider or Default (inherit)."
)
PROVIDERS_HINT = (
    "Credentials for model providers. Select one to add, replace, or remove its key. "
    "What actually runs is chosen under Models."
)
MODEL_CREDENTIAL_POINTER = "API keys and ChatGPT sign-in live under Providers."
PROVIDER_CREDENTIAL_STATUS = "Credential status: {status}"
PROVIDER_KEY_STAGED = "Key will be saved for {provider} when you Apply."
PROVIDER_KEY_NOT_STORED = (
    "There is no locally stored key for this provider. Environment credentials are not changed here."
)
PROVIDER_KEY_REMOVAL_STAGED = "The locally stored API key will be removed when you Apply."
PROVIDER_TEST_RUNNING = "Testing {provider}/{model}…"
MODEL_SLOT_INHERITED = "Using {provider}/{model} (inherited from the active model)."
MODEL_SLOT_PINNED = "Using {provider}/{model}."
MODEL_SLOT_HINT = (
    "Utility slots used outside the main conversation. Default inherits the active model; "
    "a slot may use a different provider, which needs that provider's API key."
)
ENDPOINTS_HINT = (
    "Point Kolega Code at any OpenAI/Responses/Anthropic-compatible server (LM Studio, Ollama, vLLM, ...). "
    "Each endpoint becomes a provider named custom:<id> and accepts any model id. "
    "Apply Changes to make new or changed endpoints available in the model pickers."
)
ENDPOINT_KEY_EDITED_ON_PAGE = "Endpoint API keys are edited on the Custom Endpoints page."
ENDPOINT_APPLY_REQUIRED = "Saved to the draft. Apply Changes to make it available in the model pickers."
ENDPOINT_DELETED = "Endpoint deleted from the draft. Apply Changes to persist."
ENDPOINT_NONE = "No endpoints yet. Fill in the form and Save Endpoint."

# Status dashboard
STATUS_TOKENS_UNKNOWN = "Token counts unavailable."
STATUS_WORKTREE_LABEL = "Worktree"
COMPACTING = "Compacting conversation…"
COMPACTION_SUMMARY_TITLE = "Conversation compacted — summary"

# Logs
LOG_IGNORED_EVENT = "Ignored non-display event: {event_type}"

# Sign in with ChatGPT
LOGIN_USAGE = "Usage: /login <provider>. Available providers: {targets}."
LOGIN_UNKNOWN_TARGET = "Unknown login provider '{target}'. Available providers: {targets}."
LOGOUT_USAGE = "Usage: /logout <provider>. Available providers: {targets}."
LOGOUT_UNKNOWN_TARGET = "Unknown logout provider '{target}'. Available providers: {targets}."
CHATGPT_LOGIN_STARTING = "Opening your browser to sign in to ChatGPT…"
CHATGPT_LOGIN_URL = "If your browser did not open, visit this URL to sign in:\n{url}"
CHATGPT_LOGIN_SUCCESS = "Signed in to ChatGPT as {email} on the {plan} plan."
CHATGPT_LOGIN_FAILED = "ChatGPT sign-in failed: {error}"
CHATGPT_LOGIN_SWITCH_FAILED = "Signed in, but could not switch to the ChatGPT provider: {error}"
CHATGPT_LOGOUT_DONE = "Signed out of ChatGPT. Stored credentials were removed."
CHATGPT_LOGOUT_NONE = "You are not signed in to ChatGPT."

# Session quit hint
SESSION_RESUME_HINT = "Session saved. Resume it with: kolega-code --resume {session_id}"

# Misc
COPY_MACOS_FAILED = "Copied for supported terminals, but the macOS clipboard failed."
STREAM_TRUNCATED = "[stream truncated to the last {chars} characters]"

# Goal (/goal autonomous completion loop)
GOAL_USAGE = "Usage: /goal <condition> | /goal clear | /goal  (status)"
GOAL_NONE_ACTIVE = "No active goal. Set one with /goal <condition>."
GOAL_SET = "Goal set. Working autonomously until it is met — press Esc to pause, /goal clear to stop."
GOAL_REPLACED = "Replaced the active goal. Working autonomously until it is met."
GOAL_CLEARED = "Goal cleared."
GOAL_MET = "Goal met: {condition}"
GOAL_MAX_TURNS = "Goal not met after {turns} turn(s). Paused — refine the goal with /goal <condition> or /goal clear."
GOAL_PAUSED = "Goal paused: {reason} Send a message to resume, or /goal clear to remove it."
GOAL_EVALUATING = "Evaluating goal…"
GOAL_NOT_MET_CONTINUE = "Goal not yet met — continuing. {reason}"
GOAL_RUN_TO_COMPLETION = "Running to completion (no pauses until the goal is met or capped)."
GOAL_RESUMED_NOTE = "Goal still active: {condition}  Send a message to continue, or /goal clear."
GOAL_BLOCK_STOP_FIRST = "Stop the current turn before changing the goal."
GOAL_BLOCK_SETTINGS = "Configure a provider/model and API key before setting a goal."
GOAL_BLOCK_LOOP_ACTIVE = "A scheduled loop is running. Stop it with /loop stop before setting a goal."

# Loop (/loop scheduled recurring prompts)
LOOP_USAGE = (
    'Usage: /loop <interval> <prompt>  |  /loop --cron "<expr>" <prompt>  |  '
    "/loop status  |  /loop stop\n"
    "Intervals look like 30s, 5m, 2h, 1d or 'every 2 hours'. Options: --fresh, "
    "--max-iterations <n>, --expires <duration>.\n"
    "With no prompt, /loop reads .kolega/loop.md."
)
LOOP_NONE_ACTIVE = "No scheduled loop. Start one with /loop <interval> <prompt>."
LOOP_STARTED = "Loop started: {schedule}. Next iteration {when}. Press Esc or run /loop stop to end it."
LOOP_REPLACED = "Replaced the previous loop. New schedule: {schedule}. Next iteration {when}."
LOOP_ITERATION_STARTED = "Loop iteration {iteration}/{max_iterations} ({schedule})."
LOOP_ITERATION_FRESH = "Fresh thread for this iteration — prior conversation context was cleared."
LOOP_STOPPED = "Loop stopped after {iterations} iteration(s)."
LOOP_STOPPED_BY_USER = "Loop stopped by user after {iterations} iteration(s)."
LOOP_MAX_ITERATIONS = "Loop finished: reached the {max_iterations}-iteration cap."
LOOP_EXPIRED = "Loop expired after {iterations} iteration(s). Start a new one with /loop <interval> <prompt>."
LOOP_RESTORED = "Loop restored: {schedule}. Next iteration {when}."
LOOP_MD_MISSING = "No prompt given and no .kolega/loop.md in this project."
LOOP_MD_GONE = "Loop stopped: .kolega/loop.md is no longer readable."
LOOP_MD_SYMLINK = "Refusing to read {path}: it (or its directory) is a symlink."
LOOP_MD_EMPTY = "{path} has no prompt body."
LOOP_MD_TRUNCATED = "The .kolega/loop.md prompt was truncated to 25,000 bytes."
LOOP_SCHEDULE_MISSING = "No schedule given. Add one to the command or a 'schedule:' line in .kolega/loop.md."
LOOP_SCHEDULE_EMPTY = "A loop schedule must not be empty."
LOOP_SCHEDULE_UNREADABLE = "The saved loop schedule could not be read."
LOOP_BAD_DURATION = "Could not read {value!r} as a duration. Use forms like 30s, 5m, 2h, 1d or 'every 2 hours'."
LOOP_BAD_MAX_ITERATIONS = "--max-iterations needs a positive whole number."
LOOP_UNKNOWN_OPTION = "Unknown option {option}."
LOOP_INTERVAL_TOO_SHORT = "The loop interval must be at least {minimum}s."
LOOP_SUB_MINUTE_ADVISORY = "This loop runs more than once a minute — watch token spend."
LOOP_ASK_PERMISSION_ADVISORY = (
    "Permissions are set to ask. An unattended iteration will stop at the first approval prompt — "
    "switch with /permissions if you plan to leave this running."
)
LOOP_BLOCK_STOP_FIRST = "Stop the current turn before changing the loop."
LOOP_BLOCK_SETTINGS = "Configure a provider/model and API key before starting a loop."
LOOP_BLOCK_GOAL_ACTIVE = "A goal is active. Clear it with /goal clear before starting a loop."
