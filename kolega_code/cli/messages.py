"""User-facing microcopy for the Kolega Code CLI.

Single voice for every string the user reads: sentence case, full sentences
end with a period, in-progress states end with a single ellipsis character,
no exclamation marks.
"""

from __future__ import annotations

# Composer placeholders and modal prompts
COMPOSER_PLACEHOLDER = "Ask Kolega Code..."
DISCONNECTED_COMPOSER_PLACEHOLDER = "Connect a model in Settings before chatting."
PLAN_READY_PLACEHOLDER = "Plan ready. Choose Implement plan or Discuss further."
QUESTION_PLACEHOLDER = "Choose an option below or type a custom answer..."
APPROVAL_PLACEHOLDER = "Choose whether to allow this action..."
MODEL_PLACEHOLDER = "Choose a model below or type a supported model name..."
EFFORT_PLACEHOLDER = "Choose a thinking effort below or type a supported value..."
THEME_PLACEHOLDER = "Choose a theme below or type a theme name..."

# Durable transcript messages
THREAD_RESET_MESSAGE = "Thread reset. Previous messages were cleared."
DISCONNECTED_HEADLINE = "Not connected."
DISCONNECTED_STARTUP_GUIDANCE = "Choose a provider and add an API key or sign in from the Settings tab before chatting."
DISCONNECTED_SIDEBAR_GUIDANCE = "Press Ctrl+O to open the sidebar, then select Settings."
DISCONNECTED_ACTIVITY = "Open Settings and connect a provider to start chatting."
DISCONNECTED_MODEL = "not connected"
TASK_LIST_EMPTY_MESSAGE = "No task list has been set."
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
GENERATING = "Generating…"
THINKING = "Thinking…"
READING_RESPONSE = "Reading model response…"
STOP_REQUESTED = "Stopping…"
FINISHED = "Finished."
STOPPED_BY_USER = "Stopped by user."
STOPPED_WITH_ERROR = "Stopped due to an error: {error}"
CANCEL_REQUESTED = "Cancellation requested."
WAITING_FOR_ANSWER = "Waiting for your answer…"
WAITING_FOR_PERMISSION = "Waiting for permission…"
QUEUED_MESSAGE = "Queued. It will be sent when the current turn finishes."
QUEUE_PLACEHOLDER = "Queue a follow-up…"
QUEUE_EMPTY = "No queued messages."
QUEUE_CLEARED = "Cleared {count} queued message(s)."
QUEUE_LIST_TITLE = "Queued messages:"

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
CHANGES_INSPECTOR_NO_SELECTION = "No file selected."
CHANGES_COPIED = "Copied file changes to the clipboard."

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

# Mentions
MENTIONS_NOT_FOUND = "Not found, sent as plain text: {mentions}"

# Slash commands
MODEL_SWITCHED = "Switched model to {provider}/{model} with thinking effort {effort}."
MODEL_UNKNOWN = "Unknown model {model} for provider {provider}."
MODEL_SWITCH_HINT = "Choose below, or switch with /model <name>."
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

# Settings tab
SETTINGS_SAVED = "Settings saved."
SETTINGS_INCOMPLETE = "Configuration incomplete: {error}"
SETTINGS_ACTIVE_MODEL = "Active model: {provider}/{model}"
SETTINGS_ACTIVE_MODEL_UNCONFIGURED = "Active model: not configured"
SETTINGS_API_KEY_LINE = "API key: {status}"
SETTINGS_THINKING_EFFORT_LINE = "Thinking effort: {effort}"
SETTINGS_ACTIVE_THEME = "Active theme: {theme}"

# Status dashboard
STATUS_TOKENS_UNKNOWN = "Token counts unavailable."
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
