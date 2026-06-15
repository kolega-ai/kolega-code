"""User-facing microcopy for the Kolega Code CLI.

Single voice for every string the user reads: sentence case, full sentences
end with a period, in-progress states end with a single ellipsis character,
no exclamation marks.
"""

from __future__ import annotations

# Composer placeholders and modal prompts
COMPOSER_PLACEHOLDER = "Ask Kolega Code..."
PLAN_READY_PLACEHOLDER = "Plan ready. Choose Implement plan or Discuss further."
QUESTION_PLACEHOLDER = "Choose an option below or type a custom answer..."
EFFORT_PLACEHOLDER = "Choose a thinking effort below or type a supported value..."

# Durable transcript messages
THREAD_RESET_MESSAGE = "Thread reset. Previous messages were cleared."
TASK_LIST_EMPTY_MESSAGE = "No task list has been set."
PLAN_EMPTY_MESSAGE = "No plan captured yet."

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

# Confirmations
SWITCHED_MODE = "Switched to {mode} mode."
PLAN_CAPTURED = "Plan captured. Choose Implement plan or Discuss further."
PLAN_DISCUSSION_RESUMED = "Planning discussion resumed."
SKILL_ACTIVATED = "Activated skill {name}."
SKILLS_LISTED = "Listed agent skills."

# Mentions
MENTIONS_NOT_FOUND = "Not found, sent as plain text: {mentions}"

# Slash commands
MODEL_SWITCHED = "Switched model to {provider}/{model} with thinking effort {effort}."
MODEL_UNKNOWN = "Unknown model {model} for provider {provider}."
MODEL_SWITCH_HINT = "Switch with /model <name>."
EFFORT_SWITCHED = "Switched thinking effort to {effort} for {provider}/{model}."
EFFORT_UNKNOWN = "Unknown thinking effort {effort} for {provider}/{model}."
EFFORT_UNSUPPORTED = "{provider}/{model} does not support a thinking effort setting."
EFFORT_SWITCH_HINT = "Choose below, or switch with /effort <level>."
COPY_LAST_RESPONSE = "Copied the last response to the clipboard."
COPY_NOTHING = "No response to copy yet."
VERSION_INFO = "Kolega Code version {version}."

# Blockers
BLOCK_STOP_BEFORE_RESET = "Stop the current turn before resetting the thread."
BLOCK_STOP_BEFORE_MODE_SWITCH = "Stop the current turn before switching modes."
BLOCK_STOP_BEFORE_SKILL = "Stop the current turn before activating a skill."
BLOCK_STOP_BEFORE_MODEL_SWITCH = "Stop the current turn before switching models."
BLOCK_STOP_BEFORE_EFFORT_SWITCH = "Stop the current turn before switching thinking effort."
BLOCK_PLAN_DECISION = "Choose Implement plan or Discuss further before sending another message."
BLOCK_PLAN_DECISION_MODE_SWITCH = "Choose Implement plan or Discuss further before switching modes."
BLOCK_PLAN_DECISION_SKILL = "Choose Implement plan or Discuss further before activating a skill."
BLOCK_PENDING_QUESTION_SKILL = "Answer the pending planning question before activating a skill."
SETTINGS_REQUIRED = "Configure a provider/model and API key before chatting."
SETTINGS_REQUIRED_SKILL = "Configure a provider/model and API key before activating a skill."

# Settings tab
SETTINGS_SAVED = "Settings saved."
SETTINGS_INCOMPLETE = "Configuration incomplete: {error}"
SETTINGS_ACTIVE_MODEL = "Active model: {provider}/{model}"
SETTINGS_ACTIVE_MODEL_UNCONFIGURED = "Active model: not configured"
SETTINGS_API_KEY_LINE = "API key: {status}"
SETTINGS_THINKING_EFFORT_LINE = "Thinking effort: {effort}"

# Status dashboard
STATUS_TOKENS_UNKNOWN = "Token counts unavailable."

# Logs
LOG_IGNORED_EVENT = "Ignored non-display event: {event_type}"

# Misc
COPY_MACOS_FAILED = "Copied for supported terminals, but the macOS clipboard failed."
STREAM_TRUNCATED = "[stream truncated to the last {chars} characters]"
