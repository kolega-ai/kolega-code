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

# Blockers
BLOCK_STOP_BEFORE_RESET = "Stop the current turn before resetting the thread."
BLOCK_STOP_BEFORE_MODE_SWITCH = "Stop the current turn before switching modes."
BLOCK_STOP_BEFORE_SKILL = "Stop the current turn before activating a skill."
BLOCK_PLAN_DECISION = "Choose Implement plan or Discuss further before sending another message."
BLOCK_PLAN_DECISION_MODE_SWITCH = "Choose Implement plan or Discuss further before switching modes."
BLOCK_PLAN_DECISION_SKILL = "Choose Implement plan or Discuss further before activating a skill."
BLOCK_PENDING_QUESTION_SKILL = "Answer the pending planning question before activating a skill."
SETTINGS_REQUIRED = "Save a provider, model, and API key before chatting."
SETTINGS_REQUIRED_SKILL = "Save a provider, model, and API key before activating a skill."

# Misc
COPY_MACOS_FAILED = "Copied for supported terminals, but the macOS clipboard failed."
STREAM_TRUNCATED = "[stream truncated to the last {chars} characters]"
