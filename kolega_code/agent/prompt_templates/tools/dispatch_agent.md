Dispatch an autonomous sub-agent to complete a self-contained task.

The sub-agent works without further input and returns one final report. You
cannot see its intermediate steps or send follow-up messages, so each task
must be INDEPENDENT and SELF-CONTAINED: include the goal, relevant file
paths, constraints, and exactly what the final report should contain. The
report is not automatically shown to the user - summarize the key results.
Sub-agents cannot spawn further sub-agents.

BACKGROUND EXECUTION: by default, dispatch waits for the final report.
Set background=true only when you can continue independent work while the
child runs. It returns an accepted receipt with a stable job_id, NOT the
report or a claim that the task succeeded. The session delivers completion
separately, including failures and cancellation; do not poll routinely.
Use agent_job for explicit inspection, recovery of the full report, or
cancellation. Background launches require a supported session owner and
cannot be made by sub-agents or workflow workers.

PARALLEL EXECUTION: multiple dispatch_agent calls issued in a single
response run CONCURRENTLY. Use this to fan out independent work, but never
give two parallel agents work that could overlap on the same files. Do
tasks that depend on each other's output sequentially or yourself, and
skip dispatch for small tasks you can do directly with a couple of tool
calls or anything needing back-and-forth with the user.

Returns:
    The sub-agent's final report, or an accepted job receipt when background=true
