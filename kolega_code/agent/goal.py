"""Goal-condition verification for the autonomous ``/goal`` loop.

The verifier is a read-only investigation sub-agent (see
``BaseAgent.evaluate_goal_condition``): it inspects the current codebase state to
decide whether a goal condition is met, then reports a JSON verdict. This module
holds the verdict type, the verifier instruction, and the verdict parser — it has
no CLI/TUI or LLM-client dependencies so it stays cheap to import and easy to unit
test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class GoalVerdict:
    """The outcome of one goal evaluation.

    ``met`` must only be ``True`` when the verifier produced an explicit
    ``{"ok": true}`` verdict — a malformed or unparseable reply is always treated
    as not-met so a broken evaluator can never falsely complete a goal.
    """

    met: bool
    reason: str = ""


def build_goal_verifier_instruction(condition: str) -> str:
    """Build the task given to the read-only verification sub-agent.

    The verifier receives only the condition and inspects the current codebase
    state fresh each evaluation (stateless across evaluations). It must not modify
    anything — it only reads files, searches, and runs commands/tests to verify.
    """
    return (
        "You are verifying whether the following goal has been fully achieved.\n\n"
        f"Goal:\n{condition.strip()}\n\n"
        "Inspect the current state of the codebase to verify the goal: read files, "
        "search the code, and run commands or tests as needed. Do NOT modify anything "
        "— only verify. Be strict: the goal is met only when there is concrete evidence "
        "(for example, the described files/code exist, and relevant tests or commands "
        "pass). If anything is missing or failing, the goal is not yet met.\n\n"
        "When finished, end your reply with a single JSON object on its own line:\n"
        '  {"ok": true} if the goal is fully achieved, or\n'
        '  {"ok": false, "reason": "<the specific remaining gap>"} if it is not.'
    )


def _iter_json_objects(text: str):
    """Yield ``(start, end)`` spans of top-level balanced ``{...}`` objects.

    Strings are tracked so braces inside JSON string values do not confuse the
    scanner. Yields objects in document order; the caller can take the last.
    """
    start = -1
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    yield start, i + 1
                    start = -1


def parse_goal_verdict(result_text: str) -> GoalVerdict:
    """Parse a verifier recap into a :class:`GoalVerdict`.

    Scans for balanced JSON objects and accepts the **last** one that decodes to a
    dict with an ``ok`` key. On any failure (no object, bad JSON, missing key), the
    verdict is ``met=False`` with a truncated copy of the raw text as the reason —
    so a malformed reply never completes a goal and the user can see what happened.
    """
    if not result_text:
        return GoalVerdict(met=False, reason="verifier returned no output")

    candidates = list(_iter_json_objects(result_text))
    for start, end in reversed(candidates):
        snippet = result_text[start:end]
        try:
            parsed = json.loads(snippet)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict) and "ok" in parsed:
            ok = parsed.get("ok")
            reason = parsed.get("reason")
            return GoalVerdict(
                met=bool(ok),
                reason=str(reason) if isinstance(reason, str) and reason.strip() else "",
            )

    truncated = result_text.strip()
    if len(truncated) > 280:
        truncated = truncated[:280].rstrip() + "…"
    return GoalVerdict(met=False, reason=f"verifier reply was not a valid verdict: {truncated}")
