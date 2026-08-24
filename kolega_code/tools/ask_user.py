"""Shared ``ask_user_choice`` input normalization for interactive hosts.

Both the TUI (control-channel questions) and the ACP server (elicitation
forms) expose the same ``ask_user_choice`` tool contract
(``ASK_USER_CHOICE_INPUT_SCHEMA``). This module owns the strict input
validation so the two hosts can never drift apart.
"""

from __future__ import annotations

from kolega_code.tools.core import ToolError
from kolega_code.tools.definitions import ASK_USER_CHOICE_SHAPE_HINT


def normalize_choice_questions(questions: object) -> list[tuple[str, str, list[str], list[str]]]:
    """Validate the structured questions input and return normalized questions.

    Strict: rejects malformed input with an instructive ToolError instead of coercing.
    Each result is (question_text, header, option_labels, option_descriptions).
    """
    if not isinstance(questions, list) or not questions:
        raise ToolError("'questions' must be a non-empty array of question objects. " + ASK_USER_CHOICE_SHAPE_HINT)

    normalized: list[tuple[str, str, list[str], list[str]]] = []
    for question in questions:
        if not isinstance(question, dict):
            raise ToolError("each item in 'questions' must be an object. " + ASK_USER_CHOICE_SHAPE_HINT)

        clean_question = str(question.get("question", "")).strip()
        if not clean_question:
            raise ToolError("each question must include non-empty 'question' text. " + ASK_USER_CHOICE_SHAPE_HINT)

        header = str(question.get("header", "")).strip()

        raw_options = question.get("options")
        if not isinstance(raw_options, list):
            raise ToolError(
                "each question's 'options' must be an array of {label, description} objects. "
                + ASK_USER_CHOICE_SHAPE_HINT
            )

        labels: list[str] = []
        descriptions: list[str] = []
        for option in raw_options:
            if not isinstance(option, dict):
                raise ToolError(
                    "each option must be an object with a 'label' (and ideally a 'description'). "
                    + ASK_USER_CHOICE_SHAPE_HINT
                )
            label = str(option.get("label", "")).strip()
            if not label:
                continue
            labels.append(label)
            descriptions.append(str(option.get("description", "")).strip())

        if len(labels) < 2:
            raise ToolError(
                "each question needs at least two options, each with a non-empty 'label'. " + ASK_USER_CHOICE_SHAPE_HINT
            )

        normalized.append((clean_question, header, labels, descriptions))

    return normalized
