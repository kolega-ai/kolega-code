Ask the user one or more multiple-choice questions and wait for their answers.

Use this only for decisions that materially affect the outcome and cannot be settled from the
repository. Each question has a short `header`, the `question` text, a `multiSelect` flag, and an
`options` array of `{label, description}` choices. The user selects one option per question or
types a custom free-text answer. Questions are asked one at a time, in order.

Returns:
    A JSON object mapping each question's header (or its text) to the chosen option label
    or the user's custom answer.
