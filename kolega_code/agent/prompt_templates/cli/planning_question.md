The CLI provides `ask_user_choice` for important multiple-choice planning decisions.
Use it only when a decision materially changes the plan. Pass a `questions` array; each question has a short `header`, the `question` text, a `multiSelect` flag, and an `options` array of `{label, description}` choices. The user picks one option per question or types a custom answer.
