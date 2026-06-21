You are compacting an in-progress coding session so the work can continue seamlessly in a smaller context window. Produce a TIGHT continuity briefing — not a transcript.

Hard rules:
- Respond with the summary text ONLY. Do NOT call tools. Do NOT ask questions.
- Be bounded: aim for under ~600 words. Omit anything not needed to keep working.
- Quote identifiers EXACTLY: file paths, function/class names, variable names, error strings, commands, and config keys. Do not paraphrase code or paths.
- Do not invent facts. If something is unknown or unfinished, say so briefly.
- Preserve any `<skill_content name="...">` instructions referenced earlier — note them by name so they are not lost.

Write these sections, each terse. Use `##` headers and keep them in this order:

## Goal
The user's overall objective and any explicit constraints, in 1-3 sentences.

## State
What is true RIGHT NOW: what has been built or changed, what works, what has been verified.

## Files Touched
A bullet list of exact paths read/created/modified, each with a 3-8 word note on what changed or why it matters. No code blocks unless a snippet is load-bearing.

## Decisions
Key technical decisions and their one-line rationale. Note approaches already tried and rejected.

## Open Problems
Exact open errors (quote them), failing tests, or unresolved questions. Write "None" if there are none.

## Next Steps
The concrete next actions to take, in order. Write "None" if the task appears complete.
