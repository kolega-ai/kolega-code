You are handing off an in-progress coding session to a fresh session that has NO access to this conversation. Write a handoff document that lets the new session continue seamlessly.

Hard rules:
- Output ONLY the handoff document. No preamble, no commentary, no wrapper text.
- Capture exact technical state, not abstractions: file paths, symbol names, commands run, test results, observed failures, decisions made, and partial work affecting the next step.
- Quote identifiers EXACTLY: file paths, function/class names, variable names, error strings, commands, and config keys. Do not paraphrase code or paths.
- Do not invent facts. If something is unknown or unfinished, say so briefly.
- Preserve any `<skill_content name="...">` instructions referenced earlier — note them by name so they are not lost.
- Be thorough: completeness matters more than brevity here, because the new session cannot look anything up in this conversation.

Use exactly this structure:

## Goal
What the user is trying to accomplish

## Constraints & Preferences
- Any constraints, preferences, or requirements mentioned

## Progress
### Done
- [x] Completed tasks with specifics

### In Progress
- [ ] Current work, if any

### Pending
- [ ] Tasks mentioned but not started

## Key Decisions
- **Decision**: rationale

## Critical Context
- Code snippets, file paths, function/type names, error messages, and data essential to continue
- Repository state if relevant

## Next Steps
1. What should happen next
