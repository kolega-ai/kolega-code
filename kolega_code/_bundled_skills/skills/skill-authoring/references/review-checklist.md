# Skill review checklist

## Activation

- [ ] The description states both capability and trigger conditions.
- [ ] Positive examples consistently need this skill's specialized workflow.
- [ ] Near misses are unlikely to trigger it accidentally.
- [ ] The skill does not substantially duplicate another installed skill.

## Instructions

- [ ] Steps are imperative, ordered, and executable.
- [ ] Decision points explain how to choose among meaningful alternatives.
- [ ] Non-obvious constraints include a reason or a concrete failure mode.
- [ ] Examples are realistic and shorter than the guidance they clarify.
- [ ] The final checks measure the requested outcome, not just command success.

## Resources

- [ ] Every bundled file directly supports the skill.
- [ ] `SKILL.md` says when to read or run each resource.
- [ ] Scripts validate inputs, report useful failures, and avoid destructive defaults.
- [ ] Dependencies and environment requirements are documented.
- [ ] Local links resolve with exact filename casing.

## Safety and distribution

- [ ] The skill contains no credentials, personal data, caches, or generated work.
- [ ] Third-party assets have recorded provenance and redistribution-compatible terms.
- [ ] Metadata matches the skill directory and the Agent Skills specification.
- [ ] Representative workflows and failure paths have been exercised.
- [ ] The complete skill validates and packages without relying on repository-external files.
