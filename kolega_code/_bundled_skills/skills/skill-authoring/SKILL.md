---
name: skill-authoring
description: Create, revise, review, and package standards-compliant Agent Skills with effective trigger descriptions, progressive disclosure, reusable resources, and focused validation. Use when an agent needs to design a new skill, improve an existing SKILL.md, organize scripts/references/assets, or audit a skill before distribution.
license: Apache-2.0
compatibility: Requires filesystem access to create or edit a skill directory.
metadata:
  author: Kolega
  version: "1.0"
---

# Skill authoring

Create focused skills that add procedural knowledge rather than restating general model
capabilities.

## Workflow

1. **Clarify activation.** Collect at least three realistic requests that should trigger the
   skill and two near misses that should not. Identify the user-visible outcome.
2. **Inspect the environment.** Find existing conventions, tools, dependencies, and related
   skills before choosing a structure.
3. **Plan reusable contents.** Put only the core workflow in `SKILL.md`. Use `scripts/` for
   deterministic operations, `references/` for detailed knowledge, and `assets/` for files
   copied into outputs. Omit directories that are not needed.
4. **Write metadata.** Make `name` match the skill directory. Make `description` state both
   the capability and the situations in which the skill should activate.
5. **Write instructions.** Use imperative steps, explicit decision points, concrete commands,
   and task-specific checks. Explain non-obvious constraints. Avoid generic advice.
6. **Connect resources.** Link directly from `SKILL.md` to each resource and state when to
   load or run it. Keep references one level deep where practical.
7. **Exercise the skill.** Test activation examples, near misses, normal execution, and at
   least one failure path. Run every added script on representative input.
8. **Validate and package.** Use the repository's validator and packager when available.
   Otherwise verify the Agent Skills specification manually and archive the complete skill
   directory without caches, credentials, or generated work.

## Frontmatter

Start `SKILL.md` with YAML frontmatter:

```yaml
---
name: lowercase-kebab-name
description: State what the skill does and when to use it.
license: Apache-2.0
---
```

Use `compatibility` only for genuine environment constraints. Keep `metadata` values as
strings. Treat `allowed-tools` as experimental and include it only when the target client
supports it.

## Progressive disclosure

Keep always-loaded metadata precise, activated instructions concise, and detailed resources
on demand:

- Keep `SKILL.md` under 500 lines when practical.
- Split by task or domain, not by arbitrary document size.
- Do not duplicate the same guidance in `SKILL.md` and a reference.
- Add a short table of contents to long reference files.
- Keep a skill self-contained; do not depend on files elsewhere in its source repository.

Before finishing, apply the [review checklist](references/review-checklist.md).
