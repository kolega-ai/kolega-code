---
name: loop
description: Autonomous loop engineering — auto-detects bug-fix or new-feature workflow and activates the right methodology
---

# Loop Engineering — Auto-Router

When the user invokes you with `/loop <description>`, analyze the description
and activate the appropriate sub-skill.

## Routing Logic

1. **Bug-fix keywords**: fix, bug, crash, error, broken, regression, debug,
   repair, patch, resolve, incorrect, wrong, fails, not working, 500,
   exception, traceback, segfault, timeout, hang, null pointer, type error,
   assertion error, index error, key error → Call `activate_skill("bug-fix-loop")`
   then proceed with the bug-fix methodology using the full bug description.

2. **Feature keywords**: build, create, add, implement, feature, new, make,
   develop, write, generate, scaffold, design, extend, refactor → Call
   `activate_skill("new-code-loop")` then proceed with the new-code methodology
   using the full feature specification.

3. **Ambiguous**: If the description contains both or neither → Ask the user:
   "Is this a bug fix or a new feature?" Then activate the correct skill based
   on their answer.

## Activation

After routing, activate the chosen skill immediately:
```
activate_skill("<skill-name>")
```

Then follow that skill's workflow precisely. Do not skip phases. Do not
shortcut the methodology.
