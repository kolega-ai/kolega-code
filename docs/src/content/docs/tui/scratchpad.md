---
title: Session Scratchpad
description: A per-session throwaway working directory in OS temp for scripts, venvs, and downloads.
---

Every session gets a private **scratchpad working directory** under the OS
temp dir. The agent is told its path in the system prompt and uses it for
throwaway work that should not touch your repository: one-off scripts, ad-hoc
virtual environments, downloaded files, extracted archives, generated
intermediates, and log captures.

Nothing is written to your repository, so `git status` stays clean.

## Where it lives

The path is shaped like:

```text
<os-temp>/kolega-code-<user>/<project-key>/<session-id>/scratchpad/
```

- `<os-temp>` is the platform temp dir (`$TMPDIR` on macOS, `%TEMP%` on
  Windows, `/tmp` on most Linux).
- `<user>` is the numeric user id on POSIX (temp dirs can be shared there) or
  the login name elsewhere.
- `<project-key>` is the same stable project identity used by
  [project memory](../project-memory/): linked Git worktrees share a key,
  separate clones do not.
- `<session-id>` scopes the directory to one conversation. Resuming a session
  — including across plan/build mode switches — advertises the same path, so
  earlier scratchpad files are still there until the OS cleans them.

## What belongs there (and what does not)

The scratchpad is **throwaway by design**. It is not swept, migrated, or
backed up by Kolega Code — the operating system reclaims temp on its own
schedule (for example on reboot). The agent is instructed accordingly:

- deliverables and anything you must keep belong in the working directory,
  never in the scratchpad;
- sub-agents share the session's scratchpad and use unique filenames to avoid
  collisions;
- the agent must not delete scratchpad files it did not create.

The scratchpad is the only location outside the working directory where the
agent may create files without asking. If the optional shell-command safety
checker is enabled, commands that read, write, or delete files inside the
scratchpad directory are treated as in-scope; other out-of-project paths are
still flagged.

## Relation to other state

The scratchpad is distinct from:

- [project memory](../project-memory/): durable, curated knowledge that
  persists across sessions — never transient progress or task state;
- the current plan artifact: the approved implementation plan preserved
  across compaction;
- session history: the messages and tool results used to resume a
  conversation.

If the temp directory cannot be created, the session simply runs without a
scratchpad section; nothing else changes.
