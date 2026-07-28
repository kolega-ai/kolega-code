---
title: Scheduled Loops
description: Re-run a prompt on an interval or a cron schedule inside your session.
---

A loop re-runs one prompt on a schedule — a fixed interval or a 5-field cron
expression — inside the current session. It's the answer to "check on this every
few minutes and tell me what happened": watch a deploy, poll CI, tend a branch,
or run a recurring review while you work on something else.

It's available as the `/loop` slash command in the [TUI](./tui/interface/) and as
`kolega-code ask --loop` from the [CLI](./cli/ask/).

```
/loop 5m check whether CI went green on this branch and summarize any failures
```

## Command forms

| Command | What it does |
| --- | --- |
| `/loop <interval> <prompt>` | Run `<prompt>` every `<interval>`, starting now |
| `/loop --cron "<expr>" <prompt>` | Run `<prompt>` on a cron schedule |
| `/loop "<expr>" <prompt>` | Same — a leading double-quoted argument is read as cron |
| `/loop <interval>` | Interval from the command, prompt from `.kolega/loop.md` |
| `/loop` | Schedule and prompt both from `.kolega/loop.md` |
| `/loop status` | Show the current (or last) loop |
| `/loop stop` | Stop the loop (aliases: `clear`, `off`, `cancel`, `none`, `reset`) |

Options go before the schedule, in any order:

| Option | Default | Meaning |
| --- | --- | --- |
| `--fresh` | off | Start each iteration after the first from a clean conversation thread |
| `--max-iterations <n>` | `100` | Stop after this many iterations |
| `--expires <duration>` | `7d` | Stop this long after the loop was created |

```
/loop --fresh --max-iterations 20 10m run the failing test and report the first error
/loop --cron "0 9 * * 1-5" summarize PRs opened since yesterday
```

Only one loop runs per session. Starting another replaces it.

## Schedules

### Intervals

Write `30s`, `5m`, `2h`, `1d`, or the longer `every 2 hours`. Units accept the
obvious spellings (`s`/`sec`/`second(s)`, `m`/`min`/`minute(s)`, `h`/`hr`/`hour(s)`,
`d`/`day(s)`).

Two properties matter:

- **The first iteration runs immediately.** You see output right away rather than
  waiting out the first interval.
- **The delay is measured from the end of the previous iteration**, not its start.
  A `5m` loop whose iteration takes eight minutes waits five more minutes after it
  finishes, so a slow iteration never queues an instantly-due next one.

The minimum interval is **15 seconds**. Anything under a minute prints a
token-spend advisory when the loop starts.

### Cron

Cron expressions use the standard five fields — minute, hour, day-of-month,
month, day-of-week — and, unlike intervals, **wait for the first matching time**
rather than running immediately.

| Syntax | Example | Meaning |
| --- | --- | --- |
| `*` | `* * * * *` | Every minute |
| value | `30 9 * * *` | 09:30 daily |
| range | `0 9-17 * * *` | Hourly, 09:00–17:00 |
| step | `*/15 * * * *` | Every 15 minutes |
| range + step | `0 9-17/4 * * *` | 09:00, 13:00, 17:00 |
| list | `0 9,13,17 * * *` | 09:00, 13:00, 17:00 |

Day-of-week runs `0`–`6` from Sunday, and `7` is also Sunday.

When **both** day-of-month and day-of-week are restricted, a day matches if
**either** field matches — the traditional vixie-cron rule. So `0 0 13 * 5` means
"midnight on the 13th, *and* midnight every Friday", not "Friday the 13th".

Extended syntax is **not** supported and is rejected with an explanatory message
rather than being silently mis-scheduled: `L`, `W`, `?`, `#`, month/day names
(`MON`, `JAN`), and `@daily`-style macros.

All times are your **local** wall clock, so `0 9 * * *` is 9am where you are. The
next fire is recomputed after every iteration, so a daylight-saving transition can
at worst skip or repeat a single fire.

## When iterations actually run

Loop iterations are ordinary turns. The scheduler checks once a second and starts
an iteration only when the session is genuinely idle:

- no turn is running
- no permission prompt, question, model/effort/theme picker, or plan decision is
  waiting
- no messages you typed are queued — **your input always wins over the timer**

A due iteration that can't start is deferred, and `/loop status` and the status
dashboard say `waiting for idle`.

**There is no catch-up.** If three interval windows pass while the agent is busy,
exactly one iteration runs when it goes idle — not three.

Because iterations are real turns, they show up in the transcript, create rewind
checkpoints, and are saved to the session like any other work.

## The prompt

The prompt is sent verbatim on every iteration, wrapped in a short
`[Scheduled loop iteration 3 of 100]` header. The agent also gets a system-prompt
note telling it the turn was started by a timer and the user is probably away, so
it should be concise, prefer read-only checks, and **not** ask questions nobody is
there to answer.

### Skills as loop prompts

If the prompt starts with a skill command, that skill is activated for each
iteration and the rest is used as the prompt:

```
/loop 30m /review changes on this branch
```

A leading token that names a built-in command (`/model`, `/clear`, …) is sent as
plain text instead.

### `.kolega/loop.md`

Commit a `loop.md` under `.kolega/` to give a project a standard loop prompt:

```markdown
# Branch watch

schedule: 15m

Check whether CI finished on the current branch. If it failed, read the logs and
summarize the first real error. If it passed, say so in one line.
```

The optional `schedule:` line supplies the default schedule (an interval or a cron
expression); a schedule on the command line overrides it. A leading Markdown
heading is ignored. The rest of the file is the prompt.

The file is **re-read before every iteration**, so edits take effect on the next
run. It's truncated at 25,000 bytes, and Kolega Code refuses to read it if the
file or its `.kolega` directory is a symlink. Deleting it mid-loop stops the loop.

`loop.md` never starts a loop by itself — it's only consulted when you explicitly
run `/loop`.

## `--fresh`

By default each iteration appends to the running conversation, so the agent
remembers what previous iterations found. That's usually what you want, and
context compaction handles growth.

For a watchdog that runs for hours, `--fresh` clears the conversation before every
iteration after the first. Each run starts from a clean context and has to
rediscover state from the repository, Git history, and any notes on disk — which
is the point: nothing accumulates.

## Stopping a loop

| Action | Effect |
| --- | --- |
| `Esc` while waiting | Clears the pending wakeup and ends the loop |
| `Esc` during an iteration | Cancels the turn **and** ends the loop |
| `/loop stop` | Ends the loop |
| `/clear` or `/reset` | Ends the loop along with the thread |
| Rewinding | Ends the loop |
| Iteration cap or expiry | Ends the loop and says which limit was hit |

Stopping is final: start a new loop with `/loop` rather than resuming. This
differs from [`/goal`](./goal/), which pauses and resumes on your next message —
silently restarting a timer because you typed something unrelated would be
surprising.

## Loops and goals are mutually exclusive

An [autonomous goal](./goal/) and a scheduled loop both drive the same turn
worker, so they can't run at once. `/loop` is refused while a goal is active, and
`/goal` (and the `set_goal` tool) is refused while a loop is active. Clear one
first.

## Unattended runs

An iteration that needs approval will stop and wait for you. If permissions are
set to `ask`, `/loop` says so when it starts: switch with
[`/permissions`](./tui/slash-commands/) if you plan to walk away, and keep loop
prompts read-only when you can't supervise them.

## Sessions

An active loop is saved with the session and restored on
[resume](./tui/sessions-and-resume/), including its schedule, iteration count, and
expiry. A loop that expired while the session was closed is retired instead of
fired. A fire time that passed while you were away triggers exactly one iteration
once the session is idle.

## Headless: `ask --loop`

```bash
kolega-code ask "check the deploy and report" --loop 10m --loop-max-iterations 12
kolega-code ask --loop-cron "0 9 * * 1-5" --loop-fresh   # prompt from .kolega/loop.md
```

| Flag | Meaning |
| --- | --- |
| `--loop <interval>` | Fixed-interval schedule |
| `--loop-cron "<expr>"` | Cron schedule (mutually exclusive with `--loop`) |
| `--loop-max-iterations <n>` | Iteration cap (default `100`) |
| `--loop-expires <duration>` | Wall-clock cap (default `7d`) |
| `--loop-fresh` | Clear history before each iteration after the first |

Progress goes to stderr (`[loop] iteration 3/12 (every 10m)`) so piped stdout stays
the answers. With `--json` you get `loop_iteration`, `loop_sleep`, and
`loop_result` records alongside the usual chunks. The command exits `0` when the
loop ends at its cap or expiry.

`--loop` cannot be combined with `--goal`. Unlike the TUI, a leading skill command
in a headless loop prompt is sent as plain text.

For scheduling that outlives a session entirely, use system `cron` or a CI
schedule to invoke `kolega-code ask` directly.
