---
name: deep-research
description: >
  Conduct rigorous multi-source investigation and create citation-backed Markdown
  reports with reader-appropriate structure and voice. Use for explicit deep-research
  requests, historical or cultural reports, decision comparisons, disputed claims,
  literature landscapes, and questions requiring source discovery, selective
  verification, synthesis, and uncertainty analysis. Do not use for a single-fact
  lookup, a one-URL summary, routine coding research, or a simple search-result list.
license: Apache-2.0
compatibility: >
  Requires web search and page-reading tools for live research. Best with Kolega
  Code Gigacode workflows and runtime model discovery; prompts to enable Gigacode
  before offering a bounded sequential fallback when orchestration is unavailable.
  Uses a session scratchpad directory for evidence dossiers when one is available,
  and falls back to compact in-result records when it is not.
metadata:
  owner: Kolega
  version: "3.0"
---

# Deep Research

Produce a useful report, not a transcript of research operations. Research only as
widely as the question, stakes, and remaining uncertainty require. Preserve source
quality and uncertainty while adapting the report's shape and voice to its readers.

## Activation examples

Use this skill for requests such as:

- "Run deep research and create a report on Saturnian magic throughout history."
- "Compare these three observability platforms and recommend one for a small
  regulated team."
- "Investigate the disputed claim that this policy reduced housing costs, and show
  what the best evidence supports."

Do not activate it for near misses such as:

- "What year did the policy take effect?" — answer the fact directly.
- "Summarize this one article." — read and summarize that source without a research
  workflow.

## Brief the report with the user

Before checking Gigacode availability or acquiring sources, ask one compact batch
covering all unresolved core fields and 1–3 genuinely topic-specific questions. A
silent capability check may occur at any time, but do not offer the Gigacode
enablement choice until the brief is confirmed.

### Core fields

Extract everything the initial request already answers and carry those values into
the confirmation summary. Ask only what remains unresolved:

- **Length** — offer exactly these presets:
  - `concise` — about 1,500 words;
  - `standard` — about 3,000 words;
  - `detailed` — about 6,000 words;
  - `long` — 10,000 words or more;
  - or a custom word target.

  Map `long` to a 10,000-word minimum. If the user wants materially more than
  10,000 words, obtain a concrete custom target before confirmation. Never infer
  length from tier when the user has not supplied it.

- **Audience and use** — who will read the report and what decision,
  understanding, or action it should support.

- **Scope** — relevant timeframe, geography, comparison set, definitions,
  inclusions, and exclusions. Ask only the applicable dimensions.

- **Delivery** — Markdown artifact and path versus conversation-only; preserve
  the collision-safe default when the user accepts it.

### Topic-specific questions

In the same batch, ask 1–3 concise questions whose answers will shape lanes,
evidence standards, or synthesis. Choose questions specific to this topic, for
example:

- **Historical/cultural:** endpoint, tradition or interpretive lens, primary
  practice versus later reception.
- **Product/market:** options, operating constraints, decision criteria, current
  environment.
- **Disputed/high-stakes:** jurisdiction, decision at stake, acceptable evidence
  threshold.
- **Experiential/community:** communities or platforms, period, whose experiences
  matter, desired treatment of representativeness.

Do not ask generic questions that merely restate the title, expose workflow
mechanics, or ask the user to choose tier, lanes, models, verification, audit, or
drafting mode.

### Proposed research brief

After the user answers, return a compact **Proposed research brief** containing
the research question, audience and use, scope, selected length preset and numeric
target, report style and structure requirements, delivery path or mode, and the
settled topic-specific decisions. Explicitly ask the user to confirm or correct
the brief, and wait. Begin Gigacode enablement, routing, and research only after
confirmation.

### Narrow exceptions

- **Already-answered field:** skip that question; show the extracted value in the
  confirmation summary.
- **User-directed no-questions:** if the user explicitly asks not to be asked
  follow-ups or to start immediately, state the proposed defaults—including a
  concrete word target—and proceed without the second confirmation turn. Record
  this as a user-directed exception.
- **Correction:** reopens only the affected part of the brief; do not repeat the
  complete questionnaire.

### Examples

**Underspecified request — "Research the history of Saturnian magic."**

No core fields are supplied. Ask all four plus a topic question in one batch:

> To research this well I need a few details.
>
> **Length:** concise (~1,500 words), standard (~3,000), detailed (~6,000), long
> (10,000+), or a custom target?
> **Audience and use:** who will read this and what should it help them understand
> or decide?
> **Scope:** any particular traditions, periods, or geographies to focus on or
> exclude?
> **Delivery:** Markdown file (I will propose a collision-safe path) or
> in-conversation?
> **Topic question:** Are you more interested in the ancient and Renaissance
> textual tradition, the later occult revival, or both?

After the user answers, summarize the **Proposed research brief** and wait for
confirmation before proceeding.

**Detailed initial request — "Write a detailed 6,000-word report on Saturnian magic
in Renaissance Europe for a general educated audience, focusing on Ficino and
Agrippa, as a Markdown file."**

Length (`detailed` / 6,000 words), audience, scope, and delivery are all resolved.
Skip those questions and ask only what remains:

> You have given me most of what I need. One question before I start:
>
> Should the report emphasize the philosophical underpinnings of Saturn's role, the
> practical magical techniques described, or both in roughly equal measure?

After the user answers, return the **Proposed research brief** and wait for
confirmation.

## Check Gigacode before research

After the brief is confirmed, check whether `run_workflow` is available before
acquiring sources. If it is unavailable and the user has not
already requested sequential execution or told you to continue without Gigacode,
pause and offer this choice:

> Gigacode is off. Parallel research is faster and keeps the bounded research lanes
> isolated. Run `/gigacode on` and then resend this research request, or tell me to
> continue without Gigacode.

Do not claim that the skill can execute a TUI slash command itself. Do not start the
sequential fallback until the user chooses it. If the user continues without
Gigacode, do not ask again during that research request. If `run_workflow` is
available, proceed without this prompt; absence of `list_subagent_models` alone does
not mean Gigacode is off.

After this preflight, read
[Evidence and reporting](references/evidence-and-reporting.md) before research.
When Gigacode is available, also read the concise
[Gigacode workflow guide](references/gigacode-workflow.md) and invoke the bundled
[`scripts/deep-research.workflow`](scripts/deep-research.workflow) instead of
generating a new orchestration script.

## 1. Apply the confirmed brief and settle delivery

Use the question, audience, scope, length target, and delivery mode confirmed during
intake. Do not re-ask resolved fields. If a detail needed for the artifact path or
report profile was not covered during intake, resolve it now before announcing the
output path.

### Artifact-first delivery

In an edit-enabled session, an explicit report request produces a Markdown file by
default:

1. Honor a user-supplied path.
2. Otherwise derive a short topic slug and announce
   `reports/<topic-slug>.md` before research starts.
3. Never silently overwrite. If the path exists and overwrite was not requested,
   choose `<topic-slug>-2.md`, then `-3`, and so on.
4. Treat conversation-only delivery as an explicit format choice.
5. In read-only mode, return the report in conversation and say that no file could be
   written.

After a workflow run, immediately use
[`scripts/materialize_report.py`](scripts/materialize_report.py) with its
`resultPath`. The materializer is the authoritative gate: it builds `## Sources`
from the body, validates structure, appends a sanitized `## Scope and gaps` section
for a partial result, and warns when the delivered length departs from the confirmed
target or a stage was degraded. Report any warning it prints, and check
`run_summary.degraded_stages`: a degraded stage means a decision the run was supposed
to make was discarded, which is worth telling the user about. Finish with the report
path and a short status; do not paste the full report again.

### Keep intermediate work out of the project

Research intermediates belong in the session scratchpad, not in the user's
repository. When the session advertises a scratchpad directory, pass it to the
workflow as `workspace.scratchpad_dir` with a short `run_slug`, so scouts can write
full-fidelity evidence dossiers that later stages read on demand instead of
squeezing everything through compact return values.

The scratchpad holds dossiers, section drafts, and the assembled report body. The
finished report is the only deliverable and it goes into the project. If no
scratchpad is available, the workflow runs on its inline records instead — say so
if evidence depth suffered.

## 2. Choose proportional effort

Choose by uncertainty and consequence, not by topic breadth alone.

| Tier | Use when | Base lanes | Per-lane acquisition ceiling |
| --- | --- | ---: | --- |
| Focused | Narrow, low-stakes synthesis with modest uncertainty | 2 | 4 searches, 6 fetches |
| Standard | Default multi-source report or comparison | 3–4 | 6 searches, 8 fetches |
| Extended | Explicitly exhaustive, high-stakes, or unusually disputed | 5–6 | 8 searches, 12 fetches |

These are ceilings, never quotas. Stop a lane as soon as its material claims have
adequate support. Lane ownership must be disjoint; do not add a cross-cutting lane
that re-fetches every period or option.

Focused runs skip follow-up research by default. Standard runs may add at most one
follow-up, and only when the gap could change the thesis or recommendation. Extended
runs may use one acquisition escalation under the rules below, not an open-ended
retry chain.

## 3. Plan source classes, not source counts

Before searching, identify the source classes the question needs: primary records,
official data, peer-reviewed analysis, technical documentation, contemporary
reporting, or lived-experience sources. Assign each class to one lane.

Use discovery sources to find authoritative evidence, then cite the authoritative
evidence. Prefer a smaller set of strong, representative citations over collecting
sources to meet a quota.

## 4. Route stages at runtime

When Gigacode and model discovery are available:

1. Call `list_subagent_models` once.
2. Use the user's requested model family when specified; otherwise anchor routing
   to the effective Investigation default's provider and model family.
3. Keep stage roles in that family. Prefer the same exact model with lower effort
   for bounded discovery, extraction, coverage classification, and
   mechanical/editorial checks; next prefer a clearly related faster sibling from
   the same provider and family.
4. Use higher effort or the stronger sibling in that same family for difficult
   conclusion-driving verification, conflict resolution, and synthesis.
5. If family membership is unclear, stay on the same exact model or omit the
   override. Do not assemble a sampler of unrelated providers or model families
   merely to specialize each stage.
6. Cross the family boundary only when the user directs it or the chosen family
   lacks a required capability, such as vision for the one permitted Browser
   escalation. Limit the exception to the affected role and disclose it.
7. Pass only exact configured routes in workflow `args.routes`. If no safe
   same-family alternate is clear, omit that role's override and inherit the
   configured agent-type default. An omitted role inherits silently; a route that
   is present but malformed fails the run before dispatch rather than quietly
   falling back.

Never guess a route. Do not copy a provider, model ID, or effort value into this
skill or its resources. Routes are complete runtime values returned by the current
session's discovery tool. A shared provider name alone does not establish a shared
model family; use only an obvious lineage shown by runtime model names, and treat
ambiguous cases as unrelated.

Use read-only Investigation workers for ordinary research. A Browser worker is an
exception for one approved acquisition escalation and requires a runtime-discovered
vision-capable route when an override is used.

## 5. Use the bounded workflow

With Gigacode:

1. Call `run_workflow` with `script_path` pointing at this skill's
   [`scripts/deep-research.workflow`](scripts/deep-research.workflow). Do not read
   the file into context to pass it as `script`; `script_path` takes precedence and
   avoids a large pointless round trip.
2. Pass the settled brief — including `current_as_of`, since workers have no clock —
   plus disjoint lanes, tier, ceilings, report profile, stage plan, workspace, and
   optional runtime routes.
3. Use a tier-appropriate budget. Do not increase an exhausted budget by a large
   multiplier without changing scope or workflow shape.
4. On interruption, inspect `resultPath` and `transcriptPath`. Resume only to reuse
   valid persisted calls while narrowing optional work; do not rerun merely to
   recover omitted inline output.
5. Materialize the final or explicitly supported partial report from `resultPath`.

The workflow researches and verifies lanes as one pipeline with no barrier between
the stages, verifies every lane that holds important or disputed claims, constructs
its registry deterministically, and uses one ordinary drafting agent. The
skill—not the user—decides whether section fan-out is warranted after coverage
analysis. It uses sections for an explicitly long report, usually around 5,000 words
or more, or when the supported evidence separates into genuinely independent
sections that benefit from bounded parallel drafting. An assembly pass then unifies
argument, transitions, voice, and citations.

A `failed` result means no report exists. A drafted report with residual issues
comes back as `partial` with those issues in `gaps` — never discarded.

### Sequential fallback

The confirmed brief from intake applies to sequential research unchanged. Bypassing
Gigacode does not bypass intake; if the brief has not been confirmed, complete it
before proceeding.

If Gigacode is unavailable and the user chooses to continue without it:

1. Research the same 2–6 disjoint lanes sequentially with the same ceilings.
2. Keep the source/evidence registry and failed-acquisition ledger **in scratchpad
   files**, not in conversation context, using the same layout the workflow uses:
   `lanes/<lane>.md` per lane plus a `registry.json`. A long sequential run will
   outlive its own context; citations must survive compaction.
3. Verify conclusion-driving or disputed claims with genuinely independent support.
4. Run one combined evidence/editorial audit for Focused or Standard work.
5. Revise only if the audit finds a material problem.
6. Rebuild `## Sources` from the files, restricted to URLs actually cited in the
   final body. Preserve query strings when comparing URLs: two pages that differ
   only in a query parameter are different documents.
7. Write the report directly using the artifact-first path rules.

Orchestration improves speed and cost control; its absence must not weaken the
evidence standard or prevent delivery.

## 6. Stop failed acquisition quickly

Every worker prompt must carry these rules:

- Never repeat an identical query.
- Never retry a terminally failed URL by changing only protocol, anchor, query
  string, or endpoint shape.
- Treat 403/404, login/paywall, robots denial, certificate failure, unsupported
  type, oversized document, and scanned/no-text results as terminal for that source.
- For a conclusion-driving source, try at most one obvious accessible equivalent.
- Retry a transient timeout once, then move on.
- Pass failures to verification; verifiers must not retry known failures or broadly
  re-fetch every scout source.
- Prefer a substitute source or narrower claim over downloading, conversion,
  browser automation, or OCR.
- Focused and Standard runs do not initiate OCR.
- Extended/high-stakes or explicitly exhaustive work may use at most one Browser
  **or** local conversion/OCR escalation—not both—for an irreplaceable source that
  could change the answer. If it fails, disclose the gap and continue.

## 7. Draft for the reader

All reports must answer the question early, cite material factual claims, separate
evidence from inference, preserve important disagreement, and include exactly one
deduplicated Sources section containing only cited sources.

Presentation is adaptive:

- **Historical/cultural/humanities:** use a narrative or chronological arc, a
  specific title, concrete period-appropriate headings, and an opening thesis. Do
  not default to `Executive answer`, `Methods`, or `Limitations`.
- **Product/market/policy:** lead with the answer, comparisons, trade-offs,
  recommendation, and risks. An executive summary may fit.
- **Scientific/technical/high-stakes:** foreground method, evidence quality,
  uncertainty, and limitations when they aid interpretation.
- **Community/trend/experiential:** organize around patterns and voices while making
  representativeness limits clear without repetitive disclaimers.

Match the user's register and the subject's texture without sensationalizing,
imitating an author, or sacrificing precision. Keep material uncertainty near the
affected claim, but do not expose worker mechanics, evidence IDs, audit language, or
repeated boilerplate caveats in the report.

Use the confirmed numeric target from intake for orchestration purposes. Do not
ask the user to choose single-agent versus section-based drafting.

## 8. Audit only what needs judgment

- Focused and ordinary Standard: one combined evidence/editorial audit — pass
  `stage_plan.audit: "combined"`.
- Extended or high-stakes: two independent audits — pass
  `stage_plan.audit: "dual"`. This does not happen by default; set it explicitly.
- Use `stage_plan.audit: "deterministic"` only when no judgment-based audit is
  warranted.
- Set `stage_plan.verification` and `stage_plan.followup` in the same object:
  `risk_only` verification for experiential or testimony-led work, `selective` for
  ordinary reports, `required` when nearly every lane carries checkable
  conclusions; `followup: "off"` for Focused runs.
- Revise only for material issues.
- Use an independent closure review only for unresolved critical or major evidence
  issues in high-stakes work.
- Never dispatch an agent for deterministic citation, bibliography, duplicate
  heading, empty-section, or output-file checks.

If a critical claim remains unsupported, remove or narrow it. A disclosed gap is
better than another expensive loop that is unlikely to change the answer.
