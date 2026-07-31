# Bounded Gigacode workflow

Use the bundled [`scripts/deep-research.workflow`](scripts/deep-research.workflow)
verbatim. Do not ask a coordinator agent to generate another workflow.

This guide defines the runtime arguments, routing procedure, budget behavior, and
recovery path. Kolega Code's injected Gigacode authoring guide remains authoritative
for primitive signatures.

Use this guide only after the skill's Gigacode preflight. If `run_workflow` is
unavailable, prompt the user to run `/gigacode on` or explicitly choose the
sequential fallback; do not silently fall back.

## Before invocation

1. Confirm that the research brief is complete and confirmed with the user.
   The `intake` object passed to `run_workflow` must carry `confirmed: true`
   (or `mode: "user_directed_defaults"` for the no-questions exception) with all
   four resolved fields present. The workflow validates intake before any worker
   dispatch and returns the standard failed result when intake is missing or
   invalid.
2. Settle the tier, output path, and disjoint lanes from the confirmed brief.
3. If `list_subagent_models` is available, call it exactly once.
4. Select complete exact route objects only from that response, applying the
   same-family-first policy below. Omit a role entirely when no safe alternate is
   clear; a malformed route is rejected, never silently ignored.
5. Pass `script_path` pointing at this skill's own
   `scripts/deep-research.workflow`. Do not read the file into context and pass
   `script`; `script_path` takes precedence and saves the whole round trip. Read
   the file into `script` only if `script_path` fails.
6. Announce the output path before starting.

No provider, model ID, or fixed effort selection belongs in this skill. Route
objects exist only in the current invocation's arguments.

## Arguments

Pass an object with this shape:

```json
{
  "intake": {
    "mode": "interactive",
    "confirmed": true,
    "resolved_fields": ["length", "audience_use", "scope", "delivery"],
    "topic_questions_asked": 2
  },
  "brief": {
    "question": "The settled research question",
    "audience": "Who will read and use the report",
    "scope": "Timeframe, geography, comparison set, and exclusions",
    "current_as_of": "2026-07-28",
    "high_stakes": false
  },
  "tier": "standard",
  "lanes": [
    {
      "id": "lane-1",
      "title": "A disjoint source/subject boundary",
      "question": "What this lane alone must establish",
      "source_classes": ["primary records", "scholarly synthesis"]
    }
  ],
  "acquisition": {
    "searches_per_lane": 6,
    "fetches_per_lane": 8,
    "verification_searches": 2,
    "verification_fetches": 4
  },
  "report_profile": {
    "kind": "historical-cultural",
    "length": "standard",
    "voice": "Engaging, precise, and accessible",
    "target_words": 3000,
    "required_structure": [],
    "avoid_structure": ["Executive answer", "Methods", "Limitations"]
  },
  "stage_plan": {
    "verification": "selective",
    "followup": "thesis_changing",
    "audit": "combined",
    "reason": "ordinary multi-source report"
  },
  "workspace": {
    "scratchpad_dir": "<this session's absolute scratchpad path>",
    "run_slug": "topic-slug"
  },
  "writing_reserve_tokens": 18000,
  "allow_acquisition_escalation": false,
  "escalation": null,
  "routes": {
    "discovery": "<optional complete runtime route object>",
    "verification": "<optional complete runtime route object>",
    "synthesis": "<optional complete runtime route object>",
    "audit": "<optional complete runtime route object>",
    "acquisition": "<optional complete runtime route object>"
  }
}
```

The example values describe a Standard run; adjust ceilings to the tier table in
`SKILL.md`. Do not pass string placeholders as routes in a real invocation. Include
a route key only when its value is a complete object returned for the current
session.

`brief.current_as_of` is **required**. Workflow scripts have no clock, so workers
cannot judge date fitness without it. Pass today's date.

There is no `research_batch_size`. Lanes run as one pipeline and the runtime caps
concurrency automatically; passing the key is a validation error.

### Length

Map the user's confirmed length preset to `target_words` exactly:

| `report_profile.length` | `target_words` |
| --- | ---: |
| `concise` | 1,500 |
| `standard` | 3,000 |
| `detailed` | 6,000 |
| `long` | ≥ 10,000 (use the confirmed custom numeric target) |
| `custom` | any explicit user-confirmed target ≥ 500 |

`long` requires a confirmed numeric target of at least 10,000. `custom` accepts any
target confirmed by the user of at least 500. There is no silent default for
`target_words`; an invocation without a valid `length` preset and corresponding
numeric target is rejected before dispatch.

The user does not select a drafting mode. After verification, coverage analysis
decides whether the report needs bounded section drafting. A target of roughly
5,000 words or more qualifies; a shorter report qualifies only when it has at least
two genuinely independent reader-facing sections with distinct supported claim sets.

### Stage plan

`stage_plan` is optional and defaults to `selective` + `thesis_changing` +
`combined`. Set it deliberately rather than leaving the audit mode implicit:

| Key | Values | Choose |
| --- | --- | --- |
| `verification` | `risk_only`, `selective`, `required` | `risk_only` for experiential/testimony work where external checking does not apply; `selective` for ordinary reports; `required` when nearly every lane carries checkable conclusions |
| `followup` | `off`, `thesis_changing` | `off` for Focused runs and for anything that must not grow; `thesis_changing` otherwise |
| `audit` | `deterministic`, `combined`, `dual` | `combined` for Focused and ordinary Standard; `dual` for Extended or `high_stakes: true`; `deterministic` only when no judgment-based audit is warranted |

`reason` is a short free-text note recorded in telemetry.

Every lane holding eligible claims gets one verifier call carrying up to eight
claims, ordered by risk and importance. No verifier capacity is held back for the
conditional follow-up; the follow-up is gated on the writing reserve when coverage
actually requests it.

### Lane and claim classification

Lane-count validation:

- Focused: exactly 2;
- Standard: 3 or 4;
- Extended: 5 or 6.

Lanes must not overlap. A historical run might divide by periods; a product run
might divide by options or evidence domains. Cross-lane comparison happens after
verification, not in another broad research lane.

Lane identity for evidence keying is assigned by the workflow (`lane-1…N`,
`followup-1`, `escalation-1`), so a worker cannot cross-wire evidence by reusing
another lane's id.

Scouts classify each candidate claim with `claim_type` and optional
`verification_triggers`. Both are schema enums; see
[Evidence and reporting](references/evidence-and-reporting.md) for the values and
what each one means for verification.

### Workspace and scratchpad dossiers

`workspace` is optional but should be passed whenever the session advertises a
scratchpad directory. It turns on full-fidelity evidence dossiers:

- `scratchpad_dir` must be an absolute path — the session scratchpad path from
  your own context, not a project path.
- `run_slug` must be 1–64 characters of lowercase letters, digits, and hyphens,
  starting with a letter or digit.

The workflow derives every path deterministically:

```text
<scratchpad_dir>/deep-research/<run_slug>/
├── lanes/<lane-key>.md          # scout dossier: full quotes with context
├── lanes/<lane-key>.verify.md   # verifier working notes
├── sections/<NN>-<slug>.md      # section bodies for a long report
└── report/body.md               # assembled body, no Sources section
```

Scouts write their dossier with `exec_command` and return only a compact record
plus `dossier_path`. Verification, coverage, drafting, and audit receive paths and
read the depth they need with `read_file_section`. This is what keeps the
handoffs compact without starving later stages of evidence.

The scratchpad is additive and best-effort. When `workspace` is omitted, a worker
cannot write, or a file has been reclaimed from temp, every stage falls back to the
inline compact records and the run still completes. `run_summary.dossiers` reports
how many dossiers were actually persisted.

Never write a deliverable to the scratchpad. The finished report is written into
the project by the materializer.

## Stage routing

Map work to runtime routes by capability:

- `discovery`: bounded search, source metadata, direct extraction;
- `verification`: disputed, ambiguous, or conclusion-driving claims;
- `synthesis`: coverage judgment, final argument, and material revision;
- `audit`: bounded evidence/editorial checks;
- `acquisition`: the one permitted Browser or local escalation.

Choose one routing family rather than a different provider for every role:

1. If the user names a model family, use it. Otherwise use the effective
   Investigation default's provider and model family as the anchor.
2. First vary effort on the same exact model: lower it for bounded discovery and
   mechanical checks, and raise it for verification and synthesis.
3. If useful, choose a runtime-listed faster or stronger sibling only when it is
   clearly in the same provider and model lineage.
4. If lineage is ambiguous, keep the exact anchor model or inherit the configured
   default. Sharing a provider is not enough to infer family membership.
5. Do not distribute ordinary stages across unrelated providers or families merely
   to optimize each role independently.
6. Use another family only at the user's direction or when the anchor family lacks
   a required capability. Keep a capability-driven exception to the affected role
   and disclose it.

The workflow omits `model_override` when a role key is absent. A route that is
present but malformed — a missing `effort`, an extra key, an empty `provider` or
`model`, or an unrecognized role name — is a validation error that fails the run
before dispatch. The workflow never invents a partial override and never quietly
falls back from an invalid one.

## Budget and concurrency

Set a tier-appropriate workflow budget and reserve enough completed-output capacity
for coverage, drafting, audit, and possible material revision. The workflow:

- researches and verifies lanes as one pipeline, with no barrier between the
  stages, so a fast lane's verification starts while a slow lane is still scouting;
- checks the writing reserve before research and before each optional stage;
- filters failed workers at every join;
- skips optional follow-up when it would threaten the reserve; and
- returns a supported partial status when some lanes fail but the remaining evidence
  can answer the question honestly.

`writing_reserve_tokens` defaults to `max(18000, target_words × 1.4 × 2)` — one
full draft plus one full revision. An explicit value below that floor is rejected
with the computed minimum in the message, because a 10,000-word report cannot be
drafted and revised inside an 18,000-token reserve.

Budget admission cannot stop workers already in flight, but the runtime's automatic
active-chain cap bounds the overshoot. Do not respond to exhaustion with a blanket
budget multiplier: inspect persisted artifacts, remove low-value optional work, and
resume from the run ID so unchanged successful calls are reused.

## Optional acquisition escalation

Default `allow_acquisition_escalation` to `false`.

Set it to `true` only for Extended/high-stakes or explicitly exhaustive work and
provide one bounded `escalation` object:

```json
{
  "kind": "browser",
  "target": "Exact source URL",
  "question": "The one conclusion-changing fact to recover"
}
```

`kind` is `browser` or `local`. The workflow performs one attempt total. Browser
requires an available Browser worker and, when overridden, a runtime-discovered
vision-capable route. Local means one bounded read-only conversion/OCR investigation.
Do not combine kinds or retry a failed escalation.

## Results and materialization

The workflow returns a compact object:

```json
{
  "status": "complete | partial | failed",
  "report_markdown": "Present for a single-draft report",
  "report_plan": {
    "title": "Report title",
    "body_path": "<scratchpad>/deep-research/<run_slug>/report/body.md",
    "section_paths": ["..."],
    "assembled_word_count": 11500
  },
  "cited_sources": [{"id": "S001", "title": "...", "url": "..."}],
  "source_registry": [{"id": "S001", "title": "...", "url": "..."}],
  "gaps": ["Concise unresolved gap"],
  "run_summary": {
    "stage_plan": {"verification": "selective", "followup": "thesis_changing", "audit": "combined"},
    "tier": "standard",
    "base_lanes_requested": 4,
    "base_lanes_completed": 4,
    "verifier_calls": 4,
    "eligible_claims": 12,
    "selected_claims": 12,
    "deferred_claims": 0,
    "followups_run": 0,
    "escalations_run": 0,
    "draft_mode": "single | sections",
    "section_outline_source": "coverage | derived | none",
    "degraded_stages": [],
    "target_words": 3000,
    "assembled_word_count": 3120,
    "expansion_passes": 0,
    "dossiers": 4,
    "workspace_enabled": true
  }
}
```

`degraded_stages` names any stage whose structured output was unusable. Structured
output occasionally degenerates — for example collapsing every later field into the
first string field — leaving a dict that passes an isinstance check while carrying
none of the decisions the stage was asked for. The workflow checks the keys it
actually acts on, discards such a record, records the stage here, and names it in
`gaps`. It never proceeds as though the stage had simply chosen the default, because
that silently loses a requested follow-up or a section outline.

`section_outline_source` records where the drafting outline came from. When the
target is 5,000 words or more and coverage supplied no usable outline, the workflow
derives one from the lanes rather than forcing a long report through a single
drafting call, and tells the assembly stage to rename and reorder those headings —
lanes are evidence boundaries, not reader-facing argument boundaries.

Either `report_markdown` or `report_plan` carries the report. A long, section-drafted
report is assembled in a scratchpad file rather than squeezed through one
schema-constrained JSON string, so `report_plan.body_path` is the authoritative
text and `report_markdown` is empty. `source_registry` is always present so the
bibliography can be rebuilt deterministically.

`status` is `failed` only when no report exists at all — a failed drafting worker,
or no claim set that can sustain a cited report. A drafted report with residual
structural or material issues is returned as `partial` with those issues in `gaps`.

Length is treated asymmetrically, because the two failures mean different things. A
shortfall is owned by the expansion pass and reported in `gaps` without blocking
delivery: an honest short report is still worth having. A report more than about
1.35× the target is a material issue handed to the revision pass to tighten, and if
it survives that pass the run is marked `partial` — it disregarded an explicit
instruction it was asked to fix.

Gaps in the result are written for the operator and may be numerous. Workers are
instructed to phrase them for a reader, and the materializer sanitises and caps them
before any of them reach the report.

Raw scout, verifier, coverage, and audit outputs remain visible in normal workflow
artifacts and are not copied into the final result.

Run the materializer with the manifest's `resultPath`:

```bash
python skills/deep-research/scripts/materialize_report.py \
  /path/to/workflow/result.md \
  reports/topic.md \
  --collision-safe
```

Use the available Python 3.11+ interpreter; do not assume the executable is literally
`python`. Use `--overwrite` only when the user explicitly approved replacing the
target.

The materializer is the authoritative gate. It owns Markdown link extraction, URL
identity, `## Sources` construction, and structural validation, and it reads
`report_plan.body_path` when present. For a partial result it appends a
`## Scope and gaps` section, first stripping internal lane and claim identifiers,
dropping entries that are about research machinery rather than evidence, and capping
the list so the report ends with a short disclosure instead of a research ledger. It
warns when the delivered length departs materially from the target in either
direction, and when any stage was degraded. It refuses to write anything for a
`failed` result or an unreadable body file.

## Resume and failure handling

- Read `resultPath` first when inline workflow output is omitted.
- Read `transcriptPath` only to diagnose stage failure or budget use.
- Resume to reuse an unchanged successful prefix while narrowing later work.
- The scratchpad path is stable for a session, so dossiers usually survive a
  resume. If temp has been reclaimed, stages fall back to the inline records.
- If some lanes failed, proceed only when the surviving supported evidence can
  sustain an honest answer. The result is marked `partial` with the consequential
  gaps named.
- Never rerun a completed workflow solely to recover a long result from chat output.
