# Evidence and reporting

Use this reference to keep deep research rigorous without turning every report into
an institutional audit. The evidence rules are fixed; the report's structure and
voice are reader-dependent.

## Briefing and source planning

Topic-specific intake questions should change what is researched, not serve as
procedural form-filling. Before assigning source classes to lanes, use the answers
from intake to narrow scope and select evidence:

- **Scope answer** sets the timeframe, jurisdiction, geography, or comparison set
  for every lane, eliminating irrelevant source classes before acquisition begins.
- **Interpretive-lens question** (historical/cultural) targets primary texts,
  scholarly commentary, and reception records rather than news sources.
- **Options/constraints question** (product/market) determines whether technical
  documentation, independent benchmarks, or regulatory filings belong in scope.
- **Jurisdiction/decision question** (disputed/high-stakes) specifies which legal,
  clinical, or policy evidence applies and at what threshold.
- **Communities/period question** (experiential/community) identifies the platforms,
  forums, or archival sources appropriate for lived-experience evidence.
- **Audience and use answer** calibrates evidence depth: a practitioner audience
  may need primary or technical sources where a general audience benefits from
  authoritative synthesis.

Treat intake as a research-design step. If the topic-specific answers do not change
any lane boundary, source class, or evidence standard, the questions were not
specific enough.

## 1. Build claims from evidence

Separate four layers:

1. **Source** — who published the material, when, and under what incentives.
2. **Evidence** — the specific passage, datum, observation, or record.
3. **Claim** — the proposition that the evidence supports.
4. **Inference** — the analyst's interpretation that joins multiple claims.

Record evidence atomically. One evidence record should support one proposition and
name the source URL that can be cited. Do not use an entire document as if every
claim in it had been verified.

For each candidate claim, record:

- stable lane-local claim and evidence IDs;
- claim text;
- evidence IDs;
- importance: `background`, `supporting`, or `conclusion-driving`;
- whether the claim is disputed, surprising, or high-stakes; and
- a short qualification when support is partial.

The reader-facing report must never expose internal IDs.

### Classify claims with the exact enum values

`claim_type` and `verification_triggers` are schema enums. They decide what gets
verified, so an invented value is rejected outright rather than silently changing
behavior:

| `claim_type` | Use for |
| --- | --- |
| `external_fact` | a checkable fact about the world |
| `quantitative` | a number, rate, magnitude, or trend |
| `causal` | an assertion that one thing produced another |
| `comparative` | a ranking or "more/less than" judgment |
| `attributed_report` | what a named source said, did, or experienced |
| `interpretation` | an analytic or scholarly reading of material |

| `verification_triggers` | Use when |
| --- | --- |
| `known_dispute` | credible sources are known to disagree |
| `cross_source_conflict` | this lane's own sources conflict |
| `scope_risk` | the sample, jurisdiction, or period may not match the claim |
| `source_access_uncertain` | the supporting source may not stay reachable |
| `high_stakes` | a wrong answer here carries real cost |

`attributed_report` and `interpretation` are the only types that opt out of
verification, and only when the claim is neither disputed nor conclusion-driving.
Testimony that carries the argument still gets checked. A claim whose type is
absent falls back to the conservative rule — verify it if it is
conclusion-driving or disputed — so a mislabeled claim is never skipped.

### Claim statuses

Verification assigns each claim exactly one status, and the drafter is bound by it:

| Status | Meaning | Drafting |
| --- | --- | --- |
| `verified` | independently checked and supported | may carry a factual conclusion |
| `qualified` | supported within a stated limit | keep the qualification next to the claim |
| `contested` | credible support is unresolved or conflicting | present the disagreement; conclude nothing decisive |
| `refuted` | the evidence does not support it | excluded from the report |
| `attributed` | testimony or interpretation, not externally checkable | attribute explicitly; never evidence of prevalence |
| `single-source` | background resting on one source | cite and attribute; not independently corroborated |
| `unverified` | eligible but not checked | usable with explicit attribution and hedged language |

Not being selected for verification is never a mark of quality. `single-source`
does not outrank `unverified`, and neither may be presented as corroborated.

## 2. Judge source fitness

Use the strongest available source for the claim:

1. primary record, official data, standard, or original research;
2. high-quality scholarly synthesis or authoritative technical documentation;
3. reputable specialist reporting or analysis;
4. contemporaneous reporting, trade material, or institutional commentary;
5. community or anecdotal material for experience and discovery, not universal
   factual claims.

Source class is not a universal ranking. A community post can be the primary source
for what a participant said; it is not primary evidence for broad prevalence.

Assess:

- directness: does the source actually support the proposition?
- authority: does the author or institution have relevant competence?
- independence: are apparently separate sources copying one origin?
- date fitness: is the evidence current enough for the claim?
- scope fitness: does the sample, jurisdiction, population, or period match?
- incentives and missing context;
- access stability and whether the citation is reader-verifiable.

Use discovery pages to locate authoritative evidence. Do not cite a search snippet
when the underlying source is available.

## 3. Acquire proportionally

### Lane ceilings

| Tier | Materially different searches | Fetches |
| --- | ---: | ---: |
| Focused | 4 | 6 |
| Standard | 6 | 8 |
| Extended | 8 | 12 |

Ceilings are upper bounds, not quotas. Stop when:

- every conclusion-driving claim in the lane has adequate direct support;
- another source would merely repeat established background;
- remaining uncertainty is unlikely to change the answer; or
- the writing/audit reserve would be threatened.

No source-count minimum applies. Prefer two independent strong sources for a
disputed conclusion-driving claim when they are realistically available; do not
manufacture independence by citing syndicated copies.

### Failed-acquisition ledger

Keep a compact shared ledger:

```text
canonical URL | failure class | attempts | alternate tried | claim affected
```

Canonicalize away the fragment and known tracking parameters, but **keep the rest
of the query string**: `?report=2019` and `?report=2024` are different documents,
and collapsing them mis-attributes citations.

Under orchestration, a verifier receives its own lane's ledger, which is what
prevents same-source retries. The cross-lane ledger is assembled after the research
pipeline for the follow-up and escalation stages. Working sequentially, keep one
ledger for the whole run.

Terminal failures require zero same-source retries:

- 403/404 or other access denial;
- login, paywall, or robots restriction;
- certificate or protocol failure;
- unsupported or oversized content;
- scanned/no-text content;
- a source that requires prohibited or unavailable tooling.

If the source could change a conclusion, try at most one obvious lawful accessible
equivalent: an author manuscript, official mirror, archived official copy, or another
source reporting the same primary evidence. If that attempt fails, record the gap.

A transient timeout may be retried once. A second timeout is terminal.

Do not:

- repeat an identical query;
- vary only protocol, fragment, query string, mirror endpoint, or download parameter
  to create the appearance of a new attempt;
- let a verifier revisit a URL or access strategy already marked terminal;
- download and shell-convert a document merely because ordinary fetching failed; or
- treat OCR output as strong evidence without checking the relevant passage.

Focused and Standard runs never initiate OCR by default. Extended/high-stakes or
explicitly exhaustive work may use one Browser **or** local conversion/OCR
escalation—not both—only for an irreplaceable source that could change the answer.
Bound the target and question before the attempt. Stop after failure and disclose
the gap.

## 4. Verify selectively

Verification is for:

- conclusion-driving claims;
- disputed or surprising claims;
- claims whose scope, date, or causal language may exceed the evidence;
- conflicts between credible sources; and
- pivotal translations or interpretations.

Background claims with direct authoritative support do not need a wholesale second
research pass.

Every lane holding eligible claims is verified. Verification capacity is not
rationed across lanes and none is held back for a possible follow-up: a report
whose conclusion-driving claims went unchecked is worse than one that costs a few
more bounded calls.

A verifier receives:

- the lane's summary, its selected claims, and only the evidence and source cards
  those claims depend on;
- the path to the lane's evidence dossier, when one exists;
- the lane's failed-acquisition ledger; and
- a small verification acquisition ceiling.

It reads the dossier for depth rather than being handed the whole scout record.

It returns a delta only:

- verdict per claim: `supported`, `qualified`, `unsupported`, or `disputed`;
- approved existing evidence IDs;
- concise qualifications;
- rejected evidence IDs and reason;
- genuinely new sources/evidence; and
- new failed acquisitions or gaps.

It must not echo the scout record or broadly re-fetch every source. Independence
means finding separate support for a material proposition, not mechanically
repeating the same acquisition work.

## 5. Handle disagreement honestly

When credible sources disagree:

1. identify the exact proposition in dispute;
2. compare definitions, dates, jurisdiction, samples, and incentives;
3. distinguish factual conflict from interpretive difference;
4. prefer the source closest to the underlying event or data when appropriate;
5. state what is established, what is probable, and what remains open.

Do not collapse a live disagreement into false certainty. Do not give fringe claims
equal weight merely because they exist.

Calibrate language:

- **Strong:** demonstrates, establishes, directly records.
- **Moderate:** supports, indicates, is consistent with.
- **Tentative:** suggests, may reflect, is plausibly explained by.
- **Unresolved:** evidence is insufficient or credible sources disagree.

## 6. Keep handoffs compact by writing evidence down

Research cost grows when every stage reproduces full source cards, but compressing
evidence into one-line paraphrases starves the drafter. Resolve the tension by
separating depth from transport: write full-fidelity evidence to a file and pass
paths.

### Evidence dossiers

When a session scratchpad is available, each scout writes a dossier alongside its
compact record:

```text
<scratchpad>/deep-research/<run-slug>/lanes/<lane-key>.md
```

The dossier is where depth belongs — full bibliographic metadata, access status,
and generous verbatim quotation with enough surrounding context for a later stage
to judge directness, scope, and date fitness without refetching. Record rejected
leads and the detail behind each failed acquisition too.

Downstream stages receive the path and read what they need. Verifiers add their own
working notes next to the dossier. Nothing large travels through a stage's return
value.

This is additive. When no scratchpad is available or a worker cannot write one,
every stage falls back to the inline compact records below and the run still
completes; say so in the gaps rather than failing.

Never write a deliverable to the scratchpad — it is throwaway by design. The report
is written into the project at the end.

### Scout record

- lane ID and a short synthesis;
- compact source cards: local ID, title, URL, publisher, date, type;
- atomic evidence: local ID, claim, source ID, short quote/paraphrase;
- candidate claims with importance and dispute flags;
- failed acquisitions; and
- unresolved gaps.

### Verification delta

- claim verdicts and approved evidence IDs;
- qualifications and rejections;
- only new sources/evidence;
- new failures and remaining gaps; and
- a short verifier synthesis.

### Coverage input

- one short supported synthesis per lane;
- a claim index with status and source IDs but **no evidence excerpts**;
- source-class coverage;
- dossier paths for anything that needs a closer look;
- unresolved gaps and failed-source effects; and
- the inferred target length and report profile needed to decide drafting shape.

Do not pass complete fetched text, search transcripts, duplicated source cards, or
full evidence ledgers into coverage and audit prompts. Pass the path instead.

### Drafting-shape decision

The skill decides the drafting shape; do not ask the user to choose an
implementation detail. A long report must never collapse back to a single drafting
call merely because the stage that was supposed to propose an outline failed to
return one; fall back to a deterministic outline and let the assembly pass turn it
into reader-facing sections.

Use one drafting agent by default. Use bounded section drafting when:

- the inferred target is roughly 5,000 words or more; or
- coverage analysis identifies at least two genuinely independent sections with
  distinct claim sets whose parallel drafting improves clarity or context fit.

Do not use section fan-out merely because research had multiple lanes. Lanes are
evidence boundaries; report sections are reader-facing argument boundaries.

When section drafting is warranted:

1. coverage returns a short outline, purpose, and supported claim IDs per section;
2. draft the bounded sections in parallel using only their assigned claims;
3. assemble the sections into one body, then
4. build the bibliography from the assembled body, not from section-reported
   source lists.

With a scratchpad, each section is written to its own file and the assembly stage
writes `report/body.md`: a title, an opening that answers the question early, the
sections in order with reconciled transitions and normalized voice, and an
evidence-calibrated conclusion. A long report must not be re-emitted as one giant
JSON string; that is the most truncation-prone shape available.

Verify the delivered length against the target with `wc -w` rather than trusting an
impression, and check both directions. A report materially shorter than the confirmed
target gets one bounded expansion pass over its thinnest sections — adding evidence,
mechanism, and concrete detail from the dossiers, never padding — and is then
reassembled. One pass only: a short report that is honest still ships, with the
shortfall disclosed. A report far longer than the target is a defect of a different
kind: it disregarded the brief, so tighten it rather than disclosing it.

## 7. Reader-fit report contract

### Core invariants

Every report must:

- answer the research question early;
- cite material factual claims with descriptive Markdown links;
- distinguish sourced fact from synthesis or inference;
- preserve material disagreement and uncertainty;
- conclude only as strongly as the evidence allows; and
- contain exactly one `## Sources` section with deduplicated sources actually cited
  in the body.

Keep uncertainty next to the affected claim. Consolidate secondary caveats instead
of repeating "the accessible evidence is limited" in every section. Do not expose
research lanes, worker names, evidence IDs, audit verdicts, or retry history unless
the user explicitly asks for methods.

That applies to recorded gaps too. A gap is read by a person, so name the missing
evidence and why it matters — never a lane identifier, a claim or evidence ID, or the
shape of an internal record. "Klibansky and Panofsky's study was reached only at
second hand" is a disclosure; "lane-1/C11's second half needs the reception lanes" is
leaked machinery. A reader-facing note is a short disclosure of what stayed
unresolved, not a transcript of the research ledger.

The user's requested format wins. Otherwise choose the closest profile below.

### Historical, cultural, and humanities

- Open with a clear thesis in natural prose.
- Use a narrative, chronological, or thematic arc suited to the material.
- Write a specific, engaging title and concrete period-appropriate headings.
- Integrate interpretive disputes where they arise.
- Use a brief, naturally titled note on gaps only if it changes how the history
  should be read.
- Do not default to headings named `Executive answer`, `Methods`, or `Limitations`.

For a history of Saturnian magic, for example, prefer headings tied to periods,
texts, and transformations over corporate or methodological labels. Evocative does
not mean sensational: preserve ambiguity between documentary fact, later tradition,
and modern reconstruction.

### Product, market, and policy decision

- Lead with the answer or decision frame.
- Compare options on criteria that matter to the stated audience.
- Explain trade-offs, recommendation, risks, and what would change the choice.
- Use an executive summary when the decision-maker benefits from one.

### Scientific, technical, and high-stakes

- State scope and definitions precisely.
- Explain method, evidence quality, and uncertainty where they affect
  interpretation.
- Distinguish association, mechanism, and causation.
- Give limitations their own section when needed for safe use of the findings.

### Community, trend, and experiential

- Organize around observed patterns and participant voices.
- Identify the sampled community and missing groups.
- Treat anecdotes as experience evidence, not prevalence estimates.
- State representativeness limits once clearly rather than as a disclaimer in every
  paragraph.

### Voice

Match the user's register and the subject's texture. Prefer concrete nouns and
active sentences. Avoid generic institutional phrases, audit-shaped headings, and
inflated abstractions when plain language is more accurate.

Do not:

- imitate a living author;
- manufacture scenes, quotations, or emotional certainty;
- sensationalize cultural or religious material;
- trade factual precision for color; or
- force all discovered sources into the prose.

## 8. Cite economically and immediately

Use descriptive Markdown links near the supported claim:

```markdown
The [official release notes](https://example.com/release) date the change to May.
```

Avoid:

- bare URLs in body prose;
- one citation at the end of a paragraph containing several unrelated claims;
- bibliography entries never cited in the body;
- multiple citations that all derive from one origin; and
- citation density that obscures the argument when one stronger source suffices.

The deterministic bibliography builder extracts URLs from the final body, resolves
them against the canonical registry, and creates `## Sources`. Writer-reported
source lists are advisory only.

Link extraction has to be exact, because a mis-parsed URL looks like an unsupported
citation. It must honor balanced parentheses in a destination — Wikipedia's
`Saturn_(mythology)` style is everywhere in historical research — skip images,
inline code, and fenced blocks, tolerate an optional link title, and ignore
anchors and mail links rather than reporting them as unknown sources.

## 9. Audit proportionally

Choose the audit mode deliberately rather than leaving it implicit: `combined` for
Focused and ordinary Standard work, `dual` for Extended or high-stakes work, and
`deterministic` only when no judgment-based audit is warranted.

One combined evidence/editorial audit is enough for Focused and ordinary Standard
reports. It checks:

- whether material claims are supported by cited registry sources;
- whether claim strength matches the evidence;
- whether disagreement and scope limits are represented fairly;
- whether the answer addresses the settled brief; and
- whether structure, headings, and voice fit the report profile.

Use two independent audits for Extended or high-stakes work: one evidence-focused
and one reader/editorial-focused. Revise only when an audit identifies a material
issue.

An independent closure review is justified only when high-stakes revision leaves an
unresolved critical or major evidence issue. Deterministic checks—not agents—own:

- unknown citation URLs;
- duplicate citation URLs or headings;
- missing cited sources;
- uncited bibliography entries;
- empty sections;
- malformed Markdown output; and
- missing/empty output files.

If evidence cannot support a material claim, remove or narrow the claim and name
the gap. More process is not a substitute for better evidence.

A residual defect never justifies throwing the report away. Deliver it as a
supported partial result with the unresolved issues named, and let the reader see
what is unsettled. Discarding a finished report because a citation would not
resolve serves nobody.
