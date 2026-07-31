# =============================================================================
# examples/parallel-code-review.py — a gigacode workflow authored entirely by
# an LLM, published verbatim.
#
# Authoring prompt (verbatim; this was the COMPLETE prompt):
#
#     write a gigacode workflow for parallel code review of this repo and
#     execute it. Use deepseek-v4-flash for all the workflow sub-agents
#     (deepseek provider, via model_override).
#
# Date:            2026-07-31
# Authoring model: openai_chatgpt/gpt-5.6-sol (thinking effort "medium")
# Sub-agents:      deepseek/deepseek-v4-flash (effort "high"), pinned by the
#                  script itself via a complete atomic model_override on every
#                  agent() call
# Harness:         kolega-code
# Target repo:     kolega-zapier (a small public Zapier Platform integration,
#                  ~1,000 LOC of CommonJS)
#
# Protocol: one authoring session; publish what it produces, unedited. This
# was the first and only session run with this prompt and configuration, and
# it succeeded end-to-end on that first attempt.
#
# What the session did, unassisted: explored the repo; noticed the working
# tree had uncommitted changes and explicitly defined the review boundary as
# the full working-tree snapshot at HEAD d34e4d1c; authored this script;
# executed it via run_workflow(script_path=..., args={repository, head,
# boundary}) with NO token budget — following the harness's sizing guidance
# ("when unsure, omit it") — and verified afterwards that every workflow call
# ran on deepseek-v4-flash.
#
# Execution record of THESE EXACT BYTES (hash-verified identical to the
# persisted run script):
#   run 9773faa8 — status COMPLETED. 18 agent calls, all completed: 6 parallel
#   discovery specialists, 11 adversarial challengers (one per candidate
#   finding), 1 synthesis gate. Every call ran on deepseek/deepseek-v4-flash
#   (effort high) as instructed; all workers were read-only investigation
#   agents. 267,421 output tokens; 645 s workflow duration inside a 14m22s
#   turn including authoring. Zero script-exception drops, zero duplicate
#   labels. The target repo was not modified (the enclosing agent separately
#   ran the repo's own test suite and `zapier validate` — 36 checks passed).
#
# sha256 of the model output below the delimiter:
#   ade0d591d5beb7bc356787236fc132155e3fa36f5f2c361906e772b59ce86c27
#
# Everything below the delimiter is byte-for-byte what the model produced.
# ======================= BEGIN UNEDITED MODEL OUTPUT =========================
meta = {
    "name": "parallel-repository-code-review",
    "description": "Review the complete repository snapshot with parallel specialists, adversarial challenges, and a final synthesis gate",
    "phases": [
        {"title": "Discovery"},
        {"title": "Challenge"},
        {"title": "Synthesis"},
    ],
}

MODEL = {
    "provider": "deepseek",
    "model": "deepseek-v4-flash",
    "effort": "high",
}

FINDINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "category": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "high", "medium", "low"],
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "very high"],
                    },
                    "path": {"type": "string"},
                    "line": {"type": "integer", "minimum": 1},
                    "evidence": {"type": "string"},
                    "impact": {"type": "string"},
                    "remediation": {"type": "string"},
                },
                "required": [
                    "title",
                    "category",
                    "severity",
                    "confidence",
                    "path",
                    "line",
                    "evidence",
                    "impact",
                    "remediation",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["findings"],
    "additionalProperties": False,
}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "confirmed": {"type": "boolean"},
        "reason": {"type": "string"},
        "path": {"type": "string"},
        "line": {"type": "integer", "minimum": 1},
        "evidence": {"type": "string"},
    },
    "required": ["confirmed", "reason", "path", "line", "evidence"],
    "additionalProperties": False,
}

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "boundary": {"type": "string"},
        "change_outline": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "high", "medium", "low"],
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "very high"],
                    },
                    "path": {"type": "string"},
                    "line": {"type": "integer", "minimum": 1},
                    "evidence": {"type": "string"},
                    "impact": {"type": "string"},
                    "remediation": {"type": "string"},
                },
                "required": [
                    "title",
                    "severity",
                    "confidence",
                    "path",
                    "line",
                    "evidence",
                    "impact",
                    "remediation",
                ],
                "additionalProperties": False,
            },
        },
        "coverage_notes": {"type": "string"},
    },
    "required": ["boundary", "change_outline", "findings", "coverage_notes"],
    "additionalProperties": False,
}

REVIEWERS = [
    {
        "key": "contracts",
        "focus": "Public HTTP/API contracts, endpoint construction, request/response schemas, identifiers, pagination, and externally visible compatibility.",
    },
    {
        "key": "auth-security",
        "focus": "Authentication, authorization assumptions, secret handling, input trust boundaries, injection risks, and sensitive-data exposure.",
    },
    {
        "key": "actions-dataflow",
        "focus": "Zapier triggers, creates, and searches: control flow, field mappings, state transitions, polling/deduplication behavior, and runtime failures.",
    },
    {
        "key": "sdk-runtime",
        "focus": "Zapier SDK/runtime compatibility, async behavior, error handling, resource lifetime, dependency/configuration correctness, and upgrade hazards.",
    },
    {
        "key": "tests-release",
        "focus": "Tests, package metadata, build/release configuration, documentation promises, and gaps that reveal concrete production defects. Do not report missing tests by itself.",
    },
    {
        "key": "bug-hunt",
        "focus": "Independent end-to-end bug hunt across the whole repository, emphasizing reproducible failures, wrong results, data loss, and broken integrations.",
    },
]

boundary = (
    "Review the complete current working-tree snapshot of this repository, rooted at "
    "HEAD d34e4d1c3ac841b36e4cd63fc29e0e5d4d70ca71. Include tracked files and the "
    "untracked CHANGELOG.md. Existing uncommitted edits are part of the snapshot. "
    "The review is repository-wide, not limited to the diff."
)

acceptance = (
    "Report only concrete defects with practical impact: wrong results, runtime failure, "
    "security or authority exposure, compatibility breakage, state/data loss, materially "
    "leaked resources, or a directly broken public contract. Exclude style, naming, cleanup, "
    "hypothetical edge cases without a supported execution path, generic advice, and missing "
    "tests alone. Every finding must cite an exact repository path and line and explain the "
    "execution path. Inspect source and tests directly. Do not claim to have verified live "
    "external services or web documentation because investigation agents have no web access."
)

phase("Discovery")
reviews = await parallel([
    lambda reviewer=reviewer: agent(
        f"""You are the {reviewer['key']} specialist in a parallel repository code review.

Boundary: {boundary}
Your focus: {reviewer['focus']}

{acceptance}

Search the repository, follow callers and tests as needed, and return zero or more
high-confidence candidates matching the schema. Avoid duplicating unrelated concerns
outside your focus.""",
        label=f"discover:{reviewer['key']}",
        phase="Discovery",
        schema=FINDINGS_SCHEMA,
        model_override=MODEL,
        agent_type="investigation",
    )
    for reviewer in REVIEWERS
])

candidates = []
for review_index, review in enumerate(reviews):
    if not review:
        continue
    for finding_index, finding in enumerate(review.get("findings") or []):
        item = dict(finding)
        item["candidate_id"] = f"{REVIEWERS[review_index]['key']}:{finding_index + 1}"
        candidates.append(item)

log(f"Discovery produced {len(candidates)} candidates.")

phase("Challenge")
verdicts = await parallel([
    lambda candidate=candidate: agent(
        f"""You are an adversarial challenger in a repository code review.

Boundary: {boundary}

Candidate:
{candidate}

Attempt to DISPROVE this candidate. Inspect the cited line, relevant definitions,
callers, tests, package/runtime behavior, and any compensating code. Reject it if it
is precluded by the actual execution path, merely stylistic, speculative, duplicated
by a fix elsewhere, or not practically impactful. Confirm it only when repository
evidence supports the failure with high confidence. You cannot browse the web, so do
not claim external verification. Return the required structured verdict.""",
        label=f"challenge:{candidate['candidate_id']}",
        phase="Challenge",
        schema=VERDICT_SCHEMA,
        model_override=MODEL,
        agent_type="investigation",
    )
    for candidate in candidates
])

confirmed = []
for index, verdict in enumerate(verdicts):
    if verdict and verdict.get("confirmed"):
        confirmed.append({
            "candidate": candidates[index],
            "verdict": verdict,
        })

log(f"Challenge confirmed {len(confirmed)} of {len(candidates)} candidates.")

phase("Synthesis")
final = await agent(
    f"""You are the final synthesis gate for a parallel repository code review.

Boundary: {boundary}

Confirmed candidate/verdict pairs:
{confirmed}

Produce the final structured report. Re-check the repository evidence for every item.
Remove duplicate symptoms of the same root cause, unsupported claims, and anything below
the concrete-impact threshold. Preserve only findings whose exact path/line is valid.
Order findings by severity (critical, high, medium, low). Keep evidence specific and
remediation bounded. Coverage notes must state that this was a static repository review,
that the current mutable working-tree snapshot was included, and that no live external
service contract was verified by the investigation agents.""",
    label="synthesis:final",
    phase="Synthesis",
    schema=FINAL_SCHEMA,
    model_override=MODEL,
    agent_type="investigation",
)

return {
    "model": MODEL,
    "reviewer_count": len(REVIEWERS),
    "candidate_count": len(candidates),
    "confirmed_before_synthesis": len(confirmed),
    "report": final,
}
