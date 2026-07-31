meta = {
    "name": "deep-research",
    "description": "Research disjoint lanes, adaptively verify claims, and produce a compact cited report",
    "max_agent_depth": 1,
    "phases": [
        {"title": "Research"},
        {"title": "Verify"},
        {"title": "Coverage"},
        {"title": "Draft"},
        {"title": "Audit"},
        {"title": "Revise"},
    ],
}

# Claim-type classification. The enum is enforced by the schema so scouts cannot
# invent a value that silently changes verification eligibility.
CLAIM_TYPES = [
    "attributed_report",
    "external_fact",
    "quantitative",
    "causal",
    "comparative",
    "interpretation",
]
VERIFICATION_TRIGGERS = [
    "known_dispute",
    "source_access_uncertain",
    "cross_source_conflict",
    "scope_risk",
    "high_stakes",
]

SCOUT_SCHEMA = {
    "type": "object",
    "properties": {
        "lane_id": {"type": "string"},
        "summary": {"type": "string"},
        "dossier_path": {"type": "string"},
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "publisher": {"type": "string"},
                    "date": {"type": "string"},
                    "source_type": {"type": "string"},
                },
                "required": ["id", "title", "url"],
            },
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "claim": {"type": "string"},
                    "source_id": {"type": "string"},
                    "quote_or_paraphrase": {"type": "string"},
                },
                "required": ["id", "claim", "source_id", "quote_or_paraphrase"],
            },
        },
        "candidate_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    "importance": {"type": "string"},
                    "disputed": {"type": "boolean"},
                    "claim_type": {"type": "string", "enum": CLAIM_TYPES},
                    "verification_triggers": {
                        "type": "array",
                        "items": {"type": "string", "enum": VERIFICATION_TRIGGERS},
                    },
                },
                "required": ["id", "text", "evidence_ids", "importance", "disputed"],
            },
        },
        "failures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "reason": {"type": "string"},
                    "terminal": {"type": "boolean"},
                },
                "required": ["url", "reason", "terminal"],
            },
        },
        "gaps": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "lane_id",
        "summary",
        "sources",
        "evidence",
        "candidate_claims",
        "failures",
        "gaps",
    ],
}

VERIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "lane_id": {"type": "string"},
        "summary": {"type": "string"},
        "notes_path": {"type": "string"},
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["supported", "qualified", "unsupported", "unresolved"],
                    },
                    "approved_evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "qualification": {"type": "string"},
                },
                "required": [
                    "claim_id",
                    "status",
                    "approved_evidence_ids",
                    "qualification",
                ],
            },
        },
        "rejected_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "new_sources": SCOUT_SCHEMA["properties"]["sources"],
        "new_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "claim_id": {"type": "string"},
                    "claim": {"type": "string"},
                    "source_id": {"type": "string"},
                    "quote_or_paraphrase": {"type": "string"},
                },
                "required": [
                    "id",
                    "claim_id",
                    "claim",
                    "source_id",
                    "quote_or_paraphrase",
                ],
            },
        },
        "new_failures": SCOUT_SCHEMA["properties"]["failures"],
        "gaps": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "lane_id",
        "summary",
        "verdicts",
        "rejected_evidence_ids",
        "new_sources",
        "new_evidence",
        "new_failures",
        "gaps",
    ],
}

COVERAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "followup_needed": {"type": "boolean"},
        "decision_affected": {"type": "string"},
        "followup_lane": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "question": {"type": "string"},
                "source_classes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["id", "title", "question", "source_classes"],
        },
        "gaps": {"type": "array", "items": {"type": "string"}},
        "section_drafting_needed": {"type": "boolean"},
        "section_drafting_reason": {"type": "string"},
        "section_outline": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "heading": {"type": "string"},
                    "purpose": {"type": "string"},
                    "claim_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["heading", "purpose", "claim_ids"],
            },
        },
    },
    "required": [
        "summary",
        "followup_needed",
        "decision_affected",
        "gaps",
        "section_drafting_needed",
        "section_drafting_reason",
        "section_outline",
    ],
}

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "body_markdown": {"type": "string"},
        "gaps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "body_markdown", "gaps"],
}

SECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "heading": {"type": "string"},
        "body_markdown": {"type": "string"},
        "used_claim_ids": {"type": "array", "items": {"type": "string"}},
        "gaps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["heading", "body_markdown", "used_claim_ids", "gaps"],
}

# File-mode section drafting: the body lives in the scratchpad, never in the
# workflow return value.
SECTION_FILE_SCHEMA = {
    "type": "object",
    "properties": {
        "heading": {"type": "string"},
        "section_path": {"type": "string"},
        "word_count": {"type": "integer"},
        "cited_urls": {"type": "array", "items": {"type": "string"}},
        "used_claim_ids": {"type": "array", "items": {"type": "string"}},
        "gaps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["heading", "section_path", "word_count", "cited_urls", "used_claim_ids", "gaps"],
}

SEAMS_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "body_path": {"type": "string"},
        "assembled_word_count": {"type": "integer"},
        "cited_urls": {"type": "array", "items": {"type": "string"}},
        "gaps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "body_path", "assembled_word_count", "cited_urls", "gaps"],
}

AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "revision_needed": {"type": "boolean"},
        "material_issues": {"type": "array", "items": {"type": "string"}},
        "minor_issues": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["revision_needed", "material_issues", "minor_issues", "summary"],
}

REVISION_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "body_markdown": {"type": "string"},
        "remaining_material_issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "body_markdown", "remaining_material_issues"],
}

REVISION_FILE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "body_path": {"type": "string"},
        "assembled_word_count": {"type": "integer"},
        "remaining_material_issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "body_path", "assembled_word_count", "remaining_material_issues"],
}

CLOSURE_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "unresolved_material_issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["supported", "unresolved_material_issues"],
}

KNOWN_CLAIM_TYPES = set(CLAIM_TYPES)
# Claim types whose form alone warrants a check, whatever their stated importance.
ALWAYS_CHECK_TYPES = {"quantitative", "causal", "comparative"}
TESTIMONY_TYPES = {"attributed_report", "interpretation"}
RISK_TRIGGERS = {"high_stakes", "known_dispute"}
SCOPE_TRIGGERS = {"cross_source_conflict", "scope_risk", "source_access_uncertain"}

# Stage-plan mode constants
VALID_VERIFICATION_MODES = {"risk_only", "selective", "required"}
VALID_FOLLOWUP_MODES = {"off", "thesis_changing"}
VALID_AUDIT_MODES = {"deterministic", "combined", "dual"}

# Claim statuses. `refuted` is the only status excluded from drafting.
VERDICT_STATUS = {
    "supported": "verified",
    "qualified": "qualified",
    "unsupported": "refuted",
    "unresolved": "contested",
}

# Query parameters that never identify a distinct document.
TRACKING_PARAMS = {"gclid", "fbclid", "ref", "ref_src", "s", "share"}

ROUTE_ROLES = ("discovery", "verification", "synthesis", "audit", "acquisition")

MAX_CLAIMS_PER_VERIFIER_CALL = 8
SHORTFALL_RATIO_NUMERATOR = 4
SHORTFALL_RATIO_DENOMINATOR = 5
# 1.35x the target: past this the report has stopped honoring the confirmed brief.
OVERLENGTH_RATIO_NUMERATOR = 27
OVERLENGTH_RATIO_DENOMINATOR = 20
MAX_EXPANDED_SECTIONS = 3

# Keys the workflow actually acts on. A worker whose structured output degenerates
# can return a dict that satisfies "is a dict" while omitting the decisions the
# stage exists to make; treat that as a stage failure instead of degrading quietly.
COVERAGE_ACTED_ON_KEYS = (
    "followup_needed",
    "section_drafting_needed",
    "section_outline",
    "gaps",
)

READER_FACING_GAPS = (
    "State every gap in reader-facing language: name the missing evidence and why it "
    "matters. Never mention lane identifiers, claim or evidence IDs, worker roles, "
    "or the structure of these records."
)


# ---------------------------------------------------------------------------
# Deterministic text helpers
#
# canonical_url and markdown_urls mirror the authoritative implementations in
# scripts/materialize_report.py. The copies here are advisory only: they feed
# audit hints and telemetry, never a hard failure. tests/test_deep_research.py
# asserts both implementations agree on a shared fixture corpus.
# ---------------------------------------------------------------------------
def canonical_url(value):
    """Return a comparison key that preserves meaningful query parameters."""
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.split("#", 1)[0]
    scheme = ""
    rest = text
    marker = text.find("://")
    if marker >= 0:
        scheme = text[:marker].lower()
        rest = text[marker + 3 :]
    query = ""
    question = rest.find("?")
    if question >= 0:
        query = rest[question + 1 :]
        rest = rest[:question]
    slash = rest.find("/")
    if slash >= 0:
        host = rest[:slash].lower()
        path = rest[slash:]
    else:
        host = rest.lower()
        path = ""
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    kept = []
    for part in query.split("&"):
        if not part:
            continue
        key = part.split("=", 1)[0].lower()
        if key in TRACKING_PARAMS or key.startswith("utm_"):
            continue
        kept.append(part)
    base = (scheme + "://" if scheme else "") + host + path
    if kept:
        return base + "?" + "&".join(sorted(kept))
    return base


def strip_fenced_code(text):
    """Blank out fenced code blocks so their contents cannot look like citations."""
    kept = []
    in_fence = False
    fence = ""
    for line in str(text or "").split("\n"):
        stripped = line.strip()
        marker = ""
        if stripped.startswith("```"):
            marker = "```"
        elif stripped.startswith("~~~"):
            marker = "~~~"
        if marker:
            if in_fence:
                if marker == fence:
                    in_fence = False
                    fence = ""
            else:
                in_fence = True
                fence = marker
            kept.append("")
            continue
        kept.append("" if in_fence else line)
    return "\n".join(kept)


def strip_inline_code(text):
    """Drop inline code spans so `[a](b)` is not read as a citation."""
    source = str(text or "")
    length = len(source)
    kept = []
    index = 0
    while index < length:
        if source[index] == "`":
            run = 0
            while index + run < length and source[index + run] == "`":
                run += 1
            closing = source.find("`" * run, index + run)
            if closing < 0:
                index += run
                continue
            index = closing + run
            continue
        kept.append(source[index])
        index += 1
    return "".join(kept)


def matching_bracket(text, start):
    """Index of the `]` closing the `[` at `start`, or -1."""
    depth = 0
    index = start
    length = len(text)
    while index < length:
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def link_destination(text, start):
    """Parse a Markdown link destination, returning (destination, closing index)."""
    length = len(text)
    index = start
    while index < length and text[index] in " \t\n":
        index += 1
    destination = ""
    if index < length and text[index] == "<":
        end = text.find(">", index + 1)
        if end < 0:
            return "", -1
        destination = text[index + 1 : end]
        index = end + 1
    else:
        depth = 0
        chars = []
        while index < length:
            char = text[index]
            if char in " \t\n":
                break
            if char == "(":
                depth += 1
            elif char == ")":
                if depth == 0:
                    break
                depth -= 1
            chars.append(char)
            index += 1
        destination = "".join(chars)
    while index < length and text[index] in " \t\n":
        index += 1
    if index < length and text[index] in ('"', "'", "("):
        closer = ")" if text[index] == "(" else text[index]
        end = text.find(closer, index + 1)
        if end < 0:
            return "", -1
        index = end + 1
        while index < length and text[index] in " \t\n":
            index += 1
    if index < length and text[index] == ")":
        return destination.strip(), index
    return "", -1


def markdown_urls(markdown):
    """Extract http(s) destinations from Markdown inline links, in order.

    Skips fenced blocks, inline code, and images; honors balanced parentheses in
    the destination and an optional link title; ignores non-http destinations.
    """
    text = strip_inline_code(strip_fenced_code(markdown))
    length = len(text)
    urls = []
    index = 0
    while index < length:
        if text[index] != "[":
            index += 1
            continue
        is_image = index > 0 and text[index - 1] == "!"
        label_end = matching_bracket(text, index)
        if label_end < 0 or label_end + 1 >= length or text[label_end + 1] != "(":
            index += 1
            continue
        destination, close = link_destination(text, label_end + 2)
        if close < 0:
            index += 1
            continue
        if not is_image and destination:
            lowered = destination.lower()
            if lowered.startswith("http://") or lowered.startswith("https://"):
                if destination not in urls:
                    urls.append(destination)
        index = close + 1
    return urls


def slugify(value, fallback):
    """Deterministic filesystem-safe slug (no regex, no randomness)."""
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789"
    chars = []
    for char in str(value or "").strip().lower():
        if char in allowed:
            chars.append(char)
        elif chars and chars[-1] != "-":
            chars.append("-")
    slug = "".join(chars).strip("-")
    slug = slug[:48].strip("-")
    return slug or fallback


# ---------------------------------------------------------------------------
# Routing and dispatch
# ---------------------------------------------------------------------------
def route_issues(candidate_routes):
    """Validation errors for `args.routes`; malformed routes are never ignored."""
    issues = []
    if not isinstance(candidate_routes, dict):
        return ["routes must be an object"]
    for role in sorted(candidate_routes.keys()):
        if role not in ROUTE_ROLES:
            issues.append("routes." + str(role) + " is not a recognized stage role")
            continue
        value = candidate_routes[role]
        if not isinstance(value, dict):
            issues.append("routes." + role + " must be a complete route object")
            continue
        if set(value.keys()) != {"provider", "model", "effort"}:
            issues.append(
                "routes." + role + " must contain exactly provider, model, and effort"
            )
            continue
        if not str(value.get("provider") or "").strip() or not str(value.get("model") or "").strip():
            issues.append("routes." + role + " requires a non-empty provider and model")
    return issues


def route_for(role):
    value = routes.get(role)
    if not isinstance(value, dict):
        return None
    return value


async def dispatch(prompt, label, phase_name, schema, role, agent_type="investigation"):
    override = route_for(role)
    if override is None:
        return await agent(
            prompt,
            label=label,
            phase=phase_name,
            schema=schema,
            agent_type=agent_type,
        )
    return await agent(
        prompt,
        label=label,
        phase=phase_name,
        schema=schema,
        model_override=override,
        agent_type=agent_type,
    )


# ---------------------------------------------------------------------------
# Evidence bookkeeping
# ---------------------------------------------------------------------------
def failure_entries(source):
    entries = []
    for failure in (source or {}).get("failures", []) or []:
        entries.append(
            {
                "url": failure.get("url", ""),
                "reason": failure.get("reason", ""),
                "terminal": bool(failure.get("terminal", False)),
            }
        )
    return entries


def lane_failed_ledger(record):
    """Deterministic per-lane ledger: only this lane's own failures.

    Research and verification run as one pipeline with no barrier, so a
    cross-lane ledger would make prompts depend on completion order and break
    resume caching. The global ledger is assembled after the pipeline for the
    follow-up and escalation stages.
    """
    ledger = {}
    scout = (record or {}).get("scout") or {}
    for failure in failure_entries(scout):
        key = canonical_url(failure["url"])
        if key and key not in ledger:
            ledger[key] = failure
    return [ledger[key] for key in sorted(ledger.keys())]


def compact_failed_ledger(records):
    ledger = {}
    for record in records:
        scout = (record or {}).get("scout") or {}
        for failure in failure_entries(scout):
            key = canonical_url(failure["url"])
            if key and key not in ledger:
                ledger[key] = failure
        verifier = (record or {}).get("verification") or {}
        for failure in verifier.get("new_failures", []) or []:
            entry = {
                "url": failure.get("url", ""),
                "reason": failure.get("reason", ""),
                "terminal": bool(failure.get("terminal", False)),
            }
            key = canonical_url(entry["url"])
            if key and key not in ledger:
                ledger[key] = entry
    return [ledger[key] for key in sorted(ledger.keys())]


def normalized_claim_type(claim):
    """Lowercased known claim type, or "" when absent or unrecognized."""
    raw = str((claim or {}).get("claim_type", "")).strip().lower()
    return raw if raw in KNOWN_CLAIM_TYPES else ""


def is_eligible_for_verification(claim, mode):
    """Whether a claim should be sent to the verifier under the given mode.

    Unrecognized claim types fail *open* to the conservative importance/dispute
    heuristic, so a mislabeled claim can never silently skip verification. Only
    the two testimony types opt out, and only when nothing else raises them.
    """
    claim_type = normalized_claim_type(claim)
    triggers = set(
        str(trigger).strip().lower() for trigger in (claim.get("verification_triggers") or [])
    )
    importance = str(claim.get("importance", "")).strip().lower()
    disputed = bool(claim.get("disputed", False))
    conclusion_driving = importance == "conclusion-driving"

    if claim_type in TESTIMONY_TYPES and not disputed and not conclusion_driving:
        return False

    if mode == "risk_only":
        return disputed or bool(triggers & RISK_TRIGGERS)

    if not claim_type:
        return conclusion_driving or disputed

    return (
        conclusion_driving
        or disputed
        or claim_type in ALWAYS_CHECK_TYPES
        or bool(triggers & (RISK_TRIGGERS | SCOPE_TRIGGERS))
    )


def claim_priority_key(claim):
    """Lower value = higher verification priority."""
    triggers = set(
        str(trigger).strip().lower() for trigger in (claim.get("verification_triggers") or [])
    )
    importance = str(claim.get("importance", "")).strip().lower()
    claim_type = normalized_claim_type(claim)
    if triggers & RISK_TRIGGERS:
        return 0
    if importance == "conclusion-driving" or claim_type in ALWAYS_CHECK_TYPES:
        return 1
    return 2


def claim_index(claims):
    """Claim list without evidence excerpts, for coverage and audit prompts."""
    return [
        {
            "id": claim.get("id", ""),
            "text": claim.get("text", ""),
            "status": claim.get("status", ""),
            "qualification": claim.get("qualification", ""),
            "source_ids": claim.get("source_ids", []),
        }
        for claim in claims
    ]


def missing_stage_keys(value, required_keys):
    """Required keys a stage result omitted or left null.

    Structured output occasionally degenerates — for example collapsing every
    later field into the first string field — leaving a dict that passes an
    isinstance check but carries none of the decisions the stage was asked for.
    """
    if not isinstance(value, dict):
        return sorted(required_keys)
    return sorted(key for key in required_keys if value.get(key) is None)


def derive_section_outline(records, claims):
    """One reader-facing section per lane that produced citable claims.

    Fallback only. Lanes are evidence boundaries rather than argument boundaries,
    so the assembly stage is told to rename and reorder these headings. It exists
    so that a long report is never forced through a single drafting call merely
    because the coverage stage failed to return a usable outline.
    """
    by_lane = {}
    for claim in claims:
        lane_key = str(claim.get("id", "")).split("/", 1)[0]
        by_lane.setdefault(lane_key, []).append(claim.get("id"))
    outline = []
    for record in records:
        lane_key = record.get("lane_key", "")
        claim_ids = by_lane.get(lane_key) or []
        if not claim_ids:
            continue
        heading = (
            str(record.get("lane_title", "")).strip()
            or str(record.get("lane_id", "")).strip()
            or lane_key
        )
        purpose = str(record.get("lane_question", "")).strip() or (
            "Present the supported evidence gathered for " + heading + "."
        )
        outline.append({"heading": heading, "purpose": purpose, "claim_ids": claim_ids})
    return outline


def new_record(scout, lane_key, lane_id, dossier_path, lane_title="", lane_question=""):
    return {
        "lane_key": lane_key,
        "lane_id": lane_id,
        "lane_title": lane_title,
        "lane_question": lane_question,
        "scout": scout,
        "verification": None,
        "eligible_claim_ids": [],
        "selected_claim_ids": [],
        "deferred_claim_ids": [],
        "verifier_failed": False,
        "verifier_attempted": False,
        "dossier_path": dossier_path,
    }


def budget_allows_optional(extra=0):
    if budget.total is None:
        return True
    return budget.remaining() > writing_reserve + extra


def build_registry(records):
    """Build the source registry and claim list from all research records.

    Sources are keyed by workflow-assigned lane identity, so a follow-up or
    escalation worker cannot cross-wire evidence by reusing a base lane's id.
    Verifier-added sources are admitted only when referenced by evidence approved
    by a valid verdict for a claim selected in that same verifier call.
    """
    raw_sources = []
    source_keys = {}
    for record in records:
        scout = record.get("scout") or {}
        lane_key = record.get("lane_key", "lane")
        for source in scout.get("sources", []):
            key = canonical_url(source.get("url"))
            if key:
                raw_sources.append((key, lane_key, source))
                source_keys[(lane_key, str(source.get("id", "")))] = key
        verifier = record.get("verification") or {}
        selected_claim_ids = set(str(claim_id) for claim_id in record.get("selected_claim_ids", []))
        approved_new_evidence_ids = set()
        verdicts = verifier.get("verdicts", [])
        new_evidence = verifier.get("new_evidence", [])
        for verdict in verdicts:
            if str(verdict.get("status", "")) not in VERDICT_STATUS:
                continue
            verdict_claim_id = str(verdict.get("claim_id", ""))
            if verdict_claim_id not in selected_claim_ids:
                continue
            approved_ids = set(str(eid) for eid in verdict.get("approved_evidence_ids", []))
            for evidence_item in new_evidence:
                evidence_id = str(evidence_item.get("id", ""))
                if (
                    evidence_id in approved_ids
                    and str(evidence_item.get("claim_id", "")) == verdict_claim_id
                ):
                    approved_new_evidence_ids.add(evidence_id)
        new_ev_by_source = {}
        for ev in new_evidence:
            sid = str(ev.get("source_id", ""))
            if sid:
                new_ev_by_source.setdefault(sid, set()).add(str(ev.get("id", "")))
        for source in verifier.get("new_sources", []):
            src_id = str(source.get("id", ""))
            ev_ids_for_source = new_ev_by_source.get(src_id, set())
            if ev_ids_for_source & approved_new_evidence_ids:
                key = canonical_url(source.get("url"))
                if key:
                    raw_sources.append((key, lane_key, source))
                    source_keys[(lane_key, src_id)] = key

    by_url = {}
    for key, lane_key, source in sorted(raw_sources, key=lambda item: (item[0], item[1])):
        if key not in by_url:
            by_url[key] = {
                "title": str(source.get("title", "")).strip() or key,
                "url": str(source.get("url", "")).strip(),
                "publisher": str(source.get("publisher", "")).strip(),
                "date": str(source.get("date", "")).strip(),
                "source_type": str(source.get("source_type", "")).strip(),
            }

    registry = []
    registry_ids = {}
    for index, key in enumerate(sorted(by_url.keys()), 1):
        card = dict(by_url[key])
        card["id"] = "S" + str(index).zfill(3)
        registry.append(card)
        registry_ids[key] = card["id"]

    claims = []
    lane_summaries = []
    all_gaps = []

    for record in records:
        scout = record.get("scout") or {}
        verifier = record.get("verification") or {}
        lane_key = record.get("lane_key", "lane")
        selected_ids = set(str(cid) for cid in (record.get("selected_claim_ids") or []))
        deferred_ids = set(str(cid) for cid in (record.get("deferred_claim_ids") or []))
        verifier_failed = bool(record.get("verifier_failed", False))

        lane_summaries.append(
            {
                "lane_id": lane_key,
                "summary": verifier.get("summary") or scout.get("summary", ""),
                "dossier_path": record.get("dossier_path", ""),
            }
        )
        all_gaps.extend(scout.get("gaps", []))
        all_gaps.extend(verifier.get("gaps", []))

        verdicts = {str(item.get("claim_id", "")): item for item in verifier.get("verdicts", [])}
        evidence = {item.get("id"): item for item in scout.get("evidence", [])}
        for item in verifier.get("new_evidence", []):
            evidence[item.get("id")] = item

        for claim in scout.get("candidate_claims", []):
            claim_id = str(claim.get("id", ""))
            verdict = verdicts.get(claim_id)
            qualification = ""

            if claim_id in selected_ids:
                if verifier_failed or not verdict:
                    status = "unverified"
                else:
                    status = VERDICT_STATUS.get(str(verdict.get("status", "")), "unverified")
                    qualification = str(verdict.get("qualification", ""))
            elif claim_id in deferred_ids:
                status = "unverified"
            elif normalized_claim_type(claim) in TESTIMONY_TYPES:
                status = "attributed"
            else:
                status = "single-source"

            if verdict and verdict.get("approved_evidence_ids"):
                approved_eids = verdict.get("approved_evidence_ids")
            else:
                approved_eids = claim.get("evidence_ids", [])

            source_ids = []
            excerpts = []
            for eid in approved_eids:
                ev_item = evidence.get(eid) or {}
                source_key = source_keys.get((lane_key, str(ev_item.get("source_id", ""))))
                if source_key and source_key in registry_ids:
                    source_ids.append(registry_ids[source_key])
                excerpt = ev_item.get("quote_or_paraphrase")
                if excerpt:
                    excerpts.append(str(excerpt))

            if source_ids:
                claims.append(
                    {
                        "id": lane_key + "/" + claim_id,
                        "text": claim.get("text", ""),
                        "status": status,
                        "qualification": qualification,
                        "source_ids": sorted(set(source_ids)),
                        "evidence": excerpts,
                        "dossier_path": record.get("dossier_path", ""),
                    }
                )

    return (
        registry,
        claims,
        lane_summaries,
        sorted(set(str(gap) for gap in all_gaps if gap)),
    )


def strip_sources(markdown):
    lines = str(markdown or "").strip().splitlines()
    kept = []
    for line in lines:
        if line.strip().lower() == "## sources":
            break
        kept.append(line)
    if kept and kept[0].startswith("# "):
        kept = kept[1:]
    return "\n".join(kept).strip()


def structural_issues(markdown):
    issues = []
    text = str(markdown or "").strip()
    if not text.startswith("# "):
        issues.append("missing level-one title")
    if text.count("\n## Sources\n") != 1:
        issues.append("report must contain exactly one Sources section")
    lines = text.splitlines()
    headings = []
    for index, line in enumerate(lines):
        if line.startswith("## "):
            key = line.strip().lower()
            if key in headings:
                issues.append("duplicate heading: " + line.strip())
            headings.append(key)
            next_index = index + 1
            has_content = False
            while next_index < len(lines) and not lines[next_index].startswith("## "):
                if lines[next_index].strip():
                    has_content = True
                    break
                next_index += 1
            if not has_content:
                issues.append("empty section: " + line.strip())
    return issues


def assemble_report(draft, registry):
    """Assemble an inline draft into a cited report. Issues are advisory."""
    title = str((draft or {}).get("title", "")).strip().lstrip("#").strip()
    body = strip_sources((draft or {}).get("body_markdown", ""))
    by_url = {canonical_url(source.get("url")): source for source in registry}
    cited = []
    unknown = []
    for url in markdown_urls(body):
        key = canonical_url(url)
        if key in by_url:
            if key not in cited:
                cited.append(key)
        else:
            unknown.append(url)
    source_lines = []
    cited_sources = []
    for key in cited:
        source = by_url[key]
        detail = source.get("publisher") or source.get("source_type") or ""
        if source.get("date"):
            detail = (detail + ", " + source.get("date")).strip(", ")
        suffix = " — " + detail if detail else ""
        source_lines.append(
            "- [" + source.get("title", key) + "](" + source.get("url", key) + ")" + suffix
        )
        cited_sources.append(source)
    report = "# " + title + "\n\n" + body + "\n\n## Sources\n\n" + "\n".join(source_lines)
    issues = []
    if not title:
        issues.append("missing report title")
    if not body:
        issues.append("missing report body")
    if not cited_sources:
        issues.append("report body cites no registry source")
    if unknown:
        issues.append("unknown citation URLs: " + repr(unknown))
    issues.extend(structural_issues(report))
    return report.strip() + "\n", cited_sources, issues


def word_count(text):
    return len(str(text or "").split())


# ---------------------------------------------------------------------------
# Telemetry accumulators (initialized before validation so every return path
# can use one summary helper)
# ---------------------------------------------------------------------------
stages_run = []
stages_skipped = []
degraded_stages = []
research_records = []
partial_reasons = []
draft_mode = "single"
section_outline_source = "none"
followups_run = 0
escalations_run = 0
expansion_passes = 0
base_lanes_completed = 0
assembled_words = 0
dossier_count = 0

# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------
validation_errors = []
if not isinstance(args, dict):
    validation_errors.append("args must be an object")
brief = args.get("brief", {}) if isinstance(args, dict) else {}
tier = args.get("tier", "") if isinstance(args, dict) else ""
lanes = args.get("lanes", []) if isinstance(args, dict) else []
acquisition = args.get("acquisition", {}) if isinstance(args, dict) else {}
report_profile = args.get("report_profile", {}) if isinstance(args, dict) else {}
routes = args.get("routes", {}) if isinstance(args, dict) else {}
intake = args.get("intake", {}) if isinstance(args, dict) else {}
workspace = args.get("workspace") if isinstance(args, dict) else None

lane_ranges = {"focused": (2, 2), "standard": (3, 4), "extended": (5, 6)}
tier_caps = {"focused": (4, 6), "standard": (6, 8), "extended": (8, 12)}
if not isinstance(brief, dict) or not str(brief.get("question", "")).strip():
    validation_errors.append("brief.question is required")
if not isinstance(brief, dict) or not str(brief.get("audience", "")).strip():
    validation_errors.append("brief.audience is required")
if not isinstance(brief, dict) or not str(brief.get("scope", "")).strip():
    validation_errors.append("brief.scope is required")
if not isinstance(brief, dict) or not str(brief.get("current_as_of", "")).strip():
    validation_errors.append("brief.current_as_of is required; workers have no clock")
if tier not in lane_ranges:
    validation_errors.append("tier must be focused, standard, or extended")
if not isinstance(lanes, list):
    validation_errors.append("lanes must be an array")
elif tier in lane_ranges:
    minimum, maximum = lane_ranges[tier]
    if len(lanes) < minimum or len(lanes) > maximum:
        validation_errors.append(
            tier
            + " requires "
            + str(minimum)
            + ("-" + str(maximum) if minimum != maximum else "")
            + " lanes"
        )
lane_ids = []
for lane in lanes if isinstance(lanes, list) else []:
    if not isinstance(lane, dict):
        validation_errors.append("every lane must be an object")
        continue
    lane_id = str(lane.get("id", "")).strip()
    if not lane_id or not str(lane.get("question", "")).strip():
        validation_errors.append("every lane requires id and question")
    if lane_id in lane_ids:
        validation_errors.append("lane ids must be unique")
    lane_ids.append(lane_id)

default_searches, default_fetches = tier_caps.get(tier, (0, 0))
searches_per_lane = acquisition.get("searches_per_lane", default_searches)
fetches_per_lane = acquisition.get("fetches_per_lane", default_fetches)
verification_searches = acquisition.get("verification_searches", 2)
verification_fetches = acquisition.get("verification_fetches", 4)
if (
    not isinstance(searches_per_lane, int)
    or searches_per_lane < 0
    or searches_per_lane > default_searches
):
    validation_errors.append("searches_per_lane exceeds the tier ceiling")
if (
    not isinstance(fetches_per_lane, int)
    or fetches_per_lane < 0
    or fetches_per_lane > default_fetches
):
    validation_errors.append("fetches_per_lane exceeds the tier ceiling")
if not isinstance(verification_searches, int) or verification_searches < 0:
    validation_errors.append("verification_searches must be a nonnegative integer")
if not isinstance(verification_fetches, int) or verification_fetches < 0:
    validation_errors.append("verification_fetches must be a nonnegative integer")
length_preset = report_profile.get("length", "") if isinstance(report_profile, dict) else ""
target_words = report_profile.get("target_words") if isinstance(report_profile, dict) else None
length_targets = {"concise": 1500, "standard": 3000, "detailed": 6000}
if length_preset not in ("concise", "standard", "detailed", "long", "custom"):
    validation_errors.append(
        "report_profile.length must be concise, standard, detailed, long, or custom"
    )
if not isinstance(target_words, int) or isinstance(target_words, bool) or target_words < 500:
    validation_errors.append("report_profile.target_words must be an integer of at least 500")
elif length_preset in length_targets and target_words != length_targets[length_preset]:
    validation_errors.append(
        "report_profile.target_words does not match the selected length preset"
    )
elif length_preset == "long" and target_words < 10000:
    validation_errors.append("report_profile.long requires at least 10000 target words")

validation_errors.extend(route_issues(routes))

required_intake_fields = {"length", "audience_use", "scope", "delivery"}
topic_questions_asked = 0
if not isinstance(intake, dict):
    validation_errors.append("intake must be an object")
else:
    intake_mode = intake.get("mode")
    intake_confirmed = intake.get("confirmed")
    resolved_fields = intake.get("resolved_fields")
    asked = intake.get("topic_questions_asked", 0)
    if intake_mode not in ("interactive", "user_directed_defaults"):
        validation_errors.append("intake.mode must be interactive or user_directed_defaults")
    if intake_mode == "interactive" and intake_confirmed is not True:
        validation_errors.append("interactive intake must be confirmed")
    if not isinstance(resolved_fields, list) or not required_intake_fields.issubset(
        set(str(field) for field in resolved_fields)
    ):
        validation_errors.append(
            "intake.resolved_fields must include length, audience_use, scope, and delivery"
        )
    # Caller-asserted and unverifiable: recorded for telemetry, never a hard gate.
    if isinstance(asked, int) and not isinstance(asked, bool) and asked >= 0:
        topic_questions_asked = asked
    else:
        validation_errors.append("intake.topic_questions_asked must be a nonnegative integer")

# Scratchpad workspace (optional). Absent means inline-only operation.
scratchpad_root = ""
run_slug = ""
if workspace is not None:
    if not isinstance(workspace, dict):
        validation_errors.append("workspace must be an object")
    else:
        scratchpad_dir = str(workspace.get("scratchpad_dir", "")).strip()
        run_slug = str(workspace.get("run_slug", "")).strip()
        absolute = scratchpad_dir.startswith("/") or (
            len(scratchpad_dir) > 2
            and scratchpad_dir[1] == ":"
            and scratchpad_dir[2] in ("/", "\\")
        )
        if not scratchpad_dir or not absolute:
            validation_errors.append("workspace.scratchpad_dir must be an absolute path")
        slug_allowed = "abcdefghijklmnopqrstuvwxyz0123456789-"
        slug_valid = (
            1 <= len(run_slug) <= 64
            and run_slug[0] in "abcdefghijklmnopqrstuvwxyz0123456789"
            and all(char in slug_allowed for char in run_slug)
        )
        if not slug_valid:
            validation_errors.append(
                "workspace.run_slug must be 1-64 lowercase alphanumeric or hyphen characters "
                "starting with a letter or digit"
            )
        if absolute and slug_valid:
            scratchpad_root = scratchpad_dir.rstrip("/") + "/deep-research/" + run_slug

use_workspace = bool(scratchpad_root)

# The writing reserve must cover one full draft plus one full revision.
required_reserve = 18000
if isinstance(target_words, int) and not isinstance(target_words, bool) and target_words > 0:
    required_reserve = max(18000, ((target_words * 14) // 10) * 2)
writing_reserve = args.get("writing_reserve_tokens", required_reserve) if isinstance(args, dict) else required_reserve
if not isinstance(writing_reserve, int) or isinstance(writing_reserve, bool) or writing_reserve < 0:
    validation_errors.append("writing_reserve_tokens must be a nonnegative integer")
    writing_reserve = required_reserve
elif writing_reserve < required_reserve:
    validation_errors.append(
        "writing_reserve_tokens must be at least " + str(required_reserve) + " for this target length"
    )
if isinstance(args, dict) and "research_batch_size" in args:
    validation_errors.append(
        "research_batch_size is no longer supported; lanes run as one pipeline and "
        "concurrency is capped by the runtime"
    )

# ---------------------------------------------------------------------------
# Stage-plan parsing (optional; defaults: selective + thesis_changing + combined)
# ---------------------------------------------------------------------------
stage_plan_arg = args.get("stage_plan", {}) if isinstance(args, dict) else {}
if not isinstance(stage_plan_arg, dict):
    stage_plan_arg = {}

verification_mode = str(stage_plan_arg.get("verification", "selective")).strip() or "selective"
followup_mode = str(stage_plan_arg.get("followup", "thesis_changing")).strip() or "thesis_changing"
audit_mode = str(stage_plan_arg.get("audit", "combined")).strip() or "combined"
plan_reason = str(stage_plan_arg.get("reason", "")).strip()

if verification_mode not in VALID_VERIFICATION_MODES:
    validation_errors.append("stage_plan.verification must be risk_only, selective, or required")
    verification_mode = "selective"
if followup_mode not in VALID_FOLLOWUP_MODES:
    validation_errors.append("stage_plan.followup must be off or thesis_changing")
    followup_mode = "thesis_changing"
if audit_mode not in VALID_AUDIT_MODES:
    validation_errors.append("stage_plan.audit must be deterministic, combined, or dual")
    audit_mode = "combined"

resolved_plan = {
    "verification": verification_mode,
    "followup": followup_mode,
    "audit": audit_mode,
    "reason": plan_reason,
}

high_stakes = bool(brief.get("high_stakes", False)) if isinstance(brief, dict) else False


def make_summary():
    """One authoritative telemetry snapshot, used by every return path."""
    eligible = sum(len(r.get("eligible_claim_ids", [])) for r in research_records)
    selected = sum(len(r.get("selected_claim_ids", [])) for r in research_records)
    deferred = sum(len(r.get("deferred_claim_ids", [])) for r in research_records)
    verifier_calls = sum(1 for r in research_records if r.get("verifier_attempted"))
    skipped_no_eligible = sum(1 for r in research_records if not r.get("eligible_claim_ids"))
    verdict_counts = {}
    for record in research_records:
        for verdict in (record.get("verification") or {}).get("verdicts", []):
            status = str(verdict.get("status", "unknown"))
            verdict_counts[status] = verdict_counts.get(status, 0) + 1
    return {
        "stage_plan": resolved_plan,
        "stages_run": stages_run,
        "stages_skipped": stages_skipped,
        "degraded_stages": degraded_stages,
        "section_outline_source": section_outline_source,
        "eligible_claims": eligible,
        "selected_claims": selected,
        "deferred_claims": deferred,
        "verifier_calls": verifier_calls,
        "lanes_skipped_no_eligible": skipped_no_eligible,
        "verifier_verdict_counts": verdict_counts,
        "tier": tier or "unknown",
        "base_lanes_requested": len(lanes) if isinstance(lanes, list) else 0,
        "base_lanes_completed": base_lanes_completed,
        "followups_run": followups_run,
        "escalations_run": escalations_run,
        "draft_mode": draft_mode,
        "topic_questions_asked": topic_questions_asked,
        "target_words": target_words if isinstance(target_words, int) else 0,
        "assembled_word_count": assembled_words,
        "expansion_passes": expansion_passes,
        "dossiers": dossier_count,
        "workspace_enabled": use_workspace,
    }


if validation_errors:
    return {
        "status": "failed",
        "report_markdown": "",
        "report_plan": None,
        "cited_sources": [],
        "source_registry": [],
        "gaps": validation_errors,
        "run_summary": make_summary(),
    }


# ---------------------------------------------------------------------------
# Scratchpad paths (deterministic; derived only from run_slug and lane identity)
# ---------------------------------------------------------------------------
def lane_dossier_path(lane_key):
    if not use_workspace:
        return ""
    return scratchpad_root + "/lanes/" + lane_key + ".md"


def lane_notes_path(lane_key):
    if not use_workspace:
        return ""
    return scratchpad_root + "/lanes/" + lane_key + ".verify.md"


def section_file_path(index, heading):
    return (
        scratchpad_root
        + "/sections/"
        + str(index + 1).zfill(2)
        + "-"
        + slugify(heading, "section")
        + ".md"
    )


REPORT_BODY_PATH = scratchpad_root + "/report/body.md" if use_workspace else ""

WRITE_INSTRUCTIONS = (
    "Write the file with a shell heredoc so nothing is interpolated, for example: "
    "mkdir -p <parent directory> && cat > <path> <<'DOSSIER'\\n...content...\\nDOSSIER\\n"
    "Use exec_command for this; do not assume any other write tool is available. "
    "If the write fails, continue and return your compact record without the path."
)

READ_INSTRUCTIONS = (
    "Read the referenced files with read_file_section (preferred) or read_entire_file "
    "using the absolute paths given. If a file is missing or unreadable, continue from "
    "the inline records below and note the degradation in your gaps."
)


def dossier_instruction(lane_key):
    if not use_workspace:
        return ""
    return (
        "\n\nEVIDENCE DOSSIER (required): before returning, write your full-fidelity "
        "working notes to "
        + lane_dossier_path(lane_key)
        + " and set dossier_path to that path. The dossier is where depth belongs: for "
        "every source record full bibliographic metadata, access status, and generous "
        "verbatim quotations with enough surrounding context that a later stage can judge "
        "directness and scope without refetching. Also record leads you rejected and why, "
        "and the detail behind every failed acquisition. "
        + WRITE_INSTRUCTIONS
    )


def research_prompt(lane, lane_key, searches, fetches, special_instruction=""):
    return (
        "Research only this disjoint lane for the settled brief.\n\n"
        "BRIEF: "
        + repr(brief)
        + "\nLANE: "
        + repr(lane)
        + "\nREPORT PROFILE: "
        + repr(report_profile)
        + "\n\nACQUISITION CEILING: at most "
        + str(searches)
        + " materially different searches and "
        + str(fetches)
        + " fetches. These are ceilings, not quotas; stop as soon as material "
        "claims are adequately supported. Never repeat a query. Retry a transient "
        "timeout once only. Treat access denial, 403/404, login/paywall, robots, "
        "certificate failure, unsupported/oversized content, and scanned/no-text "
        "content as terminal for that URL. Do not vary URL syntax to retry. Try at "
        "most one obvious accessible equivalent for a conclusion-driving source. "
        "Do not use browser automation, downloading, conversion, or OCR. "
        + special_instruction
        + "\n\nReturn a compact record only. Use atomic evidence and lane-local IDs. "
        "Do not include fetched full text, search transcripts, prose bibliography, "
        "or process narration."
        "\n\nClassify each candidate claim with claim_type (one of "
        + ", ".join(CLAIM_TYPES)
        + ") and verification_triggers (any of "
        + ", ".join(VERIFICATION_TRIGGERS)
        + ") where applicable. Use only these exact values; an unrecognized value is "
        "rejected by the schema.\n\n"
        + READER_FACING_GAPS
        + dossier_instruction(lane_key)
    )


def verification_prompt(record, selected_claims, failed_ledger, extra=""):
    """Verifier prompt carrying only what the verifier needs, plus the dossier path."""
    scout = record.get("scout") or {}
    lane_key = record.get("lane_key", "lane")
    selected_ids = set(str(claim.get("id", "")) for claim in selected_claims)
    wanted_evidence_ids = set()
    for claim in selected_claims:
        for eid in claim.get("evidence_ids", []) or []:
            wanted_evidence_ids.add(str(eid))
    evidence_for_claims = [
        item
        for item in scout.get("evidence", [])
        if str(item.get("id", "")) in wanted_evidence_ids
    ]
    wanted_source_ids = set(str(item.get("source_id", "")) for item in evidence_for_claims)
    sources_for_claims = [
        source
        for source in scout.get("sources", [])
        if str(source.get("id", "")) in wanted_source_ids
    ]
    dossier = record.get("dossier_path", "")
    dossier_block = ""
    if dossier:
        dossier_block = (
            "\nLANE DOSSIER: "
            + dossier
            + "\n"
            + READ_INSTRUCTIONS
            + " Read the dossier sections covering the selected claims before judging "
            "directness, scope, and date fitness."
        )
        if use_workspace:
            dossier_block += (
                " Write your own working notes to "
                + lane_notes_path(lane_key)
                + " and set notes_path. "
                + WRITE_INSTRUCTIONS
            )
    return (
        "Verify only the selected candidate claims listed below. "
        "Do not re-research background claims or broadly re-fetch the lane's "
        "sources. Return a delta; never echo the scout record.\n\nBRIEF: "
        + repr(brief)
        + "\nLANE: "
        + lane_key
        + "\nLANE SUMMARY: "
        + repr(str(scout.get("summary", "")))
        + "\nSELECTED CLAIMS: "
        + repr(selected_claims)
        + "\nEVIDENCE FOR SELECTED CLAIMS: "
        + repr(evidence_for_claims)
        + "\nSOURCES FOR SELECTED CLAIMS: "
        + repr(sources_for_claims)
        + "\nKNOWN FAILED ACQUISITIONS: "
        + repr(failed_ledger)
        + dossier_block
        + "\n\nDo not retry any terminal URL or equivalent access strategy in the "
        "failed ledger. Use at most "
        + str(verification_searches)
        + " materially different searches and "
        + str(verification_fetches)
        + " fetches. Never use OCR or conversion. Seek independent support only "
        "where it can affect the answer. Verdict status must be supported, "
        "qualified, unsupported, or unresolved. New records contain only genuinely "
        "new evidence tied to a verdict. Report only claim IDs from SELECTED CLAIMS; "
        "ignore anything else. Selected IDs: "
        + repr(sorted(selected_ids))
        + "\n\n"
        + READER_FACING_GAPS
        + extra
    )


def select_claims(record, mode):
    """Assign eligible / selected / deferred claim IDs for one record."""
    candidate_claims = (record.get("scout") or {}).get("candidate_claims", []) or []
    eligible = [claim for claim in candidate_claims if is_eligible_for_verification(claim, mode)]
    record["eligible_claim_ids"] = [str(claim.get("id", "")) for claim in eligible]
    ordered = sorted(eligible, key=claim_priority_key)
    record["selected_claim_ids"] = [
        str(claim.get("id", "")) for claim in ordered[:MAX_CLAIMS_PER_VERIFIER_CALL]
    ]
    record["deferred_claim_ids"] = [
        str(claim.get("id", "")) for claim in ordered[MAX_CLAIMS_PER_VERIFIER_CALL:]
    ]
    return ordered


def selected_claim_objects(record):
    selected = set(record.get("selected_claim_ids", []))
    return [
        claim
        for claim in (record.get("scout") or {}).get("candidate_claims", []) or []
        if str(claim.get("id", "")) in selected
    ]


async def verify_record(record, label, failed_ledger, extra=""):
    """Run one verifier call for a record that has selected claims."""
    record["verifier_attempted"] = True
    verification = await dispatch(
        verification_prompt(record, selected_claim_objects(record), failed_ledger, extra),
        label,
        "Verify",
        VERIFIER_SCHEMA,
        "verification",
    )
    if verification is None:
        record["verifier_failed"] = True
        return False
    record["verification"] = verification
    return True


# ---------------------------------------------------------------------------
# Research and verification: one pipeline, no barrier between the stages
# ---------------------------------------------------------------------------
phase("Research")
stages_run.append("Research")

indexed_lanes = [
    {"index": index, "lane": lane, "lane_key": "lane-" + str(index + 1)}
    for index, lane in enumerate(lanes)
]


# Both stages take the full (previous result, original item, index) signature the
# runtime supplies to a pipeline stage.
async def scout_stage(item, original, index):
    lane = item["lane"]
    return await dispatch(
        research_prompt(lane, item["lane_key"], searches_per_lane, fetches_per_lane),
        "scout:" + str(lane.get("id")),
        "Research",
        SCOUT_SCHEMA,
        "discovery",
    )


async def verify_stage(scout, item, index):
    if scout is None:
        return None
    lane_key = item["lane_key"]
    reported_path = str(scout.get("dossier_path", "")).strip()
    record = new_record(
        scout,
        lane_key,
        str(item["lane"].get("id", "")),
        reported_path,
        str(item["lane"].get("title", "")),
        str(item["lane"].get("question", "")),
    )
    select_claims(record, verification_mode)
    if record["selected_claim_ids"] and budget_allows_optional():
        await verify_record(record, "verify:" + str(item["lane"].get("id")), lane_failed_ledger(record))
    elif record["eligible_claim_ids"]:
        record["deferred_claim_ids"] = list(record["eligible_claim_ids"])
        record["selected_claim_ids"] = []
    return record


research_ran = budget_allows_optional()
if research_ran:
    pipeline_records = await pipeline(indexed_lanes, scout_stage, verify_stage)
else:
    partial_reasons.append("research did not start; the writing and audit reserve was exhausted")
    pipeline_records = [None for _ in indexed_lanes]

for item, record in zip(indexed_lanes, pipeline_records):
    if record is None:
        if research_ran:
            partial_reasons.append("research worker failed for " + str(item["lane"].get("id")))
    else:
        research_records.append(record)

if any(record.get("verifier_attempted") for record in research_records):
    stages_run.append("Verify")
else:
    stages_skipped.append("Verify")

base_lanes_completed = len(research_records)

# ---------------------------------------------------------------------------
# Acquisition escalation (optional, one attempt total)
# ---------------------------------------------------------------------------
escalation = args.get("escalation")
allow_escalation = bool(args.get("allow_acquisition_escalation", False))
if (
    allow_escalation
    and (tier == "extended" or high_stakes)
    and isinstance(escalation, dict)
    and escalation.get("kind") in ("browser", "local")
    and escalation.get("target")
    and escalation.get("question")
    and budget_allows_optional()
):
    kind = escalation.get("kind")
    escalation_prompt = (
        "Perform one bounded acquisition escalation for an irreplaceable source. "
        "Make exactly one "
        + str(kind)
        + " attempt against "
        + str(escalation.get("target"))
        + ". Answer only this question: "
        + str(escalation.get("question"))
        + ". Do not try another URL, mirror, method, conversion, browser path, or OCR "
        "engine. Stop after success or failure. Return the same compact source, atomic "
        "evidence, candidate-claim, failure, and gap record used by a research lane. "
        "Do not include full fetched text, a search transcript, or process narration."
        + dossier_instruction("escalation-1")
    )
    esc_scout = await dispatch(
        escalation_prompt,
        "escalation:" + str(kind),
        "Research",
        SCOUT_SCHEMA,
        "acquisition",
        "browser" if kind == "browser" else "investigation",
    )
    escalations_run = 1
    if esc_scout is None:
        partial_reasons.append("the single acquisition escalation failed")
    else:
        esc_record = new_record(
            esc_scout,
            "escalation-1",
            "acquisition-escalation",
            str(esc_scout.get("dossier_path", "")).strip(),
            "The escalated source",
            str(escalation.get("question", "")),
        )
        select_claims(esc_record, verification_mode)
        if esc_record["selected_claim_ids"] and budget_allows_optional():
            esc_ok = await verify_record(
                esc_record,
                "verify:acquisition-escalation",
                compact_failed_ledger(research_records + [esc_record]),
                "\nDo not reacquire the escalation target; assess the returned "
                "evidence and seek ordinary independent support only if essential.",
            )
            if not esc_ok:
                partial_reasons.append("the acquisition escalation could not be verified")
        elif esc_record["eligible_claim_ids"]:
            esc_record["deferred_claim_ids"] = list(esc_record["eligible_claim_ids"])
            esc_record["selected_claim_ids"] = []
        research_records.append(esc_record)

# First registry build (for coverage's lane summaries and claim index)
registry, all_claims, lane_summaries, evidence_gaps = build_registry(research_records)

# ---------------------------------------------------------------------------
# Coverage phase (only when it can change something)
# ---------------------------------------------------------------------------
required_structure = (
    report_profile.get("required_structure", []) if isinstance(report_profile, dict) else []
)
followup_possible = followup_mode == "thesis_changing" and tier != "focused"
run_coverage = (
    followup_possible
    or target_words >= 5000
    or (isinstance(required_structure, list) and len(required_structure) >= 2)
)

coverage = None
if run_coverage and research_records:
    phase("Coverage")
    stages_run.append("Coverage")
    coverage = await dispatch(
        "Assess whether one bounded follow-up could change the thesis or recommendation. "
        "Do not research. Focused work must not request a follow-up. A Standard or "
        "Extended follow-up must identify the exact decision affected and one narrow, "
        "non-overlapping lane. Return followup_needed=false for merely useful context. "
        "Also decide the reader-facing drafting shape. Section drafting is warranted "
        "for a target of roughly 5000 words or more, or for a shorter report only when "
        "at least two genuinely independent sections have distinct supported claim sets. "
        "Research lanes alone are not a reason. Return a compact section outline with "
        "purpose and claim IDs when sections are warranted; otherwise return an empty "
        "outline.\n\n"
        "BRIEF: "
        + repr(brief)
        + "\nTIER: "
        + tier
        + "\nTARGET WORDS: "
        + str(target_words)
        + "\nREPORT PROFILE: "
        + repr(report_profile)
        + "\nSUPPORTED LANE SUMMARIES: "
        + repr(lane_summaries)
        + "\nCLAIM INDEX: "
        + repr(claim_index(all_claims))
        + "\nGAPS: "
        + repr(evidence_gaps)
        + "\nFAILED ACQUISITIONS: "
        + repr(compact_failed_ledger(research_records))
        + ("\n" + READ_INSTRUCTIONS if use_workspace else "")
        + "\n\n"
        + READER_FACING_GAPS
        + "\n\nReturn every field of the schema as its own value. Do not pack later "
        "fields into the summary text; a record missing followup_needed, "
        "section_drafting_needed, section_outline, or gaps is discarded.",
        "coverage",
        "Coverage",
        COVERAGE_SCHEMA,
        "audit",
    )
    coverage_missing = missing_stage_keys(coverage, COVERAGE_ACTED_ON_KEYS) if coverage else []
    if coverage_missing:
        # Loud, not silent: the follow-up and drafting-shape decisions are lost.
        degraded_stages.append("Coverage")
        partial_reasons.append(
            "the coverage stage returned an unusable record (missing "
            + ", ".join(coverage_missing)
            + "), so its follow-up and section decisions were discarded"
        )
        coverage = None
else:
    stages_skipped.append("Coverage")

# ---------------------------------------------------------------------------
# Follow-up lane (conditional on coverage's recommendation)
# ---------------------------------------------------------------------------
if (
    followup_possible
    and coverage
    and coverage.get("followup_needed")
    and str(coverage.get("decision_affected", "")).strip()
    and isinstance(coverage.get("followup_lane"), dict)
    and budget_allows_optional(verification_fetches * 250)
):
    followup_lane = coverage["followup_lane"]
    followup_scout_result = await dispatch(
        research_prompt(
            followup_lane,
            "followup-1",
            searches_per_lane,
            fetches_per_lane,
            "This is the only follow-up. Research only the decision-changing gap: "
            + str(coverage.get("decision_affected")),
        ),
        "followup:scout",
        "Research",
        SCOUT_SCHEMA,
        "discovery",
    )

    if followup_scout_result is not None:
        followup_record = new_record(
            followup_scout_result,
            "followup-1",
            "followup",
            str(followup_scout_result.get("dossier_path", "")).strip(),
            str(followup_lane.get("title", "")),
            str(followup_lane.get("question", "")),
        )
        select_claims(followup_record, verification_mode)
        if followup_record["selected_claim_ids"] and budget_allows_optional():
            await verify_record(
                followup_record,
                "followup:verify",
                compact_failed_ledger(research_records),
                "\nDo not reacquire the follow-up target; assess the returned "
                "evidence and seek ordinary independent support only if essential.",
            )
        elif followup_record["eligible_claim_ids"]:
            followup_record["deferred_claim_ids"] = list(followup_record["eligible_claim_ids"])
            followup_record["selected_claim_ids"] = []
        research_records.append(followup_record)
        followups_run = 1
    else:
        partial_reasons.append("the single decision-changing follow-up failed")

# Final registry build (includes follow-up lane if present)
registry, all_claims, lane_summaries, evidence_gaps = build_registry(research_records)
all_gaps = sorted(set(partial_reasons + evidence_gaps + ((coverage or {}).get("gaps", []))))
dossier_count = sum(1 for record in research_records if record.get("dossier_path"))

# Claims usable for drafting: everything the evidence did not refute.
citable_claims = [claim for claim in all_claims if claim.get("status") != "refuted"]

if not citable_claims or not registry:
    return {
        "status": "failed",
        "report_markdown": "",
        "report_plan": None,
        "cited_sources": [],
        "source_registry": registry,
        "gaps": all_gaps + ["no supported claim set was sufficient to draft a cited report"],
        "run_summary": make_summary(),
    }

# ---------------------------------------------------------------------------
# Draft phase
# ---------------------------------------------------------------------------
STATUS_USE_MATRIX = (
    "Status-use matrix:\n"
    "verified – independently checked; may carry a factual conclusion.\n"
    "qualified – usable with its qualification kept next to the claim.\n"
    "contested – present the disagreement; draw no decisive conclusion.\n"
    "attributed – testimony or interpretation; attribute it explicitly and never "
    "treat it as evidence of prevalence.\n"
    "single-source – background resting on one source; cite and attribute it, and do "
    "not describe it as independently corroborated.\n"
    "unverified – not independently checked; usable with explicit attribution and "
    "hedged language, never as a bare assertion."
)

phase("Draft")
stages_run.append("Draft")
single_draft_prompt = (
    "Write the complete reader-facing report from the supported-claim registry below. "
    "Answer the question early. Match the report profile rather than using a fixed "
    "institutional template. Cite material factual claims with descriptive Markdown "
    "links using only exact URLs from the registry. Use a smaller set of strong "
    "representative citations; do not force every source into the prose. Keep material "
    "uncertainty near the affected claim without repeating boilerplate caveats. Do not "
    "mention workers, lanes, evidence IDs, audits, or retry history. Aim for about "
    + str(target_words)
    + " words. Return title and body only: do not write a Sources section because it "
    "is built deterministically.\n\n"
    "BRIEF: "
    + repr(brief)
    + "\nREPORT PROFILE: "
    + repr(report_profile)
    + "\nSUPPORTED CLAIMS: "
    + repr(citable_claims)
    + "\nSOURCE REGISTRY: "
    + repr(registry)
    + "\nCONSEQUENTIAL GAPS: "
    + repr(all_gaps)
    + ("\n" + READ_INSTRUCTIONS if use_workspace else "")
    + "\n\n"
    + STATUS_USE_MATRIX
)
coverage_outline = (coverage or {}).get("section_outline") or []
section_outline = [
    section
    for section in coverage_outline
    if isinstance(section, dict)
    and str(section.get("heading", "")).strip()
    and str(section.get("purpose", "")).strip()
]
section_decision = bool((coverage or {}).get("section_drafting_needed", False))
section_outline_source = "coverage" if len(section_outline) >= 2 else "none"

# Honor the documented rule: a long report never falls back to one drafting call
# just because coverage failed to supply a usable outline.
if len(section_outline) < 2 and target_words >= 5000:
    derived = derive_section_outline(research_records, citable_claims)
    if len(derived) >= 2:
        section_outline = derived
        section_outline_source = "derived"
        log("coverage supplied no usable section outline; derived one from the lanes")

use_section_drafting = len(section_outline) >= 2 and (target_words >= 5000 or section_decision)
draft_mode = "sections" if use_section_drafting else "single"

draft = None
report_plan = None
report = ""
cited_sources = []
deterministic_issues = []
claims_by_id = {claim.get("id"): claim for claim in citable_claims}


def section_claims(section):
    return [
        claims_by_id[claim_id]
        for claim_id in section.get("claim_ids", [])
        if claim_id in claims_by_id
    ]


def section_prompt(section, section_index, section_target, file_mode):
    if file_mode:
        destination = section_file_path(section_index, section.get("heading", ""))
        tail = (
            "Write the section heading and body to "
            + destination
            + ", set section_path to that path, count the words with `wc -w`, and return "
            "the count plus the exact registry URLs you cited. Do not return the body text. "
            + WRITE_INSTRUCTIONS
        )
    else:
        tail = "Write the section heading and body in body_markdown."
    return (
        "Draft one bounded report section. Use only the assigned supported claims and "
        "exact registry URLs. Write the section heading and body, not a title, "
        "introduction, conclusion, or Sources section. Avoid repeating general context "
        "that belongs elsewhere. Target about "
        + str(section_target)
        + " words.\n\nBRIEF: "
        + repr(brief)
        + "\nREPORT PROFILE: "
        + repr(report_profile)
        + "\nSECTION: "
        + repr(section)
        + "\nASSIGNED SUPPORTED CLAIMS: "
        + repr(section_claims(section))
        + "\nSOURCE REGISTRY: "
        + repr(registry)
        + ("\n" + READ_INSTRUCTIONS if use_workspace else "")
        + "\n\n"
        + STATUS_USE_MATRIX
        + "\n\n"
        + tail
    )


if use_section_drafting:
    section_target = max(500, target_words // len(section_outline))
    file_mode = use_workspace
    section_results = await parallel(
        [
            (
                lambda section=section, section_index=section_index: dispatch(
                    section_prompt(section, section_index, section_target, file_mode),
                    "section-draft:" + str(section_index + 1),
                    "Draft",
                    SECTION_FILE_SCHEMA if file_mode else SECTION_SCHEMA,
                    "synthesis",
                )
            )
            for section_index, section in enumerate(section_outline)
        ]
    )
    section_results = [section for section in section_results if section is not None]

    if len(section_results) < 2:
        partial_reasons.append(
            "section drafting did not produce enough usable sections; used one bounded draft"
        )
        draft_mode = "single"
    elif file_mode:
        usable_sections = [
            section
            for section in section_results
            if str(section.get("section_path", "")).strip()
        ]
        if len(usable_sections) < 2:
            partial_reasons.append(
                "section drafters did not persist enough section files; used one bounded draft"
            )
            draft_mode = "single"
        else:
            seams_prompt = (
                "Assemble the drafted sections into one coherent report file. Read each "
                "section file, then write the complete report body to "
                + REPORT_BODY_PATH
                + " in the order given: a specific level-one title, an opening that "
                "answers the question early, the sections with smooth transitions, and "
                "an evidence-calibrated conclusion. Remove repetition and normalize "
                "voice without flattening subject-specific texture. Preserve only "
                "supported citations and use exact registry URLs. Do not write a Sources "
                "section; it is built deterministically. Then count the words with "
                "`wc -w` and return the count, the title, the body path, and the exact "
                "registry URLs cited.\n\n"
                + READ_INSTRUCTIONS
                + "\n"
                + WRITE_INSTRUCTIONS
                + "\n\nBRIEF: "
                + repr(brief)
                + "\nREPORT PROFILE: "
                + repr(report_profile)
                + "\nTARGET WORDS: "
                + str(target_words)
                + "\nSECTION FILES IN ORDER: "
                + repr(usable_sections)
                + "\nSOURCE REGISTRY: "
                + repr(registry)
                + "\nCONSEQUENTIAL GAPS: "
                + repr(all_gaps)
                + (
                    "\n\nThese section boundaries were derived from research lanes, "
                    "which are evidence boundaries rather than reader-facing argument "
                    "boundaries. Rename, reorder, split, or merge the headings so the "
                    "report reads as one argument, keeping every supported citation."
                    if section_outline_source == "derived"
                    else ""
                )
            )
            seams = await dispatch(
                seams_prompt, "draft-assembly", "Draft", SEAMS_SCHEMA, "synthesis"
            )
            if seams is None or not str(seams.get("body_path", "")).strip():
                partial_reasons.append(
                    "report assembly did not persist a body file; used one bounded draft"
                )
                draft_mode = "single"
            else:
                assembled_words = int(seams.get("assembled_word_count", 0) or 0)
                shortfall = (
                    target_words * SHORTFALL_RATIO_NUMERATOR
                ) // SHORTFALL_RATIO_DENOMINATOR
                if assembled_words < shortfall and budget_allows_optional():
                    shortest = sorted(
                        usable_sections,
                        key=lambda section: int(section.get("word_count", 0) or 0),
                    )[:MAX_EXPANDED_SECTIONS]
                    await parallel(
                        [
                            (
                                lambda section=section, order=order: dispatch(
                                    "The assembled report is materially shorter than the "
                                    "requested length. Expand this one section in place at "
                                    + str(section.get("section_path"))
                                    + " using only its assigned supported claims and the "
                                    "lane dossiers behind them. Add substance, not padding: "
                                    "more evidence, mechanism, and concrete detail. Keep the "
                                    "existing heading, cite only exact registry URLs, and "
                                    "rewrite the same file. Then count the words with "
                                    "`wc -w`.\n\n"
                                    + READ_INSTRUCTIONS
                                    + "\n"
                                    + WRITE_INSTRUCTIONS
                                    + "\n\nBRIEF: "
                                    + repr(brief)
                                    + "\nSECTION: "
                                    + repr(section)
                                    + "\nASSEMBLED WORDS: "
                                    + str(assembled_words)
                                    + "\nTARGET WORDS: "
                                    + str(target_words)
                                    + "\nSOURCE REGISTRY: "
                                    + repr(registry)
                                    + "\n\n"
                                    + STATUS_USE_MATRIX,
                                    "section-expand:" + str(order + 1),
                                    "Draft",
                                    SECTION_FILE_SCHEMA,
                                    "synthesis",
                                )
                            )
                            for order, section in enumerate(shortest)
                        ]
                    )
                    expansion_passes = 1
                    reassembled = await dispatch(
                        seams_prompt
                        + "\n\nThe sections have been expanded since the last assembly. "
                        "Rebuild the body file from the current section files.",
                        "draft-assembly:2",
                        "Draft",
                        SEAMS_SCHEMA,
                        "synthesis",
                    )
                    if reassembled is not None and str(reassembled.get("body_path", "")).strip():
                        seams = reassembled
                        assembled_words = int(seams.get("assembled_word_count", 0) or 0)
                report_plan = {
                    "title": str(seams.get("title", "")).strip(),
                    "body_path": str(seams.get("body_path", "")).strip(),
                    "section_paths": [
                        str(section.get("section_path", "")) for section in usable_sections
                    ],
                    "assembled_word_count": assembled_words,
                }
                all_gaps = sorted(set(all_gaps + [str(g) for g in seams.get("gaps", []) if g]))
    else:
        draft = await dispatch(
            "Synthesize the independently drafted sections into one coherent report. "
            "Write a specific title, an opening that answers the question early, smooth "
            "transitions, and an evidence-calibrated conclusion. Remove repetition and "
            "normalize voice without flattening subject-specific texture. Preserve only "
            "supported citations and use exact registry URLs. Return title and body "
            "without a Sources section.\n\nBRIEF: "
            + repr(brief)
            + "\nREPORT PROFILE: "
            + repr(report_profile)
            + "\nSECTION DRAFTS: "
            + repr(section_results)
            + "\nSUPPORTED CLAIMS: "
            + repr(citable_claims)
            + "\nSOURCE REGISTRY: "
            + repr(registry)
            + "\nCONSEQUENTIAL GAPS: "
            + repr(all_gaps),
            "draft-assembly",
            "Draft",
            DRAFT_SCHEMA,
            "synthesis",
        )
        if draft is None:
            partial_reasons.append("section synthesis failed; used one bounded draft")
            draft_mode = "single"

if report_plan is None and draft is None:
    draft_mode = "single"
    draft = await dispatch(
        single_draft_prompt,
        "draft",
        "Draft",
        DRAFT_SCHEMA,
        "synthesis",
    )

if report_plan is None and draft is None:
    return {
        "status": "failed",
        "report_markdown": "",
        "report_plan": None,
        "cited_sources": [],
        "source_registry": registry,
        "gaps": all_gaps + ["drafting worker failed"],
        "run_summary": make_summary(),
    }

if report_plan is None:
    report, cited_sources, deterministic_issues = assemble_report(draft, registry)
    assembled_words = word_count(strip_sources(report))

# ---------------------------------------------------------------------------
# Audit phase (conditional on audit_mode)
# ---------------------------------------------------------------------------
if audit_mode == "dual":
    audit_roles = ["evidence", "editorial"]
elif audit_mode == "combined":
    audit_roles = ["combined"]
else:
    audit_roles = []

material_issues = list(deterministic_issues)

# Length is part of the confirmed brief, so a large overshoot is a material issue
# the single revision pass can act on. A shortfall is handled by the expansion
# pass and reported as a gap rather than a defect.
overlength_target = (
    target_words * OVERLENGTH_RATIO_NUMERATOR
) // OVERLENGTH_RATIO_DENOMINATOR
if assembled_words and assembled_words > overlength_target:
    material_issues.append(
        "report is far longer than the requested length: "
        + str(assembled_words)
        + " words against a target of about "
        + str(target_words)
        + "; tighten it toward the target without dropping supported substance"
    )

if audit_roles:
    phase("Audit")
    stages_run.append("Audit")
    if report_plan is None:
        report_block = "\nREPORT: " + report
    else:
        report_block = (
            "\nREPORT FILE: "
            + report_plan["body_path"]
            + "\n"
            + READ_INSTRUCTIONS
            + " The Sources section is built deterministically after the audit, so check "
            "that every cited URL appears in the source registry rather than expecting a "
            "bibliography in the file."
        )
    audit_prompt = (
        "Audit this report against the brief, report profile, supported claims, and "
        "source registry. Check support for material claims, calibrated certainty, "
        "fair scope, reader fit, structure, and voice. Be compact. A revision is "
        "needed only for a material issue, not optional polish. "
        "Treat deterministic issues as material. "
        "For experiential and testimony-based reports focus on attribution, "
        "overgeneralization, sample limits, and separation of testimony from "
        "inference — not relitigating whether the reported experience occurred.\n\n"
        "BRIEF: "
        + repr(brief)
        + "\nREPORT PROFILE: "
        + repr(report_profile)
        + report_block
        + "\nCLAIM INDEX: "
        + repr(claim_index(citable_claims))
        + "\nSOURCE REGISTRY: "
        + repr(registry)
        + "\nDETERMINISTIC ISSUES: "
        + repr(deterministic_issues)
    )
    audits = await parallel(
        [
            (
                lambda audit_role=audit_role: dispatch(
                    audit_prompt + "\nAUDIT ROLE: " + audit_role,
                    "audit:" + audit_role,
                    "Audit",
                    AUDIT_SCHEMA,
                    "audit",
                )
            )
            for audit_role in audit_roles
        ]
    )
    audits = [audit for audit in audits if audit is not None]
    for audit in audits:
        material_issues.extend(audit.get("material_issues", []))
    if not audits:
        partial_reasons.append("the report audit failed")
else:
    stages_skipped.append("Audit")
    audits = []

# ---------------------------------------------------------------------------
# Revision (at most one, only when material issues exist)
# ---------------------------------------------------------------------------
remaining_material_issues = []
if material_issues or any(audit.get("revision_needed") for audit in audits):
    phase("Revise")
    stages_run.append("Revise")
    issue_list = repr(sorted(set(str(issue) for issue in material_issues if issue)))
    if report_plan is None:
        revision = await dispatch(
            "Revise the report only to fix the material issues below. Preserve the "
            "reader-fit voice and supported useful detail. Use only exact registry URLs "
            "for citations. Return title and body without a Sources section. If an issue "
            "cannot be fixed from supported evidence, remove or narrow the claim and list "
            "the remaining issue.\n\nBRIEF: "
            + repr(brief)
            + "\nREPORT PROFILE: "
            + repr(report_profile)
            + "\nCURRENT REPORT: "
            + report
            + "\nMATERIAL ISSUES: "
            + issue_list
            + "\nSUPPORTED CLAIMS: "
            + repr(citable_claims)
            + "\nSOURCE REGISTRY: "
            + repr(registry),
            "revision",
            "Revise",
            REVISION_SCHEMA,
            "synthesis",
        )
        if revision is None:
            partial_reasons.append("material revision failed")
            remaining_material_issues = list(material_issues)
        else:
            report, cited_sources, deterministic_issues = assemble_report(revision, registry)
            assembled_words = word_count(strip_sources(report))
            remaining_material_issues = list(revision.get("remaining_material_issues", []))
            remaining_material_issues.extend(deterministic_issues)
    else:
        revision = await dispatch(
            "Revise the report file in place, only to fix the material issues below. "
            "Preserve the reader-fit voice and supported useful detail. Use only exact "
            "registry URLs for citations. Do not add a Sources section. If an issue "
            "cannot be fixed from supported evidence, remove or narrow the claim and "
            "list the remaining issue. Rewrite "
            + report_plan["body_path"]
            + ", then count the words with `wc -w` and return the count.\n\n"
            + READ_INSTRUCTIONS
            + "\n"
            + WRITE_INSTRUCTIONS
            + "\n\nBRIEF: "
            + repr(brief)
            + "\nREPORT PROFILE: "
            + repr(report_profile)
            + "\nREPORT FILE: "
            + report_plan["body_path"]
            + "\nMATERIAL ISSUES: "
            + issue_list
            + "\nCLAIM INDEX: "
            + repr(claim_index(citable_claims))
            + "\nSOURCE REGISTRY: "
            + repr(registry),
            "revision",
            "Revise",
            REVISION_FILE_SCHEMA,
            "synthesis",
        )
        if revision is None:
            partial_reasons.append("material revision failed")
            remaining_material_issues = list(material_issues)
        else:
            revised_path = str(revision.get("body_path", "")).strip()
            if revised_path:
                report_plan["body_path"] = revised_path
            revised_title = str(revision.get("title", "")).strip()
            if revised_title:
                report_plan["title"] = revised_title
            revised_words = int(revision.get("assembled_word_count", 0) or 0)
            if revised_words:
                assembled_words = revised_words
                report_plan["assembled_word_count"] = revised_words
            remaining_material_issues = list(revision.get("remaining_material_issues", []))

# ---------------------------------------------------------------------------
# High-stakes closure check
# ---------------------------------------------------------------------------
if high_stakes and remaining_material_issues:
    closure = await dispatch(
        "Perform a final evidence closure check only on the unresolved material "
        "issues. Do not edit prose and do not report formatting defects.\n\n"
        + (
            "REPORT: " + report
            if report_plan is None
            else "REPORT FILE: " + report_plan["body_path"] + "\n" + READ_INSTRUCTIONS
        )
        + "\nUNRESOLVED MATERIAL ISSUES: "
        + repr(remaining_material_issues)
        + "\nCLAIM INDEX: "
        + repr(claim_index(citable_claims)),
        "closure",
        "Audit",
        CLOSURE_SCHEMA,
        "verification",
    )
    if closure is None or not closure.get("supported"):
        remaining_material_issues = (
            closure.get("unresolved_material_issues", remaining_material_issues)
            if closure
            else remaining_material_issues
        )
    else:
        remaining_material_issues = []

# ---------------------------------------------------------------------------
# Advisory deterministic check. A drafted report is never discarded here;
# scripts/materialize_report.py is the authoritative structural gate.
# ---------------------------------------------------------------------------
advisory_issues = structural_issues(report) if report_plan is None else []

# A shortfall is a delivery warning, not an evidence defect: the expansion pass
# owns it, an honest short report still ships, and it never demotes the run. A
# report still far over the target after revision is different — it disregards an
# explicit instruction it was asked to fix, so it counts as unresolved.
length_notes = []
overlength_notes = []
shortfall_target = (target_words * SHORTFALL_RATIO_NUMERATOR) // SHORTFALL_RATIO_DENOMINATOR
if assembled_words and assembled_words < shortfall_target:
    length_notes.append(
        "report is materially shorter than the requested length: "
        + str(assembled_words)
        + " of about "
        + str(target_words)
        + " words"
    )
elif assembled_words and assembled_words > overlength_target:
    overlength_notes.append(
        "report is materially longer than the requested length: "
        + str(assembled_words)
        + " of about "
        + str(target_words)
        + " words"
    )

unresolved = sorted(
    set(
        [str(issue) for issue in advisory_issues if issue]
        + [str(issue) for issue in remaining_material_issues if issue]
        + overlength_notes
    )
)
gaps = sorted(set(all_gaps + partial_reasons + unresolved + length_notes))
status = (
    "partial"
    if partial_reasons or unresolved or base_lanes_completed < len(lanes)
    else "complete"
)

return {
    "status": status,
    "report_markdown": report,
    "report_plan": report_plan,
    "cited_sources": cited_sources,
    "source_registry": registry,
    "gaps": gaps,
    "run_summary": make_summary(),
}
