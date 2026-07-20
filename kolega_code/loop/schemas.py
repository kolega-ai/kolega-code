"""JSON schemas for structured bug-fix loop output.

These schemas are used by Gigacode workflows to ensure sub-agents
return parseable, structured data rather than free-form text.
"""

DIAGNOSTIC_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "pass_1": {
            "type": "object",
            "properties": {
                "architecture": {
                    "type": "string",
                    "description": "Module roles, design patterns, boundaries, conventions, assumptions",
                },
                "intended_behavior": {
                    "type": "string",
                    "description": "What the code should do, sources of truth, spec gaps",
                },
                "recent_changes": {
                    "type": "string",
                    "description": "Relevant commits, blames, potential introduction points",
                },
                "analogous_code": {
                    "type": "string",
                    "description": "Similar modules, how they handle this, potential same-class bugs",
                },
                "scope": {"enum": ["NEIGHBORHOOD", "SYSTEM"]},
            },
            "required": ["architecture", "intended_behavior", "scope"],
        },
        "pass_2": {
            "type": "object",
            "properties": {
                "error_path": {
                    "type": "string",
                    "description": "Full execution trace from symptom to error",
                },
                "unexpected_causes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Alternative root cause candidates",
                },
                "hypotheses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "approach": {"type": "string", "description": "What to change and where (file:line)"},
                            "rationale": {
                                "type": "string",
                                "description": "Why this approach, grounded in Pass 1 findings",
                            },
                            "risks": {"type": "string", "description": "What could go wrong, what else might break"},
                            "verification": {
                                "type": "string",
                                "description": "What tests/checks would confirm this fix",
                            },
                            "confidence": {"enum": ["HIGH", "MEDIUM", "LOW"]},
                        },
                        "required": ["approach", "rationale", "confidence"],
                    },
                    "minItems": 2,
                    "maxItems": 3,
                    "description": "2-3 distinct fix hypotheses",
                },
            },
            "required": ["error_path", "hypotheses"],
        },
        "overall_confidence": {"enum": ["HIGH", "MEDIUM", "LOW"]},
    },
    "required": ["pass_1", "pass_2"],
}

CHECK_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "branch": {"type": "string"},
        "check_a": {
            "type": "object",
            "properties": {
                "result": {"enum": ["PASS", "FAIL"]},
                "details": {"type": "string"},
            },
            "required": ["result"],
        },
        "check_b": {
            "type": "object",
            "properties": {
                "result": {"enum": ["PASS", "FAIL"]},
                "failed_tests": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["result"],
        },
        "overall": {"enum": ["PASS", "FAIL"]},
    },
    "required": ["branch", "check_a", "check_b", "overall"],
}

ADAPT_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string", "description": "Why previous attempts failed"},
        "pass_1_accuracy": {
            "enum": ["accurate", "partial", "missed"],
            "description": "Was the system understanding correct?",
        },
        "abandoned_hypotheses": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Which hypotheses are ruled out and why",
        },
        "retained_hypotheses": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Which hypotheses are still viable",
        },
        "new_hypothesis": {
            "type": "string",
            "description": "A new approach not in the original set, if needed",
        },
        "new_strategy": {"type": "string", "description": "Concise direction for the retry"},
        "investigation_scope": {
            "enum": ["NEIGHBORHOOD", "SYSTEM"],
            "description": "Scope for the retry investigation",
        },
    },
    "required": ["analysis", "investigation_scope"],
}
