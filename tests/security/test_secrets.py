"""Tests for probable-secret detection and redaction.

Focus: redaction must never corrupt the document it is scrubbing. A bug bundle was
shipped with an unparseable ``session.json`` because a match span absorbed the
backslash of a JSON ``\\"`` escape, leaving a bare quote behind.
"""

import json

from kolega_code.security import (
    SECRET_PLACEHOLDER,
    detect_secrets,
    redact_secrets,
    redact_secrets_in_obj,
)

# Obviously fake, but shaped like the real thing so the patterns fire.
FAKE_TOKEN = "apify_api_9RtQv3LmZx8KpWn2Yc7BdF4HsJ6Tg1Ae0Nu5"
SOURCE_LINE = f'api_url = f"https://api.example.com/v2/runs?token={FAKE_TOKEN}"'


def test_redacting_serialized_json_keeps_it_parseable():
    """The exact shape that corrupted a real bug bundle."""
    encoded = json.dumps({"text": SOURCE_LINE})

    out = redact_secrets(encoded)

    assert json.loads(out)  # would raise before the fix
    assert FAKE_TOKEN not in out
    assert SECRET_PLACEHOLDER in out


def test_detected_span_never_swallows_a_trailing_escape():
    encoded = json.dumps({"text": SOURCE_LINE})

    spans = [encoded[finding.start : finding.end] for finding in detect_secrets(encoded)]

    assert spans == [FAKE_TOKEN]
    assert not any(span.endswith("\\") for span in spans)


def test_redact_in_obj_redacts_nested_strings_and_preserves_shape():
    payload = {
        "text": SOURCE_LINE,
        "nested": {"deep": [SOURCE_LINE, "harmless"]},
        "count": 3,
        "ratio": 1.5,
        "flag": True,
        "missing": None,
    }

    out = redact_secrets_in_obj(payload)

    assert FAKE_TOKEN not in json.dumps(out)
    assert SECRET_PLACEHOLDER in out["text"]
    assert out["nested"]["deep"][1] == "harmless"
    assert out["count"] == 3
    assert out["ratio"] == 1.5
    assert out["flag"] is True
    assert out["missing"] is None


def test_redact_in_obj_output_is_json_serializable_and_parses():
    out = redact_secrets_in_obj({"text": SOURCE_LINE})

    assert json.loads(json.dumps(out))["text"].endswith("token=" + SECRET_PLACEHOLDER + '"')


def test_redact_in_obj_coerces_and_redacts_non_json_leaves():
    """``json.dumps(default=str)`` stringifies after scrubbing, so such leaves escaped redaction."""

    class Failure:
        def __str__(self) -> str:
            return f"request failed: token={FAKE_TOKEN}"

    out = redact_secrets_in_obj({"error": Failure()})

    assert isinstance(out["error"], str)
    assert FAKE_TOKEN not in out["error"]
    assert SECRET_PLACEHOLDER in out["error"]


def test_redact_in_obj_leaves_keys_alone():
    out = redact_secrets_in_obj({"api_key_name": "harmless"})

    assert "api_key_name" in out


def test_redact_in_obj_applies_configured_extra_values():
    out = redact_secrets_in_obj({"note": "value is hunter2hunter2"}, ["hunter2hunter2"])

    assert "hunter2hunter2" not in out["note"]


def test_quoted_credential_values_are_still_redacted():
    text = 'DEEPSEEK_API_KEY="quotedsecretvalue123"'

    out = redact_secrets(text)

    assert "quotedsecretvalue123" not in out
