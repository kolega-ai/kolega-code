"""Outbound redaction of known secrets."""

from kolega_code.gateway.redaction import REDACTED_PLACEHOLDER, scrub


def test_no_secrets_no_change() -> None:
    assert scrub("plain text", []) == "plain text"


def test_secrets_are_replaced() -> None:
    text = "key=sk-abcdef1234567890 end"
    assert scrub(text, ["sk-abcdef1234567890"]) == f"key={REDACTED_PLACEHOLDER} end"


def test_longest_secret_wins_for_overlaps() -> None:
    assert scrub("abcdefgh", ["abcdefgh", "cdef"]) == REDACTED_PLACEHOLDER


def test_short_values_are_ignored() -> None:
    # Short candidates would rewrite ordinary words.
    assert scrub("ask and new", ["ask", "new"]) == "ask and new"


def test_every_occurrence_is_replaced() -> None:
    text = "token sk-abcdef1234567890 and sk-abcdef1234567890 again"
    assert scrub(text, ["sk-abcdef1234567890"]).count(REDACTED_PLACEHOLDER) == 2


def test_telegram_token_shaped_secrets() -> None:
    token = "123456:fake-bot-token-for-tests-only"
    assert scrub(f"leaked {token}!", [token]) == f"leaked {REDACTED_PLACEHOLDER}!"
