"""Security settings cannot be silently discarded or coerced into identities."""

import pytest

from kolega_code.gateway.access_settings import AccessSettingsError, parse_allowed_users, validate_access_settings


@pytest.mark.parametrize("raw", [None, [], "telegram", 42, False])
def test_explicit_malformed_gateway_section_is_rejected(raw: object) -> None:
    with pytest.raises(AccessSettingsError, match="gateway: expected an object"):
        validate_access_settings(raw)


@pytest.mark.parametrize("key", ["allowed_users", "group_ids"])
@pytest.mark.parametrize("value", [None, "123", [123], [True], ["123", 456], [""], [" \t"]])
def test_access_lists_require_nonempty_strings(key: str, value: object) -> None:
    with pytest.raises(AccessSettingsError, match=rf"gateway\.{key}:"):
        validate_access_settings({key: value})


@pytest.mark.parametrize("value", [None, "false", "true", 0, 1, [], {}])
def test_pairing_requires_a_real_boolean(value: object) -> None:
    with pytest.raises(AccessSettingsError, match=r"gateway\.pairing_enabled:"):
        validate_access_settings({"pairing_enabled": value})


@pytest.mark.parametrize("user_id", ["@alice", "*", "0", "000", "-123", "+123", "١٢٣", "１２３", "1.2", "1 2"])
def test_telegram_users_require_positive_ascii_decimal_ids(user_id: str) -> None:
    with pytest.raises(AccessSettingsError, match=r"gateway\.allowed_users:"):
        validate_access_settings({"adapter": "telegram", "allowed_users": [user_id]})


@pytest.mark.parametrize("group_id", ["@group", "*", "0", "-0", "+000", "١٢٣", "--123", "1.2"])
def test_telegram_groups_require_signed_nonzero_ascii_decimal_ids(group_id: str) -> None:
    with pytest.raises(AccessSettingsError, match=r"gateway\.group_ids:"):
        validate_access_settings({"group_ids": [group_id]}, adapter="telegram")


def test_normalizes_without_integer_conversion_or_mutating_input() -> None:
    raw = {
        "adapter": "telegram",
        "allowed_users": [" 123 ", "123", "000123"],
        "group_ids": [" -1001 ", "-001001", "+123", "456"],
        "pairing_enabled": False,
    }
    result = validate_access_settings(raw)
    assert result["allowed_users"] == ["123"]
    assert result["group_ids"] == ["-1001", "123", "456"]
    assert raw["allowed_users"] == [" 123 ", "123", "000123"]


def test_safe_defaults_and_generic_identities_remain_supported() -> None:
    assert validate_access_settings({}) == {}
    assert validate_access_settings({"allowed_users": [], "group_ids": [], "pairing_enabled": True}) == {
        "allowed_users": [],
        "group_ids": [],
        "pairing_enabled": True,
    }
    assert validate_access_settings({"allowed_users": [" owner ", "alice", "owner"]}, adapter="echo") == {
        "allowed_users": ["owner", "alice"]
    }


def test_input_splitting_preserves_invalid_empty_entries() -> None:
    assert parse_allowed_users("  ") == []
    assert parse_allowed_users("123, ,456") == ["123", "", "456"]
