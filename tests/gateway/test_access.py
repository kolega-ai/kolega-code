"""GatewayAccessControl: allowlist, pairing issuance, and approval across processes."""

import json
import time
from pathlib import Path

import pytest

from kolega_code.gateway.access import (
    ALLOWLIST_FILE_NAME,
    PAIRING_FILE_NAME,
    AccessControlError,
    GatewayAccessControl,
)
from kolega_code.gateway.adapters.base import InboundMessage


def make_access(tmp_path: Path, **overrides: object) -> GatewayAccessControl:
    kwargs: dict[str, object] = dict(
        state_dir=tmp_path,
        allowed_users=("123",),
        pairing_enabled=True,
        now=time.time,
    )
    kwargs.update(overrides)
    return GatewayAccessControl(**kwargs)  # type: ignore[arg-type]


def inbound(sender_id: str = "999") -> InboundMessage:
    return InboundMessage(
        channel="recording", chat_id="42", sender_id=sender_id, sender_name="New Person", message_id="m-1", text="hi"
    )


def test_allowlist_union_with_persisted_file(tmp_path: Path) -> None:
    access = make_access(tmp_path)
    assert access.is_allowed("123")  # configured
    assert not access.is_allowed("999")
    (tmp_path / ALLOWLIST_FILE_NAME).write_text(
        json.dumps({"999": {"name": "Approved", "approved_at": 1000}}), encoding="utf-8"
    )
    assert access.is_allowed("999")


@pytest.mark.parametrize("pairing_enabled", [False, True])
def test_no_allowlist_denies_unknown_users(tmp_path: Path, pairing_enabled: bool) -> None:
    access = GatewayAccessControl(state_dir=tmp_path, pairing_enabled=pairing_enabled)
    assert not access.is_allowed("anyone")
    assert access.summary() == {
        "policy": "pairing-only" if pairing_enabled else "locked",
        "configured_users": 0,
        "paired_users": 0,
        "pairing_enabled": pairing_enabled,
    }


@pytest.mark.parametrize("sender_id", ["", " ", "\t", " 123", "123 "])
def test_missing_identity_cannot_be_allowed_or_paired(tmp_path: Path, sender_id: str) -> None:
    access = make_access(tmp_path, allowed_users=(sender_id,))
    assert not access.is_allowed(sender_id)
    assert access.on_unknown_sender(inbound(sender_id)) is None
    assert access.pending() == []


def test_first_user_pairing_then_last_approval_removal(tmp_path: Path) -> None:
    access = make_access(tmp_path, allowed_users=())
    reply = access.on_unknown_sender(inbound())
    assert reply is not None
    assert not access.is_allowed("999")
    code = reply.rsplit(" ", 1)[-1]
    assert access.on_unknown_sender(inbound()) == reply
    # A separate operator instance grants membership; the running checker sees it.
    operator = GatewayAccessControl(state_dir=tmp_path)
    assert operator.approve(code) == "999"
    assert access.is_allowed("999")
    assert not access.is_allowed("123")
    assert access.summary()["policy"] == "restricted"
    assert access.summary()["paired_users"] == 1
    (tmp_path / ALLOWLIST_FILE_NAME).write_text("{}", encoding="utf-8")
    assert not access.is_allowed("999")
    assert access.summary()["policy"] == "pairing-only"


@pytest.mark.parametrize(
    "record",
    [
        None,
        True,
        "approved",
        [],
        {},
        {"name": "x"},
        {"name": [], "approved_at": 1},
        {"name": "x", "approved_at": True},
        {"name": "x", "approved_at": "1000"},
        {"name": "x", "approved_at": float("nan")},
        {"name": "x", "approved_at": float("inf")},
        {"name": "x", "approved_at": -1},
    ],
)
def test_malformed_approval_does_not_authorize(tmp_path: Path, record: object) -> None:
    (tmp_path / ALLOWLIST_FILE_NAME).write_text(json.dumps({"999": record}), encoding="utf-8")
    access = make_access(tmp_path)
    assert not access.is_allowed("999")
    assert access.is_allowed("123")
    assert access.summary()["paired_users"] == 0


@pytest.mark.parametrize(
    "content",
    [b"{not JSON", b"[]", b"null", b"\xff", b'{"999":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}"],
)
def test_unreadable_approvals_never_grant_users(tmp_path: Path, content: bytes) -> None:
    (tmp_path / ALLOWLIST_FILE_NAME).write_bytes(content)
    access = make_access(tmp_path)
    assert access.is_allowed("123")
    assert not access.is_allowed("999")
    assert access.summary()["paired_users"] == 0


def test_malformed_pairing_records_are_ignored(tmp_path: Path) -> None:
    valid = {"code": "ABC123", "sender_id": "999", "created_at": 1, "expires_at": 1001}
    invalid: list[object] = [
        None,
        [],
        True,
        "bad",
        {},
        {**valid, "sender_id": ""},
        {**valid, "created_at": "bad"},
        {**valid, "expires_at": float("nan")},
        {**valid, "expires_at": float("inf")},
        {**valid, "expires_at": False},
        {**valid, "sender_name": {}},
        {**valid, "created_at": 2000},
    ]
    # Keep matching code values so each invalid field, not only the key, is exercised.
    payload = {
        f"BAD{i}": {**entry, "code": f"BAD{i}"} if isinstance(entry, dict) else entry for i, entry in enumerate(invalid)
    }
    payload["ABC123"] = valid
    (tmp_path / PAIRING_FILE_NAME).write_text(json.dumps(payload), encoding="utf-8")
    access = make_access(tmp_path, allowed_users=(), now=lambda: 1000)
    assert [request.code for request in access.pending()] == ["ABC123"]
    reply = access.on_unknown_sender(inbound())
    assert reply is not None and reply.endswith("ABC123")
    assert access.approve("ABC123") == "999"
    assert access.is_allowed("999")


def test_unknown_sender_is_silent_without_pairing(tmp_path: Path) -> None:
    access = make_access(tmp_path, pairing_enabled=False)
    assert access.on_unknown_sender(inbound()) is None


def test_pairing_issues_a_code_and_reuses_it(tmp_path: Path) -> None:
    access = make_access(tmp_path)
    first = access.on_unknown_sender(inbound())
    assert first is not None
    code = first.rsplit(" ", 1)[-1]
    assert len(code) == 6
    second = access.on_unknown_sender(inbound())
    assert second is not None and second.endswith(code)
    assert len(access.pending()) == 1


def test_approve_admits_the_sender(tmp_path: Path) -> None:
    access = make_access(tmp_path)
    reply = access.on_unknown_sender(inbound())
    assert reply is not None
    code = reply.rsplit(" ", 1)[-1]
    assert access.approve(code) == "999"
    assert access.is_allowed("999")
    assert access.pending() == []


def test_approve_rejects_unknown_or_expired_codes(tmp_path: Path) -> None:
    access = make_access(tmp_path)
    with pytest.raises(AccessControlError):
        access.approve("ZZZZZZ")
    # An expired pending code is rejected too.
    expired = {
        "EXP1RE": {
            "sender_id": "999",
            "created_at": 0.0,
            "expires_at": 1.0,
        }
    }
    (tmp_path / PAIRING_FILE_NAME).write_text(json.dumps(expired), encoding="utf-8")
    with pytest.raises(AccessControlError):
        access.approve("EXP1RE")


def test_pending_prunes_expired_entries(tmp_path: Path) -> None:
    stale = {f"CODE{i}": {"sender_id": str(i), "created_at": 0.0, "expires_at": 1.0} for i in range(10)}
    (tmp_path / PAIRING_FILE_NAME).write_text(json.dumps(stale), encoding="utf-8")
    access = make_access(tmp_path)
    assert access.pending() == []


def test_pairing_cap_keeps_the_newest(tmp_path: Path) -> None:
    from kolega_code.gateway.access import MAX_PENDING_CODES

    now = 1000.0
    many = {
        f"CODE{i:02d}": {"sender_id": str(i), "created_at": now - (100 - i), "expires_at": now + 3600}
        for i in range(MAX_PENDING_CODES + 10)
    }
    (tmp_path / PAIRING_FILE_NAME).write_text(json.dumps(many), encoding="utf-8")
    access = make_access(tmp_path, now=lambda: now)
    # Issuing one more code prunes the oldest entries down to the cap.
    access.on_unknown_sender(inbound())
    pending = json.loads((tmp_path / PAIRING_FILE_NAME).read_text(encoding="utf-8"))
    assert len(pending) == MAX_PENDING_CODES
    assert "CODE00" not in pending
