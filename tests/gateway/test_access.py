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
    (tmp_path / ALLOWLIST_FILE_NAME).write_text(json.dumps({"999": {"name": "Approved"}}), encoding="utf-8")
    assert access.is_allowed("999")


def test_no_allowlist_means_open(tmp_path: Path) -> None:
    access = GatewayAccessControl(state_dir=tmp_path, allowed_users=(), pairing_enabled=True)
    assert access.is_allowed("anyone")


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
