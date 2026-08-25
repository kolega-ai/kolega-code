"""In-process peer inbox: registration, addressing, policy, and delivery.

Like the control-channel tests, the failure modes matter as much as the happy
path: a send that silently goes nowhere, an ambiguous address that reaches the
wrong peer, or a policy that lets a bypassing sender ride its authority into a
cautious recipient are all pinned here.
"""

from __future__ import annotations

import pytest

from kolega_code.session.inbox import (
    MAX_PEER_TEXT_CHARS,
    DeliveryOutcome,
    InboxRegistration,
    InboxRegistry,
    PeerMessage,
    PeerMessageError,
    resolve_inbound_decision,
    validate_peer_text,
)


def _registration(
    session_id: str = "s1",
    title: str = "myapp",
    project_path: str = "/tmp/myapp",
    status: str = "idle",
    permission_mode: str = "ask",
    outcomes: list[str] | None = None,
) -> InboxRegistration:
    def deliver(_message: PeerMessage) -> str:
        if outcomes is not None:
            return outcomes.pop(0)
        return DeliveryOutcome.ACCEPTED.value

    async def deliver_async(message: PeerMessage) -> str:
        return deliver(message)

    return InboxRegistration(
        session_id=session_id,
        describe_title=lambda: title,
        describe_project_path=lambda: project_path,
        describe_status=lambda: status,
        describe_permission_mode=lambda: permission_mode,
        deliver_message=deliver_async,
    )


def _message(text: str = "hello", sender_session_id: str = "s2", **kwargs) -> PeerMessage:
    return PeerMessage.create(sender_session_id=sender_session_id, sender_title="peer", text=text, **kwargs)


# -- Registration lifecycle --------------------------------------------------


def test_register_and_unregister_round_trip() -> None:
    registry = InboxRegistry()
    registry.register(_registration())
    assert registry.is_registered("s1")

    registry.unregister("s1")
    assert not registry.is_registered("s1")
    assert registry.list_agents() == []


def test_unregister_unknown_session_is_a_no_op() -> None:
    InboxRegistry().unregister("never-registered")


def test_reregistration_replaces_the_entry() -> None:
    registry = InboxRegistry()
    registry.register(_registration(status="idle"))
    registry.register(_registration(status="busy"))

    (agent,) = registry.list_agents()
    assert agent.status == "busy"


# -- Discovery ---------------------------------------------------------------


def test_list_agents_reports_live_callables_and_excludes_self() -> None:
    registry = InboxRegistry()
    registry.register(_registration(session_id="self", title="self-session", status="busy", permission_mode="auto"))
    registry.register(_registration(session_id="b", title="zeta"))
    registry.register(_registration(session_id="a", title="alpha", project_path="/tmp/alpha"))

    agents = registry.list_agents(exclude_session_id="self")

    assert [agent.title for agent in agents] == ["alpha", "zeta"]  # sorted by name
    assert [agent.status for agent in agents] == ["idle", "idle"]
    assert agents[0].session_id == "a"
    assert agents[0].project_path == "/tmp/alpha"
    # Self never appears in discovery.
    assert all(agent.session_id != "self" for agent in agents)


# -- Addressing --------------------------------------------------------------


def test_resolve_by_exact_name_case_insensitive() -> None:
    registry = InboxRegistry()
    registry.register(_registration(title="MyApp"))
    registry.register(_registration(session_id="other", title="other"))

    resolved = registry.resolve(" myapp ", exclude_session_id="other")
    assert resolved.session_id == "s1"


def test_resolve_by_unique_name_prefix_and_by_full_id() -> None:
    registry = InboxRegistry()
    registry.register(_registration(title="deploy-bot", session_id="aaaa-bbbb"))
    registry.register(_registration(session_id="cccc-dddd", title="unrelated"))

    assert registry.resolve("dep").session_id == "aaaa-bbbb"
    assert registry.resolve("aaaa-bbbb").session_id == "aaaa-bbbb"
    assert registry.resolve("cccc").session_id == "cccc-dddd"  # id prefix


@pytest.mark.parametrize("query", ["", "   ", "ghost"])
@pytest.mark.asyncio
async def test_resolve_unknown_recipient_raises(query: str) -> None:
    registry = InboxRegistry()
    registry.register(_registration())

    with pytest.raises(PeerMessageError):
        registry.resolve(query)


def test_resolve_ambiguous_prefix_lists_candidates() -> None:
    registry = InboxRegistry()
    registry.register(_registration(session_id="1", title="deploy-api"))
    registry.register(_registration(session_id="2", title="deploy-web"))

    with pytest.raises(PeerMessageError, match="deploy-api.*deploy-web|deploy-web.*deploy-api"):
        registry.resolve("deploy")


def test_resolve_duplicate_titles_are_ambiguous_not_silent() -> None:
    """Two sessions sharing a name must fail loudly, not message the wrong one."""
    registry = InboxRegistry()
    registry.register(_registration(session_id="1", title="myapp"))
    registry.register(_registration(session_id="2", title="myapp"))

    with pytest.raises(PeerMessageError, match="address one by session id"):
        registry.resolve("myapp")

    # The id form still works.
    assert registry.resolve("2").session_id == "2"


# -- Inbound policy ----------------------------------------------------------


@pytest.mark.parametrize(
    "recipient_mode,sender_mode,expected",
    [
        ("ask", "ask", DeliveryOutcome.ACCEPTED),
        ("auto", "auto", DeliveryOutcome.ACCEPTED),
        ("ask", "auto", DeliveryOutcome.HELD),
        ("auto", "ask", DeliveryOutcome.HELD),
    ],
)
def test_auto_policy_is_asymmetric_by_permission_mode(
    recipient_mode: str, sender_mode: str, expected: DeliveryOutcome
) -> None:
    assert resolve_inbound_decision("auto", recipient_mode, sender_mode) is expected


@pytest.mark.parametrize("recipient_mode,sender_mode", [("ask", "ask"), ("ask", "auto"), ("auto", "ask")])
def test_explicit_policies_override_the_matrix(recipient_mode: str, sender_mode: str) -> None:
    assert resolve_inbound_decision("accept", recipient_mode, sender_mode) is DeliveryOutcome.ACCEPTED
    assert resolve_inbound_decision("hold", recipient_mode, sender_mode) is DeliveryOutcome.HELD
    assert resolve_inbound_decision("refuse", recipient_mode, sender_mode) is DeliveryOutcome.REFUSED


@pytest.mark.parametrize("policy", [None, "", "nonsense"])
def test_unknown_policy_degrades_to_auto(policy: object) -> None:
    assert resolve_inbound_decision(policy, "ask", "ask") is DeliveryOutcome.ACCEPTED  # type: ignore[arg-type]
    assert resolve_inbound_decision(policy, "auto", "ask") is DeliveryOutcome.HELD  # type: ignore[arg-type]


# -- Delivery ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_deliver_routes_to_the_registration_outcome() -> None:
    outcomes = [DeliveryOutcome.HELD.value]
    registry = InboxRegistry()
    registry.register(_registration(outcomes=outcomes))

    outcome = await registry.deliver("s1", _message(), sender_session_id="s2")

    assert outcome == DeliveryOutcome.HELD.value


@pytest.mark.asyncio
async def test_deliver_to_unknown_recipient_raises() -> None:
    registry = InboxRegistry()

    with pytest.raises(PeerMessageError, match="No live session"):
        await registry.deliver("missing", _message())


@pytest.mark.asyncio
async def test_deliver_rejects_self_addressing_even_when_registered() -> None:
    """The tool rejects self-sends first; the registry is the backstop."""
    registry = InboxRegistry()
    registry.register(_registration(session_id="me"))

    with pytest.raises(PeerMessageError, match="cannot message itself"):
        await registry.deliver("me", _message(sender_session_id="me"), sender_session_id="me")


@pytest.mark.asyncio
async def test_deliver_rejects_empty_text() -> None:
    registry = InboxRegistry()
    registry.register(_registration())

    with pytest.raises(PeerMessageError, match="must not be empty"):
        await registry.deliver("s1", _message(text="   "))


@pytest.mark.asyncio
async def test_deliver_rejects_oversized_text() -> None:
    registry = InboxRegistry()
    registry.register(_registration())

    oversized = PeerMessage.create(sender_session_id="s2", sender_title="p", text="x" * (MAX_PEER_TEXT_CHARS + 1))
    with pytest.raises(PeerMessageError, match="character limit"):
        await registry.deliver("s1", oversized)


def test_validate_peer_text_strips_whitespace() -> None:
    assert validate_peer_text("  hi \n") == "hi"


# -- Message shape -----------------------------------------------------------


def test_peer_message_create_stamps_identity_and_time() -> None:
    message = PeerMessage.create(sender_session_id="s1", sender_title="a", text="hi")

    assert message.message_id
    assert message.sender_mode == "ask"
    assert message.created_at  # utc_now_iso stamped by default factory
    assert message.reply_to is None


@pytest.mark.asyncio
async def test_delivery_failure_inside_callback_propagates() -> None:
    """A recipient-side crash must surface to the sender, never fake success."""

    async def boom(_message: PeerMessage) -> str:
        raise RuntimeError("queue exploded")

    registry = InboxRegistry()
    registry.register(_registration())
    registry.register(
        InboxRegistration(
            session_id="broken",
            describe_title=lambda: "broken",
            describe_project_path=lambda: "/x",
            describe_status=lambda: "idle",
            describe_permission_mode=lambda: "ask",
            deliver_message=boom,
        )
    )

    with pytest.raises(RuntimeError, match="queue exploded"):
        await registry.deliver("broken", _message())


# The shared module-level registry exists so ordinary hosts need no wiring.
def test_shared_registry_is_a_registry() -> None:
    from kolega_code.session.inbox import SHARED_INBOX_REGISTRY

    assert isinstance(SHARED_INBOX_REGISTRY, InboxRegistry)
