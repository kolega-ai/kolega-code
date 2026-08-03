from kolega_code.agent.tool_backend.network_status import (
    ConnectFailureTracker,
    connect_failure_message,
    looks_like_connect_failure,
    looks_like_tls_failure,
    secure_connection_failure_message,
)


def test_tracker_counts_distinct_subjects_and_dedupes() -> None:
    tracker = ConnectFailureTracker()
    assert tracker.record("example.com") == 1
    assert tracker.record("Example.com ") == 1
    assert tracker.record("other.test") == 2
    assert tracker.distinct_count == 2


def test_connect_failure_message_first_host() -> None:
    message = connect_failure_message("example.com", "connection refused", 1)
    assert message.startswith("Could not connect to example.com (connection refused).")
    assert "may restrict outbound network access" in message
    assert "Prefer resources available locally" in message


def test_connect_failure_message_omits_empty_detail() -> None:
    message = connect_failure_message("example.com", "", 1)
    assert message.startswith("Could not connect to example.com. ")
    assert "()" not in message


def test_connect_failure_message_escalates_at_two() -> None:
    message = connect_failure_message("b.test", "connection refused", 2)
    assert "2 distinct external hosts have failed to connect" in message
    assert "treat outbound network access as unavailable" in message
    assert "may restrict" not in message


def test_secure_connection_message_has_no_egress_hint() -> None:
    message = secure_connection_failure_message("example.com", "[SSL: CERTIFICATE_VERIFY_FAILED]")
    assert message.startswith("Could not establish a secure connection to example.com")
    assert "may restrict" not in message
    assert "retrying will not help" in message


def test_marker_sniffing() -> None:
    assert looks_like_connect_failure("All connection attempts failed")
    assert looks_like_connect_failure("Temporary failure in name resolution")
    assert looks_like_connect_failure("[Errno 61] Connection refused")
    # A mid-stream reset means the server WAS reachable — must not classify as unreachable.
    assert not looks_like_connect_failure("connection reset by peer")
    assert not looks_like_connect_failure("HTTP 502 Bad Gateway")
    assert looks_like_tls_failure("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    assert not looks_like_tls_failure("connection refused")
