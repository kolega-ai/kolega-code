"""Session server: discovery, ranged reads, live follow, and access scoping.

The follow tests are the important ones. "Attach and lose nothing" is the whole
promise of the streaming endpoint, so they cover reconnecting at an arbitrary
sequence and receiving events appended after the socket was already open.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kolega_code.cli.session_event_store import FileArtifactStore, FileSessionEventStore
from kolega_code.cli.session_store import SessionStore
from kolega_code.events import AgentEvent, ArtifactPurpose, KnownEventType
from kolega_code.web.server import ServerConfig, create_app


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(root=tmp_path / "state")


async def _seed(store: SessionStore, project: Path, *, count: int = 4):
    record = store.create(project, "code", {"model": "test"}, title="Fix the parser")
    events = FileSessionEventStore(store.journal(record.session_id))
    await events.append(
        AgentEvent(
            session_id=record.session_id,
            sender="agent",
            event_type=KnownEventType.TURN_STARTED,
            content={"turn_id": "t1", "user_text": "fix the parser"},
        )
    )
    for index in range(count):
        await events.append(
            AgentEvent(
                session_id=record.session_id,
                sender="agent",
                event_type=KnownEventType.ASSISTANT_DELTA,
                content={"text": f"step {index}", "complete": True},
                elapsed_ms=100 * (index + 1),
            )
        )
    await events.append(
        AgentEvent(
            session_id=record.session_id,
            sender="agent",
            event_type=KnownEventType.TURN_ENDED,
            content={"turn_id": "t1", "status": "completed"},
            elapsed_ms=900,
        )
    )
    return record


def _client(store: SessionStore, token: str | None = None) -> TestClient:
    return TestClient(create_app(ServerConfig(store=store, token=token)))


@pytest.mark.asyncio
async def test_lists_sessions_with_replay_metadata(store: SessionStore, tmp_path: Path) -> None:
    record = await _seed(store, tmp_path)

    with _client(store) as client:
        payload = client.get("/api/sessions").json()

    (summary,) = [item for item in payload if item["session_id"] == record.session_id]
    assert summary["title"] == "Fix the parser"
    assert summary["event_count"] == 6
    assert summary["replayable"] is True
    assert summary["status"] == "idle"


@pytest.mark.asyncio
async def test_project_filter_matches_the_returned_path(store: SessionStore, tmp_path: Path) -> None:
    mine = await _seed(store, tmp_path)
    other = store.create(tmp_path / "elsewhere", "code", {"model": "test"}, title="Other project")

    with _client(store) as client:
        listed = client.get("/api/sessions").json()
        by_id = {item["session_id"]: item for item in listed}
        filtered = client.get("/api/sessions", params={"project": by_id[mine.session_id]["project_path"]}).json()

    assert {item["session_id"] for item in filtered} == {mine.session_id}
    assert other.session_id not in {item["session_id"] for item in filtered}
    # A trailing slash is the one difference a client is likely to introduce.
    with _client(store) as client:
        trailing = client.get(
            "/api/sessions",
            params={"project": by_id[mine.session_id]["project_path"] + "/"},
        ).json()
    assert {item["session_id"] for item in trailing} == {mine.session_id}


@pytest.mark.asyncio
async def test_project_filter_never_reaches_the_filesystem(store: SessionStore, tmp_path: Path) -> None:
    """The filter is an opaque string match, so hostile input is inert.

    Regression test for a path-injection finding: the parameter used to be
    resolved into a real path, which put untrusted request data into a
    filesystem expression for a comparison that never needed one.
    """
    await _seed(store, tmp_path)
    hostile = ["../../../../etc", "/etc/passwd", "~/.ssh/id_rsa", "\x00/etc", "a" * 4096]

    with _client(store) as client:
        for value in hostile:
            response = client.get("/api/sessions", params={"project": value})
            assert response.status_code == 200, f"{value!r} produced {response.status_code}"
            assert response.json() == [], f"{value!r} unexpectedly matched a session"


@pytest.mark.asyncio
async def test_session_id_traversal_is_rejected(store: SessionStore, tmp_path: Path) -> None:
    await _seed(store, tmp_path)
    with _client(store) as client:
        for value in ("..", "../../etc/passwd", "%2e%2e%2f"):
            response = client.get(f"/api/sessions/{value}")
            assert response.status_code in (404, 400), f"{value!r} produced {response.status_code}"


@pytest.mark.asyncio
async def test_session_without_events_is_marked_unreplayable(store: SessionStore, tmp_path: Path) -> None:
    """Sessions predating the event spine must not offer a broken replay."""
    record = store.create(tmp_path, "code", {"model": "test"})

    with _client(store) as client:
        payload = client.get("/api/sessions").json()

    (summary,) = [item for item in payload if item["session_id"] == record.session_id]
    assert summary["replayable"] is False and summary["event_count"] == 0


@pytest.mark.asyncio
async def test_reads_an_event_range_and_reports_the_next_cursor(store: SessionStore, tmp_path: Path) -> None:
    record = await _seed(store, tmp_path)

    with _client(store) as client:
        first = client.get(f"/api/sessions/{record.session_id}/events", params={"limit": 2}).json()

    assert len(first["events"]) == 2
    assert first["complete"] is False
    # Sequence numbers are sparse, so the cursor has to come from the server.
    assert first["next_from_seq"] == first["events"][-1]["seq"] + 1

    with _client(store) as client:
        rest = client.get(
            f"/api/sessions/{record.session_id}/events",
            params={"from_seq": first["next_from_seq"]},
        ).json()
    assert [event["seq"] for event in rest["events"]] == sorted(event["seq"] for event in rest["events"])
    assert rest["complete"] is True


@pytest.mark.asyncio
async def test_server_side_fold_matches_the_projection(store: SessionStore, tmp_path: Path) -> None:
    record = await _seed(store, tmp_path)

    with _client(store) as client:
        state = client.get(f"/api/sessions/{record.session_id}/state").json()

    assert [item["kind"] for item in state["conversation"]] == ["user"] + ["assistant"] * 4
    assert state["turns"][0]["status"] == "completed"
    assert "_streams" not in state


@pytest.mark.asyncio
async def test_stream_replays_backlog_then_follows_live(store: SessionStore, tmp_path: Path) -> None:
    record = await _seed(store, tmp_path, count=2)
    events = FileSessionEventStore(store.journal(record.session_id))

    with _client(store) as client:
        with client.websocket_connect(f"/api/sessions/{record.session_id}/stream") as socket:
            backlog = [json.loads(socket.receive_text()) for _ in range(4)]
            assert [item["event_type"] for item in backlog][0] == KnownEventType.TURN_STARTED

            # Appended after the socket was already open: it must still arrive.
            await events.append(
                AgentEvent(
                    session_id=record.session_id,
                    sender="agent",
                    event_type=KnownEventType.ASSISTANT_DELTA,
                    content={"text": "appended live", "complete": True},
                    elapsed_ms=1200,
                )
            )
            live = json.loads(socket.receive_text())

    assert live["content"]["text"] == "appended live"
    assert live["seq"] > backlog[-1]["seq"], "live events must continue the sequence"


@pytest.mark.asyncio
async def test_stream_resumes_at_a_given_sequence(store: SessionStore, tmp_path: Path) -> None:
    """A client reconnecting at its last seen seq must not receive duplicates."""
    record = await _seed(store, tmp_path)

    with _client(store) as client:
        with client.websocket_connect(f"/api/sessions/{record.session_id}/stream") as socket:
            first = [json.loads(socket.receive_text()) for _ in range(3)]
        resume_at = first[-1]["seq"] + 1
        with client.websocket_connect(
            f"/api/sessions/{record.session_id}/stream",
            params={"from_seq": resume_at},
        ) as socket:
            rest = [json.loads(socket.receive_text()) for _ in range(3)]

    seen = [item["seq"] for item in first + rest]
    assert len(seen) == len(set(seen)), f"resume delivered a duplicate: {seen}"
    assert seen == sorted(seen)


@pytest.mark.asyncio
async def test_unknown_session_is_a_404(store: SessionStore, tmp_path: Path) -> None:
    with _client(store) as client:
        assert client.get("/api/sessions/does-not-exist").status_code == 404


@pytest.mark.asyncio
async def test_token_gates_every_route(store: SessionStore, tmp_path: Path) -> None:
    record = await _seed(store, tmp_path)

    with _client(store, token="s3cret") as client:
        assert client.get("/api/sessions").status_code == 401
        assert client.get(f"/api/sessions/{record.session_id}/events").status_code == 401
        assert client.get("/api/sessions", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert client.get("/api/sessions", params={"token": "s3cret"}).status_code == 200
        assert client.get("/api/sessions", headers={"Authorization": "Bearer wrong"}).status_code == 401


@pytest.mark.asyncio
async def test_a_link_with_a_token_can_load_the_players_own_files(store: SessionStore, tmp_path: Path) -> None:
    """A shared link is the only place the token can live, and it is a query string.

    The browser then resolves the player's script, stylesheet, manifest, and event
    log as plain relative URLs with no query, so without carrying the grant
    forward every one of them is unauthorized and the page renders blank.
    """
    record = await _seed(store, tmp_path)

    with _client(store, token="s3cret") as client:
        page = client.get(f"/s/{record.session_id}", params={"token": "s3cret"})
        assert page.status_code == 200

        # The client keeps the cookie the page handed back, as a browser would.
        for path in ("player.js", "player.css", "fold.js", "manifest.json", "events.jsonl"):
            assert client.get(f"/s/{record.session_id}/{path}").status_code == 200, path

    # A visitor without the link still gets nothing.
    with _client(store, token="s3cret") as bare:
        assert bare.get(f"/s/{record.session_id}/manifest.json").status_code == 401
        assert bare.get(f"/s/{record.session_id}", params={"token": "wrong"}).status_code == 401


@pytest.mark.asyncio
async def test_player_manifest_advertises_a_live_session(store: SessionStore, tmp_path: Path) -> None:
    """The player only follows when the manifest tells it there is something to follow."""
    record = await _seed(store, tmp_path)

    with _client(store) as client:
        manifest = client.get(f"/s/{record.session_id}/manifest.json").json()

    assert manifest["stream"] == f"/api/sessions/{record.session_id}/stream"
    assert manifest["last_seq"] > 0, "the player resumes the stream from here"
    assert manifest["status"] == "idle", "the seeded session's turn is closed"


@pytest.mark.asyncio
async def test_player_manifest_reports_an_open_turn_as_open(store: SessionStore, tmp_path: Path) -> None:
    record = await _seed(store, tmp_path)
    events = FileSessionEventStore(store.journal(record.session_id))
    await events.append(
        AgentEvent(
            session_id=record.session_id,
            sender="agent",
            event_type=KnownEventType.TURN_STARTED,
            content={"turn_id": "t2", "user_text": "still going"},
            elapsed_ms=1000,
        )
    )

    with _client(store) as client:
        manifest = client.get(f"/s/{record.session_id}/manifest.json").json()

    assert manifest["status"] == "open"


@pytest.mark.asyncio
async def test_artifact_read_is_scoped_to_shareable_purposes(store: SessionStore, tmp_path: Path) -> None:
    """Opaque provider payloads must not be readable through the artifact route."""
    record = await _seed(store, tmp_path)
    journal = store.journal(record.session_id)
    artifacts = FileArtifactStore(journal)
    events = FileSessionEventStore(journal)

    shareable = await artifacts.put(
        b"full tool output",
        media_type="text/plain; charset=utf-8",
        purpose=ArtifactPurpose.TOOL_RESULT,
        encoding="utf-8",
        chars=16,
    )
    secret = await artifacts.put(
        b"encrypted provider reasoning",
        media_type="application/octet-stream",
        purpose=ArtifactPurpose.ENCRYPTED_REASONING,
        encoding="utf-8",
    )
    carrier = AgentEvent(
        session_id=record.session_id,
        sender="agent",
        event_type=KnownEventType.CHAT_MESSAGE,
        content={"message_type": "tool_result", "text": "preview"},
    )
    carrier.artifacts = [shareable, secret]
    await events.append(carrier)

    with _client(store) as client:
        allowed = client.get(f"/api/sessions/{record.session_id}/artifacts/{shareable.sha256}")
        denied = client.get(f"/api/sessions/{record.session_id}/artifacts/{secret.sha256}")
        malformed = client.get(f"/api/sessions/{record.session_id}/artifacts/not-a-digest")

    assert allowed.status_code == 200 and allowed.content == b"full tool output"
    assert denied.status_code == 404, "an opaque provider artifact was served"
    assert malformed.status_code == 400


@pytest.mark.asyncio
async def test_player_and_theme_assets_are_served(store: SessionStore, tmp_path: Path) -> None:
    record = await _seed(store, tmp_path)

    with _client(store) as client:
        page = client.get(f"/s/{record.session_id}")
        manifest = client.get(f"/s/{record.session_id}/manifest.json").json()
        events = client.get(f"/s/{record.session_id}/events.jsonl")
        css = client.get("/theme.css")
        script = client.get(f"/s/{record.session_id}/player.js")
        blocked = client.get(f"/s/{record.session_id}/../../etc/passwd")

    assert page.status_code == 200 and "Kolega Code" in page.text
    assert f'<base href="/s/{record.session_id}/"' in page.text, "the player needs a base URL to resolve its fetches"
    assert manifest["event_count"] == 6 and manifest["turns"][0]["turn_id"] == "t1"
    assert len([line for line in events.text.splitlines() if line]) == 6
    assert css.status_code == 200 and "--kc-background" in css.text
    assert script.status_code == 200
    assert blocked.status_code == 404, "only known player assets may be served"


@pytest.mark.asyncio
async def test_themes_endpoint_exposes_every_theme(store: SessionStore, tmp_path: Path) -> None:
    with _client(store) as client:
        payload = client.get("/api/themes").json()

    assert payload["default"]
    assert len(payload["themes"]) >= 5
    for tokens in payload["themes"].values():
        assert tokens["colors"]["background"].startswith("#")


@pytest.mark.asyncio
async def test_index_lists_only_replayable_sessions(store: SessionStore, tmp_path: Path) -> None:
    replayable = await _seed(store, tmp_path)
    empty = store.create(tmp_path, "code", {"model": "test"}, title="Nothing recorded")

    with _client(store) as client:
        body = client.get("/").text

    assert replayable.session_id in body
    assert empty.session_id not in body


@pytest.mark.asyncio
async def test_openapi_schema_is_available_for_frontend_authors(store: SessionStore, tmp_path: Path) -> None:
    with _client(store) as client:
        schema = client.get("/openapi.json").json()

    paths = set(schema["paths"])
    assert "/api/sessions" in paths
    assert "/api/sessions/{session_id}/events" in paths
    assert "/api/themes" in paths


@pytest.mark.asyncio
async def test_a_scoped_server_serves_only_the_session_it_was_given(store: SessionStore, tmp_path: Path) -> None:
    """A share link must not be a key to the whole machine.

    The token gates routes, not sessions, so an unscoped server handed the link
    for one session let the holder read every other session in the store —
    including ones recorded for unrelated projects, in the clear.
    """
    from starlette.websockets import WebSocketDisconnect

    shared = await _seed(store, tmp_path / "shared")
    private = await _seed(store, tmp_path / "private")

    app = create_app(ServerConfig(store=store, session_ids=frozenset({shared.session_id})))
    with TestClient(app) as client:
        assert client.get(f"/api/sessions/{shared.session_id}/events").status_code == 200

        for path in (
            f"/api/sessions/{private.session_id}",
            f"/api/sessions/{private.session_id}/events",
            f"/api/sessions/{private.session_id}/state",
            f"/api/sessions/{private.session_id}/artifacts/{'a' * 64}",
            f"/s/{private.session_id}",
            f"/s/{private.session_id}/manifest.json",
            f"/s/{private.session_id}/events.jsonl",
        ):
            assert client.get(path).status_code == 404, f"{path} leaked an out-of-scope session"

        listed = [item["session_id"] for item in client.get("/api/sessions").json()]
        assert listed == [shared.session_id]
        assert private.session_id not in client.get("/").text

        with pytest.raises(WebSocketDisconnect) as closed:
            with client.websocket_connect(f"/api/sessions/{private.session_id}/stream"):
                pass
        assert closed.value.code == 4404


@pytest.mark.asyncio
async def test_an_unscoped_server_still_serves_every_session(store: SessionStore, tmp_path: Path) -> None:
    """Hosts embedding create_app serve their own sessions and must keep working."""
    first = await _seed(store, tmp_path / "one")
    second = await _seed(store, tmp_path / "two")

    with _client(store) as client:
        assert client.get(f"/api/sessions/{first.session_id}/events").status_code == 200
        assert client.get(f"/api/sessions/{second.session_id}/events").status_code == 200
        listed = {item["session_id"] for item in client.get("/api/sessions").json()}
        assert {first.session_id, second.session_id} <= listed


@pytest.mark.asyncio
async def test_scope_is_checked_after_a_thread_id_is_resolved(store: SessionStore, tmp_path: Path) -> None:
    """Naming a session by its thread must not route around the scope."""
    shared = await _seed(store, tmp_path / "shared")
    private = await _seed(store, tmp_path / "private")
    thread_id = getattr(private, "thread_id", None)
    if not thread_id:
        pytest.skip("sessions in this store are not addressable by thread id")

    app = create_app(ServerConfig(store=store, session_ids=frozenset({shared.session_id})))
    with TestClient(app) as client:
        assert client.get(f"/api/sessions/{thread_id}/events").status_code == 404
