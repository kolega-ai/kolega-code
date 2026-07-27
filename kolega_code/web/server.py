"""Local session server: HTTP and WebSocket access to recorded sessions.

Ships to every user rather than behind an extra, because a frontend surface that
has to be installed separately is one that nothing can rely on. The generated
OpenAPI schema at ``/docs`` is the contract third-party frontends build against.

Scope is read-only: list sessions, read an event range, follow a session live,
fetch an artifact, and serve the player. That covers the sharing model — a second
client attaches and watches, live or after the fact — without pretending to offer
interactive control, which needs the headless session runtime and a control lease
and is deliberately absent rather than stubbed.

Security posture: bound to loopback unless told otherwise, and an optional bearer
token gates every route. Exposing a session beyond the machine is left to tools
users already have (Tailscale, ngrok, an SSH tunnel), so this never opens a port
to the world on its own.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from starlette.staticfiles import StaticFiles

from kolega_code.cli import theme as cli_theme
from kolega_code.cli.session_event_store import FileArtifactStore, FileSessionEventStore
from kolega_code.cli.session_store import SessionStore, SessionStoreError as CliSessionStoreError
from kolega_code.events import ArtifactPurpose, ArtifactRef
from kolega_code.session.projection import replay
from kolega_code.session.store import SessionStoreError

from .theme_css import stylesheet

ASSET_DIR = Path(__file__).parent / "assets"

#: Ceiling on a single event-range response, so one request cannot pull an entire
#: long session into memory. Clients page with from_seq.
MAX_EVENTS_PER_READ = 5_000


@dataclass
class ServerConfig:
    store: SessionStore
    token: Optional[str] = None
    #: Sessions this server is allowed to expose. ``None`` means every session in
    #: the store, which is right for a host serving its own sessions.
    #:
    #: A share link is the other case. Its token gates *routes*, so without a
    #: scope the link handed to one person reads every session on the machine —
    #: including ones recorded for unrelated projects. ``ShareServer`` sets this
    #: to the single session being shared.
    session_ids: Optional[frozenset[str]] = None


def _digest_ok(digest: str) -> bool:
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def create_app(config: ServerConfig) -> FastAPI:
    """Build the session server application."""
    app = FastAPI(
        title="Kolega Code session server",
        version="1",
        summary="Read and follow recorded agent sessions.",
        description=(
            "Every session is an ordered event stream. Read a range with "
            "`GET /api/sessions/{id}/events`, or attach to the WebSocket at "
            "`/api/sessions/{id}/stream?from_seq=N` to replay the backlog from `N` "
            "and then follow live appends with no gap and no duplicate.\n\n"
            "Sequence numbers are strictly increasing but **not contiguous** — a "
            "session shares one sequence space with its provider-facing records — so "
            "page with the `next_from_seq` a response gives you rather than "
            "computing `seq + 1` yourself.\n\n"
            "The stream endpoint is absent from the schema below because OpenAPI "
            "cannot describe WebSocket routes; it is nonetheless the primary way to "
            "follow a live session."
        ),
    )
    app.state.config = config

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        query_token = request.query_params.get("token")
        if not _authorized(
            config,
            request.headers.get("authorization"),
            query_token,
            request.cookies.get(TOKEN_COOKIE),
        ):
            return JSONResponse({"detail": "Not authorized"}, status_code=401)
        response = await call_next(request)
        if config.token and query_token == config.token and request.cookies.get(TOKEN_COOKIE) != config.token:
            # Hand the link's token to the browser so the page's own subresources
            # load. Host-only, not readable from script, and not sent on
            # cross-site requests.
            response.set_cookie(
                TOKEN_COOKIE,
                config.token,
                httponly=True,
                samesite="strict",
                path="/",
            )
        return response

    # -- Discovery ---------------------------------------------------------

    @app.get("/api/sessions", summary="List recorded sessions")
    async def list_sessions(
        project: Optional[str] = Query(
            None,
            description=(
                "Filter to sessions recorded for this project, matched exactly against the "
                "`project_path` this endpoint returns. Pass back the value you were given."
            ),
        ),
    ) -> list[dict[str, Any]]:
        # Matched as an opaque string, never turned into a filesystem path.
        # Resolving it would put untrusted request data into a path expression —
        # and touch the filesystem to follow symlinks — for a filter that only
        # ever compares against metadata already in memory.
        wanted = _normalized_project(project)
        summaries: list[dict[str, Any]] = []
        for record in config.store.list():
            if not _in_scope(config, record.session_id):
                continue
            if wanted is not None and _normalized_project(record.project_path) != wanted:
                continue
            meta = await _head(config, record.session_id)
            summaries.append(
                {
                    "session_id": record.session_id,
                    "title": record.title,
                    "project_path": record.project_path,
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                    "interaction_mode": record.interaction_mode,
                    "event_count": meta.event_count if meta else 0,
                    "last_seq": meta.last_seq if meta else 0,
                    "duration_ms": meta.duration_ms if meta else 0,
                    "status": meta.status if meta else "empty",
                    # Sessions recorded before the event spine have history but no
                    # presentation events; a client should not offer replay for them.
                    "replayable": bool(meta and meta.event_count),
                }
            )
        return summaries

    @app.get("/api/sessions/{session_id}", summary="Describe one session")
    async def get_session(session_id: str) -> dict[str, Any]:
        record = _record(config, session_id)
        meta = await _head(config, record.session_id)
        return {
            "session_id": record.session_id,
            "title": record.title,
            "project_path": record.project_path,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "event_count": meta.event_count if meta else 0,
            "last_seq": meta.last_seq if meta else 0,
            "duration_ms": meta.duration_ms if meta else 0,
            "status": meta.status if meta else "empty",
            "turns": [
                {
                    "turn_id": marker.turn_id,
                    "status": marker.status,
                    "user_text": marker.user_text[:200],
                    "elapsed_ms": marker.started_ms,
                    "seq": marker.started_seq,
                    "ended_ms": marker.ended_ms,
                }
                for marker in replay(await _events(config, record.session_id)).turns
            ],
        }

    # -- Events ------------------------------------------------------------

    @app.get("/api/sessions/{session_id}/events", summary="Read an event range")
    async def get_events(
        session_id: str,
        from_seq: int = Query(1, ge=1, description="Inclusive lower bound on seq."),
        to_seq: Optional[int] = Query(None, ge=1, description="Inclusive upper bound on seq."),
        limit: int = Query(MAX_EVENTS_PER_READ, ge=1, le=MAX_EVENTS_PER_READ),
    ) -> dict[str, Any]:
        record = _record(config, session_id)
        events = await _events(config, record.session_id, from_seq=from_seq, to_seq=to_seq)
        page = events[:limit]
        return {
            "session_id": record.session_id,
            "events": [event.model_dump(mode="json") for event in page],
            # Sequence numbers are sparse, so a client cannot compute the next
            # cursor itself; hand it back explicitly.
            "next_from_seq": (page[-1].seq or 0) + 1 if page else from_seq,
            "complete": len(page) == len(events),
        }

    @app.get("/api/sessions/{session_id}/state", summary="Read the folded presentation state")
    async def get_state(session_id: str, to_seq: Optional[int] = Query(None, ge=1)) -> dict[str, Any]:
        """Server-side fold, for clients that would rather not implement one."""
        record = _record(config, session_id)
        events = await _events(config, record.session_id, to_seq=to_seq)
        return replay(events).to_dict()

    @app.websocket("/api/sessions/{session_id}/stream")
    async def stream(websocket: WebSocket, session_id: str) -> None:
        """Replay the backlog from ``from_seq``, then follow live appends."""
        token = websocket.query_params.get("token")
        if not _authorized(
            config,
            websocket.headers.get("authorization"),
            token,
            websocket.cookies.get(TOKEN_COOKIE),
        ):
            await websocket.close(code=4401)
            return
        try:
            record = _record(config, session_id)
        except HTTPException:
            await websocket.close(code=4404)
            return

        try:
            from_seq = max(1, int(websocket.query_params.get("from_seq") or 1))
        except ValueError:
            from_seq = 1

        await websocket.accept()
        store = _event_store(config, record.session_id)
        try:
            async for event in store.tail(record.session_id, from_seq=from_seq):
                await websocket.send_text(json.dumps(event.model_dump(mode="json"), separators=(",", ":")))
        except (WebSocketDisconnect, asyncio.CancelledError):
            return
        except SessionStoreError as exc:
            await websocket.close(code=4400, reason=str(exc)[:120])

    # -- Artifacts ---------------------------------------------------------

    @app.get("/api/sessions/{session_id}/artifacts/{digest}", summary="Read an artifact")
    async def get_artifact(session_id: str, digest: str) -> Response:
        """Fetch a payload an event points at: an image, or a long tool result.

        Scoped to the session and gated on purpose, so this cannot be used to
        read opaque provider state out of a session directory.
        """
        record = _record(config, session_id)
        if not _digest_ok(digest):
            raise HTTPException(status_code=400, detail="Malformed artifact digest")
        events = await _events(config, record.session_id)
        ref = next(
            (
                candidate
                for event in events
                for candidate in event.artifacts
                if candidate.sha256 == digest and candidate.purpose in ArtifactPurpose.SHAREABLE
            ),
            None,
        )
        if ref is None:
            # Either unknown, or a purpose that is never served. Both are "not
            # found" from a client's perspective; distinguishing them would leak
            # the existence of provider-internal payloads.
            raise HTTPException(status_code=404, detail="No such artifact in this session")
        journal = config.store.journal(record.session_id)
        try:
            data = await FileArtifactStore(journal).open(ref)
        except SessionStoreError as exc:
            raise HTTPException(status_code=410, detail=str(exc)) from exc
        return Response(content=data, media_type=ref.media_type or "application/octet-stream")

    # -- Theming and player ------------------------------------------------

    @app.get("/api/themes", summary="Design tokens for every theme")
    async def get_themes() -> dict[str, Any]:
        return {"default": cli_theme.default_theme_slug(), "themes": cli_theme.all_web_tokens()}

    @app.get("/theme.css", include_in_schema=False)
    async def get_theme_css() -> Response:
        return Response(content=stylesheet(), media_type="text/css")

    @app.get("/s/{session_id}", response_class=HTMLResponse, summary="Replay player for a session")
    async def player(session_id: str) -> Response:
        record = _record(config, session_id)
        html = (ASSET_DIR / "player.html").read_text(encoding="utf-8")
        # The player fetches manifest.json and events.jsonl relative to its own
        # URL, so a served session exposes those two names under /s/<id>/.
        html = html.replace('href="theme.css"', 'href="/theme.css"')
        return HTMLResponse(content=html.replace("<head>", f'<head>\n    <base href="/s/{record.session_id}/" />', 1))

    @app.get("/s/{session_id}/manifest.json", include_in_schema=False)
    async def player_manifest(session_id: str) -> dict[str, Any]:
        record = _record(config, session_id)
        events = await _events(config, record.session_id)
        state = replay(events)
        meta = await _head(config, record.session_id)
        return {
            "format": 1,
            "session_id": record.session_id,
            "title": record.title or record.session_id,
            "event_count": len(events),
            "duration_ms": max((event.elapsed_ms for event in events), default=0),
            # A served session may still be running. The player follows the
            # stream endpoint from last_seq when it is, and a static bundle omits
            # both keys, which is what makes a bundle strictly a replay.
            "status": meta.status if meta else "empty",
            "last_seq": (events[-1].seq or 0) if events else 0,
            "stream": f"/api/sessions/{record.session_id}/stream",
            "theme": cli_theme.default_theme_slug(),
            "themes": list(cli_theme.all_web_tokens().keys()),
            "turns": [
                {
                    "turn_id": marker.turn_id,
                    "user_text": marker.user_text[:200],
                    "elapsed_ms": marker.started_ms,
                    "seq": marker.started_seq,
                    "status": marker.status,
                    "ended_ms": marker.ended_ms,
                }
                for marker in state.turns
            ],
        }

    @app.get("/s/{session_id}/events.jsonl", include_in_schema=False)
    async def player_events(session_id: str) -> Response:
        record = _record(config, session_id)
        events = await _events(config, record.session_id)
        body = "\n".join(
            json.dumps(event.model_dump(mode="json"), separators=(",", ":"), ensure_ascii=False) for event in events
        )
        return PlainTextResponse(content=body, media_type="application/x-ndjson")

    @app.get("/s/{session_id}/{asset}", include_in_schema=False)
    async def player_asset(session_id: str, asset: str) -> Response:
        # Deliberately not scoped to a session: these are the player's own static
        # files, identical for every session and carrying no session data. The id
        # is in the path only because the player resolves them relative to its
        # own URL, and answering for an unknown id reveals nothing.
        if asset not in {"player.js", "fold.js", "player.css", "player.html"}:
            raise HTTPException(status_code=404, detail="Unknown asset")
        media = "text/javascript" if asset.endswith(".js") else "text/css" if asset.endswith(".css") else "text/html"
        return FileResponse(ASSET_DIR / asset, media_type=media)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> Response:
        records = [record for record in config.store.list() if _in_scope(config, record.session_id)]
        rows = []
        for record in records:
            meta = await _head(config, record.session_id)
            if not (meta and meta.event_count):
                continue
            label = record.title or record.session_id
            rows.append(
                f'<li><a href="/s/{record.session_id}">{_escape(label)}</a> <span>{meta.event_count} events</span></li>'
            )
        body = "".join(rows) or "<li><em>No replayable sessions recorded yet.</em></li>"
        return HTMLResponse(
            "<!doctype html><html data-theme='"
            + cli_theme.default_theme_slug()
            + "'><head><meta charset='utf-8'><title>Kolega Code sessions</title>"
            "<link rel='stylesheet' href='/theme.css'>"
            "<style>body{background:var(--kc-background);color:var(--kc-foreground);"
            "font-family:var(--kc-font-mono);padding:3rem;line-height:1.7}"
            "h1{font-size:1rem;letter-spacing:.14em;text-transform:uppercase;color:var(--kc-primary)}"
            "ul{list-style:none;padding:0}li{padding:.35rem 0;border-bottom:1px solid #ffffff14}"
            "a{color:var(--kc-user);text-decoration:none}a:hover{text-decoration:underline}"
            "span{color:var(--kc-text-muted);font-size:.8rem}</style></head>"
            f"<body><h1>Kolega Code sessions</h1><ul>{body}</ul></body></html>"
        )

    if ASSET_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=ASSET_DIR), name="assets")

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _normalized_project(value: Optional[str]) -> Optional[str]:
    """Normalize a project path for comparison, without touching the filesystem.

    Purely lexical on purpose. The request-supplied value is compared against
    recorded metadata and is never opened, joined, or resolved, so it must not
    reach any filesystem API.
    """
    if not value:
        return None
    trimmed = value.strip().rstrip("/")
    return trimmed or "/"


#: Name of the cookie that carries a query-string token across subresource loads.
TOKEN_COOKIE = "kc_session_token"


def _authorized(
    config: ServerConfig,
    header: Optional[str],
    query_token: Optional[str],
    cookie_token: Optional[str] = None,
) -> bool:
    """Accept the token from a header, the query string, or the handshake cookie.

    A shared link can only carry the token in its query string, but the browser
    resolves the player's script, stylesheet, manifest, and event log as plain
    relative URLs with no query. Without the cookie those subresources are all
    unauthorized and the page renders blank, so a token-protected server is only
    usable by API clients.
    """
    if not config.token:
        return True
    # An explicitly presented credential has to be the right one. Falling back to
    # the cookie when a caller sent a wrong header or query token would report
    # success for a credential the caller actually got wrong.
    if header is not None:
        return header.startswith("Bearer ") and header[7:] == config.token
    if query_token is not None:
        return query_token == config.token
    return cookie_token == config.token


def _in_scope(config: ServerConfig, session_id: str) -> bool:
    return config.session_ids is None or session_id in config.session_ids


def _record(config: ServerConfig, session_id: str):
    """Resolve a session, refusing anything this server is not scoped to.

    The scope check runs on the *resolved* id, because this accepts a thread id
    as well as a session id and a caller must not be able to reach an
    out-of-scope session by naming its thread instead.

    Out of scope is reported as 404 rather than 403: distinguishing "exists but
    not yours" from "does not exist" would confirm which other sessions this
    machine has recorded, which is exactly what the scope is for.
    """
    try:
        record = config.store.load_session_or_thread(session_id)
    except CliSessionStoreError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not _in_scope(config, record.session_id):
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return record


def _event_store(config: ServerConfig, session_id: str) -> FileSessionEventStore:
    return FileSessionEventStore(config.store.journal(session_id))


async def _events(
    config: ServerConfig,
    session_id: str,
    *,
    from_seq: int = 1,
    to_seq: Optional[int] = None,
):
    try:
        return await _event_store(config, session_id).read(session_id, from_seq=from_seq, to_seq=to_seq)
    except SessionStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _head(config: ServerConfig, session_id: str):
    try:
        return await _event_store(config, session_id).head(session_id)
    except Exception:
        # A session directory that predates the event log, or one being written
        # right now, must not break a listing.
        return None


def artifact_url(session_id: str, ref: ArtifactRef) -> str:
    """Canonical read path for an artifact reference."""
    return f"/api/sessions/{session_id}/artifacts/{ref.sha256}"
