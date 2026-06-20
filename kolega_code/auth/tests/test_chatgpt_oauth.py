import asyncio
import base64
import hashlib
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from kolega_code.auth import chatgpt_oauth as flow
from kolega_code.auth import constants
from kolega_code.auth.tests.test_tokens import make_id_token


def test_generate_pkce_pair_is_valid_s256() -> None:
    verifier, challenge = flow.generate_pkce_pair()
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert challenge == expected
    assert "=" not in verifier and "=" not in challenge


def test_build_authorize_url_carries_required_params() -> None:
    url = flow.build_authorize_url("http://localhost:1455/auth/callback", "chal", "st4te", client_id="cid")
    qs = parse_qs(urlsplit(url).query)
    assert url.startswith(constants.AUTHORIZE_URL + "?")
    assert qs["response_type"] == ["code"]
    assert qs["client_id"] == ["cid"]
    assert qs["redirect_uri"] == ["http://localhost:1455/auth/callback"]
    assert qs["code_challenge"] == ["chal"]
    assert qs["code_challenge_method"] == ["S256"]
    assert qs["state"] == ["st4te"]
    assert qs["scope"] == [constants.SCOPES]
    assert qs["codex_cli_simplified_flow"] == ["true"]


def test_is_plan_eligible() -> None:
    assert flow.is_plan_eligible("pro")
    assert flow.is_plan_eligible("plus")
    assert not flow.is_plan_eligible("free")
    # Unknown/missing plan -> permissive: only an explicit "free" is blocked, the
    # backend is the authoritative gate, so we don't lock out a paid user over a
    # missing claim.
    assert flow.is_plan_eligible(None)


@pytest.mark.asyncio
async def test_exchange_code_returns_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        assert "grant_type=authorization_code" in body
        assert "code_verifier=verif" in body
        return httpx.Response(
            200,
            json={
                "access_token": "at",
                "refresh_token": "rt",
                "id_token": make_id_token(plan_type="pro"),
                "expires_in": 3600,
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tokens = await flow.exchange_code("the-code", "verif", "http://localhost:1455/auth/callback", http_client=client)
    assert tokens.access_token == "at"
    assert tokens.plan_type == "pro"
    await client.aclose()


@pytest.mark.asyncio
async def test_exchange_code_without_access_token_raises() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"error": "x"})))
    with pytest.raises(flow.LoginError):
        await flow.exchange_code("c", "v", "http://localhost:1455/auth/callback", http_client=client)
    await client.aclose()


async def _drive_browser_callback(captured: dict, code: str = "the-code", state: str | None = None) -> None:
    """Simulate the browser redirect by GETing the loopback callback once it's up."""
    for _ in range(500):
        if "url" in captured:
            break
        await asyncio.sleep(0.01)
    qs = parse_qs(urlsplit(captured["url"]).query)
    redirect = qs["redirect_uri"][0]
    use_state = state if state is not None else qs["state"][0]
    async with httpx.AsyncClient() as browser:
        resp = await browser.get(f"{redirect}?code={code}&state={use_state}")
    captured["callback_status"] = resp.status_code


@pytest.mark.asyncio
async def test_run_login_flow_end_to_end() -> None:
    captured: dict = {}
    token_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                json={
                    "access_token": "live-at",
                    "refresh_token": "live-rt",
                    "id_token": make_id_token(account_id="acct_777", plan_type="pro"),
                    "expires_in": 3600,
                },
            )
        )
    )

    login = asyncio.create_task(
        flow.run_login_flow(
            ports=(54551, 54552),
            open_browser=False,
            on_url=lambda u: captured.__setitem__("url", u),
            http_client=token_client,
        )
    )
    await _drive_browser_callback(captured)
    tokens = await login

    assert tokens.access_token == "live-at"
    assert tokens.account_id == "acct_777"
    assert captured["callback_status"] == 200
    await token_client.aclose()


@pytest.mark.asyncio
async def test_run_login_flow_rejects_state_mismatch() -> None:
    captured: dict = {}
    token_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))

    login = asyncio.create_task(
        flow.run_login_flow(
            ports=(54553,),
            open_browser=False,
            on_url=lambda u: captured.__setitem__("url", u),
            http_client=token_client,
        )
    )
    await _drive_browser_callback(captured, state="wrong-state")
    with pytest.raises(flow.LoginError):
        await login
    assert captured["callback_status"] == 400
    await token_client.aclose()


@pytest.mark.asyncio
async def test_run_login_flow_rejects_free_plan() -> None:
    captured: dict = {}
    token_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                json={
                    "access_token": "at",
                    "refresh_token": "rt",
                    "id_token": make_id_token(plan_type="free"),
                    "expires_in": 3600,
                },
            )
        )
    )

    login = asyncio.create_task(
        flow.run_login_flow(
            ports=(54554,),
            open_browser=False,
            on_url=lambda u: captured.__setitem__("url", u),
            http_client=token_client,
        )
    )
    await _drive_browser_callback(captured)
    with pytest.raises(flow.LoginError, match="free"):
        await login
    await token_client.aclose()
