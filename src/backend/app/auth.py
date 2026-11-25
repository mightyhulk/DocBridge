"""Patient sign-in with Google (OAuth 2.0 authorization-code flow, server side) or optional web email login.

The browser is sent to Google, Google redirects back to /api/auth/google/callback with a one-time code, and this server
exchanges it (using the client secret, which never reaches the browser) for the patient's name, email and picture.
A signed, HttpOnly cookie then keeps them signed in.
"""
import base64
import hashlib
import hmac
import json
import secrets
import time
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from .config import get_settings

COOKIE = "hva_session"
STATE_COOKIE = "hva_oauth_state"
SESSION_TTL_S = 7 * 24 * 3600
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"

router = APIRouter(prefix="/api/auth")


def google_configured() -> bool:
    s = get_settings()
    return bool(s.google_client_id and s.google_client_secret)


def enabled() -> bool:
    s = get_settings()
    return (google_configured() or s.local_auth) and not s.auth_disabled


def _secret() -> bytes:
    s = get_settings()
    if s.session_secret:
        return s.session_secret.encode()
    f = s.data_dir / "session.key"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(secrets.token_hex(32))
    return f.read_text().strip().encode()


def _sign(payload: dict) -> str:
    body = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
    mac = hmac.new(_secret(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{mac}"


def _unsign(token: str) -> dict | None:
    try:
        body, mac = token.rsplit(".", 1)
        if not hmac.compare_digest(mac, hmac.new(_secret(), body.encode(), hashlib.sha256).hexdigest()):
            return None
        data = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        return data if data.get("exp", 0) > time.time() else None
    except Exception:
        return None


def current_user(request: Request) -> dict | None:
    token = request.cookies.get(COOKIE)
    return _unsign(token) if token else None


def require_user(request: Request) -> dict | None:
    """Dependency: 401 unless a patient is signed in. When sign-in is not configured it lets everyone through."""
    if not enabled():
        return None
    user = current_user(request)
    if not user:
        raise HTTPException(401, "Please sign in with Google to continue.")
    return user


def _redirect_uri(request: Request) -> str:
    if get_settings().google_redirect_uri:
        return get_settings().google_redirect_uri
    base = get_settings().public_url.rstrip("/") or str(request.base_url).rstrip("/")
    return f"{base}/api/auth/google/callback"


def is_secure_request(request: Request) -> bool:
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    return proto == "https"


@router.get("/me")
async def me(request: Request):
    user = current_user(request)
    return {"enabled": enabled(),
            "user": {k: user.get(k) for k in ("name", "email", "picture")} if user else None}


def _is_local_host(request: Request) -> bool:
    return (request.url.hostname or "") in ("localhost", "127.0.0.1", "::1")


@router.get("/methods")
async def methods(request: Request):
    """Which sign-in options this website offers to this browser."""
    from urllib.parse import urlparse
    st = get_settings()
    public_host = urlparse(st.public_url).hostname if st.public_url else None
    here = request.url.hostname or ""
    url = None
    if google_configured() and not st.auth_disabled:
        if _is_local_host(request) or not public_host or here == public_host:
            url = "/api/auth/google/login"
    return {"google": bool(url), "google_url": url, "local": st.local_auth}


class LocalLogin(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    email: str = Field(min_length=3, max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@router.post("/local")
async def local_login(body: LocalLogin, request: Request):
    if not get_settings().local_auth or get_settings().auth_disabled:
        raise HTTPException(503, "Name + email sign-in is not enabled on this server.")
    email = body.email.strip().lower()
    session = _sign({"sub": "local:" + email, "email": email, "name": body.name.strip(), "picture": "",
                     "exp": int(time.time()) + SESSION_TTL_S})
    resp = JSONResponse({"ok": True})
    resp.set_cookie(COOKIE, session, max_age=SESSION_TTL_S, httponly=True, samesite="lax", secure=is_secure_request(request))
    return resp


@router.get("/google/login")
async def google_login(request: Request):
    if not google_configured() or get_settings().auth_disabled:
        raise HTTPException(503, "Google sign-in is not configured on this server.")
    state = secrets.token_urlsafe(24)
    url = AUTH_URL + "?" + urlencode({
        "client_id": get_settings().google_client_id, "redirect_uri": _redirect_uri(request), "response_type": "code",
        "scope": "openid email profile", "state": state, "prompt": "select_account",
    })
    resp = RedirectResponse(url)
    resp.set_cookie(STATE_COOKIE, state, max_age=600, httponly=True, samesite="lax", secure=is_secure_request(request))
    return resp


@router.get("/callback/google")
@router.get("/google/callback")
async def google_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    if error or not code:
        return RedirectResponse("/?login_error=" + (error or "cancelled"))
    expected = request.cookies.get(STATE_COOKIE, "")
    if not expected or not hmac.compare_digest(expected, state):
        return RedirectResponse("/?login_error=state_mismatch")
    s = get_settings()
    import logging
    log = logging.getLogger("hva")
    try:
        async with httpx.AsyncClient(timeout=15) as http:
            tok = await http.post(TOKEN_URL, data={
                "code": code, "client_id": s.google_client_id, "client_secret": s.google_client_secret,
                "redirect_uri": _redirect_uri(request), "grant_type": "authorization_code"})
            if tok.status_code != 200:
                log.warning("Google token exchange failed: %s %s", tok.status_code, tok.text[:300])
                reason = tok.json().get("error", "") if tok.headers.get("content-type", "").startswith("application/json") else ""
                return RedirectResponse("/?login_error=" + ("code_used" if reason == "invalid_grant" else "google_rejected"))
            info = await http.get(USERINFO_URL, headers={"Authorization": f"Bearer {tok.json()['access_token']}"})
            if info.status_code != 200:
                log.warning("Google userinfo failed: %s %s", info.status_code, info.text[:300])
                return RedirectResponse("/?login_error=google_rejected")
            profile = info.json()
    except (httpx.HTTPError, KeyError, ValueError) as e:
        log.warning("Google sign-in could not reach Google: %s: %s", type(e).__name__, e)
        return RedirectResponse("/?login_error=google_unreachable")
    if not profile.get("email") or profile.get("email_verified") is False:
        return RedirectResponse("/?login_error=email_not_verified")
    session = _sign({"sub": profile.get("sub"), "email": profile["email"], "name": profile.get("name") or profile["email"],
                     "picture": profile.get("picture", ""), "exp": int(time.time()) + SESSION_TTL_S})
    resp = RedirectResponse("/")
    resp.set_cookie(COOKIE, session, max_age=SESSION_TTL_S, httponly=True, samesite="lax", secure=is_secure_request(request))
    resp.delete_cookie(STATE_COOKIE)
    return resp


@router.post("/logout")
async def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE)
    return resp
