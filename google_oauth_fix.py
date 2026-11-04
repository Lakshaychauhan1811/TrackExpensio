"""
TrackExpensio - Google sign-in that works on Vercel (serverless-safe)

Why the old flow breaks on Vercel
---------------------------------
1. Config/credentials were read from files or localhost env values that do not
   exist on Vercel, so /auth/google/login crashes with a bare 500.
2. The PKCE code_verifier was kept in server memory. Vercel runs each request
   on a different short-lived instance, so the callback cannot find it.

What this file does
-------------------
- Reads GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET from environment variables.
- Builds the redirect URI from your real domain (never localhost on Vercel).
- Keeps the PKCE verifier + state in a signed, httponly cookie (no server memory).
- Shows readable errors instead of "Internal Server Error".

How to use (api.py)
-------------------
    from google_oauth_fix import router as google_router
    app = FastAPI(...)
    app.include_router(google_router)   # put this BEFORE the old /auth/google routes

Required env vars on Vercel (then Redeploy):
    GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, SECRET_KEY,
    GOOGLE_REDIRECT_URI=https://track-expensio.vercel.app/auth/google/callback
"""

import html
import logging
import os
import secrets
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from google_auth_oauthlib.flow import Flow
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

logger = logging.getLogger(__name__)
router = APIRouter()

# Google may return extra scopes that were granted earlier; do not crash on that.
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

# Change these if your old flow asked for more (for example Gmail or Drive).
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

COOKIE_NAME = "g_oauth"
COOKIE_MAX_AGE = 600  # seconds the user has to finish signing in


def _on_vercel() -> bool:
    return bool(os.getenv("VERCEL"))


def _signer() -> URLSafeTimedSerializer:
    secret = os.getenv("SECRET_KEY")
    if not secret:
        raise HTTPException(500, "SECRET_KEY is not set in the Vercel environment variables.")
    return URLSafeTimedSerializer(secret, salt="google-oauth")


def _redirect_uri(request: Request) -> str:
    configured = os.getenv("GOOGLE_REDIRECT_URI", "").strip()
    is_local_value = "127.0.0.1" in configured or "localhost" in configured
    if configured and not (_on_vercel() and is_local_value):
        return configured
    host = request.headers.get("x-forwarded-host") or request.url.netloc
    scheme = "https" if _on_vercel() else request.url.scheme
    return f"{scheme}://{host}/auth/google/callback"


def _flow(redirect_uri: str, code_verifier: str) -> Flow:
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise HTTPException(
            500,
            "GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET are not set in the Vercel "
            "environment variables. Add them, then redeploy.",
        )
    config = {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri],
        }
    }
    flow = Flow.from_client_config(
        config,
        scopes=SCOPES,
        code_verifier=code_verifier,
        autogenerate_code_verifier=False,
    )
    flow.redirect_uri = redirect_uri
    return flow


def _result_page(ok: bool, message: str) -> HTMLResponse:
    color = "#3fb950" if ok else "#f85149"
    body = f"""<!doctype html>
<html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TrackExpensio</title></head>
<body style="font-family:system-ui,sans-serif;background:#0d1117;color:#e6edf3;
display:flex;min-height:100vh;align-items:center;justify-content:center;
margin:0;padding:24px;text-align:center">
<div>
<h2 style="color:{color}">{html.escape(message)}</h2>
<p>You can close this window and tap <b>I've connected</b> in TrackExpensio.</p>
<p><a style="color:#58a6ff" href="/">Back to TrackExpensio</a></p>
</div>
<script>setTimeout(function(){{ window.close(); }}, 1500);</script>
</body></html>"""
    return HTMLResponse(body, status_code=200 if ok else 400)


async def _link_session(session_id: str, google_user_id: str) -> bool:
    # If api.py already has a shared SessionManager instance, use that instead.
    from session_manager import SessionManager

    return await SessionManager().link_session_to_google(session_id, google_user_id)


@router.get("/auth/google/login")
async def google_login(request: Request, session_id: str = ""):
    if not session_id:
        raise HTTPException(400, "session_id is required.")

    redirect_uri = _redirect_uri(request)
    verifier = secrets.token_urlsafe(64)
    state = secrets.token_urlsafe(24)

    flow = _flow(redirect_uri, verifier)
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        state=state,
    )

    response = RedirectResponse(auth_url)
    response.set_cookie(
        COOKIE_NAME,
        _signer().dumps({"sid": session_id, "v": verifier, "st": state}),
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        secure=_on_vercel(),
        samesite="lax",
    )
    return response


@router.get("/auth/google/callback")
async def google_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    if error:
        return _result_page(False, f"Google sign-in was cancelled ({error}).")
    if not code:
        return _result_page(False, "Google did not return a sign-in code. Please try again.")

    raw = request.cookies.get(COOKIE_NAME)
    if not raw:
        return _result_page(False, "Sign-in expired. Please start again from TrackExpensio.")

    try:
        data = _signer().loads(raw, max_age=COOKIE_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return _result_page(False, "Sign-in expired. Please start again from TrackExpensio.")

    if not secrets.compare_digest(str(data.get("st", "")), state):
        return _result_page(False, "Sign-in could not be verified. Please try again.")

    try:
        flow = _flow(_redirect_uri(request), data["v"])
        flow.fetch_token(code=code)
        creds = flow.credentials

        from google.auth.transport import requests as g_requests
        from google.oauth2 import id_token

        info = id_token.verify_oauth2_token(
            creds.id_token, g_requests.Request(), os.getenv("GOOGLE_CLIENT_ID")
        )
        google_user_id = info["sub"]

        linked = await _link_session(data["sid"], google_user_id)
        if not linked:
            return _result_page(False, "Session not found. Reload TrackExpensio and try again.")
    except HTTPException:
        raise
    except Exception as exc:  # shows in Vercel logs with the full traceback
        logger.exception("Google callback failed")
        return _result_page(False, f"Sign-in failed ({type(exc).__name__}). Check the Vercel logs.")

   response = _result_page(True, "Google account connected")
   response.delete_cookie(COOKIE_NAME)
   return response
