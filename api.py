"""
FastAPI Client/Server - TrackExpensio Web Interface

This file contains the FastAPI web server that provides:
- Web UI (HTML/CSS/JS frontend)
- REST API endpoints for the frontend
- Integration with the MCP server (main.py) tools
- Google OAuth authentication
- Chat agent interface

The server acts as a bridge between the frontend and the MCP server tools.
"""

import base64
import os
from datetime import datetime, timezone
from typing import Any, Dict

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

from main import (
    TOOL_REGISTRY,
    add_expense,
    auth,
    mongo_manager,
    document_expense_from_rag,
)
from chat_agent import chat_with_agent
from pydantic import BaseModel, Field
from session_manager import SessionManager


async def _process_document_upload(file: UploadFile, api_key: str | None, session_id: str | None) -> Dict[str, Any]:
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    contents = await file.read()
    encoded = base64.b64encode(contents).decode("utf-8")
    result = await document_expense_from_rag(
        api_key=api_key,
        session_id=session_id,
        document_base64=encoded,
        filename=file.filename,
    )
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message", "Document parsing failed"))
    return result

session_manager = SessionManager()

app = FastAPI(title="TrackExpensio API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.oauth_states = {}

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def _google_flow() -> Flow:
    print("DEBUG GOOGLE CLIENT:", repr(os.getenv("GOOGLE_CLIENT_ID")))
    print("DEBUG GOOGLE REDIRECT:", repr(os.getenv("GOOGLE_REDIRECT_URI")))
    config = {
        "web": {
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "project_id": os.getenv("GOOGLE_PROJECT_ID", "expense-tracker"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
            "redirect_uris": [os.getenv("GOOGLE_REDIRECT_URI")],
        }
    }
    flow = Flow.from_client_config(
        config,
        scopes=[
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
            "openid",
        ],
    )
    flow.redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    return flow


class ChatRequest(BaseModel):
    message: str = Field(..., description="User query or command")
    session_id: str | None = None
    api_key: str | None = None
    history: list[dict[str, str]] | None = None


@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    """Conversational endpoint powered by Groq + TrackExpensio tools."""
    user_id = await auth.authenticate_user(api_key=payload.api_key, session_id=payload.session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        result = await chat_with_agent(
            payload.message,
            session_id=payload.session_id,
            api_key=payload.api_key,
            history=payload.history,
        )
        return result
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc


@app.post("/api/session/create")
async def create_session() -> Dict[str, Any]:
    """Create a new guest session for public access"""
    session_data = await session_manager.create_guest_session()
    return session_data


@app.post("/api/mcp/{tool_name}")
async def invoke_tool(tool_name: str, payload: Dict[str, Any]) -> Any:
    """Invoke MCP tool - accepts either api_key or session_id"""
    tool = TOOL_REGISTRY.get(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    # Authenticate user
    api_key = payload.get("api_key")
    session_id = payload.get("session_id")
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    return await tool(**payload)


@app.post("/api/doc-expense")
async def doc_expense(
    session_id: str = Form(None),
    api_key: str = Form(None), 
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    result = await _process_document_upload(file, api_key, session_id)
    return {
        "message": "Document parsed",
        "extracted": result.get("extracted"),
        "expense_id": result.get("expense_id"),
    }


@app.post("/api/chat/upload-doc")
async def chat_upload_doc(
    session_id: str = Form(None),
    api_key: str = Form(None),
    file: UploadFile = File(...),
    prompt: str = Form(None),  # Optional prompt/instruction from user
) -> Dict[str, Any]:
    result = await _process_document_upload(file, api_key, session_id)
    extracted = result.get("extracted", {})
    merchant = extracted.get("merchant") or "Unknown merchant"
    amount = extracted.get("amount") or 0
    category = extracted.get("category") or "Other"
    date = extracted.get("date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    summary = {
        "expense_id": result.get("expense_id"),
        "merchant": merchant,
        "amount": amount,
        "category": category,
        "date": date,
        "currency": extracted.get("currency"),
        "raw_text": extracted.get("raw_text"),
        "context": extracted.get("context"),
    }
    
    # Build success message
    currency = extracted.get("currency", "INR")
    currency_symbol = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "INR": "₹"}.get(currency, currency)
    
    message = f"✅ Expense added from document!\n\n"
    message += f"📄 {merchant}\n"
    message += f"💰 Amount: {currency_symbol}{amount:,.2f}\n"
    message += f"📂 Category: {category}\n"
    message += f"📅 Date: {date}\n"
    if extracted.get("notes"):
        message += f"📝 Notes: {extracted.get('notes')}\n"
    
    # If user provided a prompt, process it with the chatbot
    if prompt and prompt.strip():
        try:
            chat_result = await chat_with_agent(
                f"{prompt}\n\nDocument details: {merchant} - {amount} - {category} - {date}",
                session_id=session_id,
                api_key=api_key,
            )
            additional_info = chat_result.get("reply", "")
            if additional_info and len(additional_info) > 20:
                message += f"\n💬 {additional_info}"
        except Exception as e:
            message += f"\n\n⚠️ Note: Could not process your instruction: {str(e)}"
    
    return {"summary": summary, "message": message}


@app.get("/api/session/status")
async def session_status(session_id: str):
    info = await session_manager.get_session_status(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "is_linked": not info.get("is_guest", True),
        "user_id": info.get("user_id"),
    }


@app.get("/api/user/profile")
async def get_user_profile(session_id: str = None, api_key: str = None):
    """Get user profile information (name, email) from Google account"""
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Get Google profile from MongoDB
    profile_doc = await mongo_manager.google_collection.find_one({"user_id": user_id})
    if not profile_doc or "profile" not in profile_doc:
        return {"name": None, "email": None, "picture": None}
    
    profile = profile_doc["profile"]
    return {
        "name": profile.get("name"),
        "email": profile.get("email"),
        "picture": profile.get("picture"),
    }


@app.get("/auth/google/login")
async def google_login(session_id: str = None, api_key: str = None):
    """Optional Gmail login for data persistence"""
    if not session_id and not api_key:
        raise HTTPException(status_code=400, detail="session_id or api_key required")
    
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id, require_linked=False)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    flow = _google_flow()
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    app.state.oauth_states[state] = {"api_key": api_key, "session_id": session_id, "user_id": user_id}
    return RedirectResponse(authorization_url)


@app.get("/auth/google/callback")
async def google_callback(code: str, state: str):
    stored = app.state.oauth_states.pop(state, None)
    if not stored:
        raise HTTPException(status_code=400, detail="Invalid state")
    flow = _google_flow()
    flow.fetch_token(code=code)
    credentials = flow.credentials
    service = build("oauth2", "v2", credentials=credentials)
    profile = service.userinfo().get().execute()
    cred_payload = {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": credentials.scopes,
    }
    google_user_id = f"google_{profile.get('id', profile.get('email', 'unknown'))}"
    
    # If this was a guest session, link it to Google account
    session_id = stored.get("session_id")
    if session_id:
        await session_manager.link_session_to_google(session_id, google_user_id)
    
    await mongo_manager.upsert_google_profile(google_user_id, profile, cred_payload)
    
    # Return HTML that closes popup and notifies parent window
    if session_id:
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Connected</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    text-align: center;
                    padding: 50px;
                    background: #1a1a1a;
                    color: white;
                }}
                h2 {{ color: #10b981; }}
            </style>
        </head>
        <body>
            <h2>✅ Gmail Connected Successfully</h2>
            <p>Your data will now be saved permanently.</p>
            <p>Closing window...</p>
            <script>
                // Notify parent window and close
                if (window.opener) {{
                    window.opener.postMessage({{
                        type: 'google_login_success',
                        session_id: '{session_id}'
                    }}, '*');
                    setTimeout(() => window.close(), 1000);
                }} else {{
                    // If not popup, redirect
                    window.location.href = '/?google_login=success&session_id={session_id}';
                }}
            </script>
        </body>
        </html>
        """)
    else:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Connected</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    text-align: center;
                    padding: 50px;
                    background: #1a1a1a;
                    color: white;
                }
                h2 { color: #10b981; }
            </style>
        </head>
        <body>
            <h2>✅ Gmail Connected Successfully</h2>
            <p>Your data will now be saved permanently.</p>
            <p>You can close this tab.</p>
        </body>
        </html>
        """)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main chatbot UI"""
    return templates.TemplateResponse("expense_tracker.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    """Alias for home page"""
    return templates.TemplateResponse("expense_tracker.html", {"request": request})


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("FASTAPI_PORT", "8080"))
    print(f"\n🚀 Starting TrackExpensio Web Server...")
    print(f"📱 Open your browser and go to: http://localhost:{port}")
    print(f"💬 Chatbot available at: http://localhost:{port}/chat")
    print(f"\n⚠️  Note: Use 'localhost' or '127.0.0.1', NOT '0.0.0.0'\n")
    uvicorn.run(app, host="0.0.0.0", port=port)

