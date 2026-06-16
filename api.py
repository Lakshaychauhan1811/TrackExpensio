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
import asyncio
import hashlib
import secrets
import smtplib
import time
from contextlib import asynccontextmanager
from email.message import EmailMessage
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, AsyncGenerator

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
import io
import tempfile
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

from main import (
    TOOL_REGISTRY,
    add_expense,
    add_income,
    list_expenses,
    get_expenses_overview,
    generate_financial_report,
    sync_bank_transactions,
    add_savings_goal,
    set_savings_goal,
    delete_savings_goal,
    list_savings_goals,
    track_savings_progress,
    set_budget,
    check_budget_status,
    list_income,
    get_income_summary,
    yahoo_finance,
    auth,
    db,
    mongo_manager,
    document_expense_from_rag,
)
from services.bank_plaid import plaid_service
from services.notifications import notification_service
from services.bill_reminder_job import process_bill_reminders
from services.gmail_bills import gmail_bill_service
from services.account_aggregator import aa_service
from services.sms_parser import parse_sms_alert, parse_sms_batch
from chat_agent import chat_with_agent
from stock_analyzer import analyze_stock
from pydantic import BaseModel, Field
from session_manager import SessionManager
from security import security_manager
from audit_logger import audit_logger, AuditAction, DETAILED_AUDIT_TOOLS, parse_audit_dates
from fx_timing import fx_manager


def _server_host() -> str:
    """127.0.0.1 for local dev; 0.0.0.0 when PORT is set (Render/Heroku)."""
    explicit = os.getenv("FASTAPI_HOST")
    if explicit:
        return explicit
    return "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"


def _server_port() -> int:
    return int(os.getenv("FASTAPI_PORT") or os.getenv("PORT") or "8080")


def _cors_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "*")
    if raw.strip() == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


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
    try:
        await audit_logger.log(
            user_id=user_id,
            action=AuditAction.DOCUMENT_UPLOADED,
            details={
                "filename": file.filename,
                "expense_id": result.get("expense_id"),
                "merchant": (result.get("extracted") or {}).get("merchant"),
                "amount": (result.get("extracted") or {}).get("amount"),
            },
            source="api",
            resource_type="expense",
            resource_id=result.get("expense_id"),
        )
    except Exception:
        pass
    return result

session_manager = SessionManager()

async def _ensure_all_indexes():
    """Create all database indexes on startup (no Email OTP)"""
    try:
        # Audit logger indexes
        await audit_logger._ensure_indexes()
        print("✅ Audit logger indexes created")
    except Exception as e:
        print(f"⚠️ Warning: Could not create audit logger indexes: {e}")
    
    try:
        # Security manager indexes
        await security_manager._ensure_indexes()
        print("✅ Security manager indexes created")
    except Exception as e:
        print(f"⚠️ Warning: Could not create security manager indexes: {e}")
    
    try:
        # FX manager indexes
        await fx_manager._ensure_indexes()
        print("✅ FX manager indexes created")
    except Exception as e:
        print(f"⚠️ Warning: Could not create FX manager indexes: {e}")


async def _bill_reminder_scheduler() -> None:
    """Background loop: send bill due emails on schedule."""
    await asyncio.sleep(20)
    interval = int(os.getenv("BILL_REMINDER_INTERVAL_SEC", "3600"))
    while True:
        try:
            result = await process_bill_reminders(db)
            if result.get("sent"):
                print(f"📧 Bill reminder emails sent: {result['sent']}")
        except asyncio.CancelledError:
            break
        except Exception as exc:
            print(f"⚠️ Bill reminder scheduler: {exc}")
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events"""
    await _ensure_all_indexes()
    if notification_service.email_configured():
        print("✅ Bill email reminders enabled (SMTP configured)")
    else:
        print("ℹ️ Bill email reminders off — set SMTP_* in .env to enable")
    reminder_task = asyncio.create_task(_bill_reminder_scheduler())
    print("🚀 TrackExpensio API started successfully")
    yield
    reminder_task.cancel()
    try:
        await reminder_task
    except asyncio.CancelledError:
        pass
    print("👋 TrackExpensio API shutting down")


app = FastAPI(title="TrackExpensio API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def _google_flow(request: Request | None = None) -> Flow:
    """
    Build a Google OAuth flow. If GOOGLE_REDIRECT_URI is not set, derive it from the incoming request
    (works for localhost / 127.0.0.1 during local dev).
    """
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    if not redirect_uri and request is not None:
        base = str(request.base_url).rstrip("/")  # e.g. http://127.0.0.1:8080
        redirect_uri = f"{base}/auth/google/callback"

    # Helpful debug for local setup (safe)
    print("DEBUG GOOGLE CLIENT:", repr(os.getenv("GOOGLE_CLIENT_ID")))
    print("DEBUG GOOGLE REDIRECT:", repr(redirect_uri))
    config = {
        "web": {
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "project_id": os.getenv("GOOGLE_PROJECT_ID", "expense-tracker"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
            "redirect_uris": [redirect_uri],
        }
    }
    flow = Flow.from_client_config(
        config,
        scopes=[
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
            "https://www.googleapis.com/auth/gmail.readonly",
            "openid",
        ],
    )
    flow.redirect_uri = redirect_uri
    return flow


class ChatRequest(BaseModel):
    message: str = Field(..., description="User query or command")
    session_id: str | None = None
    api_key: str | None = None
    history: list[dict[str, str]] | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


class ExpenseCreateRequest(BaseModel):
    date: str
    amount: float
    category: str
    merchant: str = ""
    note: str = ""
    session_id: str | None = None
    api_key: str | None = None


class IncomeCreateRequest(BaseModel):
    date: str
    amount: float
    source: str
    category: str = "Salary"
    note: str = ""
    session_id: str | None = None
    api_key: str | None = None


class BudgetCreateRequest(BaseModel):
    category: str
    amount: float
    session_id: str | None = None
    api_key: str | None = None


class PlaidExchangeRequest(BaseModel):
    public_token: str
    session_id: str | None = None
    api_key: str | None = None
    bank_name: str = "Linked Bank"
    account_type: str = "Checking"
    account_number_last4: str = "0000"


class NotificationSettingsRequest(BaseModel):
    bill_email_enabled: bool | None = None
    alert_email: str | None = None
    session_id: str | None = None
    api_key: str | None = None


class BankSyncRequest(BaseModel):
    connection_id: str
    start_date: str
    end_date: str
    session_id: str | None = None
    api_key: str | None = None


class SavingsGoalRequest(BaseModel):
    goal_name: str
    target_amount: float
    target_date: str
    current_amount: float = 0
    goal_number: str = ""
    goal_id: str | None = None
    session_id: str | None = None
    api_key: str | None = None


class GmailSyncRequest(BaseModel):
    session_id: str | None = None
    api_key: str | None = None
    max_messages: int = 40


class AAConsentRequest(BaseModel):
    provider: str = "setu"
    bank_name: str = "HDFC Bank"
    session_id: str | None = None
    api_key: str | None = None


class AAApproveRequest(BaseModel):
    consent_id: str
    approved: bool = True
    session_id: str | None = None
    api_key: str | None = None


class AASyncRequest(BaseModel):
    connection_id: str
    session_id: str | None = None
    api_key: str | None = None


class SMSParseRequest(BaseModel):
    sms_text: str
    save: bool = False
    session_id: str | None = None
    api_key: str | None = None


async def _enforce_rate_limit_and_ip(request: Request, identifier: str) -> str | None:
    """IP block + rate limit (100 req / 15 min). Returns client IP."""
    client_ip = request.client.host if request.client else None
    if client_ip and await security_manager.is_ip_blocked(client_ip):
        await security_manager.log_security_event(
            "blocked_ip_access_attempt",
            ip_address=client_ip,
            severity="warning",
        )
        raise HTTPException(status_code=403, detail="Access denied")
    is_allowed, _ = await security_manager.check_rate_limit(
        identifier, max_requests=100, window_minutes=15
    )
    if not is_allowed:
        try:
            await audit_logger.log(
                user_id=None,
                action=AuditAction.RATE_LIMIT_HIT,
                ip_address=client_ip,
                details={"identifier": identifier[:32]},
                success=False,
                source="api",
            )
        except Exception:
            pass
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
        )
    return client_ip


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Deployment health check."""
    mongo_ok = False
    try:
        await auth.client.admin.command("ping")
        mongo_ok = True
    except Exception:
        pass
    return {
        "status": "healthy" if mongo_ok else "degraded",
        "service": "TrackExpensio",
        "version": "1.0",
        "mongodb": mongo_ok,
        "integrations": {
            "plaid": plaid_service.is_configured(),
            "email_alerts": notification_service.email_configured(),
            "push_alerts": notification_service.push_configured(),
        },
    }


@app.get("/api/bank/plaid/link-token")
async def plaid_link_token(
    request: Request,
    session_id: str = None,
    api_key: str = None,
) -> Dict[str, Any]:
    """Plaid Link token for frontend bank connection."""
    identifier = api_key or session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    result = await plaid_service.create_link_token(user_id)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message", "Plaid link token failed"))
    return result


@app.post("/api/bank/plaid/exchange")
async def plaid_exchange(body: PlaidExchangeRequest, request: Request) -> Dict[str, Any]:
    """Complete Plaid Link — save bank connection."""
    identifier = body.api_key or body.session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    user_id = await auth.authenticate_user(api_key=body.api_key, session_id=body.session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    exchanged = await plaid_service.exchange_public_token(body.public_token)
    if exchanged.get("status") != "success":
        raise HTTPException(status_code=400, detail=exchanged.get("message", "Plaid exchange failed"))
    last4 = body.account_number_last4
    if len(last4) > 4:
        last4 = last4[-4:]
    connection_id = await db.connect_bank_account(
        user_id,
        body.bank_name,
        body.account_type,
        last4,
        plaid_access_token=exchanged.get("access_token"),
        plaid_item_id=exchanged.get("item_id"),
    )
    return {
        "status": "success",
        "connection_id": connection_id,
        "message": f"Bank linked: {body.bank_name}",
    }


@app.get("/api/bank/connections")
async def list_bank_connections_api(
    request: Request,
    session_id: str = None,
    api_key: str = None,
) -> Dict[str, Any]:
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    connections = await db.list_bank_connections(user_id)
    return {"status": "success", "connections": connections}


@app.post("/api/bank/sync")
async def sync_bank_api(body: BankSyncRequest, request: Request) -> Dict[str, Any]:
    identifier = body.api_key or body.session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    result = await sync_bank_transactions(
        api_key=body.api_key,
        session_id=body.session_id,
        connection_id=body.connection_id,
        start_date=body.start_date,
        end_date=body.end_date,
    )
    if isinstance(result, dict) and result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@app.get("/api/bank/plaid/balances")
async def plaid_balances_api(
    request: Request,
    connection_id: str = None,
    session_id: str = None,
    api_key: str = None,
) -> Dict[str, Any]:
    """Current balances for a Plaid-linked bank connection."""
    identifier = api_key or session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not connection_id:
        raise HTTPException(status_code=400, detail="connection_id is required")
    conn = await db.get_bank_connection(user_id, connection_id)
    if not conn:
        raise HTTPException(status_code=404, detail="Bank connection not found")
    token = conn.get("plaid_access_token")
    if not token:
        raise HTTPException(status_code=400, detail="No Plaid token on this connection — reconnect via Link")
    result = await plaid_service.get_account_balances(token)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message", "Could not fetch balances"))
    return result


# ── Tutorial-compatible Plaid routes (delegate to TrackExpensio auth + MongoDB) ──

@app.post("/plaid/create-link-token")
async def plaid_tutorial_link_token(
    request: Request,
    session_id: str = None,
    api_key: str = None,
) -> Dict[str, Any]:
    """Legacy path from Plaid tutorials — same as /api/bank/plaid/link-token."""
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            body = await request.json()
            session_id = session_id or body.get("session_id")
            api_key = api_key or body.get("api_key")
        except Exception:
            pass
    return await plaid_link_token(request, session_id=session_id, api_key=api_key)


@app.post("/plaid/exchange-token")
async def plaid_tutorial_exchange(request: Request) -> Dict[str, Any]:
    """Legacy path — same as /api/bank/plaid/exchange."""
    body = await request.json()
    exchange_body = PlaidExchangeRequest(
        public_token=body.get("public_token", ""),
        session_id=body.get("session_id"),
        api_key=body.get("api_key"),
        bank_name=body.get("bank_name", "Linked Bank"),
        account_type=body.get("account_type", "Checking"),
        account_number_last4=body.get("account_number_last4", "0000"),
    )
    return await plaid_exchange(exchange_body, request)


@app.get("/plaid/sync-transactions")
async def plaid_tutorial_sync(
    request: Request,
    days: int = 30,
    connection_id: str = None,
    session_id: str = None,
    api_key: str = None,
) -> Dict[str, Any]:
    """Legacy path — sync last N days into TrackExpensio expenses."""
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not connection_id:
        connections = await db.list_bank_connections(user_id)
        plaid_conns = [c for c in connections if c.get("plaid_item_id")]
        if not plaid_conns:
            raise HTTPException(status_code=400, detail="No bank connected. Use Connect Bank first.")
        connection_id = plaid_conns[0]["id"]
    end_date = datetime.now(timezone.utc).date().isoformat()
    start_date = (datetime.now(timezone.utc).date() - timedelta(days=max(1, min(days, 730)))).isoformat()
    result = await sync_bank_transactions(
        api_key=api_key,
        session_id=session_id,
        connection_id=connection_id,
        start_date=start_date,
        end_date=end_date,
    )
    if isinstance(result, dict) and result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@app.get("/plaid/balances")
async def plaid_tutorial_balances(
    request: Request,
    connection_id: str = None,
    session_id: str = None,
    api_key: str = None,
) -> Dict[str, Any]:
    """Legacy path — same as /api/bank/plaid/balances."""
    return await plaid_balances_api(request, connection_id, session_id, api_key)


@app.get("/api/user/notification-settings")
async def get_notification_settings_api(
    session_id: str = None,
    api_key: str = None,
) -> Dict[str, Any]:
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    from services.bill_reminder_job import get_user_email

    prefs = await db.get_notification_settings(user_id)
    profile_email = await get_user_email(user_id, mongo_manager)
    return {
        "status": "success",
        "settings": prefs,
        "profile_email": profile_email,
        "smtp_configured": notification_service.email_configured(),
    }


@app.put("/api/user/notification-settings")
async def update_notification_settings_api(body: NotificationSettingsRequest) -> Dict[str, Any]:
    user_id = await auth.authenticate_user(api_key=body.api_key, session_id=body.session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    await db.set_notification_settings(
        user_id,
        bill_email_enabled=body.bill_email_enabled,
        alert_email=body.alert_email,
    )
    return {"status": "success", "message": "Notification settings saved"}


@app.post("/api/bills/reminders/run")
async def run_bill_reminders_now(
    request: Request,
    session_id: str = None,
    api_key: str = None,
) -> Dict[str, Any]:
    """Manually trigger bill reminder emails for all users (authenticated)."""
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    result = await process_bill_reminders(db)
    return result


@app.get("/api/integrations/status")
async def integrations_status() -> Dict[str, Any]:
    """Which optional integrations are configured."""
    return {
        "plaid": {
            "configured": plaid_service.is_configured(),
            "env": os.getenv("PLAID_ENV", "sandbox"),
            **plaid_service.coverage_info(),
        },
        "notifications": notification_service.status(),
    }


@app.post("/api/auth/login")
async def rest_login(body: LoginRequest, request: Request) -> Dict[str, Any]:
    """REST login — returns API key."""
    client_ip = await _enforce_rate_limit_and_ip(request, body.username)
    result = await auth.login_user(body.username, body.password)
    if result.get("status") != "success":
        try:
            await audit_logger.log(
                user_id=None,
                action=AuditAction.FAILED_LOGIN,
                ip_address=client_ip,
                details={"username": body.username, "via": "rest"},
                success=False,
                source="api",
            )
        except Exception:
            pass
        raise HTTPException(status_code=401, detail=result.get("message", "Login failed"))
    await audit_logger.log(
        user_id=result.get("user_id"),
        action=AuditAction.LOGIN,
        ip_address=client_ip,
        details={"username": body.username, "via": "rest"},
        source="api",
    )
    return result


@app.post("/api/auth/register")
async def rest_register(body: RegisterRequest, request: Request) -> Dict[str, Any]:
    """REST user registration."""
    await _enforce_rate_limit_and_ip(request, body.username)
    result = await auth.create_user(body.username, body.password)
    if result.get("status") != "success":
        raise HTTPException(status_code=400, detail=result.get("message", "Registration failed"))
    return result


@app.get("/api/expenses")
async def rest_list_expenses(
    request: Request,
    session_id: str = None,
    api_key: str = None,
    start_date: str = None,
    end_date: str = None,
) -> Any:
    """REST list expenses (alternative to MCP list_expenses)."""
    identifier = api_key or session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not start_date or not end_date:
        today = datetime.now(timezone.utc)
        end_date = end_date or today.strftime("%Y-%m-%d")
        start = today - timedelta(days=30)
        start_date = start_date or start.strftime("%Y-%m-%d")
    result = await list_expenses(
        api_key=api_key,
        session_id=session_id,
        start_date=start_date,
        end_date=end_date,
    )
    if isinstance(result, dict) and result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return {"status": "success", "expenses": result, "count": len(result) if isinstance(result, list) else 0}


@app.get("/api/expenses/overview")
async def rest_expenses_overview(
    request: Request,
    session_id: str = None,
    api_key: str = None,
    period: str = "30d",
    start_date: str = None,
    end_date: str = None,
) -> Dict[str, Any]:
    """Expenses total, count, category breakdown, and full list for a period."""
    identifier = api_key or session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    result = await get_expenses_overview(
        api_key=api_key,
        session_id=session_id,
        period=period,
        start_date=start_date,
        end_date=end_date,
    )
    if isinstance(result, dict) and result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@app.post("/api/expenses")
async def rest_create_expense(body: ExpenseCreateRequest, request: Request) -> Dict[str, Any]:
    """REST add expense."""
    identifier = body.api_key or body.session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    result = await add_expense(
        api_key=body.api_key,
        session_id=body.session_id,
        date=body.date,
        amount=body.amount,
        category=body.category,
        merchant=body.merchant,
        note=body.note,
    )
    if isinstance(result, dict) and result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@app.get("/api/income")
async def rest_list_income(
    request: Request,
    session_id: str = None,
    api_key: str = None,
    start_date: str = None,
    end_date: str = None,
) -> Any:
    """REST list income entries."""
    identifier = api_key or session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not start_date or not end_date:
        today = datetime.now(timezone.utc)
        end_date = end_date or today.strftime("%Y-%m-%d")
        start = today - timedelta(days=30)
        start_date = start_date or start.strftime("%Y-%m-%d")
    result = await list_income(
        api_key=api_key,
        session_id=session_id,
        start_date=start_date,
        end_date=end_date,
    )
    if isinstance(result, dict) and result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return {"status": "success", "income": result, "count": len(result) if isinstance(result, list) else 0}


@app.get("/api/income/summary")
async def rest_income_summary(
    request: Request,
    session_id: str = None,
    api_key: str = None,
    start_date: str = None,
    end_date: str = None,
) -> Dict[str, Any]:
    """REST income summary for a date range."""
    identifier = api_key or session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    result = await get_income_summary(
        api_key=api_key,
        session_id=session_id,
        start_date=start_date,
        end_date=end_date,
    )
    if isinstance(result, dict) and result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@app.post("/api/income")
async def rest_create_income(body: IncomeCreateRequest, request: Request) -> Dict[str, Any]:
    """REST add income."""
    identifier = body.api_key or body.session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    result = await add_income(
        api_key=body.api_key,
        session_id=body.session_id,
        date=body.date,
        amount=body.amount,
        source=body.source,
        category=body.category,
        note=body.note,
    )
    if isinstance(result, dict) and result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@app.get("/api/budget/status")
async def rest_budget_status(
    request: Request,
    session_id: str = None,
    api_key: str = None,
) -> Dict[str, Any]:
    """REST budget status with spend vs limit."""
    identifier = api_key or session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    result = await check_budget_status(api_key=api_key, session_id=session_id)
    if isinstance(result, dict) and result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@app.post("/api/budget")
async def rest_set_budget(body: BudgetCreateRequest, request: Request) -> Dict[str, Any]:
    """REST set or update a monthly category budget."""
    identifier = body.api_key or body.session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    result = await set_budget(
        api_key=body.api_key,
        session_id=body.session_id,
        category=body.category,
        amount=body.amount,
    )
    if isinstance(result, dict) and result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@app.get("/api/market/quote")
async def rest_market_quote(
    request: Request,
    symbol: str,
    session_id: str = None,
    api_key: str = None,
) -> Dict[str, Any]:
    """REST Yahoo Finance quote lookup for dashboard market widget."""
    identifier = api_key or session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id, require_linked=False)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not symbol or not symbol.strip():
        raise HTTPException(status_code=400, detail="Symbol is required")
    result = await yahoo_finance(symbol=symbol.strip(), api_key=api_key, session_id=session_id)
    if isinstance(result, dict) and result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@app.get("/api/gmail/status")
async def gmail_sync_status(
    request: Request,
    session_id: str = None,
    api_key: str = None,
) -> Dict[str, Any]:
    """Check whether Gmail is connected and bill import is available."""
    identifier = api_key or session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id, require_linked=False)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    status = await gmail_bill_service.get_status(user_id)
    return {"status": "success", **status}


@app.post("/api/gmail/sync-bills")
async def gmail_sync_bills(body: GmailSyncRequest, request: Request) -> Dict[str, Any]:
    """Scan Gmail for payment receipts and import them as expenses."""
    identifier = body.api_key or body.session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    user_id = await auth.authenticate_user(
        api_key=body.api_key, session_id=body.session_id, require_linked=False
    )
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        result = await gmail_bill_service.sync_bills(
            user_id, max_messages=min(max(body.max_messages, 1), 80)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gmail sync failed: {exc}") from exc

    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message", "Sync failed"))
    return result


@app.post("/api/gmail/sync-bank-transactions")
async def gmail_sync_bank_transactions(body: GmailSyncRequest, request: Request) -> Dict[str, Any]:
    """Scan Gmail for Indian bank transaction alert emails and import as expenses."""
    identifier = body.api_key or body.session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    user_id = await auth.authenticate_user(
        api_key=body.api_key, session_id=body.session_id, require_linked=False
    )
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        result = await gmail_bill_service.sync_bank_transaction_emails(
            user_id, max_messages=min(max(body.max_messages, 1), 80)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gmail bank sync failed: {exc}") from exc
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message", "Sync failed"))
    return result


@app.post("/api/gmail/purge-promos")
async def gmail_purge_promotional_imports(
    body: GmailSyncRequest, request: Request
) -> Dict[str, Any]:
    """Delete Gmail-imported expenses that match promotional/marketing filters."""
    identifier = body.api_key or body.session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    user_id = await auth.authenticate_user(
        api_key=body.api_key, session_id=body.session_id, require_linked=False
    )
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        return await gmail_bill_service.purge_promotional_imports(user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Purge failed: {exc}") from exc


@app.get("/api/aa/status")
async def aa_status() -> Dict[str, Any]:
    """Account Aggregator (India) integration status."""
    return {"status": "success", **aa_service.status()}


@app.get("/api/aa/connections")
async def aa_list_connections(
    request: Request,
    session_id: str = None,
    api_key: str = None,
) -> Dict[str, Any]:
    identifier = api_key or session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    connections = await aa_service.list_connections(user_id)
    return {"status": "success", "connections": connections, "count": len(connections)}


@app.post("/api/aa/consent")
async def aa_create_consent(body: AAConsentRequest, request: Request) -> Dict[str, Any]:
    """Start Account Aggregator consent flow (Setu / Finvu / OneMoney)."""
    identifier = body.api_key or body.session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    user_id = await auth.authenticate_user(api_key=body.api_key, session_id=body.session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    result = await aa_service.create_consent(user_id, body.provider, body.bank_name)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@app.post("/api/aa/approve")
async def aa_approve_consent(body: AAApproveRequest, request: Request) -> Dict[str, Any]:
    """Complete AA consent after user approves on consent screen."""
    identifier = body.api_key or body.session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    user_id = await auth.authenticate_user(api_key=body.api_key, session_id=body.session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    result = await aa_service.approve_consent(user_id, body.consent_id, body.approved)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@app.post("/api/aa/sync")
async def aa_sync_transactions(body: AASyncRequest, request: Request) -> Dict[str, Any]:
    """Fetch transactions from linked AA bank account and store as expenses."""
    identifier = body.api_key or body.session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    user_id = await auth.authenticate_user(api_key=body.api_key, session_id=body.session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    result = await aa_service.sync_transactions(user_id, body.connection_id)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@app.post("/api/sms/parse")
async def sms_parse_expense(body: SMSParseRequest, request: Request) -> Dict[str, Any]:
    """Parse Indian bank SMS alert(s). Optionally save as expenses."""
    identifier = body.api_key or body.session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    user_id = await auth.authenticate_user(api_key=body.api_key, session_id=body.session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not body.sms_text or not body.sms_text.strip():
        raise HTTPException(status_code=400, detail="SMS text is required")

    parsed_list = parse_sms_batch(body.sms_text)
    if not parsed_list:
        single = parse_sms_alert(body.sms_text)
        if single:
            parsed_list = [single]

    if not parsed_list:
        raise HTTPException(
            status_code=400,
            detail='Could not parse amount from SMS. Try: "Rs. 450 spent on Amazon" or "INR 599 debited from HDFC"',
        )

    if not body.save:
        return {"status": "success", "parsed": parsed_list, "count": len(parsed_list)}

    saved = []
    skipped = 0
    for item in parsed_list:
        ext_id = item["metadata"]["sms_hash"]
        if await db.expense_exists_by_source_id(user_id, "sms_parse", ext_id):
            skipped += 1
            continue
        expense_id = await db.add_expense(
            user_id,
            item["date"],
            item["amount"],
            item["category"],
            item["note"],
            item["merchant"],
            metadata=item["metadata"],
        )
        if expense_id:
            saved.append({
                "expense_id": expense_id,
                "merchant": item["merchant"],
                "amount": item["amount"],
                "date": item["date"],
                "category": item["category"],
            })
        else:
            skipped += 1

    return {
        "status": "success",
        "saved": saved,
        "imported": len(saved),
        "skipped": skipped,
        "message": f"Saved {len(saved)} expense(s) from SMS ({skipped} skipped).",
    }


@app.get("/api/savings-goals")
async def rest_list_savings_goals(
    request: Request,
    session_id: str = None,
    api_key: str = None,
) -> Dict[str, Any]:
    """List savings goals with progress."""
    identifier = api_key or session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    result = await track_savings_progress(api_key=api_key, session_id=session_id)
    if isinstance(result, dict) and result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@app.post("/api/savings-goals")
async def rest_save_savings_goal(body: SavingsGoalRequest, request: Request) -> Dict[str, Any]:
    """Create or update a savings goal."""
    identifier = body.api_key or body.session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    if body.goal_id:
        result = await set_savings_goal(
            api_key=body.api_key,
            session_id=body.session_id,
            goal_id=body.goal_id,
            goal_name=body.goal_name,
            target_amount=body.target_amount,
            target_date=body.target_date,
            current_amount=body.current_amount,
            goal_number=body.goal_number,
        )
    else:
        result = await add_savings_goal(
            api_key=body.api_key,
            session_id=body.session_id,
            goal_name=body.goal_name,
            target_amount=body.target_amount,
            target_date=body.target_date,
            current_amount=body.current_amount,
            goal_number=body.goal_number,
        )
    if isinstance(result, dict) and result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@app.delete("/api/savings-goals/{goal_id}")
async def rest_delete_savings_goal(
    goal_id: str,
    request: Request,
    session_id: str = None,
    api_key: str = None,
) -> Dict[str, Any]:
    """Remove a savings goal."""
    identifier = api_key or session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    result = await delete_savings_goal(api_key=api_key, session_id=session_id, goal_id=goal_id)
    if isinstance(result, dict) and result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@app.get("/api/reports/pdf")
async def rest_report_pdf(
    request: Request,
    session_id: str = None,
    api_key: str = None,
    start_date: str = None,
    end_date: str = None,
):
    """Download financial report as PDF."""
    from fastapi.responses import Response
    from services.pdf_reports import build_financial_report_pdf

    identifier = api_key or session_id or (request.client.host if request.client else "anonymous")
    await _enforce_rate_limit_and_ip(request, identifier)
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    today = datetime.now(timezone.utc)
    end_date = end_date or today.strftime("%Y-%m-%d")
    start_date = start_date or (today - timedelta(days=30)).strftime("%Y-%m-%d")
    report = await generate_financial_report(
        api_key=api_key,
        session_id=session_id,
        start_date=start_date,
        end_date=end_date,
    )
    if report.get("status") == "error":
        raise HTTPException(status_code=400, detail=report.get("message"))
    try:
        pdf_bytes = build_financial_report_pdf(report)
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    filename = f"report_{start_date}_{end_date}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest, request: Request):
    """Conversational endpoint powered by Groq + TrackExpensio tools."""
    identifier = payload.api_key or payload.session_id or (request.client.host if request.client else "anonymous")
    client_ip = await _enforce_rate_limit_and_ip(request, identifier)
    try:
        # Authenticate user
        user_id = await auth.authenticate_user(
            api_key=payload.api_key, session_id=payload.session_id
        )
        if not user_id:
            # Log failed authentication (best-effort)
            try:
                await audit_logger.log(
                    user_id=None,
                    action=AuditAction.FAILED_LOGIN,
                    ip_address=client_ip,
                    success=False,
                    error_message="Authentication failed",
                )
            except Exception:
                pass
            raise HTTPException(status_code=401, detail="Authentication required")

        # Call chat agent
        result = await chat_with_agent(
            payload.message,
            session_id=payload.session_id,
            api_key=payload.api_key,
            history=payload.history,
        )

        # Log successful chat interaction (best-effort)
        try:
            await audit_logger.log(
                user_id=user_id,
                action=AuditAction.CHAT_INTERACTION,
                ip_address=client_ip,
                details={"message_length": len(payload.message)},
                success=True,
                source="api",
            )
        except Exception:
            pass

        return result

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc


@app.post("/api/session/create")
async def create_session() -> Dict[str, Any]:
    """Create a new guest session for public access"""
    session_data = await session_manager.create_guest_session()
    return session_data


@app.post("/api/mcp/{tool_name}")
async def invoke_tool(tool_name: str, payload: Dict[str, Any], request: Request) -> Any:
    """Invoke MCP tool - accepts either api_key or session_id"""
    tool = TOOL_REGISTRY.get(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    api_key = payload.get("api_key")
    session_id = payload.get("session_id")
    identifier = api_key or session_id or (request.client.host if request.client else "anonymous")
    client_ip = await _enforce_rate_limit_and_ip(request, identifier)
    user_agent = request.headers.get("user-agent")
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id)
    if not user_id:
        try:
            await audit_logger.log(
                user_id=None,
                action=AuditAction.FAILED_LOGIN,
                details={"tool": tool_name},
                success=False,
                error_message="Authentication required",
                source="api",
                ip_address=client_ip,
                user_agent=user_agent,
                tool_name=tool_name,
                session_id=session_id,
            )
        except Exception:
            pass
        raise HTTPException(status_code=401, detail="Authentication required")
    
    started = time.perf_counter()
    result = await tool(**payload)
    duration_ms = (time.perf_counter() - started) * 1000
    ok = not (isinstance(result, dict) and result.get("status") == "error")

    if tool_name not in DETAILED_AUDIT_TOOLS or not ok:
        try:
            await audit_logger.log(
                user_id=user_id,
                action=AuditAction.MCP_TOOL_CALLED,
                details={
                    "tool": tool_name,
                    **({"error": result.get("message")} if not ok and isinstance(result, dict) else {}),
                },
                success=ok,
                error_message=result.get("message") if not ok and isinstance(result, dict) else None,
                source="api",
                ip_address=client_ip,
                user_agent=user_agent,
                tool_name=tool_name,
                session_id=session_id,
                duration_ms=duration_ms,
            )
        except Exception:
            pass
    return result


@app.get("/api/audit/dashboard")
async def audit_dashboard(
    request: Request,
    session_id: str = None,
    api_key: str = None,
    hours: int = 168,
) -> Dict[str, Any]:
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    return {
        "status": "success",
        "dashboard": await audit_logger.get_dashboard(user_id, hours=min(hours, 720)),
    }


@app.get("/api/audit/logs")
async def audit_logs_api(
    request: Request,
    session_id: str = None,
    api_key: str = None,
    start_date: str = None,
    end_date: str = None,
    action: str = None,
    success: bool = None,
    source: str = None,
    category: str = None,
    severity: str = None,
    search: str = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    start_dt = end_dt = None
    if start_date or end_date:
        try:
            start_dt, end_dt = parse_audit_dates(start_date, end_date)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    action_enum = None
    if action:
        try:
            action_enum = AuditAction(action.lower())
        except ValueError:
            pass
    result = await audit_logger.search_logs(
        user_id=user_id,
        start_date=start_dt,
        end_date=end_dt,
        action=action_enum,
        success=success,
        source=source,
        category=category,
        severity=severity,
        search=search,
        limit=min(limit, 500),
        offset=max(0, offset),
    )
    return {"status": "success", **result}


@app.get("/api/audit/export")
async def audit_export_api(
    request: Request,
    session_id: str = None,
    api_key: str = None,
    start_date: str = None,
    end_date: str = None,
    action: str = None,
):
    from fastapi.responses import Response
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    start_dt = end_dt = None
    if start_date or end_date:
        start_dt, end_dt = parse_audit_dates(
            start_date or "2000-01-01",
            end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )
    action_enum = AuditAction(action.lower()) if action else None
    csv_data = await audit_logger.export_logs_csv(user_id, start_dt, end_dt, action_enum)
    await audit_logger.log(
        user_id=user_id,
        action=AuditAction.DATA_EXPORTED,
        details={"via": "rest_export"},
        source="api",
        ip_address=request.client.host if request.client else None,
    )
    filename = f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.csv"
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/doc-expense")
async def doc_expense(
    session_id: str = Form(None),
    api_key: str = Form(None), 
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    result = await _process_document_upload(file, api_key, session_id)
    extracted = result.get("extracted") or {}
    merchant = extracted.get("merchant") or "Unknown merchant"
    amount = extracted.get("amount") or 0
    return {
        "message": f"Expense saved from bill: {merchant} — ₹{float(amount):,.2f}",
        "extracted": extracted,
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
    """Get user profile information (name, email) from Google account or email OTP"""
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Handle email OTP users (user_id format: "email_<email>")
    if user_id.startswith("email_"):
        email = user_id.replace("email_", "", 1)
        return {
            "name": email.split("@")[0],  # Use email prefix as name
            "email": email,
            "picture": None,
        }
    
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
async def google_login(request: Request, session_id: str = None, api_key: str = None):
    """Optional Gmail login for data persistence"""
    if not session_id and not api_key:
        raise HTTPException(status_code=400, detail="session_id or api_key required")
    
    user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id, require_linked=False)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    flow = _google_flow(request)
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    await mongo_manager.save_oauth_state(
        state, {"api_key": api_key, "session_id": session_id, "user_id": user_id}
    )
    return RedirectResponse(authorization_url)


@app.get("/auth/google/callback")
async def google_callback(request: Request, code: str = None, state: str = None, error: str = None):
    """Handle Google OAuth callback - success or error"""
    if error:
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Authentication Error - TrackExpensio</title>
            <meta charset="UTF-8">
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    text-align: center;
                    padding: 50px 20px;
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white;
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .container {{
                    background: rgba(255, 255, 255, 0.1);
                    backdrop-filter: blur(10px);
                    border-radius: 20px;
                    padding: 40px;
                    max-width: 400px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                }}
                .error-icon {{ font-size: 64px; margin-bottom: 20px; }}
                h2 {{ color: white; margin-bottom: 10px; }}
                p {{ color: rgba(255, 255, 255, 0.9); margin-bottom: 15px; }}
                button {{
                    background: white;
                    color: #f5576c;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 8px;
                    font-weight: 600;
                    cursor: pointer;
                    margin-top: 10px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="error-icon">❌</div>
                <h2>Authentication Failed</h2>
                <p>{error.replace('_', ' ').title()}</p>
                <p style="font-size: 14px;">Please try again or use Sync with Google on the main app.</p>
                <button onclick="window.close()">Close Window</button>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(error_html)
    
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state parameter")
    
    stored = await mongo_manager.pop_oauth_state(state)
    if not stored:
        raise HTTPException(status_code=400, detail="Invalid or expired state. Please try again.")
    
    try:
        flow = _google_flow(request)
        flow.fetch_token(code=code)
        credentials = flow.credentials
        service = build("oauth2", "v2", credentials=credentials)
        profile = service.userinfo().get().execute()
    except Exception as e:
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error - TrackExpensio</title>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    text-align: center;
                    padding: 50px;
                    background: #1a1a1a;
                    color: white;
                }}
                .error {{ color: #ff6b6b; }}
            </style>
        </head>
        <body>
            <h2 class="error">❌ Authentication Error</h2>
            <p>Failed to authenticate with Google: {str(e)}</p>
            <p>Please try again or use Sync with Google on the main app.</p>
            <button onclick="window.close()" style="margin-top: 20px; padding: 10px 20px; cursor: pointer;">Close</button>
        </body>
        </html>
        """
        return HTMLResponse(error_html)
    
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

    try:
        await audit_logger.log(
            user_id=google_user_id,
            action=AuditAction.SESSION_LINKED,
            details={"email": profile.get("email"), "session_id": session_id},
            source="oauth",
        )
    except Exception:
        pass
    
    # Return HTML that closes popup and notifies parent window
    if session_id:
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Connected - TrackExpensio</title>
            <meta charset="UTF-8">
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    text-align: center;
                    padding: 50px 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .container {{
                    background: rgba(255, 255, 255, 0.1);
                    backdrop-filter: blur(10px);
                    border-radius: 20px;
                    padding: 40px;
                    max-width: 400px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                }}
                .checkmark {{
                    font-size: 64px;
                    margin-bottom: 20px;
                    animation: scaleIn 0.5s ease-out;
                }}
                @keyframes scaleIn {{
                    from {{ transform: scale(0); }}
                    to {{ transform: scale(1); }}
                }}
                h2 {{
                    color: white;
                    margin-bottom: 10px;
                    font-size: 24px;
                }}
                p {{
                    color: rgba(255, 255, 255, 0.9);
                    margin-bottom: 8px;
                    font-size: 16px;
                }}
                .spinner {{
                    border: 3px solid rgba(255, 255, 255, 0.3);
                    border-top: 3px solid white;
                    border-radius: 50%;
                    width: 30px;
                    height: 30px;
                    animation: spin 1s linear infinite;
                    margin: 20px auto;
                }}
                @keyframes spin {{
                    0% {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(360deg); }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="checkmark">✅</div>
                <h2>Successfully Connected!</h2>
                <p>Your Google account has been linked.</p>
                <p style="font-size: 14px; margin-top: 15px;">Your data will now be saved permanently.</p>
                <div class="spinner"></div>
                <p style="font-size: 12px; margin-top: 10px; opacity: 0.8;">Closing window...</p>
            </div>
            <script>
                // Notify parent window and close
                if (window.opener) {{
                    window.opener.postMessage({{
                        type: 'google_login_success',
                        session_id: '{session_id}'
                    }}, '*');
                    setTimeout(() => window.close(), 1500);
                }} else {{
                    // If not popup, redirect
                    setTimeout(() => {{
                        window.location.href = '/?google_login=success&session_id={session_id}';
                    }}, 1500);
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


@app.post("/api/speech-to-text")
async def speech_to_text(
    audio: UploadFile = File(...),
    session_id: str = Form(None),
    api_key: str = Form(None),
):
    """Convert speech to text using Whisper and process the command"""
    try:
        # Authenticate user
        user_id = await auth.authenticate_user(api_key=api_key, session_id=session_id)
        if not user_id:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Read audio file
        audio_data = await audio.read()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_file:
            tmp_file.write(audio_data)
            tmp_file_path = tmp_file.name
        
        try:
            # Transcribe using Whisper
            transcribed_text = None
            
            # Try OpenAI Whisper API first (more reliable)
            try:
                from openai import OpenAI
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if openai_api_key:
                    client = OpenAI(api_key=openai_api_key)
                    with open(tmp_file_path, "rb") as audio_file:
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            language="en"
                        )
                    transcribed_text = transcript.text.strip()
            except Exception as e:
                # Fallback to local Whisper
                try:
                    import whisper
                    model = whisper.load_model("base")  # Use base model for speed
                    result = model.transcribe(tmp_file_path, language="en")
                    transcribed_text = result["text"].strip()
                except ImportError:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "status": "error",
                            "message": "Whisper not configured. Please set OPENAI_API_KEY or install: pip install openai-whisper"
                        }
                    )
                except Exception as e2:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "status": "error",
                            "message": f"Whisper transcription failed: {str(e2)}"
                        }
                )
            
            if not transcribed_text:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": "No speech detected"}
                )
            
            # Process the transcribed text through chat agent
            from chat_agent import chat_with_agent
            chat_result = await chat_with_agent(
                transcribed_text,
                session_id=session_id,
                api_key=api_key,
            )
            
            try:
                await audit_logger.log(
                    user_id=user_id,
                    action=AuditAction.VOICE_COMMAND,
                    details={
                        "transcribed_length": len(transcribed_text),
                        "preview": transcribed_text[:120],
                    },
                    success=True,
                    source="api",
                )
            except Exception:
                pass
            
            return JSONResponse(content={
                "status": "success",
                "transcribed_text": transcribed_text,
                "reply": chat_result.get("reply", ""),
                "tool_results": chat_result.get("tool_results", []),
            })
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass
                
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Speech-to-text failed: {str(exc)}"}
        )


@app.get("/stocks/analyze/{ticker}")
async def stock_analyze(ticker: str, request: Request):
    """Full stock analysis: chart, indicators, signals, projection, news."""
    identifier = request.client.host if request.client else "anonymous"
    await _enforce_rate_limit_and_ip(request, identifier)
    result = await asyncio.to_thread(analyze_stock, ticker)
    if "error" in result:
        return JSONResponse(status_code=404, content=result)
    return result


@app.get("/stocks/quick/{ticker}")
async def stock_quick(ticker: str, request: Request):
    """Quick stats without heavy chart JSON or news."""
    identifier = request.client.host if request.client else "anonymous"
    await _enforce_rate_limit_and_ip(request, identifier)
    result = await asyncio.to_thread(analyze_stock, ticker)
    if "error" in result:
        return JSONResponse(status_code=404, content=result)
    result = dict(result)
    result.pop("chart_json", None)
    result.pop("news", None)
    return result


@app.get("/stocks/page", response_class=HTMLResponse)
async def stock_page(request: Request):
    """Smart Stock Analyzer UI."""
    return templates.TemplateResponse(request=request, name="stock_analysis.html")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main chatbot UI"""
    return templates.TemplateResponse("expense_tracker.html", context={"request": request})


@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    """Alias for home page"""
    return templates.TemplateResponse("expense_tracker.html", context={"request": request})


if __name__ == "__main__":
    import uvicorn
    host = _server_host()
    port = _server_port()
    print(f"\n🚀 Starting TrackExpensio Web Server...")
    print(f"📱 Open your browser and go to: http://{host}:{port}")
    print(f"💬 Chatbot available at: http://{host}:{port}/chat")
    print(f"\n✅ Server running on {host}:{port}\n")
    uvicorn.run(app, host=host, port=port)

