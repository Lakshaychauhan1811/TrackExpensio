"""
Fetch payment receipts and bills from a user's Gmail inbox and import them as expenses.
Requires Google OAuth with gmail.readonly scope (stored in google_profiles collection).
"""

from __future__ import annotations

import asyncio
import base64
import email
import json
import os
import re
from datetime import datetime, timezone
from email.utils import parseaddr
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from groq import Groq

from database import Database
from services.storage import MongoManager
from services.transaction_filters import (
    categorize_text,
    extract_debit_amount,
    extract_merchant_amount,
    is_promotional,
    should_keep_gmail_expense,
    validate_email_for_import,
)

GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
APP_TZ = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Kolkata"))

# Only payment-confirmation subjects — do NOT pull all merchant marketing mail.
BILL_SEARCH_QUERY = (
    "newer_than:180d "
    "-subject:(voucher OR cashback OR offer OR discount OR welcome OR unlock "
    "OR congratulations OR promo OR \"joining fee\" OR \"get up to\" OR \"win \" OR \"book now\") "
    "subject:(invoice OR receipt OR \"order confirmed\" OR \"payment successful\" "
    "OR \"amount paid\" OR \"amount debited\" OR \"payment received\" OR \"payment of\" "
    "OR \"booking confirmed\" OR \"order delivered\" OR \"transaction alert\" OR bill)"
)

# Indian bank transaction alert emails (HDFC, SBI, ICICI, etc.)
INDIAN_BANK_TXN_QUERY = (
    "newer_than:90d "
    "-subject:(voucher OR cashback OR offer OR welcome OR promo) "
    "(from:(hdfcbank OR icicibank OR onlinesbi OR sbicard OR axisbank OR kotak "
    "OR yesbank OR idfcbank OR indusind OR pnb OR bankofbaroda OR canarabank) "
    "subject:(debited OR \"amount debited\" OR \"spent on\" OR \"payment of\" OR withdrawn))"
)

KNOWN_MERCHANTS = {
    "flipkart": "Flipkart",
    "amazon": "Amazon",
    "swiggy": "Swiggy",
    "zomato": "Zomato",
    "paytm": "Paytm",
    "phonepe": "PhonePe",
    "googlepay": "Google Pay",
    "gpay": "Google Pay",
    "uber": "Uber",
    "ola": "Ola",
    "makemytrip": "MakeMyTrip",
    "irctc": "IRCTC",
    "myntra": "Myntra",
    "ajio": "Ajio",
    "bigbasket": "BigBasket",
    "blinkit": "Blinkit",
    "zepto": "Zepto",
    "cred": "CRED",
    "razorpay": "Razorpay",
    "bookmyshow": "BookMyShow",
}


def _today_str() -> str:
    return datetime.now(APP_TZ).strftime("%Y-%m-%d")


def _strip_html(html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _decode_part(data: str) -> str:
    try:
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_body(payload: Dict[str, Any]) -> str:
    """Walk MIME parts and return best plain-text or HTML body."""
    if not payload:
        return ""

    mime = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data")
    if body_data:
        raw = _decode_part(body_data)
        if "html" in mime.lower():
            return _strip_html(raw)
        return raw

    parts = payload.get("parts") or []
    plain, html = "", ""
    for part in parts:
        part_mime = part.get("mimeType", "")
        if part_mime == "text/plain" and not plain:
            plain = _extract_body(part)
        elif part_mime == "text/html" and not html:
            html = _extract_body(part)
        elif part.get("parts"):
            nested = _extract_body(part)
            if nested:
                if "html" in part_mime.lower():
                    html = html or nested
                else:
                    plain = plain or nested
    return plain or html


def _merchant_from_sender(sender: str, subject: str) -> str:
    _, addr = parseaddr(sender or "")
    addr_lower = (addr or "").lower()
    combined = f"{addr_lower} {subject.lower()}"
    for key, label in KNOWN_MERCHANTS.items():
        if key in combined:
            return label
    if addr:
        domain = addr.split("@")[-1].split(".")[0]
        return domain.replace("-", " ").title()
    return "Unknown"


def _parse_with_groq(subject: str, sender: str, body: str) -> Optional[Dict[str, Any]]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    snippet = (body or "")[:4000]
    today = _today_str()
    prompt = f"""Extract bill/payment info from this email. Return JSON only.

From: {sender}
Subject: {subject}
Body:
{snippet}

IMPORTANT: If this is promotional/marketing (voucher, cashback offer, welcome bonus,
discount ad, "get up to", EMI ad, not an actual completed payment), return:
{{"skip": true, "reason": "promotional"}}

Fields for real payments only:
- merchant: store or service name
- amount: total amount actually paid/debited as number (not string)
- currency: INR unless clearly otherwise
- category: Food, Travel, Bills, Shopping, Entertainment, Utilities, Rent, Health, Education, Other
- date: YYYY-MM-DD (email date if no transaction date found; today is {today})
- notes: order id, invoice no, or short summary (max 120 chars)

Return null for missing fields except amount which is required for real payments."""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("GROQ_RAG_MODEL", "llama-3.3-70b-versatile"),
            messages=[
                {
                    "role": "system",
                    "content": "Return valid JSON only. No markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        if parsed.get("skip"):
            return None
        if parsed.get("amount"):
            parsed["amount"] = float(str(parsed["amount"]).replace(",", ""))
            return parsed
    except Exception:
        pass
    return None


def _build_credentials(creds_data: Dict[str, Any]) -> Optional[Credentials]:
    if not creds_data or not creds_data.get("token"):
        return None
    credentials = Credentials(
        token=creds_data.get("token"),
        refresh_token=creds_data.get("refresh_token"),
        token_uri=creds_data.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=creds_data.get("client_id") or os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=creds_data.get("client_secret") or os.getenv("GOOGLE_CLIENT_SECRET"),
        scopes=creds_data.get("scopes") or [],
    )
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    return credentials


class GmailBillService:
    def __init__(
        self,
        mongo: Optional[MongoManager] = None,
        database: Optional[Database] = None,
    ) -> None:
        self.mongo = mongo or MongoManager()
        self.db = database or Database()

    async def get_status(self, user_id: str) -> Dict[str, Any]:
        doc = await self.mongo.google_collection.find_one({"user_id": user_id})
        if not doc:
            return {
                "connected": False,
                "gmail_read_enabled": False,
                "email": None,
                "last_sync_at": None,
                "synced_count": 0,
            }

        profile = doc.get("profile") or {}
        creds = doc.get("credentials") or {}
        scopes = creds.get("scopes") or []
        gmail_ok = GMAIL_READONLY_SCOPE in scopes
        sync_meta = doc.get("gmail_sync") or {}

        return {
            "connected": True,
            "gmail_read_enabled": gmail_ok,
            "email": profile.get("email"),
            "last_sync_at": sync_meta.get("last_sync_at"),
            "synced_count": sync_meta.get("total_imported", 0),
        }

    async def _save_refreshed_credentials(self, user_id: str, credentials: Credentials) -> None:
        await self.mongo.google_collection.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "credentials.token": credentials.token,
                    "credentials.scopes": list(credentials.scopes or []),
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

    async def _get_gmail_service(self, user_id: str):
        doc = await self.mongo.google_collection.find_one({"user_id": user_id})
        if not doc:
            raise ValueError("Connect your Google account first to import Gmail bills.")
        creds_data = doc.get("credentials") or {}
        scopes = creds_data.get("scopes") or []
        if GMAIL_READONLY_SCOPE not in scopes:
            raise ValueError(
                "Gmail read access not granted. Click 'Connect Gmail' again to allow bill import."
            )
        credentials = _build_credentials(creds_data)
        if not credentials:
            raise ValueError("Google credentials missing. Please reconnect Gmail.")

        if credentials.expired and credentials.refresh_token:
            await asyncio.to_thread(credentials.refresh, Request())
            await self._save_refreshed_credentials(user_id, credentials)

        service = await asyncio.to_thread(
            build, "gmail", "v1", credentials=credentials, cache_discovery=False
        )
        return service

    def _parse_message(
        self, msg: Dict[str, Any], full_message: Dict[str, Any]
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        headers = {
            h["name"].lower(): h["value"]
            for h in full_message.get("payload", {}).get("headers", [])
        }
        subject = headers.get("subject", "")
        sender = headers.get("from", "")
        date_hdr = headers.get("date", "")
        body = _extract_body(full_message.get("payload", {}))
        combined = f"{subject}\n{body}"

        should_import, skip_reason = validate_email_for_import(sender, subject, body)
        if not should_import:
            return None, skip_reason

        email_date = _today_str()
        try:
            parsed_dt = email.utils.parsedate_to_datetime(date_hdr)
            if parsed_dt:
                email_date = parsed_dt.astimezone(APP_TZ).strftime("%Y-%m-%d")
        except Exception:
            pass

        extracted = _parse_with_groq(subject, sender, body)
        regex_amount = extract_merchant_amount(combined) or extract_debit_amount(combined)
        if not extracted:
            if not regex_amount:
                return None, "no_amount"
            extracted = {
                "merchant": _merchant_from_sender(sender, subject),
                "amount": regex_amount,
                "currency": "INR",
                "category": categorize_text(subject, body, sender),
                "date": email_date,
                "notes": subject[:120],
            }
        elif regex_amount:
            # Never trust LLM amount over regex extraction from the email body.
            extracted["amount"] = regex_amount

        amount = extracted.get("amount")
        if not amount or float(amount) <= 0:
            return None, "invalid_amount"

        if is_promotional(subject, body):
            return None, "promotional"

        merchant = extracted.get("merchant") or _merchant_from_sender(sender, subject)
        category = extracted.get("category") or categorize_text(subject, body, merchant)
        date = extracted.get("date") or email_date
        notes = extracted.get("notes") or subject[:120]

        return {
            "message_id": msg["id"],
            "thread_id": msg.get("threadId"),
            "subject": subject,
            "sender": sender,
            "merchant": merchant,
            "amount": float(amount),
            "category": category,
            "date": date,
            "note": notes,
            "metadata": {
                "source": "gmail_sync",
                "gmail_message_id": msg["id"],
                "gmail_subject": subject[:200],
                "gmail_sender": sender[:200],
                "gmail_body_snippet": body[:500],
            },
        }, None

    async def sync_bills(
        self,
        user_id: str,
        *,
        max_messages: int = 40,
        query: Optional[str] = None,
        purge_invalid: bool = True,
    ) -> Dict[str, Any]:
        purge_result: Dict[str, Any] = {"removed": 0, "items": []}
        if purge_invalid:
            purge_result = await self.purge_promotional_imports(user_id)

        service = await self._get_gmail_service(user_id)
        search_q = query or BILL_SEARCH_QUERY

        def _list_messages():
            return (
                service.users()
                .messages()
                .list(userId="me", q=search_q, maxResults=max_messages)
                .execute()
            )

        listing = await asyncio.to_thread(_list_messages)
        messages = listing.get("messages") or []
        if not messages:
            return {
                "status": "success",
                "imported": 0,
                "skipped": 0,
                "message": "No bill emails found in the last 6 months.",
                "items": [],
            }

        imported = 0
        skipped = 0
        skip_reasons: Dict[str, int] = {}
        items: List[Dict[str, Any]] = []

        for msg in messages:
            msg_id = msg["id"]
            if await self.db.expense_exists_by_gmail_id(user_id, msg_id):
                skipped += 1
                continue

            def _get_full(mid=msg_id):
                return (
                    service.users()
                    .messages()
                    .get(userId="me", id=mid, format="full")
                    .execute()
                )

            try:
                full = await asyncio.to_thread(_get_full)
                parsed, skip_reason = self._parse_message(msg, full)
                if not parsed:
                    skipped += 1
                    if skip_reason:
                        skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
                    continue

                expense_id = await self.db.add_expense(
                    user_id,
                    parsed["date"],
                    parsed["amount"],
                    parsed["category"],
                    parsed["note"],
                    parsed["merchant"],
                    metadata=parsed["metadata"],
                )
                if expense_id:
                    imported += 1
                    await self.mongo.mark_gmail_message_synced(
                        user_id, msg_id, expense_id, parsed
                    )
                    items.append(
                        {
                            "expense_id": expense_id,
                            "merchant": parsed["merchant"],
                            "amount": parsed["amount"],
                            "date": parsed["date"],
                            "category": parsed["category"],
                            "subject": parsed["subject"],
                        }
                    )
                else:
                    skipped += 1
            except Exception:
                skipped += 1

        now = datetime.now(timezone.utc).isoformat()
        await self.mongo.google_collection.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "gmail_sync.last_sync_at": now,
                },
                "$inc": {"gmail_sync.total_imported": imported},
            },
        )

        filter_note = ""
        if skip_reasons.get("promotional"):
            filter_note = f" Blocked {skip_reasons['promotional']} promotional email(s)."
        purge_note = ""
        if purge_result.get("removed"):
            purge_note = f" Removed {purge_result['removed']} invalid Gmail import(s)."

        return {
            "status": "success",
            "imported": imported,
            "skipped": skipped,
            "scanned": len(messages),
            "skip_reasons": skip_reasons,
            "purged": purge_result.get("removed", 0),
            "purged_items": purge_result.get("items", []),
            "message": (
                f"Imported {imported} bill(s) from Gmail "
                f"({skipped} skipped as duplicates, promos, or unreadable)."
                f"{purge_note}{filter_note}"
            ),
            "items": items,
        }

    async def sync_bank_transaction_emails(
        self,
        user_id: str,
        *,
        max_messages: int = 50,
    ) -> Dict[str, Any]:
        """Import Indian bank debit/transaction alert emails as expenses."""
        result = await self.sync_bills(
            user_id,
            max_messages=max_messages,
            query=INDIAN_BANK_TXN_QUERY,
        )
        result["message"] = (
            f"Imported {result.get('imported', 0)} bank transaction(s) from Gmail "
            f"({result.get('skipped', 0)} skipped)."
        )
        result["source"] = "gmail_bank_txn"
        return result

    async def purge_promotional_imports(self, user_id: str) -> Dict[str, Any]:
        """Remove Gmail imports that are promotional or lack payment confirmation."""
        expenses = await self.db.get_expenses(user_id)
        removed: List[Dict[str, Any]] = []

        for expense in expenses:
            metadata = expense.get("metadata") or {}
            if metadata.get("source") != "gmail_sync":
                continue

            subject = metadata.get("gmail_subject") or ""
            note = expense.get("note") or ""
            merchant = expense.get("merchant") or ""
            snippet = metadata.get("gmail_body_snippet") or ""
            combined_note = f"{note}\n{snippet}".strip()

            if should_keep_gmail_expense(subject, combined_note, merchant):
                continue

            expense_id = expense.get("id")
            if not expense_id:
                continue
            deleted = await self.db.delete_expense(user_id, expense_id)
            if deleted:
                removed.append(
                    {
                        "expense_id": expense_id,
                        "merchant": merchant,
                        "amount": expense.get("amount"),
                        "date": expense.get("date"),
                        "subject": subject[:120] or note[:120],
                    }
                )

        return {
            "status": "success",
            "removed": len(removed),
            "message": f"Removed {len(removed)} invalid Gmail import(s).",
            "items": removed,
        }


gmail_bill_service = GmailBillService()
