"""
Automatic bill reminder emails.

Runs on a schedule (hourly) and sends emails when bills are due in 7, 3, 1, or 0 days.
Requires SMTP_* in .env and user email from Google profile.
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from database import Database
from services.notifications import notification_service
from services.storage import MongoManager

# Days before due date to send reminders
DEFAULT_REMINDER_DAYS = [7, 3, 1, 0]


def _parse_reminder_days() -> List[int]:
    raw = os.getenv("BILL_REMINDER_DAYS", "7,3,1,0")
    try:
        return sorted({int(x.strip()) for x in raw.split(",") if x.strip() != ""}, reverse=True)
    except ValueError:
        return DEFAULT_REMINDER_DAYS


async def get_user_email(user_id: str, mongo: MongoManager) -> Optional[str]:
    if user_id.startswith("email_"):
        return user_id.replace("email_", "", 1)
    doc = await mongo.google_collection.find_one({"user_id": user_id})
    if doc and doc.get("profile"):
        return doc["profile"].get("email")
    return None


async def process_bill_reminders(db: Optional[Database] = None) -> Dict[str, Any]:
    """
    Scan all bill reminders and send emails where appropriate.
    Returns summary stats.
    """
    if not notification_service.email_configured():
        return {
            "status": "skipped",
            "message": "SMTP not configured",
            "sent": 0,
        }

    database = db or Database()
    mongo = MongoManager()
    reminder_days = _parse_reminder_days()
    today = datetime.now(timezone.utc).date()

    sent = 0
    skipped = 0
    errors: List[str] = []

    bills = await database.get_all_bills_for_reminders()
    for bill in bills:
        user_id = bill.get("user_id")
        due_str = bill.get("due_date")
        if not user_id or not due_str:
            skipped += 1
            continue

        try:
            due = datetime.strptime(due_str[:10], "%Y-%m-%d").date()
        except ValueError:
            skipped += 1
            continue

        days_until = (due - today).days
        if days_until not in reminder_days:
            continue

        bill_id = bill.get("id") or bill.get("_id")
        reminder_key = f"{due_str}:{days_until}"
        if reminder_key in (bill.get("reminders_sent") or []):
            skipped += 1
            continue

        prefs = await database.get_notification_settings(user_id)
        if not prefs.get("bill_email_enabled", True):
            skipped += 1
            continue

        email = prefs.get("alert_email") or await get_user_email(user_id, mongo)
        if not email:
            skipped += 1
            continue

        result = await notification_service.send_bill_reminder(
            email,
            bill.get("bill_name", "Bill"),
            due_str,
            float(bill.get("amount", 0)),
            currency=prefs.get("currency", "INR"),
        )
        if result.get("status") == "success":
            await database.mark_bill_reminder_sent(str(bill_id), reminder_key)
            sent += 1
        else:
            errors.append(f"{bill.get('bill_name')}: {result.get('message', 'failed')}")

    return {
        "status": "success",
        "sent": sent,
        "skipped": skipped,
        "errors": errors[:10],
        "reminder_days": reminder_days,
    }
