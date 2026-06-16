"""
Bill reminders & alerts via email (optional).

Set in .env:
  SMTP_HOST=smtp.gmail.com
  SMTP_PORT=587
  SMTP_USER=...
  SMTP_PASSWORD=...
  SMTP_FROM=noreply@you.com

Push (FCM) — set FCM_SERVER_KEY when ready.
"""

import os
import smtplib
from email.message import EmailMessage
from typing import Any, Dict, List, Optional


class NotificationService:
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "").strip()
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "").strip()
        self.smtp_password = os.getenv("SMTP_PASSWORD", "").strip()
        self.smtp_from = os.getenv("SMTP_FROM", self.smtp_user).strip()
        self.fcm_key = os.getenv("FCM_SERVER_KEY", "").strip()

    def email_configured(self) -> bool:
        return bool(self.smtp_host and self.smtp_user and self.smtp_password)

    def push_configured(self) -> bool:
        return bool(self.fcm_key)

    async def send_email(
        self,
        to_address: str,
        subject: str,
        body: str,
        html: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.email_configured():
            return {
                "status": "skipped",
                "message": "Email not configured. Set SMTP_HOST, SMTP_USER, SMTP_PASSWORD in .env",
            }
        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = self.smtp_from
            msg["To"] = to_address
            msg.set_content(body)
            if html:
                msg.add_alternative(html, subtype="html")

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            return {"status": "success", "message": f"Email sent to {to_address}"}
        except Exception as exc:
            return {"status": "error", "message": f"Email failed: {exc}"}

    async def send_bill_reminder(
        self,
        to_address: str,
        bill_name: str,
        due_date: str,
        amount: float,
        currency: str = "INR",
    ) -> Dict[str, Any]:
        subject = f"Bill reminder: {bill_name} due {due_date}"
        body = (
            f"TrackExpensio reminder\n\n"
            f"Bill: {bill_name}\n"
            f"Amount: {currency} {amount:,.2f}\n"
            f"Due: {due_date}\n"
        )
        return await self.send_email(to_address, subject, body)

    async def send_push(self, device_token: str, title: str, body: str) -> Dict[str, Any]:
        if not self.push_configured():
            return {
                "status": "skipped",
                "message": "Push not configured. Set FCM_SERVER_KEY in .env",
            }
        # FCM HTTP v1 would go here; stub for integration point
        return {
            "status": "error",
            "message": "FCM send not implemented — add firebase-admin or httpx FCM call",
        }

    def status(self) -> Dict[str, bool]:
        return {
            "email": self.email_configured(),
            "push": self.push_configured(),
        }


notification_service = NotificationService()
