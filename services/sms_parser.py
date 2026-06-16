"""
Parse Indian bank SMS alerts and create expenses.

Examples:
  Rs. 450 spent on Amazon
  INR 599 debited from A/C XXXX1234
  Rs. 1200 debited from HDFC Bank
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import os

from services.transaction_filters import (
    categorize_text,
    extract_debit_amount,
    is_credit_only,
    is_promotional,
)

APP_TZ = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Kolkata"))

AMOUNT_PATTERNS = [
    r"(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d{1,2})?)\s*(?:spent|debited|paid|withdrawn|deducted)",
    r"(?:spent|debited|paid|withdrawn)\s*(?:Rs\.?|INR|₹)?\s*([\d,]+(?:\.\d{1,2})?)",
    r"(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d{1,2})?)",
    r"([\d,]+(?:\.\d{1,2})?)\s*(?:Rs\.?|INR|₹)\s*(?:debited|spent|paid)",
    r"(?:debited|credited).*?(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d{1,2})?)",
]

BANK_HINTS = {
    "hdfc": "HDFC Bank",
    "sbi": "SBI",
    "icici": "ICICI Bank",
    "axis": "Axis Bank",
    "kotak": "Kotak Mahindra",
    "pnb": "PNB",
    "bob": "Bank of Baroda",
    "canara": "Canara Bank",
    "idfc": "IDFC First Bank",
    "yes bank": "Yes Bank",
    "indusind": "IndusInd Bank",
}

MERCHANT_HINTS = {
    "amazon": ("Amazon", "Shopping"),
    "flipkart": ("Flipkart", "Shopping"),
    "swiggy": ("Swiggy", "Food"),
    "zomato": ("Zomato", "Food"),
    "uber": ("Uber", "Travel"),
    "ola": ("Ola", "Travel"),
    "paytm": ("Paytm", "Bills"),
    "phonepe": ("PhonePe", "Bills"),
    "gpay": ("Google Pay", "Bills"),
    "netflix": ("Netflix", "Entertainment"),
    "spotify": ("Spotify", "Entertainment"),
}


def _today() -> str:
    return datetime.now(APP_TZ).strftime("%Y-%m-%d")


def _parse_amount(text: str) -> Optional[float]:
    for pattern in AMOUNT_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1).replace(",", ""))
                if val > 0:
                    return val
            except ValueError:
                continue
    return None


def _detect_bank(text: str) -> str:
    lower = text.lower()
    for key, label in BANK_HINTS.items():
        if key in lower:
            return label
    return "Bank Transaction"


def _detect_merchant_and_category(text: str) -> tuple[str, str]:
    lower = text.lower()
    for key, (merchant, category) in MERCHANT_HINTS.items():
        if key in lower:
            return merchant, category
    if "atm" in lower:
        return "ATM Withdrawal", "Other"
    if "upi" in lower:
        return "UPI Payment", "Bills"
    return _detect_bank(text), "Other"


def parse_sms_alert(sms_text: str) -> Optional[Dict[str, Any]]:
    """Parse a single Indian bank SMS into expense fields."""
    text = (sms_text or "").strip()
    if not text or len(text) < 10:
        return None

    if is_promotional("", text) or is_credit_only(text):
        return None

    amount = extract_debit_amount(text) or _parse_amount(text)
    if not amount:
        return None

    merchant, category = _detect_merchant_and_category(text)
    mapped = categorize_text(text, merchant)
    if mapped != "Other":
        category = mapped

    return {
        "amount": amount,
        "merchant": merchant,
        "category": category,
        "date": _today(),
        "note": text[:200],
        "currency": "INR",
        "metadata": {
            "source": "sms_parse",
            "sms_hash": str(abs(hash(text))),
            "raw_sms": text[:500],
        },
    }


def parse_sms_batch(sms_text: str) -> List[Dict[str, Any]]:
    """Parse multiple SMS messages separated by blank lines."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", sms_text) if b.strip()]
    if len(blocks) <= 1:
        blocks = [ln.strip() for ln in sms_text.splitlines() if ln.strip()]
    results = []
    seen = set()
    for block in blocks:
        parsed = parse_sms_alert(block)
        if not parsed:
            continue
        key = parsed["metadata"]["sms_hash"]
        if key in seen:
            continue
        seen.add(key)
        results.append(parsed)
    return results
