"""
Shared rules for parsing real expenses from bank alerts, SMS, and Gmail receipts.
Filters out promotional/marketing emails (vouchers, cashback offers, etc.).
"""

from __future__ import annotations

import re
from email.utils import parseaddr
from typing import Dict, List, Optional, Tuple

# Skip promotional keywords (subject + body)
SKIP_KEYWORDS = [
    "voucher",
    "cashback",
    "offer",
    "discount",
    "welcome",
    "joining fee",
    "unlock",
    "win",
    "free",
]

SKIP_IF_CONTAINS = [
    "voucher",
    "offer",
    "cashback",
    "win",
    "free",
    "discount",
    "congratulations",
    "welcome",
    "joining",
    "unlock",
    "promo",
    "get up to",
    "get ₹",
    "get rs",
    "limited time",
    "claim now",
    "click here",
    "emi!",
]

# Debit keywords (real expenses)
DEBIT_WORDS = ["debited", "deducted", "paid", "payment of", "spent", "withdrawn"]

# Credit keywords (income — not expenses)
CREDIT_WORDS = ["credited", "received", "refund", "cashback added", "salary credited"]

TRUSTED_SENDERS = [
    "alerts@hdfcbank.net",
    "alerts@sbi.co.in",
    "alerts@icicibank.com",
    "noreply@axisbank.com",
]

TRUSTED_SENDER_DOMAINS = [
    "hdfcbank.net",
    "hdfcbank.com",
    "sbi.co.in",
    "onlinesbi.com",
    "icicibank.com",
    "axisbank.com",
    "kotak.com",
    "yesbank.in",
    "idfcbank.com",
    "indusind.com",
    "pnb.co.in",
    "bankofbaroda.com",
    "canarabank.com",
]

DEBIT_PATTERNS = [
    r"(?:rs\.?|inr|₹)\s*([\d,]+\.?\d*)\s+(?:debited|deducted|spent|paid|withdrawn)",
    r"(?:debited|deducted|spent|paid|withdrawn).*?(?:rs\.?|inr|₹)\s*([\d,]+\.?\d*)",
    r"payment.*?(?:rs\.?|inr|₹)\s*([\d,]+\.?\d*)",
]

REQUIRED_FIELD_PATTERNS: Dict[str, str] = {
    "account number": r"(?:account|a/c|ac)\s*(?:no|number|#)?",
    "balance": r"(?:balance|avl\.?\s*bal|available\s+balance|bal\s+)",
    "transaction id": r"(?:transaction\s*id|txn\s*id|utr|ref\.?\s*no|reference\s*(?:no|number|id)|order\s*id)",
}

CATEGORY_MAP: Dict[str, str] = {
    r"swiggy|zomato|food|snacks|restaurant": "Food",
    r"metro|uber|ola|irctc|travel|flight|makemytrip|delhi\s*metro": "Travel",
    r"amazon|flipkart|myntra|shopping|ajio": "Shopping",
    r"electricity|water|gas|utility|bill payment": "Bills",
    r"netflix|spotify|bookmyshow|entertainment": "Entertainment",
    r"pharmacy|medical|health|hospital": "Health",
}

MERCHANT_PAYMENT_MARKERS = [
    "payment successful",
    "payment of",
    "amount paid",
    "amount debited",
    "order confirmed",
    "booking confirmed",
    "debited",
    "paid rs",
    "paid ₹",
    "transaction successful",
    "successfully paid",
    "we received your payment",
    "payment received for",
]

# Weak markers alone are not enough to import (order id in promos, etc.)
WEAK_MERCHANT_MARKERS = [
    "your order",
    "order id",
    "invoice",
    "receipt",
    "e-ticket",
    "boarding pass",
    "ticket",
    "transaction id",
    "utr:",
]

STRONG_PAYMENT_MARKERS = [
    "payment successful",
    "amount paid",
    "amount debited",
    "payment of",
    "successfully paid",
    "we received your payment",
    "payment received for",
    "order confirmed",
    "booking confirmed",
    "paid rs",
    "paid ₹",
    "debited",
    "spent on",
    "withdrawn",
]


def _normalize_text(*parts: str) -> str:
    return " ".join(p for p in parts if p).lower()


def is_promotional(subject: str, body: str) -> bool:
    text = _normalize_text(subject, body)
    for kw in SKIP_IF_CONTAINS:
        if kw in text:
            return True
    for kw in SKIP_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", text):
            return True
    if re.search(r"get\s+(?:up\s+to\s+)?[₹rs]", text):
        return True
    if re.search(r"₹0\s+joining", text):
        return True
    return False


def is_credit_only(text: str) -> bool:
    lower = text.lower()
    has_credit = any(w in lower for w in CREDIT_WORDS)
    has_debit = any(w in lower for w in DEBIT_WORDS)
    return has_credit and not has_debit


def is_debit_transaction(text: str) -> bool:
    lower = text.lower()
    if any(w in lower for w in DEBIT_WORDS):
        return True
    for pattern in DEBIT_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return True
    return False


def sender_address(sender: str) -> str:
    _, addr = parseaddr(sender or "")
    return (addr or "").lower()


def is_trusted_bank_sender(sender: str) -> bool:
    addr = sender_address(sender)
    if not addr:
        return False
    if addr in TRUSTED_SENDERS:
        return True
    domain = addr.split("@")[-1]
    return any(domain.endswith(d) for d in TRUSTED_SENDER_DOMAINS)


def has_required_bank_fields(text: str) -> bool:
    lower = text.lower()
    for pattern in REQUIRED_FIELD_PATTERNS.values():
        if not re.search(pattern, lower, re.IGNORECASE):
            return False
    return True


def has_strong_payment_proof(text: str) -> bool:
    lower = text.lower()
    if any(marker in lower for marker in STRONG_PAYMENT_MARKERS):
        return True
    if is_debit_transaction(lower):
        return True
    return False


def has_merchant_payment_proof(text: str) -> bool:
    lower = text.lower()
    if has_strong_payment_proof(lower):
        return True
    if any(marker in lower for marker in WEAK_MERCHANT_MARKERS):
        return bool(extract_merchant_amount(text))
    return False


def categorize_text(*parts: str) -> str:
    text = _normalize_text(*parts)
    for pattern, category in CATEGORY_MAP.items():
        if re.search(pattern, text, re.IGNORECASE):
            return category
    return "Other"


def extract_debit_amount(text: str) -> Optional[float]:
    if not text:
        return None
    for pattern in DEBIT_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1).replace(",", ""))
                if val >= 1:
                    return val
            except ValueError:
                continue

    payment_patterns = [
        r"(?:amount paid|paid|total paid|payment of|fare|amount)[:\s]*(?:₹|rs\.?|inr)?\s*([\d,]+(?:\.\d{1,2})?)",
        r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d{1,2})?)\s*(?:paid|debited|spent)",
        r"(?:order total|grand total|total amount)[:\s]*(?:₹|rs\.?|inr)?\s*([\d,]+(?:\.\d{1,2})?)",
    ]
    for pattern in payment_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1).replace(",", ""))
                if val >= 1:
                    return val
            except ValueError:
                continue
    return None


def extract_merchant_amount(text: str) -> Optional[float]:
    """Extract amount from merchant receipts (order confirmations, tickets)."""
    amount = extract_debit_amount(text)
    if amount:
        return amount
    if not has_merchant_payment_proof(text):
        return None
    if re.search(REQUIRED_FIELD_PATTERNS["transaction id"], text, re.IGNORECASE):
        match = re.search(
            r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d{1,2})?)",
            text,
            re.IGNORECASE,
        )
        if match:
            try:
                val = float(match.group(1).replace(",", ""))
                if val >= 1:
                    return val
            except ValueError:
                pass
    return None


def validate_email_for_import(sender: str, subject: str, body: str) -> Tuple[bool, Optional[str]]:
    """
    Decide whether an email should become an expense.
    Returns (should_import, skip_reason).
    """
    combined = _normalize_text(subject, body)

    if is_promotional(subject, body):
        return False, "promotional"

    if is_credit_only(combined):
        return False, "credit_not_expense"

    if is_trusted_bank_sender(sender) or any(
        bank in combined for bank in ("hdfc bank", "sbi", "icici bank", "axis bank", "debited from a/c")
    ):
        if not has_required_bank_fields(combined):
            return False, "missing_bank_fields"
        if not is_debit_transaction(combined):
            return False, "not_debit"
        if not extract_debit_amount(combined):
            return False, "no_amount"
        return True, None

    if not has_merchant_payment_proof(combined):
        return False, "no_payment_confirmation"

    if not has_strong_payment_proof(combined):
        return False, "not_confirmed_payment"

    amount = extract_merchant_amount(combined)
    if not amount:
        return False, "no_amount"

    return True, None


def should_keep_gmail_expense(subject: str, note: str, merchant: str = "") -> bool:
    """Return True if a stored Gmail import looks like a real payment."""
    combined = _normalize_text(subject, note, merchant)
    if is_promotional(subject, combined):
        return False
    if is_credit_only(combined):
        return False
    if has_strong_payment_proof(combined):
        return bool(extract_merchant_amount(combined) or extract_debit_amount(combined))
    if has_required_bank_fields(combined) and is_debit_transaction(combined):
        return bool(extract_debit_amount(combined))
    if re.search(REQUIRED_FIELD_PATTERNS["transaction id"], combined, re.IGNORECASE):
        return bool(extract_merchant_amount(combined))
    return False
