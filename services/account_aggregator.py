"""
India Account Aggregator (AA) integration — demo + provider-ready structure.

Supported providers: Setu, Finvu, OneMoney
Flow: Connect Bank → AA Consent → User Approves → Fetch Transactions → Store in MongoDB

When AA_API_KEY / SETU_API_KEY are not set, runs in sandbox mode with sample Indian bank data.
"""

from __future__ import annotations

import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from database import Database

AA_PROVIDERS = ("setu", "finvu", "onemoney")

INDIAN_BANKS = (
    "HDFC Bank",
    "SBI",
    "ICICI Bank",
    "Axis Bank",
    "Kotak Mahindra",
    "PNB",
)

# Sample transactions for sandbox / interview demo
SAMPLE_AA_TRANSACTIONS = [
    {"date_offset": 1, "amount": 450.0, "merchant": "Amazon India", "category": "Shopping", "note": "UPI — Amazon order"},
    {"date_offset": 2, "amount": 1200.0, "merchant": "HDFC Debit", "category": "Bills", "note": "Electricity bill payment"},
    {"date_offset": 3, "amount": 299.0, "merchant": "Swiggy", "category": "Food", "note": "Food delivery"},
    {"date_offset": 5, "amount": 85.0, "merchant": "Uber", "category": "Travel", "note": "Cab ride"},
    {"date_offset": 7, "amount": 1500.0, "merchant": "Flipkart", "category": "Shopping", "note": "Online purchase"},
    {"date_offset": 10, "amount": 350.0, "merchant": "IRCTC", "category": "Travel", "note": "Train ticket"},
]


class AccountAggregatorService:
    def __init__(self, database: Optional[Database] = None) -> None:
        self.db = database or Database()

    def is_configured(self) -> bool:
        return bool(
            os.getenv("SETU_API_KEY")
            or os.getenv("FINVU_API_KEY")
            or os.getenv("AA_API_KEY")
        )

    def status(self) -> Dict[str, Any]:
        return {
            "configured": self.is_configured(),
            "mode": "production" if self.is_configured() else "sandbox",
            "providers": list(AA_PROVIDERS),
            "banks": list(INDIAN_BANKS),
            "flow": [
                "Connect Bank Account",
                "AA Consent Screen",
                "User Approves",
                "Fetch Transactions",
                "Categorize Expenses",
                "Store in MongoDB",
            ],
        }

    async def create_consent(
        self,
        user_id: str,
        provider: str,
        bank_name: str,
    ) -> Dict[str, Any]:
        provider = (provider or "setu").lower()
        if provider not in AA_PROVIDERS:
            return {"status": "error", "message": f"Provider must be one of: {', '.join(AA_PROVIDERS)}"}

        bank = bank_name or "HDFC Bank"
        consent_id = f"aa_{secrets.token_hex(8)}"
        connection_id = await self.db.create_aa_connection(
            user_id=user_id,
            provider=provider,
            bank_name=bank,
            consent_id=consent_id,
            status="pending",
        )
        return {
            "status": "success",
            "consent_id": consent_id,
            "connection_id": connection_id,
            "provider": provider,
            "bank_name": bank,
            "consent_url": f"/bank?aa_consent={consent_id}",
            "message": f"AA consent created via {provider.title()}. Approve to link {bank}.",
            "sandbox": not self.is_configured(),
        }

    async def approve_consent(
        self,
        user_id: str,
        consent_id: str,
        approved: bool = True,
    ) -> Dict[str, Any]:
        conn = await self.db.get_aa_connection_by_consent(user_id, consent_id)
        if not conn:
            return {"status": "error", "message": "Consent session not found or expired."}
        if not approved:
            await self.db.update_aa_connection_status(str(conn["_id"]), "rejected")
            return {"status": "success", "message": "Consent declined.", "approved": False}

        await self.db.update_aa_connection_status(str(conn["_id"]), "active")
        return {
            "status": "success",
            "approved": True,
            "connection_id": str(conn["_id"]),
            "bank_name": conn.get("bank_name"),
            "provider": conn.get("provider"),
            "message": f"{conn.get('bank_name')} linked via Account Aggregator ({conn.get('provider')}).",
        }

    def _fetch_from_provider(
        self,
        provider: str,
        bank_name: str,
        connection_id: str,
    ) -> List[Dict[str, Any]]:
        """Sandbox: return sample Indian bank transactions."""
        today = datetime.now(timezone.utc).date()
        txns = []
        for i, sample in enumerate(SAMPLE_AA_TRANSACTIONS):
            txn_date = (today - timedelta(days=sample["date_offset"])).isoformat()
            txns.append({
                "external_id": f"aa_{connection_id}_{i}_{sample['date_offset']}",
                "date": txn_date,
                "amount": sample["amount"],
                "merchant": sample["merchant"],
                "category": sample["category"],
                "note": sample["note"],
                "bank": bank_name,
                "provider": provider,
            })
        return txns

    async def sync_transactions(self, user_id: str, connection_id: str) -> Dict[str, Any]:
        conn = await self.db.get_aa_connection(user_id, connection_id)
        if not conn:
            return {"status": "error", "message": "AA connection not found."}
        if conn.get("status") != "active":
            return {"status": "error", "message": "Complete AA consent before syncing transactions."}

        provider = conn.get("provider", "setu")
        bank_name = conn.get("bank_name", "HDFC Bank")
        raw_txns = self._fetch_from_provider(provider, bank_name, connection_id)

        imported = 0
        skipped = 0
        items = []
        for txn in raw_txns:
            external_id = txn["external_id"]
            if await self.db.expense_exists_by_source_id(user_id, "aa_sync", external_id):
                skipped += 1
                continue
            expense_id = await self.db.add_expense(
                user_id,
                txn["date"],
                txn["amount"],
                txn["category"],
                txn["note"],
                txn["merchant"],
                metadata={
                    "source": "aa_sync",
                    "external_id": external_id,
                    "aa_provider": provider,
                    "aa_connection_id": connection_id,
                    "bank_name": bank_name,
                },
            )
            if expense_id:
                imported += 1
                items.append({
                    "expense_id": expense_id,
                    "merchant": txn["merchant"],
                    "amount": txn["amount"],
                    "date": txn["date"],
                    "category": txn["category"],
                })
            else:
                skipped += 1

        await self.db.record_aa_sync(user_id, connection_id, imported)
        return {
            "status": "success",
            "imported": imported,
            "skipped": skipped,
            "message": f"Synced {imported} transaction(s) from {bank_name} via {provider.title()}.",
            "items": items,
            "sandbox": not self.is_configured(),
        }

    async def list_connections(self, user_id: str) -> List[Dict[str, Any]]:
        return await self.db.list_aa_connections(user_id)


aa_service = AccountAggregatorService()
