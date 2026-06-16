import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient


def _serialize(document: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize MongoDB document to JSON-serializable dict, converting datetime objects to strings."""
    if not document:
        return document
    doc = document.copy()
    
    # Handle _id separately
    if "_id" in doc:
        doc["id"] = str(doc.pop("_id"))
    
    # Convert datetime objects to ISO format strings
    def convert_value(v):
        if isinstance(v, datetime):
            return v.isoformat()
        elif isinstance(v, dict):
            return {k: convert_value(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [convert_value(item) for item in v]
        else:
            return v
    
    # Apply conversion to all values
    for key, value in doc.items():
        doc[key] = convert_value(value)
    
    return doc


class Database:
    """
    MongoDB implementation of the storage layer. Each logical resource has its
    own collection and every document is partitioned by `user_id`, ensuring
    user isolation.
    """

    def __init__(self) -> None:
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGODB_DB", "ai_expense_tracker")
        self.client = AsyncIOMotorClient(uri)
        self.db = self.client[db_name]
        self.expenses = self.db["expenses"]
        self.income = self.db["income"]
        self.budgets = self.db["budgets"]
        self.recurring = self.db["recurring_expenses"]
        self.recurring_log = self.db["recurring_log"]
        self.savings = self.db["savings_goals"]
        self.debts = self.db["debts"]
        self.debt_payments = self.db["debt_payments"]
        self.investments = self.db["investments"]
        self.bills = self.db["bill_reminders"]
        self.settings = self.db["user_settings"]
        self.roles = self.db["roles"]
        self.access = self.db["shared_access"]
        self.bank_connections = self.db["bank_connections"]
        self.aa_connections = self.db["aa_connections"]
        self.bank_sync = self.db["bank_sync_log"]
        self.audit_logs = self.db["audit_logs"]
        self.fx_rates = self.db["fx_rates"]
        self.tax_settings = self.db["tax_settings"]

    # -------------------- Expense --------------------
    async def add_expense(
        self,
        user_id: str,
        date: str,
        amount: float,
        category: str,
        note: str,
        merchant: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        doc = {
            "user_id": user_id,
            "date": date,
            "amount": float(amount),
            "category": category,
            "note": note,
            "merchant": merchant,
            "created_at": datetime.now(timezone.utc),
        }
        if metadata:
            doc["metadata"] = metadata
        result = await self.expenses.insert_one(doc)
        return str(result.inserted_id)

    async def expense_exists_by_gmail_id(self, user_id: str, gmail_message_id: str) -> bool:
        if not gmail_message_id:
            return False
        existing = await self.expenses.find_one(
            {"user_id": user_id, "metadata.gmail_message_id": gmail_message_id}
        )
        return existing is not None

    async def expense_exists_by_plaid_id(self, user_id: str, external_id: str) -> bool:
        if not external_id:
            return False
        existing = await self.expenses.find_one(
            {"user_id": user_id, "metadata.external_id": external_id, "metadata.source": "plaid"}
        )
        return existing is not None

    async def expense_exists_by_source_id(
        self, user_id: str, source: str, external_id: str
    ) -> bool:
        if not external_id:
            return False
        existing = await self.expenses.find_one(
            {
                "user_id": user_id,
                "metadata.external_id": external_id,
                "metadata.source": source,
            }
        )
        return existing is not None

    async def update_expense(
        self, user_id: str, expense_id: str, updates: Dict[str, Any]
    ) -> bool:
        allowed = {"date", "amount", "category", "note", "merchant"}
        payload = {k: v for k, v in updates.items() if k in allowed and v is not None}
        if not payload:
            return False
        result = await self.expenses.update_one(
            {"_id": ObjectId(expense_id), "user_id": user_id}, {"$set": payload}
        )
        return result.modified_count > 0

    async def delete_expense(self, user_id: str, expense_id: str) -> bool:
        result = await self.expenses.delete_one(
            {"_id": ObjectId(expense_id), "user_id": user_id}
        )
        return result.deleted_count > 0

    async def get_expenses(
        self, user_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {"user_id": user_id}
        if start_date and end_date:
            query["date"] = {"$gte": start_date, "$lte": end_date}
        elif start_date:
            query["date"] = {"$gte": start_date}
        elif end_date:
            query["date"] = {"$lte": end_date}

        cursor = self.expenses.find(query).sort("date", -1).limit(2000)
        docs = await cursor.to_list(length=None)
        return [_serialize(doc) for doc in docs]

    async def get_expense_overview(
        self, user_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        expenses = await self.get_expenses(user_id, start_date, end_date)
        total = sum(float(e.get("amount") or 0) for e in expenses)
        categories: Dict[str, Dict[str, Any]] = {}

        for expense in expenses:
            category = expense.get("category") or "Other"
            amount = float(expense.get("amount") or 0)
            metadata = expense.get("metadata") or {}
            if category not in categories:
                categories[category] = {
                    "category": category,
                    "total": 0.0,
                    "count": 0,
                    "items": [],
                }
            bucket = categories[category]
            bucket["total"] += amount
            bucket["count"] += 1
            bucket["items"].append(
                {
                    "id": expense.get("id"),
                    "date": expense.get("date"),
                    "amount": amount,
                    "merchant": expense.get("merchant") or "—",
                    "note": expense.get("note") or "",
                    "source": metadata.get("source") or metadata.get("import_source") or "",
                }
            )

        category_list = sorted(categories.values(), key=lambda row: row["total"], reverse=True)
        for row in category_list:
            row["items"].sort(key=lambda item: item.get("date") or "", reverse=True)
            row["total"] = round(row["total"], 2)

        return {
            "total": round(total, 2),
            "count": len(expenses),
            "categories": category_list,
            "expenses": expenses,
            "start_date": start_date,
            "end_date": end_date,
        }

    async def get_expense_summary(
        self, user_id: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        pipeline = [
            {
                "$match": {
                    "user_id": user_id,
                    "date": {"$gte": start_date, "$lte": end_date},
                }
            },
            {
                "$group": {
                    "_id": "$category",
                    "total": {"$sum": "$amount"},
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"total": -1}},
        ]
        docs = await self.expenses.aggregate(pipeline).to_list(length=None)
        return [
            {"category": doc["_id"], "total": doc["total"], "count": doc["count"]}
            for doc in docs
        ]

    async def get_spending_by_category(
        self, user_id: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        return await self.get_expense_summary(user_id, start_date, end_date)

    # -------------------- Income --------------------
    async def add_income(
        self, user_id: str, date: str, amount: float, source: str, category: str, note: str
    ) -> str:
        doc = {
            "user_id": user_id,
            "date": date,
            "amount": float(amount),
            "source": source,
            "category": category,
            "note": note,
            "created_at": datetime.now(timezone.utc),
        }
        result = await self.income.insert_one(doc)
        return str(result.inserted_id)

    async def get_income(
        self, user_id: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        cursor = (
            self.income.find(
                {"user_id": user_id, "date": {"$gte": start_date, "$lte": end_date}}
            )
            .sort("date", -1)
            .limit(500)
        )
        docs = await cursor.to_list(length=None)
        return [_serialize(doc) for doc in docs]

    async def get_income_summary(
        self, user_id: str, start_date: str, end_date: str
    ) -> Dict[str, Any]:
        pipeline = [
            {
                "$match": {
                    "user_id": user_id,
                    "date": {"$gte": start_date, "$lte": end_date},
                }
            },
            {
                "$group": {
                    "_id": "$category",
                    "total": {"$sum": "$amount"},
                    "count": {"$sum": 1},
                }
            },
        ]
        docs = await self.income.aggregate(pipeline).to_list(length=None)
        total = sum(doc["total"] for doc in docs)
        return {"total": total, "by_category": docs}

    # -------------------- Budgets --------------------
    async def set_budget(self, user_id: str, category: str, amount: float) -> str:
        result = await self.budgets.update_one(
            {"user_id": user_id, "category": category},
            {
                "$set": {
                    "user_id": user_id,
                    "category": category,
                    "amount": float(amount),
                    "updated_at": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )
        if result.upserted_id:
            return str(result.upserted_id)
        doc = await self.budgets.find_one({"user_id": user_id, "category": category})
        return str(doc["_id"])

    async def get_budgets(self, user_id: str) -> List[Dict[str, Any]]:
        cursor = self.budgets.find({"user_id": user_id})
        docs = await cursor.to_list(length=None)
        return [_serialize(doc) for doc in docs]

    # -------------------- Recurring --------------------
    async def add_recurring_expense(
        self,
        user_id: str,
        amount: float,
        category: str,
        frequency: str,
        merchant: str,
    ) -> str:
        doc = {
            "user_id": user_id,
            "amount": float(amount),
            "category": category,
            "frequency": frequency,
            "merchant": merchant,
            "created_at": datetime.now(timezone.utc),
        }
        result = await self.recurring.insert_one(doc)
        return str(result.inserted_id)

    async def get_recurring_expenses(self, user_id: str) -> List[Dict[str, Any]]:
        cursor = self.recurring.find({"user_id": user_id})
        return [_serialize(doc) for doc in await cursor.to_list(length=None)]

    async def check_recurring_expense_generated(
        self, user_id: str, recurring_id: str, start_date: str, end_date: str
    ) -> bool:
        doc = await self.recurring_log.find_one(
            {
                "user_id": user_id,
                "recurring_id": recurring_id,
                "period_start": start_date,
                "period_end": end_date,
            }
        )
        return doc is not None

    async def mark_recurring_generated(self, recurring_id: str, expense_date: str) -> None:
        await self.recurring_log.insert_one(
            {
                "recurring_id": recurring_id,
                "period_start": expense_date,
                "period_end": expense_date,
                "created_at": datetime.now(timezone.utc),
            }
        )

    # -------------------- Savings --------------------
    async def save_savings_goal(
        self,
        user_id: str,
        goal_name: str,
        target_amount: float,
        target_date: str,
        current_amount: float = 0,
        goal_number: str = "",
        goal_id: Optional[str] = None,
    ) -> str:
        goal_name = goal_name.strip()
        doc = {
            "user_id": user_id,
            "goal_name": goal_name,
            "goal_number": str(goal_number).strip() if goal_number else "",
            "target_amount": float(target_amount),
            "target_date": target_date,
            "current_amount": float(current_amount),
            "updated_at": datetime.now(timezone.utc),
        }
        if goal_id:
            await self.savings.update_one(
                {"_id": ObjectId(goal_id), "user_id": user_id},
                {"$set": doc},
            )
            return goal_id
        doc["created_at"] = datetime.now(timezone.utc)
        result = await self.savings.insert_one(doc)
        return str(result.inserted_id)

    async def set_savings_goal(
        self, user_id: str, goal_name: str, target_amount: float, target_date: str, current_amount: float
    ) -> str:
        """Legacy upsert by goal name (chat agent)."""
        doc = {
            "user_id": user_id,
            "goal_name": goal_name.strip(),
            "target_amount": float(target_amount),
            "target_date": target_date,
            "current_amount": float(current_amount),
            "updated_at": datetime.now(timezone.utc),
        }
        result = await self.savings.update_one(
            {"user_id": user_id, "goal_name": goal_name.strip()}, {"$set": doc}, upsert=True
        )
        if result.upserted_id:
            return str(result.upserted_id)
        record = await self.savings.find_one({"user_id": user_id, "goal_name": goal_name.strip()})
        return str(record["_id"])

    async def delete_savings_goal(self, user_id: str, goal_id: str) -> bool:
        try:
            result = await self.savings.delete_one(
                {"_id": ObjectId(goal_id), "user_id": user_id}
            )
            return result.deleted_count > 0
        except Exception:
            return False

    async def get_savings_goal(self, user_id: str, goal_id: str) -> Optional[Dict[str, Any]]:
        try:
            doc = await self.savings.find_one({"_id": ObjectId(goal_id), "user_id": user_id})
            return _serialize(doc) if doc else None
        except Exception:
            return None

    async def get_savings_goals(self, user_id: str) -> List[Dict[str, Any]]:
        cursor = self.savings.find({"user_id": user_id}).sort("updated_at", -1)
        return [_serialize(doc) for doc in await cursor.to_list(length=None)]

    # -------------------- Debt --------------------
    async def add_debt(
        self,
        user_id: str,
        creditor_name: str,
        total_amount: float,
        interest_rate: float,
        minimum_payment: float,
        due_date: str,
        debt_type: str,
    ) -> str:
        doc = {
            "user_id": user_id,
            "creditor_name": creditor_name,
            "total_amount": float(total_amount),
            "interest_rate": float(interest_rate),
            "minimum_payment": float(minimum_payment),
            "due_date": due_date,
            "debt_type": debt_type,
            "created_at": datetime.now(timezone.utc),
        }
        result = await self.debts.insert_one(doc)
        return str(result.inserted_id)

    async def record_debt_payment(
        self, user_id: str, debt_id: str, amount: float, payment_date: str
    ) -> str:
        doc = {
            "user_id": user_id,
            "debt_id": debt_id,
            "amount": float(amount),
            "payment_date": payment_date,
            "created_at": datetime.now(timezone.utc),
        }
        result = await self.debt_payments.insert_one(doc)
        return str(result.inserted_id)

    async def get_debts(self, user_id: str) -> List[Dict[str, Any]]:
        cursor = self.debts.find({"user_id": user_id})
        return [_serialize(doc) for doc in await cursor.to_list(length=None)]

    async def get_debt_payments(self, user_id: str, debt_id: str) -> List[List[Any]]:
        cursor = self.debt_payments.find({"user_id": user_id, "debt_id": debt_id})
        docs = await cursor.to_list(length=None)
        return [[doc["debt_id"], doc["payment_date"], doc["amount"]] for doc in docs]

    # -------------------- Investments --------------------
    async def add_investment(
        self,
        user_id: str,
        investment_name: str,
        investment_type: str,
        amount: float,
        purchase_date: str,
        current_value: float,
    ) -> str:
        doc = {
            "user_id": user_id,
            "investment_name": investment_name,
            "investment_type": investment_type,
            "amount": float(amount),
            "purchase_date": purchase_date,
            "current_value": float(current_value),
            "updated_at": datetime.utcnow(),
        }
        result = await self.investments.insert_one(doc)
        return str(result.inserted_id)

    async def update_investment_value(
        self, user_id: str, investment_id: str, current_value: float, update_date: str
    ) -> None:
        await self.investments.update_one(
            {"_id": ObjectId(investment_id), "user_id": user_id},
            {
                "$set": {
                    "current_value": float(current_value),
                    "last_valuation_date": update_date,
                }
            },
        )

    async def get_investments(self, user_id: str) -> List[Dict[str, Any]]:
        cursor = self.investments.find({"user_id": user_id})
        return [_serialize(doc) for doc in await cursor.to_list(length=None)]

    # -------------------- Bills --------------------
    async def add_bill_reminder(
        self,
        user_id: str,
        bill_name: str,
        amount: float,
        due_date: str,
        frequency: str,
        category: str,
    ) -> str:
        doc = {
            "user_id": user_id,
            "bill_name": bill_name,
            "amount": float(amount),
            "due_date": due_date,
            "frequency": frequency,
            "category": category,
            "reminders_sent": [],
            "created_at": datetime.now(timezone.utc),
        }
        result = await self.bills.insert_one(doc)
        return str(result.inserted_id)

    async def get_all_bills_for_reminders(self) -> List[Dict[str, Any]]:
        """All bill reminders (for scheduled email job)."""
        cursor = self.bills.find({})
        return [_serialize(doc) for doc in await cursor.to_list(length=None)]

    async def mark_bill_reminder_sent(self, bill_id: str, reminder_key: str) -> None:
        try:
            await self.bills.update_one(
                {"_id": ObjectId(bill_id)},
                {"$addToSet": {"reminders_sent": reminder_key}},
            )
        except Exception:
            pass

    async def get_notification_settings(self, user_id: str) -> Dict[str, Any]:
        doc = await self.settings.find_one({"user_id": user_id}) or {}
        return {
            "bill_email_enabled": doc.get("bill_email_enabled", True),
            "alert_email": doc.get("alert_email"),
            "currency": doc.get("currency", "INR"),
        }

    async def set_notification_settings(
        self,
        user_id: str,
        bill_email_enabled: Optional[bool] = None,
        alert_email: Optional[str] = None,
    ) -> None:
        update: Dict[str, Any] = {"user_id": user_id}
        if bill_email_enabled is not None:
            update["bill_email_enabled"] = bill_email_enabled
        if alert_email is not None:
            update["alert_email"] = alert_email.strip() if alert_email else None
        await self.settings.update_one(
            {"user_id": user_id},
            {"$set": update},
            upsert=True,
        )

    async def list_bank_connections(self, user_id: str) -> List[Dict[str, Any]]:
        cursor = self.bank_connections.find({"user_id": user_id}).sort("connected_at", -1)
        docs = await cursor.to_list(length=None)
        out = []
        for doc in docs:
            s = _serialize(doc)
            s.pop("plaid_access_token", None)
            out.append(s)
        return out

    async def get_upcoming_bills(
        self, user_id: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        cursor = self.bills.find(
            {
                "user_id": user_id,
                "due_date": {"$gte": start_date, "$lte": end_date},
            }
        ).sort("due_date", 1)
        return [_serialize(doc) for doc in await cursor.to_list(length=None)]

    # -------------------- FX / Settings --------------------
    async def set_base_currency(self, user_id: str, currency: str) -> None:
        await self.settings.update_one(
            {"user_id": user_id},
            {"$set": {"user_id": user_id, "base_currency": currency.upper()}},
            upsert=True,
        )

    async def get_user_base_currency(self, user_id: str) -> Optional[str]:
        doc = await self.settings.find_one({"user_id": user_id})
        return doc.get("base_currency") if doc else None

    async def get_exchange_rate(self, from_currency: str, to_currency: str) -> float:
        if from_currency.upper() == to_currency.upper():
            return 1.0
        url = f"https://open.er-api.com/v6/latest/{from_currency.upper()}"
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            rate = data["rates"].get(to_currency.upper())
            if not rate:
                raise RuntimeError("Unsupported currency conversion")
            return float(rate)

    # -------------------- Roles & Access --------------------
    async def create_role(self, role_name: str, permissions: List[str]) -> str:
        doc = {
            "role_name": role_name,
            "permissions": permissions,
            "created_at": datetime.now(timezone.utc),
        }
        result = await self.roles.insert_one(doc)
        return str(result.inserted_id)

    async def share_account_access(
        self, owner_user_id: str, target_username: str, access_level: str
    ) -> str:
        doc = {
            "owner_user_id": owner_user_id,
            "target_username": target_username,
            "access_level": access_level,
            "created_at": datetime.now(timezone.utc),
        }
        result = await self.access.insert_one(doc)
        return str(result.inserted_id)

    # -------------------- Bank Integrations --------------------
    async def connect_bank_account(
        self,
        user_id: str,
        bank_name: str,
        account_type: str,
        account_number_last4: str,
        plaid_access_token: Optional[str] = None,
        plaid_item_id: Optional[str] = None,
    ) -> str:
        now = datetime.now(timezone.utc)
        if plaid_item_id:
            existing = await self.bank_connections.find_one(
                {"user_id": user_id, "plaid_item_id": plaid_item_id}
            )
            if existing:
                update: Dict[str, Any] = {
                    "bank_name": bank_name,
                    "account_type": account_type,
                    "account_number_last4": account_number_last4,
                    "connected_at": now,
                }
                if plaid_access_token:
                    update["plaid_access_token"] = plaid_access_token
                await self.bank_connections.update_one(
                    {"_id": existing["_id"]},
                    {"$set": update},
                )
                return str(existing["_id"])

        doc = {
            "user_id": user_id,
            "bank_name": bank_name,
            "account_type": account_type,
            "account_number_last4": account_number_last4,
            "connected_at": now,
        }
        if plaid_access_token:
            doc["plaid_access_token"] = plaid_access_token
        if plaid_item_id:
            doc["plaid_item_id"] = plaid_item_id
        result = await self.bank_connections.insert_one(doc)
        return str(result.inserted_id)

    async def get_bank_connection(self, user_id: str, connection_id: str) -> Optional[Dict[str, Any]]:
        from bson import ObjectId
        try:
            return await self.bank_connections.find_one(
                {"_id": ObjectId(connection_id), "user_id": user_id}
            )
        except Exception:
            return None

    async def sync_bank_transactions(
        self, user_id: str, connection_id: str, start_date: str, end_date: str
    ) -> Dict[str, Any]:
        from services.bank_plaid import plaid_service

        conn = await self.get_bank_connection(user_id, connection_id)
        synced_count = 0
        note = "manual_connection"

        if conn and plaid_service.is_configured():
            token = conn.get("plaid_access_token")
            if token:
                transactions, plaid_err = await plaid_service.fetch_transactions(
                    token, start_date, end_date
                )
                for txn in transactions:
                    external_id = txn.get("external_id")
                    if external_id and await self.expense_exists_by_plaid_id(user_id, external_id):
                        continue
                    await self.add_expense(
                        user_id,
                        txn["date"],
                        txn["amount"],
                        txn.get("category", "Other"),
                        txn.get("note", ""),
                        txn.get("merchant", "Unknown"),
                        metadata={
                            "source": "plaid",
                            "external_id": external_id,
                            "connection_id": connection_id,
                        },
                    )
                    synced_count += 1
                note = "plaid_sync" if not plaid_err else f"plaid_sync_partial: {plaid_err}"
            else:
                note = "plaid_token_missing_use_link_flow"
        elif not plaid_service.is_configured():
            note = "set_PLAID_CLIENT_ID_and_PLAID_SECRET_in_env"

        await self.bank_sync.insert_one(
            {
                "user_id": user_id,
                "connection_id": connection_id,
                "start_date": start_date,
                "end_date": end_date,
                "synced_count": synced_count,
                "note": note,
                "created_at": datetime.now(timezone.utc),
            }
        )
        return {"synced_count": synced_count, "note": note}

    # -------------------- Account Aggregator (India) --------------------
    async def create_aa_connection(
        self,
        user_id: str,
        provider: str,
        bank_name: str,
        consent_id: str,
        status: str = "pending",
    ) -> str:
        doc = {
            "user_id": user_id,
            "provider": provider,
            "bank_name": bank_name,
            "consent_id": consent_id,
            "status": status,
            "connected_at": datetime.now(timezone.utc),
            "last_sync_at": None,
            "synced_count": 0,
        }
        result = await self.aa_connections.insert_one(doc)
        return str(result.inserted_id)

    async def list_aa_connections(self, user_id: str) -> List[Dict[str, Any]]:
        cursor = self.aa_connections.find({"user_id": user_id}).sort("connected_at", -1)
        return [_serialize(doc) for doc in await cursor.to_list(length=None)]

    async def get_aa_connection(self, user_id: str, connection_id: str) -> Optional[Dict[str, Any]]:
        from bson import ObjectId
        try:
            return await self.aa_connections.find_one(
                {"_id": ObjectId(connection_id), "user_id": user_id}
            )
        except Exception:
            return None

    async def get_aa_connection_by_consent(
        self, user_id: str, consent_id: str
    ) -> Optional[Dict[str, Any]]:
        return await self.aa_connections.find_one(
            {"user_id": user_id, "consent_id": consent_id}
        )

    async def update_aa_connection_status(self, connection_id: str, status: str) -> None:
        from bson import ObjectId
        await self.aa_connections.update_one(
            {"_id": ObjectId(connection_id)},
            {"$set": {"status": status, "updated_at": datetime.now(timezone.utc)}},
        )

    async def record_aa_sync(self, user_id: str, connection_id: str, count: int) -> None:
        from bson import ObjectId
        await self.aa_connections.update_one(
            {"_id": ObjectId(connection_id), "user_id": user_id},
            {
                "$set": {
                    "last_sync_at": datetime.now(timezone.utc),
                },
                "$inc": {"synced_count": count},
            },
        )

