"""
Bank sync via Plaid (official plaid-python SDK).

Set in .env:
  PLAID_CLIENT_ID=...
  PLAID_SECRET=...
  PLAID_ENV=sandbox   # or development / production
  PLAID_DAYS_REQUESTED=90  # optional, max 730
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import plaid
from dotenv import load_dotenv
from plaid.api import plaid_api
from plaid.exceptions import ApiException
from plaid.model.accounts_get_request import AccountsGetRequest
from plaid.model.country_code import CountryCode
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.model.link_token_transactions import LinkTokenTransactions
from plaid.model.products import Products
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions

load_dotenv()

# Plaid institution coverage — India (IN) is NOT supported as of 2026.
# https://plaid.com/docs/api/institutions/
PLAID_SUPPORTED_COUNTRY_CODES = (
    "US", "GB", "ES", "NL", "FR", "IE", "CA", "DE", "IT", "PL",
    "DK", "NO", "SE", "EE", "LT", "LV", "PT", "BE", "AT", "FI",
)


def _model_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return dict(obj)


class PlaidBankService:
    """Plaid Link / transactions sync using the official Plaid Python SDK."""

    def __init__(self):
        self._reload_config()

    def _reload_config(self) -> None:
        self.client_id = os.getenv("PLAID_CLIENT_ID", "").strip()
        self.secret = os.getenv("PLAID_SECRET", "").strip()
        env = os.getenv("PLAID_ENV", "sandbox").strip().lower()
        if env == "production":
            self.plaid_env = plaid.Environment.Production
        else:
            # sandbox and development both use Sandbox in plaid-python v39+
            self.plaid_env = plaid.Environment.Sandbox
        try:
            days = int(os.getenv("PLAID_DAYS_REQUESTED", "90"))
        except ValueError:
            days = 90
        self.days_requested = max(1, min(days, 730))
        raw_countries = os.getenv("PLAID_COUNTRY_CODES", "US")
        requested = [c.strip().upper() for c in raw_countries.split(",") if c.strip()]
        self.country_codes = [c for c in requested if c in PLAID_SUPPORTED_COUNTRY_CODES] or ["US"]

    def coverage_info(self) -> Dict[str, Any]:
        """Return Plaid regional coverage for the UI."""
        self._reload_config()
        return {
            "supported_country_codes": list(PLAID_SUPPORTED_COUNTRY_CODES),
            "active_country_codes": self.country_codes,
            "india_supported": False,
            "india_note": (
                "Plaid does not connect to Indian banks (HDFC, SBI, ICICI, etc.). "
                "Coverage is US, Canada, UK, and parts of Europe only."
            ),
        }

    def is_configured(self) -> bool:
        self._reload_config()
        return bool(self.client_id and self.secret)

    def _get_client(self) -> plaid_api.PlaidApi:
        self._reload_config()
        configuration = plaid.Configuration(
            host=self.plaid_env,
            api_key={
                "clientId": self.client_id,
                "secret": self.secret,
            },
        )
        return plaid_api.PlaidApi(plaid.ApiClient(configuration))

    @staticmethod
    def _plaid_error(exc: ApiException) -> str:
        try:
            body = json.loads(exc.body) if exc.body else {}
            code = body.get("error_code", "")
            msg = body.get("error_message") or body.get("display_message") or str(exc)
            return f"{code}: {msg}" if code else msg
        except Exception:
            return str(exc)

    @staticmethod
    def _normalize_category(txn: Dict[str, Any]) -> str:
        pfc = txn.get("personal_finance_category") or {}
        if isinstance(pfc, dict):
            primary = pfc.get("primary")
        else:
            primary = getattr(pfc, "primary", None)
        if primary:
            return str(primary).replace("_", " ").title()
        categories = txn.get("category") or []
        if categories:
            return str(categories[0])
        return "Other"

    @classmethod
    def _normalize_transaction(cls, txn: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Map Plaid txn to expense fields. Skips pending and credits (inflows)."""
        if txn.get("pending"):
            return None
        raw_amount = float(txn.get("amount", 0))
        if raw_amount <= 0:
            return None
        return {
            "external_id": txn.get("transaction_id"),
            "date": str(txn.get("date", "")),
            "amount": raw_amount,
            "merchant": txn.get("merchant_name") or txn.get("name", "Unknown"),
            "category": cls._normalize_category(txn),
            "note": txn.get("name", ""),
            "pending": False,
            "source": "plaid",
            "currency": txn.get("iso_currency_code") or "USD",
            "synced_at": datetime.now(timezone.utc).isoformat(),
        }

    def _create_link_token_sync(self, user_id: str) -> Dict[str, Any]:
        client = self._get_client()
        countries = [CountryCode(c) for c in self.country_codes]
        request = LinkTokenCreateRequest(
            products=[Products("transactions")],
            client_name="TrackExpensio",
            country_codes=countries,
            language="en",
            user=LinkTokenCreateRequestUser(client_user_id=str(user_id)),
            transactions=LinkTokenTransactions(days_requested=self.days_requested),
        )
        response = client.link_token_create(request)
        return {
            "status": "success",
            "link_token": response.link_token,
            "expiration": str(response.expiration),
        }

    def _exchange_public_token_sync(self, public_token: str) -> Dict[str, Any]:
        client = self._get_client()
        request = ItemPublicTokenExchangeRequest(public_token=public_token)
        response = client.item_public_token_exchange(request)
        return {
            "status": "success",
            "access_token": response.access_token,
            "item_id": response.item_id,
        }

    def _fetch_transactions_sync(
        self,
        access_token: str,
        start_date: str,
        end_date: str,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        client = self._get_client()
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        normalized: List[Dict[str, Any]] = []
        offset = 0
        page_size = 500
        total = None

        while total is None or offset < total:
            options = TransactionsGetRequestOptions(count=page_size, offset=offset)
            request = TransactionsGetRequest(
                access_token=access_token,
                start_date=start,
                end_date=end,
                options=options,
            )
            try:
                response = client.transactions_get(request)
            except ApiException as exc:
                err = self._plaid_error(exc)
                if "PRODUCT_NOT_READY" in err:
                    return [], (
                        "Transactions not ready yet — wait a minute after linking, then sync again"
                    )
                return normalized, err

            total = int(response.total_transactions or 0)
            txns = response.transactions or []
            for txn in txns:
                mapped = self._normalize_transaction(_model_to_dict(txn))
                if mapped and mapped.get("external_id"):
                    normalized.append(mapped)
            offset += page_size
            if not txns:
                break

        return normalized, None

    def _get_account_balances_sync(self, access_token: str) -> List[Dict[str, Any]]:
        client = self._get_client()
        request = AccountsGetRequest(access_token=access_token)
        response = client.accounts_get(request)
        accounts = []
        for account in response.accounts or []:
            acct = _model_to_dict(account)
            balances = acct.get("balances") or {}
            accounts.append(
                {
                    "name": acct.get("name", "Account"),
                    "type": str(acct.get("type", "")),
                    "subtype": str(acct.get("subtype", "")),
                    "balance_current": balances.get("current"),
                    "balance_available": balances.get("available"),
                    "currency": balances.get("iso_currency_code") or "USD",
                    "account_id": acct.get("account_id"),
                    "mask": acct.get("mask"),
                }
            )
        return accounts

    async def create_link_token(self, user_id: str) -> Dict[str, Any]:
        if not self.is_configured():
            return {
                "status": "error",
                "message": "Plaid not configured. Set PLAID_CLIENT_ID and PLAID_SECRET in .env",
            }
        try:
            return await asyncio.to_thread(self._create_link_token_sync, user_id)
        except ApiException as exc:
            return {"status": "error", "message": self._plaid_error(exc)}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    async def exchange_public_token(self, public_token: str) -> Dict[str, Any]:
        if not self.is_configured():
            return {"status": "error", "message": "Plaid not configured"}
        if not public_token:
            return {"status": "error", "message": "public_token is required"}
        try:
            return await asyncio.to_thread(self._exchange_public_token_sync, public_token)
        except ApiException as exc:
            return {"status": "error", "message": self._plaid_error(exc)}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    async def fetch_transactions(
        self,
        access_token: str,
        start_date: str,
        end_date: str,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        if not self.is_configured():
            return [], "Plaid not configured"
        if not access_token:
            return [], "Missing Plaid access token"
        try:
            return await asyncio.to_thread(
                self._fetch_transactions_sync, access_token, start_date, end_date
            )
        except ApiException as exc:
            return [], self._plaid_error(exc)
        except Exception as exc:
            return [], str(exc)

    async def get_account_balances(self, access_token: str) -> Dict[str, Any]:
        if not self.is_configured():
            return {"status": "error", "message": "Plaid not configured"}
        if not access_token:
            return {"status": "error", "message": "Missing Plaid access token"}
        try:
            accounts = await asyncio.to_thread(self._get_account_balances_sync, access_token)
            return {"status": "success", "accounts": accounts}
        except ApiException as exc:
            return {"status": "error", "message": self._plaid_error(exc)}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}


plaid_service = PlaidBankService()
