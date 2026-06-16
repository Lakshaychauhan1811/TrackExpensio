"""
Audit Logging System for TrackExpensio

Enterprise-style activity trail: severity, categories, change tracking,
security alerts, analytics dashboard, and export.
"""

import csv
import io
import os
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from motor.motor_asyncio import AsyncIOMotorClient


class AuditAction(str, Enum):
    """Types of actions that can be audited"""
    EXPENSE_ADDED = "expense_added"
    EXPENSE_UPDATED = "expense_updated"
    EXPENSE_DELETED = "expense_deleted"
    EXPENSE_VIEWED = "expense_viewed"
    INCOME_ADDED = "income_added"
    INCOME_UPDATED = "income_updated"
    INCOME_DELETED = "income_deleted"
    BUDGET_SET = "budget_set"
    BUDGET_VIEWED = "budget_viewed"
    REPORT_GENERATED = "report_generated"
    TAX_ESTIMATED = "tax_estimated"
    LOGIN = "login"
    LOGOUT = "logout"
    PASSWORD_CHANGED = "password_changed"
    PROFILE_UPDATED = "profile_updated"
    DATA_EXPORTED = "data_exported"
    DATA_DELETED = "data_deleted"
    ACCESS_SHARED = "access_shared"
    SETTINGS_CHANGED = "settings_changed"
    CURRENCY_CHANGED = "currency_changed"
    FAILED_LOGIN = "failed_login"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    API_KEY_GENERATED = "api_key_generated"
    API_KEY_REVOKED = "api_key_revoked"
    CHAT_INTERACTION = "chat_interaction"
    MCP_TOOL_CALLED = "mcp_tool_called"
    DOCUMENT_UPLOADED = "document_uploaded"
    SESSION_LINKED = "session_linked"
    VOICE_COMMAND = "voice_command"
    RATE_LIMIT_HIT = "rate_limit_hit"
    IP_BLOCKED = "ip_blocked"


class AuditSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AuditCategory(str, Enum):
    FINANCIAL = "financial"
    SECURITY = "security"
    AUTH = "auth"
    SYSTEM = "system"
    AI = "ai"


# action -> (label, category, default severity)
ACTION_META: Dict[str, Tuple[str, AuditCategory, AuditSeverity]] = {
    AuditAction.EXPENSE_ADDED.value: ("Expense added", AuditCategory.FINANCIAL, AuditSeverity.INFO),
    AuditAction.EXPENSE_UPDATED.value: ("Expense updated", AuditCategory.FINANCIAL, AuditSeverity.INFO),
    AuditAction.EXPENSE_DELETED.value: ("Expense deleted", AuditCategory.FINANCIAL, AuditSeverity.WARNING),
    AuditAction.EXPENSE_VIEWED.value: ("Data viewed", AuditCategory.FINANCIAL, AuditSeverity.INFO),
    AuditAction.INCOME_ADDED.value: ("Income added", AuditCategory.FINANCIAL, AuditSeverity.INFO),
    AuditAction.INCOME_UPDATED.value: ("Income updated", AuditCategory.FINANCIAL, AuditSeverity.INFO),
    AuditAction.INCOME_DELETED.value: ("Income deleted", AuditCategory.FINANCIAL, AuditSeverity.WARNING),
    AuditAction.BUDGET_SET.value: ("Budget set", AuditCategory.FINANCIAL, AuditSeverity.INFO),
    AuditAction.BUDGET_VIEWED.value: ("Budget viewed", AuditCategory.FINANCIAL, AuditSeverity.INFO),
    AuditAction.REPORT_GENERATED.value: ("Report generated", AuditCategory.FINANCIAL, AuditSeverity.INFO),
    AuditAction.TAX_ESTIMATED.value: ("Tax estimated", AuditCategory.FINANCIAL, AuditSeverity.INFO),
    AuditAction.LOGIN.value: ("Login", AuditCategory.AUTH, AuditSeverity.INFO),
    AuditAction.LOGOUT.value: ("Logout", AuditCategory.AUTH, AuditSeverity.INFO),
    AuditAction.FAILED_LOGIN.value: ("Failed login", AuditCategory.SECURITY, AuditSeverity.WARNING),
    AuditAction.PASSWORD_CHANGED.value: ("Password changed", AuditCategory.SECURITY, AuditSeverity.WARNING),
    AuditAction.API_KEY_GENERATED.value: ("API key created", AuditCategory.SECURITY, AuditSeverity.WARNING),
    AuditAction.API_KEY_REVOKED.value: ("API key revoked", AuditCategory.SECURITY, AuditSeverity.WARNING),
    AuditAction.SESSION_LINKED.value: ("Google account linked", AuditCategory.AUTH, AuditSeverity.INFO),
    AuditAction.CHAT_INTERACTION.value: ("AI chat", AuditCategory.AI, AuditSeverity.INFO),
    AuditAction.VOICE_COMMAND.value: ("Voice command", AuditCategory.AI, AuditSeverity.INFO),
    AuditAction.MCP_TOOL_CALLED.value: ("Tool invoked", AuditCategory.SYSTEM, AuditSeverity.INFO),
    AuditAction.DOCUMENT_UPLOADED.value: ("Document uploaded", AuditCategory.FINANCIAL, AuditSeverity.INFO),
    AuditAction.DATA_EXPORTED.value: ("Data exported", AuditCategory.SECURITY, AuditSeverity.WARNING),
    AuditAction.SUSPICIOUS_ACTIVITY.value: ("Suspicious activity", AuditCategory.SECURITY, AuditSeverity.CRITICAL),
    AuditAction.RATE_LIMIT_HIT.value: ("Rate limit exceeded", AuditCategory.SECURITY, AuditSeverity.WARNING),
    AuditAction.IP_BLOCKED.value: ("IP blocked", AuditCategory.SECURITY, AuditSeverity.CRITICAL),
}

AUDIT_ACTION_LABELS: Dict[str, str] = {k: v[0] for k, v in ACTION_META.items()}

# Tools that emit detailed audit entries in main.py — API layer skips duplicate success logs
DETAILED_AUDIT_TOOLS = frozenset({
    "add_expense", "delete_expense", "update_expense", "add_income", "set_budget",
    "estimate_taxes", "login_user", "register_user", "get_audit_logs",
    "get_audit_dashboard", "export_audit_logs",
})


def _meta_for_action(action: AuditAction, success: bool) -> Tuple[str, str, str]:
    label, category, severity = ACTION_META.get(
        action.value,
        (action.value.replace("_", " ").title(), AuditCategory.SYSTEM, AuditSeverity.INFO),
    )
    if not success and action != AuditAction.FAILED_LOGIN:
        severity = AuditSeverity.WARNING
    if action in (AuditAction.FAILED_LOGIN, AuditAction.SUSPICIOUS_ACTIVITY, AuditAction.IP_BLOCKED):
        severity = AuditSeverity.CRITICAL if not success else severity
    return label, category.value, severity.value


class AuditLogger:
    """Centralized audit logging system"""

    def __init__(self):
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGODB_DB", "ai_expense_tracker")
        self.client = AsyncIOMotorClient(uri)
        self.db = self.client[db_name]
        self.audit_logs = self.db["audit_logs"]

    async def _ensure_indexes(self):
        await self.audit_logs.create_index([("user_id", 1), ("timestamp", -1)])
        await self.audit_logs.create_index([("action", 1), ("timestamp", -1)])
        await self.audit_logs.create_index([("user_id", 1), ("success", 1), ("timestamp", -1)])
        await self.audit_logs.create_index([("user_id", 1), ("category", 1), ("timestamp", -1)])
        await self.audit_logs.create_index([("user_id", 1), ("severity", 1), ("timestamp", -1)])
        await self.audit_logs.create_index("request_id")
        await self.audit_logs.create_index("timestamp", expireAfterSeconds=31536000)

    async def log(
        self,
        user_id: Optional[str],
        action: AuditAction,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        source: str = "system",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        request_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        changes: Optional[Dict[str, Any]] = None,
        severity: Optional[AuditSeverity] = None,
        category: Optional[AuditCategory] = None,
    ) -> str:
        label, default_category, default_severity = _meta_for_action(action, success)
        log_entry: Dict[str, Any] = {
            "user_id": user_id,
            "action": action.value,
            "action_label": label,
            "category": (category or AuditCategory(default_category)).value
            if isinstance(category, AuditCategory)
            else (category or default_category),
            "severity": (severity or AuditSeverity(default_severity)).value
            if isinstance(severity, AuditSeverity)
            else (severity or default_severity),
            "timestamp": datetime.now(timezone.utc),
            "success": success,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "source": source,
            "request_id": request_id or str(uuid.uuid4()),
        }
        if resource_type:
            log_entry["resource_type"] = resource_type
        if resource_id:
            log_entry["resource_id"] = resource_id
        if session_id:
            log_entry["session_id"] = session_id
        if tool_name:
            log_entry["tool_name"] = tool_name
        if duration_ms is not None:
            log_entry["duration_ms"] = round(duration_ms, 2)
        if details:
            log_entry["details"] = details
        if changes:
            log_entry["changes"] = changes
        if error_message:
            log_entry["error_message"] = error_message

        result = await self.audit_logs.insert_one(log_entry)
        return str(result.inserted_id)

    def _serialize_log(self, log: Dict[str, Any]) -> Dict[str, Any]:
        action = log["action"]
        label, cat, sev = ACTION_META.get(
            action,
            (action.replace("_", " ").title(), AuditCategory.SYSTEM, AuditSeverity.INFO),
        )
        out: Dict[str, Any] = {
            "id": str(log["_id"]),
            "user_id": log.get("user_id"),
            "action": action,
            "action_label": log.get("action_label", label),
            "category": log.get("category", cat.value if isinstance(cat, AuditCategory) else cat),
            "severity": log.get("severity", sev.value if isinstance(sev, AuditSeverity) else sev),
            "timestamp": log["timestamp"].isoformat()
            if hasattr(log["timestamp"], "isoformat")
            else str(log["timestamp"]),
            "success": log.get("success", True),
            "ip_address": log.get("ip_address"),
            "user_agent": log.get("user_agent"),
            "source": log.get("source", "system"),
            "resource_type": log.get("resource_type"),
            "resource_id": log.get("resource_id"),
            "session_id": log.get("session_id"),
            "tool_name": log.get("tool_name"),
            "request_id": log.get("request_id"),
            "duration_ms": log.get("duration_ms"),
            "details": log.get("details", {}),
            "changes": log.get("changes"),
        }
        if "error_message" in log:
            out["error_message"] = log["error_message"]
        return out

    def _build_query(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        action: Optional[AuditAction] = None,
        success: Optional[bool] = None,
        source: Optional[str] = None,
        category: Optional[str] = None,
        severity: Optional[str] = None,
        resource_type: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        query: Dict[str, Any] = {"user_id": user_id}
        if start_date or end_date:
            query["timestamp"] = {}
            if start_date:
                query["timestamp"]["$gte"] = start_date
            if end_date:
                end_exclusive = end_date + timedelta(days=1) if end_date.hour == 0 else end_date
                query["timestamp"]["$lt"] = end_exclusive
        if action:
            query["action"] = action.value
        if success is not None:
            query["success"] = success
        if source:
            query["source"] = source
        if category:
            query["category"] = category
        if severity:
            query["severity"] = severity
        if resource_type:
            query["resource_type"] = resource_type
        if search and search.strip():
            term = search.strip()
            query["$or"] = [
                {"action_label": {"$regex": term, "$options": "i"}},
                {"action": {"$regex": term, "$options": "i"}},
                {"tool_name": {"$regex": term, "$options": "i"}},
                {"error_message": {"$regex": term, "$options": "i"}},
                {"resource_id": {"$regex": term, "$options": "i"}},
                {"ip_address": {"$regex": term, "$options": "i"}},
            ]
        return query

    async def search_logs(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        action: Optional[AuditAction] = None,
        success: Optional[bool] = None,
        source: Optional[str] = None,
        category: Optional[str] = None,
        severity: Optional[str] = None,
        resource_type: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        query = self._build_query(
            user_id, start_date, end_date, action, success, source,
            category, severity, resource_type, search,
        )
        total = await self.audit_logs.count_documents(query)
        cursor = (
            self.audit_logs.find(query)
            .sort("timestamp", -1)
            .skip(max(0, offset))
            .limit(min(limit, 500))
        )
        logs = await cursor.to_list(length=None)
        return {
            "logs": [self._serialize_log(log) for log in logs],
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": offset + len(logs) < total,
        }

    async def get_user_logs(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        action: Optional[AuditAction] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        result = await self.search_logs(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            action=action,
            limit=limit,
            offset=0,
        )
        return result["logs"]

    async def get_dashboard(
        self,
        user_id: str,
        hours: int = 168,
    ) -> Dict[str, Any]:
        """Analytics dashboard: counts, trends, security signals."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        match = {"user_id": user_id, "timestamp": {"$gte": cutoff}}

        total = await self.audit_logs.count_documents(match)
        failures = await self.audit_logs.count_documents({**match, "success": False})
        critical = await self.audit_logs.count_documents({
            **match, "severity": AuditSeverity.CRITICAL.value,
        })

        by_action = await self.audit_logs.aggregate([
            {"$match": match},
            {"$group": {"_id": "$action", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 12},
        ]).to_list(length=None)

        by_category = await self.audit_logs.aggregate([
            {"$match": match},
            {"$group": {"_id": "$category", "count": {"$sum": 1}}},
        ]).to_list(length=None)

        by_source = await self.audit_logs.aggregate([
            {"$match": match},
            {"$group": {"_id": "$source", "count": {"$sum": 1}}},
        ]).to_list(length=None)

        by_day = await self.audit_logs.aggregate([
            {"$match": match},
            {
                "$group": {
                    "_id": {
                        "$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"},
                    },
                    "count": {"$sum": 1},
                    "failures": {
                        "$sum": {"$cond": [{"$eq": ["$success", False]}, 1, 0]},
                    },
                },
            },
            {"$sort": {"_id": 1}},
        ]).to_list(length=None)

        recent_failures = await self.audit_logs.find(
            {**match, "success": False},
        ).sort("timestamp", -1).limit(5).to_list(length=None)

        unique_ips = await self.audit_logs.distinct("ip_address", {
            **match, "ip_address": {"$nin": [None, ""]},
        })

        alerts = await self.get_security_alerts(user_id, hours=min(hours, 72))

        success_rate = round(((total - failures) / total) * 100, 1) if total else 100.0

        return {
            "period_hours": hours,
            "total_events": total,
            "failed_events": failures,
            "critical_events": critical,
            "success_rate_pct": success_rate,
            "unique_ips": len(unique_ips),
            "by_action": {r["_id"]: r["count"] for r in by_action if r["_id"]},
            "by_category": {r["_id"]: r["count"] for r in by_category if r["_id"]},
            "by_source": {r["_id"]: r["count"] for r in by_source if r["_id"]},
            "timeline": [
                {"date": r["_id"], "count": r["count"], "failures": r.get("failures", 0)}
                for r in by_day
            ],
            "recent_failures": [self._serialize_log(log) for log in recent_failures],
            "security_alerts": alerts,
        }

    async def get_security_alerts(
        self,
        user_id: str,
        hours: int = 24,
    ) -> List[Dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        alerts: List[Dict[str, Any]] = []

        failed_logins = await self.audit_logs.count_documents({
            "user_id": user_id,
            "action": AuditAction.FAILED_LOGIN.value,
            "timestamp": {"$gte": cutoff},
        })
        if failed_logins >= 3:
            alerts.append({
                "level": "critical",
                "code": "multiple_failed_logins",
                "message": f"{failed_logins} failed login attempts in the last {hours}h",
                "count": failed_logins,
            })

        failed_ops = await self.audit_logs.count_documents({
            "user_id": user_id,
            "success": False,
            "timestamp": {"$gte": cutoff},
            "action": {"$ne": AuditAction.FAILED_LOGIN.value},
        })
        if failed_ops >= 5:
            alerts.append({
                "level": "warning",
                "code": "high_failure_rate",
                "message": f"{failed_ops} failed operations in the last {hours}h",
                "count": failed_ops,
            })

        burst = await self.audit_logs.count_documents({
            "user_id": user_id,
            "timestamp": {"$gte": datetime.now(timezone.utc) - timedelta(minutes=10)},
        })
        if burst >= 50:
            alerts.append({
                "level": "warning",
                "code": "activity_burst",
                "message": f"{burst} events in the last 10 minutes",
                "count": burst,
            })

        return alerts

    async def export_logs_csv(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        action: Optional[AuditAction] = None,
        limit: int = 5000,
    ) -> str:
        result = await self.search_logs(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            action=action,
            limit=limit,
            offset=0,
        )
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "timestamp", "action", "action_label", "category", "severity",
            "success", "source", "tool_name", "resource_type", "resource_id",
            "ip_address", "duration_ms", "error_message", "details",
        ])
        for row in result["logs"]:
            writer.writerow([
                row.get("timestamp"),
                row.get("action"),
                row.get("action_label"),
                row.get("category"),
                row.get("severity"),
                row.get("success"),
                row.get("source"),
                row.get("tool_name", ""),
                row.get("resource_type", ""),
                row.get("resource_id", ""),
                row.get("ip_address", ""),
                row.get("duration_ms", ""),
                row.get("error_message", ""),
                str(row.get("details", {})),
            ])
        return output.getvalue()

    async def get_failed_logins(
        self,
        user_id: Optional[str] = None,
        hours: int = 24,
    ) -> List[Dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        query: Dict[str, Any] = {
            "action": AuditAction.FAILED_LOGIN.value,
            "timestamp": {"$gte": cutoff},
        }
        if user_id:
            query["user_id"] = user_id
        cursor = self.audit_logs.find(query).sort("timestamp", -1)
        logs = await cursor.to_list(length=None)
        return [self._serialize_log(log) for log in logs]

    async def clear_old_logs(self, days: int = 365) -> int:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        result = await self.audit_logs.delete_many({"timestamp": {"$lt": cutoff_date}})
        return result.deleted_count

    async def get_action_counts(
        self,
        user_id: str,
        hours: int = 24,
    ) -> Dict[str, int]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        pipeline = [
            {"$match": {"user_id": user_id, "timestamp": {"$gte": cutoff}}},
            {"$group": {"_id": "$action", "count": {"$sum": 1}}},
        ]
        rows = await self.audit_logs.aggregate(pipeline).to_list(length=None)
        return {row["_id"]: row["count"] for row in rows}


def parse_audit_dates(start_date: Optional[str], end_date: Optional[str]):
    """Parse YYYY-MM-DD strings to UTC datetimes."""
    start_dt = end_dt = None
    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return start_dt, end_dt


audit_logger = AuditLogger()
