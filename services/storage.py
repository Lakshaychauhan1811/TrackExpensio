import asyncio
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from motor.motor_asyncio import AsyncIOMotorClient


class MongoManager:
    """
    Centralized MongoDB helper used for storing user-specific assets such as
    speech transcripts, parsed documents, and OAuth tokens. Every document
    carries the `user_id` so isolation is guaranteed.

    The Motor client is bound to the asyncio event loop active when it was
    created. On some hosts (e.g. Render free tier) the worker's event loop
    can be recycled between requests, which leaves the old client pointing
    at a closed loop ("RuntimeError: Event loop is closed"). To avoid that,
    we lazily recreate the client whenever the running loop differs from
    the one the client was built on.
    """

    def __init__(self) -> None:
        self._uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self._db_name = os.getenv("MONGODB_DB", "ai_expense_tracker")
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connect()

    def _connect(self) -> None:
        self.client = AsyncIOMotorClient(self._uri)
        self.db = self.client[self._db_name]
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def _ensure_loop(self) -> None:
        """Recreate the Motor client if the event loop has changed or closed."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._loop is None or self._loop is not current_loop or self._loop.is_closed():
            self._connect()

    # ------------------------------------------------------------------
    # Collections are exposed as properties (not plain attributes) so that
    # ANY access — whether from MongoManager's own methods or from external
    # callers like api.py reaching in as `mongo_manager.google_collection`
    # — re-checks the event loop first. Previously only MongoManager's own
    # methods called _ensure_loop(), so direct external access could still
    # use a stale client bound to a closed loop ("RuntimeError: Event loop
    # is closed"), exactly as seen when api.py's get_user_profile() read
    # mongo_manager.google_collection directly.
    # ------------------------------------------------------------------
    @property
    def voice_collection(self):
        self._ensure_loop()
        return self.db["voice_expenses"]

    @property
    def doc_collection(self):
        self._ensure_loop()
        return self.db["document_expenses"]

    @property
    def google_collection(self):
        self._ensure_loop()
        return self.db["google_profiles"]

    @property
    def gmail_sync_collection(self):
        self._ensure_loop()
        return self.db["gmail_synced_messages"]

    @property
    def oauth_states_collection(self):
        self._ensure_loop()
        return self.db["oauth_states"]

    async def ensure_indexes(self) -> None:
        """Create indexes, handle connection errors gracefully"""
        self._ensure_loop()
        try:
            await self.voice_collection.create_index([("user_id", 1), ("created_at", -1)])
            await self.doc_collection.create_index([("user_id", 1), ("created_at", -1)])
            await self.google_collection.create_index("user_id", unique=True)
            await self.gmail_sync_collection.create_index(
                [("user_id", 1), ("message_id", 1)], unique=True
            )
            await self.db["expenses"].create_index(
                [("user_id", 1), ("metadata.gmail_message_id", 1)],
                sparse=True,
            )
            await self.oauth_states_collection.create_index("state", unique=True)
            await self.oauth_states_collection.create_index(
                "expires_at", expireAfterSeconds=0
            )
        except Exception as e:
            # Re-raise to be caught by caller
            raise ConnectionError(f"MongoDB connection failed: {str(e)}") from e

    async def save_voice_expense(self, user_id: str, payload: Dict[str, Any]) -> None:
        self._ensure_loop()
        await self.voice_collection.insert_one(
            {
                "user_id": user_id,
                "created_at": datetime.utcnow(),
                **payload,
            }
        )

    async def save_document_expense(self, user_id: str, payload: Dict[str, Any]) -> None:
        self._ensure_loop()
        await self.doc_collection.insert_one(
            {
                "user_id": user_id,
                "created_at": datetime.utcnow(),
                **payload,
            }
        )

    async def upsert_google_profile(
        self, user_id: str, profile: Dict[str, Any], credentials: Dict[str, Any]
    ) -> None:
        self._ensure_loop()
        await self.google_collection.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "user_id": user_id,
                    "profile": profile,
                    "credentials": credentials,
                    "updated_at": datetime.utcnow(),
                }
            },
            upsert=True,
        )

    async def save_oauth_state(self, state: str, payload: Dict[str, Any], ttl_minutes: int = 15) -> None:
        self._ensure_loop()
        now = datetime.utcnow()
        await self.oauth_states_collection.update_one(
            {"state": state},
            {
                "$set": {
                    "state": state,
                    "payload": payload,
                    "created_at": now,
                    "expires_at": now + timedelta(minutes=ttl_minutes),
                }
            },
            upsert=True,
        )

    async def pop_oauth_state(self, state: str) -> Optional[Dict[str, Any]]:
        self._ensure_loop()
        doc = await self.oauth_states_collection.find_one_and_delete({"state": state})
        if doc:
            return doc.get("payload")
        return None

    async def mark_gmail_message_synced(
        self, user_id: str, message_id: str, expense_id: str, payload: Dict[str, Any]
    ) -> None:
        self._ensure_loop()
        await self.gmail_sync_collection.update_one(
            {"user_id": user_id, "message_id": message_id},
            {
                "$set": {
                    "user_id": user_id,
                    "message_id": message_id,
                    "expense_id": expense_id,
                    "merchant": payload.get("merchant"),
                    "amount": payload.get("amount"),
                    "synced_at": datetime.utcnow(),
                }
            },
            upsert=True,
        )