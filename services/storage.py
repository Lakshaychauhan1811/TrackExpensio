import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from motor.motor_asyncio import AsyncIOMotorClient


class MongoManager:
    """
    Centralized MongoDB helper used for storing user-specific assets such as
    speech transcripts, parsed documents, and OAuth tokens. Every document
    carries the `user_id` so isolation is guaranteed.
    """

    def __init__(self) -> None:
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGODB_DB", "ai_expense_tracker")
        self.client = AsyncIOMotorClient(uri)
        self.db = self.client[db_name]
        self.voice_collection = self.db["voice_expenses"]
        self.doc_collection = self.db["document_expenses"]
        self.google_collection = self.db["google_profiles"]
        self.gmail_sync_collection = self.db["gmail_synced_messages"]
        self.oauth_states_collection = self.db["oauth_states"]

    async def ensure_indexes(self) -> None:
        """Create indexes, handle connection errors gracefully"""
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
        await self.voice_collection.insert_one(
            {
                "user_id": user_id,
                "created_at": datetime.utcnow(),
                **payload,
            }
        )

    async def save_document_expense(self, user_id: str, payload: Dict[str, Any]) -> None:
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
        doc = await self.oauth_states_collection.find_one_and_delete({"state": state})
        if doc:
            return doc.get("payload")
        return None

    async def mark_gmail_message_synced(
        self, user_id: str, message_id: str, expense_id: str, payload: Dict[str, Any]
    ) -> None:
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

