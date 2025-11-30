import os
from datetime import datetime
from typing import Any, Dict

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

    async def ensure_indexes(self) -> None:
        """Create indexes, handle connection errors gracefully"""
        try:
            await self.voice_collection.create_index([("user_id", 1), ("created_at", -1)])
            await self.doc_collection.create_index([("user_id", 1), ("created_at", -1)])
            await self.google_collection.create_index("user_id", unique=True)
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

