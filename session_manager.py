```python
"""
TrackExpensio - Guest and Google Session Management
MongoDB-compatible and Vercel serverless-safe.
"""

import asyncio
import os
import secrets
import uuid
import logging

from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)


class SessionManager:
    """Manage guest sessions and verified Google account linking."""

    def __init__(self):
        self._uri = os.getenv("MONGODB_URI")
        self._db_name = os.getenv(
            "MONGODB_DB",
            "ai_expense_tracker"
        )

        if not self._uri:
            raise RuntimeError("MONGODB_URI is not configured.")

        self.client = None
        self.db = None
        self._loop = None

        self._connect()

    def _connect(self):
        """Create or recreate MongoDB connection."""

        self.client = AsyncIOMotorClient(
            self._uri,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
        )

        self.db = self.client[self._db_name]

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def _ensure_loop(self):
        """Reconnect if the active event loop has changed."""

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        if (
            self._loop is None
            or self._loop is not current_loop
            or self._loop.is_closed()
        ):
            self._connect()

    @property
    def sessions(self):
        self._ensure_loop()
        return self.db["guest_sessions"]

    @property
    def session_users(self):
        self._ensure_loop()
        return self.db["session_users"]

    @staticmethod
    def _generate_session_id() -> str:
        return str(uuid.uuid4())

    async def create_guest_session(self) -> Dict[str, str]:
        """Create a new guest session."""

        session_id = self._generate_session_id()
        user_id = f"guest_{secrets.token_urlsafe(16)}"

        now = datetime.utcnow()

        session_document = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": now,
            "expires_at": now + timedelta(days=365),
            "is_guest": True,
            "linked_at": None,
        }

        await self.sessions.insert_one(session_document)

        await self.session_users.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "user_id": user_id,
                    "created_at": now,
                }
            },
            upsert=True,
        )

        return {
            "session_id": session_id,
            "user_id": user_id,
            "status": "success",
        }

    async def _get_session(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a valid session."""

        if not session_id:
            return None

        session = await self.sessions.find_one(
            {"session_id": session_id}
        )

        if not session:
            return None

        expires_at = session.get("expires_at")

        if expires_at and expires_at < datetime.utcnow():
            return None

        return session

    async def get_user_from_session(
        self,
        session_id: str
    ) -> Optional[str]:
        """Get the user ID associated with a session."""

        session = await self._get_session(session_id)

        if not session:
            return None

        return session.get("user_id")

    async def get_session_status(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return session status."""

        session = await self._get_session(session_id)

        if not session:
            return None

        return {
            "session_id": session_id,
            "user_id": session.get("user_id"),
            "is_guest": session.get("is_guest", True),
            "linked_at": session.get("linked_at"),
        }

    async def link_session_to_google(
        self,
        session_id: str,
        google_user_id: str
    ) -> bool:
        """
        Link an existing guest session to a Google account.

        The previous guest user ID is preserved as
        original_user_id so existing data can be migrated safely.
        """

        if not session_id or not google_user_id:
            return False

        session = await self._get_session(session_id)

        if not session:
            logger.warning(
                "Google linking failed: session not found."
            )
            return False

        now = datetime.utcnow()

        previous_user_id = session.get("user_id")

        # Preserve the original guest ID for future data migration.
        await self.sessions.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "user_id": google_user_id,
                    "is_guest": False,
                    "linked_at": now,
                    "google_user_id": google_user_id,
                },
                "$setOnInsert": {
                    "created_at": now,
                },
            },
        )

        # Keep the session mapping synchronized.
        await self.session_users.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "user_id": google_user_id,
                    "linked_at": now,
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "created_at": now,
                },
            },
            upsert=True,
        )

        # Preserve guest ID without overwriting an earlier original ID.
        if previous_user_id and previous_user_id != google_user_id:
            await self.sessions.update_one(
                {"session_id": session_id},
                {
                    "$setOnInsert": {
                        "original_user_id": previous_user_id,
                    }
                },
            )

        logger.info(
            "Google account linked to session successfully."
        )

        return True

    async def link_session_to_user(
        self,
        session_id: str,
        user_id: str
    ) -> bool:
        """Link a session to another verified user."""

        if not session_id or not user_id:
            return False

        session = await self._get_session(session_id)

        if not session:
            return False

        now = datetime.utcnow()
        previous_user_id = session.get("user_id")

        await self.sessions.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "user_id": user_id,
                    "is_guest": False,
                    "linked_at": now,
                },
                "$setOnInsert": {
                    "original_user_id": previous_user_id,
                },
            },
        )

        await self.session_users.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "user_id": user_id,
                    "linked_at": now,
                    "updated_at": now,
                }
            },
            upsert=True,
        )

        return True

    async def close(self):
        """Close MongoDB client when appropriate."""

        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self._loop = None
```
