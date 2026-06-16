import hashlib
import os
import secrets
from datetime import datetime
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient
from session_manager import SessionManager
from security import security_manager


class UserAuth:
    """User authentication and API key management"""

    def __init__(self):
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGODB_DB", "ai_expense_tracker")
        self.client = AsyncIOMotorClient(uri)
        self.db = self.client[db_name]
        self.users = self.db["users"]
        self.api_keys = self.db["api_keys"]
        self.session_manager = SessionManager()

    def _hash_password(self, password: str) -> str:
        """Hash password with bcrypt (via security_manager)."""
        return security_manager.hash_password(password)

    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password; supports bcrypt and legacy SHA256 hex hashes."""
        if not stored_hash:
            return False
        # Legacy SHA256 (64 hex chars, no bcrypt prefix)
        if len(stored_hash) == 64 and all(c in "0123456789abcdef" for c in stored_hash.lower()):
            return hashlib.sha256(password.encode()).hexdigest() == stored_hash
        return security_manager.verify_password(password, stored_hash)

    async def _upgrade_password_hash(self, user_id: str, password: str) -> None:
        """Re-hash legacy SHA256 passwords to bcrypt on successful login."""
        new_hash = self._hash_password(password)
        from bson import ObjectId
        await self.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"password_hash": new_hash, "password_upgraded_at": datetime.utcnow()}},
        )

    def _generate_api_key(self) -> str:
        return secrets.token_urlsafe(32)

    async def create_user(self, username: str, password: str) -> dict:
        existing = await self.users.find_one({"username": username})
        if existing:
            return {"status": "error", "message": "Username already exists"}

        user_id = str(
            (
                await self.users.insert_one(
                    {
                        "username": username,
                        "password_hash": self._hash_password(password),
                        "created_at": datetime.utcnow(),
                    }
                )
            ).inserted_id
        )

        api_key = self._generate_api_key()
        await self.api_keys.insert_one(
            {
                "user_id": user_id,
                "api_key": api_key,
                "created_at": datetime.utcnow(),
            }
        )

        return {
            "status": "success",
            "user_id": user_id,
            "api_key": api_key,
            "message": "User created successfully",
        }

    async def authenticate_user(
        self,
        api_key: str = None,
        session_id: str = None,
        require_linked: bool = True,
    ) -> Optional[str]:
        if session_id:
            session_info = await self.session_manager.get_session_status(session_id)
            if session_info:
                if require_linked and session_info.get("is_guest", True):
                    return None
                return session_info.get("user_id")

        if api_key:
            key_doc = await self.api_keys.find_one({"api_key": api_key})
            if key_doc:
                return key_doc.get("user_id")

        return None

    async def login_user(self, username: str, password: str) -> dict:
        user = await self.users.find_one({"username": username})
        if not user:
            return {"status": "error", "message": "Invalid username or password"}

        stored = user.get("password_hash", "")
        if not self._verify_password(password, stored):
            return {"status": "error", "message": "Invalid username or password"}

        user_id = str(user["_id"])

        # Upgrade legacy SHA256 → bcrypt
        if len(stored) == 64 and all(c in "0123456789abcdef" for c in stored.lower()):
            await self._upgrade_password_hash(user_id, password)

        key_doc = await self.api_keys.find_one({"user_id": user_id})
        if key_doc:
            api_key = key_doc.get("api_key")
        else:
            api_key = self._generate_api_key()
            await self.api_keys.insert_one(
                {
                    "user_id": user_id,
                    "api_key": api_key,
                    "created_at": datetime.utcnow(),
                }
            )

        return {
            "status": "success",
            "user_id": user_id,
            "api_key": api_key,
            "message": "Login successful",
        }
