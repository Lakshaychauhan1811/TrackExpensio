import os
import secrets
import hashlib
from datetime import datetime
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
from session_manager import SessionManager


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
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def _generate_api_key(self) -> str:
        """Generate a secure API key"""
        return secrets.token_urlsafe(32)

    async def create_user(self, username: str, password: str) -> dict:
        """Create a new user and return API key"""
        # Check if user exists
        existing = await self.users.find_one({"username": username})
        if existing:
            return {"status": "error", "message": "Username already exists"}

        # Create user
        user_id = str(await self.users.insert_one({
            "username": username,
            "password_hash": self._hash_password(password),
            "created_at": datetime.utcnow()
        }).inserted_id)

        # Generate API key
        api_key = self._generate_api_key()
        await self.api_keys.insert_one({
            "user_id": user_id,
            "api_key": api_key,
            "created_at": datetime.utcnow()
        })

        return {
            "status": "success",
            "user_id": user_id,
            "api_key": api_key,
            "message": "User created successfully"
        }

    async def authenticate_user(self, api_key: str = None, session_id: str = None, require_linked: bool = True) -> Optional[str]:
        """Authenticate user by API key or session_id, returns user_id if valid"""
        # Try session_id first
        if session_id:
            session_info = await self.session_manager.get_session_status(session_id)
            if session_info:
                if require_linked and session_info.get("is_guest", True):
                    return None
                return session_info.get("user_id")
        
        # Try API key (for registered users)
        if api_key:
            key_doc = await self.api_keys.find_one({"api_key": api_key})
            if key_doc:
                return key_doc.get("user_id")
        
        return None

    async def login_user(self, username: str, password: str) -> dict:
        """Login user and return API key"""
        user = await self.users.find_one({"username": username})
        if not user:
            return {"status": "error", "message": "Invalid username or password"}

        password_hash = self._hash_password(password)
        if user.get("password_hash") != password_hash:
            return {"status": "error", "message": "Invalid username or password"}

        user_id = str(user["_id"])
        # Get or create API key
        key_doc = await self.api_keys.find_one({"user_id": user_id})
        if key_doc:
            api_key = key_doc.get("api_key")
        else:
            api_key = self._generate_api_key()
            await self.api_keys.insert_one({
                "user_id": user_id,
                "api_key": api_key,
                "created_at": datetime.utcnow()
            })

        return {
            "status": "success",
            "user_id": user_id,
            "api_key": api_key,
            "message": "Login successful"
        }

