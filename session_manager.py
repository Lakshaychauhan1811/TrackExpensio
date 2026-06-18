"""Guest session management for public access"""
import asyncio
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient
import os


class SessionManager:
    """Manages guest sessions for public access.

    Self-heals against Render's event-loop recycling the same way
    storage.py / database.py do — see those files for the full rationale.
    """

    def __init__(self):
        self._uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self._db_name = os.getenv("MONGODB_DB", "ai_expense_tracker")
        self._loop = None
        self._connect()

    def _connect(self) -> None:
        self.client = AsyncIOMotorClient(self._uri)
        self.db = self.client[self._db_name]
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def _ensure_loop(self) -> None:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._loop is None or self._loop is not current_loop or self._loop.is_closed():
            self._connect()

    @property
    def sessions(self):
        self._ensure_loop()
        return self.db["guest_sessions"]

    @property
    def session_users(self):
        self._ensure_loop()
        return self.db["session_users"]  # Maps session_id to user_id
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        return str(uuid.uuid4())
    
    async def create_guest_session(self) -> Dict[str, str]:
        """Create a new guest session and return session_id and user_id"""
        session_id = self._generate_session_id()
        user_id = f"guest_{secrets.token_urlsafe(16)}"
        
        # Store session
        await self.sessions.insert_one({
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(days=365),  # Long-lived
            "is_guest": True
        })
        
        # Map session to user
        await self.session_users.insert_one({
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.utcnow()
        })
        
        return {
            "session_id": session_id,
            "user_id": user_id,
            "status": "success"
        }
    
    async def _get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        session = await self.sessions.find_one({"session_id": session_id})
        if not session:
            return None
        if session.get("expires_at") and session["expires_at"] < datetime.utcnow():
            return None
        return session
    
    async def get_user_from_session(self, session_id: str) -> Optional[str]:
        """Get user_id from session_id"""
        session = await self._get_session(session_id)
        if not session:
            return None
        return session.get("user_id")
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        session = await self._get_session(session_id)
        if not session:
            return None
        return {
            "session_id": session_id,
            "user_id": session.get("user_id"),
            "is_guest": session.get("is_guest", True),
            "linked_at": session.get("linked_at"),
        }
    
    async def link_session_to_google(self, session_id: str, google_user_id: str) -> bool:
        """Link a guest session to a Google account for persistence"""
        session = await self.sessions.find_one({"session_id": session_id})
        if not session:
            return False
        
        # Update session to use Google user_id
        await self.sessions.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "user_id": google_user_id,
                    "is_guest": False,
                    "linked_at": datetime.utcnow()
                }
            }
        )
        
        # Update mapping
        await self.session_users.update_one(
            {"session_id": session_id},
            {"$set": {"user_id": google_user_id}}
        )
        
        return True

    async def link_session_to_user(self, session_id: str, user_id: str) -> bool:
        """Link a guest session to any verified user_id (e.g., email OTP)."""
        session = await self.sessions.find_one({"session_id": session_id})
        if not session:
            return False

        await self.sessions.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "user_id": user_id,
                    "is_guest": False,
                    "linked_at": datetime.utcnow(),
                }
            },
        )
        await self.session_users.update_one(
            {"session_id": session_id},
            {"$set": {"user_id": user_id}},
            upsert=True,
        )
        return True