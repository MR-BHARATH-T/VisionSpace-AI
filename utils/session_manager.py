"""Session Manager Module for VisionSpace AI.

Manages chat session history in a thread-safe manner, keeping track of conversation contexts.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger("VisionSpaceAI.SessionManager")


class SessionManager:
    """Manages chat session histories in memory for conversational context."""

    def __init__(self) -> None:
        """Initializes the SessionManager with a lock and session history storage."""
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        logger.info("SessionManager initialized.")

    def get_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves history for a given session ID in a thread-safe manner.

        Args:
            session_id: Unique string identifier for the chat session.

        Returns:
            List of message dictionaries representing session conversation history.
        """
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = []
            # Return a copy of the list to prevent external mutation side-effects
            return list(self._sessions[session_id])

    def add_message(self, session_id: str, role: str, content: Any, images: Optional[List[Any]] = None) -> None:
        """Appends a new message to the session history in a thread-safe manner.

        Args:
            session_id: Unique string identifier for the chat session.
            role: The author role of the message (e.g. 'user', 'assistant', 'system').
            content: The text content or structured list of the message.
            images: Optional list of images or URLs associated with the message.
        """
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = []
            msg = {"role": role, "content": content}
            if images is not None:
                msg["images"] = images
            self._sessions[session_id].append(msg)
            logger.debug("Added message for session_id '%s' (role: %s, with %d images)", session_id, role, len(images) if images else 0)

    def set_session_history(self, session_id: str, history: List[Dict[str, Any]]) -> None:
        """Overwrites or sets the history for a given session ID.

        Args:
            session_id: Unique string identifier for the chat session.
            history: List of message dictionaries representing conversation history.
        """
        with self._lock:
            self._sessions[session_id] = list(history)
            logger.debug("Set session history for session_id '%s' with %d messages", session_id, len(history))

    def clear_session(self, session_id: str) -> None:
        """Clears the history for a session ID.

        Args:
            session_id: Unique string identifier for the chat session.
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info("Cleared session history for session_id '%s'", session_id)
