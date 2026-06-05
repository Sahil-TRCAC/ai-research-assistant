"""
services/session_service.py
────────────────────────────
Business logic for managing research sessions.
"""

import logging
from datetime import datetime, timezone

from models import db
from models.document import Document
from models.research_session import ResearchSession

logger = logging.getLogger("ai_research.session_service")


def create_session(document_id: str, title: str | None = None) -> ResearchSession:
    """Create a new research session linked to a document."""
    doc = Document.query.get(document_id)
    if not doc:
        raise ValueError(f"Document {document_id!r} not found.")

    session = ResearchSession(
        document_id=document_id,
        title=title or f"Session — {doc.title[:60]}",
        messages=[],
        message_count=0,
    )
    db.session.add(session)
    db.session.commit()
    logger.info("Session created: id=%s doc=%s", session.id, document_id)
    return session


def get_sessions_for_document(document_id: str) -> list[ResearchSession]:
    """Return all active sessions for a document."""
    return (
        ResearchSession.query
        .filter_by(document_id=document_id, status="active")
        .order_by(ResearchSession.created_at.desc())
        .all()
    )


def get_session_by_id(session_id: str) -> ResearchSession | None:
    return ResearchSession.query.get(session_id)


def archive_session(session_id: str) -> ResearchSession | None:
    """Mark a session as archived."""
    session = ResearchSession.query.get(session_id)
    if not session:
        return None
    session.status = "archived"
    session.updated_at = datetime.now(timezone.utc)
    db.session.commit()
    logger.info("Session archived: id=%s", session_id)
    return session


def delete_session(session_id: str) -> bool:
    """Hard-delete a session."""
    session = ResearchSession.query.get(session_id)
    if not session:
        return False
    db.session.delete(session)
    db.session.commit()
    logger.info("Session deleted: id=%s", session_id)
    return True


def add_message(session_id: str, role: str, content: str) -> ResearchSession | None:
    """
    Append a message to the session conversation history.

    Parameters
    ----------
    role    : "user" or "assistant"
    content : Message text
    """
    session = ResearchSession.query.get(session_id)
    if not session:
        return None

    now = datetime.now(timezone.utc)
    message = {
        "role": role,
        "content": content,
        "timestamp": now.isoformat(),
    }

    messages = list(session.messages or [])
    messages.append(message)
    session.messages = messages
    session.message_count = len(messages)
    session.last_activity_at = now
    session.updated_at = now

    db.session.commit()
    return session
