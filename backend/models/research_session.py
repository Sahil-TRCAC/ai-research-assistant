"""
models/research_session.py
──────────────────────────
Tracks individual research/Q&A sessions tied to a document.
Each session stores a conversation history as JSON.
"""

import uuid
from datetime import datetime, timezone
from models import db


class ResearchSession(db.Model):
    __tablename__ = "research_sessions"

    # ── Primary key ──────────────────────────────────────────────────────────
    id = db.Column(
        db.String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # ── Foreign key ──────────────────────────────────────────────────────────
    document_id = db.Column(
        db.String(36),
        db.ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ── Session metadata ─────────────────────────────────────────────────────
    title = db.Column(db.String(500), nullable=False, default="New Session")
    status = db.Column(
        db.String(30),
        nullable=False,
        default="active",
        index=True,
    )
    # Possible statuses: active | archived | deleted

    # ── Conversation history ──────────────────────────────────────────────────
    # Stored as a list of {"role": "user"|"assistant", "content": "...", "timestamp": "..."}
    messages = db.Column(db.JSON, nullable=False, default=list)

    # ── Stats ────────────────────────────────────────────────────────────────
    message_count = db.Column(db.Integer, nullable=False, default=0)
    last_activity_at = db.Column(
        db.DateTime(timezone=True),
        nullable=True,
    )

    # ── Timestamps ───────────────────────────────────────────────────────────
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # ── Relationships ────────────────────────────────────────────────────────
    document = db.relationship("Document", back_populates="sessions")

    # ── Repr ─────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"<ResearchSession id={self.id!r} "
            f"doc={self.document_id!r} status={self.status!r}>"
        )

    def to_dict(self, include_messages: bool = False) -> dict:
        """Serialise to a plain dict."""
        data = {
            "id": self.id,
            "document_id": self.document_id,
            "title": self.title,
            "status": self.status,
            "message_count": self.message_count,
            "last_activity_at": (
                self.last_activity_at.isoformat() if self.last_activity_at else None
            ),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_messages:
            data["messages"] = self.messages or []
        return data
