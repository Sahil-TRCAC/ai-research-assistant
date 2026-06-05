"""
models/document.py
──────────────────
Represents a research document uploaded by the user.
Stores file metadata AND extracted text content in PostgreSQL.

Key columns
───────────
- extracted_text   : Full text extracted from the PDF (TEXT, potentially large)
- content_preview  : First 500 chars of extracted_text (for list views)
- is_scanned       : True when pypdf found pages but no extractable text
- extraction_error : Human-readable error if extraction failed
"""

import uuid
from datetime import datetime, timezone
from models import db


class Document(db.Model):
    __tablename__ = "documents"

    # ── Primary key ──────────────────────────────────────────────────────────
    id = db.Column(
        db.String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # ── Core fields ──────────────────────────────────────────────────────────
    title = db.Column(db.String(500), nullable=False)
    original_filename = db.Column(db.String(500), nullable=False)
    stored_filename = db.Column(db.String(500), nullable=False, unique=True)
    file_path = db.Column(db.String(1000), nullable=False)
    file_size = db.Column(db.BigInteger, nullable=False, default=0)  # bytes
    file_type = db.Column(db.String(50), nullable=False)             # pdf, docx, …
    mime_type = db.Column(db.String(100), nullable=True)

    # ── Content ──────────────────────────────────────────────────────────────
    # Full text stored in PostgreSQL — can be multi-MB for large PDFs
    extracted_text = db.Column(db.Text, nullable=True)
    # Short preview for list endpoints (first ~500 chars of extracted_text)
    content_preview = db.Column(db.Text, nullable=True)
    page_count = db.Column(db.Integer, nullable=True)
    word_count = db.Column(db.Integer, nullable=True)
    char_count = db.Column(db.Integer, nullable=True)
    # Extraction outcome flags
    is_scanned = db.Column(db.Boolean, nullable=False, default=False)
    extraction_error = db.Column(db.Text, nullable=True)

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_count = db.Column(db.Integer, nullable=True)  # set after chunking

    # ── Status ───────────────────────────────────────────────────────────────
    status = db.Column(
        db.String(30),
        nullable=False,
        default="uploaded",
        index=True,
    )
    # Possible statuses: uploaded | processing | ready | error

    error_message = db.Column(db.Text, nullable=True)

    # ── Metadata ─────────────────────────────────────────────────────────────
    description = db.Column(db.Text, nullable=True)
    tags = db.Column(db.JSON, nullable=False, default=list)

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
    sessions = db.relationship(
        "ResearchSession",
        back_populates="document",
        lazy="dynamic",
        cascade="all, delete-orphan",
    )
    chunks = db.relationship(
        "DocumentChunk",
        back_populates="document",
        lazy="dynamic",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.chunk_index",
    )

    # ── Repr ─────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return f"<Document id={self.id!r} title={self.title!r} status={self.status!r}>"

    def to_dict(
        self,
        include_preview: bool = False,
        include_content: bool = False,
    ) -> dict:
        """
        Serialise to a plain dict (safe for JSON responses).

        Parameters
        ----------
        include_preview : Include the 500-char content_preview field.
        include_content : Include the full extracted_text (can be large!).
        """
        data = {
            "id": self.id,
            "title": self.title,
            "original_filename": self.original_filename,
            "file_size": self.file_size,
            "file_type": self.file_type,
            "mime_type": self.mime_type,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "status": self.status,
            "is_scanned": self.is_scanned,
            "extraction_error": self.extraction_error,
            "chunk_count": self.chunk_count,
            "description": self.description,
            "tags": self.tags or [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_preview:
            data["content_preview"] = self.content_preview
        if include_content:
            data["extracted_text"] = self.extracted_text
        return data
