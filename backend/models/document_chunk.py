"""
models/document_chunk.py
─────────────────────────
Represents a single text chunk produced by the RAG chunking pipeline.

Each DocumentChunk holds a slice of a Document's extracted_text,
ready to be embedded and stored in a vector database.

Why chunking?
─────────────
LLM context windows and embedding models have token limits (~512–8192 tokens).
A 50-page PDF may contain 25,000+ words — far too large to process as one unit.
Chunking splits the text into overlapping windows so that:
  1. Every chunk fits within the embedding model's token limit.
  2. Semantic meaning is preserved by respecting sentence boundaries.
  3. Overlap ensures that context spanning two adjacent chunks is not lost.

Why overlap?
────────────
Without overlap, a sentence split across two chunks would be semantically
broken in both. A 100-character overlap repeats the tail of chunk N at the
head of chunk N+1, so retrieval can find the relevant passage regardless of
which chunk boundary it falls near.
"""

import uuid
from datetime import datetime, timezone
from models import db


class DocumentChunk(db.Model):
    __tablename__ = "document_chunks"

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

    # ── Chunk content ─────────────────────────────────────────────────────────
    chunk_index = db.Column(db.Integer, nullable=False)   # 0-based order
    chunk_text  = db.Column(db.Text, nullable=False)
    char_count  = db.Column(db.Integer, nullable=False, default=0)

    # ── Chunking provenance ───────────────────────────────────────────────────
    # Character offsets in the original extracted_text (useful for highlighting)
    start_char  = db.Column(db.Integer, nullable=True)
    end_char    = db.Column(db.Integer, nullable=True)

    # ── Embedding status (for future vector store integration) ────────────────
    is_embedded     = db.Column(db.Boolean, nullable=False, default=False, index=True)
    embedding_model = db.Column(db.String(100), nullable=True)

    # Serialised as JSON: a list of 384 floats, e.g. "[0.12, -0.45, ...]"
    # ~6 KB per chunk. Use pgvector extension for ANN search later.
    embedding_vector = db.Column(db.Text, nullable=True)

    # ── Timestamps ───────────────────────────────────────────────────────────
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # ── Relationships ────────────────────────────────────────────────────────
    document = db.relationship("Document", back_populates="chunks")

    # ── Constraints ───────────────────────────────────────────────────────────
    __table_args__ = (
        db.UniqueConstraint("document_id", "chunk_index", name="uq_doc_chunk_index"),
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentChunk doc={self.document_id!r} "
            f"idx={self.chunk_index} chars={self.char_count}>"
        )

    def to_dict(self, include_text: bool = True, include_vector: bool = False) -> dict:
        data = {
            "id":            self.id,
            "document_id":   self.document_id,
            "chunk_index":   self.chunk_index,
            "char_count":    self.char_count,
            "start_char":    self.start_char,
            "end_char":      self.end_char,
            "is_embedded":   self.is_embedded,
            "embedding_model": self.embedding_model,
            "created_at":    self.created_at.isoformat() if self.created_at else None,
        }
        if include_text:
            data["chunk_text"] = self.chunk_text
        if include_vector and self.embedding_vector:
            import json as _json
            data["embedding_vector"] = _json.loads(self.embedding_vector)
        return data
