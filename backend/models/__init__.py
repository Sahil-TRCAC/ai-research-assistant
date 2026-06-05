"""
models/__init__.py
──────────────────
Exposes the shared SQLAlchemy instance and all models so that
`from models import db, Document, DocumentChunk, ResearchSession` works anywhere.
"""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# Import models after db is defined to avoid circular imports
from .document import Document                  # noqa: E402, F401
from .document_chunk import DocumentChunk       # noqa: E402, F401
from .research_session import ResearchSession   # noqa: E402, F401

__all__ = ["db", "Document", "DocumentChunk", "ResearchSession"]
