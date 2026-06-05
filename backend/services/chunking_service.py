"""
services/chunking_service.py
─────────────────────────────
Splits extracted document text into overlapping chunks for the RAG pipeline.

Why Chunking is Necessary for RAG
──────────────────────────────────
Retrieval-Augmented Generation works by:
  1. Embedding text chunks into vectors
  2. Storing those vectors in a vector database
  3. At query time: embedding the user's question, finding the top-K most
     similar chunks via cosine similarity, and injecting them into the LLM prompt

This only works if each chunk fits within the embedding model's token limit
(typically 256–512 tokens ≈ 1,000–2,000 characters). A full PDF cannot be
embedded as a single unit — it must be chunked first.

How Overlap Improves Retrieval
───────────────────────────────
Without overlap:
    [chunk 0: chars 0–499]  [chunk 1: chars 500–999]
    A sentence starting at char 480 is split: its beginning lives in chunk 0,
    its end in chunk 1. Neither chunk captures the full thought.

With 100-char overlap:
    [chunk 0: chars 0–499]  [chunk 1: chars 400–899]  ...
    The tail of chunk 0 reappears at the head of chunk 1.
    The split sentence now appears complete in chunk 1.
    Retrieval is more likely to return a chunk that contains the full context.

Full Data Flow (PDF → Chunks)
──────────────────────────────
  POST /api/documents/upload
      │
      ▼
  document_service.upload_document()
      │  saves file, calls pdf_extractor.extract()
      │  commits Document row (status=ready, extracted_text=...)
      │
      ▼
  chunking_service.chunk_document(document_id)
      │  reads Document.extracted_text from DB
      │  calls split_text_into_chunks(text, size=500, overlap=100)
      │     ├─ splits on sentence boundaries when possible
      │     ├─ skips empty chunks
      │     └─ tracks start_char / end_char offsets
      │  bulk-inserts DocumentChunk rows into PostgreSQL
      │  updates Document.chunk_count
      │
      ▼
  PostgreSQL: document_chunks table
      chunk_index │ chunk_text          │ start_char │ end_char
      0           │ "Introduction..."   │ 0          │ 497
      1           │ "...the model..."   │ 400        │ 897   ← 100-char overlap
      2           │ "...results show"   │ 800        │ ...
"""

from __future__ import annotations

import re
import logging
from datetime import datetime, timezone
from typing import Generator

from models import db
from models.document import Document
from models.document_chunk import DocumentChunk

logger = logging.getLogger("ai_research.chunking_service")

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 500    # characters per chunk
DEFAULT_OVERLAP    = 100    # characters of overlap between adjacent chunks
MIN_CHUNK_CHARS    = 50     # skip chunks shorter than this


# ── Core splitting logic ──────────────────────────────────────────────────────

def split_text_into_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[dict]:
    """
    Split `text` into overlapping chunks, preferring sentence boundaries.

    Algorithm
    ─────────
    1. Start at position `cursor = 0`.
    2. Take a window of `chunk_size` characters.
    3. Search backwards from the end of the window for the last sentence-ending
       punctuation (. ! ?). If found, cut there to keep sentences whole.
    4. If no sentence boundary exists in the window, fall back to the last
       whitespace. If no whitespace either, hard-cut at `chunk_size`.
    5. Record start_char / end_char offsets for each chunk.
    6. Advance cursor by `(chunk_size - overlap)` to create the next window.
    7. Skip chunks shorter than `min_chars`.

    Parameters
    ----------
    text       : Full extracted text.
    chunk_size : Target character length per chunk.
    overlap    : Overlap in characters between consecutive chunks.
    min_chars  : Minimum character length to keep a chunk.

    Returns
    -------
    List of dicts with keys: chunk_index, chunk_text, char_count,
    start_char, end_char.
    """
    text = text.strip()
    if not text:
        return []

    step    = max(1, chunk_size - overlap)
    cursor  = 0
    index   = 0
    chunks: list[dict] = []

    while cursor < len(text):
        window_end = cursor + chunk_size

        if window_end >= len(text):
            # Last chunk — take everything remaining
            raw_chunk = text[cursor:]
        else:
            raw_chunk = text[cursor:window_end]
            # ── Prefer sentence boundary ──────────────────────────────────────
            cut = _find_sentence_boundary(raw_chunk)
            if cut is not None:
                raw_chunk = raw_chunk[: cut + 1]
            else:
                # Fall back to whitespace boundary
                ws_cut = raw_chunk.rfind(" ")
                if ws_cut > chunk_size // 2:
                    raw_chunk = raw_chunk[:ws_cut]
                # else: hard cut at chunk_size — rare for natural language

        chunk_text = raw_chunk.strip()
        char_count = len(chunk_text)

        if char_count >= min_chars:
            chunks.append({
                "chunk_index": index,
                "chunk_text":  chunk_text,
                "char_count":  char_count,
                "start_char":  cursor,
                "end_char":    cursor + len(raw_chunk),
            })
            index += 1

        cursor += step

    return chunks


def _find_sentence_boundary(text: str) -> int | None:
    """
    Search backwards in `text` for the last sentence-ending character
    followed by whitespace (or end-of-string).

    Returns the index of the sentence-ending character, or None.
    """
    # Match . ! ? optionally followed by " ' ) and then whitespace
    pattern = re.compile(r'[.!?]["\')]?\s')
    matches = list(pattern.finditer(text))
    if matches:
        last = matches[-1]
        return last.start()   # index of the . or ! or ?
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_document(
    document_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    replace_existing: bool = True,
) -> list[DocumentChunk]:
    """
    Read a Document's extracted_text, split it into chunks, and persist
    all DocumentChunk rows to PostgreSQL in a single bulk operation.

    Parameters
    ----------
    document_id      : UUID of the target Document.
    chunk_size       : Characters per chunk (default 500).
    overlap          : Overlap between chunks (default 100).
    replace_existing : Delete old chunks before inserting new ones (default True).

    Returns
    -------
    List of persisted DocumentChunk instances.

    Raises
    ------
    ValueError if document not found or has no extracted_text.
    """
    doc: Document | None = Document.query.get(document_id)
    if not doc:
        raise ValueError(f"Document {document_id!r} not found.")

    if not doc.extracted_text:
        raise ValueError(
            f"Document {document_id!r} has no extracted_text. "
            "Ensure PDF extraction completed successfully before chunking."
        )

    logger.info(
        "Chunking document: id=%s chars=%d chunk_size=%d overlap=%d",
        document_id, len(doc.extracted_text), chunk_size, overlap,
    )

    # ── Delete existing chunks if re-chunking ─────────────────────────────────
    if replace_existing:
        deleted = DocumentChunk.query.filter_by(document_id=document_id).delete()
        if deleted:
            logger.info("Deleted %d existing chunks for doc=%s", deleted, document_id)

    # ── Split text ────────────────────────────────────────────────────────────
    raw_chunks = split_text_into_chunks(
        doc.extracted_text, chunk_size=chunk_size, overlap=overlap
    )

    if not raw_chunks:
        logger.warning("No chunks produced for document: %s", document_id)
        return []

    # ── Bulk insert ───────────────────────────────────────────────────────────
    chunk_objects: list[DocumentChunk] = [
        DocumentChunk(
            document_id = document_id,
            chunk_index = c["chunk_index"],
            chunk_text  = c["chunk_text"],
            char_count  = c["char_count"],
            start_char  = c["start_char"],
            end_char    = c["end_char"],
        )
        for c in raw_chunks
    ]

    db.session.bulk_save_objects(chunk_objects)

    # ── Update Document.chunk_count ───────────────────────────────────────────
    doc.chunk_count  = len(raw_chunks)
    doc.updated_at   = datetime.now(timezone.utc)
    db.session.commit()

    logger.info(
        "Chunking complete: doc=%s  chunks=%d  avg_chars=%.0f",
        document_id,
        len(raw_chunks),
        sum(c["char_count"] for c in raw_chunks) / len(raw_chunks),
    )

    return DocumentChunk.query.filter_by(document_id=document_id).order_by(
        DocumentChunk.chunk_index
    ).all()


def get_chunks_for_document(
    document_id: str,
    page: int = 1,
    per_page: int = 50,
) -> tuple[list[DocumentChunk], int]:
    """
    Return paginated chunks for a document.

    Returns
    -------
    (list_of_chunks, total_count)
    """
    query = (
        DocumentChunk.query
        .filter_by(document_id=document_id)
        .order_by(DocumentChunk.chunk_index)
    )
    total  = query.count()
    chunks = query.offset((page - 1) * per_page).limit(per_page).all()
    return chunks, total


def delete_chunks_for_document(document_id: str) -> int:
    """Delete all chunks for a document. Returns count deleted."""
    count = DocumentChunk.query.filter_by(document_id=document_id).delete()
    db.session.commit()
    logger.info("Deleted %d chunks for doc=%s", count, document_id)
    return count
