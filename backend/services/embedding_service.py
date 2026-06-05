"""
services/embedding_service.py
──────────────────────────────
Generates and stores vector embeddings for DocumentChunk rows.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WHAT ARE EMBEDDINGS?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
An embedding is a fixed-length list of floating-point numbers (a
"vector") that encodes the *semantic meaning* of a piece of text.

  "The capital of France is Paris."  →  [0.12, -0.45, 0.03, ...]
  "Paris is France's capital city."  →  [0.13, -0.44, 0.02, ...]  ← nearly identical
  "I love chocolate ice cream."      →  [0.89,  0.21, -0.67, ...]  ← very different

Two texts that mean the same thing produce vectors that are close
together in the high-dimensional space; unrelated texts are far apart.

Model used: all-MiniLM-L6-v2
  - 384-dimensional output vectors
  - ~80 MB download (cached after first use)
  - ~1–5 ms per chunk on CPU; sub-millisecond on GPU
  - State-of-the-art quality-to-speed ratio for semantic search

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WHY EMBEDDINGS ARE NEEDED FOR RAG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Keyword search (LIKE / full-text search) only finds exact words.
A user asking "What are the main findings?" will not match a chunk
that says "Our results demonstrate..." — different words, same meaning.

Embedding-based (semantic) search solves this:
  1. Embed every chunk at upload time → store as vector
  2. At query time: embed the user's question → get its vector
  3. Find chunks whose vectors are closest (cosine similarity)
  4. Those chunks contain the most *semantically relevant* content
  5. Inject them into the LLM prompt → grounded, accurate answer

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  HOW VECTORS ENABLE SEMANTIC SEARCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cosine similarity between two vectors A and B:

    similarity = (A · B) / (|A| × |B|)   ∈ [-1, 1]

  1.0  = identical meaning
  0.9  = very similar
  0.0  = unrelated
 -1.0  = opposite meaning (rare in practice)

At retrieval time (implemented later):
  - Embed query                 → q_vec
  - Compute similarity(q_vec, chunk_vec) for every chunk
  - Return top-K most similar chunks
  - Inject chunk texts into LLM prompt

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FULL WORKFLOW: CHUNK → EMBEDDING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  POST /api/documents/<id>/embed
        │
        ▼ embedding_service.embed_document_chunks(document_id)
        ├─ 1. Query: SELECT chunks WHERE is_embedded=False
        ├─ 2. Extract chunk_text strings into a list
        ├─ 3. model.encode(texts, batch_size=32)
        │       → numpy array shape (N, 384)
        ├─ 4. For each chunk:
        │       chunk.embedding_vector = vector.tolist()   (JSON in DB)
        │       chunk.embedding_model  = "all-MiniLM-L6-v2"
        │       chunk.is_embedded      = True
        ├─ 5. db.session.commit()
        └─ 6. Return EmbeddingResult(embedded=N, skipped=M, errors=[])

Storage: embedding_vector stored as JSON TEXT in PostgreSQL.
         Each vector is a Python list of 384 floats ≈ 6 KB per chunk.
         pgvector integration (for ANN search) is the next step.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from models import db
from models.document import Document
from models.document_chunk import DocumentChunk

logger = logging.getLogger("ai_research.embedding_service")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_DIMS = 384
DEFAULT_BATCH_SIZE = 32

# ── Singleton model loader ────────────────────────────────────────────────────
_model = None


def _get_model():
    """
    Load the embedding model once per process and cache it.
    Uses fastembed (ONNX runtime) — much lighter than PyTorch.
    """
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s (first-time download may take ~1 min)", MODEL_NAME)
        t0 = time.time()
        try:
            from fastembed import TextEmbedding
            _model = TextEmbedding(model_name=MODEL_NAME, cache_dir=".model_cache")
            logger.info("Model loaded in %.1fs  dims=%d", time.time() - t0, MODEL_DIMS)
        except ImportError:
            raise RuntimeError(
                "fastembed is not installed. "
                "Run: pip install fastembed"
            )
    return _model


def _get_dims() -> int:
    return MODEL_DIMS


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class EmbeddingResult:
    """
    Summary of one embed_document_chunks() call.

    Attributes
    ----------
    document_id  : UUID of the processed document.
    total_chunks : Total chunks in the document.
    embedded     : Chunks successfully embedded in this call.
    skipped      : Chunks that were already embedded (is_embedded=True).
    errors       : List of (chunk_id, error_message) for any failures.
    duration_sec : Wall-clock seconds taken.
    model        : Model name used.
    dims         : Embedding dimension.
    """
    document_id:  str
    total_chunks: int = 0
    embedded:     int = 0
    skipped:      int = 0
    errors:       list[tuple[str, str]] = field(default_factory=list)
    duration_sec: float = 0.0
    model:        str = MODEL_NAME
    dims:         int = 384

    @property
    def success(self) -> bool:
        return len(self.errors) == 0 and (self.embedded + self.skipped) == self.total_chunks

    def to_dict(self) -> dict:
        return {
            "document_id":  self.document_id,
            "total_chunks": self.total_chunks,
            "embedded":     self.embedded,
            "skipped":      self.skipped,
            "error_count":  len(self.errors),
            "errors":       [{"chunk_id": cid, "error": err} for cid, err in self.errors],
            "duration_sec": round(self.duration_sec, 3),
            "model":        self.model,
            "dims":         self.dims,
            "success":      self.success,
        }


# ── Core public API ───────────────────────────────────────────────────────────

def embed_document_chunks(
    document_id: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    re_embed: bool = False,
) -> EmbeddingResult:
    """
    Generate and store embeddings for all unembedded chunks of a document.

    Parameters
    ----------
    document_id : UUID of the Document to embed.
    batch_size  : Number of chunks sent to the model at once (default 32).
                  Increase for GPU; decrease if RAM is limited.
    re_embed    : If True, re-embed chunks that already have embeddings.

    Returns
    -------
    EmbeddingResult with counts and timing.

    Raises
    ------
    ValueError if document not found or has no chunks.
    RuntimeError if the embedding model cannot be loaded.
    """
    t_start = time.time()

    doc: Document | None = Document.query.get(document_id)
    if not doc:
        raise ValueError(f"Document {document_id!r} not found.")

    # Fetch chunks
    query = DocumentChunk.query.filter_by(document_id=document_id).order_by(
        DocumentChunk.chunk_index
    )
    if not re_embed:
        pending = query.filter_by(is_embedded=False).all()
        skipped = query.filter_by(is_embedded=True).count()
    else:
        pending = query.all()
        skipped = 0

    total = query.count()
    result = EmbeddingResult(
        document_id=document_id,
        total_chunks=total,
        skipped=skipped,
        dims=384,
    )

    if not pending:
        logger.info("All %d chunks already embedded for doc=%s", total, document_id)
        result.duration_sec = time.time() - t_start
        return result

    logger.info(
        "Embedding %d/%d chunks for doc=%s  batch_size=%d",
        len(pending), total, document_id, batch_size,
    )

    # Load model (lazy singleton)
    model = _get_model()
    result.dims = _get_dims()

    # Process in batches
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start: batch_start + batch_size]
        texts = [c.chunk_text for c in batch]

        try:
            vectors = list(model.encode(texts))

            for chunk, vector in zip(batch, vectors):
                chunk.embedding_vector = json.dumps(vector.tolist())
                chunk.embedding_model  = MODEL_NAME
                chunk.is_embedded      = True
                result.embedded       += 1

            db.session.flush()

            logger.debug(
                "Batch [%d:%d] embedded OK",
                batch_start, batch_start + len(batch),
            )

        except Exception as exc:
            logger.error("Batch [%d:%d] failed: %s", batch_start, batch_start + len(batch), exc)
            for chunk in batch:
                result.errors.append((chunk.id, str(exc)))

    db.session.commit()
    result.duration_sec = time.time() - t_start

    logger.info(
        "Embedding complete: doc=%s  embedded=%d  skipped=%d  errors=%d  %.2fs",
        document_id, result.embedded, result.skipped,
        len(result.errors), result.duration_sec,
    )
    return result


def get_embedding_status(document_id: str) -> dict:
    """
    Return a summary of embedding progress for a document.

    Returns
    -------
    dict with keys: total, embedded, pending, percent_complete, model
    """
    doc: Document | None = Document.query.get(document_id)
    if not doc:
        raise ValueError(f"Document {document_id!r} not found.")

    total    = DocumentChunk.query.filter_by(document_id=document_id).count()
    embedded = DocumentChunk.query.filter_by(
        document_id=document_id, is_embedded=True
    ).count()
    pending  = total - embedded
    pct      = round((embedded / total * 100), 1) if total > 0 else 0.0

    return {
        "document_id":      document_id,
        "document_title":   doc.title,
        "total_chunks":     total,
        "embedded_chunks":  embedded,
        "pending_chunks":   pending,
        "percent_complete": pct,
        "is_fully_embedded": pending == 0 and total > 0,
        "model":            MODEL_NAME,
        "dims":             384,
    }


def embed_single_text(text: str) -> list[float]:
    """
    Embed a single string and return its vector as a Python list.
    Used later for query embedding at retrieval time.
    """
    model = _get_model()
    vec = list(model.encode([text]))[0]
    return vec.tolist()
