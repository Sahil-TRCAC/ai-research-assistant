"""
services/retrieval_service.py
──────────────────────────────
Semantic (vector) retrieval over stored DocumentChunk embeddings.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WHAT IS SEMANTIC SEARCH?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Keyword search (LIKE / full-text) matches *exact words*.
If a user asks "What are the key findings?" but the paper says
"Our results demonstrate...", keyword search finds nothing.

Semantic search instead measures *meaning proximity*:
  1. Convert every chunk to a 384-dim vector at upload time (done ✓)
  2. At query time, convert the question to the same vector space
  3. Rank all chunks by how close they are to the query vector
  4. Return the top-K most relevant chunks

This works because all-MiniLM-L6-v2 maps semantically similar
sentences to nearby points in 384-dimensional space — regardless
of the exact words used.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  COSINE SIMILARITY — THE DISTANCE METRIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Given two vectors A (query) and B (chunk):

    cosine_similarity(A, B) = (A · B) / (|A| × |B|)

  ┌─────────────────────────────────────────────────┐
  │  Score │ Interpretation                         │
  │────────│────────────────────────────────────────│
  │  1.00  │ Identical meaning                      │
  │  0.90+ │ Very strong match — almost the answer  │
  │  0.70+ │ Good match — clearly relevant          │
  │  0.50+ │ Weak match — loosely related            │
  │  0.00  │ Unrelated                              │
  │ -1.00  │ Opposite meaning (very rare)           │
  └─────────────────────────────────────────────────┘

Because embedding_service normalises vectors to unit length
(normalize_embeddings=True), the denominator |A|×|B| = 1×1 = 1,
so cosine similarity simplifies to the dot product:

    cosine_similarity(A, B) = A · B   (when both are unit vectors)

This makes the computation extremely fast — just a dot product.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TOP-K RETRIEVAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After computing similarity scores for all chunks, we sort
descending and return the top K (default 5).

K = 5 is the standard RAG starting point:
  - Small enough to fit in any LLM context window
  - Large enough to cover multi-facet questions
  - Empirically shown to give best precision/recall trade-off

Later you can tune K, add a minimum-score threshold, or use
approximate nearest-neighbor (ANN) search with pgvector for
large corpora (100k+ chunks). For this project's document count,
brute-force cosine over all chunks is fast enough (<50 ms).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FULL QUERY → RETRIEVAL FLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  POST /api/retrieval/search  {"query": "What are the key findings?"}
        │
        ▼ retrieval_service.search(query, top_k=5)
        │
        ├─ 1. Validate query (non-empty string)
        │
        ├─ 2. Embed query
        │       embedding_service.embed_single_text(query)
        │       → q_vec: list[float] of length 384
        │         (unit-normalised, same space as chunk vectors)
        │
        ├─ 3. Load embedded chunks from DB
        │       SELECT * FROM document_chunks WHERE is_embedded=True
        │       Deserialise embedding_vector (JSON → list[float])
        │       → chunks: list[DocumentChunk]
        │         chunk_vecs: numpy array  shape (N, 384)
        │
        ├─ 4. Compute cosine similarity (vectorised)
        │       scores = chunk_vecs @ q_vec          ← dot product
        │       (valid because all vectors are unit-normalised)
        │       → scores: numpy array  shape (N,)
        │
        ├─ 5. Top-K selection
        │       top_indices = argsort(scores)[-top_k:][::-1]
        │       → ordered indices of the K highest-scoring chunks
        │
        ├─ 6. Build results
        │       For each top chunk:
        │         - chunk_text       (the passage to show/inject)
        │         - similarity_score (float 0–1, rounded to 4 dp)
        │         - document_id      (UUID of source document)
        │         - chunk_index      (position in document)
        │         - document metadata (title, filename, page_count…)
        │
        └─ 7. Return RetrievalResult(results=[...], query, top_k, …)
"""

from __future__ import annotations

import heapq
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from models.document_chunk import DocumentChunk
from models.document import Document
from services.embedding_service import embed_single_text

logger = logging.getLogger("ai_research.retrieval_service")

DEFAULT_TOP_K = 5
MIN_SCORE_THRESHOLD = 0.0   # Return all top-K regardless of score (caller may filter)
BATCH_SIZE = 500            # Load & score chunks in batches to limit memory


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class ChunkResult:
    """
    A single retrieved chunk with its similarity score and source metadata.

    Attributes
    ----------
    chunk_id        : Primary key of the DocumentChunk row.
    chunk_text      : The raw text passage.
    chunk_index     : Zero-based position in the source document.
    similarity_score: Cosine similarity to the query vector (0–1).
    document_id     : UUID of the parent Document.
    document_title  : Human-readable title of the document.
    original_filename: Original uploaded filename.
    page_count      : Number of pages in the source PDF.
    """
    chunk_id:          str
    chunk_text:        str
    chunk_index:       int
    similarity_score:  float
    document_id:       str
    document_title:    str
    original_filename: str
    page_count:        Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "chunk_id":          self.chunk_id,
            "chunk_text":        self.chunk_text,
            "chunk_index":       self.chunk_index,
            "similarity_score":  round(self.similarity_score, 4),
            "document_id":       self.document_id,
            "document_title":    self.document_title,
            "original_filename": self.original_filename,
            "page_count":        self.page_count,
        }


@dataclass
class RetrievalResult:
    """
    Complete result of one search() call.

    Attributes
    ----------
    query           : The original query string.
    results         : Ordered list of ChunkResult (best first).
    top_k           : How many results were requested.
    total_chunks_searched: Total embedded chunks that were scored.
    duration_sec    : Wall-clock time for the entire retrieval.
    query_dims      : Dimensionality of the query embedding.
    document_filter : document_id filter applied (None = all docs).
    """
    query:                  str
    results:                list[ChunkResult] = field(default_factory=list)
    top_k:                  int = DEFAULT_TOP_K
    total_chunks_searched:  int = 0
    duration_sec:           float = 0.0
    query_dims:             int = 384
    document_filter:        Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "query":                 self.query,
            "results":               [r.to_dict() for r in self.results],
            "top_k":                 self.top_k,
            "results_returned":      len(self.results),
            "total_chunks_searched": self.total_chunks_searched,
            "duration_sec":          round(self.duration_sec, 3),
            "query_dims":            self.query_dims,
            "document_filter":       self.document_filter,
        }


# ── Core retrieval function ───────────────────────────────────────────────────

def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    document_id: Optional[str] = None,
    min_score: float = MIN_SCORE_THRESHOLD,
) -> RetrievalResult:
    """
    Perform semantic search over all embedded DocumentChunks.

    Parameters
    ----------
    query       : The user's natural-language question.
    top_k       : How many top results to return (default 5).
    document_id : If provided, restrict search to chunks of that document.
                  If None, search across all documents.
    min_score   : Minimum cosine similarity to include in results.
                  Chunks below this threshold are dropped even if top-K.

    Returns
    -------
    RetrievalResult with ordered ChunkResult list (highest score first).

    Raises
    ------
    ValueError  : query is empty or top_k < 1.
    RuntimeError: embedding model cannot be loaded.
    """
    t_start = time.time()

    # ── 1. Validate inputs ────────────────────────────────────────────────────
    query = query.strip()
    if not query:
        raise ValueError("Query must be a non-empty string.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    top_k = min(top_k, 50)  # hard cap — prevents absurdly large payloads

    logger.info(
        "Retrieval query=%r  top_k=%d  document_filter=%s",
        query[:80], top_k, document_id,
    )

    # ── 2. Embed the query ────────────────────────────────────────────────────
    # embed_single_text uses the same model + normalisation as chunk embedding,
    # so query and chunk vectors live in the same unit-normalised 384-dim space.
    q_vec = np.array(embed_single_text(query), dtype=np.float32)  # shape (384,)
    query_dims = len(q_vec)

    logger.debug("Query embedded: dims=%d  norm=%.4f", query_dims, float(np.linalg.norm(q_vec)))

    # ── 3. Load embedded chunks in batches ────────────────────────────────────
    chunk_query = DocumentChunk.query.filter_by(is_embedded=True)
    if document_id:
        chunk_query = chunk_query.filter_by(document_id=document_id)

    total_count = chunk_query.count()
    if total_count == 0:
        logger.warning(
            "No embedded chunks found%s. Embed documents first.",
            f" for document {document_id!r}" if document_id else "",
        )
        return RetrievalResult(
            query=query,
            top_k=top_k,
            document_filter=document_id,
            duration_sec=time.time() - t_start,
            query_dims=query_dims,
        )

    total_batches = max(1, math.ceil(total_count / BATCH_SIZE))
    logger.info("Scoring %d embedded chunks (%d batches of %d)", total_count, total_batches, BATCH_SIZE)

    # Min-heap tracking top-K across batches: (score, chunk_id, doc_id, chunk_index, chunk_text)
    # Only keeps top-K entries so we never hold all chunks in memory at once.
    heap: list[tuple[float, str, str, int, str]] = []
    total_valid = 0

    for batch_num in range(total_batches):
        offset = batch_num * BATCH_SIZE
        batch_chunks: list[DocumentChunk] = (
            chunk_query
            .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
            .offset(offset)
            .limit(BATCH_SIZE)
            .all()
        )

        rows: list[list[float]] = []
        for chunk in batch_chunks:
            if not chunk.embedding_vector:
                continue
            try:
                vec = json.loads(chunk.embedding_vector)
                rows.append(vec)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Corrupt embedding_vector for chunk %s: %s", chunk.id, exc)

        if not rows:
            continue

        vecs = np.array(rows, dtype=np.float32)
        scores: np.ndarray = vecs @ q_vec

        for local_idx in range(len(scores)):
            chunk = batch_chunks[local_idx]
            score = float(scores[local_idx])
            total_valid += 1
            entry = (score, chunk.id, chunk.document_id, chunk.chunk_index, chunk.chunk_text)
            if len(heap) < top_k:
                heapq.heappush(heap, entry)
            elif score > heap[0][0]:
                heapq.heappushpop(heap, entry)

    if total_valid == 0:
        logger.error("No valid vectors found after deserialisation.")
        return RetrievalResult(
            query=query,
            top_k=top_k,
            document_filter=document_id,
            duration_sec=time.time() - t_start,
            query_dims=query_dims,
        )

    # ── 4. Build top-K results ────────────────────────────────────────────────
    # Heap sorted ascending by score; extract in descending order.
    top_items = sorted(heap, key=lambda x: -x[0])

    doc_ids_needed: set[str] = set()
    result_refs: list[tuple[float, str, str, int, str]] = []
    for item in top_items:
        score, chunk_id, doc_id, chunk_index, chunk_text = item
        if score < min_score:
            continue
        doc_ids_needed.add(doc_id)
        result_refs.append(item)

    # Batch-fetch parent Documents to avoid N+1
    docs_by_id: dict[str, Document] = {
        doc.id: doc
        for doc in Document.query.filter(Document.id.in_(doc_ids_needed)).all()
    } if doc_ids_needed else {}

    results: list[ChunkResult] = []
    for score, chunk_id, doc_id, chunk_index, chunk_text in result_refs:
        doc = docs_by_id.get(doc_id)
        results.append(ChunkResult(
            chunk_id          = chunk_id,
            chunk_text        = chunk_text,
            chunk_index       = chunk_index,
            similarity_score  = score,
            document_id       = doc_id,
            document_title    = doc.title if doc else "Unknown",
            original_filename = doc.original_filename if doc else "Unknown",
            page_count        = doc.page_count if doc else None,
        ))

    duration = time.time() - t_start
    top_score = results[0].similarity_score if results else 0.0
    logger.info(
        "Retrieval done: returned=%d  searched=%d  top_score=%.4f  %.3fs",
        len(results), total_valid, top_score, duration,
    )

    return RetrievalResult(
        query                 = query,
        results               = results,
        top_k                 = top_k,
        total_chunks_searched = total_valid,
        duration_sec          = duration,
        query_dims            = query_dims,
        document_filter       = document_id,
    )
