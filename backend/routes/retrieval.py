"""
routes/retrieval.py
────────────────────
Semantic search endpoint — the query side of the RAG pipeline.

POST /api/retrieval/search
    Accept a natural-language question, embed it with the same
    all-MiniLM-L6-v2 model used at upload time, compute cosine
    similarity against every stored chunk vector, and return the
    top-K most relevant passages with their similarity scores and
    source document metadata.

This endpoint does NOT call any LLM.
It is the retrieval step only — the R in RAG.
"""

import logging
from flask import Blueprint, request

from services import retrieval_service
from utils.response import success_response, error_response
from utils.validators import validate_uuid

logger = logging.getLogger("ai_research.retrieval")

retrieval_bp = Blueprint("retrieval", __name__, url_prefix="/api/retrieval")


# ── POST /api/retrieval/search ────────────────────────────────────────────────

@retrieval_bp.post("/search")
def search():
    """
    Semantic search over all embedded document chunks.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      HOW THIS WORKS (step by step)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. Parse the query string from the JSON body.

    2. Embed the query with all-MiniLM-L6-v2 (same model used
       when the document was uploaded) → 384-float unit vector.

    3. Load all embedded DocumentChunk rows from the database and
       deserialise their embedding_vector JSON columns into a
       NumPy matrix of shape (N, 384).

    4. Compute cosine similarity between the query vector and
       every chunk vector using a single matrix-vector dot product:
           scores = chunk_matrix @ query_vector   shape (N,)
       This is valid because all vectors are already unit-normalised
       (normalize_embeddings=True in embedding_service.py).

    5. Pick the top-K indices (default 5) by descending score.

    6. Attach document metadata (title, filename, page count) for
       each result — no extra per-chunk DB query (batched fetch).

    7. Return the ranked list as JSON.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      REQUEST BODY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {
      "query":       "What are the key findings?",   // required
      "top_k":       5,                              // optional, default 5, max 50
      "document_id": "uuid...",                      // optional — restrict to one doc
      "min_score":   0.0                             // optional — minimum similarity
    }

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      RESPONSE (200 OK)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {
      "success": true,
      "message": "Found 5 relevant chunks.",
      "data": {
        "query":                 "What are the key findings?",
        "results": [
          {
            "chunk_id":          "uuid...",
            "chunk_text":        "Our results demonstrate that ...",
            "chunk_index":       3,
            "similarity_score":  0.9142,
            "document_id":       "uuid...",
            "document_title":    "Attention Is All You Need",
            "original_filename": "attention.pdf",
            "page_count":        15
          },
          ...
        ],
        "results_returned":      5,
        "top_k":                 5,
        "total_chunks_searched": 48,
        "duration_sec":          0.034,
        "query_dims":            384,
        "document_filter":       null
      }
    }

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      SIMILARITY SCORE INTERPRETATION
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    0.90+   Excellent match — essentially the answer
    0.70–0.89 Good match — clearly relevant
    0.50–0.69 Weak match — loosely related
    <0.50   Poor match — likely irrelevant (raise min_score to filter)
    """
    body = request.get_json(silent=True) or {}

    # ── Required: query ───────────────────────────────────────────────────────
    query = body.get("query", "")
    if not isinstance(query, str) or not query.strip():
        return error_response(
            "Field 'query' is required and must be a non-empty string.",
            400,
            "MISSING_QUERY",
        )

    # ── Optional: top_k ───────────────────────────────────────────────────────
    try:
        top_k = int(body.get("top_k", 5))
    except (TypeError, ValueError):
        return error_response("'top_k' must be an integer.", 400, "INVALID_PARAMS")

    if top_k < 1 or top_k > 50:
        return error_response("'top_k' must be between 1 and 50.", 400, "INVALID_PARAMS")

    # ── Optional: document_id filter ──────────────────────────────────────────
    document_id = body.get("document_id") or None
    if document_id and not validate_uuid(document_id):
        return error_response(
            "'document_id' must be a valid UUID.", 400, "INVALID_ID"
        )

    # ── Optional: min_score ───────────────────────────────────────────────────
    try:
        min_score = float(body.get("min_score", 0.0))
    except (TypeError, ValueError):
        return error_response("'min_score' must be a float.", 400, "INVALID_PARAMS")

    if not (-1.0 <= min_score <= 1.0):
        return error_response(
            "'min_score' must be between -1.0 and 1.0.", 400, "INVALID_PARAMS"
        )

    # ── Execute retrieval ─────────────────────────────────────────────────────
    logger.info(
        "Search request: query=%r  top_k=%d  document_id=%s  min_score=%.2f",
        query[:80], top_k, document_id, min_score,
    )

    try:
        result = retrieval_service.search(
            query=query,
            top_k=top_k,
            document_id=document_id,
            min_score=min_score,
        )
    except ValueError as exc:
        return error_response(str(exc), 400, "INVALID_QUERY")
    except RuntimeError as exc:
        # Embedding model failed to load
        return error_response(str(exc), 503, "MODEL_UNAVAILABLE")
    except Exception as exc:
        logger.exception("Unexpected error during retrieval")
        return error_response("Retrieval failed.", 500, "RETRIEVAL_FAILED", str(exc))

    # ── Build response message ────────────────────────────────────────────────
    n = len(result.results)
    if n == 0:
        message = (
            "No embedded chunks found. "
            "Make sure documents are uploaded and embedded first."
        )
    elif n == 1:
        message = "Found 1 relevant chunk."
    else:
        message = f"Found {n} relevant chunks."

    return success_response(
        data=result.to_dict(),
        message=message,
    )
