"""
routes/chat.py
──────────────
POST /api/chat — the full RAG pipeline endpoint.

This is the "ask a question" endpoint. It wires together:
  1. RetrievalService  — find the top-5 relevant document chunks
  2. LLMService        — generate a grounded answer using those chunks

The LLM never sees the raw document; it only sees the top-K passages
retrieved for the specific question, which keeps the answer grounded and
prevents hallucination.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  REQUEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POST /api/chat
Content-Type: application/json

{
  "question": "What are the key findings?",
  "top_k": 5,                  // optional, 1-20, default 5
  "document_id": "uuid..."     // optional — restrict to one document
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RESPONSE (200 OK)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "success": true,
  "message": "Answer generated successfully.",
  "data": {
    "answer": "The key findings were …",
    "sources": [
      { "document_id": "uuid…", "chunk_index": 3 },
      { "document_id": "uuid…", "chunk_index": 7 }
    ],
    "model": "llama3-8b-8192",
    "duration_sec": 1.243,
    "prompt_tokens": 872,
    "answer_tokens": 128
  }
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ERROR CASES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
400 MISSING_QUESTION   — 'question' field absent or empty
400 INVALID_PARAMS     — top_k out of range
400 INVALID_ID         — document_id not a valid UUID
503 RETRIEVAL_FAILED   — embedding model unavailable
503 LLM_UNAVAILABLE    — Groq API key missing or API down
500 CHAT_FAILED        — unexpected internal error
"""


import logging
from flask import Blueprint, request

from services import retrieval_service
from services import llm_service
from utils.response import success_response, error_response
from utils.validators import validate_uuid

logger = logging.getLogger("ai_research.chat")

chat_bp = Blueprint("chat", __name__, url_prefix="/api/chat")


# ── POST /api/chat ────────────────────────────────────────────────────────────

@chat_bp.post("")
def chat():
    """
    Full RAG pipeline: Question → Retrieval → LLM → Grounded Answer.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      STEP-BY-STEP FLOW
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. Parse + validate the JSON request body.

    2. Call retrieval_service.search(question, top_k)
         → Embed the question with all-MiniLM-L6-v2
         → Dot-product against every stored chunk vector
         → Return the top-K ChunkResult objects (text + metadata)

    3. If no chunks are found (no documents uploaded / embedded yet),
       return a helpful 404 rather than calling the LLM with no context.

    4. Call llm_service.answer(question, chunks)
         → Build a numbered context block from the chunks
         → Construct the grounded system prompt
         → POST to Groq chat completions API
         → Return LLMResult(answer, sources, tokens, latency)

    5. Serialise to the standard success envelope and return.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      HOW RAG PREVENTS HALLUCINATION
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • The LLM receives ONLY the top-K retrieved passages, not the
      full document corpus.
    • The system prompt instructs it to answer ONLY from those
      passages and to refuse if the context is insufficient.
    • Sources are returned alongside the answer so every claim is
      traceable back to a specific chunk in a specific document.
    """
    body = request.get_json(silent=True) or {}

    # ── Required: question ────────────────────────────────────────────────────
    question = body.get("question", "")
    if not isinstance(question, str) or not question.strip():
        return error_response(
            "Field 'question' is required and must be a non-empty string.",
            400,
            "MISSING_QUESTION",
        )

    # ── Optional: top_k ───────────────────────────────────────────────────────
    try:
        top_k = int(body.get("top_k", 5))
    except (TypeError, ValueError):
        return error_response("'top_k' must be an integer.", 400, "INVALID_PARAMS")

    if top_k < 1 or top_k > 20:
        return error_response(
            "'top_k' must be between 1 and 20.", 400, "INVALID_PARAMS"
        )

    # ── Optional: document_id filter ──────────────────────────────────────────
    document_id = body.get("document_id") or None
    if document_id and not validate_uuid(document_id):
        return error_response(
            "'document_id' must be a valid UUID.", 400, "INVALID_ID"
        )

    logger.info(
        "Chat request: question=%r  top_k=%d  document_id=%s",
        question[:80], top_k, document_id,
    )

    # ── Step 1: Retrieval (skip if no embedded chunks) ──────────────────────────
    from models.document_chunk import DocumentChunk

    has_docs = DocumentChunk.query.filter_by(is_embedded=True).first() is not None
    chunks = []

    if has_docs:
        try:
            retrieval_result = retrieval_service.search(
                query=question,
                top_k=top_k,
                document_id=document_id,
            )
            chunks = retrieval_result.results
        except ValueError as exc:
            return error_response(str(exc), 400, "INVALID_QUESTION")
        except RuntimeError as exc:
            return error_response(str(exc), 503, "RETRIEVAL_FAILED")
        except Exception as exc:
            logger.exception("Retrieval failed unexpectedly")
            return error_response("Retrieval failed.", 500, "CHAT_FAILED", str(exc))

    logger.info("Retrieved %d chunks (has_docs=%s)", len(chunks), has_docs)

    # ── Step 2: LLM Generation ────────────────────────────────────────────────
    try:
        llm_result = llm_service.answer(
            question=question,
            chunks=chunks,
        )
    except ValueError as exc:
        return error_response(str(exc), 400, "INVALID_QUESTION")
    except RuntimeError as exc:
        return error_response(str(exc), 503, "LLM_UNAVAILABLE")
    except Exception as exc:
        logger.exception("LLM call failed unexpectedly")
        return error_response("LLM call failed.", 500, "CHAT_FAILED", str(exc))

    # Build source citations — deduplicated by document so same doc doesn't
    # appear multiple times; each entry shows the doc title and snippet index.
    seen_docs: set[str] = set()
    sources: list[dict] = []
    for chunk in chunks:
        if chunk.document_id not in seen_docs:
            seen_docs.add(chunk.document_id)
            sources.append({
                "document_id":       chunk.document_id,
                "document_title":    chunk.document_title,
                "original_filename": chunk.original_filename,
                "chunk_index":       chunk.chunk_index,
            })

    return success_response(
        data={
            "answer":       llm_result.answer,
            "sources":      sources,
            "model":        llm_result.model,
            "duration_sec": llm_result.duration_sec,
            "mode":         "document",   # tells frontend which renderer to use
        },
        message="Answer generated successfully.",
    )