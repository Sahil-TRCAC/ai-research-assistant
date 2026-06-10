"""
routes/research.py
───────────────────
Research session endpoints.

POST   /api/research/sessions                         — create session
GET    /api/research/sessions/<doc_id>                — list sessions for a document
GET    /api/research/sessions/<session_id>/detail     — get session with messages
DELETE /api/research/sessions/<session_id>            — delete session
PATCH  /api/research/sessions/<session_id>/archive    — archive session
POST   /api/research/sessions/<session_id>/messages   — add a message (placeholder for AI)
"""

import logging
from flask import Blueprint, request

from services import research_engine, llm_service, session_service
from utils.response import success_response, error_response
from utils.validators import validate_uuid

logger = logging.getLogger("ai_research.research")

research_bp = Blueprint("research", __name__, url_prefix="/api/research")


# ── Create session ────────────────────────────────────────────────────────────

@research_bp.post("/sessions")
def create_session():
    """
    Create a new research session.

    JSON body
    ---------
    { "document_id": "<uuid>", "title": "optional title" }
    """
    body = request.get_json(silent=True) or {}
    document_id = body.get("document_id", "").strip()
    title = body.get("title", "").strip() or None

    if not document_id:
        return error_response("document_id is required.", 400, "MISSING_FIELD")
    if not validate_uuid(document_id):
        return error_response("Invalid document_id format.", 400, "INVALID_ID")

    try:
        session = session_service.create_session(document_id=document_id, title=title)
        return success_response(
            data=session.to_dict(include_messages=True),
            message="Session created.",
            status_code=201,
        )
    except ValueError as exc:
        return error_response(str(exc), 404, "NOT_FOUND")
    except Exception as exc:
        logger.exception("Failed to create session")
        return error_response("Internal server error.", 500, "SESSION_CREATE_FAILED", str(exc))


# ── Live web research query ───────────────────────────────────────────────────

@research_bp.route("/query", methods=["POST", "OPTIONS"])
def live_research_query():
    """
    Perform live web research for a query, then summarize findings with Groq.

    JSON body
    ---------
    { "query": "...", "max_sources": 4 }
    """
    if request.method == "OPTIONS":
        return "", 204

    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return error_response("Field 'query' is required.", 400, "MISSING_QUERY")

    try:
        max_sources = int(body.get("max_sources", 4))
    except (TypeError, ValueError):
        return error_response("'max_sources' must be an integer.", 400, "INVALID_PARAMS")

    if max_sources < 1 or max_sources > 8:
        return error_response("'max_sources' must be between 1 and 8.", 400, "INVALID_PARAMS")

    logger.info("Live research query: %r max_sources=%d", query[:120], max_sources)

    try:
        research_context = research_engine.perform_live_research(query, max_sources=max_sources)
        llm_result = llm_service.summarize_web_research(
            query=query,
            context_text=research_context.combined_text,
            sources=[{"title": source.title, "url": source.url} for source in research_context.sources],
        )
        return success_response(
            data={
                "query": llm_result.query,
                "answer": llm_result.answer,
                "sources": llm_result.sources,
                "research_sources": [
                    {
                        "title": source.title,
                        "url": source.url,
                        "domain": source.domain,
                        "relevance_score": round(source.relevance_score, 2),
                        "snippets": source.snippets,
                    }
                    for source in research_context.sources
                ],
            },
            message="Live research completed successfully.",
        )
    except ValueError as exc:
        return error_response(str(exc), 400, "INVALID_QUERY")
    except RuntimeError as exc:
        return error_response(str(exc), 503, "RESEARCH_FAILED")
    except Exception as exc:
        logger.exception("Live research failed")
        return error_response("Live research failed.", 500, "RESEARCH_FAILED", str(exc))


# ── List sessions for a document ──────────────────────────────────────────────

@research_bp.get("/sessions/<string:document_id>")
def list_sessions(document_id: str):
    """List all active sessions for a given document."""
    if not validate_uuid(document_id):
        return error_response("Invalid document_id format.", 400, "INVALID_ID")

    sessions = session_service.get_sessions_for_document(document_id)
    return success_response(
        data=[s.to_dict() for s in sessions],
        message=f"{len(sessions)} session(s) found.",
    )


# ── Get session detail ────────────────────────────────────────────────────────

@research_bp.get("/sessions/<string:session_id>/detail")
def get_session(session_id: str):
    """Retrieve a session including its full message history."""
    if not validate_uuid(session_id):
        return error_response("Invalid session_id format.", 400, "INVALID_ID")

    session = session_service.get_session_by_id(session_id)
    if not session:
        return error_response("Session not found.", 404, "NOT_FOUND")

    return success_response(data=session.to_dict(include_messages=True))


# ── Archive session ───────────────────────────────────────────────────────────

@research_bp.patch("/sessions/<string:session_id>/archive")
def archive_session(session_id: str):
    """Archive (soft-delete) a session."""
    if not validate_uuid(session_id):
        return error_response("Invalid session_id format.", 400, "INVALID_ID")

    session = session_service.archive_session(session_id)
    if not session:
        return error_response("Session not found.", 404, "NOT_FOUND")

    return success_response(data=session.to_dict(), message="Session archived.")


# ── Delete session ────────────────────────────────────────────────────────────

@research_bp.delete("/sessions/<string:session_id>")
def delete_session(session_id: str):
    """Hard-delete a session and all its messages."""
    if not validate_uuid(session_id):
        return error_response("Invalid session_id format.", 400, "INVALID_ID")

    deleted = session_service.delete_session(session_id)
    if not deleted:
        return error_response("Session not found.", 404, "NOT_FOUND")

    return success_response(data={"id": session_id}, message="Session deleted.")


# ── Add message (stub — AI logic added later) ─────────────────────────────────

@research_bp.post("/sessions/<string:session_id>/messages")
def add_message(session_id: str):
    """
    Append a user message to a session.
    Returns a placeholder assistant response (AI will be wired in later).

    JSON body
    ---------
    { "content": "What is the main argument of this paper?" }
    """
    if not validate_uuid(session_id):
        return error_response("Invalid session_id format.", 400, "INVALID_ID")

    body = request.get_json(silent=True) or {}
    content = (body.get("content") or "").strip()

    if not content:
        return error_response("Message content is required.", 400, "MISSING_FIELD")

    # Save user message
    session = session_service.add_message(session_id, role="user", content=content)
    if not session:
        return error_response("Session not found.", 404, "NOT_FOUND")

    # ── Placeholder assistant response (will be replaced by AI service) ───────
    placeholder = (
        "AI response placeholder — the AI service will be integrated in a future update."
    )
    session = session_service.add_message(session_id, role="assistant", content=placeholder)

    return success_response(
        data=session.to_dict(include_messages=True),
        message="Message added.",
        status_code=201,
    )
