"""
routes/documents.py
────────────────────
RESTful endpoints for document management + text extraction.

POST   /api/documents/upload                    — upload & extract
GET    /api/documents                           — list (paginated)
GET    /api/documents/<id>                      — metadata + preview
GET    /api/documents/<id>/content              — full extracted text
POST   /api/documents/<id>/re-extract           — retry extraction
PATCH  /api/documents/<id>                      — update title/desc/tags
DELETE /api/documents/<id>                      — delete document + file
"""

import logging
from flask import Blueprint, request

from services import document_service
from services import chunking_service
from utils.response import success_response, error_response, paginated_response
from utils.validators import validate_pagination_params, validate_uuid

logger = logging.getLogger("ai_research.documents")

documents_bp = Blueprint("documents", __name__, url_prefix="/api/documents")


# ── Upload + Extract ───────────────────────────────────────────────────────────

@documents_bp.post("/upload")
def upload_document():
    """
    Upload a document, extract its text, and persist everything.

    Form fields
    -----------
    file        : (required) multipart file — PDF, DOCX, TXT, MD, CSV
    title       : (optional) human-readable title
    description : (optional) free-text description
    tags        : (optional) comma-separated tag string

    Response includes extraction stats (word_count, page_count, is_scanned).
    Full extracted_text is NOT returned here — use GET /<id>/content.
    """
    if "file" not in request.files:
        return error_response("No file part in the request.", 400, "MISSING_FILE")

    file = request.files["file"]
    if file.filename == "":
        return error_response("No file selected.", 400, "EMPTY_FILENAME")

    title       = request.form.get("title", "").strip() or None
    description = request.form.get("description", "").strip() or None
    raw_tags    = request.form.get("tags", "")
    tags        = [t.strip() for t in raw_tags.split(",") if t.strip()]

    try:
        doc = document_service.upload_document(
            file=file,
            title=title,
            description=description,
            tags=tags,
        )
        # Return metadata + preview (not full text — that can be megabytes)
        return success_response(
            data=doc.to_dict(include_preview=True),
            message="Document uploaded and text extracted successfully."
                    if doc.status == "ready"
                    else "Document uploaded but text extraction failed.",
            status_code=201,
        )
    except ValueError as exc:
        return error_response(str(exc), 400, "INVALID_FILE")
    except Exception as exc:
        logger.exception("Unexpected error during upload")
        return error_response("Internal server error.", 500, "UPLOAD_FAILED", str(exc))


# ── List ──────────────────────────────────────────────────────────────────────

@documents_bp.get("")
def list_documents():
    """
    Paginated document list.

    Query params
    ------------
    page     : int  (default 1)
    per_page : int  (default 20, max 100)
    status   : str  filter by status (ready | error | processing)
    """
    try:
        page     = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 20))
    except ValueError:
        return error_response("page and per_page must be integers.", 400, "INVALID_PARAMS")

    page, per_page = validate_pagination_params(page, per_page)
    status_filter  = request.args.get("status") or None

    docs, total = document_service.get_all_documents(
        page=page, per_page=per_page, status=status_filter
    )
    return paginated_response(
        data=[d.to_dict(include_preview=False) for d in docs],
        page=page,
        per_page=per_page,
        total=total,
    )


# ── Get metadata ──────────────────────────────────────────────────────────────

@documents_bp.get("/<string:document_id>")
def get_document(document_id: str):
    """
    Fetch document metadata and content preview (first 500 chars).
    Does NOT return the full extracted text — use /content for that.
    """
    if not validate_uuid(document_id):
        return error_response("Invalid document ID format.", 400, "INVALID_ID")

    doc = document_service.get_document_by_id(document_id)
    if not doc:
        return error_response("Document not found.", 404, "NOT_FOUND")

    return success_response(data=doc.to_dict(include_preview=True))


# ── Get full extracted text ───────────────────────────────────────────────────

@documents_bp.get("/<string:document_id>/content")
def get_document_content(document_id: str):
    """
    Return the full extracted text of a document.

    This endpoint exists separately from GET /<id> because extracted_text
    can be several megabytes for long PDFs — clients should only request
    it when they actually need the content (e.g., for RAG / AI queries).

    Response
    --------
    {
      "id": "...",
      "title": "...",
      "extracted_text": "...full text...",
      "word_count": 12345,
      "char_count": 67890,
      "page_count": 42,
      "is_scanned": false,
      "extraction_error": null
    }
    """
    if not validate_uuid(document_id):
        return error_response("Invalid document ID format.", 400, "INVALID_ID")

    doc = document_service.get_document_text(document_id)
    if not doc:
        return error_response("Document not found.", 404, "NOT_FOUND")

    if doc.status == "error" or doc.is_scanned:
        return error_response(
            message=doc.extraction_error or "Text extraction failed for this document.",
            status_code=422,
            error_code="EXTRACTION_FAILED",
            details={
                "id": doc.id,
                "is_scanned": doc.is_scanned,
                "page_count": doc.page_count,
            },
        )

    if not doc.extracted_text:
        return error_response(
            "No extracted text available for this document.",
            404,
            "NO_CONTENT",
        )

    return success_response(
        data={
            "id": doc.id,
            "title": doc.title,
            "file_type": doc.file_type,
            "extracted_text": doc.extracted_text,
            "word_count": doc.word_count,
            "char_count": doc.char_count,
            "page_count": doc.page_count,
            "is_scanned": doc.is_scanned,
            "extraction_error": doc.extraction_error,
        },
        message="Extracted text retrieved successfully.",
    )


# ── Re-extract ────────────────────────────────────────────────────────────────

@documents_bp.post("/<string:document_id>/re-extract")
def re_extract_document(document_id: str):
    """
    Retry text extraction on an existing document.
    Useful when a document was previously in error/scanned state.
    """
    if not validate_uuid(document_id):
        return error_response("Invalid document ID format.", 400, "INVALID_ID")

    doc = document_service.re_extract_document(document_id)
    if not doc:
        return error_response("Document not found.", 404, "NOT_FOUND")

    return success_response(
        data=doc.to_dict(include_preview=True),
        message="Re-extraction complete." if doc.status == "ready" else "Re-extraction failed.",
        status_code=200,
    )


# ── Update metadata ───────────────────────────────────────────────────────────

@documents_bp.patch("/<string:document_id>")
def update_document(document_id: str):
    """
    Update mutable metadata: title, description, tags.

    JSON body (all optional)
    ────────────────────────
    { "title": "...", "description": "...", "tags": ["tag1"] }
    """
    if not validate_uuid(document_id):
        return error_response("Invalid document ID format.", 400, "INVALID_ID")

    body    = request.get_json(silent=True) or {}
    allowed = {"title", "description", "tags"}
    updates = {k: v for k, v in body.items() if k in allowed}

    if not updates:
        return error_response(
            f"Provide at least one of: {', '.join(allowed)}", 400, "MISSING_FIELDS"
        )

    doc = document_service.update_document_metadata(document_id, **updates)
    if not doc:
        return error_response("Document not found.", 404, "NOT_FOUND")

    return success_response(data=doc.to_dict(), message="Document updated.")


# ── Delete ────────────────────────────────────────────────────────────────────

@documents_bp.delete("/<string:document_id>")
def delete_document(document_id: str):
    """Delete document record and remove file from disk."""
    if not validate_uuid(document_id):
        return error_response("Invalid document ID format.", 400, "INVALID_ID")

    deleted = document_service.delete_document(document_id)
    if not deleted:
        return error_response("Document not found.", 404, "NOT_FOUND")

    return success_response(data={"id": document_id}, message="Document deleted.")


# ── Get chunks ─────────────────────────────────────────────────────────────────

@documents_bp.get("/<string:document_id>/chunks")
def get_document_chunks(document_id: str):
    """
    Return paginated text chunks for a document.

    Each chunk is a ~500-character slice of the extracted_text with
    100-character overlap with its neighbours — ready for RAG embedding.

    Query params
    ------------
    page     : int (default 1)
    per_page : int (default 50, max 200)

    Response fields per chunk
    ─────────────────────────
    chunk_index : zero-based position in the document
    chunk_text  : the text slice
    char_count  : length of this chunk in characters
    start_char  : offset in original extracted_text where this chunk begins
    end_char    : offset where this chunk ends
    is_embedded : whether this chunk has been vectorised yet
    """
    if not validate_uuid(document_id):
        return error_response("Invalid document ID format.", 400, "INVALID_ID")

    # Verify document exists
    doc = document_service.get_document_by_id(document_id)
    if not doc:
        return error_response("Document not found.", 404, "NOT_FOUND")

    if not doc.extracted_text:
        return error_response(
            "Document has no extracted text. Upload a PDF first.",
            422,
            "NO_CONTENT",
        )

    try:
        page     = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 50))
    except ValueError:
        return error_response("page and per_page must be integers.", 400, "INVALID_PARAMS")

    page     = max(1, page)
    per_page = max(1, min(per_page, 200))

    chunks, total = chunking_service.get_chunks_for_document(
        document_id, page=page, per_page=per_page
    )

    # If no chunks exist yet, generate them on-the-fly
    if total == 0 and doc.extracted_text:
        logger.info("No chunks found for doc=%s — generating on-the-fly", document_id)
        try:
            chunking_service.chunk_document(document_id)
            chunks, total = chunking_service.get_chunks_for_document(
                document_id, page=page, per_page=per_page
            )
        except Exception as exc:
            logger.exception("On-the-fly chunking failed for doc=%s", document_id)
            return error_response(
                "Chunking failed.", 500, "CHUNKING_FAILED", str(exc)
            )

    return paginated_response(
        data=[c.to_dict(include_text=True) for c in chunks],
        page=page,
        per_page=per_page,
        total=total,
        message=f"{total} chunk(s) available.",
    )


# ── On-demand rechunk ────────────────────────────────────────────────────────────

@documents_bp.post("/<string:document_id>/rechunk")
def rechunk_document(document_id: str):
    """
    Re-run chunking with custom parameters.

    JSON body (all optional — defaults used if omitted)
    ────────────────────────────────────────────────────
    {
      "chunk_size": 500,
      "overlap":    100
    }
    """
    if not validate_uuid(document_id):
        return error_response("Invalid document ID format.", 400, "INVALID_ID")

    doc = document_service.get_document_by_id(document_id)
    if not doc:
        return error_response("Document not found.", 404, "NOT_FOUND")

    if not doc.extracted_text:
        return error_response(
            "Document has no extracted text — cannot chunk.", 422, "NO_CONTENT"
        )

    body       = request.get_json(silent=True) or {}
    chunk_size = int(body.get("chunk_size", 500))
    overlap    = int(body.get("overlap", 100))

    if chunk_size < 100:
        return error_response("chunk_size must be at least 100.", 400, "INVALID_PARAMS")
    if overlap >= chunk_size:
        return error_response("overlap must be less than chunk_size.", 400, "INVALID_PARAMS")

    try:
        chunks = chunking_service.chunk_document(
            document_id, chunk_size=chunk_size, overlap=overlap, replace_existing=True
        )
    except ValueError as exc:
        return error_response(str(exc), 404, "NOT_FOUND")
    except Exception as exc:
        logger.exception("Rechunking failed for doc=%s", document_id)
        return error_response("Rechunking failed.", 500, "CHUNKING_FAILED", str(exc))

    return success_response(
        data={
            "document_id": document_id,
            "chunk_count":  len(chunks),
            "chunk_size":   chunk_size,
            "overlap":      overlap,
            "sample":       chunks[0].to_dict() if chunks else None,
        },
        message=f"Document rechunked into {len(chunks)} chunks.",
    )


# ── Generate embeddings ───────────────────────────────────────────────────────

@documents_bp.post("/<string:document_id>/embed")
def embed_document(document_id: str):
    """
    Generate vector embeddings for all unembedded chunks of a document.

    Uses sentence-transformers model: all-MiniLM-L6-v2
    Each chunk_text is converted to a 384-dimensional float vector
    and stored as JSON in the embedding_vector column.

    JSON body (all optional)
    ────────────────────────
    {
      "batch_size": 32,      # chunks per model.encode() call (default 32)
      "re_embed":   false    # re-embed already-embedded chunks (default false)
    }

    Response
    --------
    {
      "total_chunks":  6,
      "embedded":      6,
      "skipped":       0,
      "error_count":   0,
      "duration_sec":  1.23,
      "model":         "all-MiniLM-L6-v2",
      "dims":          384
    }

    Note: On first call the model (~80 MB) is downloaded and cached.
    Subsequent calls are fast.
    """
    if not validate_uuid(document_id):
        return error_response("Invalid document ID format.", 400, "INVALID_ID")

    doc = document_service.get_document_by_id(document_id)
    if not doc:
        return error_response("Document not found.", 404, "NOT_FOUND")

    if not doc.extracted_text:
        return error_response(
            "Document has no extracted text. Upload and extract first.",
            422, "NO_CONTENT",
        )

    # Check that chunks exist
    from models.document_chunk import DocumentChunk
    chunk_count = DocumentChunk.query.filter_by(document_id=document_id).count()
    if chunk_count == 0:
        return error_response(
            "No chunks found. Run /rechunk or re-upload the document first.",
            422, "NO_CHUNKS",
        )

    body       = request.get_json(silent=True) or {}
    batch_size = int(body.get("batch_size", 32))
    re_embed   = bool(body.get("re_embed",  False))

    if batch_size < 1 or batch_size > 512:
        return error_response("batch_size must be between 1 and 512.", 400, "INVALID_PARAMS")

    try:
        from services import embedding_service
        result = embedding_service.embed_document_chunks(
            document_id, batch_size=batch_size, re_embed=re_embed
        )
    except ValueError as exc:
        return error_response(str(exc), 404, "NOT_FOUND")
    except RuntimeError as exc:
        return error_response(str(exc), 503, "MODEL_UNAVAILABLE")
    except Exception as exc:
        logger.exception("Embedding failed for doc=%s", document_id)
        return error_response("Embedding failed.", 500, "EMBED_FAILED", str(exc))

    status_code = 200 if result.success else 207  # 207 = partial success
    return success_response(
        data=result.to_dict(),
        message=(
            f"Embedding complete: {result.embedded} chunks embedded."
            if result.success
            else f"Partial embedding: {result.embedded} embedded, "
                 f"{len(result.errors)} errors."
        ),
        status_code=status_code,
    )


# ── Embedding status ──────────────────────────────────────────────────────────

@documents_bp.get("/<string:document_id>/embedding-status")
def get_embedding_status(document_id: str):
    """
    Return the current embedding progress for a document.

    Response
    --------
    {
      "document_id":       "...",
      "document_title":    "...",
      "total_chunks":      6,
      "embedded_chunks":   4,
      "pending_chunks":    2,
      "percent_complete":  66.7,
      "is_fully_embedded": false,
      "model":             "all-MiniLM-L6-v2",
      "dims":              384
    }
    """
    if not validate_uuid(document_id):
        return error_response("Invalid document ID format.", 400, "INVALID_ID")

    try:
        from services import embedding_service
        status = embedding_service.get_embedding_status(document_id)
    except ValueError as exc:
        return error_response(str(exc), 404, "NOT_FOUND")
    except Exception as exc:
        logger.exception("Failed to get embedding status for doc=%s", document_id)
        return error_response("Could not retrieve status.", 500, "STATUS_ERROR", str(exc))

    return success_response(
        data=status,
        message="Fully embedded." if status["is_fully_embedded"] else
                f"{status['pending_chunks']} chunk(s) still pending.",
    )
