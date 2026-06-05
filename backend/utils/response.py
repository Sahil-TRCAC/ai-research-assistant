"""
utils/response.py
─────────────────
Standardised JSON response helpers used by every route.
All API responses follow the same envelope structure:

Success:
    { "success": true, "data": {...}, "message": "...", "meta": {...} }

Error:
    { "success": false, "error": { "code": "...", "message": "...", "details": {...} } }
"""

from flask import jsonify
from typing import Any


def success_response(
    data: Any = None,
    message: str = "OK",
    status_code: int = 200,
    meta: dict | None = None,
) -> tuple:
    """Return a standardised success JSON response."""
    body = {
        "success": True,
        "message": message,
        "data": data,
    }
    if meta:
        body["meta"] = meta
    return jsonify(body), status_code


def error_response(
    message: str,
    status_code: int = 400,
    error_code: str = "BAD_REQUEST",
    details: Any = None,
) -> tuple:
    """Return a standardised error JSON response."""
    body = {
        "success": False,
        "error": {
            "code": error_code,
            "message": message,
        },
    }
    if details:
        body["error"]["details"] = details
    return jsonify(body), status_code


def paginated_response(
    data: list,
    page: int,
    per_page: int,
    total: int,
    message: str = "OK",
) -> tuple:
    """Return a paginated success response."""
    return success_response(
        data=data,
        message=message,
        meta={
            "page": page,
            "per_page": per_page,
            "total": total,
            "pages": (total + per_page - 1) // per_page,
            "has_next": (page * per_page) < total,
            "has_prev": page > 1,
        },
    )
