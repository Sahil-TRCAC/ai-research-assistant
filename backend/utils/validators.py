"""
utils/validators.py
────────────────────
Shared request validation utilities.
"""

import os
from werkzeug.datastructures import FileStorage
from flask import current_app


def allowed_file(filename: str) -> bool:
    """Return True if the file extension is in the ALLOWED_EXTENSIONS set."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in current_app.config.get("ALLOWED_EXTENSIONS", set())


def validate_pagination_params(page: int, per_page: int) -> tuple[int, int]:
    """
    Clamp pagination params to safe defaults.

    Returns
    -------
    (page, per_page) with sane bounds applied.
    """
    from config import get_config
    cfg = get_config()
    page = max(1, page)
    per_page = max(1, min(per_page, cfg.MAX_PAGE_SIZE))
    return page, per_page


def validate_uuid(value: str) -> bool:
    """Return True if value is a valid UUID string."""
    import uuid
    try:
        uuid.UUID(str(value))
        return True
    except ValueError:
        return False
