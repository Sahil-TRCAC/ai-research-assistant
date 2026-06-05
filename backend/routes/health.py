"""
routes/health.py
─────────────────
Health-check blueprint.
GET /api/health          — lightweight ping
GET /api/health/db       — verify DB connectivity
GET /api/health/detailed — full system status
"""

import time
import platform
import logging
from datetime import datetime, timezone

from flask import Blueprint
from sqlalchemy import text

from models import db
from utils.response import success_response, error_response

logger = logging.getLogger("ai_research.health")

health_bp = Blueprint("health", __name__, url_prefix="/api/health")

# Record server start time
_START_TIME = time.time()


@health_bp.get("")
def ping():
    """Lightweight liveness probe."""
    return success_response(
        data={"status": "ok"},
        message="Service is running",
    )


@health_bp.get("/db")
def database_check():
    """Verify database connectivity."""
    try:
        db.session.execute(text("SELECT 1"))
        return success_response(
            data={"database": "connected"},
            message="Database is reachable",
        )
    except Exception as exc:
        logger.error("DB health check failed: %s", exc)
        return error_response(
            message="Database unreachable",
            status_code=503,
            error_code="DB_UNAVAILABLE",
            details=str(exc),
        )


@health_bp.get("/detailed")
def detailed_status():
    """Full system status including uptime and environment info."""
    uptime_seconds = round(time.time() - _START_TIME, 2)
    db_ok = False
    try:
        db.session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    status = "healthy" if db_ok else "degraded"

    return success_response(
        data={
            "status": status,
            "uptime_seconds": uptime_seconds,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.system(),
            "services": {
                "database": "ok" if db_ok else "error",
            },
        },
        message=f"System is {status}",
        status_code=200 if db_ok else 207,
    )
