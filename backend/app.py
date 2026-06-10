"""
app.py
───────
Application factory for the AI Research Assistant Flask backend.

Usage
─────
  python app.py                    # development server
  gunicorn "app:create_app()"      # production

Environment
───────────
Copy .env.example → .env and fill in your values before running.
"""

import os
import logging
from flask import Flask, jsonify
from flask_cors import CORS

from config import get_config
from utils.logger import configure_logger


def create_app() -> Flask:
    """
    Flask application factory.

    Returns a fully configured Flask app with:
    - Database (SQLAlchemy + PostgreSQL)
    - CORS
    - All blueprints registered
    - Global error handlers
    - Structured logging
    """
    cfg = get_config()

    # ── Logging (configure before anything else) ─────────────────────────────
    configure_logger(log_level=cfg.LOG_LEVEL, log_file=cfg.LOG_FILE)
    logger = logging.getLogger("ai_research")
    logger.info("Starting AI Research Assistant — env=%s", os.getenv("FLASK_ENV", "development"))

    # ── Flask app ─────────────────────────────────────────────────────────────
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object(cfg)

    # ── Ensure upload folder exists ───────────────────────────────────────────
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # ── CORS ──────────────────────────────────────────────────────────────────
    is_prod = os.getenv("FLASK_ENV", "development").lower() == "production"
    if is_prod:
        origins = list(app.config["CORS_ORIGINS"])
    else:
        origins = "*"

    CORS(
        app,
        resources={r"/api/*": {"origins": origins}},
        supports_credentials=False,  # frontend sends no cookies / auth headers
        allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    logger.info("CORS enabled for origins: %s", origins)

    # ── Database ──────────────────────────────────────────────────────────────
    _init_db(app)

    # ── Blueprints ────────────────────────────────────────────────────────────
    _register_blueprints(app)

    # ── Error handlers ────────────────────────────────────────────────────────
    _register_error_handlers(app)

    logger.info("App initialised successfully.")
    return app


# ── Private helpers ───────────────────────────────────────────────────────────

def _init_db(app: Flask) -> None:
    """Initialise SQLAlchemy and create all tables if they don't exist."""
    from models import db

    # Set engine options based on database type
    db_url = app.config.get("SQLALCHEMY_DATABASE_URI", "")
    if db_url.startswith("sqlite"):
        app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True}
    else:
        app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
            "pool_size": int(os.getenv("DB_POOL_SIZE", 10)),
            "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", 20)),
            "pool_pre_ping": True,
            "pool_recycle": 300,
        }

    db.init_app(app)

    with app.app_context():
        try:
            from models import Document, DocumentChunk, ResearchSession  # noqa: F401
            db.create_all()
            logging.getLogger("ai_research").info("Database tables verified / created.")
        except Exception as exc:
            logging.getLogger("ai_research").warning(
                "Database not available at startup (will retry on first request): %s", exc
            )


def _register_blueprints(app: Flask) -> None:
    """Register all route blueprints."""
    from routes.health import health_bp
    from routes.documents import documents_bp
    from routes.research import research_bp
    from routes.retrieval import retrieval_bp
    from routes.chat import chat_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(documents_bp)
    app.register_blueprint(research_bp)
    app.register_blueprint(retrieval_bp)
    app.register_blueprint(chat_bp)

    logger = logging.getLogger("ai_research")
    logger.info("Blueprints registered: health, documents, research, retrieval, chat")


def _register_error_handlers(app: Flask) -> None:
    """Global error handlers for consistent JSON error responses."""
    logger = logging.getLogger("ai_research")

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"success": False, "error": {"code": "BAD_REQUEST", "message": str(e)}}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"success": False, "error": {"code": "NOT_FOUND", "message": "The requested resource was not found."}}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"success": False, "error": {"code": "METHOD_NOT_ALLOWED", "message": str(e)}}), 405

    @app.errorhandler(413)
    def request_entity_too_large(e):
        return jsonify({"success": False, "error": {"code": "FILE_TOO_LARGE", "message": "Uploaded file exceeds the maximum allowed size."}}), 413

    @app.errorhandler(500)
    def internal_server_error(e):
        logger.exception("Unhandled 500 error")
        return jsonify({"success": False, "error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred."}}), 500

    @app.errorhandler(Exception)
    def handle_unexpected(e):
        logger.exception("Unhandled exception: %s", e)
        return jsonify({"success": False, "error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred."}}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    flask_app = create_app()
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() in ("1", "true", "yes")
    flask_app.run(host="0.0.0.0", port=port, debug=debug)
