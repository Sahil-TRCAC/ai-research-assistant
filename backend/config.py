"""
config.py
─────────
Centralised configuration for the AI Research Assistant backend.
All settings are loaded from environment variables (via .env).
"""

import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()


class BaseConfig:
    """Shared defaults across all environments."""

    # ── Flask core ──────────────────────────────────────────────────────────
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-please-change")
    TESTING: bool = False

    # ── Database ────────────────────────────────────────────────────────────
    SQLALCHEMY_DATABASE_URI: str = os.getenv(
        "DATABASE_URL", "postgresql://postgres:password@localhost:5432/ai_research_db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
    SQLALCHEMY_ENGINE_OPTIONS: dict = {
        "pool_pre_ping": True,  # actual options set in app._init_db() per DB type
    }

    # ── File uploads ────────────────────────────────────────────────────────
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "uploads")
    MAX_CONTENT_LENGTH: int = int(os.getenv("MAX_CONTENT_LENGTH", 50 * 1024 * 1024))  # 50 MB
    ALLOWED_EXTENSIONS: set = {"pdf", "docx", "txt", "md", "csv", "xlsx"}

    # ── CORS ────────────────────────────────────────────────────────────────
    CORS_ORIGINS: list = [
        o.strip()
        for o in os.getenv(
            "CORS_ORIGINS",
            # Default: local dev origins + the deployed Render frontend.
            # Override via CORS_ORIGINS env var (comma-separated) on Render.
            "http://localhost:3000,"
            "http://localhost:5173,"
            "http://localhost:8000,"
            "http://127.0.0.1:8000,"
            "https://ai-research-frontend-qyxc.onrender.com",
        ).split(",")
        if o.strip()
    ]
    # Optional regex for extra origins (e.g. preview deployments).
    # Set CORS_ORIGINS_REGEX env var if you need dynamic origin matching.
    CORS_ORIGINS_REGEX: str = os.getenv("CORS_ORIGINS_REGEX", "")

    # ── Logging ─────────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/app.log")

    # ── Redis / Celery (future) ──────────────────────────────────────────────
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL: str = REDIS_URL
    CELERY_RESULT_BACKEND: str = REDIS_URL

    # ── Pagination defaults ──────────────────────────────────────────────────
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100


class DevelopmentConfig(BaseConfig):
    FLASK_ENV = "development"
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class TestingConfig(BaseConfig):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    LOG_LEVEL = "WARNING"


class ProductionConfig(BaseConfig):
    DEBUG = False
    FLASK_ENV = "production"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")

    # In production the SECRET_KEY MUST be set via env
    @classmethod
    def validate(cls) -> None:
        if cls.SECRET_KEY == "dev-secret-key-please-change":
            raise RuntimeError(
                "SECRET_KEY must be set to a secure random value in production."
            )


# ── Mapping ───────────────────────────────────────────────────────────────────
_config_map = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}


def get_config() -> BaseConfig:
    """Return the correct config class based on FLASK_ENV."""
    env = os.getenv("FLASK_ENV", "development").lower()
    cfg = _config_map.get(env, DevelopmentConfig)
    if env == "production" and hasattr(cfg, "validate"):
        cfg.validate()
    return cfg
