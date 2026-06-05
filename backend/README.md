# AI Research Assistant — Backend

A production-ready Flask backend providing the API foundation for the AI Research Assistant.

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | Flask 3.0 |
| Database | PostgreSQL + SQLAlchemy 2.0 |
| CORS | Flask-CORS |
| Config | python-dotenv |
| Server | Gunicorn (production) |

## Project Structure

```
backend/
├── app.py                    # Application factory & entry point
├── config.py                 # Environment-based configuration
├── requirements.txt
├── .env.example              # Copy → .env and fill values
├── routes/
│   ├── health.py             # GET /api/health/*
│   ├── documents.py          # CRUD /api/documents
│   └── research.py           # Sessions /api/research/sessions
├── models/
│   ├── document.py           # Document ORM model
│   └── research_session.py   # ResearchSession ORM model
├── services/
│   ├── document_service.py   # File I/O + DB logic
│   └── session_service.py    # Session + message logic
├── utils/
│   ├── logger.py             # Rotating file + console logger
│   ├── response.py           # Standardised JSON envelopes
│   └── validators.py         # Pagination, UUID, file-type checks
└── uploads/                  # Uploaded documents (not committed)
```

## Quick Start

### 1. Create a virtual environment

```bash
cd backend
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS / Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
copy .env.example .env
# Edit .env with your PostgreSQL credentials and secret key
```

### 4. Create the PostgreSQL database

```sql
CREATE DATABASE ai_research_db;
```

### 5. Run the development server

```bash
python app.py
```

Server starts at **http://localhost:5000**

---

## API Reference

### Health

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Liveness ping |
| GET | `/api/health/db` | Database connectivity check |
| GET | `/api/health/detailed` | Full system status + uptime |

### Documents

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/documents/upload` | Upload a document (multipart/form-data) |
| GET | `/api/documents` | List documents (paginated) |
| GET | `/api/documents/<id>` | Get single document |
| PATCH | `/api/documents/<id>` | Update title / description / tags |
| DELETE | `/api/documents/<id>` | Delete document + file |

### Research Sessions

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/research/sessions` | Create session |
| GET | `/api/research/sessions/<doc_id>` | List sessions for a document |
| GET | `/api/research/sessions/<id>/detail` | Get session with messages |
| PATCH | `/api/research/sessions/<id>/archive` | Archive session |
| DELETE | `/api/research/sessions/<id>` | Delete session |
| POST | `/api/research/sessions/<id>/messages` | Add message (AI stub) |

### Response Envelope

All responses follow a consistent structure:

**Success**
```json
{ "success": true, "message": "OK", "data": {}, "meta": {} }
```

**Error**
```json
{ "success": false, "error": { "code": "NOT_FOUND", "message": "..." } }
```

---

## Production Deployment

```bash
gunicorn "app:create_app()" \
  --workers 4 \
  --bind 0.0.0.0:5000 \
  --timeout 120 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FLASK_ENV` | `development` | Environment name |
| `SECRET_KEY` | *(required)* | Flask secret key |
| `DATABASE_URL` | `postgresql://...` | PostgreSQL connection string |
| `CORS_ORIGINS` | `http://localhost:3000` | Allowed CORS origins (comma-separated) |
| `UPLOAD_FOLDER` | `uploads` | Directory for uploaded files |
| `MAX_CONTENT_LENGTH` | `52428800` | Max upload size in bytes (50 MB) |
| `LOG_LEVEL` | `DEBUG` | Logging level |
| `LOG_FILE` | `logs/app.log` | Log file path |
