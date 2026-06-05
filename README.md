<div align="center">
  <h1>🧠 AI Research Assistant</h1>
  <p><strong>Retrieval-Augmented Generation over your PDFs — powered by Groq LLMs</strong></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
    <img src="https://img.shields.io/badge/Flask-3.0-black?logo=flask" alt="Flask">
    <img src="https://img.shields.io/badge/Groq-LLM-ff6600?logo=groq" alt="Groq">
    <img src="https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql" alt="PostgreSQL">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  </p>

  <br>

  <p align="center">
    <b>Upload a PDF → Ask questions → Get grounded answers with source citations</b>
  </p>

  <br>
</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **Smart PDF Ingestion** | Upload PDF/DOCX/TXT files — automatically extracted, chunked, and embedded |
| 🔍 **Semantic Search** | Queries matched by meaning, not keywords (384-dim all-MiniLM-L6-v2 vectors) |
| 🧠 **LLM-Powered Answers** | Grounded generation via Groq's `llama-3.3-70b-versatile` |
| 🔗 **Source Citations** | Every answer traces back to specific chunks in your documents |
| 💬 **Chat Interface** | Clean, dark-mode UI with conversation history (persisted in browser) |
| ⚡ **10x Faster LLM** | Groq LPU inference — answers in <500ms |
| 🔐 **Privacy-First** | Your data stays in your PostgreSQL database. No third-party storage. |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Frontend (HTML/JS)                 │
│              Dark-mode chat + sidebar                │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP (fetch API)
                   ▼
┌─────────────────────────────────────────────────────┐
│              Flask Backend (Python)                   │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  Routes   │  │ Services  │  │      Models       │  │
│  │ (API)     │──│(Business) │──│   (SQLAlchemy)    │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
│       │               │               │              │
│       ▼               ▼               ▼              │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  Groq    │  │Sentence- │  │   PostgreSQL      │  │
│  │  LLM API │  │Transformers│  │   (pgvector)     │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### The RAG Pipeline

```
Upload PDF  ──→  Extract Text  ──→  Chunk (500 chars, 100 overlap)
                                            │
                                            ▼
                                     Embed (384-dim vector)
                                            │
              ┌──────────────────────────────┘
              ▼
User Question  ──→  Embed Query  ──→  Cosine Similarity  ──→  Top-5 Chunks
                                                                   │
                                                                   ▼
                                                            Groq LLM
                                                                   │
                                                                   ▼
                                                          Grounded Answer
```

---

## ⚡ Quick Start

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| PostgreSQL | 14+ |
| Groq API Key | [Get free key](https://console.groq.com) |

### 1. Clone & Setup

```bash
git clone https://github.com/Sahil-TRCAC/ai-research-assistant.git
cd ai-research-assistant

# Backend
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

### 2. Database

```bash
# Create the PostgreSQL database
createdb ai_research_db

# Or via psql:
psql -U postgres -c "CREATE DATABASE ai_research_db;"
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your credentials:
#   - DATABASE_URL=postgresql://user:password@localhost:5432/ai_research_db
#   - GROQ_API_KEY=gsk_your_key_here
```

### 4. Run

```bash
python app.py
```

The server starts at **http://localhost:5000**

### 5. Open the Frontend

Open `frontend/index.html` in your browser — it connects to the backend automatically.

---

## 🧪 Testing

```bash
cd backend
python -m pytest test_smoke.py -v
python -m pytest test_pdf_extraction.py -v
python -m pytest test_chunking.py -v
python -m pytest test_embeddings.py -v
```

---

## 📡 API Reference

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Liveness check |
| GET | `/api/health/db` | Database connectivity |
| GET | `/api/health/detailed` | Full system status |

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/documents/upload` | Upload file (multipart) |
| GET | `/api/documents` | List documents (paginated) |
| GET | `/api/documents/<id>` | Get document metadata |
| PATCH | `/api/documents/<id>` | Update document |
| DELETE | `/api/documents/<id>` | Delete document |
| POST | `/api/documents/<id>/rechunk` | Re-chunk with custom params |
| POST | `/api/documents/<id>/embed` | Generate embeddings |

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Full RAG: question → answer |
| POST | `/api/retrieval/search` | Semantic search (no LLM) |

### Sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/research/sessions` | Create session |
| GET | `/api/research/sessions/<doc_id>` | List sessions |
| GET | `/api/research/sessions/<id>/detail` | Session with messages |

---

## 🛠️ Tech Stack

### Backend

| Category | Technology |
|----------|-----------|
| Framework | Flask 3.0 + Flask-CORS |
| Database | PostgreSQL 16 + SQLAlchemy 2.0 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Groq API (llama-3.3-70b-versatile) |
| PDF | pypdf, python-docx |
| Server | Gunicorn (production) |

### Frontend

| Category | Technology |
|----------|-----------|
| UI | Tailwind CSS (CDN) |
| Icons | Material Symbols |
| Fonts | Inter, JetBrains Mono |
| State | localStorage |

---

## 📁 Project Structure

```
ai-research-assistant/
├── backend/
│   ├── app.py                    # Application factory & entry point
│   ├── config.py                 # Environment configuration
│   ├── models/                   # SQLAlchemy ORM models
│   │   ├── document.py           # Document model
│   │   ├── document_chunk.py     # Chunk model (text + vector)
│   │   └── research_session.py   # Session model
│   ├── routes/                   # Flask blueprints (API endpoints)
│   │   ├── health.py             # Health checks
│   │   ├── documents.py          # Document CRUD + chunk/embed
│   │   ├── research.py           # Session management
│   │   ├── retrieval.py          # Semantic search
│   │   └── chat.py               # RAG pipeline endpoint
│   ├── services/                 # Business logic
│   │   ├── pdf_extractor.py      # PDF text extraction
│   │   ├── chunking_service.py   # Text splitting
│   │   ├── embedding_service.py  # Vector embedding
│   │   ├── retrieval_service.py  # Similarity search
│   │   └── llm_service.py        # Groq integration
│   └── utils/                    # Utilities & helpers
├── frontend/
│   └── index.html                # Single-page chat UI
└── README.md
```

---

## 🚀 Deployment

### Backend on Render

1. Push to GitHub
2. Create a **Web Service** on [Render](https://render.com)
3. Connect your repo
4. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn "app:create_app()" --workers 4 --bind 0.0.0.0:$PORT --timeout 120`
5. Add environment variables from `.env.example`
6. Deploy

### Frontend on Vercel

1. Push to GitHub
2. Import project on [Vercel](https://vercel.com)
3. Set:
   - **Root Directory**: `frontend`
   - **Build Command**: None (static)
   - **Output**: `index.html`
4. Set `AI_RESEARCH_API` env var to your Render backend URL
5. Deploy

---

## ⚠️ Security Notes

- **Never commit `.env`** — it contains API keys and database credentials
- The `.env` file is in `.gitignore` and will not be pushed
- Rotate your Groq API key if it has been exposed
- Use strong passwords for your PostgreSQL database
- In production, use environment variables (not `.env` files)

---

## 📄 License

MIT © Sahil-TRCAC

---

<div align="center">
  <p>Built with Flask, Groq, sentence-transformers, and ❤️</p>
  <p>
    <a href="https://github.com/Sahil-TRCAC/ai-research-assistant/issues">Report Bug</a>
    ·
    <a href="https://github.com/Sahil-TRCAC/ai-research-assistant/issues">Request Feature</a>
  </p>
</div>
