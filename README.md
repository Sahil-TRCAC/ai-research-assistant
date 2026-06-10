<div align="center">
  <div style="padding: 20px;">
    <h1>🧠 AI Research Assistant</h1>
    <p><strong>Your Ultimate Retrieval-Augmented Generation (RAG) & Live Web Research Engine powered by Groq LLMs</strong></p>
  </div>

  <p>
    <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Flask-3.0-black?logo=flask&logoColor=white" alt="Flask">
    <img src="https://img.shields.io/badge/Groq-LLM-ff6600?logo=groq&logoColor=white" alt="Groq">
    <img src="https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL">
    <img src="https://img.shields.io/badge/Tailwind_CSS-38B2AC?logo=tailwind-css&logoColor=white" alt="Tailwind CSS">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </p>

  <br>

  <p align="center">
    <b>Upload your documents or query the web → Get instant, grounded answers with inline citations!</b>
  </p>

  <br>
</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **Smart Document Ingestion** | Upload PDF, DOCX, TXT, CSV, and MD files. Automatically extracted, intelligently chunked, and embedded. |
| 🌐 **Live Web Research** | Ask any question and the AI will search the web (DuckDuckGo), scrape the top live pages, and synthesize the latest findings. |
| 🔍 **Deep Semantic Search** | Local embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`) match your queries by meaning, not just keywords. |
| 🧠 **Lightning Fast LLMs** | Grounded generation via Groq's `llama-3.3-70b-versatile`. Powered by Groq LPUs for <500ms inference. |
| 🔗 **Precision Source Citations** | Every claim traces back to specific document chunks or URLs with clickable Perplexity-style citation pills `[1]`. |
| 💬 **Premium UI/UX** | Stunning glassmorphism dark-mode UI built with Vanilla JS and TailwindCSS, including local conversation history. |
| 🔐 **Privacy-First** | All documents stay entirely within your local PostgreSQL database. No external vector storage providers required. |

---

## 🏗️ Architecture

```mermaid
graph TD
    UI[Frontend UI <br> Vanilla JS + Tailwind] -->|Fetch API| API(Flask Backend)
    
    subgraph "Backend System"
        API -->|Upload| DOCS[Document Service]
        API -->|Search Web| WEB[Research Engine <br> DuckDuckGo + Scraper]
        
        DOCS -->|Text| CHUNK[Chunking Service]
        CHUNK -->|Chunks| EMBED[Embedding Service <br> Local all-MiniLM]
        EMBED -->|Vectors| DB[(PostgreSQL + pgvector)]
        
        API -->|RAG Query| RETRIEVE[Retrieval Service]
        RETRIEVE <--> DB
    end
    
    RETRIEVE -->|Context| LLM[Groq LLM Service]
    WEB -->|Live Context| LLM
    LLM -->|Grounded Answer| UI
```

### The Dual-Mode Pipeline

1. **Document RAG Mode:** Upload a file ──→ Extract & Embed ──→ Semantic Search ──→ LLM Synthesis with Chunk Citations.
2. **Web Research Mode:** Ask a question ──→ Web Search ──→ Scrape Pages ──→ Extract Content ──→ LLM Synthesis with URL Citations.

---

## ⚡ Quick Start

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | |
| PostgreSQL | 14+ | Requires `pgvector` extension |
| Groq API Key | - | [Get a free key here](https://console.groq.com) |

### 1. Clone & Setup

```bash
git clone https://github.com/Sahil-TRCAC/ai-research-assistant.git
cd ai-research-assistant

# Setup Backend Virtual Environment
cd backend
python -m venv venv

# Activate venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Initialization

```bash
# Create the PostgreSQL database (if you have PostgreSQL CLI installed)
createdb ai_research_db

# Or manually via psql:
# psql -U postgres -c "CREATE DATABASE ai_research_db;"
```

### 3. Environment Variables

```bash
cp .env.example .env
```
Edit the `.env` file to include your database URL and API key:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/ai_research_db
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 4. Run the Server

```bash
python app.py
```
*Note: On the first run, the local embedding model (~80 MB) will be downloaded automatically.*

The server will start at **http://localhost:5000**

### 5. Open the Frontend

Simply double-click `frontend/index.html` to open it in your browser. It automatically connects to `http://localhost:5000/api`.

---

## 📡 Core API Reference

| Mode | Method | Endpoint | Description |
|------|--------|----------|-------------|
| **Setup** | POST | `/api/documents/upload` | Upload and extract text from a document |
| **Setup** | POST | `/api/documents/<id>/embed` | Generate vectors for the document chunks |
| **RAG** | POST | `/api/chat` | Ask a question against an uploaded document |
| **Web** | POST | `/api/research/query` | Live web research query |

---

## 🛠️ Tech Stack

### Backend
- **Framework:** Flask 3.0 + Flask-CORS
- **Database:** PostgreSQL 16 + SQLAlchemy 2.0 (`pgvector`)
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **LLM:** Groq API (`llama-3.3-70b-versatile`)
- **Web Scraping:** BeautifulSoup4, DuckDuckGo-Search
- **File Parsing:** `pypdf`, `python-docx`

### Frontend
- **UI Toolkit:** Tailwind CSS (CDN)
- **Icons & Fonts:** Material Symbols, Inter, JetBrains Mono
- **Markdown:** `marked.js`
- **State Management:** `localStorage`

---

## 🚀 Deployment Guides

### Deploying the Backend (Render)
1. Create a **Web Service** on [Render](https://render.com) and link your GitHub repository.
2. Build Command: `pip install -r requirements.txt`
3. Start Command: `gunicorn "app:create_app()" --workers 4 --bind 0.0.0.0:$PORT --timeout 120`
4. Inject your `.env` variables in the Render Dashboard.
5. Make sure you attach a Render PostgreSQL database to the service.

### Deploying the Frontend (Vercel)
1. Import the project into [Vercel](https://vercel.com).
2. Set the **Root Directory** to `frontend`.
3. Clear the Build Command (leave empty).
4. Output Directory: Leave default or `.`
5. Inject the environment variable `AI_RESEARCH_API` pointing to your deployed backend URL (e.g. `https://your-backend.onrender.com/api`).

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">
  <p>Built for speed, accuracy, and beautiful user experiences. 🚀</p>
  <p>
    <a href="https://github.com/Sahil-TRCAC/ai-research-assistant/issues">Report Bug</a>
    ·
    <a href="https://github.com/Sahil-TRCAC/ai-research-assistant/issues">Request Feature</a>
  </p>
</div>
