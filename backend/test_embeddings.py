"""
test_embeddings.py
───────────────────
Tests for the embedding service and API endpoints.

Strategy: sentence-transformers is mocked for unit tests so the real
~80MB model isn't needed. The HTTP route tests also use the mock.
A separate live-model section at the end runs only if the env var
RUN_LIVE_EMBEDDING_TEST=1 is set — useful to verify the real model.
"""

import os, sys, json, tempfile

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["FLASK_ENV"]    = "development"
os.environ["SECRET_KEY"]   = "test-key"
os.environ["CORS_ORIGINS"] = "http://localhost:3000"

PASS, FAIL = "PASS", "FAIL"
results = []

def check(label, condition, detail=""):
    tag = PASS if condition else FAIL
    results.append((tag, label, detail))
    print(f"  {tag}  {label}" + (f"  [{detail}]" if detail else ""))


# ── Mock sentence-transformers ────────────────────────────────────────────────
import numpy as np
from unittest.mock import MagicMock, patch

DIMS = 384

def make_mock_model():
    """Return a mock SentenceTransformer that produces deterministic unit vectors."""
    mock = MagicMock()
    def fake_encode(texts, **kwargs):
        vecs = np.random.default_rng(seed=42).random((len(texts), DIMS)).astype("float32")
        # Normalise to unit vectors
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms
    mock.encode.side_effect = fake_encode
    mock.get_sentence_embedding_dimension.return_value = DIMS
    return mock


# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] EmbeddingResult dataclass")
# ─────────────────────────────────────────────────────────────────────────────
from services.embedding_service import EmbeddingResult

r = EmbeddingResult(document_id="abc", total_chunks=5, embedded=5, skipped=0)
check("success=True when embedded+skipped==total",     r.success)
check("to_dict() returns dict",                        isinstance(r.to_dict(), dict))
check("to_dict() has all required keys",
      all(k in r.to_dict() for k in [
          "document_id","total_chunks","embedded","skipped",
          "error_count","duration_sec","model","dims","success"
      ]))

r_fail = EmbeddingResult(document_id="abc", total_chunks=5, embedded=3, skipped=0,
                         errors=[("id1","err")])
check("success=False when errors present",             not r_fail.success)
check("to_dict errors list populated",                 len(r_fail.to_dict()["errors"]) == 1)

r_partial = EmbeddingResult(document_id="abc", total_chunks=5, embedded=3, skipped=2)
check("success=True when embedded+skipped==total (partial)", r_partial.success)


# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] embed_single_text (mocked model)")
# ─────────────────────────────────────────────────────────────────────────────
import services.embedding_service as emb_svc

with patch("services.embedding_service._get_model", return_value=make_mock_model()):
    vec = emb_svc.embed_single_text("Hello world")
    check("embed_single_text returns list",         isinstance(vec, list))
    check("embed_single_text returns 384 dims",     len(vec) == DIMS, f"got {len(vec)}")
    check("embed_single_text returns floats",       all(isinstance(v, float) for v in vec[:5]))
    # Unit vector check: norm ≈ 1.0
    norm = sum(v**2 for v in vec) ** 0.5
    check("embed_single_text vector is unit-norm",  abs(norm - 1.0) < 0.01, f"norm={norm:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] embed_document_chunks — DB integration (mocked model)")
# ─────────────────────────────────────────────────────────────────────────────
from app import create_app
app = create_app()

with app.app_context():
    from models import db, Document, DocumentChunk
    from services.chunking_service import chunk_document
    from services.embedding_service import embed_document_chunks, get_embedding_status

    # Seed a document with extracted_text
    text = ("The quick brown fox jumps over the lazy dog. " * 40).strip()
    doc = Document(
        title="Embed Test",
        original_filename="embed_test.pdf",
        stored_filename="embed_test_stored.pdf",
        file_path="/tmp/embed_test.pdf",
        file_size=1000, file_type="pdf",
        status="ready", extracted_text=text,
        word_count=len(text.split()), char_count=len(text), tags=[],
    )
    db.session.add(doc)
    db.session.commit()
    doc_id = doc.id

    # Create chunks
    chunks = chunk_document(doc_id)
    total_chunks = len(chunks)
    check("chunks created for doc",    total_chunks >= 1, f"got {total_chunks}")

    # Verify all is_embedded=False initially
    pending = DocumentChunk.query.filter_by(document_id=doc_id, is_embedded=False).count()
    check("all chunks start un-embedded", pending == total_chunks)

    # status BEFORE embedding
    status = get_embedding_status(doc_id)
    check("status.total_chunks correct",      status["total_chunks"] == total_chunks)
    check("status.embedded_chunks == 0",      status["embedded_chunks"] == 0)
    check("status.pending_chunks == total",   status["pending_chunks"] == total_chunks)
    check("status.percent_complete == 0",     status["percent_complete"] == 0.0)
    check("status.is_fully_embedded False",   not status["is_fully_embedded"])
    check("status.model == all-MiniLM-L6-v2", status["model"] == "all-MiniLM-L6-v2")

    # Embed with mocked model
    with patch("services.embedding_service._get_model", return_value=make_mock_model()):
        result = embed_document_chunks(doc_id, batch_size=4)

    check("result.embedded == total_chunks",  result.embedded == total_chunks,
          f"embedded={result.embedded} total={total_chunks}")
    check("result.skipped == 0",              result.skipped == 0)
    check("result.errors == []",              result.errors == [])
    check("result.success == True",           result.success)
    check("result.duration_sec > 0",          result.duration_sec > 0)
    check("result.dims == 384",               result.dims == DIMS)

    # Verify DB state
    embedded_count = DocumentChunk.query.filter_by(
        document_id=doc_id, is_embedded=True
    ).count()
    check("DB: all chunks marked is_embedded=True", embedded_count == total_chunks,
          f"embedded={embedded_count}")

    # Verify vector stored
    sample_chunk = DocumentChunk.query.filter_by(document_id=doc_id).first()
    check("chunk has embedding_vector set",      sample_chunk.embedding_vector is not None)
    check("chunk has embedding_model set",       sample_chunk.embedding_model == "all-MiniLM-L6-v2")
    vec_data = json.loads(sample_chunk.embedding_vector)
    check("embedding_vector is list of 384",     len(vec_data) == DIMS, f"got {len(vec_data)}")
    check("embedding_vector contains floats",    all(isinstance(v, float) for v in vec_data[:5]))

    # to_dict without vector
    d = sample_chunk.to_dict(include_text=True, include_vector=False)
    check("to_dict: no embedding_vector without flag", "embedding_vector" not in d)
    check("to_dict: embedding_model present",          "embedding_model" in d)
    check("to_dict: is_embedded = True",               d["is_embedded"] is True)

    # to_dict WITH vector
    d_vec = sample_chunk.to_dict(include_text=True, include_vector=True)
    check("to_dict: embedding_vector present with flag", "embedding_vector" in d_vec)
    check("to_dict: vector has 384 dims",                len(d_vec["embedding_vector"]) == DIMS)

    # status AFTER embedding
    status2 = get_embedding_status(doc_id)
    check("status after: embedded_chunks == total", status2["embedded_chunks"] == total_chunks)
    check("status after: pending_chunks == 0",      status2["pending_chunks"] == 0)
    check("status after: percent_complete == 100",  status2["percent_complete"] == 100.0)
    check("status after: is_fully_embedded True",   status2["is_fully_embedded"])

    # re_embed=False skips already-embedded chunks
    with patch("services.embedding_service._get_model", return_value=make_mock_model()):
        result2 = embed_document_chunks(doc_id, re_embed=False)
    check("re_embed=False: skipped == total",  result2.skipped == total_chunks,
          f"skipped={result2.skipped}")
    check("re_embed=False: embedded == 0",     result2.embedded == 0)

    # re_embed=True re-processes all chunks
    with patch("services.embedding_service._get_model", return_value=make_mock_model()):
        result3 = embed_document_chunks(doc_id, re_embed=True)
    check("re_embed=True: embedded == total",  result3.embedded == total_chunks)
    check("re_embed=True: skipped == 0",       result3.skipped == 0)

    # Error: document not found
    try:
        embed_document_chunks("00000000-0000-0000-0000-000000000000")
        check("ValueError on missing doc", False)
    except ValueError:
        check("ValueError raised for missing doc", True)


# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] HTTP route tests — /embed and /embedding-status")
# ─────────────────────────────────────────────────────────────────────────────
def make_test_pdf(path, text):
    stream = text.encode("latin-1", errors="replace")
    sd = b"BT /F1 12 Tf 72 720 Td (" + stream + b") Tj ET"
    c = (
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
        b"   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length " + str(len(sd)).encode() + b" >>\nstream\n"
        + sd + b"\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n"
        b"0000000115 00000 n \n0000000266 00000 n \n0000000360 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n430\n%%EOF\n"
    )
    with open(path, "wb") as f:
        f.write(c)

client = app.test_client()

# Upload a PDF to get a real doc_id
with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
    pdf_path = tmp.name
make_test_pdf(pdf_path, "The quick brown fox jumps over the lazy dog. " * 50)

http_doc_id = None
with open(pdf_path, "rb") as f:
    r = client.post(
        "/api/documents/upload",
        data={"file": (f, "embed_http_test.pdf"), "title": "Embed HTTP Test"},
        content_type="multipart/form-data",
    )
os.unlink(pdf_path)
body = json.loads(r.data)
if body["success"]:
    http_doc_id = body["data"]["id"]

check("Upload succeeded for HTTP tests", http_doc_id is not None)

if http_doc_id:
    # GET /embedding-status (before embed)
    r = client.get(f"/api/documents/{http_doc_id}/embedding-status")
    check("GET /embedding-status -> 200",          r.status_code == 200, f"got {r.status_code}")
    body = json.loads(r.data)
    check("embedding-status success=True",         body["success"])
    st = body.get("data", {})
    check("embedding-status has total_chunks",     "total_chunks" in st)
    check("embedding-status has embedded_chunks",  "embedded_chunks" in st)
    check("embedding-status has percent_complete", "percent_complete" in st)
    check("embedding-status has is_fully_embedded","is_fully_embedded" in st)
    check("embedding-status has model field",      "model" in st)
    check("not fully embedded yet",                not st.get("is_fully_embedded"))
    print(f"      total={st.get('total_chunks')}  embedded={st.get('embedded_chunks')}  "
          f"pct={st.get('percent_complete')}%")

    # POST /embed (with mocked model)
    with patch("services.embedding_service._get_model", return_value=make_mock_model()):
        r = client.post(f"/api/documents/{http_doc_id}/embed", json={"batch_size": 8})
    check("POST /embed -> 200",                    r.status_code == 200, f"got {r.status_code}")
    body = json.loads(r.data)
    check("embed response success=True",           body["success"])
    d = body.get("data", {})
    check("embed response has embedded count",     "embedded" in d)
    check("embed response has total_chunks",       "total_chunks" in d)
    check("embed response has duration_sec",       "duration_sec" in d)
    check("embed response has model",              "model" in d)
    check("embed response has dims=384",           d.get("dims") == DIMS)
    check("all chunks embedded",                   d.get("embedded") == d.get("total_chunks"),
          f"embedded={d.get('embedded')} total={d.get('total_chunks')}")
    print(f"      embedded={d.get('embedded')}  duration={d.get('duration_sec')}s  dims={d.get('dims')}")

    # GET /embedding-status (after embed)
    r = client.get(f"/api/documents/{http_doc_id}/embedding-status")
    body = json.loads(r.data)
    st2 = body.get("data", {})
    check("status after embed: is_fully_embedded=True",  st2.get("is_fully_embedded"))
    check("status after embed: percent_complete=100",    st2.get("percent_complete") == 100.0)
    check("status after embed: pending_chunks=0",        st2.get("pending_chunks") == 0)

    # re_embed=False on already-embedded (skips all)
    with patch("services.embedding_service._get_model", return_value=make_mock_model()):
        r = client.post(f"/api/documents/{http_doc_id}/embed", json={"re_embed": False})
    body = json.loads(r.data)
    check("re_embed=False: skipped == total in response",
          body["data"].get("skipped") == body["data"].get("total_chunks"))

    # Validation: bad batch_size
    r = client.post(f"/api/documents/{http_doc_id}/embed", json={"batch_size": 0})
    check("batch_size=0 -> 400",                   r.status_code == 400)

    r = client.post(f"/api/documents/{http_doc_id}/embed", json={"batch_size": 1000})
    check("batch_size=1000 -> 400",                r.status_code == 400)

    # Invalid UUID
    r = client.post("/api/documents/bad-uuid/embed")
    check("embed invalid UUID -> 400",             r.status_code == 400)

    r = client.get("/api/documents/bad-uuid/embedding-status")
    check("status invalid UUID -> 400",            r.status_code == 400)

    # Non-existent doc
    r = client.post("/api/documents/00000000-0000-0000-0000-000000000000/embed")
    check("embed unknown doc -> 404",              r.status_code == 404)

    r = client.get("/api/documents/00000000-0000-0000-0000-000000000000/embedding-status")
    check("status unknown doc -> 404",             r.status_code == 404)


# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Route registration check")
# ─────────────────────────────────────────────────────────────────────────────
routes = {str(r.rule) for r in app.url_map.iter_rules()}
check("POST /embed route registered",
      "/api/documents/<string:document_id>/embed" in routes)
check("GET /embedding-status route registered",
      "/api/documents/<string:document_id>/embedding-status" in routes)


# ─────────────────────────────────────────────────────────────────────────────
print()
passed = sum(1 for t, *_ in results if t == PASS)
failed = sum(1 for t, *_ in results if t == FAIL)
print(f"Results: {passed} passed, {failed} failed out of {len(results)} tests")
if failed:
    print("\nFAILED:")
    for tag, label, detail in results:
        if tag == FAIL:
            print(f"  FAIL  {label}" + (f"  [{detail}]" if detail else ""))
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
