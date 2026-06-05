"""
test_pdf_extraction.py
───────────────────────
End-to-end smoke test for PDF extraction + document upload pipeline.
Runs entirely in-memory (SQLite + temp files) — no PostgreSQL needed.
"""

import os
import sys
import json
import tempfile

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["FLASK_ENV"]    = "development"
os.environ["SECRET_KEY"]   = "test-key"
os.environ["CORS_ORIGINS"] = "http://localhost:3000"

# ── Build a minimal PDF in memory using only stdlib + pypdf ──────────────────
def create_test_pdf(path: str, text: str = "Hello PDF World from AI Research Assistant.") -> None:
    """Create a minimal valid single-page PDF without reportlab."""
    # Minimal PDF structure — hand-crafted for testing
    stream = text.encode("latin-1", errors="replace")
    content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
        b"   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    )
    stream_data = b"BT /F1 12 Tf 72 720 Td (" + stream + b") Tj ET"
    content += (
        b"4 0 obj\n<< /Length " + str(len(stream_data)).encode() + b" >>\nstream\n"
        + stream_data + b"\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000360 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n430\n%%EOF\n"
    )
    with open(path, "wb") as f:
        f.write(content)


PASS = "PASS"
FAIL = "FAIL"
results = []

def check(label, condition, detail=""):
    tag = PASS if condition else FAIL
    results.append((tag, label, detail))
    print(f"  {tag}  {label}" + (f" | {detail}" if detail else ""))


# ── Test 1: pdf_extractor module directly ─────────────────────────────────────
print("\n[1] pdf_extractor unit tests")
from services.pdf_extractor import extract, ExtractionResult, _clean_text

# Clean text helper
check("_clean_text strips trailing spaces", _clean_text("hello   \nworld  ") == "hello\nworld")
check("_clean_text collapses blank lines", "\n\n\n" not in _clean_text("a\n\n\n\nb"))
check("_clean_text handles empty string", _clean_text("") == "")

# Non-existent file
res = extract("/nonexistent/path/file.pdf")
check("extract() handles missing file gracefully", not res.success)
check("extract() sets error on missing file", res.error is not None)

# Real PDF file
with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
    pdf_path = tmp.name

try:
    create_test_pdf(pdf_path)
    res = extract(pdf_path)
    check("extract() returns ExtractionResult", isinstance(res, ExtractionResult))
    check("extract() finds 1 page", res.page_count == 1, f"got {res.page_count}")
    check("extract() has page_texts list", isinstance(res.page_texts, list))
    check("ExtractionResult.preview() works", len(res.preview(10)) <= 10)
finally:
    os.unlink(pdf_path)


# ── Test 2: Flask app + routes ────────────────────────────────────────────────
print("\n[2] Flask app + route tests")
from app import create_app
app = create_app()
client = app.test_client()

# Health checks
r = client.get("/api/health")
check("GET /api/health -> 200", r.status_code == 200)
body = json.loads(r.data)
check("Health response success=True", body["success"] is True)

r = client.get("/api/health/db")
check("GET /api/health/db -> 200", r.status_code == 200)

r = client.get("/api/health/detailed")
check("GET /api/health/detailed -> 200", r.status_code == 200)
body = json.loads(r.data)
check("Detailed health has uptime_seconds", "uptime_seconds" in body.get("data", {}))

# Document list (empty)
r = client.get("/api/documents")
check("GET /api/documents -> 200", r.status_code == 200)
body = json.loads(r.data)
check("Empty documents list returns []", body["data"] == [])
check("Pagination meta present", "meta" in body)

# Upload a PDF
with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
    pdf_path = tmp.name
create_test_pdf(pdf_path, "The quick brown fox jumps over the lazy dog.")

doc_id = None
with open(pdf_path, "rb") as f:
    r = client.post(
        "/api/documents/upload",
        data={"file": (f, "test_research.pdf"), "title": "Test PDF", "tags": "test,pdf"},
        content_type="multipart/form-data",
    )
os.unlink(pdf_path)

check("POST /api/documents/upload -> 201", r.status_code == 201, f"got {r.status_code}")
body = json.loads(r.data)
check("Upload response success=True", body["success"] is True)

if body["success"]:
    doc_id = body["data"]["id"]
    d = body["data"]
    check("Document has id", bool(d.get("id")))
    check("Document title set", d.get("title") == "Test PDF")
    check("Document status is ready or error", d.get("status") in ("ready", "error"))
    check("Document page_count present", "page_count" in d)
    check("Document word_count present", "word_count" in d)
    check("Document has is_scanned flag", "is_scanned" in d)
    check("Tags saved correctly", "test" in (d.get("tags") or []))
    check("content_preview present", "content_preview" in d)
    print(f"      status={d['status']}  words={d.get('word_count')}  pages={d.get('page_count')}")

# Get single document
if doc_id:
    r = client.get(f"/api/documents/{doc_id}")
    check("GET /api/documents/<id> -> 200", r.status_code == 200)

    # Get full content
    r = client.get(f"/api/documents/{doc_id}/content")
    check("GET /api/documents/<id>/content -> 200 or 422", r.status_code in (200, 422),
          f"got {r.status_code}")
    body = json.loads(r.data)
    if r.status_code == 200:
        check("Content response has extracted_text key", "extracted_text" in body.get("data", {}))

    # Re-extract endpoint
    r = client.post(f"/api/documents/{doc_id}/re-extract")
    check("POST /api/documents/<id>/re-extract -> 200", r.status_code == 200)

    # Update metadata
    r = client.patch(f"/api/documents/{doc_id}",
                     json={"title": "Updated Title", "tags": ["updated"]})
    check("PATCH /api/documents/<id> -> 200", r.status_code == 200)

    # Delete
    r = client.delete(f"/api/documents/{doc_id}")
    check("DELETE /api/documents/<id> -> 200", r.status_code == 200)

    # Confirm deleted
    r = client.get(f"/api/documents/{doc_id}")
    check("GET deleted doc -> 404", r.status_code == 404)

# Validation tests
r = client.get("/api/documents/not-a-uuid")
check("Invalid UUID -> 400", r.status_code == 400)

r = client.get("/api/documents/00000000-0000-0000-0000-000000000000")
check("Non-existent doc -> 404", r.status_code == 404)

r = client.post("/api/documents/upload", data={}, content_type="multipart/form-data")
check("Upload with no file -> 400", r.status_code == 400)


# ── Summary ───────────────────────────────────────────────────────────────────
print()
passed = sum(1 for t, *_ in results if t == PASS)
failed = sum(1 for t, *_ in results if t == FAIL)
print(f"Results: {passed} passed, {failed} failed out of {len(results)} tests")
if failed:
    print("FAILED TESTS:")
    for tag, label, detail in results:
        if tag == FAIL:
            print(f"  FAIL  {label}" + (f" | {detail}" if detail else ""))
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
