"""
test_chunking.py
─────────────────
End-to-end tests for the document chunking pipeline.
Runs entirely in-memory (SQLite + temp files) — no PostgreSQL needed.
"""

import os, sys, json, tempfile

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["FLASK_ENV"]    = "development"
os.environ["SECRET_KEY"]   = "test-key"
os.environ["CORS_ORIGINS"] = "http://localhost:3000"

# ── Minimal PDF builder (no external deps) ────────────────────────────────────
def create_test_pdf(path: str, text: str) -> None:
    stream = text.encode("latin-1", errors="replace")
    stream_data = b"BT /F1 12 Tf 72 720 Td (" + stream + b") Tj ET"
    content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
        b"   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length " + str(len(stream_data)).encode() + b" >>\nstream\n"
        + stream_data + b"\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n"
        b"0000000115 00000 n \n0000000266 00000 n \n0000000360 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n430\n%%EOF\n"
    )
    with open(path, "wb") as f:
        f.write(content)


PASS, FAIL = "PASS", "FAIL"
results = []

def check(label, condition, detail=""):
    tag = PASS if condition else FAIL
    results.append((tag, label, detail))
    print(f"  {tag}  {label}" + (f"  [{detail}]" if detail else ""))


# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1] split_text_into_chunks — unit tests")
# ═══════════════════════════════════════════════════════════════════════════════
from services.chunking_service import split_text_into_chunks, _find_sentence_boundary

# Empty / blank input
check("empty string yields no chunks",      split_text_into_chunks("") == [])
check("whitespace-only yields no chunks",   split_text_into_chunks("   ") == [])

# Sentence boundary detection
check("finds . boundary",  _find_sentence_boundary("Hello world. Next sentence.") is not None)
check("finds ! boundary",  _find_sentence_boundary("Wow! Amazing.") is not None)
check("finds ? boundary",  _find_sentence_boundary("Really? Yes.") is not None)
check("returns None for no boundary", _find_sentence_boundary("no sentence end here") is None)

# Short text fits in one chunk (must be > min_chars=50)
short = "The quick brown fox jumps over the lazy dog and runs away."
chunks = split_text_into_chunks(short, chunk_size=500, overlap=100)
check("short text -> 1 chunk",          len(chunks) == 1, f"got {len(chunks)}")
check("chunk has correct keys",
      all(k in chunks[0] for k in ("chunk_index","chunk_text","char_count","start_char","end_char")))
check("chunk_index starts at 0",        chunks[0]["chunk_index"] == 0)
check("chunk_text matches input",       chunks[0]["chunk_text"] == short.strip())
check("char_count is accurate",         chunks[0]["char_count"] == len(short.strip()))
check("start_char is 0",                chunks[0]["start_char"] == 0)

# Long text produces multiple chunks
long_text = "The quick brown fox jumps over the lazy dog. " * 30   # ~1350 chars
chunks = split_text_into_chunks(long_text, chunk_size=500, overlap=100)
check("long text -> multiple chunks",   len(chunks) > 1, f"got {len(chunks)}")
check("chunks are ordered by index",
      all(chunks[i]["chunk_index"] == i for i in range(len(chunks))))
check("all chunks have char_count > 0", all(c["char_count"] > 0 for c in chunks))
check("no empty chunk_text",            all(c["chunk_text"].strip() for c in chunks))

# Overlap: tail of chunk[i] appears in head of chunk[i+1]
if len(chunks) >= 2:
    tail = chunks[0]["chunk_text"][-80:]
    head = chunks[1]["chunk_text"][:200]
    check("overlap: chunk[0] tail in chunk[1] head", tail[:30] in head,
          f"tail={tail[:30]!r}")

# skip_min: tiny chunks are dropped
tiny_chunks = split_text_into_chunks("Hi!", chunk_size=500, overlap=100, min_chars=50)
check("chunk shorter than min_chars is skipped", len(tiny_chunks) == 0)

# Custom chunk_size / overlap
chunks_custom = split_text_into_chunks(long_text, chunk_size=200, overlap=50)
check("custom chunk_size respected (max char_count)",
      all(c["char_count"] <= 250 for c in chunks_custom))   # some buffer for sentence cuts
check("custom overlap produces more chunks than default",
      len(chunks_custom) >= len(chunks))

print(f"  (produced {len(chunks)} chunks @ 500/100,  {len(chunks_custom)} chunks @ 200/50)")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2] chunk_document — database integration tests")
# ═══════════════════════════════════════════════════════════════════════════════
from app import create_app
app = create_app()

with app.app_context():
    from models import db, Document, DocumentChunk
    from services.chunking_service import (
        chunk_document, get_chunks_for_document, delete_chunks_for_document
    )

    # Create a synthetic Document directly (bypass file I/O)
    long_extracted = ("The quick brown fox jumps over the lazy dog. " * 40).strip()
    doc = Document(
        title="Test Doc",
        original_filename="test.pdf",
        stored_filename="test_stored.pdf",
        file_path="/tmp/test.pdf",
        file_size=1000,
        file_type="pdf",
        status="ready",
        extracted_text=long_extracted,
        word_count=len(long_extracted.split()),
        char_count=len(long_extracted),
        tags=[],
    )
    db.session.add(doc)
    db.session.commit()
    doc_id = doc.id

    # chunk_document() call
    chunks = chunk_document(doc_id)
    check("chunk_document returns list",        isinstance(chunks, list))
    check("at least 1 chunk produced",          len(chunks) >= 1, f"got {len(chunks)}")
    check("chunk objects are DocumentChunk",    all(isinstance(c, DocumentChunk) for c in chunks))
    check("all chunks have correct document_id",
          all(c.document_id == doc_id for c in chunks))
    check("chunk_index sequential from 0",
          [c.chunk_index for c in chunks] == list(range(len(chunks))))
    check("chunk_text is non-empty for all",    all(c.chunk_text.strip() for c in chunks))
    check("char_count matches len(chunk_text)",
          all(c.char_count == len(c.chunk_text) for c in chunks))

    # Document.chunk_count updated
    db.session.refresh(doc)
    check("Document.chunk_count updated",       doc.chunk_count == len(chunks),
          f"doc.chunk_count={doc.chunk_count} len={len(chunks)}")

    # get_chunks_for_document pagination
    page1, total = get_chunks_for_document(doc_id, page=1, per_page=2)
    check("pagination: total == len(chunks)",   total == len(chunks))
    check("pagination: per_page=2 returns <=2", len(page1) <= 2)
    check("pagination: page1 starts at idx 0",  page1[0].chunk_index == 0)

    if total > 2:
        page2, _ = get_chunks_for_document(doc_id, page=2, per_page=2)
        check("pagination: page2 is different from page1",
              page2[0].chunk_index != page1[0].chunk_index)

    # replace_existing: rechunk replaces old chunks
    old_count = total
    new_chunks = chunk_document(doc_id, chunk_size=300, overlap=50, replace_existing=True)
    db_count = DocumentChunk.query.filter_by(document_id=doc_id).count()
    check("replace_existing removes old chunks", db_count == len(new_chunks),
          f"db={db_count} new={len(new_chunks)}")

    # Error: non-existent document
    try:
        chunk_document("00000000-0000-0000-0000-000000000000")
        check("ValueError on missing doc", False)
    except ValueError:
        check("ValueError raised for missing document", True)

    # Error: document with no extracted_text
    empty_doc = Document(
        title="Empty", original_filename="e.pdf", stored_filename="e.pdf",
        file_path="/tmp/e.pdf", file_size=0, file_type="pdf",
        status="error", tags=[],
    )
    db.session.add(empty_doc)
    db.session.commit()
    try:
        chunk_document(empty_doc.id)
        check("ValueError on no extracted_text", False)
    except ValueError:
        check("ValueError raised for missing extracted_text", True)

    # delete_chunks_for_document
    deleted = delete_chunks_for_document(doc_id)
    check("delete_chunks returns correct count", deleted == len(new_chunks),
          f"deleted={deleted}")
    remaining = DocumentChunk.query.filter_by(document_id=doc_id).count()
    check("chunks removed from DB after delete", remaining == 0)


# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3] HTTP route tests — /chunks and /rechunk")
# ═══════════════════════════════════════════════════════════════════════════════
client = app.test_client()

# Build a real PDF and upload it
with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
    pdf_path = tmp.name
long_pdf_text = "The quick brown fox jumps over the lazy dog. " * 50
create_test_pdf(pdf_path, long_pdf_text)

doc_id = None
with open(pdf_path, "rb") as f:
    r = client.post(
        "/api/documents/upload",
        data={"file": (f, "chunking_test.pdf"), "title": "Chunking Test"},
        content_type="multipart/form-data",
    )
os.unlink(pdf_path)

check("Upload -> 201", r.status_code == 201, f"got {r.status_code}")
body = json.loads(r.data)
if body["success"]:
    doc_id = body["data"]["id"]
    check("Upload: chunk_count in response", "chunk_count" in body["data"])
    print(f"      doc_id={doc_id}  status={body['data']['status']}  "
          f"chunk_count={body['data'].get('chunk_count')}")

if doc_id:
    # GET /api/documents/<id>/chunks
    r = client.get(f"/api/documents/{doc_id}/chunks")
    check("GET /chunks -> 200", r.status_code == 200, f"got {r.status_code}")
    body = json.loads(r.data)
    check("GET /chunks success=True",            body["success"])
    check("GET /chunks has data list",           isinstance(body.get("data"), list))
    check("GET /chunks has pagination meta",     "meta" in body)
    check("GET /chunks meta.total > 0",          body["meta"]["total"] > 0,
          f"total={body['meta']['total']}")

    if body["data"]:
        c0 = body["data"][0]
        check("chunk has chunk_index",  "chunk_index" in c0)
        check("chunk has chunk_text",   "chunk_text"  in c0)
        check("chunk has char_count",   "char_count"  in c0)
        check("chunk has start_char",   "start_char"  in c0)
        check("chunk has end_char",     "end_char"    in c0)
        check("chunk has is_embedded",  "is_embedded" in c0)
        check("chunk_index == 0 for first", c0["chunk_index"] == 0)
        check("is_embedded starts False",   c0["is_embedded"] == False)
        print(f"      chunk[0]: chars={c0['char_count']}  "
              f"start={c0['start_char']}  end={c0['end_char']}")

    # GET with pagination params
    r = client.get(f"/api/documents/{doc_id}/chunks?page=1&per_page=2")
    check("GET /chunks?per_page=2 -> 200",     r.status_code == 200)
    body = json.loads(r.data)
    check("per_page=2 returns at most 2 items", len(body["data"]) <= 2)

    # POST /rechunk with custom params
    r = client.post(f"/api/documents/{doc_id}/rechunk",
                    json={"chunk_size": 300, "overlap": 50})
    check("POST /rechunk -> 200",              r.status_code == 200)
    body = json.loads(r.data)
    check("rechunk success=True",              body["success"])
    check("rechunk returns chunk_count",       "chunk_count" in body.get("data", {}))
    check("rechunk chunk_size reflected",      body["data"].get("chunk_size") == 300)
    check("rechunk overlap reflected",         body["data"].get("overlap") == 50)
    check("rechunk has sample chunk",          body["data"].get("sample") is not None)
    print(f"      rechunk(300/50): chunk_count={body['data'].get('chunk_count')}")

    # /rechunk validation errors
    r = client.post(f"/api/documents/{doc_id}/rechunk", json={"chunk_size": 50})
    check("rechunk chunk_size < 100 -> 400",  r.status_code == 400)

    r = client.post(f"/api/documents/{doc_id}/rechunk",
                    json={"chunk_size": 200, "overlap": 200})
    check("rechunk overlap >= chunk_size -> 400", r.status_code == 400)

    # Invalid UUID
    r = client.get("/api/documents/not-a-uuid/chunks")
    check("GET /chunks invalid UUID -> 400",   r.status_code == 400)

    # Non-existent document
    r = client.get("/api/documents/00000000-0000-0000-0000-000000000000/chunks")
    check("GET /chunks unknown doc -> 404",    r.status_code == 404)


# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4] Route list verification")
# ═══════════════════════════════════════════════════════════════════════════════
routes = {str(r.rule) for r in app.url_map.iter_rules()}
check("GET  /api/documents/<id>/chunks registered",
      "/api/documents/<string:document_id>/chunks" in routes)
check("POST /api/documents/<id>/rechunk registered",
      "/api/documents/<string:document_id>/rechunk" in routes)


# ═══════════════════════════════════════════════════════════════════════════════
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
