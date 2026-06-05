"""
services/pdf_extractor.py
──────────────────────────
PDF text extraction using pypdf.

Responsibilities
────────────────
- Extract full text from all pages of a PDF file
- Return per-page breakdown for fine-grained access
- Compute word count and character count
- Detect scanned/image-only PDFs (no extractable text)
- All errors are non-fatal — returns an ExtractionResult describing
  success or failure without raising.

Data flow
─────────
  Upload route
      │
      ▼
  document_service.upload_document()
      │  saves file to disk
      ▼
  pdf_extractor.extract(file_path)          ← this module
      │  reads PDF page-by-page with pypdf
      ▼
  ExtractionResult
      │  full_text, page_texts, word_count, char_count, page_count
      ▼
  document_service  commits result to Document.extracted_text in PostgreSQL
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("ai_research.pdf_extractor")


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """
    Holds the outcome of a PDF extraction attempt.

    Attributes
    ----------
    success      : True if at least some text was extracted.
    full_text    : Concatenated text from every page (newline-separated).
    page_texts   : List of per-page text strings (index 0 = page 1).
    page_count   : Total number of pages found in the PDF.
    word_count   : Number of whitespace-delimited tokens in full_text.
    char_count   : Number of characters in full_text.
    is_scanned   : True when pages exist but no text could be extracted
                   (likely a scanned/image-based PDF).
    error        : Human-readable error message if success is False.
    """
    success: bool = False
    full_text: str = ""
    page_texts: list[str] = field(default_factory=list)
    page_count: int = 0
    word_count: int = 0
    char_count: int = 0
    is_scanned: bool = False
    error: Optional[str] = None

    # Convenience: first N characters for preview
    def preview(self, max_chars: int = 500) -> str:
        return self.full_text[:max_chars]


# ── Public API ────────────────────────────────────────────────────────────────

def extract(file_path: str) -> ExtractionResult:
    """
    Extract all text from a PDF file using pypdf.

    Parameters
    ----------
    file_path : Absolute path to the PDF file on disk.

    Returns
    -------
    ExtractionResult — never raises; errors are captured in .error field.
    """
    try:
        import pypdf  # lazy import — only required for PDFs
    except ImportError:
        logger.error("pypdf is not installed. Run: pip install pypdf")
        return ExtractionResult(error="pypdf is not installed.")

    try:
        return _extract_with_pypdf(file_path)
    except Exception as exc:
        logger.exception("Unexpected error extracting PDF: %s", file_path)
        return ExtractionResult(error=f"Extraction failed: {exc}")


# ── Private implementation ────────────────────────────────────────────────────

def _extract_with_pypdf(file_path: str) -> ExtractionResult:
    """Core extraction logic."""
    import pypdf

    result = ExtractionResult()

    with open(file_path, "rb") as fh:
        reader = pypdf.PdfReader(fh)
        result.page_count = len(reader.pages)

        if result.page_count == 0:
            result.error = "PDF contains no pages."
            return result

        page_texts: list[str] = []

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                raw = page.extract_text() or ""
                cleaned = _clean_text(raw)
                page_texts.append(cleaned)

                if cleaned:
                    logger.debug(
                        "Page %d/%d: extracted %d chars",
                        page_num, result.page_count, len(cleaned),
                    )
                else:
                    logger.debug("Page %d/%d: no text (possibly image-only)", page_num, result.page_count)

            except Exception as exc:  # noqa: BLE001 — partial failure is fine
                logger.warning("Failed to extract page %d: %s", page_num, exc)
                page_texts.append("")  # keep index aligned with page number

    result.page_texts = page_texts
    result.full_text = "\n\n".join(t for t in page_texts if t)
    result.word_count = len(result.full_text.split())
    result.char_count = len(result.full_text)
    result.success = bool(result.full_text.strip())

    if not result.success and result.page_count > 0:
        result.is_scanned = True
        result.error = (
            f"No text could be extracted from {result.page_count} page(s). "
            "The PDF may be scanned or image-based. OCR support coming soon."
        )
        logger.warning("PDF appears to be scanned: %s", file_path)
    else:
        logger.info(
            "Extracted %d words / %d chars from %d pages: %s",
            result.word_count, result.char_count, result.page_count, file_path,
        )

    return result


def _clean_text(raw: str) -> str:
    """
    Normalise extracted text:
    - Collapse multiple blank lines to a single blank line
    - Strip trailing whitespace from each line
    - Normalise Unicode whitespace characters
    """
    if not raw:
        return ""
    # Replace non-breaking spaces and other unicode spaces with regular space
    text = re.sub(r"[\u00a0\u2000-\u200b\u202f\u205f\u3000]", " ", raw)
    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.splitlines()]
    # Collapse 3+ consecutive newlines → exactly 2 (one blank line between paragraphs)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return cleaned.strip()
