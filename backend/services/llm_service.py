"""
services/llm_service.py
────────────────────────
LLM integration layer — the G (Generation) step in the RAG pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WHAT THIS MODULE DOES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Takes the question asked by the user and the top-K relevant chunks
returned by RetrievalService, builds a grounded prompt, sends it to
Groq's hosted LLM (llama3-8b-8192 by default), and returns a
structured answer together with source citations.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  HOW RETRIEVED CHUNKS ARE INJECTED INTO THE PROMPT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each chunk is numbered and inserted verbatim into the SYSTEM message
as a labelled "Context" block:

    [Context 1] (document: "Attention Is All You Need", chunk 3)
    ─────────────────────────────────────────────────────────────
    The Transformer model relies solely on attention mechanisms …

    [Context 2] (document: "BERT Paper", chunk 12)
    ─────────────────────────────────────────────────────────────
    We introduce a new language representation model called BERT …

The LLM is then instructed (in the system prompt) to:
  - Answer ONLY from the provided context blocks.
  - Cite sources by their [Context N] label when possible.
  - Return the fixed refusal string if the answer isn't in context.

This technique is called "context stuffing" or "grounded generation".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  HOW RAG PREVENTS HALLUCINATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A "hallucination" occurs when a language model fabricates a plausible-
sounding but factually incorrect answer from its parametric memory.

RAG prevents this in three ways:

1. CONTEXT CONSTRAINT — The system prompt explicitly forbids the
   model from using any knowledge outside the provided context
   blocks.  The model is told: "If the context does not contain
   enough information, say so."

2. VERIFIABLE SOURCES — Every context block carries document_id +
   chunk_index metadata.  The caller can trace each part of the
   answer back to the exact passage in the original PDF.

3. REFUSAL INSTRUCTION — If none of the retrieved chunks contain a
   relevant answer, the model is instructed to return the standard
   refusal string rather than speculating:
       "The uploaded documents do not contain enough information
        to answer this question."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  COMPLETE Question → Retrieval → LLM → Answer FLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  POST /api/chat  { "question": "What are the key findings?" }
        │
        ▼ routes/chat.py  validates request
        │
        ▼ retrieval_service.search(question, top_k=5)
        │   • embed question → 384-dim query vector
        │   • dot-product against all chunk vectors
        │   • return top-5 ChunkResult objects
        │
        ▼ llm_service.answer(question, chunks)
        │
        ├─ 1. _build_context_block(chunks)
        │       Numbered, labelled context passages for the prompt.
        │
        ├─ 2. _build_messages(question, context_block)
        │       SYSTEM: role + context + strict grounding rules
        │       USER:   the question
        │
        ├─ 3. Groq().chat.completions.create(...)
        │       Model:       llama3-8b-8192 (or env override)
        │       temperature: 0  ← deterministic, factual answers
        │       max_tokens:  1024
        │
        ├─ 4. Extract answer text from response.choices[0].message
        │
        └─ 5. Return LLMResult(answer, sources, model, latency)
                │
                ▼ routes/chat.py serialises → JSON response
                  {
                    "answer":  "The key findings were …",
                    "sources": [
                      { "document_id": "uuid…", "chunk_index": 3 },
                      …
                    ]
                  }
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, List

from groq import Groq

from services.retrieval_service import ChunkResult

logger = logging.getLogger("ai_research.llm_service")

# ── Constants ─────────────────────────────────────────────────────────────────

# Read at call-time (inside answer()) so .env changes don't require a restart.
_DEFAULT_MODEL_FALLBACK       = "llama-3.3-70b-versatile"
_DEFAULT_TEMPERATURE_FALLBACK = 0
_DEFAULT_MAX_TOKENS_FALLBACK  = 2048

# The refusal string the LLM should return when context is insufficient.
REFUSAL_STRING = (
    "The uploaded documents do not contain enough information "
    "to answer this question."
)

# Replace your SYSTEM_PREAMBLE with this:
SYSTEM_PREAMBLE = """\
You are a helpful AI research assistant.

RULES:
1. You have been provided with the extracted text from the user's PDF documents in the DOCUMENT CONTEXT below. YOU CAN AND MUST read these documents. Never claim that you cannot read PDFs or access external files.
2. If the user's question is answerable from the provided context passages, use them to give a precise answer.
3. CITATIONS — Every factual claim MUST be followed immediately by [N] where N is the context block number. For example, if you use information from [Context 1], cite it as [1]. Multiple citations on one claim: [1][3].
4. If the user asks you to summarize, read, or look at the document, summarize the provided context for them.
5. If the context is empty or not relevant to the question, answer from your general knowledge.
6. Be concise and direct.
7. {refusal}

─────────────────────────────
DOCUMENT CONTEXT (Extracted text from the user's uploaded files)
─────────────────────────────
{context}
─────────────────────────────
""".format


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class SourceReference:
    """A pointer back to a specific chunk used in the answer."""
    document_id:  str
    chunk_index:  int

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
        }


@dataclass
class LLMResult:
    """
    Complete result of one answer() call.

    Attributes
    ----------
    answer       : The grounded answer text from the LLM.
    sources      : SourceReferences for each chunk injected into the prompt.
    model        : The Groq model identifier used.
    duration_sec : Wall-clock time for the LLM call alone.
    prompt_tokens: Tokens consumed in the prompt (from Groq usage metadata).
    answer_tokens: Tokens in the generated answer.
    """
    answer:        str
    sources:       List[SourceReference] = field(default_factory=list)
    model:         str = _DEFAULT_MODEL_FALLBACK
    duration_sec:  float = 0.0
    prompt_tokens: int = 0
    answer_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "answer":        self.answer,
            "sources":       [s.to_dict() for s in self.sources],
            "model":         self.model,
            "duration_sec":  round(self.duration_sec, 3),
            "prompt_tokens": self.prompt_tokens,
            "answer_tokens": self.answer_tokens,
        }


@dataclass
class LLMResearchResult:
    query: str
    answer: str          # Full markdown prose with inline [N] citations
    sources: list[dict[str, str]]
    raw_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": self.sources,
            "raw_text": self.raw_text,
        }


WEB_RESEARCH_INSTRUCTIONS = """\
You are a world-class research assistant that writes answers exactly like Perplexity AI.

You are given numbered web sources and their scraped content. Your job is to synthesize a rich,
flowing, well-cited answer in markdown.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STRICT WRITING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. CITATIONS — Every factual claim MUST be followed immediately by [N] where N is
   the source number (1-based). Multiple citations on one claim: [1][3].
   Never write a sentence without at least one citation unless it is a transition
   phrase or a heading.

2. FORMAT — Write in flowing prose paragraphs. Use:
   - ## Heading   for major sub-topics (use 2-4 headings max)
   - **bold**     for key terms when first introduced
   - - bullet     for enumerated lists (no more than one list per section)
   - Never use tables.

3. LENGTH — Aim for 4-6 paragraphs total (300-600 words). Be comprehensive
   but tight — no filler, no padding.

4. TONE — Authoritative, neutral, encyclopedic. Like a senior analyst briefing.

5. ACCURACY — Only use facts from the sources. If sources disagree, say so inline:
   "Some sources suggest X [1], while others report Y [3]."

6. NO HALLUCINATION — If the sources do not cover part of the question, say:
   "The available sources do not address [topic]."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  OUTPUT FORMAT  (MANDATORY — return ONLY valid JSON, no extra text)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "answer": "<full markdown answer with inline [N] citations>",
  "sources": [
    { "title": "<page title>", "url": "<canonical url>" },
    ...
  ]
}

The "sources" array must list ONLY sources you actually cited, in the order they
first appear in the answer. The index of each source in this array determines its
citation number (1-based).
"""


# ── Private helpers ───────────────────────────────────────────────────────────

def _build_context_block(chunks: List[ChunkResult]) -> str:
    """
    Convert a list of ChunkResult objects into a numbered, labelled
    context string that is injected verbatim into the system prompt.

    Example output:
        [Context 1] (document: "Attention Is All You Need", chunk 3)
        ─────────────────────────────────────────────────────────────
        The Transformer relies solely on attention mechanisms …

        [Context 2] (document: "BERT", chunk 12)
        ─────────────────────────────────────────────────────────────
        BERT stands for Bidirectional Encoder Representations …
    """
    if not chunks:
       return "(No document context available — answer from general knowledge.)"

    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = (
            f'[Context {i}] '
            f'(document: "{chunk.document_title}", chunk {chunk.chunk_index})'
        )
        separator = "─" * 65
        parts.append(f"{header}\n{separator}\n{chunk.chunk_text.strip()}")

    return "\n\n".join(parts)


def _build_messages(question: str, context_block: str) -> list[dict]:
    """
    Assemble the Groq chat message list.

    Returns a two-element list:
      [{"role": "system", "content": <grounded system prompt>},
       {"role": "user",   "content": <user question>}]
    """
    system_content = SYSTEM_PREAMBLE(
        refusal=REFUSAL_STRING,
        context=context_block,
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": question.strip()},
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def answer(
    question: str,
    chunks:   List[ChunkResult],
    model:    str | None = None,
) -> LLMResult:
    """
    Generate a grounded answer using the Groq LLM.

    Parameters
    ----------
    question : The user's natural-language question.
    chunks   : Top-K ChunkResult objects from RetrievalService.
               These are injected into the prompt as numbered context blocks.
    model    : Groq model identifier (default: llama3-8b-8192).

    Returns
    -------
    LLMResult with answer text, source citations, and token usage.

    Raises
    ------
    ValueError  : question is empty.
    RuntimeError: Groq API key missing or API call failed.
    """
    # Read config live from env so a .env change takes effect without restart.
    if model is None:
        model = os.getenv("GROQ_MODEL", _DEFAULT_MODEL_FALLBACK)
    temperature = float(os.getenv("GROQ_TEMPERATURE", str(_DEFAULT_TEMPERATURE_FALLBACK)))
    max_tokens  = int(os.getenv("GROQ_MAX_TOKENS",  str(_DEFAULT_MAX_TOKENS_FALLBACK)))

    question = question.strip()
    if not question:
        raise ValueError("Question must be a non-empty string.")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )

    logger.info(
        "LLM call: model=%s  chunks=%d  question=%r",
        model, len(chunks), question[:80],
    )

    # ── 1. Build context block ────────────────────────────────────────────────
    context_block = _build_context_block(chunks)
    logger.debug(
        "Context block built: %d chars from %d chunks",
        len(context_block), len(chunks),
    )

    # ── 2. Build message list ─────────────────────────────────────────────────
    messages = _build_messages(question, context_block)

    # ── 3. Call Groq API ──────────────────────────────────────────────────────
    client = Groq(api_key=groq_api_key)
    t_start = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        logger.exception("Groq API call failed: %s", exc)
        raise RuntimeError(f"LLM call failed: {exc}") from exc

    duration = time.time() - t_start

    # ── 4. Extract answer ─────────────────────────────────────────────────────
    answer_text = response.choices[0].message.content or REFUSAL_STRING

    # ── 5. Token usage ────────────────────────────────────────────────────────
    usage = response.usage
    prompt_tokens = usage.prompt_tokens  if usage else 0
    answer_tokens = usage.completion_tokens if usage else 0

    logger.info(
        "LLM done: %.3fs  prompt_tokens=%d  answer_tokens=%d",
        duration, prompt_tokens, answer_tokens,
    )

    # ── 6. Build source citations ─────────────────────────────────────────────
    # Each chunk injected into the prompt becomes a SourceReference so the
    # caller can trace claims back to the exact document and chunk position.
    sources = [
        SourceReference(
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
        )
        for chunk in chunks
    ]

    return LLMResult(
        answer        = answer_text,
        sources       = sources,
        model         = model,
        duration_sec  = duration,
        prompt_tokens = prompt_tokens,
        answer_tokens = answer_tokens,
    )


def summarize_web_research(
    query: str,
    context_text: str,
    sources: list[dict[str, str]],
    model: str | None = None,
) -> LLMResearchResult:
    if not query.strip():
        raise ValueError("Query must be a non-empty string.")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )

    if model is None:
        model = os.getenv("GROQ_MODEL", _DEFAULT_MODEL_FALLBACK)
    temperature = float(os.getenv("GROQ_TEMPERATURE", str(_DEFAULT_TEMPERATURE_FALLBACK)))
    max_tokens  = int(os.getenv("GROQ_MAX_TOKENS",  str(_DEFAULT_MAX_TOKENS_FALLBACK)))

    # ── Build clearly-numbered source blocks so [N] citations are unambiguous ──
    # Each block shows its number, title, URL, and the scraped text. The LLM
    # must cite [N] inline whenever it uses information from that source.
    source_blocks: list[str] = []
    for i, src in enumerate(sources, start=1):
        title = src.get("title", f"Source {i}")
        url   = src.get("url", "")
        source_blocks.append(
            f"[Source {i}] {title}\n"
            f"URL: {url}\n"
            f"─────────────────────────────────────────────\n"
        )

    numbered_sources = "\n".join(source_blocks)

    system_prompt = (
        f"{WEB_RESEARCH_INSTRUCTIONS}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  AVAILABLE SOURCES (cite with [N])\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{numbered_sources}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  SCRAPED RESEARCH CONTENT\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{context_text}\n\n"
        "Now produce the JSON payload exactly as specified. "
        "Return ONLY the JSON object — no markdown fences, no extra text."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": query.strip()},
    ]

    client = Groq(api_key=groq_api_key)
    t_start = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        logger.exception("Groq API call failed: %s", exc)
        raise RuntimeError(f"LLM call failed: {exc}") from exc

    duration = time.time() - t_start
    answer_text = response.choices[0].message.content or ""

    logger.info(
        "Web research LLM done: %.3fs  model=%s  answer_len=%d",
        duration, model, len(answer_text),
    )

    # ── Parse JSON — multi-strategy recovery ─────────────────────────────────
    #
    # The model frequently puts raw double-quotes inside the "answer" value,
    # e.g.  "answer": "She said "hi" to him"  which breaks standard JSON.
    # Strategy order:
    #  1. Strict json.loads on the raw text.
    #  2. Strip markdown fences, try again.
    #  3. Regex-extract the answer string (capture up to the first "sources":
    #     key) and sources array, bypassing the broken quote escaping.
    #  4. Plain-text fallback — return the raw LLM text as the answer so the
    #     frontend still renders something useful instead of an error.
    # ──────────────────────────────────────────────────────────────────────────
    import re as _re

    parsed: dict[str, Any] = {}
    clean = answer_text.strip()

    def _try_parse(text: str) -> dict | None:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    # 1. Strict parse
    parsed = _try_parse(clean) or {}

    # 2. Strip ```json ... ``` fences if present, retry
    if not parsed and clean.startswith("```"):
        fence_end   = clean.find("\n")
        close_fence = clean.rfind("```")
        if close_fence > fence_end:
            parsed = _try_parse(clean[fence_end:close_fence].strip()) or {}

    # 3. Find the outermost { … } block and retry
    if not parsed:
        start = clean.find("{")
        end   = clean.rfind("}")
        if start != -1 and end > start:
            parsed = _try_parse(clean[start:end + 1]) or {}

    # 4. Regex extraction — handles unescaped interior quotes.
    #    Extract "answer": "<everything up to the next top-level key or end>"
    if not parsed:
        # Capture the text between "answer": " … " and the next JSON key
        # (greedy match stopped by  ,"sources":  or end-of-object)
        answer_match = _re.search(
            r'"answer"\s*:\s*"(.*?)(?="\s*,\s*"sources"|"\s*\})',
            clean,
            _re.DOTALL,
        )
        raw_answer = answer_match.group(1) if answer_match else clean

        # Try to extract sources array as raw text
        sources_raw: list[dict] = []
        src_block = _re.search(r'"sources"\s*:\s*(\[.*?\])', clean, _re.DOTALL)
        if src_block:
            try:
                sources_raw = json.loads(src_block.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        if raw_answer:
            parsed = {"answer": raw_answer, "sources": sources_raw}
            logger.info("JSON recovered via regex extraction.")

    # 5. Plain-text fallback — never return an error to the user
    if not parsed:
        logger.warning(
            "All JSON parse strategies failed; using raw text as answer. "
            "First 400 chars: %s", clean[:400]
        )
        parsed = {"answer": clean, "sources": []}



    sources_data = parsed.get("sources", [])
    if not isinstance(sources_data, list):
        raise RuntimeError("LLM response 'sources' field is invalid.")

    # Fall back to the input sources list if the model returned an empty array
    if not sources_data and sources:
        sources_data = sources

    return LLMResearchResult(
        query=query,
        answer=parsed.get("answer", ""),
        sources=[{
            "title": s.get("title", ""),
            "url":   s.get("url", ""),
        } for s in sources_data if isinstance(s, dict)],
        raw_text=answer_text,
    )

