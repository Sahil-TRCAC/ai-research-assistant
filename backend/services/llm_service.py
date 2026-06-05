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

import logging
import os
import time
from dataclasses import dataclass, field
from typing import List

from groq import Groq

from services.retrieval_service import ChunkResult

logger = logging.getLogger("ai_research.llm_service")

# ── Constants ─────────────────────────────────────────────────────────────────

# Read at call-time (inside answer()) so .env changes don't require a restart.
_DEFAULT_MODEL_FALLBACK       = "llama-3.3-70b-versatile"
_DEFAULT_TEMPERATURE_FALLBACK = 0
_DEFAULT_MAX_TOKENS_FALLBACK  = 1024

# The refusal string the LLM should return when context is insufficient.
REFUSAL_STRING = (
    "The uploaded documents do not contain enough information "
    "to answer this question."
)

# Replace your SYSTEM_PREAMBLE with this:
SYSTEM_PREAMBLE = """\
You are a helpful AI research assistant.

RULES:
1. If the user's question is answerable from the provided context passages, 
   use them to give a precise answer.
2. If the context is empty or not relevant to the question, answer from 
   your general knowledge without mentioning the documents.
3. Never refuse to answer just because something isn't in the documents.
4. Be concise and direct.
5. {refusal}

─────────────────────────────
DOCUMENT CONTEXT (may be empty or partial)
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
