from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from services.scraper import ScrapedPage, build_source_preview, scrape_url, validate_source_url

logger = logging.getLogger("ai_research.research_engine")

SEARCH_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

CACHE_TTL_SECONDS = 3600
MAX_SOURCES = 4
MAX_CHARACTERS = 16000
CACHE: dict[str, dict[str, Any]] = {}

@dataclass
class ResearchSource:
    title: str
    url: str
    domain: str
    snippets: list[str]
    relevance_score: float


@dataclass
class ResearchContext:
    query: str
    sources: list[ResearchSource]
    combined_text: str
    summary_prompt: str


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.lower().strip())


def _cache_key(query: str) -> str:
    return _normalize_query(query)


def _get_cached_query(query: str) -> ResearchContext | None:
    key = _cache_key(query)
    entry = CACHE.get(key)
    if not entry:
        return None
    if time.time() - entry["timestamp"] > CACHE_TTL_SECONDS:
        del CACHE[key]
        return None
    return entry["value"]


def _set_cached_query(query: str, context: ResearchContext) -> None:
    CACHE[_cache_key(query)] = {
        "timestamp": time.time(),
        "value": context,
    }


def _duckduckgo_search(query: str, max_results: int = 8) -> list[str]:
    url = "https://html.duckduckgo.com/html/"
    try:
        response = requests.post(
            url,
            data={"q": query},
            headers=SEARCH_REQUEST_HEADERS,
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Search request failed: %s", exc)
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    urls: list[str] = []
    for link in soup.select("a.result__a, a[data-testid='result-title-a'], a[href]"):
        href = link.get("href")
        if not href or not href.startswith("http"):
            continue
        if href in urls:
            continue
        if validate_source_url(href):
            urls.append(href)
        if len(urls) >= max_results:
            break
    return urls


def _score_source(page: ScrapedPage, query: str) -> float:
    query_terms = set(re.findall(r"\w+", query.lower()))
    if not query_terms:
        return 0.0

    content = " ".join([page.title, page.text]).lower()
    matches = sum(content.count(term) for term in query_terms)
    score = matches + len(page.headings) * 2 + len(page.paragraphs) * 0.1
    return score


def _build_combined_text(sources: list[ResearchSource]) -> str:
    chunks: list[str] = []
    total_chars = 0
    for source in sources:
        snippet = "\n\n".join(source.snippets)
        if not snippet:
            continue
        snippet = snippet.strip()
        if not snippet:
            continue
        if total_chars + len(snippet) > MAX_CHARACTERS:
            remaining = MAX_CHARACTERS - total_chars
            if remaining > 0:
                chunks.append(snippet[:remaining].rstrip())
            break
        chunks.append(f"[{source.title}]({source.url})\n{snippet}")
        total_chars += len(snippet)
    return "\n\n".join(chunks)


def _create_summary_prompt(query: str, context_text: str, sources: list[ResearchSource]) -> str:
    source_list = "\n".join(f"- {source.title}: {source.url}" for source in sources)
    return (
        f"Research query: {query}\n"
        f"Sources:\n{source_list}\n\n"
        f"Research content:\n{context_text}\n\n"
        "Based only on the research content above, produce a structured JSON answer with keys:\n"
        "query, summary, key_findings, sources, conclusion.\n"
        "Do not invent facts. If the sources conflict, identify the disagreement."
    )


def _extract_snippets(page: ScrapedPage, query: str, max_snippets: int = 6) -> list[str]:
    normalized_query = query.lower()
    snippets: list[str] = []
    for heading in page.headings[:4]:
        snippets.append(heading)
    for paragraph in page.paragraphs:
        if normalized_query in paragraph.lower() or len(snippets) < 3:
            snippets.append(paragraph)
        if len(snippets) >= max_snippets:
            break
    for items in page.lists:
        if len(snippets) >= max_snippets:
            break
        bullets = " ".join(items)
        if normalized_query in bullets.lower() or len(snippets) < 3:
            snippets.append("; ".join(items))
    return snippets


def perform_live_research(query: str, max_sources: int = MAX_SOURCES) -> ResearchContext:
    query = query.strip()
    if not query:
        raise ValueError("Query must be a non-empty string.")

    cached = _get_cached_query(query)
    if cached is not None:
        logger.info("Using cached research results for query=%r", query)
        return cached

    urls = _duckduckgo_search(query, max_results=max_sources * 2)
    pages: list[ScrapedPage] = []
    for url in urls:
        try:
            page = scrape_url(url)
            if page.text:
                pages.append(page)
        except Exception as exc:
            logger.warning("Skipping URL %s due to error: %s", url, exc)
        if len(pages) >= max_sources:
            break

    if not pages:
        raise RuntimeError("No research sources could be retrieved.")

    scored = [
        ResearchSource(
            title=page.title,
            url=page.url,
            domain=page.domain,
            snippets=_extract_snippets(page, query),
            relevance_score=_score_source(page, query),
        )
        for page in pages
    ]
    scored.sort(key=lambda s: s.relevance_score, reverse=True)

    sources = scored[:max_sources]
    combined_text = _build_combined_text(sources)
    prompt = _create_summary_prompt(query, combined_text, sources)

    context = ResearchContext(
        query=query,
        sources=sources,
        combined_text=combined_text,
        summary_prompt=prompt,
    )
    _set_cached_query(query, context)
    return context


def serialize_research_context(context: ResearchContext) -> dict[str, Any]:
    return {
        "query": context.query,
        "sources": [
            {
                "title": source.title,
                "url": source.url,
                "domain": source.domain,
                "relevance_score": round(source.relevance_score, 2),
                "snippets": source.snippets,
            }
            for source in context.sources
        ],
        "combined_text": context.combined_text,
    }
