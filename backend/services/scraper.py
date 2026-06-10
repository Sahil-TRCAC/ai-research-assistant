from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

logger = logging.getLogger("ai_research.scraper")

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

DEFAULT_TIMEOUT = 10
MAX_CONTENT_CHARS = 20000

BAD_CLASS_ID_PATTERNS = (
    "nav", "footer", "header", "sidebar", "ads", "advert", "cookie", "subscribe",
    "breadcrumb", "modal", "popup", "banner", "promo",
)

@dataclass
class ScrapedPage:
    url: str
    title: str
    domain: str
    headings: list[str]
    paragraphs: list[str]
    lists: list[list[str]]
    text: str


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# Domains that should never be used as research sources (search engines, ad redirects)
_BLOCKED_DOMAINS = {
    "duckduckgo.com",
    "google.com",
    "bing.com",
    "yahoo.com",
    "yandex.com",
    "baidu.com",
    "ask.com",
}


def validate_source_url(url: str) -> bool:
    if not isinstance(url, str) or not url.strip():
        return False

    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        return False
    if not parsed.netloc:
        return False

    hostname = (parsed.hostname or "").lower()
    if hostname.startswith("localhost"):
        return False
    if hostname in ("127.0.0.1", "0.0.0.0"):
        return False
    if hostname in ("[::1]", "::1"):
        return False

    # Block search engines and ad redirect domains
    for blocked in _BLOCKED_DOMAINS:
        if hostname == blocked or hostname.endswith("." + blocked):
            return False

    return True


def fetch_html(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    if not validate_source_url(url):
        raise ValueError("Invalid or unsupported URL.")

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
    except RequestException as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        raise RuntimeError(f"Unable to fetch URL: {url}") from exc

    content_type = response.headers.get("Content-Type", "")
    if "html" not in content_type:
        raise RuntimeError("URL did not return HTML content.")

    return response.text


def _remove_irrelevant_nodes(soup: BeautifulSoup) -> None:
    selectors = [
        "script",
        "style",
        "noscript",
        "iframe",
        "header",
        "footer",
        "nav",
        "aside",
        "form",
        "svg",
        "button",
        "figure",
        "picture",
        ".cookie",
        ".consent",
        ".promo",
        ".modal",
        ".advert",
        ".banner",
        ".subscription",
        ".newsletter",
    ]
    for selector in selectors:
        for element in soup.select(selector):
            element.decompose()

    # Snapshot the list first so decomposing elements doesn't invalidate the iterator
    for element in list(soup.find_all(True)):
        if element.parent is None:
            # Already decomposed (was a child of a decomposed parent)
            continue
        class_attr = " ".join(element.get("class", []) or [])
        id_attr = element.get("id", "") or ""
        if any(pattern in class_attr.lower() for pattern in BAD_CLASS_ID_PATTERNS):
            element.decompose()
        elif any(pattern in id_attr.lower() for pattern in BAD_CLASS_ID_PATTERNS):
            element.decompose()


def _select_best_content_node(soup: BeautifulSoup) -> BeautifulSoup:
    candidates = soup.select("article, main, section, div")
    best_node = soup.body or soup
    best_score = 0

    for node in candidates:
        text = _normalize_text(node.get_text(separator=" ", strip=True))
        if len(text) < 200:
            continue

        heading_count = len(node.find_all(re.compile(r"^h[1-6]$")))
        link_density = len(node.find_all("a")) / max(1, len(node.get_text(separator=" ", strip=True).split()))
        score = len(text) + heading_count * 200 - int(link_density * 100)

        if score > best_score:
            best_score = score
            best_node = node

    return best_node


def _extract_text_blocks(root: BeautifulSoup) -> tuple[list[str], list[str], list[list[str]]]:
    headings: list[str] = []
    paragraphs: list[str] = []
    lists: list[list[str]] = []
    seen: set[str] = set()

    for element in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        text = _normalize_text(element.get_text(separator=" ", strip=True))
        if text and text not in seen:
            headings.append(text)
            seen.add(text)

    for paragraph in root.find_all("p"):
        text = _normalize_text(paragraph.get_text(separator=" ", strip=True))
        if len(text) >= 40 and text not in seen:
            paragraphs.append(text)
            seen.add(text)

    for list_root in root.find_all(["ul", "ol"]):
        items = [
            _normalize_text(li.get_text(separator=" ", strip=True))
            for li in list_root.find_all("li")
            if li.get_text(strip=True)
        ]
        items = [item for item in items if len(item) > 15]
        if items:
            key = "|".join(items)
            if key not in seen:
                lists.append(items)
                seen.add(key)

    return headings, paragraphs, lists


def scrape_url(url: str, timeout: int = DEFAULT_TIMEOUT) -> ScrapedPage:
    html = fetch_html(url, timeout=timeout)
    soup = BeautifulSoup(html, "html.parser")

    # Extract title BEFORE removing nodes (avoids soup.title becoming None)
    title_tag = soup.find("title")
    raw_title = title_tag.get_text(separator=" ", strip=True) if title_tag else ""

    _remove_irrelevant_nodes(soup)

    title = _normalize_text(raw_title)
    if not title:
        first_heading = soup.find(re.compile(r"^h[1-3]$"))
        title = _normalize_text(first_heading.get_text(separator=" ", strip=True)) if first_heading else "Untitled"

    content_root = _select_best_content_node(soup)
    headings, paragraphs, lists = _extract_text_blocks(content_root)

    raw_text = "\n\n".join(
        [title] + headings + paragraphs + ["\n".join(item) for item in lists]
    )
    raw_text = _normalize_text(raw_text)[:MAX_CONTENT_CHARS]

    domain = urlparse(url).netloc

    return ScrapedPage(
        url=url,
        title=title or "Untitled",
        domain=domain,
        headings=headings,
        paragraphs=paragraphs,
        lists=lists,
        text=raw_text,
    )


def build_source_preview(page: ScrapedPage) -> dict[str, Any]:
    preview = {
        "title": page.title,
        "url": page.url,
        "domain": page.domain,
        "headings": page.headings[:10],
        "paragraph_count": len(page.paragraphs),
    }
    return preview
