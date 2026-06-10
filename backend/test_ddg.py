import sys
sys.path.insert(0, ".")
from services.scraper import scrape_url, validate_source_url
from services.research_engine import _duckduckgo_search

print("=== Testing DuckDuckGo search ===")
urls = _duckduckgo_search("python programming", max_results=8)
print("URLs found:", len(urls))
for u in urls:
    print(" -", u)

print("\n=== Testing scraping first 3 URLs ===")
for url in urls[:3]:
    try:
        page = scrape_url(url)
        print("OK:", url[:60], "| text_len:", len(page.text), "| paragraphs:", len(page.paragraphs))
    except Exception as e:
        print("FAIL:", url[:60], "|", type(e).__name__, str(e)[:80])
