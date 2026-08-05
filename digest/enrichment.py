"""
Content enrichment module.

Fetches full article text from URLs to replace thin RSS summaries before
classification. Dispatches to domain-specific extractors for known sources
(Substack, NBER, arXiv) and falls back to generic article extraction.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from digest.crawlers.resilient import resilient_get

logger = logging.getLogger(__name__)

# Items with content shorter than this are candidates for enrichment
_MIN_CONTENT_LEN = 500

# Enriched text is truncated to this length
_MAX_TEXT_LEN = 5000

# Tags to strip before extracting text
_STRIP_TAGS = ("script", "style", "nav", "footer", "aside", "header")


class DomainRateLimiter:
    """Thread-safe per-domain rate limiter."""

    def __init__(
        self, default_delay: float = 1.0, overrides: Optional[Dict[str, float]] = None
    ):
        self._default_delay = default_delay
        self._overrides = overrides or {}
        self._last_request: Dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, domain: str) -> None:
        delay = self._overrides.get(domain, self._default_delay)
        with self._lock:
            last = self._last_request.get(domain, 0.0)
            elapsed = time.monotonic() - last
            if elapsed < delay:
                time.sleep(delay - elapsed)
            self._last_request[domain] = time.monotonic()


_rate_limiter = DomainRateLimiter(
    default_delay=1.0,
    overrides={"arxiv.org": 3.0},
)


# ---------------------------------------------------------------------------
# Domain-specific extractors
# ---------------------------------------------------------------------------


def _clean_soup(soup: BeautifulSoup) -> None:
    """Remove boilerplate tags in-place."""
    for tag_name in _STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()


def _text_from(element) -> str:
    """Extract and truncate text from a BeautifulSoup element."""
    if element is None:
        return ""
    text = element.get_text(separator=" ", strip=True)
    return text[:_MAX_TEXT_LEN]


def _is_substack(soup: BeautifulSoup) -> bool:
    """Detect Substack pages including custom domains (slowboring.com, etc.)."""
    meta = soup.find("meta", attrs={"content": "Substack"})
    if meta:
        return True
    for link in soup.find_all("link", href=True):
        if "substack.com" in link["href"]:
            return True
    return False


def _extract_substack(soup: BeautifulSoup) -> str:
    _clean_soup(soup)
    for selector in ("div.body.markup", "div.available-content"):
        el = soup.select_one(selector)
        if el:
            return _text_from(el)
    article = soup.find("article")
    if article:
        return _text_from(article)
    return ""


def _extract_nber(soup: BeautifulSoup) -> str:
    _clean_soup(soup)
    el = soup.select_one("div.page-header__intro-inner")
    if el:
        return _text_from(el)
    return ""


def _extract_arxiv(soup: BeautifulSoup) -> str:
    _clean_soup(soup)
    el = soup.select_one("blockquote.abstract.mathjax")
    if el:
        return _text_from(el)
    return ""


def _extract_generic(soup: BeautifulSoup) -> str:
    _clean_soup(soup)
    for selector in (
        "article",
        "main",
        "[role=main]",
        ".post-content",
        ".entry-content",
    ):
        el = soup.select_one(selector)
        if el:
            return _text_from(el)
    # Fallback: largest <div> by text length
    best = ""
    for div in soup.find_all("div"):
        text = div.get_text(separator=" ", strip=True)
        if len(text) > len(best):
            best = text
    return best[:_MAX_TEXT_LEN]


def _get_extractor(url: str, soup: BeautifulSoup):
    """Return the appropriate extractor function for a URL/page."""
    domain = urlparse(url).netloc.lower()
    if "nber.org" in domain:
        return _extract_nber
    if "arxiv.org" in domain:
        return _extract_arxiv
    if "substack.com" in domain or _is_substack(soup):
        return _extract_substack
    return _extract_generic


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enrich_item(item: Dict, session: requests.Session) -> bool:
    """Fetch full text for a single item and update its content in-place.

    Returns True if enrichment succeeded and content was replaced.
    """
    url = item.get("url")
    if not url:
        return False

    domain = urlparse(url).netloc.lower()
    _rate_limiter.wait(domain)

    try:
        response = resilient_get(
            session,
            url,
            timeout=20,
            max_retries=2,
            source_label=f"enrich:{domain}",
        )
    except requests.exceptions.RequestException as exc:
        logger.warning("Enrichment fetch failed for %s: %s", url, exc)
        return False

    try:
        soup = BeautifulSoup(response.content, "html.parser")
        extractor = _get_extractor(url, soup)
        text = extractor(soup)
    except Exception as exc:
        logger.warning("Enrichment parse failed for %s: %s", url, exc)
        return False

    if not text:
        return False

    # Only replace if extracted text is longer than what we already have
    existing = item.get("content", "") or item.get("abstract", "") or ""
    if len(text) <= len(existing):
        return False

    item["content"] = text
    return True


_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

_thread_local = threading.local()


def _get_thread_session() -> requests.Session:
    """Return a per-thread requests.Session (thread-safe)."""
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": _USER_AGENT})
        _thread_local.session = s
    return s


def _enrich_item_thread(item: Dict) -> bool:
    """Thread-pool wrapper: enrich using a per-thread session."""
    return enrich_item(item, _get_thread_session())


def enrich_items(
    items: List[Dict],
    session: Optional[requests.Session] = None,
    max_workers: int = 4,
) -> None:
    """Enrich a list of items in-place with full article text.

    Skips items that already have content longer than _MIN_CONTENT_LEN.
    Failures on individual items are logged and do not affect others.
    """
    candidates = [
        item for item in items if len(item.get("content", "") or "") < _MIN_CONTENT_LEN
    ]

    if not candidates:
        return

    logger.info("Enriching %d/%d items with thin content", len(candidates), len(items))

    if session is not None:
        # Caller-provided session: run single-threaded to stay safe
        for item in candidates:
            try:
                if enrich_item(item, session):
                    item["_enriched"] = True
            except Exception as exc:
                logger.warning(
                    "Unexpected enrichment error for %s: %s", item.get("url", "?"), exc
                )
        return

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_enrich_item_thread, item): item for item in candidates}
        for future in as_completed(futures):
            item = futures[future]
            try:
                if future.result():
                    item["_enriched"] = True
            except Exception as exc:
                logger.warning(
                    "Unexpected enrichment error for %s: %s",
                    item.get("url", "?"),
                    exc,
                )
