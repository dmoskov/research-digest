"""
HTML scraper crawler for sources without RSS feeds.

Scrapes article listings from HTML pages using per-source CSS selectors
stored in the source's crawl_config. Supports sites like Anthropic Research,
Epoch AI, and Gates Foundation that don't provide RSS/Atom feeds.

Each source's crawl_config should contain:
    listing_url: URL of the page to scrape for article links
    article_selector: CSS selector for article containers
    title_selector: CSS selector for title within article container
    url_selector: CSS selector for link element (extracts href)
    date_selector: CSS selector for date text (optional)
    date_format: strftime format string for parsing dates (optional)
    content_selector: CSS selector for description/summary (optional)
    author_selector: CSS selector for author (optional)
    base_url: Base URL for resolving relative links
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from digest.crawlers.resilient import cloudflare_get, html_headers, resilient_get

logger = logging.getLogger(__name__)

# Common month abbreviation mappings for flexible date parsing
_MONTH_NAMES = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _parse_flexible_date(text: str, fmt: Optional[str] = None) -> Optional[str]:
    """Parse a date string into YYYY-MM-DD format.

    Tries the provided strftime format first, then falls back to common
    patterns like "Mar 5, 2026", "2026-03-05", "March 5, 2026", etc.

    Args:
        text: Raw date string from the page.
        fmt: Optional strftime format to try first.

    Returns:
        Date string in YYYY-MM-DD format, or None if unparsable.
    """
    text = text.strip()
    if not text:
        return None

    # Try explicit format first
    if fmt:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # ISO format: 2026-03-05 or 2026-03-05T...
    iso_match = re.match(r"(\d{4}-\d{2}-\d{2})", text)
    if iso_match:
        return iso_match.group(1)

    # "Mon DD, YYYY" or "Month DD, YYYY" (e.g., "Mar 5, 2026")
    month_day_year = re.match(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", text)
    if month_day_year:
        month_str, day, year = month_day_year.groups()
        month_num = _MONTH_NAMES.get(month_str.lower())
        if month_num:
            return f"{year}-{month_num:02d}-{int(day):02d}"

    # "DD Mon YYYY" (e.g., "5 Mar 2026")
    day_month_year = re.match(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", text)
    if day_month_year:
        day, month_str, year = day_month_year.groups()
        month_num = _MONTH_NAMES.get(month_str.lower())
        if month_num:
            return f"{year}-{month_num:02d}-{int(day):02d}"

    # "Mon DD" without year (e.g., "Mar 10") — assume current year
    month_day_only = re.match(r"([A-Za-z]+)\s+(\d{1,2})$", text)
    if month_day_only:
        month_str, day = month_day_only.groups()
        month_num = _MONTH_NAMES.get(month_str.lower())
        if month_num:
            year = datetime.now().year
            return f"{year}-{month_num:02d}-{int(day):02d}"

    return None


class HtmlScraperCrawler:
    """Scrapes article listings from HTML pages using CSS selectors."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(html_headers())

    def fetch_items(self, source_config: dict, days_back: int = 7) -> List[Dict]:
        """Fetch articles by scraping an HTML listing page.

        Args:
            source_config: Source row dict from the sources table.
                Must have crawl_config with at least listing_url,
                article_selector, title_selector, url_selector, and base_url.
            days_back: Number of days to look back (used for date filtering).

        Returns:
            List of item dicts compatible with store_item().
        """
        cfg = self._resolve_crawl_config(source_config)
        if cfg is None:
            return []

        source_key = source_config.get("key", "html_scraper")
        source_name = source_config.get("name", "")
        label = f"HtmlScraper:{source_key}"

        soup = self._fetch_listing_html(cfg["listing_url"], label)
        if soup is None:
            return []

        articles = soup.select(cfg["article_selector"])
        if not articles:
            logger.warning(
                "%s: No articles found with selector '%s' on %s",
                label,
                cfg["article_selector"],
                cfg["listing_url"],
            )
            return []

        items = self._collect_items(
            articles,
            cfg=cfg,
            source_key=source_key,
            source_name=source_name,
            label=label,
            days_back=days_back,
        )
        logger.info("%s: Found %d articles from %s", label, len(items), cfg["listing_url"])
        return items

    def _resolve_crawl_config(self, source_config: dict) -> Optional[Dict]:
        """Parse and validate crawl_config from a source config dict.

        Returns the crawl_config dict with all selector keys, or None if
        required fields are missing.
        """
        crawl_config = source_config.get("crawl_config", {})
        if isinstance(crawl_config, str):
            import json

            crawl_config = json.loads(crawl_config)

        listing_url = crawl_config.get("listing_url")
        if not listing_url:
            logger.error(
                "No listing_url in crawl_config for source %s",
                source_config.get("key"),
            )
            return None

        article_selector = crawl_config.get("article_selector")
        title_selector = crawl_config.get("title_selector")
        url_selector = crawl_config.get("url_selector")

        if not all([article_selector, title_selector, url_selector]):
            logger.error(
                "Missing required selectors in crawl_config for source %s "
                "(need article_selector, title_selector, url_selector)",
                source_config.get("key"),
            )
            return None

        return {
            "listing_url": listing_url,
            "article_selector": article_selector,
            "title_selector": title_selector,
            "url_selector": url_selector,
            "base_url": crawl_config.get("base_url", listing_url),
            "date_selector": crawl_config.get("date_selector"),
            "date_format": crawl_config.get("date_format"),
            "content_selector": crawl_config.get("content_selector"),
            "author_selector": crawl_config.get("author_selector"),
        }

    def _fetch_listing_html(self, listing_url: str, label: str) -> Optional[BeautifulSoup]:
        """Fetch a listing page and return parsed HTML.

        Falls back to curl_cffi on 403 (Cloudflare protection).
        Returns None on any failure.
        """
        try:
            response = resilient_get(
                self.session,
                listing_url,
                timeout=self.timeout,
                source_label=label,
            )
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                logger.info("%s: 403 from normal request, trying curl_cffi bypass", label)
                response = cloudflare_get(listing_url, timeout=self.timeout, source_label=label)
                if response is None or response.status_code != 200:
                    logger.error("Failed to fetch %s (Cloudflare block): %s", listing_url, e)
                    return None
            else:
                logger.error("Failed to fetch %s: %s", listing_url, e)
                return None
        except requests.exceptions.RequestException as e:
            logger.error("Failed to fetch %s: %s", listing_url, e)
            return None

        return BeautifulSoup(response.text, "html.parser")

    def _collect_items(
        self,
        articles,
        *,
        cfg: Dict,
        source_key: str,
        source_name: str,
        label: str,
        days_back: Optional[int],
    ) -> List[Dict]:
        """Parse article elements, dedup by URL, and apply date filtering."""
        cutoff_date = None
        if days_back:
            cutoff_date = (datetime.now() - timedelta(days=days_back)).date()

        items: List[Dict] = []
        seen_urls: set = set()

        for article in articles:
            try:
                item = self._parse_article(
                    article,
                    base_url=cfg["base_url"],
                    title_selector=cfg["title_selector"],
                    url_selector=cfg["url_selector"],
                    date_selector=cfg["date_selector"],
                    date_format=cfg["date_format"],
                    content_selector=cfg["content_selector"],
                    author_selector=cfg["author_selector"],
                    source_key=source_key,
                    source_name=source_name,
                )
                if not item:
                    continue

                if item["url"] in seen_urls:
                    continue
                seen_urls.add(item["url"])

                if cutoff_date and item.get("date"):
                    try:
                        item_date = datetime.strptime(item["date"], "%Y-%m-%d").date()
                        if item_date < cutoff_date:
                            continue
                    except ValueError:
                        pass

                items.append(item)
            except Exception as e:
                logger.debug("%s: Error parsing article element: %s", label, e)
                continue

        return items

    def _parse_article(
        self,
        article,
        *,
        base_url: str,
        title_selector: str,
        url_selector: str,
        date_selector: Optional[str],
        date_format: Optional[str],
        content_selector: Optional[str],
        author_selector: Optional[str],
        source_key: str,
        source_name: str,
    ) -> Optional[Dict]:
        """Parse a single article element into an item dict.

        Returns item dict or None if essential fields (title, url) are missing.
        """
        title = self._extract_title(article, title_selector)
        if not title:
            return None

        url = self._extract_url(article, url_selector, base_url)
        if not url:
            return None

        return {
            "title": title,
            "url": url,
            "date": self._extract_date(article, date_selector, date_format),
            "content": self._extract_optional_text(article, content_selector),
            "author": self._extract_optional_text(article, author_selector),
            "source": source_key,
            "source_name": source_name,
            "images": [],
        }

    @staticmethod
    def _extract_title(article, selector: str) -> Optional[str]:
        """Extract title text from an article element."""
        el = article.select_one(selector)
        if not el:
            return None
        text = el.get_text(strip=True)
        return text or None

    @staticmethod
    def _extract_url(article, selector: str, base_url: str) -> Optional[str]:
        """Extract and resolve the article URL.

        Checks inside the article first, then falls back to the article
        element itself (e.g. Epoch AI's <a class="blog-post-card">).
        """
        url_el = article.select_one(selector)
        href = url_el.get("href", "") if url_el else article.get("href", "")
        if not href:
            return None
        return urljoin(base_url, href)

    @staticmethod
    def _extract_date(article, selector: Optional[str], date_format: Optional[str]) -> Optional[str]:
        """Try all matching date elements, return the first that parses."""
        if not selector:
            return None
        for el in article.select(selector):
            parsed = _parse_flexible_date(el.get_text(strip=True), fmt=date_format)
            if parsed:
                return parsed
        return None

    @staticmethod
    def _extract_optional_text(article, selector: Optional[str]) -> str:
        """Extract text from an optional selector, returning empty string if absent."""
        if not selector:
            return ""
        el = article.select_one(selector)
        return el.get_text(strip=True) if el else ""
