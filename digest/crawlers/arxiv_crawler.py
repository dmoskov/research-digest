"""
arXiv paper crawler using the arXiv Atom API.

Fetches recent papers by category (e.g. econ.GN, cs.AI, q-bio.PE).
Respects arXiv ToS: 3-second delay between requests.

API docs: https://info.arxiv.org/help/api/index.html
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from xml.etree import ElementTree

import requests

from digest.crawlers.resilient import api_headers, resilient_get

logger = logging.getLogger(__name__)


# arXiv Atom namespace
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_ARXIV_NS = {"arxiv": "http://arxiv.org/schemas/atom"}


class ArxivCrawler:
    """Crawler for arXiv papers via the Atom API."""

    API_BASE = "http://export.arxiv.org/api/query"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(api_headers())

    def fetch_items(self, source_config: dict, days_back: int = 7) -> List[Dict]:
        """
        Fetch recent arXiv papers for configured categories.

        Args:
            source_config: Source row dict from the sources table. Expected keys:
                - crawl_config: {"categories": ["econ.GN", "cs.AI", ...]}
                - key: source key for tagging items
                - name: source display name
            days_back: Number of days to look back.

        Returns:
            List of item dicts compatible with store_item().
        """
        crawl_config = source_config.get("crawl_config") or {}
        categories = crawl_config.get("categories", [])

        if not categories:
            logger.warning(
                f"No categories configured for {source_config.get('key')}"
            )
            return []

        all_items = []
        for category in categories:
            items = self._fetch_category(category, source_config, days_back)
            all_items.extend(items)
            # arXiv ToS: 3-second delay between requests
            if len(categories) > 1:
                time.sleep(3)

        # Deduplicate by URL
        seen_urls = set()
        unique_items = []
        for item in all_items:
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                unique_items.append(item)

        return unique_items

    def _fetch_category(
        self, category: str, source_config: dict, days_back: int
    ) -> List[Dict]:
        """Fetch papers from a single arXiv category."""
        # arXiv API uses search_query for category filtering
        # submittedDate range for date filtering
        # Note: arXiv date filtering via API is limited; we fetch recent
        # papers and filter client-side
        params = {
            "search_query": f"cat:{category}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": 0,
            "max_results": 50,
        }

        cutoff = datetime.now() - timedelta(days=days_back)
        items = []

        try:
            response = resilient_get(
                self.session,
                self.API_BASE,
                params=params,
                timeout=self.timeout,
                source_label=f"arXiv:{category}",
            )

            root = ElementTree.fromstring(response.content)  # noqa: S314
            entries = root.findall("atom:entry", _ATOM_NS)

            for entry in entries:
                item = self._parse_entry(entry, source_config, cutoff)
                if item:
                    items.append(item)

        except requests.exceptions.RequestException as e:
            logger.error(f"arXiv error for {category}: {e}")
        except ElementTree.ParseError as e:
            logger.error(f"arXiv XML parse error for {category}: {e}")

        return items

    def _parse_entry(
        self, entry, source_config: dict, cutoff: datetime
    ) -> Optional[Dict]:
        """Parse an arXiv Atom entry to a pipeline item dict."""
        title = _extract_title(entry)
        if not title:
            return None

        url = _extract_url(entry)
        if not url:
            return None

        pub_date = _extract_pub_date(entry)
        if _is_before_cutoff(pub_date, cutoff):
            return None

        abstract = _extract_abstract(entry)
        authors = _extract_authors(entry)
        categories = _extract_categories(entry)
        arxiv_id = url.split("/abs/")[-1] if "/abs/" in url else ""

        return {
            "title": title,
            "abstract": abstract,
            "authors": ", ".join(authors),
            "url": url,
            "date": pub_date,
            "source": source_config.get("key", "arxiv"),
            "source_name": source_config.get("name", "arXiv"),
            "source_id": arxiv_id,
            "raw_metadata": {
                "arxiv_id": arxiv_id,
                "categories": categories,
            },
        }


def _extract_title(entry) -> str:
    """Extract and normalize the title from an Atom entry, or return empty string."""
    title_elem = entry.find("atom:title", _ATOM_NS)
    if title_elem is None or not title_elem.text:
        return ""
    return " ".join(title_elem.text.strip().split())


def _extract_url(entry) -> str:
    """Extract the paper URL, preferring the abstract page link."""
    for link_elem in entry.findall("atom:link", _ATOM_NS):
        if link_elem.get("type") == "text/html":
            return link_elem.get("href", "")
    id_elem = entry.find("atom:id", _ATOM_NS)
    if id_elem is not None and id_elem.text:
        return id_elem.text.strip()
    return ""


def _extract_pub_date(entry) -> Optional[str]:
    """Extract the publication date as a YYYY-MM-DD string."""
    published_elem = entry.find("atom:published", _ATOM_NS)
    if published_elem is not None and published_elem.text:
        return published_elem.text[:10]
    return None


def _is_before_cutoff(pub_date: Optional[str], cutoff: datetime) -> bool:
    """Return True if pub_date is before the cutoff date."""
    if not pub_date or not cutoff:
        return False
    try:
        return datetime.strptime(pub_date, "%Y-%m-%d") < cutoff
    except ValueError:
        logger.debug("Failed to parse pub_date '%s' for arXiv entry", pub_date)
        return False


def _extract_abstract(entry) -> str:
    """Extract and normalize the abstract/summary text."""
    summary_elem = entry.find("atom:summary", _ATOM_NS)
    if summary_elem is not None and summary_elem.text:
        return " ".join(summary_elem.text.strip().split())
    return ""


def _extract_authors(entry) -> List[str]:
    """Extract author names from the entry."""
    authors = []
    for author_elem in entry.findall("atom:author", _ATOM_NS):
        name_elem = author_elem.find("atom:name", _ATOM_NS)
        if name_elem is not None and name_elem.text:
            authors.append(name_elem.text.strip())
    return authors


def _extract_categories(entry) -> List[str]:
    """Extract arXiv category terms from the entry."""
    categories = []
    for cat_elem in entry.findall("atom:category", _ATOM_NS):
        term = cat_elem.get("term", "")
        if term:
            categories.append(term)
    return categories
