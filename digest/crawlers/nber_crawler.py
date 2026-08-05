"""
NBER Working Papers crawler.

Fetches recent working papers from NBER RSS feed.
Note: NBER's listing page uses JavaScript rendering, so we use RSS feed instead.
Individual paper dates are extracted by fetching paper detail pages.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup

from digest.crawlers.resilient import resilient_get, rss_headers

logger = logging.getLogger(__name__)


class NBERCrawler:
    """Crawler for NBER working papers via RSS feed with date extraction."""

    RSS_URL = "https://back.nber.org/rss/new.xml"
    PAPER_URL_BASE = "https://www.nber.org/papers/"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(rss_headers())

    def fetch_papers(
        self,
        per_page: int = 50,
        cutoff_date: Optional[datetime] = None,
        fetch_dates: bool = False,
    ) -> List[Dict]:
        """
        Fetch recent NBER working papers from RSS feed.

        Args:
            per_page: Maximum number of papers to return
            cutoff_date: Only fetch papers published after this date
            fetch_dates: If True, fetch actual publication dates from paper pages (slower)

        Returns:
            List of paper dictionaries with keys: title, abstract, authors, url, date, paper_number
        """
        papers = []

        try:
            response = resilient_get(
                self.session,
                self.RSS_URL,
                timeout=self.timeout,
                source_label="NBER RSS",
            )

            root = ElementTree.fromstring(response.content)  # noqa: S314
            items = root.findall(".//item")

            for item in items:
                try:
                    paper = self._parse_rss_item(item)
                    if paper:
                        # Extract date from paper page if requested
                        if fetch_dates:
                            actual_date = self._fetch_paper_date(paper["url"])
                            if actual_date:
                                paper["date"] = actual_date

                        # Filter by date if specified
                        if cutoff_date and paper.get("date"):
                            paper_date = datetime.strptime(paper["date"], "%Y-%m-%d")
                            if paper_date <= cutoff_date:
                                continue

                        papers.append(paper)

                        # Limit to per_page
                        if len(papers) >= per_page:
                            break

                except Exception as e:
                    logger.warning(f"Error parsing RSS item: {e}")
                    continue

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching NBER RSS feed: {e}")
            return []
        except ElementTree.ParseError as e:
            logger.error(f"Error parsing NBER RSS XML: {e}")
            return []

        return papers

    def _parse_rss_item(self, item) -> Optional[Dict]:
        """Parse a single RSS feed item."""
        paper = {}

        # Extract title (includes authors)
        title_elem = item.find("title")
        if title_elem is not None and title_elem.text:
            # NBER format: "Title -- by Author1, Author2"
            full_title = title_elem.text.strip()
            if " -- by " in full_title:
                paper["title"], paper["authors"] = full_title.split(" -- by ", 1)
                paper["title"] = paper["title"].strip()
                paper["authors"] = paper["authors"].strip()
            else:
                paper["title"] = full_title
                paper["authors"] = ""
        else:
            return None

        # Extract link
        link_elem = item.find("link")
        if link_elem is not None and link_elem.text:
            paper["url"] = link_elem.text.strip()
            # Remove #fromrss suffix if present
            paper["url"] = paper["url"].replace("#fromrss", "")
        else:
            return None

        # Extract description (abstract)
        description_elem = item.find("description")
        if description_elem is not None and description_elem.text:
            paper["abstract"] = description_elem.text.strip()
        else:
            paper["abstract"] = ""

        # Extract images from enclosure tags (media:content)
        paper["images"] = self._extract_images(item)

        # Extract paper number from URL
        if paper.get("url"):
            match = re.search(r"/papers/(w\d+)", paper["url"])
            if match:
                paper["paper_number"] = match.group(1)

        # RSS feed doesn't include dates - use current date as approximation
        # or fetch from paper page
        paper["date"] = datetime.now().strftime("%Y-%m-%d")

        paper["source"] = "nber"

        return paper

    def _fetch_paper_date(self, url: str) -> Optional[str]:
        """
        Fetch publication date from paper detail page.

        Args:
            url: URL of paper detail page

        Returns:
            Date string in YYYY-MM-DD format, or None if not found
        """
        try:
            response = resilient_get(
                self.session,
                url,
                timeout=self.timeout,
                source_label="NBER paper date",
            )

            soup = BeautifulSoup(response.content, "html.parser")

            # Look for <time> tag with date
            time_elem = soup.find("time")
            if time_elem:
                date_text = time_elem.get_text(strip=True)
                return self._parse_date(date_text)

            # Fallback: look for "Issue Date" text
            for div in soup.find_all("div"):
                text = div.get_text(strip=True)
                if "Issue Date" in text:
                    # Extract date following "Issue Date"
                    match = re.search(r"Issue Date\s*([A-Za-z]+\s+\d{4})", text)
                    if match:
                        return self._parse_date(match.group(1))

        except Exception as e:
            logger.warning(f"Could not fetch date for {url}: {e}")

        return None

    def _extract_images(self, item) -> List[str]:
        """
        Extract image URLs from RSS item.

        Looks for images in enclosure tags and media:content elements.
        Returns max 2 images to avoid cluttering the digest.

        Args:
            item: RSS feed item element

        Returns:
            List of image URLs (max 2)
        """
        images = []

        # Check for enclosure tags with image types
        for enclosure in item.findall("enclosure"):
            enclosure_type = enclosure.get("type", "")
            if enclosure_type.startswith("image/"):
                url = enclosure.get("url")
                if url:
                    images.append(url)

        # Check for media:content tags (common in RSS feeds)
        # Use explicit namespace
        media_ns = {"media": "http://search.yahoo.com/mrss/"}
        for media_content in item.findall(".//media:content", media_ns):
            medium = media_content.get("medium", "")
            media_type = media_content.get("type", "")
            if medium == "image" or media_type.startswith("image/"):
                url = media_content.get("url")
                if url:
                    images.append(url)

        # Also check media:thumbnail
        for media_thumb in item.findall(".//media:thumbnail", media_ns):
            url = media_thumb.get("url")
            if url:
                images.append(url)

        # Remove duplicates while preserving order
        seen = set()
        unique_images = []
        for img in images:
            if img not in seen:
                seen.add(img)
                unique_images.append(img)

        # Return max 2 images
        return unique_images[:2]

    def _parse_date(self, date_text: str) -> str:
        """Parse date from various NBER formats."""
        # Try YYYY-MM-DD format
        if re.match(r"\d{4}-\d{2}-\d{2}", date_text):
            return date_text

        # Try "Month YYYY" format (most common on NBER)
        try:
            dt = datetime.strptime(date_text, "%B %Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            logger.debug("NBER date '%s' doesn't match '%%B %%Y' format", date_text)
            pass

        # Try "Month DD, YYYY" format
        try:
            dt = datetime.strptime(date_text, "%B %d, %Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            logger.debug(
                "NBER date '%s' doesn't match '%%B %%d, %%Y' format", date_text
            )
            pass

        # Try "Mon YYYY" format (abbreviated)
        try:
            dt = datetime.strptime(date_text, "%b %Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            logger.debug("NBER date '%s' doesn't match '%%b %%Y' format", date_text)
            pass

        # Fallback to current date
        return datetime.now().strftime("%Y-%m-%d")

    def fetch_multiple_pages(
        self,
        max_pages: int = 3,
        per_page: int = 50,
        cutoff_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Fetch papers from RSS feed.

        Note: RSS feed doesn't support pagination, so max_pages parameter is for
        API compatibility. We fetch all available papers from the feed and filter
        by cutoff_date.

        Args:
            max_pages: Ignored (RSS has no pagination)
            per_page: Maximum number of papers to return (default: 50)
            cutoff_date: Only return papers published after this date

        Returns:
            List of papers published after cutoff_date
        """
        # Fetch papers from RSS (no pagination)
        papers = self.fetch_papers(
            per_page=per_page, cutoff_date=cutoff_date, fetch_dates=False
        )

        if not papers and cutoff_date:
            logger.info(
                f"No NBER papers found after cutoff date {cutoff_date.strftime('%Y-%m-%d')}"
            )
            logger.info(
                "This is expected if cutoff is very recent (RSS shows latest ~40 papers)"
            )

        return papers

    def fetch_items(self, source_config: dict, days_back: int = 7) -> List[Dict]:
        """
        Adapter method for source-driven dispatch.

        Delegates to fetch_multiple_pages with appropriate arguments.

        Args:
            source_config: Source row dict from the sources table.
            days_back: Number of days to look back.

        Returns:
            List of paper dicts compatible with store_item().
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        per_page = source_config.get("crawl_config", {}).get("per_page", 50)
        return self.fetch_multiple_pages(per_page=per_page, cutoff_date=cutoff_date)


def main():
    """Test the NBER crawler."""
    crawler = NBERCrawler()

    # Test fetching papers without date filtering
    print("Test 1: Fetching recent papers (no date filter)")
    papers = crawler.fetch_papers(per_page=5, fetch_dates=False)
    print(f"Found {len(papers)} papers")
    for paper in papers[:3]:
        print(f"\n  Title: {paper['title']}")
        print(f"  Authors: {paper['authors']}")
        print(f"  Date: {paper.get('date', 'N/A')}")
        print(f"  Paper #: {paper.get('paper_number', 'N/A')}")
        print(f"  URL: {paper['url']}")

    # Test with cutoff date (30 days back)
    print("\n" + "=" * 70)
    print("Test 2: Fetching papers with 30-day cutoff")
    cutoff = datetime.now() - timedelta(days=30)
    papers = crawler.fetch_multiple_pages(per_page=10, cutoff_date=cutoff)
    print(f"Found {len(papers)} papers published after {cutoff.strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    main()
