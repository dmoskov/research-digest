"""
Academic journal crawler with layered fetching strategy.

Layer 1: RSS/Atom feed (if feed_url exists)
Layer 2: CrossRef API by ISSN (free, no key required)
Layer 3: OpenAlex API for enrichment (free, no key required)

CrossRef docs: https://api.crossref.org/swagger-ui/index.html
OpenAlex docs: https://docs.openalex.org/
"""

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from xml.etree import ElementTree

import requests

from digest.crawlers.resilient import api_headers, bot_contact, resilient_get

logger = logging.getLogger(__name__)




class AcademicJournalCrawler:
    """Crawler for academic journals via CrossRef, OpenAlex, or RSS."""

    CROSSREF_BASE = "https://api.crossref.org"
    OPENALEX_BASE = "https://api.openalex.org"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(api_headers())

    def fetch_items(self, source_config: dict, days_back: int = 7) -> List[Dict]:
        """
        Fetch recent items from an academic journal source.

        Dispatches to the appropriate fetching method based on crawler_type
        and available config fields.

        Args:
            source_config: Source row dict from the sources table. Expected keys:
                - crawler_type: 'crossref', 'openalex', or 'rss'
                - issn: ISSN for CrossRef lookup
                - openalex_source_id: OpenAlex source ID
                - feed_url: RSS/Atom feed URL
                - key: source key for tagging items
                - name: source display name
            days_back: Number of days to look back.

        Returns:
            List of item dicts compatible with store_item().
        """
        crawler_type = source_config.get("crawler_type", "crossref")
        items = []

        # Layer 1: Try RSS if feed_url is available
        if source_config.get("feed_url") and crawler_type == "rss":
            items = self._fetch_via_rss(source_config, days_back)
            if items:
                return items

        # Layer 2: CrossRef by ISSN
        if source_config.get("issn") and crawler_type in ("crossref", "rss"):
            items = self._fetch_via_crossref(source_config, days_back)

        # Layer 3: OpenAlex by source ID
        if source_config.get("openalex_source_id") and crawler_type == "openalex":
            items = self._fetch_via_openalex(source_config, days_back)

        # Enrich with OpenAlex abstracts if we have the ID and items lack abstracts
        if source_config.get("openalex_source_id") and items:
            items_needing_abstract = [i for i in items if not i.get("abstract")]
            if items_needing_abstract:
                self._enrich_abstracts_openalex(items_needing_abstract, source_config)

        return items

    def _fetch_via_crossref(self, source_config: dict, days_back: int) -> List[Dict]:
        """Fetch works from CrossRef API by ISSN."""
        issn = source_config["issn"]
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        url = f"{self.CROSSREF_BASE}/works"
        params = {
            "filter": f"issn:{issn},from-created-date:{from_date}",
            "rows": 200,
            "sort": "published",
            "order": "desc",
            "mailto": bot_contact(),
        }

        items = []
        try:
            response = resilient_get(
                self.session,
                url,
                params=params,
                timeout=self.timeout,
                source_label=f"CrossRef:{source_config['key']}",
            )
            data = response.json()

            for work in data.get("message", {}).get("items", []):
                item = self._crossref_work_to_item(work, source_config)
                if item:
                    items.append(item)

            # Respect rate limits
            time.sleep(1.0)

        except requests.exceptions.RequestException as e:
            logger.error(f"CrossRef error for {source_config['key']}: {e}")
        except (ValueError, KeyError) as e:
            logger.error(f"CrossRef parse error for {source_config['key']}: {e}")

        return items

    def _fetch_via_openalex(self, source_config: dict, days_back: int) -> List[Dict]:
        """Fetch works from OpenAlex API by source ID."""
        source_id = source_config["openalex_source_id"]
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        url = f"{self.OPENALEX_BASE}/works"
        params = {
            "filter": f"primary_location.source.id:{source_id},from_publication_date:{from_date}",
            "per_page": 200,
            "sort": "publication_date:desc",
            "mailto": bot_contact(),
        }

        items = []
        try:
            response = resilient_get(
                self.session,
                url,
                params=params,
                timeout=self.timeout,
                source_label=f"OpenAlex:{source_config['key']}",
            )
            data = response.json()

            for work in data.get("results", []):
                item = self._openalex_work_to_item(work, source_config)
                if item:
                    items.append(item)

            time.sleep(0.5)

        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAlex error for {source_config['key']}: {e}")
        except (ValueError, KeyError) as e:
            logger.error(f"OpenAlex parse error for {source_config['key']}: {e}")

        return items

    def _fetch_via_rss(self, source_config: dict, days_back: int) -> List[Dict]:
        """Fetch items from an RSS/Atom feed."""
        feed_url = source_config["feed_url"]
        cutoff = datetime.now() - timedelta(days=days_back)

        items = []
        try:
            response = resilient_get(
                self.session,
                feed_url,
                timeout=self.timeout,
                source_label=f"RSS:{source_config['key']}",
            )

            root = ElementTree.fromstring(response.content)  # noqa: S314

            # Handle both RSS and Atom feeds
            rss_items = root.findall(".//item")
            if rss_items:
                for rss_item in rss_items:
                    item = self._rss_item_to_item(rss_item, source_config, cutoff)
                    if item:
                        items.append(item)
            else:
                # Try Atom format
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                entries = root.findall(".//atom:entry", ns) or root.findall(".//entry")
                for entry in entries:
                    item = self._atom_entry_to_item(entry, ns, source_config, cutoff)
                    if item:
                        items.append(item)

            time.sleep(0.5)

        except requests.exceptions.RequestException as e:
            logger.error(f"RSS error for {source_config['key']}: {e}")
        except ElementTree.ParseError as e:
            logger.error(f"RSS parse error for {source_config['key']}: {e}")

        return items

    def _enrich_abstracts_openalex(
        self, items: List[Dict], source_config: dict
    ) -> None:
        """Enrich items with abstracts from OpenAlex by DOI lookup."""
        for item in items[:10]:  # Limit enrichment requests
            doi = item.get("raw_metadata", {}).get("doi")
            if not doi:
                continue

            try:
                url = f"{self.OPENALEX_BASE}/works/doi:{doi}"
                response = self.session.get(
                    url,
                    params={"mailto": bot_contact()},
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    data = response.json()
                    abstract = self._reconstruct_abstract(
                        data.get("abstract_inverted_index")
                    )
                    if abstract:
                        item["abstract"] = abstract
                time.sleep(0.5)
            except (requests.exceptions.RequestException, ValueError, KeyError) as e:
                # Handle network errors, JSON decode errors, and missing keys
                logger.warning("Failed to enrich abstract for DOI %s: %s", doi, e)
                continue

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _crossref_work_to_item(self, work: dict, source_config: dict) -> Optional[Dict]:
        """Convert a CrossRef work object to a pipeline item dict."""
        title_parts = work.get("title", [])
        if not title_parts:
            return None
        title = title_parts[0]

        # Build URL: prefer DOI URL
        doi = work.get("DOI", "")
        url = f"https://doi.org/{doi}" if doi else ""
        if not url:
            links = work.get("link", [])
            url = links[0].get("URL", "") if links else ""
        if not url:
            return None

        # Authors
        authors = []
        for author in work.get("author", []):
            given = author.get("given", "")
            family = author.get("family", "")
            if given and family:
                authors.append(f"{given} {family}")
            elif family:
                authors.append(family)

        # Publication date
        date_parts = work.get("published", work.get("created", {})).get(
            "date-parts", [[]]
        )
        pub_date = self._date_parts_to_str(date_parts[0] if date_parts else [])

        # Abstract (CrossRef sometimes has it in HTML)
        abstract = work.get("abstract", "")
        if abstract:
            # Strip JATS XML tags
            abstract = re.sub(r"<[^>]+>", "", abstract).strip()

        return {
            "title": title,
            "abstract": abstract,
            "authors": ", ".join(authors),
            "url": url,
            "date": pub_date,
            "source": source_config.get("key", "academic"),
            "source_name": source_config.get("name", ""),
            "raw_metadata": {"doi": doi},
        }

    def _openalex_work_to_item(self, work: dict, source_config: dict) -> Optional[Dict]:
        """Convert an OpenAlex work object to a pipeline item dict."""
        title = work.get("title")
        if not title:
            return None

        doi = work.get("doi", "")
        url = doi if doi else work.get("id", "")
        if not url:
            return None

        # Authors
        authors = []
        for authorship in work.get("authorships", []):
            name = authorship.get("author", {}).get("display_name", "")
            if name:
                authors.append(name)

        pub_date = work.get("publication_date", "")

        # Clamp ahead-of-print future dates
        if pub_date:
            try:
                pd = datetime.strptime(pub_date[:10], "%Y-%m-%d")
                if pd > datetime.now() + timedelta(days=7):
                    logger.debug("Clamping future OpenAlex date %s to today", pub_date)
                    pub_date = datetime.now().strftime("%Y-%m-%d")
            except ValueError:
                pass

        # Reconstruct abstract from inverted index
        abstract = self._reconstruct_abstract(work.get("abstract_inverted_index"))

        return {
            "title": title,
            "abstract": abstract,
            "authors": ", ".join(authors),
            "url": url,
            "date": pub_date,
            "source": source_config.get("key", "academic"),
            "source_name": source_config.get("name", ""),
            "raw_metadata": {"doi": doi, "openalex_id": work.get("id", "")},
        }

    @staticmethod
    def _extract_rss_description(rss_item) -> str:
        """Extract and clean description text from an RSS item."""
        desc_elem = rss_item.find("description")
        if desc_elem is not None and desc_elem.text:
            return re.sub(r"<[^>]+>", " ", desc_elem.text).strip()
        return ""

    @staticmethod
    def _extract_rss_author(rss_item) -> str:
        """Extract author from RSS item, checking dc:creator as fallback."""
        author_elem = rss_item.find("author") or rss_item.find(
            "{http://purl.org/dc/elements/1.1/}creator"
        )
        return (author_elem.text or "").strip() if author_elem is not None else ""

    def _extract_rss_pub_date(
        self, rss_item, cutoff: datetime
    ) -> tuple[Optional[str], bool]:
        """Parse publication date from RSS item and check against cutoff.

        Returns:
            (pub_date, keep): pub_date as YYYY-MM-DD string or None,
            and keep=False if the item is before the cutoff.
        """
        pub_date_elem = rss_item.find("pubDate")
        pub_date = None
        if pub_date_elem is not None and pub_date_elem.text:
            pub_date = self._parse_rss_date(pub_date_elem.text)

        if pub_date and cutoff:
            try:
                if datetime.strptime(pub_date, "%Y-%m-%d") < cutoff:
                    return pub_date, False
            except ValueError:
                logger.debug("Failed to parse pub_date '%s' for RSS item", pub_date)

        return pub_date, True

    def _rss_item_to_item(
        self, rss_item, source_config: dict, cutoff: datetime
    ) -> Optional[Dict]:
        """Convert an RSS <item> to a pipeline item dict."""
        title_elem = rss_item.find("title")
        link_elem = rss_item.find("link")
        if title_elem is None or link_elem is None:
            return None

        title = (title_elem.text or "").strip()
        url = (link_elem.text or "").strip()
        if not title or not url:
            return None

        pub_date, keep = self._extract_rss_pub_date(rss_item, cutoff)
        if not keep:
            return None

        return {
            "title": title,
            "abstract": self._extract_rss_description(rss_item),
            "authors": self._extract_rss_author(rss_item),
            "url": url,
            "date": pub_date,
            "source": source_config.get("key", "academic"),
            "source_name": source_config.get("name", ""),
        }

    def _atom_entry_to_item(
        self, entry, ns: dict, source_config: dict, cutoff: datetime
    ) -> Optional[Dict]:
        """Convert an Atom <entry> to a pipeline item dict."""
        title_elem = entry.find("atom:title", ns) or entry.find("title")
        if title_elem is None:
            return None
        title = (title_elem.text or "").strip()
        if not title:
            return None

        # Link
        link_elem = entry.find("atom:link", ns) or entry.find("link")
        url = ""
        if link_elem is not None:
            url = link_elem.get("href", "") or (link_elem.text or "").strip()
        if not url:
            return None

        # Date
        pub_date = self._extract_atom_date(entry, ns)
        if self._is_before_cutoff(pub_date, cutoff):
            return None

        # Summary
        summary_elem = entry.find("atom:summary", ns) or entry.find("summary")
        abstract = ""
        if summary_elem is not None and summary_elem.text:
            abstract = self._strip_html_tags(summary_elem.text)

        return {
            "title": title,
            "abstract": abstract,
            "authors": self._extract_atom_authors(entry, ns),
            "url": url,
            "date": pub_date,
            "source": source_config.get("key", "academic"),
            "source_name": source_config.get("name", ""),
        }

    def _extract_atom_date(self, entry, ns: dict) -> Optional[str]:
        """Extract publication date from an Atom entry as YYYY-MM-DD."""
        updated_elem = (
            entry.find("atom:updated", ns)
            or entry.find("atom:published", ns)
            or entry.find("updated")
            or entry.find("published")
        )
        if updated_elem is not None and updated_elem.text:
            return updated_elem.text[:10]  # YYYY-MM-DD
        return None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_html_tags(text: str) -> str:
        """Remove HTML/XML tags from text."""
        return re.sub(r"<[^>]+>", " ", text).strip()

    @staticmethod
    def _is_before_cutoff(pub_date: Optional[str], cutoff: Optional[datetime]) -> bool:
        """Return True if pub_date is before the cutoff datetime."""
        if not pub_date or not cutoff:
            return False
        try:
            return datetime.strptime(pub_date, "%Y-%m-%d") < cutoff
        except ValueError:
            logger.debug("Failed to parse pub_date '%s' for cutoff check", pub_date)
            return False

    def _extract_atom_authors(self, entry, ns: dict) -> str:
        """Extract author names from an Atom entry as a comma-separated string."""
        authors = []
        for author_elem in entry.findall("atom:author", ns) or entry.findall("author"):
            name_elem = author_elem.find("atom:name", ns) or author_elem.find("name")
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())
        return ", ".join(authors)

    @staticmethod
    def _reconstruct_abstract(inverted_index: Optional[dict]) -> str:
        """Reconstruct abstract text from OpenAlex inverted index format."""
        if not inverted_index:
            return ""
        # Build (position, word) pairs and sort by position
        words: list[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                words.append((pos, word))
        words.sort(key=lambda x: x[0])
        return " ".join(w for _, w in words)

    @staticmethod
    def _date_parts_to_str(parts: list) -> str:
        """Convert CrossRef date-parts [year, month, day] to YYYY-MM-DD.

        Clamps future dates to today — CrossRef returns ahead-of-print dates
        that can be months or even years in the future.
        """
        if not parts:
            return datetime.now().strftime("%Y-%m-%d")
        year = parts[0]
        month = parts[1] if len(parts) > 1 else 1
        day = parts[2] if len(parts) > 2 else 1
        try:
            parsed = datetime(year, month, day)
            if parsed > datetime.now() + timedelta(days=7):
                logger.debug("Clamping future CrossRef date %s to today", parts)
                return datetime.now().strftime("%Y-%m-%d")
            return f"{year:04d}-{month:02d}-{day:02d}"
        except (TypeError, ValueError) as e:
            logger.debug("Failed to format date parts %s: %s", parts, e)
            return datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def _parse_rss_date(date_text: str) -> Optional[str]:
        """Parse RSS date to YYYY-MM-DD."""
        # Try RFC 2822
        for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S"):
            try:
                return datetime.strptime(date_text.strip(), fmt).strftime("%Y-%m-%d")
            except ValueError:
                logger.debug("RSS date '%s' doesn't match format '%s'", date_text, fmt)
                continue
        # Fallback: extract date-like portion
        m = re.search(r"(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})", date_text)
        if m:
            try:
                return datetime.strptime(
                    f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %b %Y"
                ).strftime("%Y-%m-%d")
            except ValueError:
                logger.debug("Failed to parse RSS date fallback '%s'", date_text)
                pass
        return None
