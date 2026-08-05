"""
OSF preprint crawler for SocArXiv and other OSF-hosted preprint servers.

Fetches recent preprints from OSF preprint providers (SocArXiv, MetaArXiv,
EarthArXiv, etc.) via the OSF v2 JSON API.

API docs: https://developer.osf.io/
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import requests

from digest.crawlers.resilient import api_headers, resilient_get

logger = logging.getLogger(__name__)


def _flatten_subjects(subjects_raw: list) -> list:
    """Flatten OSF's nested subject arrays into a flat list of text labels.

    OSF subjects come as [[{"id": ..., "text": "Social Sciences"}, ...], ...]
    """
    out = []
    for group in subjects_raw:
        if isinstance(group, list):
            for s in group:
                if isinstance(s, dict) and s.get("text"):
                    out.append(s["text"])
        elif isinstance(group, dict) and group.get("text"):
            out.append(group["text"])
    return out


class OsfPreprintCrawler:
    """Crawler for OSF-hosted preprint servers (SocArXiv, etc.)."""

    API_BASE = "https://api.osf.io/v2/preprints/"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(api_headers())

    def fetch_items(self, source_config: dict, days_back: int = 7) -> List[Dict]:
        """
        Fetch recent preprints from an OSF preprint provider.

        Args:
            source_config: Source row dict from the sources table. Expected keys:
                - crawl_config: {"provider": "socarxiv", "subjects": [...]}
                  provider: OSF provider slug (e.g. "socarxiv", "metaarxiv")
                  subjects: optional list of OSF subject strings to filter on
                - key: source key for tagging items
                - name: source display name
            days_back: Number of days to look back.

        Returns:
            List of item dicts compatible with store_item().
        """
        crawl_config = source_config.get("crawl_config") or {}
        provider = crawl_config.get("provider", "socarxiv")
        subjects = crawl_config.get("subjects", [])

        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%S")

        items: List[Dict] = []
        url = self.API_BASE
        params = {
            "filter[provider]": provider,
            "filter[date_created][gte]": cutoff_str,
            "sort": "-date_created",
            "page[size]": 50,
        }

        # Add subject filter if specified
        if subjects:
            params["filter[subjects]"] = ",".join(subjects)

        pages_fetched = 0
        max_pages = 5  # Safety limit

        while url and pages_fetched < max_pages:
            try:
                response = resilient_get(
                    self.session,
                    url,
                    params=params if pages_fetched == 0 else None,
                    timeout=self.timeout,
                    source_label=f"OSF:{provider}",
                )
                data = response.json()

                for entry in data.get("data", []):
                    item = self._parse_entry(entry, source_config, cutoff)
                    if item:
                        items.append(item)

                # Follow pagination
                url = data.get("links", {}).get("next")
                pages_fetched += 1

            except requests.exceptions.RequestException as e:
                logger.error("OSF API error for %s: %s", provider, e)
                break
            except (ValueError, KeyError) as e:
                logger.error("OSF JSON parse error for %s: %s", provider, e)
                break

        # Deduplicate by URL
        seen_urls: set = set()
        unique_items: List[Dict] = []
        for item in items:
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                unique_items.append(item)

        logger.info(
            "OSF:%s — fetched %d preprints (%d pages)",
            provider, len(unique_items), pages_fetched,
        )
        return unique_items

    def _parse_entry(
        self, entry: dict, source_config: dict, cutoff: datetime
    ) -> Optional[Dict]:
        """Parse an OSF preprint JSON entry to a pipeline item dict."""
        attrs = entry.get("attributes", {})

        title = (attrs.get("title") or "").strip()
        if not title:
            return None

        # Parse date
        date_str = attrs.get("date_created") or attrs.get("date_published")
        pub_date = None
        if date_str:
            try:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                # Ensure timezone-aware for comparison with cutoff
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt < cutoff:
                    return None
                pub_date = dt.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                pass

        # Extract URL — prefer DOI link, fall back to OSF preprint URL
        doi = attrs.get("doi")
        links = entry.get("links", {})
        preprint_url = links.get("preprint_doi") or links.get("html") or ""

        if doi:
            url = f"https://doi.org/{doi}"
        elif preprint_url:
            url = preprint_url
        else:
            osf_id = entry.get("id", "")
            url = f"https://osf.io/{osf_id}/" if osf_id else ""

        if not url:
            return None

        abstract = (attrs.get("description") or "").strip()
        # Clean up HTML tags that sometimes appear in OSF abstracts
        if "<" in abstract:
            import re
            abstract = re.sub(r"<[^>]+>", "", abstract).strip()

        # Extract authors from contributors relationship
        authors = self._extract_authors(entry)

        provider = (source_config.get("crawl_config") or {}).get("provider", "socarxiv")
        osf_id = entry.get("id", "")

        return {
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "url": url,
            "date": pub_date,
            "source": source_config.get("key", f"osf-{provider}"),
            "source_name": source_config.get("name", provider.title()),
            "source_id": osf_id,
            "raw_metadata": {
                "osf_id": osf_id,
                "provider": provider,
                "doi": doi,
                "subjects": _flatten_subjects(attrs.get("subjects", [])),
            },
        }

    @staticmethod
    def _extract_authors(entry: dict) -> str:
        """Extract author names from the contributors relationship.

        The OSF preprints list endpoint embeds contributor data if available,
        otherwise we fall back to an empty string. A separate API call to
        the contributors endpoint could be made but would be too slow for
        bulk crawling.
        """
        # Try embedded contributors (sometimes available)
        contributors = entry.get("embeds", {}).get("contributors", {}).get("data", [])
        if contributors:
            names = []
            for c in contributors:
                user = c.get("embeds", {}).get("users", {}).get("data", {})
                full_name = user.get("attributes", {}).get("full_name", "")
                if full_name:
                    names.append(full_name)
            if names:
                return ", ".join(names)

        return ""
