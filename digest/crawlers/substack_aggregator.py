"""
Substack feed aggregator.

Aggregates posts from Substack RSS feeds.
"""

import html.entities
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from defusedxml import ElementTree  # untrusted feed XML
from lxml import etree as lxml_etree  # recovery fallback only; see _recover_feed_xml

from digest.crawlers.resilient import cloudflare_get, resilient_get, rss_headers

logger = logging.getLogger(__name__)


class FeedFetchError(Exception):
    """A feed could not be fetched or parsed at all.

    Distinct from a feed that answered with nothing new. Both used to come
    back as an empty list, so the pipeline recorded a 404 or a parse failure
    as status "empty" and reset consecutive_failures to zero on every run: a
    source broken for weeks looked exactly like a quiet one, and dead-feed
    detection could never fire. fetch_items raises this so the pipeline's
    error path records it; fetch_feed keeps its list-returning contract for
    other callers unless asked otherwise.
    """


# The five entities XML itself defines. Anything else written as &name; is an
# HTML entity that a feed author pasted in without declaring.
_XML_ENTITIES = frozenset({"amp", "lt", "gt", "quot", "apos"})
_NAMED_ENTITY = re.compile(r"&([A-Za-z][A-Za-z0-9]*);")
_BARE_AMPERSAND = re.compile(r"&(?!(?:#\d+|#x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]*);)")
_XML_ILLEGAL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_XML_DECLARATION = re.compile(r"^\s*<\?xml[^>]*\?>")
_DECLARED_ENCODING = re.compile(rb"""^\s*<\?xml[^>]*encoding=["']([A-Za-z0-9._-]+)["']""")

# Feed roots this parser understands. A tree whose root is anything else --
# typically <html>, from a source configured with a web page instead of its
# feed -- is not a feed with zero items, it is a misconfiguration, and must
# fail loudly rather than parse into an empty list.
_FEED_ROOTS = frozenset({"rss", "RDF", "feed"})


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _repair_feed_xml(content: bytes) -> str:
    """Fix the cheap, common ways real feeds break XML.

    Undeclared HTML entities (&nbsp;, &mdash;) become numeric character
    references, bare ampersands are escaped, and control characters XML 1.0
    forbids are dropped. Returned as text with the XML declaration removed,
    since the declared encoding no longer describes it.
    """
    m = _DECLARED_ENCODING.match(content)
    encoding = m.group(1).decode("ascii") if m else "utf-8"
    try:
        text = content.decode(encoding, errors="replace")
    except LookupError:
        text = content.decode("utf-8", errors="replace")
    text = _XML_DECLARATION.sub("", text, count=1)

    def _entity(match: "re.Match") -> str:
        name = match.group(1)
        if name in _XML_ENTITIES:
            return match.group(0)
        codepoint = html.entities.name2codepoint.get(name)
        return f"&#{codepoint};" if codepoint else f"&amp;{name};"

    text = _NAMED_ENTITY.sub(_entity, text)
    text = _BARE_AMPERSAND.sub("&amp;", text)
    return _XML_ILLEGAL_CHARS.sub("", text)


def _recover_feed_xml(text: str) -> Optional[bytes]:
    """Last-resort structural recovery, returning well-formed XML or None.

    For feeds broken beyond entity fixes: raw HTML such as <br> inside an
    element, or an unrendered template tag in the output. lxml in recover mode
    rebuilds a tree around the damage. It is hardened the usual way -- no
    entity resolution, no network, no DTD, no huge trees -- and its output is
    only ever serialised back to bytes for defusedxml to parse again, so
    defusedxml stays the parser of record and every caller still receives
    stdlib Elements.

    Returns None unless the recovered root is a feed. lxml will happily build a
    tree out of an HTML page, and accepting that would turn a misconfigured
    source back into an apparently empty one.
    """
    parser = lxml_etree.XMLParser(
        recover=True,
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        huge_tree=False,
    )
    try:
        root = lxml_etree.fromstring(text.encode("utf-8"), parser=parser)
    except lxml_etree.XMLSyntaxError:
        return None
    if root is None or _local_name(root.tag) not in _FEED_ROOTS:
        return None
    return lxml_etree.tostring(root)


class SubstackAggregator:
    """Aggregator for Substack RSS feeds."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(rss_headers())

    def fetch_feed(
        self,
        feed_url: str,
        days_back: Optional[int] = 7,
        raise_on_error: bool = False,
    ) -> List[Dict]:
        """
        Fetch posts from a Substack RSS feed.

        Args:
            feed_url: URL of the RSS feed
            days_back: Only fetch posts from last N days (None for all)
            raise_on_error: Raise FeedFetchError when the feed cannot be
                fetched or parsed, instead of returning an empty list. The
                per-source path sets this so a broken feed is recorded as an
                error; see FeedFetchError.

        Returns:
            List of post dictionaries
        """

        def _fail(message: str, cause: Exception) -> List[Dict]:
            logger.error(message)
            if raise_on_error:
                raise FeedFetchError(message) from cause
            return []

        posts = []

        cutoff_date = None
        if days_back:
            cutoff_date = datetime.now() - timedelta(days=days_back)

        try:
            response = resilient_get(
                self.session,
                feed_url,
                timeout=self.timeout,
                source_label=f"Substack:{feed_url}",
            )

            posts = self._parse_feed_xml(response.content, feed_url, cutoff_date)

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                logger.info("Substack:%s: 403, trying curl_cffi bypass", feed_url)
                cf_resp = cloudflare_get(
                    feed_url, timeout=self.timeout, source_label=f"Substack:{feed_url}"
                )
                if cf_resp is not None and cf_resp.status_code == 200:
                    try:
                        posts = self._parse_feed_xml(
                            cf_resp.content, feed_url, cutoff_date
                        )
                    except ElementTree.ParseError as ex:
                        return _fail(
                            f"Error parsing XML from {feed_url} (curl_cffi): {ex}", ex
                        )
                else:
                    return _fail(
                        f"Error fetching feed {feed_url}: Cloudflare block (403)", e
                    )
            else:
                return _fail(f"Error fetching feed {feed_url}: {e}", e)
            return posts
        except requests.exceptions.RequestException as e:
            return _fail(f"Error fetching feed {feed_url}: {e}", e)
        except ElementTree.ParseError as e:
            return _fail(f"Error parsing XML from {feed_url}: {e}", e)

        return posts

    def _parse_feed_xml(
        self, content: bytes, feed_url: str, cutoff_date: Optional[datetime]
    ) -> List[Dict]:
        """Parse raw feed XML into a list of post dicts.

        Detects the feed format (RSS 2.0, RSS 1.0/RDF, or Atom), parses each
        item with the matching parser, and filters by cutoff_date.

        Raises:
            ElementTree.ParseError: If the XML cannot be parsed.
        """
        root = self._parse_tolerant(content, feed_url)

        # Find items: try standard RSS 2.0, then RSS 1.0/RDF namespace,
        # then Atom entries.
        items = root.findall(".//item")
        parse_fn = self._parse_feed_item
        if not items:
            # RSS 1.0 (RDF) feeds put items in {http://purl.org/rss/1.0/}
            items = root.findall(".//{http://purl.org/rss/1.0/}item")
            if items:
                parse_fn = self._parse_rdf_item
        if not items:
            # Atom feeds use <entry> elements
            atom_ns = "http://www.w3.org/2005/Atom"
            items = root.findall(f".//{{{atom_ns}}}entry")
            if not items:
                items = root.findall(".//entry")
            if items:
                parse_fn = self._parse_atom_entry

        posts = []
        for item in items:
            try:
                post = parse_fn(item, feed_url)
                if post:
                    # Filter by date if specified
                    if cutoff_date and post.get("date"):
                        post_date = datetime.strptime(post["date"], "%Y-%m-%d").date()
                        if post_date < cutoff_date.date():
                            continue

                    posts.append(post)
            except Exception as e:
                logger.warning(f"Error parsing feed item: {e}")
                continue

        return posts

    @staticmethod
    def _parse_tolerant(content: bytes, feed_url: str):
        """Parse feed XML, repairing the common breakages before giving up.

        Tries the bytes as-is, then with entities and ampersands repaired,
        then with structural recovery. A repaired or recovered parse logs a
        warning, so the publisher's broken output stays visible without being
        treated as a crawl failure. If nothing produces a feed, the original
        ParseError is raised -- including for a well-formed document whose
        root is not a feed, such as an HTML page configured as a feed URL.
        """
        try:
            root = ElementTree.fromstring(content.lstrip())
        except ElementTree.ParseError as original:
            repaired = _repair_feed_xml(content)
            try:
                root = ElementTree.fromstring(repaired.lstrip())
                how = "repaired entities in"
            except ElementTree.ParseError:
                recovered = _recover_feed_xml(repaired)
                if recovered is None:
                    raise original from None
                root = ElementTree.fromstring(recovered)
                how = "recovered structurally broken"
            logger.warning("%s: %s feed XML (%s)", feed_url, how, original)

        if _local_name(root.tag) not in _FEED_ROOTS:
            raise ElementTree.ParseError(
                f"not an RSS or Atom feed: document root is <{_local_name(root.tag)}>"
            )
        return root

    def _parse_feed_item(self, item, feed_url: str) -> Optional[Dict]:
        """Parse a single RSS feed item."""
        post = {}

        # Extract title
        title_elem = item.find("title")
        if title_elem is not None and title_elem.text:
            post["title"] = title_elem.text.strip()
        else:
            return None

        # Extract link
        link_elem = item.find("link")
        if link_elem is not None and link_elem.text:
            post["url"] = link_elem.text.strip()
        else:
            return None

        # Extract description/content
        description_elem = item.find("description")
        if description_elem is not None and description_elem.text:
            # Strip HTML tags from description
            post["content"] = self._strip_html(description_elem.text)
        else:
            post["content"] = ""

        # Try content:encoded as fallback
        if not post["content"]:
            content_elem = item.find(
                "{http://purl.org/rss/1.0/modules/content/}encoded"
            )
            if content_elem is not None and content_elem.text:
                post["content"] = self._strip_html(content_elem.text)

        # Extract author
        author_elem = item.find("author")
        if author_elem is not None and author_elem.text:
            post["author"] = author_elem.text.strip()
        else:
            # Try dc:creator as fallback
            creator_elem = item.find("{http://purl.org/dc/elements/1.1/}creator")
            if creator_elem is not None and creator_elem.text:
                post["author"] = creator_elem.text.strip()
            else:
                post["author"] = ""

        # Extract publication date
        pub_date_elem = item.find("pubDate")
        if pub_date_elem is not None and pub_date_elem.text:
            post["date"] = self._parse_rss_date(pub_date_elem.text)
        else:
            post["date"] = None

        # Extract Substack name from feed URL
        match = re.search(r"https?://([^.]+)\.substack\.com", feed_url)
        if match:
            post["substack_name"] = match.group(1)
        else:
            post["substack_name"] = ""

        # Extract images from the feed item
        post["images"] = self._extract_images(item, description_elem)

        post["source"] = "substack"

        return post

    def _parse_rdf_item(self, item, feed_url: str) -> Optional[Dict]:
        """Parse an RSS 1.0 (RDF) namespaced <item>."""
        ns = "http://purl.org/rss/1.0/"
        dc = "http://purl.org/dc/elements/1.1/"

        title_elem = item.find(f"{{{ns}}}title")
        link_elem = item.find(f"{{{ns}}}link")
        if title_elem is None or not (title_elem.text or "").strip():
            return None
        if link_elem is None or not (link_elem.text or "").strip():
            return None

        post: Dict = {
            "title": title_elem.text.strip(),
            "url": link_elem.text.strip(),
        }

        desc_elem = item.find(f"{{{ns}}}description")
        post["content"] = (
            self._strip_html(desc_elem.text)
            if desc_elem is not None and desc_elem.text
            else ""
        )

        creator_elem = item.find(f"{{{dc}}}creator")
        post["author"] = (
            creator_elem.text.strip()
            if creator_elem is not None and creator_elem.text
            else ""
        )

        date_elem = item.find(f"{{{dc}}}date")
        post["date"] = (
            date_elem.text[:10] if date_elem is not None and date_elem.text else None
        )

        post["substack_name"] = ""
        post["source"] = "rss"
        post["images"] = []
        return post

    # -- Atom parsing helpers --------------------------------------------------

    _ATOM_NS = "http://www.w3.org/2005/Atom"

    @staticmethod
    def _find_atom(parent, tag: str):
        """Find an Atom-namespaced element, falling back to plain tag."""
        elem = parent.find(f"{{{SubstackAggregator._ATOM_NS}}}{tag}")
        if elem is not None:
            return elem
        return parent.find(tag)

    def _extract_atom_content(self, entry) -> str:
        """Return plain-text content from an Atom entry's <content> or <summary>."""
        content_elem = self._find_atom(entry, "content")
        summary_elem = self._find_atom(entry, "summary")
        raw = ""
        if content_elem is not None and content_elem.text:
            raw = content_elem.text
        elif summary_elem is not None and summary_elem.text:
            raw = summary_elem.text
        return self._strip_html(raw) if raw else ""

    def _extract_atom_authors(self, entry) -> str:
        """Return comma-separated author names from an Atom entry."""
        author_elems = entry.findall(f"{{{self._ATOM_NS}}}author")
        if not author_elems:
            author_elems = entry.findall("author")
        names = []
        for a in author_elems:
            name_elem = self._find_atom(a, "name")
            if name_elem is not None and name_elem.text:
                names.append(name_elem.text.strip())
        return ", ".join(names)

    def _extract_atom_date(self, entry) -> Optional[str]:
        """Return ISO date string (YYYY-MM-DD) from an Atom entry, or None."""
        pub_elem = self._find_atom(entry, "published")
        if pub_elem is None:
            pub_elem = self._find_atom(entry, "updated")
        if pub_elem is not None and pub_elem.text:
            return pub_elem.text[:10]
        return None

    def _parse_atom_entry(self, entry, feed_url: str) -> Optional[Dict]:
        """Parse an Atom <entry> element."""
        title_elem = self._find_atom(entry, "title")
        if title_elem is None or not (title_elem.text or "").strip():
            return None

        link_elem = self._find_atom(entry, "link")
        url = ""
        if link_elem is not None:
            url = link_elem.get("href", "") or (link_elem.text or "").strip()
        if not url:
            return None

        return {
            "title": title_elem.text.strip(),
            "url": url,
            "content": self._extract_atom_content(entry),
            "author": self._extract_atom_authors(entry),
            "date": self._extract_atom_date(entry),
            "substack_name": "",
            "source": "rss",
            "images": [],
        }

    def _extract_images_from_html(self, html_text: str, source_label: str) -> List[str]:
        """Extract image URLs from HTML, filtering out small icons/tracking pixels."""
        images = []
        try:
            soup = BeautifulSoup(html_text, "html.parser")
            for img in soup.find_all("img"):
                src = img.get("src")
                if not src or not src.startswith("http"):
                    continue
                width = img.get("width")
                height = img.get("height")
                if width and height:
                    try:
                        if int(width) < 100 or int(height) < 100:
                            continue
                    except (ValueError, TypeError):
                        logger.debug(
                            "Failed to parse image dimensions in %s: width=%s, height=%s",
                            source_label,
                            width,
                            height,
                        )
                images.append(src)
        except (AttributeError, TypeError) as e:
            logger.warning("Failed to parse images from %s: %s", source_label, e)
        return images

    def _extract_images(self, item, description_elem) -> List[str]:
        """
        Extract image URLs from RSS item.

        Looks for images in enclosure tags, media:content, and HTML content.
        Returns max 2 images to avoid cluttering the digest.

        Args:
            item: RSS feed item element
            description_elem: Description element that may contain HTML

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

        # Check for media:content tags
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

        # Parse HTML content for img tags (common in Substack)
        if description_elem is not None and description_elem.text:
            images.extend(
                self._extract_images_from_html(description_elem.text, "description")
            )

        # Also check content:encoded for images
        content_elem = item.find("{http://purl.org/rss/1.0/modules/content/}encoded")
        if content_elem is not None and content_elem.text:
            images.extend(
                self._extract_images_from_html(content_elem.text, "content:encoded")
            )

        # Remove duplicates while preserving order
        seen = set()
        unique_images = []
        for img in images:
            if img not in seen:
                seen.add(img)
                unique_images.append(img)

        # Return max 2 images
        return unique_images[:2]

    def _strip_html(self, html: str) -> str:
        """Strip HTML tags from text."""
        # Remove HTML tags
        clean = re.sub(r"<[^>]+>", " ", html)
        # Remove extra whitespace
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()

    def _parse_rss_date(self, date_text: str) -> Optional[str]:
        """Parse RSS date to ISO format."""
        # RSS dates are typically in RFC 2822 format
        # Example: "Wed, 15 Jan 2026 10:00:00 GMT"
        try:
            # Try parsing RFC 2822 format
            date_obj = datetime.strptime(date_text, "%a, %d %b %Y %H:%M:%S %Z")
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            logger.debug(
                "RSS date '%s' doesn't match RFC 2822 with timezone format", date_text
            )
            pass

        try:
            # Try without timezone
            date_obj = datetime.strptime(date_text, "%a, %d %b %Y %H:%M:%S")
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            logger.debug(
                "RSS date '%s' doesn't match RFC 2822 without timezone format",
                date_text,
            )
            pass

        # If parsing fails, try to extract just the date part
        match = re.search(r"(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})", date_text)
        if match:
            try:
                day, month, year = match.groups()
                date_obj = datetime.strptime(f"{day} {month} {year}", "%d %b %Y")
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                logger.debug(
                    "Failed to parse extracted date parts from '%s'", date_text
                )
                pass

        return None

    def fetch_multiple_feeds(
        self, feed_urls: List[str], days_back: Optional[int] = 7
    ) -> List[Dict]:
        """
        Fetch posts from multiple Substack feeds.

        Args:
            feed_urls: List of RSS feed URLs
            days_back: Only fetch posts from last N days

        Returns:
            Combined list of posts
        """
        all_posts = []

        for feed_url in feed_urls:
            posts = self.fetch_feed(feed_url, days_back=days_back)
            all_posts.extend(posts)

        # Sort by date (newest first)
        all_posts.sort(key=lambda x: x.get("date", ""), reverse=True)

        return all_posts

    def fetch_items(self, source_config: dict, days_back: int = 7) -> List[Dict]:
        """
        Adapter method for source-driven dispatch.

        Fetches from a single source's feed_url and stamps each item with the
        correct source key and name (overriding the generic "substack" tag).

        Args:
            source_config: Source row dict from the sources table.
                Must have 'feed_url'.
            days_back: Number of days to look back.

        Returns:
            List of post dicts compatible with store_item().
        """
        feed_url = source_config.get("feed_url")
        if not feed_url:
            logger.warning(f"No feed_url for source {source_config.get('key')}")
            return []

        # Raise rather than return [] on failure: this is the per-source path,
        # and the pipeline records an exception as status "error" with a
        # climbing consecutive_failures count. An empty list would be recorded
        # as "empty" and reset that count -- see FeedFetchError.
        items = self.fetch_feed(feed_url, days_back=days_back, raise_on_error=True)

        # Override source fields so items.source matches sources.key
        source_key = source_config.get("key", "substack")
        source_name = source_config.get("name", "")
        for item in items:
            item["source"] = source_key
            if source_name:
                item["source_name"] = source_name

        return items


def main():
    """Test the Substack aggregator."""
    aggregator = SubstackAggregator()

    # Example Substack feeds (you can add actual feeds here)
    test_feeds = [
        # "https://example.substack.com/feed",
    ]

    if not test_feeds:
        print("No test feeds configured. Add Substack RSS feed URLs to test.")
        return

    posts = aggregator.fetch_multiple_feeds(test_feeds, days_back=7)

    print(f"Found {len(posts)} posts")
    for post in posts[:3]:
        print(f"\nTitle: {post['title']}")
        print(f"Author: {post.get('author', 'N/A')}")
        print(f"Substack: {post.get('substack_name', 'N/A')}")
        print(f"Date: {post.get('date', 'N/A')}")
        print(f"URL: {post['url']}")
        print(f"Content: {post['content'][:200]}...")


if __name__ == "__main__":
    main()
