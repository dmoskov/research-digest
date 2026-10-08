"""Tests for RSS/Substack crawler parsing and error handling."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>Test Blog</title>
    <item>
      <title>Test Post Title</title>
      <link>https://example.com/post-1</link>
      <description><![CDATA[<p>A test post about housing policy.</p>]]></description>
      <pubDate>Mon, 06 Jan 2025 12:00:00 GMT</pubDate>
      <dc:creator>Test Author</dc:creator>
    </item>
    <item>
      <title>Another Post</title>
      <link>https://example.com/post-2</link>
      <description><![CDATA[<p>Another test post about energy.</p>]]></description>
      <pubDate>Tue, 07 Jan 2025 12:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>"""


class TestSubstackAggregator:
    """Tests for SubstackAggregator RSS parsing."""

    @patch("digest.crawlers.substack_aggregator.resilient_get")
    def test_parse_rss_items(self, mock_get):
        from digest.crawlers.substack_aggregator import SubstackAggregator

        mock_response = MagicMock()
        mock_response.content = SAMPLE_RSS.encode("utf-8")
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        aggregator = SubstackAggregator()
        items = aggregator.fetch_feed("https://example.com/feed", days_back=None)

        assert len(items) == 2
        assert items[0]["title"] == "Test Post Title"
        assert items[0]["url"] == "https://example.com/post-1"

    @patch("digest.crawlers.substack_aggregator.resilient_get")
    def test_handles_http_error(self, mock_get):
        import requests

        from digest.crawlers.substack_aggregator import SubstackAggregator

        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        aggregator = SubstackAggregator()
        items = aggregator.fetch_feed("https://bad-feed.com/feed", days_back=7)

        assert isinstance(items, list)
        assert items == []

    @patch("digest.crawlers.substack_aggregator.resilient_get")
    def test_handles_malformed_xml(self, mock_get):
        from digest.crawlers.substack_aggregator import SubstackAggregator

        mock_response = MagicMock()
        mock_response.content = b"<not valid xml"
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        aggregator = SubstackAggregator()
        items = aggregator.fetch_feed("https://broken.com/feed", days_back=7)

        assert isinstance(items, list)
        assert items == []

    @patch("digest.crawlers.substack_aggregator.resilient_get")
    def test_empty_feed_returns_empty_list(self, mock_get):
        from digest.crawlers.substack_aggregator import SubstackAggregator

        empty_rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0"><channel><title>Empty</title></channel></rss>"""
        mock_response = MagicMock()
        mock_response.content = empty_rss.encode("utf-8")
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        aggregator = SubstackAggregator()
        items = aggregator.fetch_feed("https://empty.com/feed", days_back=7)

        assert items == []


# ---------------------------------------------------------------------------
# Broken-but-real feeds, and telling a broken feed from a quiet one
# ---------------------------------------------------------------------------

def _resp(body: bytes):
    r = MagicMock()
    r.content = body
    r.status_code = 200
    return r


_RSS_WITH_HTML_ENTITIES = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>T</title>
<item><title>Humane slaughter&nbsp;for shrimp &mdash; a primer</title>
<link>https://example.com/a</link>
<pubDate>Mon, 06 Jan 2025 12:00:00 GMT</pubDate></item>
</channel></rss>"""

# Raw HTML in content, as Bruegel publishes it.
_RSS_WITH_UNCLOSED_HTML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>T</title>
<item><title>Trade policy note</title><link>https://example.com/b</link>
<description>First line<br>second line</description>
<pubDate>Mon, 06 Jan 2025 12:00:00 GMT</pubDate></item>
</channel></rss>"""

# An unrendered PHP tag in the output, as On Think Tanks publishes it.
_RSS_WITH_PHP_TAG = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>T</title>
<item><title>Think tank events</title><link>https://example.com/c</link>
<guid isPermaLink="false">https://example.com/?p=<? echo get_the_ID(); ?></guid>
<pubDate>Mon, 06 Jan 2025 12:00:00 GMT</pubDate></item>
</channel></rss>"""

_HTML_PAGE = b"""<!DOCTYPE html><html><head><title>Housing policy initiative</title></head>
<body><p>Our work<br>on housing</p></body></html>"""


class TestBrokenFeedRecovery:
    @patch("digest.crawlers.substack_aggregator.resilient_get")
    def test_undeclared_html_entities_are_repaired(self, mock_get):
        from digest.crawlers.substack_aggregator import SubstackAggregator

        mock_get.return_value = _resp(_RSS_WITH_HTML_ENTITIES)
        items = SubstackAggregator().fetch_feed("https://x/feed", days_back=None)
        assert len(items) == 1
        assert " " in items[0]["title"]  # &nbsp;
        assert "—" in items[0]["title"]  # &mdash;

    @patch("digest.crawlers.substack_aggregator.resilient_get")
    def test_raw_html_inside_an_element_is_recovered(self, mock_get):
        from digest.crawlers.substack_aggregator import SubstackAggregator

        mock_get.return_value = _resp(_RSS_WITH_UNCLOSED_HTML)
        items = SubstackAggregator().fetch_feed("https://x/feed", days_back=None)
        assert [i["title"] for i in items] == ["Trade policy note"]

    @patch("digest.crawlers.substack_aggregator.resilient_get")
    def test_unrendered_template_tag_is_recovered(self, mock_get):
        from digest.crawlers.substack_aggregator import SubstackAggregator

        mock_get.return_value = _resp(_RSS_WITH_PHP_TAG)
        items = SubstackAggregator().fetch_feed("https://x/feed", days_back=None)
        assert [i["url"] for i in items] == ["https://example.com/c"]

    @patch("digest.crawlers.substack_aggregator.resilient_get")
    def test_html_page_is_not_mistaken_for_an_empty_feed(self, mock_get):
        """A web page configured as a feed URL must fail, not parse to [].

        Recovery would happily build a tree out of HTML; accepting it would
        make a misconfigured source look like a quiet one again.
        """
        import pytest

        from digest.crawlers.substack_aggregator import (
            FeedFetchError,
            SubstackAggregator,
        )

        mock_get.return_value = _resp(_HTML_PAGE)
        with pytest.raises(FeedFetchError, match="not an RSS or Atom feed|not well-formed|mismatched"):
            SubstackAggregator().fetch_feed(
                "https://x/page", days_back=None, raise_on_error=True
            )

    @patch("digest.crawlers.substack_aggregator.resilient_get")
    def test_well_formed_non_feed_root_is_rejected(self, mock_get):
        import pytest

        from digest.crawlers.substack_aggregator import (
            FeedFetchError,
            SubstackAggregator,
        )

        mock_get.return_value = _resp(b"<html><body><p>hi</p></body></html>")
        with pytest.raises(FeedFetchError, match="document root is <html>"):
            SubstackAggregator().fetch_feed(
                "https://x/page", days_back=None, raise_on_error=True
            )

    def test_external_entities_stay_inert(self):
        """The recovery path must not open the XXE hole defusedxml closes."""
        from digest.crawlers.substack_aggregator import (
            _recover_feed_xml,
            _repair_feed_xml,
        )

        evil = (
            b'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY x SYSTEM '
            b'"file:///etc/passwd">]><rss><channel><item><title>&x;</title>'
            b"</item></channel></rss>"
        )
        recovered = _recover_feed_xml(_repair_feed_xml(evil))
        assert recovered is None or b"root:" not in recovered


class TestBrokenFeedIsAnError:
    """fetch_items raises, so the pipeline records status 'error'."""

    @patch("digest.crawlers.substack_aggregator.resilient_get")
    def test_connection_failure_raises_from_fetch_items(self, mock_get):
        import pytest
        import requests

        from digest.crawlers.substack_aggregator import (
            FeedFetchError,
            SubstackAggregator,
        )

        mock_get.side_effect = requests.exceptions.ConnectionError("refused")
        with pytest.raises(FeedFetchError):
            SubstackAggregator().fetch_items(
                {"key": "s", "name": "S", "feed_url": "https://x/feed"}
            )

    @patch("digest.crawlers.substack_aggregator.resilient_get")
    def test_404_raises_from_fetch_items(self, mock_get):
        import pytest
        import requests

        from digest.crawlers.substack_aggregator import (
            FeedFetchError,
            SubstackAggregator,
        )

        resp = MagicMock(status_code=404)
        mock_get.side_effect = requests.exceptions.HTTPError("404", response=resp)
        with pytest.raises(FeedFetchError, match="404"):
            SubstackAggregator().fetch_items(
                {"key": "s", "name": "S", "feed_url": "https://x/feed"}
            )

    @patch("digest.crawlers.substack_aggregator.resilient_get")
    def test_quiet_feed_still_returns_empty_without_raising(self, mock_get):
        """A healthy feed with nothing in the window is 'empty', not an error."""
        from digest.crawlers.substack_aggregator import SubstackAggregator

        mock_get.return_value = _resp(
            b'<?xml version="1.0"?><rss version="2.0"><channel><title>E</title>'
            b"</channel></rss>"
        )
        assert SubstackAggregator().fetch_items(
            {"key": "s", "name": "S", "feed_url": "https://x/feed"}
        ) == []

    @patch("digest.crawlers.substack_aggregator.resilient_get")
    def test_fetch_feed_default_still_returns_list(self, mock_get):
        """Other callers keep the list-returning contract."""
        import requests

        from digest.crawlers.substack_aggregator import SubstackAggregator

        mock_get.side_effect = requests.exceptions.ConnectionError("refused")
        assert SubstackAggregator().fetch_feed("https://x/feed") == []
