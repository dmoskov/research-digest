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
