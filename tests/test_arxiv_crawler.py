"""Tests for ArxivCrawler fetch_items and parsing logic."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Sample arXiv Atom feed with two entries
SAMPLE_ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>ArXiv Query: cat:cs.AI</title>
  <entry>
    <id>http://arxiv.org/abs/2501.00001v1</id>
    <title>  A Novel Approach to
      AI Safety  </title>
    <summary>  This paper proposes a new framework
      for ensuring AI alignment.  </summary>
    <published>2099-01-10T00:00:00Z</published>
    <link href="http://arxiv.org/abs/2501.00001v1" rel="alternate" type="text/html"/>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <category term="cs.AI"/>
    <category term="cs.LG"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2501.00002v1</id>
    <title>Deep Learning for Biosecurity</title>
    <summary>We apply deep learning to detect biosecurity threats.</summary>
    <published>2099-01-11T00:00:00Z</published>
    <link href="http://arxiv.org/abs/2501.00002v1" rel="alternate" type="text/html"/>
    <author><name>Carol White</name></author>
    <category term="cs.AI"/>
  </entry>
</feed>"""


# Entry with no title
SAMPLE_NO_TITLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2501.00099v1</id>
    <summary>No title here.</summary>
    <published>2099-01-10T00:00:00Z</published>
  </entry>
</feed>"""


# Entry with no link but has id
SAMPLE_NO_LINK = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2501.00050v1</id>
    <title>Fallback URL Paper</title>
    <summary>Uses id as url fallback.</summary>
    <published>2099-01-10T00:00:00Z</published>
    <author><name>Test Author</name></author>
  </entry>
</feed>"""


# Old entry (before cutoff)
SAMPLE_OLD_ENTRY = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2001.00001v1</id>
    <title>Very Old Paper</title>
    <summary>Published years ago.</summary>
    <published>2001-01-01T00:00:00Z</published>
    <link href="http://arxiv.org/abs/2001.00001v1" rel="alternate" type="text/html"/>
    <author><name>Old Author</name></author>
  </entry>
</feed>"""


EMPTY_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query: cat:cs.AI</title>
</feed>"""


SOURCE_CONFIG = {
    "key": "arxiv_ai",
    "name": "arXiv AI Safety",
    "crawl_config": {"categories": ["cs.AI"]},
}


def _mock_response(content_bytes):
    """Helper to build a mock response object."""
    resp = MagicMock()
    resp.content = content_bytes
    resp.status_code = 200
    return resp


class TestArxivCrawlerFetchItems:
    """Tests for ArxivCrawler.fetch_items()."""

    @patch("digest.crawlers.arxiv_crawler.resilient_get")
    def test_fetch_items_parses_entries(self, mock_get):
        from digest.crawlers.arxiv_crawler import ArxivCrawler

        mock_get.return_value = _mock_response(SAMPLE_ARXIV_ATOM.encode())

        crawler = ArxivCrawler()
        items = crawler.fetch_items(SOURCE_CONFIG, days_back=365000)

        assert len(items) == 2
        assert items[0]["title"] == "A Novel Approach to AI Safety"
        assert items[0]["url"] == "http://arxiv.org/abs/2501.00001v1"
        assert items[0]["authors"] == "Alice Smith, Bob Jones"
        assert items[0]["date"] == "2099-01-10"
        assert items[0]["source"] == "arxiv_ai"
        assert items[0]["source_name"] == "arXiv AI Safety"
        assert items[0]["source_id"] == "2501.00001v1"
        assert items[0]["raw_metadata"]["categories"] == ["cs.AI", "cs.LG"]

    @patch("digest.crawlers.arxiv_crawler.resilient_get")
    def test_fetch_items_second_entry(self, mock_get):
        from digest.crawlers.arxiv_crawler import ArxivCrawler

        mock_get.return_value = _mock_response(SAMPLE_ARXIV_ATOM.encode())

        crawler = ArxivCrawler()
        items = crawler.fetch_items(SOURCE_CONFIG, days_back=365000)

        assert items[1]["title"] == "Deep Learning for Biosecurity"
        assert items[1]["authors"] == "Carol White"
        assert items[1]["raw_metadata"]["categories"] == ["cs.AI"]

    @patch("digest.crawlers.arxiv_crawler.resilient_get")
    def test_fetch_items_deduplicates_by_url(self, mock_get):
        """When multiple categories return the same paper, it appears only once."""
        from digest.crawlers.arxiv_crawler import ArxivCrawler

        mock_get.return_value = _mock_response(SAMPLE_ARXIV_ATOM.encode())

        config = {
            "key": "arxiv_ai",
            "name": "arXiv AI",
            "crawl_config": {"categories": ["cs.AI", "cs.LG"]},
        }
        crawler = ArxivCrawler()
        with patch("digest.crawlers.arxiv_crawler.time.sleep"):
            items = crawler.fetch_items(config, days_back=365000)

        # Same feed returned for both categories → duplicates removed
        assert len(items) == 2

    @patch("digest.crawlers.arxiv_crawler.resilient_get")
    def test_fetch_items_multiple_categories_sleeps(self, mock_get):
        """Respects arXiv ToS: sleeps between category requests."""
        from digest.crawlers.arxiv_crawler import ArxivCrawler

        mock_get.return_value = _mock_response(EMPTY_FEED.encode())

        config = {
            "key": "arxiv_multi",
            "name": "arXiv Multi",
            "crawl_config": {"categories": ["cs.AI", "cs.LG"]},
        }
        crawler = ArxivCrawler()
        with patch("digest.crawlers.arxiv_crawler.time.sleep") as mock_sleep:
            crawler.fetch_items(config, days_back=7)

        # Sleep is called after each category when len(categories) > 1
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(3)

    def test_fetch_items_no_categories_returns_empty(self):
        from digest.crawlers.arxiv_crawler import ArxivCrawler

        config = {"key": "empty", "name": "Empty", "crawl_config": {"categories": []}}
        crawler = ArxivCrawler()
        items = crawler.fetch_items(config, days_back=7)
        assert items == []

    def test_fetch_items_missing_crawl_config_returns_empty(self):
        from digest.crawlers.arxiv_crawler import ArxivCrawler

        config = {"key": "no_config", "name": "No Config"}
        crawler = ArxivCrawler()
        items = crawler.fetch_items(config, days_back=7)
        assert items == []

    @patch("digest.crawlers.arxiv_crawler.resilient_get")
    def test_fetch_items_http_error_returns_empty(self, mock_get):
        from digest.crawlers.arxiv_crawler import ArxivCrawler

        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        crawler = ArxivCrawler()
        items = crawler.fetch_items(SOURCE_CONFIG, days_back=7)
        assert items == []

    @patch("digest.crawlers.arxiv_crawler.resilient_get")
    def test_fetch_items_malformed_xml_returns_empty(self, mock_get):
        from digest.crawlers.arxiv_crawler import ArxivCrawler

        mock_get.return_value = _mock_response(b"<not valid xml")

        crawler = ArxivCrawler()
        items = crawler.fetch_items(SOURCE_CONFIG, days_back=7)
        assert items == []

    @patch("digest.crawlers.arxiv_crawler.resilient_get")
    def test_fetch_items_empty_feed(self, mock_get):
        from digest.crawlers.arxiv_crawler import ArxivCrawler

        mock_get.return_value = _mock_response(EMPTY_FEED.encode())

        crawler = ArxivCrawler()
        items = crawler.fetch_items(SOURCE_CONFIG, days_back=7)
        assert items == []

    @patch("digest.crawlers.arxiv_crawler.resilient_get")
    def test_fetch_items_filters_old_entries(self, mock_get):
        from digest.crawlers.arxiv_crawler import ArxivCrawler

        mock_get.return_value = _mock_response(SAMPLE_OLD_ENTRY.encode())

        crawler = ArxivCrawler()
        items = crawler.fetch_items(SOURCE_CONFIG, days_back=7)
        assert items == []

    @patch("digest.crawlers.arxiv_crawler.resilient_get")
    def test_entry_without_title_is_skipped(self, mock_get):
        from digest.crawlers.arxiv_crawler import ArxivCrawler

        mock_get.return_value = _mock_response(SAMPLE_NO_TITLE.encode())

        crawler = ArxivCrawler()
        items = crawler.fetch_items(SOURCE_CONFIG, days_back=365000)
        assert items == []

    @patch("digest.crawlers.arxiv_crawler.resilient_get")
    def test_entry_falls_back_to_id_for_url(self, mock_get):
        from digest.crawlers.arxiv_crawler import ArxivCrawler

        mock_get.return_value = _mock_response(SAMPLE_NO_LINK.encode())

        crawler = ArxivCrawler()
        items = crawler.fetch_items(SOURCE_CONFIG, days_back=365000)
        assert len(items) == 1
        assert items[0]["url"] == "http://arxiv.org/abs/2501.00050v1"
        assert items[0]["source_id"] == "2501.00050v1"

    @patch("digest.crawlers.arxiv_crawler.resilient_get")
    def test_abstract_whitespace_normalized(self, mock_get):
        from digest.crawlers.arxiv_crawler import ArxivCrawler

        mock_get.return_value = _mock_response(SAMPLE_ARXIV_ATOM.encode())

        crawler = ArxivCrawler()
        items = crawler.fetch_items(SOURCE_CONFIG, days_back=365000)
        # Multiline abstract should be joined into single line
        assert "\n" not in items[0]["abstract"]
        assert "  " not in items[0]["abstract"]
