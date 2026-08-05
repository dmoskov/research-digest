"""Tests for HTML scraper crawler parsing and error handling."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from digest.crawlers.html_scraper import (
    HtmlScraperCrawler,
    _parse_flexible_date,
)

# --- Sample HTML fixtures ---

SAMPLE_ANTHROPIC_HTML = """
<html>
<body>
<ul class="PublicationList-module-scss-module__KxYrHG__list">
  <li>
    <a class="PublicationList-module-scss-module__KxYrHG__listItem" href="/research/labor-market-impacts">
      <div class="PublicationList-module-scss-module__KxYrHG__meta">
        <time class="PublicationList-module-scss-module__KxYrHG__date body-3">Mar 5, 2026</time>
        <span class="PublicationList-module-scss-module__KxYrHG__subject body-3">Economic Research</span>
      </div>
      <span class="PublicationList-module-scss-module__KxYrHG__title body-3">Labor market impacts of AI</span>
    </a>
  </li>
  <li>
    <a class="PublicationList-module-scss-module__KxYrHG__listItem" href="/research/alignment-faking">
      <div class="PublicationList-module-scss-module__KxYrHG__meta">
        <time class="PublicationList-module-scss-module__KxYrHG__date body-3">Dec 18, 2024</time>
        <span class="PublicationList-module-scss-module__KxYrHG__subject body-3">Alignment</span>
      </div>
      <span class="PublicationList-module-scss-module__KxYrHG__title body-3">Alignment faking in large language models</span>
    </a>
  </li>
</ul>
</body>
</html>
"""  # noqa: E501

SAMPLE_EPOCH_HTML = """
<html>
<body>
<div class="latest-posts">
  <a class="blog-post-card filtereable filtered-in paged-in" href="/blog/expanding-analysis">
    <div class="post-body">
      <div class="post-title">Expanding our analysis of biological AI models</div>
      <div class="post-description">We release a database of over 1,100 biological AI models.</div>
      <div class="post-date-and-authors">Feb 20, 2026 · By David Atanasov</div>
    </div>
  </a>
  <a class="blog-post-card filtereable filtered-in paged-in" href="/blog/economic-value-benchmarks">
    <div class="post-body">
      <div class="post-title">What do economic value benchmarks tell us?</div>
      <div class="post-description">These benchmarks track digital work.</div>
      <div class="post-date-and-authors">Feb 13, 2026 · By Florian Brand</div>
    </div>
  </a>
</div>
</body>
</html>
"""

SAMPLE_GATES_HTML = """
<html>
<body>
<section class="article-promo component">
  <div class="article-promo__content">
    <div class="article-promo__content-head">
      <h2 class="article-promo__title">
        <a href="/ideas/articles/core-mission-child-health/">
          <span>Helping every child reach their full potential</span>
        </a>
      </h2>
    </div>
    <div class="article-promo__content-body">
      <div class="article-promo__description">Every child has potential.</div>
    </div>
  </div>
</section>
<section class="article-promo component">
  <div class="article-promo__content">
    <div class="article-promo__content-head">
      <h2 class="article-promo__title">
        <a href="/ideas/articles/cowpea-farmer/">
          <span>The cowpea farmer breaking norms</span>
        </a>
      </h2>
    </div>
    <div class="article-promo__content-body">
      <div class="article-promo__description">Meet the Nigerian farmer.</div>
    </div>
  </div>
</section>
</body>
</html>
"""


# --- Date parsing tests ---


class TestParseFlexibleDate:
    def test_iso_format(self):
        assert _parse_flexible_date("2026-03-05") == "2026-03-05"

    def test_iso_with_time(self):
        assert _parse_flexible_date("2026-03-05T19:59:21.508Z") == "2026-03-05"

    def test_month_day_year(self):
        assert _parse_flexible_date("Mar 5, 2026") == "2026-03-05"

    def test_month_day_year_no_comma(self):
        assert _parse_flexible_date("Mar 5 2026") == "2026-03-05"

    def test_full_month_name(self):
        assert _parse_flexible_date("February 20, 2026") == "2026-02-20"

    def test_day_month_year(self):
        assert _parse_flexible_date("5 Mar 2026") == "2026-03-05"

    def test_explicit_format(self):
        assert _parse_flexible_date("05/03/2026", fmt="%d/%m/%Y") == "2026-03-05"

    def test_empty_string(self):
        assert _parse_flexible_date("") is None

    def test_whitespace(self):
        assert _parse_flexible_date("  Mar 5, 2026  ") == "2026-03-05"

    def test_unparsable(self):
        assert _parse_flexible_date("not a date") is None

    def test_dec_18(self):
        assert _parse_flexible_date("Dec 18, 2024") == "2024-12-18"

    def test_feb_with_extra_text(self):
        # Epoch AI style: "Feb 20, 2026 · By David Atanasov"
        assert _parse_flexible_date("Feb 20, 2026 · By David Atanasov") == "2026-02-20"


# --- Crawler tests ---


def _make_source_config(key, name, crawl_config):
    return {
        "key": key,
        "name": name,
        "crawler_type": "html_scraper",
        "crawl_config": crawl_config,
    }


ANTHROPIC_CONFIG = _make_source_config(
    "anthropic_research",
    "Anthropic Research",
    {
        "listing_url": "https://www.anthropic.com/research",
        "article_selector": "ul[class*='PublicationList'] li",
        "title_selector": "span[class*='title']",
        "url_selector": "a[class*='listItem']",
        "date_selector": "time",
        "base_url": "https://www.anthropic.com",
    },
)

EPOCH_CONFIG = _make_source_config(
    "epoch_ai",
    "Epoch AI Blog",
    {
        "listing_url": "https://epoch.ai/blog",
        "article_selector": "a.blog-post-card",
        "title_selector": ".post-title",
        "url_selector": "a.blog-post-card",
        "date_selector": ".post-date-and-authors",
        "content_selector": ".post-description",
        "base_url": "https://epoch.ai",
    },
)

GATES_CONFIG = _make_source_config(
    "gates_foundation",
    "Gates Foundation Ideas",
    {
        "listing_url": "https://www.gatesfoundation.org/ideas",
        "article_selector": "section.article-promo",
        "title_selector": ".article-promo__title a span",
        "url_selector": ".article-promo__title a",
        "content_selector": ".article-promo__description",
        "base_url": "https://www.gatesfoundation.org",
    },
)


class TestHtmlScraperCrawler:
    @patch("digest.crawlers.html_scraper.resilient_get")
    def test_anthropic_parsing(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = SAMPLE_ANTHROPIC_HTML
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        crawler = HtmlScraperCrawler()
        items = crawler.fetch_items(ANTHROPIC_CONFIG, days_back=None)

        assert len(items) == 2
        assert items[0]["title"] == "Labor market impacts of AI"
        assert (
            items[0]["url"] == "https://www.anthropic.com/research/labor-market-impacts"
        )
        assert items[0]["date"] == "2026-03-05"
        assert items[0]["source"] == "anthropic_research"
        assert items[0]["source_name"] == "Anthropic Research"

        assert items[1]["title"] == "Alignment faking in large language models"
        assert items[1]["url"] == "https://www.anthropic.com/research/alignment-faking"
        assert items[1]["date"] == "2024-12-18"

    @patch("digest.crawlers.html_scraper.resilient_get")
    def test_epoch_parsing(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = SAMPLE_EPOCH_HTML
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        crawler = HtmlScraperCrawler()
        items = crawler.fetch_items(EPOCH_CONFIG, days_back=None)

        assert len(items) == 2
        assert items[0]["title"] == "Expanding our analysis of biological AI models"
        assert items[0]["url"] == "https://epoch.ai/blog/expanding-analysis"
        assert items[0]["date"] == "2026-02-20"
        assert (
            items[0]["content"]
            == "We release a database of over 1,100 biological AI models."
        )
        assert items[0]["source"] == "epoch_ai"

    @patch("digest.crawlers.html_scraper.resilient_get")
    def test_gates_parsing(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = SAMPLE_GATES_HTML
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        crawler = HtmlScraperCrawler()
        items = crawler.fetch_items(GATES_CONFIG, days_back=None)

        assert len(items) == 2
        assert items[0]["title"] == "Helping every child reach their full potential"
        assert (
            items[0]["url"]
            == "https://www.gatesfoundation.org/ideas/articles/core-mission-child-health/"
        )
        assert items[0]["content"] == "Every child has potential."
        assert items[0]["source"] == "gates_foundation"

        assert items[1]["title"] == "The cowpea farmer breaking norms"

    @patch("digest.crawlers.html_scraper.resilient_get")
    def test_date_filtering(self, mock_get):
        from datetime import datetime, timedelta

        # Use dynamic dates so the test doesn't rot over time:
        # - "recent_date" is yesterday (always within 7-day window)
        # - "old_date" is 60 days ago (always outside 7-day window)
        recent_date = (datetime.now() - timedelta(days=1)).strftime("%b %-d, %Y")
        old_date = (datetime.now() - timedelta(days=60)).strftime("%b %-d, %Y")

        date_filter_html = f"""
<html>
<body>
<ul class="PublicationList-module-scss-module__KxYrHG__list">
  <li>
    <a class="PublicationList-module-scss-module__KxYrHG__listItem" href="/research/recent-article">
      <div class="PublicationList-module-scss-module__KxYrHG__meta">
        <time class="PublicationList-module-scss-module__KxYrHG__date body-3">{recent_date}</time>
      </div>
      <span class="PublicationList-module-scss-module__KxYrHG__title body-3">Recent article</span>
    </a>
  </li>
  <li>
    <a class="PublicationList-module-scss-module__KxYrHG__listItem" href="/research/old-article">
      <div class="PublicationList-module-scss-module__KxYrHG__meta">
        <time class="PublicationList-module-scss-module__KxYrHG__date body-3">{old_date}</time>
      </div>
      <span class="PublicationList-module-scss-module__KxYrHG__title body-3">Old article</span>
    </a>
  </li>
</ul>
</body>
</html>
"""
        mock_response = MagicMock()
        mock_response.text = date_filter_html
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        crawler = HtmlScraperCrawler()
        # Only look back 7 days - should filter out the old article
        items = crawler.fetch_items(ANTHROPIC_CONFIG, days_back=7)

        # Only the recent article should remain
        assert len(items) == 1
        assert items[0]["title"] == "Recent article"

    @patch("digest.crawlers.html_scraper.resilient_get")
    def test_dedup_within_scrape(self, mock_get):
        # HTML with duplicate URLs
        html = """
        <html><body>
        <ul class="PublicationList-test">
          <li>
            <a class="listItem-test" href="/research/same-article">
              <span class="title-test">Article Title</span>
            </a>
          </li>
          <li>
            <a class="listItem-test" href="/research/same-article">
              <span class="title-test">Article Title Duplicate</span>
            </a>
          </li>
        </ul>
        </body></html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        config = _make_source_config(
            "test",
            "Test",
            {
                "listing_url": "https://example.com",
                "article_selector": "li",
                "title_selector": "span[class*='title']",
                "url_selector": "a[class*='listItem']",
                "base_url": "https://example.com",
            },
        )

        crawler = HtmlScraperCrawler()
        items = crawler.fetch_items(config, days_back=None)

        assert len(items) == 1

    @patch("digest.crawlers.html_scraper.resilient_get")
    def test_handles_http_error(self, mock_get):
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        crawler = HtmlScraperCrawler()
        items = crawler.fetch_items(ANTHROPIC_CONFIG, days_back=7)

        assert isinstance(items, list)
        assert items == []

    @patch("digest.crawlers.html_scraper.resilient_get")
    def test_handles_no_matching_articles(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>No articles here</p></body></html>"
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        crawler = HtmlScraperCrawler()
        items = crawler.fetch_items(ANTHROPIC_CONFIG, days_back=7)

        assert items == []

    def test_missing_listing_url(self):
        config = _make_source_config(
            "test",
            "Test",
            {"article_selector": "li"},  # missing listing_url
        )

        crawler = HtmlScraperCrawler()
        items = crawler.fetch_items(config, days_back=7)

        assert items == []

    def test_missing_required_selectors(self):
        config = _make_source_config(
            "test",
            "Test",
            {"listing_url": "https://example.com"},  # missing selectors
        )

        crawler = HtmlScraperCrawler()
        items = crawler.fetch_items(config, days_back=7)

        assert items == []

    def test_crawl_config_as_json_string(self):
        """crawl_config may arrive as a JSON string from the database."""
        import json

        config = _make_source_config(
            "test",
            "Test",
            json.dumps({"listing_url": "https://example.com"}),
        )

        crawler = HtmlScraperCrawler()
        # Should not crash even with missing selectors
        items = crawler.fetch_items(config, days_back=7)
        assert items == []
