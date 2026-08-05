"""Content crawlers for various sources."""

from .academic_journal_crawler import AcademicJournalCrawler
from .arxiv_crawler import ArxivCrawler
from .html_scraper import HtmlScraperCrawler
from .nber_crawler import NBERCrawler
from .osf_preprint_crawler import OsfPreprintCrawler
from .substack_aggregator import SubstackAggregator

__all__ = [
    "NBERCrawler",
    "SubstackAggregator",
    "AcademicJournalCrawler",
    "ArxivCrawler",
    "HtmlScraperCrawler",
    "OsfPreprintCrawler",
]
