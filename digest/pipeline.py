#!/usr/bin/env python3
"""
Phase 1 of the digest pipeline: crawl → enrich → store.

Two-phase architecture:
  Phase 1 (this module):        crawl → enrich → store per source, no classification
  Phase 2 (digest.classify_worker): parallel async LLM classification

Each source's items are stored immediately after crawling and enrichment, so an
interrupted run loses at most the source in flight. Phase 2 then picks up
whatever is unclassified, which is why the phases can fail independently
without losing work.

Usage:
    research-digest crawl --store-db --use-state --days-back 30
    research-digest crawl --test-mode          # crawl only, store nothing

    # Legacy single-pass mode: crawl + classify + store together
    research-digest crawl --store-db --use-state --classify
"""

import argparse
import logging
import sys
import threading
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from digest.classifiers.relevance_classifier import RelevanceClassifier
from digest.crawlers.academic_journal_crawler import AcademicJournalCrawler
from digest.crawlers.arxiv_crawler import ArxivCrawler
from digest.crawlers.html_scraper import HtmlScraperCrawler
from digest.crawlers.nber_crawler import NBERCrawler
from digest.crawlers.osf_preprint_crawler import OsfPreprintCrawler
from digest.crawlers.substack_aggregator import SubstackAggregator
from digest.llm import get_anthropic_client
from digest.logging_config import setup_logging
from digest.scoring import get_subtopic_info
from digest.settings import get_config
from digest.state import DigestState

logger = logging.getLogger(__name__)

# Registry mapping crawler_type -> crawler class.
CRAWLER_REGISTRY = {
    "nber": NBERCrawler,
    "rss": SubstackAggregator,
    "crossref": AcademicJournalCrawler,
    "openalex": AcademicJournalCrawler,
    "arxiv_atom": ArxivCrawler,
    "html_scraper": HtmlScraperCrawler,
    "osf_preprint": OsfPreprintCrawler,
}


def _crawl_and_store_sources(args, days_back: int) -> tuple[int, int, int]:
    """Crawl all enabled sources, parallelized by crawler type.

    Sources are grouped by crawler_type. All groups run concurrently.
    Within the RSS group, sources are further parallelized (different domains).
    Rate-limited APIs (arXiv, CrossRef, OpenAlex) process sequentially within
    their group to respect API rate limits, but run concurrently with other groups.

    Each source's items are stored to DB immediately after crawling + enrichment,
    so no work is lost if the process is interrupted.

    Returns:
        Tuple of (total_items_stored, sources_succeeded, sources_failed)
    """
    logger.info("Step 1: Crawling all enabled sources...")
    from digest.storage import (
        deduplicate_items,
        get_enabled_sources,
        get_known_urls,
        store_item,
        update_source_crawl_timestamp,
    )

    sources = get_enabled_sources()
    logger.info(f"Found {len(sources)} total enabled sources")

    # Pre-fetch known URLs for dedup
    known_urls: set = set()
    if args.store_db:
        known_urls = get_known_urls(days_back=max(days_back * 2, 30))
        logger.info(f"Known URLs in DB: {len(known_urls)}")

    # Thread-safe access to shared state
    lock = threading.Lock()
    counters = {"succeeded": 0, "failed": 0, "stored": 0, "skipped": 0}

    # Group sources by crawler_type, filtering skipped/unknown types
    groups: dict[str, list] = defaultdict(list)
    for source in sources:
        ctype = source["crawler_type"]
        if ctype not in CRAWLER_REGISTRY:
            logger.warning(
                f"Unknown crawler_type '{ctype}' for {source['key']}, skipping"
            )
            with lock:
                counters["failed"] += 1
            continue
        if args.skip_nber and ctype == "nber":
            continue
        if args.skip_substack and ctype == "rss":
            continue
        groups[ctype].append(source)

    logger.info(
        f"Crawler groups: {', '.join(f'{k}({len(v)})' for k, v in groups.items())}"
    )

    def _process_source(source, crawler):
        """Process a single source: fetch, enrich, store. Thread-safe."""
        try:
            items = crawler.fetch_items(source, days_back=days_back)
            if not items:
                logger.info(f"{source['name']}: 0 items")
                update_source_crawl_timestamp(source["key"], status="ok", item_count=0)
                with lock:
                    counters["succeeded"] += 1
                return

            # Filter known URLs under lock for consistent snapshot
            with lock:
                new_items = [i for i in items if i.get("url") not in known_urls]
            skipped = len(items) - len(new_items)

            if not new_items:
                logger.info(f"{source['name']}: {len(items)} items (all known)")
                update_source_crawl_timestamp(
                    source["key"], status="ok", item_count=len(items)
                )
                with lock:
                    counters["succeeded"] += 1
                    counters["skipped"] += skipped
                return

            # In-memory dedup within this source batch
            new_items = deduplicate_items(new_items)

            # Enrich thin content (per-source, so failures are isolated)
            if not args.skip_enrichment and not args.test_mode:
                try:
                    from digest.enrichment import enrich_items

                    enrich_items(new_items)
                except Exception as e:
                    logger.warning(f"Enrichment failed for {source['name']}: {e}")

            # Generate abstracts for items without summaries
            if not args.skip_abstracts and not args.test_mode:
                try:
                    from digest.summarizer import (
                        retry_refused_with_gemini,
                        summarize_items,
                    )

                    client = get_anthropic_client()
                    summarize_items(new_items, client)
                    # Second pass: retry refused items with Gemini Flash
                    retry_refused_with_gemini(new_items)
                except Exception as e:
                    logger.warning(
                        f"Abstract generation failed for {source['name']}: {e}"
                    )

            # Store immediately to DB
            stored = 0
            if args.store_db:
                for item in new_items:
                    try:
                        store_item(item)
                        stored += 1
                        # Add to known set so later sources dedup against it
                        if item.get("url"):
                            with lock:
                                known_urls.add(item["url"])
                    except Exception as e:
                        logger.error(f"Failed to store item: {e}")

            with lock:
                counters["stored"] += stored
                counters["skipped"] += skipped
                counters["succeeded"] += 1

            if skipped:
                logger.info(
                    f"{source['name']}: {len(items)} items "
                    f"({stored} stored, {skipped} known)"
                )
            else:
                logger.info(f"{source['name']}: {stored} items stored")

            update_source_crawl_timestamp(
                source["key"], status="ok", item_count=len(items)
            )

        except Exception as e:
            logger.error(f"Error crawling {source['name']}: {e}")
            update_source_crawl_timestamp(
                source["key"], status="error", error=str(e)[:500]
            )
            with lock:
                counters["failed"] += 1

    def _process_group_sequential(ctype, group_sources):
        """Process sources sequentially (for rate-limited APIs like arXiv, CrossRef)."""
        crawler = CRAWLER_REGISTRY[ctype]()
        for source in group_sources:
            _process_source(source, crawler)

    def _process_group_parallel(ctype, group_sources, max_workers=8):
        """Process sources in parallel (for independent-domain crawlers like RSS)."""
        _local = threading.local()

        def _worker(source):
            # Each thread gets its own crawler instance (own requests.Session)
            if not hasattr(_local, "crawler"):
                _local.crawler = CRAWLER_REGISTRY[ctype]()
            _process_source(source, _local.crawler)

        workers = min(max_workers, len(group_sources))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=ctype) as pool:
            futures = [pool.submit(_worker, s) for s in group_sources]
            for f in as_completed(futures):
                f.result()  # propagate unexpected errors

    # Crawler types safe to parallelize within the group (independent domains)
    _PARALLEL_TYPES = {"rss", "html_scraper"}
    _RSS_WORKERS = 8

    group_count = len(groups)
    if group_count == 0:
        logger.info("No sources to crawl.")
        return 0, 0, 0

    # Run all crawler groups concurrently
    t0 = time.monotonic()
    with ThreadPoolExecutor(
        max_workers=group_count, thread_name_prefix="group"
    ) as pool:
        futures = {}
        for ctype, group_sources in groups.items():
            if ctype in _PARALLEL_TYPES:
                future = pool.submit(
                    _process_group_parallel, ctype, group_sources, _RSS_WORKERS
                )
            else:
                future = pool.submit(_process_group_sequential, ctype, group_sources)
            futures[future] = ctype

        for future in as_completed(futures):
            ctype = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error("Crawler group '%s' failed: %s", ctype, e)

    elapsed = time.monotonic() - t0
    logger.info(
        f"Crawl complete in {elapsed:.1f}s: {counters['stored']} new items stored, "
        f"{counters['skipped']} already known"
    )
    if counters["failed"]:
        logger.warning(
            f"{counters['succeeded']} sources succeeded, "
            f"{counters['failed']} failed"
        )

    return counters["stored"], counters["succeeded"], counters["failed"]


def _crawl_and_store_legacy(args, cutoff_date) -> tuple[int, int, int]:
    """Crawl legacy hardcoded sources, storing items per-source.

    Args:
        args: Parsed command-line arguments
        cutoff_date: Optional datetime cutoff

    Returns:
        Tuple of (total_items_stored, sources_succeeded, sources_failed)
    """
    from digest.storage import get_known_urls, store_item

    known_urls = (
        get_known_urls(days_back=max(args.days_back * 2, 30))
        if args.store_db
        else set()
    )

    total_stored = 0
    sources_succeeded = 0
    sources_failed = 0

    if not args.skip_nber and get_config().static_sources.get("nber", {}).get("enabled", False):
        logger.info("Step 1a: Crawling NBER working papers...")
        try:
            nber_crawler = NBERCrawler()
            papers = nber_crawler.fetch_multiple_pages(
                max_pages=args.nber_pages, per_page=50, cutoff_date=cutoff_date
            )
            new_papers = [p for p in papers if p.get("url") not in known_urls]
            if args.store_db:
                for item in new_papers:
                    store_item(item)
                    if item.get("url"):
                        known_urls.add(item["url"])
            total_stored += len(new_papers)
            logger.info(f"{len(new_papers)} NBER papers stored ({len(papers)} total)")
            sources_succeeded += 1
        except Exception as e:
            logger.error(f"Error crawling NBER: {e}")
            sources_failed += 1

    if not args.skip_substack and get_config().static_sources.get("substack", {}).get("enabled", False):
        logger.info("Step 1b: Aggregating Substack posts...")
        try:
            substack_feeds = get_config().static_sources["substack"].get("feeds", [])
            if substack_feeds:
                aggregator = SubstackAggregator()
                posts = aggregator.fetch_multiple_feeds(
                    substack_feeds, days_back=args.days_back
                )
                new_posts = [p for p in posts if p.get("url") not in known_urls]
                if args.store_db:
                    for item in new_posts:
                        store_item(item)
                        if item.get("url"):
                            known_urls.add(item["url"])
                total_stored += len(new_posts)
                logger.info(
                    f"{len(new_posts)} Substack posts stored ({len(posts)} total)"
                )
                sources_succeeded += 1
        except Exception as e:
            logger.error(f"Error aggregating Substack: {e}")
            sources_failed += 1

    return total_stored, sources_succeeded, sources_failed


def _classify_and_audit(all_items: list, subtopics_to_process: list) -> list:
    """Classify items for relevance and run category audit.

    Args:
        all_items: List of items to classify
        subtopics_to_process: List of subtopic keys to audit

    Returns:
        List of classified items

    Raises:
        Exception: If classification fails
    """
    logger.info("Step 2: Classifying items for relevance...")
    classifier = RelevanceClassifier()

    # Classify all items (against all subtopics in one pass)
    classified_items = classifier.classify_batch(all_items)
    logger.info(f"Classified {len(classified_items)} items")

    # Show cross-subtopic stats
    logger.info("Cross-subtopic relevance:")
    for st_key, st_info in get_config().subtopics.items():
        st_count = len(classifier.filter_relevant(classified_items, subtopic=st_key))
        logger.info(f"  - {st_info['name']}: {st_count} items")

    # Category audit pass for each subtopic
    for subtopic in subtopics_to_process:
        st_info = get_subtopic_info(subtopic)
        logger.info(f"Running category audit for {st_info['name']}...")
        classifier.category_audit_pass(classified_items, subtopic=subtopic)
        relevant = classifier.filter_relevant(classified_items, subtopic=subtopic)
        logger.info(f"{len(relevant)} items relevant to {st_info['name']} after audit")

    return classified_items


def main(argv=None):
    setup_logging()
    parser = argparse.ArgumentParser(description="Crawl enabled sources, enrich and store items")
    parser.add_argument(
        "--subtopic",
        type=str,
        default=None,
        choices=list(get_config().subtopics.keys()),
        help="Single subtopic mode (legacy). Omit to crawl all sources once.",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=30,
        help="Number of days to look back for content (default: 30)",
    )
    parser.add_argument(
        "--nber-pages",
        type=int,
        default=2,
        help="Number of NBER pages to crawl (default: 2)",
    )
    parser.add_argument(
        "--skip-nber",
        action="store_true",
        help="Skip NBER crawler",
    )
    parser.add_argument(
        "--skip-substack",
        action="store_true",
        help="Skip Substack aggregator",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test mode: crawl but don't store or classify (saves API costs)",
    )
    parser.add_argument(
        "--use-state",
        action="store_true",
        help="Use persistent state to track last run (recommended for production)",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="Path to state file (default: .digest_state.json)",
    )
    parser.add_argument(
        "--store-db",
        action="store_true",
        help="Store items to PostgreSQL database as they are crawled",
    )
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Skip full-text enrichment (use RSS content only)",
    )
    parser.add_argument(
        "--skip-abstracts",
        action="store_true",
        help="Skip LLM abstract generation for items without summaries",
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Also classify items inline (legacy). Default: crawl + store only.",
    )
    parser.add_argument(
        "--legacy-sources",
        action="store_true",
        help="[DEPRECATED] Use hardcoded sources from config/topics.py instead of database",
    )

    args = parser.parse_args(argv)

    if args.legacy_sources:
        warnings.warn(
            "--legacy-sources is deprecated and will be removed in a future release. "
            "Database sources (the default) should be used instead.",
            DeprecationWarning,
            stacklevel=1,
        )

    # Determine which subtopics to process
    if args.subtopic:
        subtopics_to_process = [args.subtopic]
    else:
        subtopics_to_process = list(get_config().subtopics.keys())

    logger.info("Research Digest Pipeline")
    logger.info(f"Subtopics: {', '.join(subtopics_to_process)}")
    logger.info(
        f"Mode: {'crawl + classify' if args.classify else 'crawl + store (classify separately)'}"
    )

    # Initialize state management
    state = DigestState(state_file=args.state_file) if args.use_state else None

    # Determine cutoff date
    if state:
        cutoff_date = state.get_cutoff_for_run(days_back=args.days_back)
        if cutoff_date:
            logger.info(
                f"Using state-based cutoff: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        else:
            logger.info(f"No previous state, looking back: {args.days_back} days")
    else:
        cutoff_date = (
            datetime.now() - timedelta(days=args.days_back) if args.days_back else None
        )
        logger.info(f"Looking back: {args.days_back} days")

    # Track pipeline-wide issues
    pipeline_warnings: list[str] = []
    crawl_failures = 0

    # ── Step 1: Crawl + Store per-source ──────────────────────────────
    if not args.legacy_sources:
        total_stored, _ok, crawl_failures = _crawl_and_store_sources(
            args, args.days_back
        )
    else:
        total_stored, _ok, crawl_failures = _crawl_and_store_legacy(args, cutoff_date)

    if total_stored == 0 and not args.test_mode:
        logger.info("No new items to process. Everything is up to date.")
        if state:
            state.mark_run_complete()
        return

    # Mark crawl as complete
    if state:
        state.mark_run_complete()
        logger.info(
            f"State updated: last_8am_cutoff = {state.get_last_8am_cutoff()}"
        )

    # ── Step 2 (optional): Inline classification ──────────────────────
    if args.classify and not args.test_mode:
        # Legacy mode: classify in this process (slow, sequential)
        # Fetch the items we just stored back from DB for classification
        from digest.storage import get_cursor, store_classifications

        logger.info(f"Step 2: Classifying {total_stored} items (inline mode)...")

        with get_cursor() as cur:
            cur.execute(
                """
                SELECT i.id, i.title, i.abstract, i.content, i.authors,
                       i.source, i.source_name
                FROM items i
                LEFT JOIN classifications c ON c.item_id = i.id
                WHERE c.id IS NULL
                  AND COALESCE(i.published_date, i.crawled_at::date) >= NOW() - INTERVAL '%s days'
                  AND COALESCE(i.published_date, i.crawled_at::date) <= NOW() + INTERVAL '1 day'
                ORDER BY COALESCE(i.published_date, i.crawled_at::date) DESC
            """,
                [args.days_back],
            )
            rows = cur.fetchall()

        if rows:
            items_to_classify = [
                {
                    "title": r["title"] or "",
                    "content": r["abstract"] or r["content"] or "",
                    "abstract": r["abstract"] or "",
                    "authors": r["authors"] or "",
                    "source": r["source_name"] or r["source"] or "",
                    "_db_id": r["id"],
                }
                for r in rows
            ]

            classified = _classify_and_audit(items_to_classify, subtopics_to_process)

            # Store classifications
            for item in classified:
                if item.get("classification"):
                    store_classifications(
                        item["_db_id"], item["classification"],
                        source_key=item.get("source", ""),
                    )

            logger.info(f"Classified and stored {len(classified)} items")
        else:
            logger.info("No unclassified items found.")
    elif not args.test_mode:
        logger.info(f"Crawl complete. {total_stored} items stored to database.")
        logger.info(
            "Classification will run separately via: research-digest classify"
        )

    # ── Final summary ─────────────────────────────────────────────────
    if args.store_db:
        try:
            from digest.storage import get_cursor

            with get_cursor() as cur:
                cur.execute(
                    "SELECT key, consecutive_failures FROM sources "
                    "WHERE is_enabled = true AND consecutive_failures >= 3 "
                    "ORDER BY consecutive_failures DESC"
                )
                dead = cur.fetchall()
            if dead:
                logger.warning(
                    "Dead feeds (>=3 consecutive failures): %s",
                    ", ".join(f"{row['key']} ({row['consecutive_failures']})" for row in dead),
                )
        except Exception as e:
            logger.error(f"Dead-feed check failed: {e}")

    if pipeline_warnings or crawl_failures:
        logger.warning("Digest pipeline completed with warnings:")
        if crawl_failures:
            logger.warning(f"  - {crawl_failures} source(s) failed to crawl")
        for w in pipeline_warnings:
            logger.warning(f"  - {w}")
        sys.exit(1)
    else:
        logger.info(f"Digest pipeline complete! ({total_stored} items stored)")


if __name__ == "__main__":
    main()
