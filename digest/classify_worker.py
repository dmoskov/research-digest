#!/usr/bin/env python3
"""
Phase 2 of the digest pipeline: classify stored items.

Reads items that have no classification rows yet and scores them against the
configured taxonomy, using asyncio + AsyncAnthropic for concurrency. Runs as a
separate step after phase 1 (:mod:`digest.pipeline`) has stored raw items, so
neither phase loses the other's work when it fails.

Usage:
    research-digest classify                    # classify all unclassified
    research-digest classify --workers 8        # 8 concurrent API calls
    research-digest classify --max-items 500    # limit batch size
    research-digest classify --days-back 7      # only recent items
"""

import argparse
import asyncio
import logging
import sys
import time

from digest.classifiers.relevance_classifier import RelevanceClassifier
from digest.logging_config import setup_logging
from digest.scoring import get_subtopic_info
from digest.settings import get_config
from digest.storage import get_cursor, store_classifications

logger = logging.getLogger(__name__)


def get_unclassified_items(days_back: int = 30, max_items: int = 0) -> list:
    """Fetch items from DB that have no classification rows."""
    with get_cursor() as cur:
        sql = """
            SELECT i.id, i.title, i.abstract, i.content, i.authors, i.source,
                   i.source_name, i.published_date, i.url
            FROM items i
            LEFT JOIN classifications c ON c.item_id = i.id
            WHERE c.id IS NULL
              AND COALESCE(i.published_date, i.crawled_at::date) >= NOW() - INTERVAL '%s days'
              AND COALESCE(i.published_date, i.crawled_at::date) <= NOW() + INTERVAL '1 day'
            ORDER BY COALESCE(i.published_date, i.crawled_at::date) DESC
        """
        params = [days_back]

        if max_items > 0:
            sql += " LIMIT %s"
            params.append(max_items)

        cur.execute(sql, params)
        rows = cur.fetchall()

    items = []
    for row in rows:
        items.append({
            "db_id": row["id"],
            "title": row["title"] or "",
            "abstract": row["abstract"] or "",
            "content": row["content"] or "",
            "authors": row["authors"] or "",
            "source": row["source"] or "",
            "source_name": row["source_name"] or "",
            "published_date": row["published_date"],
            "url": row["url"] or "",
        })
    return items


async def classify_batch_async(
    classifier: RelevanceClassifier,
    batch: list,
    semaphore: asyncio.Semaphore,
    batch_num: int,
    total_batches: int,
) -> list:
    """Classify a batch of items using the sync classifier inside a thread.

    We use a semaphore to limit concurrency and run_in_executor to avoid
    blocking the event loop (the Anthropic sync client blocks on HTTP).
    """
    async with semaphore:
        loop = asyncio.get_event_loop()

        # Format items for the batch classifier
        batch_items = []
        for item in batch:
            batch_items.append({
                "title": item["title"],
                "content": item["abstract"] or item["content"] or "",
                "abstract": item["abstract"],
                "authors": item["authors"],
                "source": item["source_name"] or item["source"],
            })

        try:
            results = await loop.run_in_executor(
                None,
                lambda: classifier._classify_batch_items(batch_items),
            )

            # Store results immediately
            classified = 0
            for item, result in zip(batch, results):
                if result is not None:
                    store_classifications(
                        item["db_id"], result,
                        source_key=item.get("source", ""),
                    )
                    classified += 1

            logger.info(
                f"  Batch {batch_num}/{total_batches}: "
                f"classified {classified}/{len(batch)} items"
            )
            return results

        except Exception as e:
            logger.error(f"  Batch {batch_num}/{total_batches} failed: {e}")
            # Fall back to individual classification
            results = []
            for item in batch:
                try:
                    result = await loop.run_in_executor(
                        None,
                        lambda i=item: classifier._classify_item_fallback(
                            {
                                "title": i["title"],
                                "content": i["abstract"] or i["content"] or "",
                                "authors": i["authors"],
                                "source": i["source_name"] or i["source"],
                            }
                        ),
                    )
                    store_classifications(
                        item["db_id"], result,
                        source_key=item.get("source", ""),
                    )
                    results.append(result)
                except Exception as e2:
                    logger.error(f"  Individual fallback failed for {item['db_id']}: {e2}")
                    results.append(None)
            return results


async def run_parallel_classification(
    items: list,
    workers: int = 4,
    batch_size: int = 5,
):
    """Classify items in parallel batches.

    Args:
        items: List of unclassified items from DB
        workers: Number of concurrent API calls
        batch_size: Items per API call (max 5 for the batch prompt)
    """
    classifier = RelevanceClassifier()
    semaphore = asyncio.Semaphore(workers)

    # Split into batches
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i : i + batch_size])

    total_batches = len(batches)
    logger.info(
        f"Classifying {len(items)} items in {total_batches} batches "
        f"({batch_size}/batch, {workers} concurrent workers)"
    )

    start_time = time.time()

    # Run all batches with concurrency limit
    tasks = [
        classify_batch_async(classifier, batch, semaphore, i + 1, total_batches)
        for i, batch in enumerate(batches)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start_time

    # Count results
    classified = 0
    errors = 0
    for r in results:
        if isinstance(r, Exception):
            errors += 1
        elif isinstance(r, list):
            classified += sum(1 for x in r if x is not None)

    logger.info("\nClassification complete:")
    logger.info(f"  Items: {classified}/{len(items)} classified")
    logger.info(f"  Errors: {errors} batch failures")
    logger.info(f"  Time: {elapsed:.1f}s ({len(items) / max(elapsed, 1):.1f} items/sec)")

    return classified, errors


def run_category_audit(days_back: int = 7):
    """Run category audit on recently classified items.

    This is a lightweight post-pass that checks for empty topic categories
    and tries to fill them with uncertain matches.
    """
    logger.info("\nRunning category audit...")
    classifier = RelevanceClassifier()

    # Get recently classified items with their classifications
    with get_cursor() as cur:
        cur.execute("""
            SELECT i.id, i.title, i.abstract, i.content, i.authors, i.source,
                   i.source_name,
                   json_agg(json_build_object(
                       'subtopic', c.subtopic,
                       'relevant', c.relevant,
                       'confidence', c.confidence,
                       'reasoning', c.reasoning
                   )) as classifications
            FROM items i
            JOIN classifications c ON c.item_id = i.id
            WHERE COALESCE(i.published_date, i.crawled_at::date) >= NOW() - INTERVAL '%s days'
              AND COALESCE(i.published_date, i.crawled_at::date) <= NOW() + INTERVAL '1 day'
            GROUP BY i.id, i.title, i.abstract, i.content, i.authors, i.source, i.source_name
        """, [days_back])
        rows = cur.fetchall()

    if not rows:
        logger.info("  No items to audit")
        return

    # Reconstruct items with classification dicts for the audit pass
    items = []
    for row in rows:
        classification = {"subtopics": {}, "cg_connection": {}}
        for c in row["classifications"]:
            classification["subtopics"][c["subtopic"]] = {
                "relevant": c["relevant"],
                "confidence": c["confidence"],
                "reasoning": c["reasoning"],
            }
        items.append({
            "title": row["title"] or "",
            "content": row["abstract"] or row["content"] or "",
            "abstract": row["abstract"] or "",
            "authors": row["authors"] or "",
            "source": row["source_name"] or row["source"] or "",
            "classification": classification,
        })

    for subtopic in get_config().subtopics:
        st_info = get_subtopic_info(subtopic)
        classifier.category_audit_pass(items, subtopic=subtopic)
        relevant = classifier.filter_relevant(items, subtopic=subtopic)
        logger.info(f"  {st_info['name']}: {len(relevant)} relevant items")


def main(argv=None):
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Classify unclassified items in parallel"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent API calls (default: 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Items per API call (default: 5, max 5)",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=30,
        help="Only classify items from last N days (default: 30)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Max items to classify (0 = all, default: 0)",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip the category audit pass",
    )
    args = parser.parse_args(argv)

    # Fetch unclassified items
    items = get_unclassified_items(
        days_back=args.days_back,
        max_items=args.max_items,
    )

    if not items:
        logger.info("No unclassified items found. Nothing to do.")
        return

    logger.info(f"Found {len(items)} unclassified items")

    # Run parallel classification
    classified, errors = asyncio.run(
        run_parallel_classification(
            items,
            workers=args.workers,
            batch_size=min(args.batch_size, 5),
        )
    )

    # Category audit
    if not args.skip_audit:
        run_category_audit(days_back=args.days_back)

    # Summary
    print(f"\n{'=' * 60}")
    if errors == 0:
        print(f"✓ Classified {classified} items ({args.workers} workers)")
    else:
        print(f"⚠ Classified {classified} items with {errors} batch errors")
    print(f"{'=' * 60}")

    sys.exit(1 if errors > 0 else 0)


if __name__ == "__main__":
    main()
