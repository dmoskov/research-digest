"""Managing the ``sources`` registry — what the pipeline crawls.

Sources live in the database, not in code, so they can be enabled, disabled and
have their crawl status tracked without a deploy. Seed them from a list of dicts:

    from digest.sources import upsert_sources

    upsert_sources([
        {
            "key": "nber",                       # stable identifier; the upsert key
            "name": "NBER Working Papers",
            "source_type": "academic_journal",   # free text, for display/filtering
            "crawler_type": "nber",              # must be a key of CRAWLER_REGISTRY
            "subtopics": ["economics"],          # which feeds this source belongs to
            "url": "https://www.nber.org/papers",
            "feed_url": "https://back.nber.org/rss/new.xml",
        },
    ])

Idempotent: re-running updates existing rows by ``key`` and leaves
``is_enabled`` / crawl status alone, so operator changes survive a reseed.
"""

import json
import logging
from typing import Iterable, List

from digest.db import get_connection

logger = logging.getLogger(__name__)

# crawler_type values the pipeline can dispatch. Kept in sync with
# digest.pipeline.CRAWLER_REGISTRY, and validated on seed rather than at crawl
# time — a typo here would otherwise show up as a source that silently never
# crawls.
VALID_CRAWLER_TYPES = frozenset(
    {"nber", "rss", "crossref", "openalex", "arxiv_atom", "html_scraper", "osf_preprint"}
)

REQUIRED_FIELDS = ("key", "name", "source_type", "crawler_type")

_UPSERT_SQL = """
    INSERT INTO sources (key, name, source_type, crawler_type,
                         subtopics, url, feed_url, issn,
                         openalex_source_id, crawl_config)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (key) DO UPDATE SET
        name = EXCLUDED.name,
        source_type = EXCLUDED.source_type,
        crawler_type = EXCLUDED.crawler_type,
        subtopics = EXCLUDED.subtopics,
        url = EXCLUDED.url,
        feed_url = EXCLUDED.feed_url,
        issn = EXCLUDED.issn,
        openalex_source_id = EXCLUDED.openalex_source_id,
        crawl_config = EXCLUDED.crawl_config,
        updated_at = NOW()
"""


def validate_sources(sources: Iterable[dict]) -> List[dict]:
    """Check a source list for missing fields, bad crawler types and duplicate keys.

    Raises ValueError listing every problem found, rather than the first — a
    seed file is usually fixed in one pass.
    """
    sources = list(sources)
    problems = []
    seen = {}
    for idx, source in enumerate(sources):
        label = source.get("key") or f"index {idx}"
        for required in REQUIRED_FIELDS:
            if not source.get(required):
                problems.append(f"{label}: missing required field '{required}'")
        crawler_type = source.get("crawler_type")
        if crawler_type and crawler_type not in VALID_CRAWLER_TYPES:
            problems.append(
                f"{label}: unknown crawler_type '{crawler_type}' "
                f"(valid: {', '.join(sorted(VALID_CRAWLER_TYPES))})"
            )
        key = source.get("key")
        if key:
            if key in seen:
                problems.append(f"{label}: duplicate key, also at index {seen[key]}")
            seen[key] = idx
    if problems:
        raise ValueError(
            f"{len(problems)} problem(s) in source definitions:\n  "
            + "\n  ".join(problems)
        )
    return sources


def upsert_sources(sources: Iterable[dict], dry_run: bool = False) -> int:
    """Insert or update sources by ``key``. Returns the number of rows written."""
    sources = validate_sources(sources)

    if dry_run:
        for source in sources:
            logger.info(
                "would upsert %-30s %s (%s)",
                source["key"], source["name"], source["crawler_type"],
            )
        return len(sources)

    with get_connection() as conn:
        with conn.cursor() as cur:
            for source in sources:
                cur.execute(
                    _UPSERT_SQL,
                    (
                        source["key"],
                        source["name"],
                        source["source_type"],
                        source["crawler_type"],
                        source.get("subtopics", []),
                        source.get("url"),
                        source.get("feed_url"),
                        source.get("issn"),
                        source.get("openalex_source_id"),
                        json.dumps(source.get("crawl_config", {})),
                    ),
                )
    logger.info("Upserted %d sources", len(sources))
    return len(sources)


def load_sources_from_json(path: str) -> List[dict]:
    """Read a source list from a JSON file (a bare array of source objects)."""
    with open(path) as fh:
        sources = json.load(fh)
    if not isinstance(sources, list):
        raise ValueError(f"{path}: expected a JSON array of source objects")
    return validate_sources(sources)
