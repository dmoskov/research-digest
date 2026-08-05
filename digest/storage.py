"""
Storage layer for persisting digest pipeline items to PostgreSQL.

Called after classification to upsert items, classifications, topics,
and network connections into the database.

Database access goes through digest.db (psycopg2 with RealDictCursor).
"""

import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

from digest.db import get_connection, get_cursor
from digest.dedup import compute_content_hash
from digest.settings import get_config

logger = logging.getLogger(__name__)

# Ordered confidence levels for filtering queries
CONFIDENCE_LEVELS = ["high", "medium", "low", "uncertain"]


_FEED_BASE_QUERY = """
        SELECT DISTINCT i.id, i.url, i.title, i.abstract, i.content,
               i.authors, i.source, i.source_id, i.source_name,
               i.published_date, i.raw_metadata
        FROM items i
    """


def _apply_subtopic_filter(filter_config: dict, joins: list, where_clauses: list, params: list) -> None:
    """Add subtopic and confidence filters to the query."""
    subtopics = filter_config.get("subtopics", [])
    if not subtopics:
        return

    joins.append("JOIN classifications c ON c.item_id = i.id")
    where_clauses.append("c.subtopic = ANY(%s)")
    params.append(subtopics)
    where_clauses.append("c.relevant = true")

    min_confidence = filter_config.get("min_confidence")
    if min_confidence and min_confidence in CONFIDENCE_LEVELS:
        allowed = CONFIDENCE_LEVELS[: CONFIDENCE_LEVELS.index(min_confidence) + 1]
        where_clauses.append("c.confidence = ANY(%s)")
        params.append(allowed)


def _apply_topic_filter(filter_config: dict, joins: list, where_clauses: list, params: list) -> None:
    """Add topic filters to the query."""
    topics = filter_config.get("topics", [])
    if not topics:
        return

    joins.append("JOIN item_topics it ON it.item_id = i.id")
    where_clauses.append("it.topic_key = ANY(%s)")
    params.append(topics)

    # Scope topics to the requested subtopics if both are specified
    subtopics = filter_config.get("subtopics", [])
    if subtopics:
        where_clauses.append("it.subtopic = ANY(%s)")
        params.append(subtopics)


def _apply_cg_filter(filter_config: dict, joins: list) -> None:
    """Add network-connection filter to the query."""
    if filter_config.get("cg_connected_only", False):
        joins.append("JOIN cg_connections cg ON cg.item_id = i.id")


def _apply_date_filter(week_start: str, week_end: str, where_clauses: list, params: list) -> None:
    """Add date range filters to the query."""
    if week_start:
        where_clauses.append("i.published_date >= %s")
        params.append(week_start)
    if week_end:
        where_clauses.append("i.published_date <= %s")
        params.append(week_end)


def _build_feed_query(filter_config: dict, week_start: str, week_end: str) -> tuple[str, list]:
    """Build SQL query for feed filtering.

    Args:
        filter_config: Filter configuration dict
        week_start: Start date filter (inclusive)
        week_end: End date filter (inclusive)

    Returns:
        Tuple of (query string, parameters list)
    """
    params: list = []
    where_clauses: list = []
    joins: list = []

    _apply_subtopic_filter(filter_config, joins, where_clauses, params)
    _apply_topic_filter(filter_config, joins, where_clauses, params)
    _apply_cg_filter(filter_config, joins)
    _apply_date_filter(week_start, week_end, where_clauses, params)

    # Assemble query
    query = _FEED_BASE_QUERY
    for join in joins:
        query += "\n" + join
    if where_clauses:
        query += "\nWHERE " + " AND ".join(where_clauses)
    query += "\nORDER BY i.published_date DESC, i.id DESC"

    return query, params


def _assemble_feed_results(rows: list, classifications_by_item: dict, connections_by_item: dict) -> list[dict]:
    """Assemble feed results from database rows and related data.

    Args:
        rows: List of database row dicts
        classifications_by_item: Dict mapping item_id to classifications
        connections_by_item: Dict mapping item_id to network connections

    Returns:
        List of enriched item dicts
    """
    results = []
    for row in rows:
        item_id = row["id"]
        published = row["published_date"]
        if isinstance(published, (date, datetime)):
            published = published.isoformat()

        raw_metadata = row["raw_metadata"]
        if isinstance(raw_metadata, str):
            raw_metadata = json.loads(raw_metadata)

        result = {
            "id": item_id,
            "url": row["url"],
            "title": row["title"],
            "abstract": row["abstract"] or "",
            "content": row["content"] or "",
            "authors": row["authors"] or "",
            "source": row["source"],
            "source_id": row["source_id"] or "",
            "source_name": row["source_name"] or "",
            "published_date": published,
            "raw_metadata": raw_metadata or {},
            "classifications": classifications_by_item.get(item_id, {}),
            "cg_connection": connections_by_item.get(
                item_id,
                {
                    "has_connection": False,
                    "connection_name": "",
                    "connection_description": "",
                },
            ),
        }
        results.append(result)

    return results


def deduplicate_items(items: list) -> list:
    """Remove cross-source duplicates from a list of crawled items.

    Uses content_hash (normalized title + first-author last name) to detect
    duplicates.  When duplicates are found, keeps the item with the longest
    abstract (most content) and logs what was dropped.

    Returns a new list with duplicates removed.
    """
    seen: dict[str, int] = {}  # content_hash -> index in result list
    result: list = []

    for item in items:
        title = item.get("title", "")
        authors = item.get("authors", "") or item.get("author", "")
        if not title:
            result.append(item)
            continue

        h = compute_content_hash(title, authors)
        item["_content_hash"] = h  # stash for later DB storage

        if h in seen:
            existing = result[seen[h]]
            existing_len = len(existing.get("abstract", "") or "")
            new_len = len(item.get("abstract", "") or "")
            if new_len > existing_len:
                # Replace with the richer version
                logger.info(
                    "Dedup: replacing '%s' (%s) with '%s' (%s) — longer abstract",
                    existing.get("title", "")[:60],
                    existing.get("source", ""),
                    item.get("title", "")[:60],
                    item.get("source", ""),
                )
                result[seen[h]] = item
            else:
                logger.info(
                    "Dedup: dropping duplicate '%s' from %s (keeping %s)",
                    item.get("title", "")[:60],
                    item.get("source", ""),
                    existing.get("source", ""),
                )
        else:
            seen[h] = len(result)
            result.append(item)

    dropped = len(items) - len(result)
    if dropped:
        logger.info("Dedup: removed %d duplicates from %d items", dropped, len(items))
    return result


def get_enabled_sources(
    crawler_type: Optional[str] = None,
    subtopic: Optional[str] = None,
) -> List[dict]:
    """
    Fetch enabled sources from the database, optionally filtered.

    Args:
        crawler_type: Filter by crawler type (e.g. 'crossref', 'rss', 'arxiv_atom').
        subtopic: Filter to sources that include this subtopic.

    Returns:
        List of source dicts with all columns from the sources table.
    """
    conditions = ["is_enabled = true"]
    params: list = []

    if crawler_type:
        conditions.append("crawler_type = %s")
        params.append(crawler_type)

    if subtopic:
        conditions.append("%s = ANY(subtopics)")
        params.append(subtopic)

    where_sql = " AND ".join(conditions)

    with get_cursor() as cur:
        cur.execute(
            f"SELECT * FROM sources WHERE {where_sql} ORDER BY name",  # noqa: S608
            params,
        )
        return [dict(row) for row in cur.fetchall()]


def get_known_urls(days_back: int = 30) -> set:
    """
    Fetch URLs of items already in the database from the recent window.

    Used to skip re-crawling and re-classifying items we already have.

    Args:
        days_back: Only return URLs for items crawled within this many days.

    Returns:
        Set of URL strings.
    """
    with get_cursor() as cur:
        cur.execute(
            "SELECT url FROM items WHERE crawled_at >= NOW() - interval '%s days'",
            (days_back,),
        )
        return {row["url"] for row in cur.fetchall()}


def update_source_crawl_timestamp(
    source_key: str,
    status: str = "ok",
    error: str = None,
    item_count: int = None,
) -> None:
    """
    Update the last_crawled_at timestamp and crawl status for a source.

    Args:
        source_key: The unique key of the source.
        status: Crawl status — 'ok', 'empty', 'error', or 'pending'.
        error: Error message (set when status is 'error', cleared otherwise).
        item_count: If provided and 0, automatically sets status to 'empty'.
    """
    if item_count is not None and item_count == 0 and status == "ok":
        status = "empty"

    with get_cursor() as cur:
        cur.execute(
            "UPDATE sources SET last_crawled_at = NOW(), updated_at = NOW(), "
            "crawl_status = %s, crawl_error = %s, "
            "consecutive_failures = CASE WHEN %s = 'error' "
            "THEN consecutive_failures + 1 ELSE 0 END "
            "WHERE key = %s",
            (status, error, status, source_key),
        )


def _sanitize_future_date(item: dict) -> None:
    """Null out dates more than 7 days in the future (data quality guard)."""
    pub = item.get("date")
    if not pub:
        return
    try:
        if isinstance(pub, str):
            parsed = datetime.fromisoformat(pub.replace("Z", "+00:00")).replace(tzinfo=None)
        elif isinstance(pub, datetime):
            parsed = pub.replace(tzinfo=None) if pub.tzinfo else pub
        else:
            return
        if parsed > datetime.now() + timedelta(days=7):
            logger.warning(
                "Nulling future date on item %s: %s",
                item.get("url", "?")[:60],
                pub,
            )
            item["date"] = None
    except (ValueError, TypeError):
        pass


def _extract_item_fields(item: dict) -> dict:
    """Pull storage-relevant fields from a pipeline item dict."""
    title = item.get("title", "")
    authors = item.get("authors", "") or item.get("author", "")

    content_hash = item.get("_content_hash") or (compute_content_hash(title, authors) if title else None)

    raw_metadata = {}
    if item.get("images"):
        raw_metadata["images"] = item["images"]
    if item.get("keyword_scores"):
        raw_metadata["keyword_scores"] = item["keyword_scores"]

    refused_at = None
    refusal_type = None
    if item.get("_abstract_refused"):
        refused_at = datetime.now(timezone.utc)
        refusal_type = "abstract"

    return {
        "url": item["url"],
        "content_hash": content_hash,
        "title": title,
        "abstract": item.get("abstract", ""),
        "content": item.get("content", ""),
        "authors": authors,
        "source": item.get("source", ""),
        "source_id": item.get("paper_number", ""),
        "source_name": item.get("source_name", "") or item.get("substack_name", ""),
        "published_date": item.get("date"),
        "raw_metadata": json.dumps(raw_metadata),
        "refused_at": refused_at,
        "refusal_type": refusal_type,
    }


_UPSERT_ITEM_SQL = """
    INSERT INTO items (url, content_hash, title, abstract, content, authors, source,
                       source_id, source_name, published_date, raw_metadata,
                       api_refused_at, api_refusal_type)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (url) DO UPDATE SET
        content_hash = EXCLUDED.content_hash,
        title = EXCLUDED.title,
        abstract = EXCLUDED.abstract,
        content = EXCLUDED.content,
        authors = EXCLUDED.authors,
        source = EXCLUDED.source,
        source_id = EXCLUDED.source_id,
        source_name = EXCLUDED.source_name,
        published_date = COALESCE(EXCLUDED.published_date, items.published_date),
        raw_metadata = CASE WHEN EXCLUDED.raw_metadata IS NOT NULL AND EXCLUDED.raw_metadata != '{}'::jsonb
                       THEN EXCLUDED.raw_metadata
                       ELSE COALESCE(items.raw_metadata, EXCLUDED.raw_metadata) END,
        crawled_at = NOW(),
        api_refused_at = COALESCE(EXCLUDED.api_refused_at, items.api_refused_at),
        api_refusal_type = COALESCE(EXCLUDED.api_refusal_type, items.api_refusal_type)
    RETURNING id
"""


def store_item(item: dict) -> int:
    """
    Upsert a single crawled item by URL. Returns the item ID.

    Uses INSERT ... ON CONFLICT (url) DO UPDATE to refresh fields when
    the same URL is re-crawled.

    Args:
        item: Pipeline item dict with keys like title, abstract/content,
              authors/author, url, date, source, paper_number, substack_name,
              images, and optionally classification.

    Returns:
        The database ID of the inserted or updated item.

    Raises:
        ValueError: If item is missing required 'url' field.
        psycopg2.Error: On database errors.
    """
    if not item.get("url"):
        raise ValueError("Item must have a 'url' field")

    _sanitize_future_date(item)
    fields = _extract_item_fields(item)

    with get_cursor() as cur:
        cur.execute(
            _UPSERT_ITEM_SQL,
            (
                fields["url"],
                fields["content_hash"],
                fields["title"],
                fields["abstract"],
                fields["content"],
                fields["authors"],
                fields["source"],
                fields["source_id"],
                fields["source_name"],
                fields["published_date"],
                fields["raw_metadata"],
                fields["refused_at"],
                fields["refusal_type"],
            ),
        )
        return cur.fetchone()["id"]


def store_classifications(item_id: int, classification: dict, source_key: str = "") -> None:
    """
    Store per-subtopic classification results, topics, and network connections.

    Upserts into classifications, item_topics, and cg_connections tables.
    Old topic rows for a given (item_id, subtopic) are deleted and re-inserted
    to handle topic changes on reclassification.

    Args:
        item_id: Database ID of the item (from store_item).
        classification: The full item["classification"] dict containing:
            - subtopics: dict of subtopic_key -> {relevant, topics, confidence, reasoning}
            - cg_connection: {has_connection, connection_name, connection_description}

    Raises:
        ValueError: If item_id is not a positive integer.
        psycopg2.Error: On database errors.
    """
    if not isinstance(item_id, int) or item_id <= 0:
        raise ValueError(f"item_id must be a positive integer, got: {item_id}")

    subtopics = classification.get("subtopics", {})
    cg_connection = classification.get("cg_connection", {})

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Apply source-based auto-tags.
            # Only auto-tag if the item has no substantive topic from the LLM: a
            # movement-building org's "monthly update" with no specific policy
            # topic should get its catch-all tag, but the same org's housing
            # policy piece is left alone to sort under housing.
            config = get_config()
            secondary_topics = config.secondary_topics
            if source_key:
                for auto_subtopic, auto_topic in config.source_auto_tags.get(source_key, []):
                    if auto_subtopic in subtopics:
                        st_data = subtopics[auto_subtopic]
                        if st_data.get("relevant", False):
                            existing_topics = st_data.get("topics", [])
                            # Skip auto-tag if a substantive (non-secondary) topic exists
                            has_substantive = any(t not in secondary_topics for t in existing_topics)
                            if not has_substantive and auto_topic not in existing_topics:
                                existing_topics.append(auto_topic)
                                st_data["topics"] = existing_topics

            # Upsert each subtopic classification
            for subtopic_key, data in subtopics.items():
                relevant = data.get("relevant", False)
                confidence = data.get("confidence", "low")
                reasoning = data.get("reasoning", "")
                topics = data.get("topics", [])

                cur.execute(
                    """
                    INSERT INTO classifications (item_id, subtopic, relevant, confidence, reasoning)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (item_id, subtopic) DO UPDATE SET
                        relevant = EXCLUDED.relevant,
                        confidence = EXCLUDED.confidence,
                        reasoning = EXCLUDED.reasoning,
                        classified_at = NOW()
                    """,
                    (item_id, subtopic_key, relevant, confidence, reasoning),
                )

                # Delete existing topics for this (item_id, subtopic) then re-insert.
                # This handles cases where topics changed on reclassification.
                cur.execute(
                    """
                    DELETE FROM item_topics
                    WHERE item_id = %s AND subtopic = %s
                    """,
                    (item_id, subtopic_key),
                )

                for topic_key in topics:
                    # Normalize: LLM sometimes returns Title Case instead of snake_case
                    topic_key = topic_key.lower().replace(" ", "_").replace("-", "_")
                    cur.execute(
                        """
                        INSERT INTO item_topics (item_id, subtopic, topic_key)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (item_id, subtopic, topic_key) DO NOTHING
                        """,
                        (item_id, subtopic_key, topic_key),
                    )

            # Upsert network connection
            if cg_connection.get("has_connection"):
                cur.execute(
                    """
                    INSERT INTO cg_connections (item_id, connection_name, connection_description)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (item_id) DO UPDATE SET
                        connection_name = EXCLUDED.connection_name,
                        connection_description = EXCLUDED.connection_description
                    """,
                    (
                        item_id,
                        cg_connection.get("connection_name", ""),
                        cg_connection.get("connection_description", ""),
                    ),
                )
            else:
                # Remove stale connection if item was reclassified as not connected
                cur.execute(
                    """
                    DELETE FROM cg_connections WHERE item_id = %s
                    """,
                    (item_id,),
                )


def store_items(items: list) -> List[int]:
    """
    Store multiple items and their classifications. Returns list of item IDs.

    Each item is upserted individually. If an item has a 'classification' key,
    its classifications are stored as well.

    Args:
        items: List of pipeline item dicts. Each may optionally include
               a 'classification' key with subtopic/connection data.

    Returns:
        List of database item IDs, in the same order as the input list.

    Raises:
        psycopg2.Error: On database errors (partial writes may occur).
    """
    item_ids = []

    for item in items:
        item_id = store_item(item)
        item_ids.append(item_id)

        classification = item.get("classification")
        if classification:
            store_classifications(
                item_id,
                classification,
                source_key=item.get("source", ""),
            )

    return item_ids


def get_items_for_feed(
    filter_config: dict,
    week_start: Optional[str] = None,
    week_end: Optional[str] = None,
) -> List[dict]:
    """
    Query items matching a feed's filter configuration.

    Joins items with classifications, item_topics, and cg_connections
    to return enriched item dicts ready for digest generation.

    Args:
        filter_config: Dict with filtering keys:
            - subtopics (list[str]): Required. Subtopic keys to include (e.g. ["abundance"]).
            - topics (list[str], optional): If set, only items with at least one matching topic_key.
            - min_confidence (str, optional): Minimum confidence level. One of
              "high", "medium", "low", "uncertain". Items at or above this level
              are included.
            - cg_connected_only (bool, optional): If True, only return items with
              a network connection.
        week_start: Start date filter (inclusive), as "YYYY-MM-DD". Defaults to no lower bound.
        week_end: End date filter (inclusive), as "YYYY-MM-DD". Defaults to no upper bound.

    Returns:
        List of dicts, each containing item fields plus nested classification data:
        {
            "id": int,
            "url": str,
            "title": str,
            "abstract": str,
            "content": str,
            "authors": str,
            "source": str,
            "source_id": str,
            "source_name": str,
            "published_date": str,
            "raw_metadata": dict,
            "classifications": {
                subtopic_key: {
                    "relevant": bool,
                    "confidence": str,
                    "reasoning": str,
                    "topics": [str],
                }
            },
            "cg_connection": {
                "has_connection": bool,
                "connection_name": str,
                "connection_description": str,
            }
        }

    Raises:
        ValueError: If filter_config is missing required 'subtopics' key
                    (unless cg_connected_only is True).
        psycopg2.Error: On database errors.
    """
    if not filter_config.get("subtopics") and not filter_config.get("cg_connected_only", False):
        raise ValueError("filter_config must include 'subtopics' list or set 'cg_connected_only' to true")

    # Build the query dynamically based on filters
    query, params = _build_feed_query(filter_config, week_start, week_end)

    with get_cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    if not rows:
        return []

    # Collect item IDs for batch-fetching related data
    item_ids = [row["id"] for row in rows]

    # Fetch all classifications for these items
    classifications_by_item = _fetch_classifications(item_ids)

    # Fetch all network connections for these items
    connections_by_item = _fetch_cg_connections(item_ids)

    # Assemble and return results
    return _assemble_feed_results(rows, classifications_by_item, connections_by_item)


def _fetch_classifications(item_ids: List[int]) -> Dict[int, dict]:
    """
    Batch-fetch classifications and topics for a list of item IDs.

    Returns:
        Dict mapping item_id -> {subtopic_key -> {relevant, confidence, reasoning, topics}}.
    """
    if not item_ids:
        return {}

    result = {}

    with get_cursor() as cur:
        # Fetch classifications
        cur.execute(
            """
            SELECT item_id, subtopic, relevant, confidence, reasoning
            FROM classifications
            WHERE item_id = ANY(%s)
            """,
            (item_ids,),
        )
        for row in cur.fetchall():
            iid = row["item_id"]
            if iid not in result:
                result[iid] = {}
            result[iid][row["subtopic"]] = {
                "relevant": row["relevant"],
                "confidence": row["confidence"] or "",
                "reasoning": row["reasoning"] or "",
                "topics": [],
            }

        # Fetch topics
        cur.execute(
            """
            SELECT item_id, subtopic, topic_key
            FROM item_topics
            WHERE item_id = ANY(%s)
            """,
            (item_ids,),
        )
        for row in cur.fetchall():
            iid = row["item_id"]
            subtopic = row["subtopic"]
            if iid in result and subtopic in result[iid]:
                result[iid][subtopic]["topics"].append(row["topic_key"])

    return result


def _fetch_cg_connections(item_ids: List[int]) -> Dict[int, dict]:
    """
    Batch-fetch network connections for a list of item IDs.

    Returns:
        Dict mapping item_id -> {has_connection, connection_name, connection_description}.
    """
    if not item_ids:
        return {}

    result = {}

    with get_cursor() as cur:
        cur.execute(
            """
            SELECT item_id, connection_name, connection_description
            FROM cg_connections
            WHERE item_id = ANY(%s)
            """,
            (item_ids,),
        )
        for row in cur.fetchall():
            result[row["item_id"]] = {
                "has_connection": True,
                "connection_name": row["connection_name"] or "",
                "connection_description": row["connection_description"] or "",
            }

    return result
