"""
AI-powered topic summarization for digest feeds.

Generates concise summaries of groups of articles on a topic. Summaries are
cached in the database, keyed by a hash of the item IDs they cover, and include
deep links back to the rendering site (DigestConfig.site_base_url).
"""

import hashlib
import logging
from datetime import date
from typing import Optional

from digest.db import get_cursor
from digest.llm import get_anthropic_client
from digest.scoring import get_subtopic_topics
from digest.settings import get_config
from digest.usage import log_usage

logger = logging.getLogger(__name__)


def compute_item_hash(item_ids: list[int]) -> str:
    """Compute a stable hash of item IDs for cache invalidation."""
    sorted_ids = sorted(item_ids)
    blob = ",".join(str(i) for i in sorted_ids)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def get_cached_summary(
    feed_id: int,
    topic_key: str,
    week_ending: date,
    item_ids: list[int],
) -> Optional[str]:
    """
    Retrieve cached summary if available and items haven't changed.

    Args:
        feed_id: Feed ID
        topic_key: Topic identifier (e.g., "housing", "ai_alignment")
        week_ending: Week ending date
        item_ids: List of item IDs for this topic

    Returns:
        Cached summary text if valid, None otherwise
    """
    current_hash = compute_item_hash(item_ids)

    with get_cursor() as cur:
        cur.execute(
            """
            SELECT summary, item_hash
            FROM topic_summaries
            WHERE feed_id = %s AND topic_key = %s AND week_ending = %s
            """,
            (feed_id, topic_key, week_ending),
        )
        row = cur.fetchone()

        if row:
            # get_cursor yields RealDictRow: index by column name. Unpacking the
            # row as a tuple binds the *keys*, so the hash never matches and the
            # cache never hits — an expensive silent failure.
            cached_summary = row["summary"]
            cached_hash = row["item_hash"]
            if cached_hash == current_hash:
                logger.info(
                    "Cache hit: feed=%d topic=%s week=%s",
                    feed_id,
                    topic_key,
                    week_ending,
                )
                return cached_summary
            else:
                logger.info(
                    "Cache stale: feed=%d topic=%s week=%s (items changed)",
                    feed_id,
                    topic_key,
                    week_ending,
                )
    return None


def save_summary(
    feed_id: int,
    topic_key: str,
    week_ending: date,
    summary: str,
    item_ids: list[int],
) -> None:
    """
    Save generated summary to database with item hash for cache validation.

    Args:
        feed_id: Feed ID
        topic_key: Topic identifier
        week_ending: Week ending date
        summary: Generated summary text
        item_ids: List of item IDs included in summary
    """
    item_hash = compute_item_hash(item_ids)

    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO topic_summaries (feed_id, topic_key, week_ending, summary, item_hash)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (feed_id, topic_key, week_ending)
            DO UPDATE SET
                summary = EXCLUDED.summary,
                item_hash = EXCLUDED.item_hash,
                created_at = NOW()
            """,
            (feed_id, topic_key, week_ending, summary, item_hash),
        )

    logger.info(
        "Saved summary: feed=%d topic=%s week=%s (%d items)",
        feed_id,
        topic_key,
        week_ending,
        len(item_ids),
    )


def _format_items_for_prompt(items: list[dict], feed_slug: str) -> str:
    """Format item dicts into a numbered text block with deeplinks."""
    base_url = get_config().site_base_url.rstrip("/")
    items_text = []
    for idx, item in enumerate(items, 1):
        title = item.get("title", "Untitled")
        abstract = item.get("abstract", "")
        pub_date = item.get("published_date", "")
        item_id = item.get("id", "")

        card_link = f"{base_url}/digests/{feed_slug}#item-{item_id}"

        items_text.append(
            f"[{idx}] {title}\n"
            f"    Date: {pub_date}\n"
            f"    Link: {card_link}\n"
            f"    Abstract: {abstract[:300]}...\n"
        )

    return "\n".join(items_text)


def _build_summary_prompt(
    topic_name: str, topic_description: str, items_block: str, num_items: int
) -> str:
    """Build the Claude prompt for topic summarization."""
    return f"""You are generating a concise digest summary for research items on the topic "{topic_name}" ({topic_description}).

Your task: Write a 2-4 sentence summary that:
1. Identifies the key themes or trends across these {num_items} items
2. Highlights the most significant or interesting findings
3. Uses inline markdown links to reference specific items like [this](URL)
4. Is written for an expert audience familiar with the topic

Guidelines:
- Be specific and concrete - mention actual findings, not just "several papers discuss X"
- Link directly to item cards using the provided card_link URLs
- Focus on what's NEW or INTERESTING, not just describing the topic
- Keep it under 100 words
- Use present tense ("Smith finds that..." not "Smith found that...")

Items:
{items_block}

Write your summary now (2-4 sentences, include inline links):"""  # noqa: E501


def generate_topic_summary(
    topic_key: str,
    topic_name: str,
    items: list[dict],
    subtopic: str,
    feed_slug: str,
) -> str:
    """
    Generate AI summary for a group of items on a topic.

    Args:
        topic_key: Topic identifier (e.g., "housing")
        topic_name: Human-readable topic name (e.g., "Housing")
        items: List of item dicts with title, abstract, url, published_date
        subtopic: Subtopic key (e.g., "abundance", "ai_safety")
        feed_slug: Feed slug for generating deeplinks

    Returns:
        Generated summary as markdown text with inline links
    """
    if not items:
        return ""

    topics_config = get_subtopic_topics(subtopic)
    topic_info = topics_config.get(topic_key, {})
    topic_description = topic_info.get("description", "")

    items_block = _format_items_for_prompt(items, feed_slug)
    prompt = _build_summary_prompt(
        topic_name, topic_description, items_block, len(items)
    )

    try:
        client = get_anthropic_client(timeout=60.0)
        response = client.messages.create(
            model=get_config().claude_model,
            max_tokens=500,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        log_usage(response, "digest_topic_summary")

        summary = response.content[0].text.strip()
        logger.info(
            "Generated summary for topic=%s: %d chars from %d items",
            topic_key,
            len(summary),
            len(items),
        )
        return summary

    except Exception as e:
        logger.error("Failed to generate summary for topic=%s: %s", topic_key, e)
        return f"{len(items)} recent items on {topic_name}."


def _extract_item_ids(items: list[dict]) -> list[int]:
    """Extract item IDs from a list of item dicts."""
    return [item["id"] for item in items]


def _generate_and_save_summary(
    feed_id: int,
    feed_slug: str,
    topic_key: str,
    topic_name: str,
    items: list[dict],
    subtopic: str,
    week_ending: date,
    item_ids: list[int],
) -> str:
    """Generate a new summary via AI and save it to the cache."""
    summary = generate_topic_summary(
        topic_key=topic_key,
        topic_name=topic_name,
        items=items,
        subtopic=subtopic,
        feed_slug=feed_slug,
    )
    if summary:
        save_summary(feed_id, topic_key, week_ending, summary, item_ids)
    return summary


def generate_and_cache_summary(
    feed_id: int,
    feed_slug: str,
    topic_key: str,
    topic_name: str,
    items: list[dict],
    subtopic: str,
    week_ending: date,
    use_cache: bool = True,
) -> str:
    """Generate or retrieve cached summary for a topic."""
    if not items:
        return ""

    item_ids = _extract_item_ids(items)

    if use_cache:
        cached = get_cached_summary(feed_id, topic_key, week_ending, item_ids)
        if cached:
            return cached

    return _generate_and_save_summary(
        feed_id,
        feed_slug,
        topic_key,
        topic_name,
        items,
        subtopic,
        week_ending,
        item_ids,
    )


def generate_summaries_for_feed(
    feed_id: int,
    feed_slug: str,
    by_topic: dict[str, list[dict]],
    subtopic: str,
    week_ending: date,
    use_cache: bool = True,
) -> dict[str, str]:
    """
    Generate summaries for all topics in a feed.

    Args:
        feed_id: Feed ID
        feed_slug: Feed slug for deeplinks
        by_topic: Dict mapping topic_key -> list of items
        subtopic: Subtopic key (e.g., "abundance")
        week_ending: Week ending date
        use_cache: Whether to use cached summaries

    Returns:
        Dict mapping topic_key -> summary text
    """
    summaries = {}
    topics_config = get_subtopic_topics(subtopic)

    for topic_key, items in by_topic.items():
        if not items:
            continue

        topic_info = topics_config.get(topic_key, {})
        topic_name = topic_info.get(
            "name", topic_key.replace("_", " ").title().replace("Ai ", "AI ")
        )

        summary = generate_and_cache_summary(
            feed_id=feed_id,
            feed_slug=feed_slug,
            topic_key=topic_key,
            topic_name=topic_name,
            items=items,
            subtopic=subtopic,
            week_ending=week_ending,
            use_cache=use_cache,
        )

        if summary:
            summaries[topic_key] = summary

    return summaries
