"""End-to-end tests against a real PostgreSQL.

Skipped unless DIGEST_DB_HOST is set, so the default `pytest` run stays
database-free. CI provides a Postgres service; locally:

    docker run -d --name rd-test -e POSTGRES_PASSWORD=test -e POSTGRES_USER=test \\
        -e POSTGRES_DB=digest_test -p 5433:5432 postgres:16

    DIGEST_DB_HOST=localhost DIGEST_DB_PORT=5433 DIGEST_DB_NAME=digest_test \\
    DIGEST_DB_USER=test DIGEST_DB_PASSWORD=test pytest tests/test_integration_db.py

These cover what the mocked tests structurally cannot: that the shipped schema
actually executes, that the SQL in the storage layer matches it, and that writes
commit. The last one matters because `get_connection()` commits on clean exit
and the storage layer never calls `commit()` itself — a connection factory that
does not commit makes every write vanish with no error anywhere.
"""

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DIGEST_DB_HOST"),
    reason="set DIGEST_DB_* to run database integration tests",
)

EXPECTED_TABLES = {
    "items", "classifications", "item_topics", "cg_connections", "sources",
    "feeds", "digest_snapshots", "topic_summaries", "user_read_items",
    "user_feed_preferences",
}


@pytest.fixture(scope="module", autouse=True)
def migrated(request):
    """Apply the schema once for the module, against whatever DB is configured."""
    from digest.migrate import run_migrations

    run_migrations()
    return True


@pytest.fixture
def clean_items():
    """Remove rows this test file created, leaving anything else alone."""
    yield
    from digest.db import get_cursor

    with get_cursor() as cur:
        cur.execute("DELETE FROM items WHERE url LIKE 'https://itest.invalid/%'")
        cur.execute("DELETE FROM sources WHERE key LIKE 'itest_%'")


def _item(**overrides):
    item = {
        "title": "Interconnection queue reform and grid storage",
        "abstract": "We study transmission interconnection delays.",
        "content": "Full text about grid storage and interconnection queue reform.",
        "authors": "Jane Quimby",
        "url": f"https://itest.invalid/{uuid.uuid4()}",
        "source": "itest_source",
        "source_name": "Integration Test Source",
        "date": "2026-01-15",
    }
    item.update(overrides)
    return item


class TestSchema:
    def test_migrate_creates_every_documented_table(self):
        from digest.db import get_cursor

        with get_cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
            )
            present = {r["table_name"] for r in cur.fetchall()}
        assert EXPECTED_TABLES <= present, f"missing: {EXPECTED_TABLES - present}"

    def test_migrate_is_idempotent(self):
        from digest.migrate import run_migrations

        assert run_migrations() == [], "a second migrate re-applied something"

    def test_items_has_the_columns_storage_writes(self):
        """The 0.1.0 regression: migrate succeeded, every insert then failed."""
        from digest.db import get_cursor

        with get_cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name='items'"
            )
            columns = {r["column_name"] for r in cur.fetchall()}
        for required in ("api_refused_at", "api_refusal_type", "content_hash", "raw_metadata"):
            assert required in columns


class TestStorageRoundTrip:
    def test_store_item_commits(self, clean_items):
        """Writes must survive the connection closing — nothing calls commit()."""
        from digest.db import get_cursor
        from digest.storage import store_item

        item = _item()
        store_item(item)

        # Fresh cursor, therefore a fresh connection from the pool.
        with get_cursor() as cur:
            cur.execute("SELECT title, source FROM items WHERE url = %s", (item["url"],))
            row = cur.fetchone()
        assert row is not None, "stored item was not committed"
        assert row["title"] == item["title"]

    def test_store_item_is_idempotent_on_url(self, clean_items):
        from digest.db import get_cursor
        from digest.storage import store_item

        item = _item()
        store_item(item)
        store_item({**item, "title": "Revised title"})

        with get_cursor() as cur:
            cur.execute("SELECT count(*) AS n, max(title) AS t FROM items WHERE url = %s", (item["url"],))
            row = cur.fetchone()
        assert row["n"] == 1, "re-storing the same URL created a duplicate row"
        assert row["t"] == "Revised title"

    def test_classification_round_trip(self, clean_items):
        from digest.db import get_cursor
        from digest.storage import get_items_for_feed, store_classifications, store_item

        item = _item()
        store_item(item)
        with get_cursor() as cur:
            cur.execute("SELECT id FROM items WHERE url = %s", (item["url"],))
            item_id = cur.fetchone()["id"]

        store_classifications(
            item_id,
            {
                "subtopics": {
                    "abundance": {
                        "relevant": True,
                        "topics": ["housing"],
                        "confidence": "high",
                        "reasoning": "zoning and housing supply",
                    }
                },
                "cg_connection": {
                    "has_connection": True,
                    "connection_name": "Jane Quimby",
                    "connection_description": "fixture",
                },
            },
            source_key=item["source"],
        )

        results = get_items_for_feed({"subtopics": ["abundance"], "min_confidence": "medium"}, "", "")
        mine = [r for r in results if r["url"] == item["url"]]
        assert len(mine) == 1
        assert mine[0]["classifications"]["abundance"]["topics"] == ["housing"]
        assert mine[0]["cg_connection"]["connection_name"] == "Jane Quimby"

    def test_feed_filter_excludes_other_subtopics(self, clean_items):
        from digest.db import get_cursor
        from digest.storage import get_items_for_feed, store_classifications, store_item

        item = _item()
        store_item(item)
        with get_cursor() as cur:
            cur.execute("SELECT id FROM items WHERE url = %s", (item["url"],))
            item_id = cur.fetchone()["id"]
        store_classifications(
            item_id,
            {
                "subtopics": {
                    "abundance": {
                        "relevant": True,
                        "topics": ["housing"],
                        "confidence": "high",
                        "reasoning": "zoning",
                    }
                }
            },
            source_key=item["source"],
        )

        results = get_items_for_feed({"subtopics": ["ai_safety"]}, "", "")
        assert not [r for r in results if r["url"] == item["url"]]

    def test_dedup_by_content_hash_across_sources(self, clean_items):
        from digest.storage import deduplicate_items

        a = _item(source="itest_a")
        b = _item(source="itest_b", abstract="A much longer abstract " * 10)
        deduped = deduplicate_items([a, b])
        assert len(deduped) == 1, "same title+author from two sources should collapse"
        assert deduped[0]["source"] == "itest_b", "should keep the richer abstract"


class TestSourcesRegistry:
    def test_upsert_and_read_back(self, clean_items):
        from digest.sources import upsert_sources
        from digest.storage import get_enabled_sources

        upsert_sources([{
            "key": "itest_rss",
            "name": "Integration Test Feed",
            "source_type": "blog",
            "crawler_type": "rss",
            "subtopics": ["abundance"],
            "feed_url": "https://itest.invalid/feed",
        }])
        keys = {s["key"] for s in get_enabled_sources()}
        assert "itest_rss" in keys

    def test_crawl_status_is_recorded(self, clean_items):
        from digest.db import get_cursor
        from digest.sources import upsert_sources
        from digest.storage import update_source_crawl_timestamp

        upsert_sources([{
            "key": "itest_status",
            "name": "Status Test",
            "source_type": "blog",
            "crawler_type": "rss",
            "subtopics": ["abundance"],
        }])
        update_source_crawl_timestamp("itest_status", status="error", item_count=0)

        with get_cursor() as cur:
            cur.execute(
                "SELECT crawl_status, last_crawled_at FROM sources WHERE key='itest_status'"
            )
            row = cur.fetchone()
        assert row["crawl_status"] == "error"
        assert row["last_crawled_at"] is not None
