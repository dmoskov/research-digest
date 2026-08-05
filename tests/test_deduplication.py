"""Tests for content hashing and deduplication."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from digest.dedup import compute_content_hash
from digest.storage import deduplicate_items


class TestComputeContentHash:
    """Tests for compute_content_hash normalisation."""

    def test_case_insensitive(self):
        h1 = compute_content_hash("Housing Supply Effects")
        h2 = compute_content_hash("housing supply effects")
        assert h1 == h2

    def test_punctuation_ignored(self):
        h1 = compute_content_hash("Housing: Supply & Effects!")
        h2 = compute_content_hash("Housing Supply Effects")
        assert h1 == h2

    def test_different_authors_different_hash(self):
        h1 = compute_content_hash("Same Title", "Alice Smith")
        h2 = compute_content_hash("Same Title", "Bob Jones")
        assert h1 != h2

    def test_same_author_same_hash(self):
        h1 = compute_content_hash("Same Title", "Alice Smith")
        h2 = compute_content_hash("Same Title", "Alice Smith")
        assert h1 == h2

    def test_empty_author_consistent(self):
        h1 = compute_content_hash("Title Only")
        h2 = compute_content_hash("Title Only", "")
        assert h1 == h2

    def test_whitespace_normalised(self):
        h1 = compute_content_hash("Housing   Supply   Effects")
        h2 = compute_content_hash("Housing Supply Effects")
        assert h1 == h2

    def test_returns_16_hex_chars(self):
        h = compute_content_hash("Any Title", "Any Author")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


class TestDeduplicateItems:
    """Tests for deduplicate_items."""

    def test_keeps_unique_items(self):
        items = [
            {"title": "Paper A", "authors": "Smith", "abstract": "short"},
            {"title": "Paper B", "authors": "Jones", "abstract": "short"},
        ]
        result = deduplicate_items(items)
        assert len(result) == 2

    def test_removes_exact_duplicate(self):
        items = [
            {"title": "Same Paper", "authors": "Smith", "abstract": "first version"},
            {"title": "Same Paper", "authors": "Smith", "abstract": "second"},
        ]
        result = deduplicate_items(items)
        assert len(result) == 1

    def test_keeps_richer_version(self):
        items = [
            {"title": "Paper X", "authors": "Smith", "abstract": "short", "source": "a"},
            {"title": "Paper X", "authors": "Smith", "abstract": "a much longer abstract with more detail", "source": "b"},  # noqa: E501
        ]
        result = deduplicate_items(items)
        assert len(result) == 1
        assert result[0]["source"] == "b"  # longer abstract wins

    def test_preserves_unique_across_sources(self):
        items = [
            {"title": "Paper A", "authors": "Smith", "source": "nber"},
            {"title": "Paper B", "authors": "Jones", "source": "substack"},
            {"title": "Paper C", "authors": "Lee", "source": "arxiv"},
        ]
        result = deduplicate_items(items)
        assert len(result) == 3

    def test_no_title_items_preserved(self):
        items = [
            {"title": "", "abstract": "no title 1"},
            {"title": "", "abstract": "no title 2"},
        ]
        result = deduplicate_items(items)
        assert len(result) == 2

    def test_content_hash_attached(self):
        items = [
            {"title": "Paper With Hash", "authors": "Author"},
        ]
        result = deduplicate_items(items)
        assert "_content_hash" in result[0]
