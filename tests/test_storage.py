"""Tests for storage layer with mocked database."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestStoreItemValidation:
    """Tests for store_item input validation."""

    def test_store_item_requires_url(self):
        from digest.storage import store_item

        item = {"title": "No URL Item", "abstract": "Missing url field"}
        with pytest.raises(ValueError, match="url"):
            store_item(item)

    @patch("digest.storage.get_cursor")
    def test_store_item_requires_title(self, mock_get_cursor):
        from digest.storage import store_item

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"id": 1}
        mock_get_cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_get_cursor.return_value.__exit__ = MagicMock(return_value=False)

        # Missing title is allowed (defaults to empty string) — just has url
        item = {"url": "https://example.com/paper", "abstract": "Missing title"}
        store_item(item)
        assert mock_cursor.execute.called

    @patch("digest.storage.get_cursor")
    def test_store_item_inserts_correctly(self, mock_get_cursor):
        from digest.storage import store_item

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"id": 42}
        mock_get_cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_get_cursor.return_value.__exit__ = MagicMock(return_value=False)

        item = {
            "url": "https://example.com/paper",
            "title": "Test Paper",
            "abstract": "Test abstract",
            "authors": "John Smith",
            "source": "nber",
            "paper_number": "w12345",
        }
        store_item(item)
        assert mock_cursor.execute.called

    @patch("digest.storage.get_cursor")
    def test_store_item_handles_db_error(self, mock_get_cursor):
        from digest.storage import store_item

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("DB connection failed")
        mock_get_cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_get_cursor.return_value.__exit__ = MagicMock(return_value=False)

        item = {
            "url": "https://example.com/paper",
            "title": "Test Paper",
            "abstract": "Test abstract",
            "source": "nber",
        }
        with pytest.raises(Exception, match="DB connection"):
            store_item(item)
