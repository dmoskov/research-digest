"""Tests for the summarizer module (abstract generation)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from digest.summarizer import (
    _MAX_CONTENT_CHARS,
    generate_abstract,
    needs_abstract,
    summarize_items,
)


class TestNeedsAbstract:
    """Tests for the needs_abstract helper."""

    def test_none_needs_abstract(self):
        assert needs_abstract(None) is True

    def test_empty_string_needs_abstract(self):
        assert needs_abstract("") is True

    def test_whitespace_only_needs_abstract(self):
        assert needs_abstract("   ") is True

    def test_short_string_needs_abstract(self):
        assert needs_abstract("Too short") is True

    def test_exactly_49_chars_needs_abstract(self):
        text = "a" * 49
        assert needs_abstract(text) is True

    def test_exactly_50_chars_does_not_need_abstract(self):
        text = "a" * 50
        assert needs_abstract(text) is False

    def test_long_string_does_not_need_abstract(self):
        text = "This is a sufficiently long abstract that should not need regeneration at all."
        assert needs_abstract(text) is False

    def test_short_after_strip_needs_abstract(self):
        text = "   short   "
        assert needs_abstract(text) is True


class TestGenerateAbstract:
    """Tests for generate_abstract with mocked Claude API."""

    def _make_client(self, response_text):
        """Helper to create a mock Anthropic client returning given text."""
        client = MagicMock()
        msg = MagicMock()
        msg.content = [MagicMock(text=response_text)]
        client.messages.create.return_value = msg
        return client

    def test_returns_bullet_summary(self):
        bullets = "- Key finding about AI alignment\n- New method improves safety"
        client = self._make_client(bullets)
        result, was_refused = generate_abstract("Test Title", "x" * 100, "arxiv", client, delay=0)
        assert result == bullets
        assert was_refused is False

    def test_truncates_long_content(self):
        long_content = "x" * (_MAX_CONTENT_CHARS + 500)
        client = self._make_client("- Summary bullet one\n- Summary bullet two")
        generate_abstract("Title", long_content, "source", client, delay=0)

        call_args = client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        # The content in the user message should be truncated
        assert len(user_msg) < _MAX_CONTENT_CHARS + 500

    def test_returns_none_for_empty_content(self):
        client = self._make_client("anything")
        result, was_refused = generate_abstract("Title", "", "source", client, delay=0)
        assert result is None
        assert was_refused is False
        client.messages.create.assert_not_called()

    def test_returns_none_for_short_content(self):
        client = self._make_client("anything")
        result, was_refused = generate_abstract("Title", "too short", "source", client, delay=0)
        assert result is None
        assert was_refused is False
        client.messages.create.assert_not_called()

    def test_returns_none_on_api_error(self):
        client = MagicMock()
        client.messages.create.side_effect = Exception("API timeout")
        result, was_refused = generate_abstract("Title", "x" * 100, "source", client, delay=0)
        assert result is None
        assert was_refused is False

    def test_returns_none_on_empty_response(self):
        client = MagicMock()
        msg = MagicMock()
        msg.content = []
        msg.stop_reason = "end_turn"
        client.messages.create.return_value = msg
        result, was_refused = generate_abstract("Title", "x" * 100, "source", client, delay=0)
        assert result is None
        assert was_refused is False

    def test_returns_none_for_too_short_output(self):
        client = self._make_client("short")
        result, was_refused = generate_abstract("Title", "x" * 100, "source", client, delay=0)
        assert result is None
        assert was_refused is False

    def test_returns_none_on_refusal_i_cannot(self):
        client = self._make_client("I cannot summarize this content")
        result, was_refused = generate_abstract("Title", "x" * 100, "source", client, delay=0)
        assert result is None
        assert was_refused is True

    def test_returns_none_on_refusal_im_sorry(self):
        client = self._make_client("I'm sorry, I can't help with that request")
        result, was_refused = generate_abstract("Title", "x" * 100, "source", client, delay=0)
        assert result is None
        assert was_refused is True

    def test_uses_default_title_when_none(self):
        client = self._make_client(
            "- Bullet one about findings\n- Bullet two about results"
        )
        generate_abstract(None, "x" * 100, "source", client, delay=0)
        call_args = client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "Untitled" in user_msg

    def test_uses_default_source_when_none(self):
        client = self._make_client(
            "- Bullet one about findings\n- Bullet two about results"
        )
        generate_abstract("Title", "x" * 100, None, client, delay=0)
        call_args = client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "Unknown" in user_msg

    @patch("digest.summarizer.time.sleep")
    def test_respects_delay(self, mock_sleep):
        client = self._make_client(
            "- Bullet one about findings\n- Bullet two about results"
        )
        generate_abstract("Title", "x" * 100, "source", client, delay=0.5)
        mock_sleep.assert_called_once_with(0.5)

    @patch("digest.summarizer.time.sleep")
    def test_no_sleep_when_delay_zero(self, mock_sleep):
        client = self._make_client(
            "- Bullet one about findings\n- Bullet two about results"
        )
        generate_abstract("Title", "x" * 100, "source", client, delay=0)
        mock_sleep.assert_not_called()


class TestSummarizeItems:
    """Tests for the summarize_items batch function."""

    def _make_client(self, response_text):
        client = MagicMock()
        msg = MagicMock()
        msg.content = [MagicMock(text=response_text)]
        client.messages.create.return_value = msg
        return client

    def test_updates_items_missing_abstracts(self):
        client = self._make_client(
            "- Finding one from the study\n- Finding two about results"
        )
        items = [
            {"title": "Paper A", "content": "x" * 100, "abstract": None},
            {"title": "Paper B", "content": "x" * 100, "abstract": None},
        ]
        updated = summarize_items(items, client, delay=0)
        assert updated == 2
        assert items[0]["abstract"] is not None
        assert items[1]["abstract"] is not None

    def test_skips_items_with_good_abstracts(self):
        client = self._make_client("- Bullet")
        items = [
            {"title": "Paper A", "content": "x" * 100, "abstract": "a" * 60},
        ]
        updated = summarize_items(items, client, delay=0)
        assert updated == 0
        client.messages.create.assert_not_called()

    def test_skips_items_without_content(self):
        client = self._make_client("- Bullet")
        items = [
            {"title": "Paper A", "content": None, "abstract": None},
            {"title": "Paper B", "content": "", "abstract": None},
        ]
        updated = summarize_items(items, client, delay=0)
        assert updated == 0
        client.messages.create.assert_not_called()

    def test_returns_zero_for_empty_list(self):
        client = self._make_client("anything")
        assert summarize_items([], client, delay=0) == 0

    def test_skips_previously_refused_items(self):
        client = self._make_client("- Bullet")
        items = [
            {
                "title": "Paper A",
                "content": "x" * 100,
                "abstract": None,
                "_abstract_refused": True,
            },
        ]
        updated = summarize_items(items, client, delay=0)
        assert updated == 0
        client.messages.create.assert_not_called()

    def test_marks_refused_on_failure(self):
        client = MagicMock()
        client.messages.create.side_effect = Exception("API error")
        items = [
            {"title": "Paper A", "content": "x" * 100, "abstract": None},
        ]
        updated = summarize_items(items, client, delay=0)
        assert updated == 0
        assert items[0].get("_abstract_refused") is True

    def test_mixed_items_only_updates_candidates(self):
        client = self._make_client(
            "- Key insight from the research\n- Secondary finding noted"
        )
        items = [
            {"title": "Has abstract", "content": "x" * 100, "abstract": "a" * 60},
            {"title": "Needs abstract", "content": "x" * 100, "abstract": None},
            {"title": "No content", "abstract": None},
        ]
        updated = summarize_items(items, client, delay=0)
        assert updated == 1
        assert items[0]["abstract"] == "a" * 60  # unchanged
        assert items[1]["abstract"] is not None  # updated
        assert client.messages.create.call_count == 1

    def test_uses_source_name_field(self):
        client = self._make_client(
            "- Research finding one here\n- Another important finding"
        )
        items = [
            {
                "title": "Paper",
                "content": "x" * 100,
                "abstract": None,
                "source_name": "Nature",
            },
        ]
        summarize_items(items, client, delay=0)
        call_args = client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "Nature" in user_msg

    def test_falls_back_to_source_field(self):
        client = self._make_client(
            "- Research finding one here\n- Another important finding"
        )
        items = [
            {
                "title": "Paper",
                "content": "x" * 100,
                "abstract": None,
                "source": "arxiv",
            },
        ]
        summarize_items(items, client, delay=0)
        call_args = client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "arxiv" in user_msg
