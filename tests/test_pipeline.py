"""Tests for digest.pipeline — state management and error handling.

Covers:
1. store failure (crawl_and_store raises) should not advance state
2. mark_run_complete() must not execute when crawl_and_store crashes
3. Crawl errors (sources_failed > 0) must produce non-zero exit code
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _make_args(**overrides):
    """Build a fake args namespace mimicking argparse output."""
    defaults = {
        "subtopic": None,
        "days_back": 7,
        "nber_pages": 2,
        "skip_nber": False,
        "skip_substack": False,
        "test_mode": False,
        "use_state": True,
        "state_file": None,
        "store_db": True,
        "skip_enrichment": True,
        "skip_abstracts": True,
        "legacy_sources": False,
        "classify": False,
    }
    defaults.update(overrides)
    args = MagicMock()
    for k, v in defaults.items():
        setattr(args, k, v)
    return args


def _sample_items(n=3):
    """Return a list of n sample classified items."""
    return [
        {
            "title": f"Paper {i}",
            "url": f"https://example.com/paper{i}",
            "abstract": f"Abstract for paper {i}",
            "authors": f"Author {i}",
            "source": "nber",
            "date": "2025-01-15",
            "classification": {
                "subtopics": {
                    "abundance": {
                        "relevant": True,
                        "topics": ["housing"],
                        "confidence": "high",
                        "reasoning": "test",
                    }
                },
                "cg_connection": {"has_connection": False},
            },
        }
        for i in range(n)
    ]


class TestStateNotAdvancedOnStorageFailure:
    """Bug 1 & 2: state.mark_run_complete() must NOT be called when crawl_and_store crashes."""

    @patch("digest.pipeline._crawl_and_store_sources")
    @patch("digest.pipeline.argparse.ArgumentParser.parse_args")
    def test_state_not_advanced_on_store_failure(self, mock_parse_args, mock_crawl):
        """When _crawl_and_store_sources raises, mark_run_complete must NOT be called."""
        from digest.pipeline import main

        mock_parse_args.return_value = _make_args()
        mock_crawl.side_effect = Exception("DB connection refused")

        mock_state = MagicMock()
        mock_state.get_cutoff_for_run.return_value = None

        with patch(
            "digest.pipeline.DigestState",
            return_value=mock_state,
        ):
            with pytest.raises(Exception, match="DB connection refused"):
                main()

            # The critical assertion: state must NOT advance on storage failure
            mock_state.mark_run_complete.assert_not_called()

    @patch("digest.pipeline._crawl_and_store_sources")
    @patch("digest.pipeline.argparse.ArgumentParser.parse_args")
    def test_state_advanced_on_successful_store(self, mock_parse_args, mock_crawl):
        """When crawl+store succeeds, mark_run_complete should be called."""
        from digest.pipeline import main

        mock_parse_args.return_value = _make_args()
        # (total_stored, sources_ok, sources_failed)
        mock_crawl.return_value = (5, 3, 0)

        mock_state = MagicMock()
        mock_state.get_cutoff_for_run.return_value = None
        mock_state.get_last_8am_cutoff.return_value = "2025-01-15T08:00:00"

        with patch(
            "digest.pipeline.DigestState",
            return_value=mock_state,
        ):
            main()

            # State SHOULD advance on success
            mock_state.mark_run_complete.assert_called_once()

    @patch("digest.pipeline._crawl_and_store_sources")
    @patch("digest.pipeline.argparse.ArgumentParser.parse_args")
    def test_nonzero_exit_on_store_failure(self, mock_parse_args, mock_crawl):
        """When _crawl_and_store_sources raises, the error must propagate."""
        from digest.pipeline import main

        mock_parse_args.return_value = _make_args(use_state=False)
        mock_crawl.side_effect = Exception("DB write failure")

        with pytest.raises((SystemExit, Exception)):
            main()


class TestCrawlFailureExitCode:
    """Bug 3: Crawl failures should produce non-zero exit code."""

    @patch("digest.pipeline._crawl_and_store_sources")
    @patch("digest.pipeline.argparse.ArgumentParser.parse_args")
    def test_nonzero_exit_on_crawl_failures(self, mock_parse_args, mock_crawl):
        """When some crawl sources fail, exit code should be non-zero."""
        from digest.pipeline import main

        mock_parse_args.return_value = _make_args(store_db=True, use_state=False)
        # 3 items stored, 2 sources ok, 1 source failed
        mock_crawl.return_value = (3, 2, 1)

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0

    @patch("digest.pipeline._crawl_and_store_sources")
    @patch("digest.pipeline.argparse.ArgumentParser.parse_args")
    def test_zero_exit_on_no_failures(self, mock_parse_args, mock_crawl):
        """When all sources succeed and no warnings, exit code should be 0."""
        from digest.pipeline import main

        mock_parse_args.return_value = _make_args(store_db=True, use_state=False)
        # 5 items stored, 3 sources ok, 0 failed
        mock_crawl.return_value = (5, 3, 0)

        # Should not raise SystemExit (exits 0 implicitly via return)
        main()


class TestStorageFailureDoesNotSwallow:
    """The storage failure must not be silently swallowed — it should be raised or cause exit."""

    @patch("digest.pipeline._crawl_and_store_sources")
    @patch("digest.pipeline.argparse.ArgumentParser.parse_args")
    def test_store_failure_is_not_silently_swallowed(self, mock_parse_args, mock_crawl):
        """Crash in _crawl_and_store_sources must cause the run to fail."""
        from digest.pipeline import main

        mock_parse_args.return_value = _make_args(use_state=False)
        mock_crawl.side_effect = Exception("Connection refused")

        with pytest.raises((SystemExit, Exception)):
            main()


class TestStateManagementWithoutStoreDb:
    """State management when --store-db is not used."""

    @patch("digest.pipeline._crawl_and_store_sources")
    @patch("digest.pipeline.argparse.ArgumentParser.parse_args")
    def test_state_advances_when_no_store_db(self, mock_parse_args, mock_crawl):
        """Without --store-db, state should still advance on success."""
        from digest.pipeline import main

        mock_parse_args.return_value = _make_args(store_db=False)
        # 0 items stored (no DB), 3 sources ok, 0 failed
        mock_crawl.return_value = (0, 3, 0)

        mock_state = MagicMock()
        mock_state.get_cutoff_for_run.return_value = None
        mock_state.get_last_8am_cutoff.return_value = None

        with patch(
            "digest.pipeline.DigestState",
            return_value=mock_state,
        ):
            main()
            mock_state.mark_run_complete.assert_called_once()


class TestDigestState:
    """Unit tests for the DigestState class itself."""

    def test_mark_run_complete_updates_state(self, tmp_path):
        from digest.state import DigestState

        state_file = tmp_path / "test_state.json"
        state = DigestState(state_file=str(state_file))

        assert state.get_last_run() is None
        assert state.get_last_8am_cutoff() is None

        state.mark_run_complete()

        assert state.get_last_run() is not None
        assert state.get_last_8am_cutoff() is not None

    def test_state_persists_across_instances(self, tmp_path):
        from digest.state import DigestState

        state_file = tmp_path / "test_state.json"
        state1 = DigestState(state_file=str(state_file))
        state1.mark_run_complete()
        cutoff1 = state1.get_last_8am_cutoff()

        # New instance should load persisted state
        state2 = DigestState(state_file=str(state_file))
        assert state2.get_last_8am_cutoff() == cutoff1

    def test_state_not_updated_means_same_cutoff(self, tmp_path):
        """If mark_run_complete is NOT called, the cutoff should not change."""
        from digest.state import DigestState

        state_file = tmp_path / "test_state.json"
        state = DigestState(state_file=str(state_file))

        # No previous state
        assert state.get_last_8am_cutoff() is None

        # Don't call mark_run_complete — simulate failed run
        # Next instance should still have no cutoff
        state2 = DigestState(state_file=str(state_file))
        assert state2.get_last_8am_cutoff() is None
