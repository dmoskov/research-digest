"""Tests for the sources registry: validation and upsert."""

import json
from unittest.mock import MagicMock, patch

import pytest

from digest.sources import (
    VALID_CRAWLER_TYPES,
    load_sources_from_json,
    upsert_sources,
    validate_sources,
)

GOOD = {
    "key": "volts",
    "name": "Volts",
    "source_type": "blog",
    "crawler_type": "rss",
    "subtopics": ["climate"],
    "feed_url": "https://www.volts.wtf/feed",
}


class TestValidation:
    def test_valid_source_passes(self):
        assert validate_sources([GOOD]) == [GOOD]

    @pytest.mark.parametrize("field", ["key", "name", "source_type", "crawler_type"])
    def test_missing_required_field_reported(self, field):
        source = {k: v for k, v in GOOD.items() if k != field}
        with pytest.raises(ValueError, match=f"missing required field '{field}'"):
            validate_sources([source])

    def test_unknown_crawler_type_reported(self):
        """A typo here would otherwise surface as a source that never crawls."""
        with pytest.raises(ValueError, match="unknown crawler_type 'rrs'"):
            validate_sources([{**GOOD, "crawler_type": "rrs"}])

    def test_duplicate_keys_reported(self):
        with pytest.raises(ValueError, match="duplicate key"):
            validate_sources([GOOD, {**GOOD, "name": "Volts again"}])

    def test_all_problems_reported_at_once(self):
        """A seed file is fixed in one pass, so report everything."""
        with pytest.raises(ValueError) as exc:
            validate_sources([{**GOOD, "crawler_type": "bogus"}, {"name": "no key"}])
        message = str(exc.value)
        assert "unknown crawler_type" in message
        assert "missing required field 'key'" in message
        assert message.startswith("4 problem(s)")

    def test_registry_covers_every_documented_crawler(self):
        assert VALID_CRAWLER_TYPES == {
            "nber", "rss", "crossref", "openalex",
            "arxiv_atom", "html_scraper", "osf_preprint",
        }

    def test_crawler_types_match_the_pipeline_registry(self):
        """Drift here means a seedable source type the pipeline cannot dispatch."""
        from digest.pipeline import CRAWLER_REGISTRY

        assert set(CRAWLER_REGISTRY) == set(VALID_CRAWLER_TYPES)


class TestUpsert:
    def test_dry_run_writes_nothing(self):
        with patch("digest.sources.get_connection") as conn:
            assert upsert_sources([GOOD], dry_run=True) == 1
            conn.assert_not_called()

    def test_dry_run_still_validates(self):
        with pytest.raises(ValueError):
            upsert_sources([{**GOOD, "crawler_type": "bogus"}], dry_run=True)

    def test_upsert_executes_one_statement_per_source(self):
        cursor = MagicMock()
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor
        with patch("digest.sources.get_connection") as get_conn:
            get_conn.return_value.__enter__.return_value = conn
            count = upsert_sources([GOOD, {**GOOD, "key": "other"}])

        assert count == 2
        assert cursor.execute.call_count == 2

    def test_upsert_serialises_crawl_config_as_json(self):
        cursor = MagicMock()
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor
        with patch("digest.sources.get_connection") as get_conn:
            get_conn.return_value.__enter__.return_value = conn
            upsert_sources([{**GOOD, "crawl_config": {"max_pages": 3}}])

        params = cursor.execute.call_args[0][1]
        assert json.loads(params[-1]) == {"max_pages": 3}

    def test_upsert_defaults_optional_columns_to_none(self):
        cursor = MagicMock()
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor
        with patch("digest.sources.get_connection") as get_conn:
            get_conn.return_value.__enter__.return_value = conn
            upsert_sources([{k: v for k, v in GOOD.items() if k != "feed_url"}])

        params = cursor.execute.call_args[0][1]
        assert None in params  # url/feed_url/issn/openalex_source_id


class TestLoadFromJson:
    def test_loads_and_validates(self, tmp_path):
        path = tmp_path / "sources.json"
        path.write_text(json.dumps([GOOD]))
        assert load_sources_from_json(str(path)) == [GOOD]

    def test_rejects_non_array(self, tmp_path):
        path = tmp_path / "sources.json"
        path.write_text(json.dumps({"key": "volts"}))
        with pytest.raises(ValueError, match="expected a JSON array"):
            load_sources_from_json(str(path))

    def test_propagates_validation_errors(self, tmp_path):
        path = tmp_path / "sources.json"
        path.write_text(json.dumps([{"name": "incomplete"}]))
        with pytest.raises(ValueError, match="missing required field"):
            load_sources_from_json(str(path))
