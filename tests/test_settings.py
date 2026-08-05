"""Tests for DigestConfig validation and the configure/get_config seam.

A misconfigured pipeline that starts anyway is the expensive failure mode: it
crawls, classifies everything as irrelevant, and produces an empty digest that
looks like a quiet week. These tests pin the "refuse to start" behaviour.
"""

import dataclasses

import pytest

from digest.settings import (
    DigestConfig,
    configure,
    get_config,
    is_configured,
    register_cache_clear,
)

MINIMAL = {
    "org_name": "Acme",
    "subtopics": {
        "climate": {
            "name": "Climate",
            "description": "Decarbonisation",
            "team_context": "We fund decarbonisation.",
        }
    },
    "subtopic_topics": {
        "climate": {"grid": {"name": "Grid", "description": "Transmission", "keywords": ["grid"]}}
    },
}


def _config(**overrides):
    return DigestConfig(**{**MINIMAL, **overrides})


class TestValidation:
    def test_minimal_config_is_valid(self):
        config = _config()
        assert config.org_name == "Acme"
        assert config.claude_model  # defaults applied

    def test_empty_subtopics_rejected(self):
        with pytest.raises(ValueError, match="subtopics is empty"):
            _config(subtopics={}, subtopic_topics={})

    def test_missing_org_name_rejected(self):
        with pytest.raises(ValueError, match="org_name is required"):
            _config(org_name="")

    def test_subtopic_without_topics_rejected(self):
        with pytest.raises(ValueError, match="without any topic areas"):
            _config(subtopic_topics={})

    def test_orphan_topic_set_rejected(self):
        orphaned = dict(MINIMAL["subtopic_topics"])
        orphaned["nonexistent"] = {"x": {"name": "X", "description": "", "keywords": []}}
        with pytest.raises(ValueError, match="no matching subtopic"):
            _config(subtopic_topics=orphaned)

    @pytest.mark.parametrize("field", ["name", "description", "team_context"])
    def test_subtopic_missing_required_field_rejected(self, field):
        subtopics = {"climate": dict(MINIMAL["subtopics"]["climate"])}
        del subtopics["climate"][field]
        with pytest.raises(ValueError, match=f"missing '{field}'"):
            _config(subtopics=subtopics)

    def test_team_context_cannot_be_blank(self):
        """A blank team_context passes a naive presence check but ruins precision."""
        subtopics = {"climate": {**MINIMAL["subtopics"]["climate"], "team_context": ""}}
        with pytest.raises(ValueError, match="missing 'team_context'"):
            _config(subtopics=subtopics)


class TestConfigureSeam:
    def test_configure_then_get_returns_same_object(self):
        config = _config()
        configure(config)
        assert get_config() is config
        assert is_configured()

    def test_defaults_are_empty_not_none(self):
        config = _config()
        assert config.topic_groups == {}
        assert config.audit_keywords == {}
        assert config.source_auto_tags == {}
        assert config.secondary_topics == frozenset()
        assert config.static_sources == {}
        assert config.network_connections is None

    def test_config_is_frozen(self):
        config = _config()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.org_name = "Other"

    def test_reconfigure_fires_registered_cache_clears(self, monkeypatch):
        """Prompt builders memoise the taxonomy; a stale cache would classify
        against the previous config's subtopics."""
        import digest.settings as settings

        calls = []
        monkeypatch.setattr(settings, "_cache_clearers", [])
        register_cache_clear(lambda: calls.append(1))
        configure(_config())
        assert calls, "cache clear callback was not invoked on configure()"


class TestUnconfigured:
    def test_scoring_raises_a_pointed_error_when_unconfigured(self, monkeypatch):
        import digest.settings as settings

        monkeypatch.setattr(settings, "_config", None)
        assert not settings.is_configured()
        with pytest.raises(RuntimeError, match="digest is not configured"):
            settings.get_config()


class TestCrawlerIdentity:
    """The crawler identity is what every publisher we fetch sees.

    If it ever went back to being a module constant, every deployment would
    crawl the web under one name and the resulting rate-limit reputation,
    blocklisting and abuse complaints would land on whoever that name belongs
    to. These tests exist to keep it configuration.
    """

    def test_default_identity_is_generic(self):
        from digest.crawlers.resilient import rss_headers

        configure(_config())
        assert rss_headers()["User-Agent"] == "ResearchDigestBot/1.0"

    def test_configured_name_and_info_url_appear_in_user_agent(self):
        from digest.crawlers.resilient import html_headers, rss_headers

        configure(_config(bot_name="AcmeDigest", bot_info_url="https://acme.org/bot"))
        expected = "AcmeDigest/1.0 (+https://acme.org/bot)"
        assert rss_headers()["User-Agent"] == expected
        assert html_headers()["User-Agent"] == expected

    def test_api_headers_carry_the_mailto_for_the_polite_pool(self):
        """CrossRef and OpenAlex route callers with a mailto into a faster pool."""
        from digest.crawlers.resilient import api_headers

        configure(_config(bot_name="AcmeDigest", bot_contact="tech@acme.org"))
        assert api_headers()["User-Agent"] == "AcmeDigest/1.0 (mailto:tech@acme.org)"

    def test_api_headers_omit_an_empty_mailto(self):
        from digest.crawlers.resilient import api_headers

        configure(_config(bot_name="AcmeDigest"))
        agent = api_headers()["User-Agent"]
        assert "mailto" not in agent
        assert agent == "AcmeDigest/1.0"

    def test_identity_is_read_per_call_not_frozen_at_import(self):
        from digest.crawlers.resilient import rss_headers

        configure(_config(bot_name="First"))
        assert rss_headers()["User-Agent"].startswith("First/")
        configure(_config(bot_name="Second"))
        assert rss_headers()["User-Agent"].startswith("Second/")

    def test_no_identity_constants_survive_in_the_module(self):
        import digest.crawlers.resilient as resilient

        for leaked in ("BOT_USER_AGENT", "BOT_CONTACT", "RSS_HEADERS", "HTML_HEADERS", "API_HEADERS"):
            assert not hasattr(resilient, leaked), (
                f"{leaked} is a module constant again — the crawler identity must "
                "come from DigestConfig so it cannot be inherited by another deployment"
            )
