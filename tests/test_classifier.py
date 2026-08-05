"""Tests for the relevance classifier."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from digest.scoring import (
    calculate_keyword_score,
    calculate_subtopic_score,
    check_network_connection,
    get_all_keyword_candidates,
)


class TestKeywordScoring:
    """Tests for keyword-based scoring functions."""

    def test_housing_keywords_detected(self):
        text = "The impact of zoning reform on housing supply in major cities"
        score = calculate_keyword_score(text, "housing", "abundance")
        assert score > 0

    def test_irrelevant_text_scores_zero(self):
        text = "A study on medieval pottery techniques in 12th century France"
        score = calculate_keyword_score(text, "housing", "abundance")
        assert score == 0.0

    def test_ai_safety_keywords_detected(self):
        text = "New approaches to mechanistic interpretability in large language models"
        score = calculate_keyword_score(text, "interpretability", "ai_safety")
        assert score > 0

    def test_multiple_keyword_matches_increase_score(self):
        text_one = "zoning reform in cities"
        text_many = "zoning reform housing supply affordable housing construction multifamily housing"
        score_one = calculate_keyword_score(text_one, "housing", "abundance")
        score_many = calculate_keyword_score(text_many, "housing", "abundance")
        assert score_many > score_one

    def test_subtopic_score_takes_max(self):
        text = "Nuclear energy policy and permitting reform for power plants"
        score = calculate_subtopic_score(text, "abundance")
        energy_score = calculate_keyword_score(text, "energy", "abundance")
        assert score >= energy_score

    def test_unknown_topic_returns_zero(self):
        score = calculate_keyword_score("anything", "nonexistent_topic", "abundance")
        assert score == 0.0


class TestCGConnectionDetection:
    """Tests for CG network connection detection."""

    def test_author_connection_detected(self):
        result = check_network_connection("Some text", "Jane Quimby")
        assert result["has_connection"] is True
        assert result["connection_type"] == "author"

    def test_organization_detected_in_text(self):
        result = check_network_connection("Research from the Institute for Widgets on housing")
        assert result["has_connection"] is True
        assert result["connection_type"] == "organization"

    def test_publication_detected(self):
        result = check_network_connection("Analysis from The Widget Review newsletter")
        assert result["has_connection"] is True
        assert result["connection_type"] == "publication"

    def test_no_connection_for_unrelated_text(self):
        result = check_network_connection("Random unrelated content about sports", "Unknown Author")
        assert result["has_connection"] is False


class TestGetAllKeywordCandidates:
    """Tests for cross-subtopic keyword matching."""

    def test_housing_text_matches_abundance(self):
        text = "The effect of zoning on housing supply and affordability"
        candidates = get_all_keyword_candidates(text)
        assert "abundance" in candidates

    def test_ai_text_matches_ai_subtopics(self):
        text = "New results in AI alignment and interpretability research"
        candidates = get_all_keyword_candidates(text)
        assert "ai_safety" in candidates

    def test_irrelevant_text_no_matches(self):
        text = "Ancient Roman architecture and its lasting influence"
        candidates = get_all_keyword_candidates(text, threshold=0.05)
        assert len(candidates) == 0


class TestClassifyItem:
    """Tests for classify_item with mocked API."""

    def test_classify_item_with_mock(self, mock_anthropic_client):
        from digest.classifiers.relevance_classifier import RelevanceClassifier

        with patch(
            "digest.classifiers.relevance_classifier.get_anthropic_client",
            return_value=mock_anthropic_client,
        ):
            classifier = RelevanceClassifier(api_key="test-key")
            result = classifier.classify_item(
                title="Housing Supply and Zoning Reform",
                content="This paper studies the effect of upzoning on housing construction.",
                item_type="research paper",
            )

        assert "subtopics" in result
        assert "cg_connection" in result
        # The keyword pre-filter should pass this to the API
        # and we should get back the mocked response for abundance
        if "abundance" in result["subtopics"]:
            assert result["subtopics"]["abundance"]["relevant"] is True

    def test_classify_item_api_failure_returns_conservative(self, mock_anthropic_client):
        from digest.classifiers.relevance_classifier import RelevanceClassifier

        mock_anthropic_client.messages.create.side_effect = Exception("API error")

        with patch(
            "digest.classifiers.relevance_classifier.get_anthropic_client",
            return_value=mock_anthropic_client,
        ):
            classifier = RelevanceClassifier(api_key="test-key")
            result = classifier.classify_item(
                title="Housing Supply and Zoning Reform",
                content="This paper studies the effect of upzoning on housing construction.",
            )

        assert "subtopics" in result
        # On error, all should be marked not relevant
        for st_data in result["subtopics"].values():
            assert st_data["relevant"] is False


class TestPromptCaching:
    """Tests for the cached system prompt used by classification calls."""

    def test_system_prompt_is_byte_stable(self):
        from digest.classifiers.relevance_classifier import (
            _build_cached_system_prompt,
        )

        assert _build_cached_system_prompt() == _build_cached_system_prompt()

    def test_system_prompt_contains_all_subtopics_sorted(self):
        from digest.classifiers.relevance_classifier import (
            _build_cached_system_prompt,
        )
        from digest.settings import get_config

        prompt = _build_cached_system_prompt()
        positions = []
        for st_key in sorted(get_config().subtopics.keys()):
            pos = prompt.find(f"({st_key})")
            assert pos != -1, f"subtopic {st_key} missing from system prompt"
            positions.append(pos)
        assert positions == sorted(positions), "subtopics not in sorted order"

    def test_classify_item_sends_cached_system_block(self, mock_anthropic_client):
        from digest.classifiers.relevance_classifier import (
            RelevanceClassifier,
            _build_cached_system_prompt,
        )

        with patch(
            "digest.classifiers.relevance_classifier.get_anthropic_client",
            return_value=mock_anthropic_client,
        ):
            classifier = RelevanceClassifier(api_key="test-key")
            classifier.classify_item(
                title="Housing Supply and Zoning Reform",
                content="This paper studies the effect of upzoning on housing construction.",
            )

        kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        assert kwargs["system"] == [
            {
                "type": "text",
                "text": _build_cached_system_prompt(),
                "cache_control": {"type": "ephemeral"},
            }
        ]
        # The volatile user message must not duplicate the cached team context
        user_content = kwargs["messages"][0]["content"]
        assert "Team context:" not in user_content
