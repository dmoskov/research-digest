"""Tests for the relevance classifier."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def _refusal():
    """What the API returns when the model declines: no content, stop=refusal."""
    refusal = MagicMock()
    refusal.content = []
    refusal.stop_reason = "refusal"
    return refusal


_HOUSING_VERDICT = (
    '{"abundance": {"relevant": true, "topics": ["housing"], '
    '"confidence": "high", "reasoning": "from the fallback"}}'
)


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

    def test_classify_item_api_failure_returns_none(self, mock_anthropic_client):
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

        # None, not a fabricated verdict. Writing `not relevant` here used to
        # be permanent: get_unclassified_items only selects items with no
        # classification rows, so one transient failure removed the item from
        # every feed for good. None leaves it unclassified and retryable.
        assert result is None

    def test_truncated_response_leaves_the_item_unclassified(self, mock_anthropic_client):
        """A model that spends max_tokens on thinking returns no JSON at all."""
        from digest.classifiers.relevance_classifier import RelevanceClassifier

        truncated = MagicMock()
        truncated.content = [MagicMock(type="thinking", thinking="")]
        truncated.stop_reason = "max_tokens"
        mock_anthropic_client.messages.create.return_value = truncated

        with patch(
            "digest.classifiers.relevance_classifier.get_anthropic_client",
            return_value=mock_anthropic_client,
        ):
            classifier = RelevanceClassifier(api_key="test-key")
            result = classifier.classify_item(
                title="Housing Supply and Zoning Reform",
                content="This paper studies the effect of upzoning on housing construction.",
            )

        assert result is None

    def test_refusal_leaves_the_item_unclassified(self, mock_anthropic_client):
        """A stop_reason of refusal must not become a `not relevant` verdict.

        Observed in production 2026-09-25: 266 rows were written this way
        during a reclassification run, silently dropping 124 previously
        relevant items. With no fallback model to ask, the item stays
        unclassified.
        """
        from digest.classifiers.relevance_classifier import RelevanceClassifier

        mock_anthropic_client.messages.create.return_value = _refusal()

        with patch(
            "digest.classifiers.relevance_classifier.get_anthropic_client",
            return_value=mock_anthropic_client,
        ), patch(
            "digest.classifiers.relevance_classifier.gemini_generate",
            return_value=None,
        ):
            classifier = RelevanceClassifier(api_key="test-key")
            result = classifier.classify_item(
                title="Housing Supply and Zoning Reform",
                content="This paper studies the effect of upzoning on housing construction.",
            )

        assert result is None

    def test_filter_relevant_tolerates_an_unclassified_item(self):
        """classify_batch may now carry a None classification through."""
        from digest.classifiers.relevance_classifier import RelevanceClassifier

        with patch(
            "digest.classifiers.relevance_classifier.anthropic_credentials_available",
            return_value=True,
        ), patch("digest.classifiers.relevance_classifier.get_anthropic_client"):
            classifier = RelevanceClassifier(api_key="test-key")

        items = [
            {"title": "a", "classification": None},
            {
                "title": "b",
                "classification": {
                    "subtopics": {"abundance": {"relevant": True}}
                },
            },
        ]
        assert classifier.filter_relevant(items, "abundance") == [items[1]]


class TestRefusalFallback:
    """An item the primary model declines is classified by the fallback model.

    Observed in production 2026-11: the classifier declined about three items
    a day, almost all papers on dangerous pathogens or software exploits. With
    nowhere else to go they aged out unclassified, so the teams that follow
    exactly that subject matter never saw them.
    """

    def _classifier(self, client):
        from digest.classifiers.relevance_classifier import RelevanceClassifier

        with patch(
            "digest.classifiers.relevance_classifier.get_anthropic_client",
            return_value=client,
        ):
            return RelevanceClassifier(api_key="test-key")

    def test_refused_item_is_classified_by_the_fallback(self, mock_anthropic_client):
        from digest.classifiers.relevance_classifier import _build_cached_system_prompt

        mock_anthropic_client.messages.create.return_value = _refusal()
        classifier = self._classifier(mock_anthropic_client)

        with patch(
            "digest.classifiers.relevance_classifier.gemini_generate",
            return_value=f"```json\n{_HOUSING_VERDICT}\n```",
        ) as fallback:
            result = classifier.classify_item(
                title="Housing Supply and Zoning Reform",
                content="This paper studies the effect of upzoning on housing construction.",
            )

        assert result["subtopics"]["abundance"]["relevant"] is True
        assert result["subtopics"]["abundance"]["reasoning"] == "from the fallback"
        assert result["fallback_model"] == "gemini-3.5-flash"
        # Same taxonomy and the same per-item prompt the primary model was sent.
        prompt = mock_anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
        fallback.assert_called_once_with(prompt, system=_build_cached_system_prompt())

    def test_fallback_is_not_consulted_when_the_model_answers(self, mock_anthropic_client):
        classifier = self._classifier(mock_anthropic_client)

        with patch("digest.classifiers.relevance_classifier.gemini_generate") as fallback:
            result = classifier.classify_item(
                title="Housing Supply and Zoning Reform",
                content="This paper studies the effect of upzoning on housing construction.",
            )

        fallback.assert_not_called()
        assert "fallback_model" not in result

    def test_other_failures_do_not_reach_the_fallback(self, mock_anthropic_client):
        """Only a refusal is a reason to ask another vendor. A timeout or a
        truncated response is transient, and the next run retries it."""
        mock_anthropic_client.messages.create.side_effect = Exception("API error")
        classifier = self._classifier(mock_anthropic_client)

        with patch("digest.classifiers.relevance_classifier.gemini_generate") as fallback:
            result = classifier.classify_item(
                title="Housing Supply and Zoning Reform",
                content="This paper studies the effect of upzoning on housing construction.",
            )

        fallback.assert_not_called()
        assert result is None

    def test_unparseable_fallback_answer_leaves_the_item_unclassified(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = _refusal()
        classifier = self._classifier(mock_anthropic_client)

        with patch(
            "digest.classifiers.relevance_classifier.gemini_generate",
            return_value="I cannot help with that.",
        ):
            result = classifier.classify_item(
                title="Housing Supply and Zoning Reform",
                content="This paper studies the effect of upzoning on housing construction.",
            )

        assert result is None

    def test_refused_batch_sends_only_the_declined_item_to_the_fallback(self, mock_anthropic_client):
        """One declined item refuses the whole batch. Each item is then retried
        alone, so the fallback sees only what the primary model still declines."""
        answered = mock_anthropic_client._make_response(
            '{"abundance": {"relevant": true, "topics": ["housing"], "confidence": "high", "reasoning": "primary"}}'
        )
        mock_anthropic_client.messages.create.side_effect = [_refusal(), answered, _refusal()]
        classifier = self._classifier(mock_anthropic_client)
        items = [
            {"title": "Zoning reform and housing supply", "content": "Upzoning raised housing construction."},
            {"title": "Affordable housing and zoning reform", "content": "Multifamily housing supply grew."},
        ]

        with patch(
            "digest.classifiers.relevance_classifier.gemini_generate",
            return_value=_HOUSING_VERDICT,
        ) as fallback:
            results = classifier._classify_batch_items(items)

        assert mock_anthropic_client.messages.create.call_count == 3
        assert fallback.call_count == 1
        assert results[0]["subtopics"]["abundance"]["reasoning"] == "primary"
        assert "fallback_model" not in results[0]
        assert results[1]["subtopics"]["abundance"]["reasoning"] == "from the fallback"
        assert results[1]["fallback_model"] == "gemini-3.5-flash"


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
