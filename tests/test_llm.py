"""Tests for the shared Anthropic client factory (digest.llm)."""

from unittest.mock import patch

import anthropic
import pytest

from digest.llm import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    get_anthropic_client,
)


class TestGetAnthropicClient:
    def test_default_retry_and_timeout_policy(self):
        with patch("digest.llm.anthropic.Anthropic") as mock_cls:
            get_anthropic_client(api_key="test-key")
            mock_cls.assert_called_once_with(
                timeout=DEFAULT_TIMEOUT_SECONDS,
                max_retries=DEFAULT_MAX_RETRIES,
                api_key="test-key",
            )

    def test_overrides_passed_through(self):
        with patch("digest.llm.anthropic.Anthropic") as mock_cls:
            get_anthropic_client(api_key="test-key", timeout=60.0, max_retries=5)
            mock_cls.assert_called_once_with(
                timeout=60.0, max_retries=5, api_key="test-key"
            )

    def test_api_key_omitted_falls_back_to_env(self):
        # Without an explicit key the factory must not pass api_key=None,
        # so the SDK reads ANTHROPIC_API_KEY itself.
        with patch("digest.llm.anthropic.Anthropic") as mock_cls:
            get_anthropic_client()
            assert "api_key" not in mock_cls.call_args.kwargs

    def test_returns_real_client(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        client = get_anthropic_client()
        assert isinstance(client, anthropic.Anthropic)
        assert client.max_retries == DEFAULT_MAX_RETRIES


class TestClassifierFailLoud:
    """AuthenticationError must propagate out of classification, not degrade
    to everything-not-relevant."""

    def _make_classifier(self):
        from digest.classifiers.relevance_classifier import (
            RelevanceClassifier,
        )

        return RelevanceClassifier(api_key="test-key", use_keyword_prefilter=False)

    def _auth_error(self):
        import httpx

        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(401, request=request)
        return anthropic.AuthenticationError(
            "invalid api key", response=response, body=None
        )

    def test_classify_item_reraises_auth_error(self):
        classifier = self._make_classifier()
        with patch.object(
            classifier.client.messages, "create", side_effect=self._auth_error()
        ):
            with pytest.raises(anthropic.AuthenticationError):
                classifier.classify_item(
                    title="AI safety paper", content="alignment research"
                )

    def test_transient_error_still_degrades_gracefully(self):
        classifier = self._make_classifier()
        with patch.object(
            classifier.client.messages, "create", side_effect=RuntimeError("boom")
        ):
            result = classifier.classify_item(
                title="AI safety paper", content="alignment research"
            )
            assert result["subtopics"]
            assert all(not s["relevant"] for s in result["subtopics"].values())
