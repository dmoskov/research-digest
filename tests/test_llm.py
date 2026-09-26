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
        # httpx2 rather than httpx: the anthropic 1.x SDK moved to it, and the
        # error constructor wants a response object of the SDK's own HTTP
        # library. It arrives as a transitive dependency of anthropic.
        import httpx2

        request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx2.Response(401, request=request)
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

    def test_transient_error_leaves_the_item_unclassified(self):
        """A transient failure must not be recorded as a judgement.

        The auth error above is re-raised because retrying is pointless. A
        transient one returns None so nothing is stored and the next run picks
        the item up again -- previously it wrote `not relevant` for every
        subtopic, which the classifier could never revisit.
        """
        classifier = self._make_classifier()
        with patch.object(
            classifier.client.messages, "create", side_effect=RuntimeError("boom")
        ):
            result = classifier.classify_item(
                title="AI safety paper", content="alignment research"
            )
            assert result is None


class TestConfiguredClientFactory:
    """A deployment without a static key supplies its own client factory.

    Workload identity federation is the motivating case: the workload mints a
    short-lived token per exchange, so there is no secret to put in
    ANTHROPIC_API_KEY and the SDK cannot build the client on its own.
    """

    def _configure_with_factory(self, base, factory):
        """Reinstall the autouse test config with a client factory attached."""
        from dataclasses import replace

        from digest.settings import configure

        configure(replace(base, anthropic_client_factory=factory))

    def test_factory_is_used_and_gets_the_retry_policy(self, monkeypatch, configured):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        calls = []
        sentinel = object()

        def factory(**kwargs):
            calls.append(kwargs)
            return sentinel

        self._configure_with_factory(configured, factory)
        assert get_anthropic_client() is sentinel
        assert calls == [
            {"timeout": DEFAULT_TIMEOUT_SECONDS, "max_retries": DEFAULT_MAX_RETRIES}
        ]

    def test_explicit_api_key_wins_over_factory(self, monkeypatch, configured):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        self._configure_with_factory(
            configured, lambda **kwargs: pytest.fail("factory used")
        )
        with patch("digest.llm.anthropic.Anthropic") as mock_cls:
            get_anthropic_client(api_key="test-key")
            assert mock_cls.call_args.kwargs["api_key"] == "test-key"

    def test_no_factory_configured_builds_client_directly(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("digest.llm.anthropic.Anthropic") as mock_cls:
            get_anthropic_client()
            assert "api_key" not in mock_cls.call_args.kwargs

    def test_classifier_accepts_a_factory_instead_of_a_key(self, monkeypatch, configured):
        from digest.classifiers.relevance_classifier import RelevanceClassifier

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        self._configure_with_factory(
            configured,
            lambda **kwargs: anthropic.Anthropic(
                api_key="minted-by-the-factory", **kwargs
            ),
        )
        classifier = RelevanceClassifier(use_keyword_prefilter=False)
        assert classifier.api_key is None
        assert classifier.client.max_retries == DEFAULT_MAX_RETRIES

    def test_classifier_still_raises_when_nothing_is_configured(self, monkeypatch):
        from digest.classifiers.relevance_classifier import RelevanceClassifier

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Anthropic credentials required"):
            RelevanceClassifier(use_keyword_prefilter=False)


class TestResponseText:
    """The pipeline must read the first *text* block, not content[0]."""

    def _response(self, *blocks):
        from unittest.mock import MagicMock

        return MagicMock(content=list(blocks))

    def _block(self, type_, **kwargs):
        from unittest.mock import MagicMock

        return MagicMock(type=type_, **kwargs)

    def test_reads_the_text_block(self):
        from digest.llm import response_text

        assert response_text(self._response(self._block("text", text="hi"))) == "hi"

    def test_skips_a_leading_thinking_block(self):
        """The case that breaks content[0]: thinking is on by default on the
        newest models, and .text does not exist on a thinking block."""
        from digest.llm import response_text

        resp = self._response(
            self._block("thinking", thinking="deliberating"),
            self._block("text", text="the answer"),
        )
        assert response_text(resp) == "the answer"

    def test_no_text_block_returns_empty(self):
        from digest.llm import response_text

        assert response_text(self._response(self._block("thinking", thinking="x"))) == ""

    def test_empty_content_returns_empty(self):
        from digest.llm import response_text

        assert response_text(self._response()) == ""
