"""Every Claude request the pipeline builds must be one the installed SDK accepts.

The call sites are otherwise tested against MagicMock clients, which accept any
keyword argument. That is how ``temperature=0.3`` survived in the topic
summariser after the anthropic 1.x SDK dropped the parameter: the real client
raised TypeError on every call, the summariser's except clause turned that into
its fallback line, and no test noticed.
"""

import inspect
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
from anthropic.resources.messages import Messages

from digest.settings import configure

ITEM = {
    "id": 1,
    "title": "Housing Supply and Zoning Reform",
    "content": "This paper studies the effect of upzoning on housing construction. " * 3,
    "abstract": "Upzoning raised housing construction.",
    "published_date": "2026-01-15",
    "source": "nber",
}


def _client():
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(type="text", text="- First finding stated plainly.\n- Second finding stated plainly.")]
    msg.stop_reason = "end_turn"
    client.messages.create.return_value = msg
    return client


def _classify_single(client):
    from digest.classifiers.relevance_classifier import RelevanceClassifier

    with patch("digest.classifiers.relevance_classifier.get_anthropic_client", return_value=client):
        RelevanceClassifier(api_key="test-key", use_keyword_prefilter=False).classify_item(
            title=ITEM["title"], content=ITEM["content"]
        )


def _classify_batch(client):
    from digest.classifiers.relevance_classifier import RelevanceClassifier

    with patch("digest.classifiers.relevance_classifier.get_anthropic_client", return_value=client):
        RelevanceClassifier(api_key="test-key", use_keyword_prefilter=False)._classify_batch_items([dict(ITEM)])


def _abstract(client):
    from digest.summarizer import generate_abstract

    generate_abstract(ITEM["title"], ITEM["content"], "nber", client, delay=0)


def _topic_summary(client):
    from digest.topic_summarizer import generate_topic_summary

    with patch("digest.topic_summarizer.get_anthropic_client", return_value=client):
        generate_topic_summary("housing", "Housing", [dict(ITEM)], "abundance", "abundance")


CALL_SITES = [_classify_single, _classify_batch, _abstract, _topic_summary]


@pytest.mark.parametrize("call_site", CALL_SITES, ids=lambda f: f.__name__.lstrip("_"))
@pytest.mark.parametrize("effort", [None, "low"])
def test_request_is_accepted_by_the_installed_sdk(call_site, effort, configured):
    configure(replace(configured, claude_effort=effort))
    client = _client()

    call_site(client)

    assert client.messages.create.called, "the call site never reached the API"
    signature = inspect.signature(Messages.create)
    for call in client.messages.create.call_args_list:
        # Raises TypeError for any keyword the real client would reject.
        signature.bind(None, *call.args, **call.kwargs)
        if effort:
            assert call.kwargs["output_config"] == {"effort": effort}
        else:
            assert "output_config" not in call.kwargs


def test_a_declined_topic_summary_is_empty_so_it_is_not_cached():
    from digest.topic_summarizer import generate_topic_summary

    client = MagicMock()
    refusal = MagicMock()
    refusal.content = []
    refusal.stop_reason = "refusal"
    refusal.stop_details = MagicMock(category="frontier_llm")
    client.messages.create.return_value = refusal

    with patch("digest.topic_summarizer.get_anthropic_client", return_value=client):
        summary = generate_topic_summary("housing", "Housing", [dict(ITEM)], "abundance", "abundance")

    assert summary == ""
