"""Shared fixtures for the digest pipeline tests.

The suite runs against a small self-contained taxonomy defined here, not
against any real deployment's configuration — so it stays meaningful for
whoever installs the package. Tests that need to assert on a *particular*
organisation's taxonomy belong in that organisation's own test suite.
"""

from unittest.mock import MagicMock

import pytest

from digest.settings import DigestConfig, configure

# A two-subtopic taxonomy exercising everything the pipeline reads: keyword
# prefiltering, catch-all topics, source auto-tags and network connections.
TEST_CONFIG = DigestConfig(
    org_name="Test Foundation",
    subtopics={
        "abundance": {
            "name": "Abundance",
            "description": "Growth, housing and energy supply",
            "team_context": "We fund work on removing supply-side constraints to growth.",
        },
        "ai_safety": {
            "name": "AI Safety",
            "description": "Alignment and interpretability",
            "team_context": "We fund technical work on making advanced AI systems safe.",
        },
    },
    subtopic_topics={
        "abundance": {
            "housing": {
                "name": "Housing",
                "description": "Housing supply and land use",
                "keywords": [
                    "zoning reform",
                    "housing supply",
                    "affordable housing",
                    "multifamily housing",
                    "housing construction",
                ],
            },
            "energy": {
                "name": "Energy",
                "description": "Energy abundance and permitting",
                "keywords": [
                    "nuclear energy",
                    "permitting reform",
                    "power plants",
                    "energy policy",
                    "clean energy",
                ],
            },
            "general_abundance": {
                "name": "General Abundance",
                "description": "Cross-cutting abundance work",
                "keywords": ["abundance agenda", "supply-side"],
            },
        },
        "ai_safety": {
            "interpretability": {
                "name": "Interpretability",
                "description": "Understanding model internals",
                "keywords": [
                    "mechanistic interpretability",
                    "interpretability",
                    "large language models",
                    "probing classifier",
                    "sparse autoencoder",
                ],
            },
            "alignment": {
                "name": "Alignment",
                "description": "Aligning models with human intent",
                "keywords": ["ai alignment", "alignment", "reward model", "rlhf"],
            },
        },
    },
    # Invented names — a fixture should never carry real people's affiliations.
    network_connections={
        "authors": {"jane quimby": "Wrote our 2024 grid report"},
        "organizations": {"institute for widgets": "Policy think tank"},
        "publications": {"the widget review": "Newsletter"},
    },
    source_auto_tags={"movement_blog": [("abundance", "general_abundance")]},
    secondary_topics=frozenset({"general_abundance"}),
    site_base_url="https://digests.example.org",
)


@pytest.fixture(autouse=True)
def configured():
    """Install the test configuration for every test.

    Autouse because the pipeline raises rather than guessing when unconfigured,
    and re-applied per test so one test mutating config cannot leak into another.
    """
    configure(TEST_CONFIG)
    return TEST_CONFIG


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client that returns a configurable JSON response."""
    client = MagicMock()

    def make_response(text):
        msg = MagicMock()
        msg.content = [MagicMock(text=text)]
        return msg

    client._make_response = make_response
    client.messages.create.return_value = make_response('{"abundance": {"relevant": true, "topics": ["housing"], "confidence": "high", "reasoning": "test"}}')  # noqa: E501
    return client


@pytest.fixture
def sample_item():
    """A minimal crawled item dict."""
    return {
        "title": "The Effect of Land Use Regulation on Housing Supply",
        "abstract": "This paper examines how zoning restrictions affect housing construction.",
        "authors": "John Smith, Jane Doe",
        "source": "nber",
        "url": "https://example.com/paper1",
        "date": "2025-01-15",
    }


@pytest.fixture
def sample_classification():
    """A sample classification result dict."""
    return {
        "subtopics": {
            "abundance": {
                "relevant": True,
                "topics": ["housing"],
                "confidence": "high",
                "reasoning": "Directly about housing supply and zoning",
            },
            "ai_safety": {
                "relevant": False,
                "topics": [],
                "confidence": "low",
                "reasoning": "Not related to AI safety",
            },
        },
        # Legacy key name for network-connection matches; see schema/001_initial.sql.
        "cg_connection": {
            "has_connection": False,
            "connection_type": None,
            "connection_name": None,
            "connection_description": None,
        },
        "keyword_scores": {},
    }
