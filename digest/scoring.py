"""Keyword scoring and prompt construction over the configured taxonomy.

Everything here reads the active :class:`~digest.settings.DigestConfig`, so the
same functions serve any subject matter. The keyword scores are a cheap
prefilter in front of the LLM classifier, not a classifier in their own right:
they decide which items are worth spending a token on.
"""

import re
from typing import Optional

from digest.settings import get_config

# Number of a topic's keywords reused as its audit-pass keyword set when the
# configuration does not supply one explicitly.
DERIVED_AUDIT_KEYWORDS = 8


def get_subtopic_info(subtopic_key: str) -> dict:
    """Return a subtopic's ``{name, description, team_context}``."""
    subtopics = get_config().subtopics
    if subtopic_key not in subtopics:
        raise ValueError(
            f"Unknown subtopic: {subtopic_key}. Valid: {list(subtopics.keys())}"
        )
    return subtopics[subtopic_key]


def get_subtopic_topics(subtopic_key: str) -> dict:
    """Return the topic areas defined for a subtopic."""
    subtopic_topics = get_config().subtopic_topics
    if subtopic_key not in subtopic_topics:
        raise ValueError(
            f"Unknown subtopic: {subtopic_key}. Valid: {list(subtopic_topics.keys())}"
        )
    return subtopic_topics[subtopic_key]


def get_topic_groups_for_subtopic(subtopic_key: str) -> Optional[dict]:
    """Return a subtopic's display grouping, or None if its topics are flat."""
    return get_config().topic_groups.get(subtopic_key)


def get_all_topics() -> dict:
    """Return every topic across every subtopic, keyed ``subtopic:topic``."""
    all_topics = {}
    for subtopic_key, topics in get_config().subtopic_topics.items():
        for topic_key, topic_data in topics.items():
            all_topics[f"{subtopic_key}:{topic_key}"] = {
                **topic_data,
                "subtopic": subtopic_key,
                "original_key": topic_key,
            }
    return all_topics


def get_topic_prompt(topic_key: str, subtopic_key: str) -> str:
    """Render one topic as a single prompt line."""
    topics = get_subtopic_topics(subtopic_key)
    if topic_key not in topics:
        raise ValueError(f"Unknown topic: {topic_key} in subtopic {subtopic_key}")
    topic = topics[topic_key]
    return f"{topic['name']}: {topic['description']}"


def get_subtopic_topics_prompt(subtopic_key: str) -> str:
    """Render one subtopic's topic areas as a prompt block."""
    subtopic_info = get_subtopic_info(subtopic_key)
    topics = get_subtopic_topics(subtopic_key)
    topics_list = "\n".join(
        f"    - {topic['name']}: {topic['description']}" for topic in topics.values()
    )
    return f"""{subtopic_info["name"]} topic areas:
{topics_list}"""


def get_all_subtopics_prompt() -> str:
    """Render every subtopic and its topics as a prompt block."""
    config = get_config()
    lines = [f"{config.org_name} focus areas:\n"]
    for subtopic_key, subtopic_info in config.subtopics.items():
        lines.append(f"## {subtopic_info['name']}")
        lines.append(f"*{subtopic_info['description']}*\n")
        for topic_key, topic in get_subtopic_topics(subtopic_key).items():
            lines.append(
                f"  - {subtopic_key}:{topic_key} - {topic['name']}: {topic['description']}"
            )
        lines.append("")
    return "\n".join(lines)


def calculate_keyword_score(text: str, topic_key: str, subtopic_key: str) -> float:
    """Fraction of a topic's keywords present in ``text``, scaled by its weight."""
    topics = get_subtopic_topics(subtopic_key)
    if topic_key not in topics:
        return 0.0

    topic = topics[topic_key]
    keywords = topic.get("keywords", [])
    if not keywords or not text:
        return 0.0

    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    base_score = matches / len(keywords)
    return min(base_score * topic.get("weight", 1.0), 1.0)


def calculate_subtopic_score(text: str, subtopic_key: str) -> float:
    """Best keyword score across a subtopic's topics."""
    topics = get_subtopic_topics(subtopic_key)
    if not topics:
        return 0.0
    return max(
        (calculate_keyword_score(text, topic_key, subtopic_key) for topic_key in topics),
        default=0.0,
    )


def get_keyword_candidates(text: str, subtopic_key: str, threshold: float = 0.02) -> list:
    """Topics in a subtopic scoring at or above ``threshold``, best first."""
    candidates = []
    for topic_key in get_subtopic_topics(subtopic_key):
        score = calculate_keyword_score(text, topic_key, subtopic_key)
        if score >= threshold:
            candidates.append((topic_key, score))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def get_all_keyword_candidates(text: str, threshold: float = 0.02) -> dict:
    """Keyword candidates across all subtopics, omitting subtopics with none."""
    results = {}
    for subtopic_key in get_config().subtopic_topics:
        candidates = get_keyword_candidates(text, subtopic_key, threshold)
        if candidates:
            results[subtopic_key] = candidates
    return results


def is_potentially_relevant(
    text: str, subtopic_key: Optional[str] = None, min_threshold: float = 0.02
) -> bool:
    """Whether ``text`` clears the prefilter for a subtopic, or for any subtopic."""
    if subtopic_key:
        return calculate_subtopic_score(text, subtopic_key) >= min_threshold
    return any(
        calculate_subtopic_score(text, sk) >= min_threshold
        for sk in get_config().subtopic_topics
    )


def get_audit_keywords(subtopic_key: str) -> dict:
    """Audit-pass keyword sets for a subtopic, ``{topic_key: [keyword]}``.

    The audit pass rescans candidates for topics that came back empty after
    classification, so its keyword sets should be broader and simpler than the
    prefilter's. Configuration may supply them per subtopic; otherwise they are
    derived from the first :data:`DERIVED_AUDIT_KEYWORDS` topic keywords.
    """
    configured = get_config().audit_keywords.get(subtopic_key)
    if configured:
        return configured
    return {
        topic_key: topic_data.get("keywords", [])[:DERIVED_AUDIT_KEYWORDS]
        for topic_key, topic_data in get_subtopic_topics(subtopic_key).items()
    }


def check_audit_keywords(text: str, category_key: str, subtopic_key: str) -> bool:
    """Whether ``text`` matches any audit keyword for a category."""
    keywords = get_audit_keywords(subtopic_key).get(category_key, [])
    if not keywords:
        return False
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _word_boundary_match(pattern: str, text: str) -> bool:
    """Match short patterns on word boundaries, long ones as substrings.

    Short names ("MIT", "IFP") produce constant false positives as bare
    substrings; long ones are distinctive enough that requiring a boundary only
    costs recall on hyphenation and possessives.
    """
    if len(pattern) < 5:
        return bool(re.search(r"\b" + re.escape(pattern) + r"\b", text))
    return pattern in text


def check_network_connection(text: str, authors: str = "") -> dict:
    """Match an item against the configured network of authors, orgs and publications.

    Returns ``{has_connection, connection_type, connection_name,
    connection_description}``. First match wins, checked authors →
    organizations → publications. Always returns a no-connection result when
    ``DigestConfig.network_connections`` is unset.
    """
    result = {
        "has_connection": False,
        "connection_type": None,
        "connection_name": None,
        "connection_description": None,
    }
    network = get_config().network_connections
    if not network:
        return result

    combined = f"{text} {authors}".lower()
    for kind, key in (
        ("author", "authors"),
        ("organization", "organizations"),
        ("publication", "publications"),
    ):
        for name, description in (network.get(key) or {}).items():
            if _word_boundary_match(name, combined):
                return {
                    "has_connection": True,
                    "connection_type": kind,
                    "connection_name": name.title(),
                    "connection_description": description,
                }
    return result
