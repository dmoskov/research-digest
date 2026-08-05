"""Runtime configuration for the digest pipeline.

The pipeline itself is topic-agnostic: what it crawls, what taxonomy it
classifies against, and whose voice the classifier prompt speaks in are all
supplied by the host application through a :class:`DigestConfig`.

    from digest.settings import DigestConfig, configure

    configure(DigestConfig(
        org_name="Acme Foundation",
        subtopics={"climate": {...}},
        subtopic_topics={"climate": {...}},
    ))

Call :func:`configure` once, before importing anything that classifies or
stores. Reading configuration before it is set raises — there is deliberately
no default taxonomy, because a silently-empty one would classify every item as
irrelevant and look like a working pipeline producing an empty digest.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

# Model defaults. Override per-deployment via DigestConfig; these are the
# combination the pipeline is tuned and cost-modelled for.
DEFAULT_CLAUDE_MODEL = "claude-opus-4-6"
DEFAULT_CLAUDE_MODEL_FAST = "claude-opus-4-6"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


@dataclass(frozen=True)
class DigestConfig:
    """Everything the pipeline needs to know about *your* subject matter.

    Args:
        org_name: Named in the classifier's system prompt, e.g. "Acme
            Foundation". The model is told it is classifying research for this
            organisation, so use the name your teams would recognise.
        subtopics: Top-level focus areas, one per team or feed.
            ``{key: {"name": str, "description": str, "team_context": str}}``
            ``team_context`` is the single biggest lever on classification
            precision — it is injected verbatim into the prompt. State scope
            *and* explicit out-of-scope examples; see README.
        subtopic_topics: Topic areas within each subtopic.
            ``{subtopic_key: {topic_key: {"name", "description", "keywords",
            "weight"}}}``. ``keywords`` drives the cheap keyword prefilter,
            ``weight`` (default 1.0) scales its score.
        topic_groups: Optional display grouping of topics within a subtopic.
        audit_keywords: Optional per-subtopic simplified keyword sets used by
            the category audit pass (which rescans candidates for topics that
            came back empty). Derived from the first 8 topic keywords when a
            subtopic is absent.
        network_connections: Optional "is this author/org already in our
            network" lookup, as ``{"authors": {name: note}, "organizations":
            {...}, "publications": {...}}``. Omit to disable the feature.
        source_auto_tags: Optional ``{source_key: [(subtopic, topic_key)]}``
            rules that force a topic tag onto items from a given source.
        secondary_topics: Topic keys that are catch-alls rather than substantive
            subjects (e.g. a "movement updates" or "general" bucket). A
            ``source_auto_tags`` rule only fires when the classifier found no
            topic outside this set, so an org's newsletter gets its catch-all
            tag while its policy analysis still sorts under the real topic.
        static_sources: Legacy in-code source registry, kept for the two
            built-in crawlers that predate the ``sources`` table
            (``{"nber": {...}, "substack": {"feeds": [url, ...]}}``, each with
            an ``enabled`` flag). Leave empty and seed the ``sources`` table
            instead — that is the supported path, and the only one that gives
            you per-source crawl status and failure tracking.
        site_base_url: Absolute base URL of the site that renders the digests
            (e.g. ``https://digests.example.org``); used to build item links in
            generated summaries.
        bot_name: Crawler identity sent to every site fetched, as
            ``{bot_name}/1.0``. **Set this.** The default is deliberately
            generic, and leaving it means your traffic is indistinguishable from
            every other deployment's — rate-limit reputation and abuse
            complaints follow the User-Agent.
        bot_info_url: Optional page describing your crawler, appended to the
            User-Agent as ``(+url)``. Publishers check it before blocking.
        bot_contact: Optional email. Worth setting: CrossRef and OpenAlex route
            callers who supply a mailto into their faster "polite pool", and
            publishers email before blocking rather than after.
        claude_model / claude_model_fast: Model IDs for summarisation and for
            classification respectively. Classification runs on every item, so
            ``claude_model_fast`` dominates cost.
        gemini_model: Optional non-Anthropic fallback for abstracts Claude
            declines to summarise. Requires the ``gemini`` extra.
    """

    org_name: str
    subtopics: Dict[str, dict]
    subtopic_topics: Dict[str, dict]
    topic_groups: Dict[str, dict] = field(default_factory=dict)
    audit_keywords: Dict[str, dict] = field(default_factory=dict)
    network_connections: Optional[Dict[str, dict]] = None
    source_auto_tags: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    secondary_topics: FrozenSet[str] = frozenset()
    static_sources: Dict[str, dict] = field(default_factory=dict)
    site_base_url: str = ""
    bot_name: str = "ResearchDigestBot"
    bot_info_url: str = ""
    bot_contact: str = ""
    claude_model: str = DEFAULT_CLAUDE_MODEL
    claude_model_fast: str = DEFAULT_CLAUDE_MODEL_FAST
    gemini_model: Optional[str] = DEFAULT_GEMINI_MODEL

    def __post_init__(self):
        if not self.org_name:
            raise ValueError("DigestConfig.org_name is required")
        if not self.subtopics:
            raise ValueError(
                "DigestConfig.subtopics is empty — the pipeline would classify "
                "every item as irrelevant and produce an empty digest"
            )
        missing = [k for k in self.subtopics if k not in self.subtopic_topics]
        if missing:
            raise ValueError(
                f"subtopics without any topic areas in subtopic_topics: {missing}. "
                "Every subtopic needs at least one topic area to classify into."
            )
        orphans = [k for k in self.subtopic_topics if k not in self.subtopics]
        if orphans:
            raise ValueError(
                f"subtopic_topics keys with no matching subtopic: {orphans}"
            )
        for st_key, info in self.subtopics.items():
            for required in ("name", "description", "team_context"):
                if not info.get(required):
                    raise ValueError(
                        f"subtopic '{st_key}' is missing '{required}'"
                    )


_config: Optional[DigestConfig] = None
_cache_clearers: List[Callable[[], None]] = []


def configure(config: DigestConfig) -> None:
    """Install the active configuration and invalidate config-derived caches."""
    global _config
    _config = config
    for clear in _cache_clearers:
        clear()


def get_config() -> DigestConfig:
    """Return the active configuration, raising if none was installed."""
    if _config is None:
        raise RuntimeError(
            "digest is not configured — call digest.settings.configure(DigestConfig(...)) "
            "before using the pipeline, or set DIGEST_CONFIG=module.path:ATTRIBUTE when "
            "using the research-digest CLI."
        )
    return _config


def is_configured() -> bool:
    """Whether a configuration has been installed."""
    return _config is not None


def register_cache_clear(clear: Callable[[], None]) -> None:
    """Register a callback that must fire when the configuration changes.

    Prompt builders memoise on config contents (Anthropic prompt caching is a
    byte-prefix match, so the built prompt must be stable within a run). Any
    such cache must be dropped on reconfigure or a second config would silently
    reuse the first one's prompt.
    """
    _cache_clearers.append(clear)
