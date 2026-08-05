"""
Relevance classifier using the Claude API.

Classifies research items by relevance to the configured subtopics and their
topic areas.

Two-stage approach:
1. Fast keyword-based pre-filtering to identify candidates
2. LLM-based classification for nuanced relevance determination

Every item is scored against every configured subtopic in one call, with
subtopic-specific reasoning, so a single crawl feeds every team's digest.
"""

import functools
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import anthropic

from digest.llm import get_anthropic_client
from digest.scoring import (
    calculate_subtopic_score,
    check_audit_keywords,
    check_network_connection,
    get_all_keyword_candidates,
    get_audit_keywords,
    get_subtopic_info,
    get_subtopic_topics,
)
from digest.settings import get_config, register_cache_clear
from digest.usage import log_usage

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _build_cached_system_prompt() -> str:
    """Build the static system prompt shared by every classification call.

    Always includes ALL subtopics, sorted, so the bytes are identical across
    calls — Anthropic prompt caching is a prefix match, and any variation
    (e.g. the keyword prefilter narrowing the subtopic set) would defeat it.
    The per-request subtopic subset is communicated in the user message
    instead. Must stay above the model's minimum cacheable prefix
    (1024 tokens on Sonnet 4); the full subtopic context clears that easily.
    """
    config = get_config()
    subtopics_prompt = RelevanceClassifier._format_subtopic_sections(sorted(config.subtopics.keys()))
    return f"""You are classifying research items for {config.org_name}, an organization with multiple focus areas. Each request asks you to evaluate one or more items against a subset of the teams below.

{subtopics_prompt}

General rules:
- Each subtopic is evaluated independently
- Only mark an item relevant if it directly relates to that team's work
- General research is not automatically relevant - it must connect to the specific topics
- If relevance is ambiguous, include it but mark confidence as "uncertain"
- Reasoning should be specific to each team's interests and help that team understand what they would learn from the item
- Only evaluate the subtopics named in the request, and respond only with JSON in the exact format requested"""


# The prompt above memoises the taxonomy, so a reconfigure must drop it or the
# next run would classify against the previous config's subtopics.
register_cache_clear(_build_cached_system_prompt.cache_clear)


def _cached_system_block() -> list:
    """System parameter with a cache_control breakpoint on the static prompt."""
    return [
        {
            "type": "text",
            "text": _build_cached_system_prompt(),
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _log_cache_usage(message, label: str) -> None:
    """Surface cache effectiveness; zero reads across a run means a silent
    cache invalidator regressed."""
    usage = message.usage
    logger.info(
        "%s usage: input=%s cache_write=%s cache_read=%s output=%s",
        label,
        usage.input_tokens,
        getattr(usage, "cache_creation_input_tokens", None),
        getattr(usage, "cache_read_input_tokens", None),
        usage.output_tokens,
    )


class RelevanceClassifier:
    """Classifier for determining relevance to the configured subtopics using Claude."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_keyword_prefilter: bool = True,
        keyword_threshold: float = 0.02,
    ):
        """
        Initialize classifier.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            use_keyword_prefilter: Whether to use keyword-based pre-filtering
            keyword_threshold: Minimum keyword score to pass pre-filter
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")

        self.client = get_anthropic_client(api_key=self.api_key)
        self.use_keyword_prefilter = use_keyword_prefilter
        self.keyword_threshold = keyword_threshold

    def classify_item(
        self,
        title: str,
        content: str,
        item_type: str = "research paper",
        authors: str = "",
        subtopic_filter: Optional[str] = None,
        source: str = "",
    ) -> Dict:
        """
        Classify a single item for relevance to the configured subtopics.

        Args:
            title: Item title
            content: Item abstract/content
            item_type: Type of item (e.g., "research paper", "blog post")
            authors: Author string if available
            subtopic_filter: If set, only classify for this subtopic

        Returns:
            Dict with keys:
                - subtopics: dict of subtopic_key -> {relevant, topics, confidence, reasoning}
                - cg_connection: dict with network connection info
                - keyword_scores: dict of keyword scores (if prefilter enabled)
        """
        combined_text = f"{title} {content}"
        result = {
            "subtopics": {},
            "cg_connection": check_network_connection(combined_text, authors),
            "keyword_scores": {},
        }

        subtopics_to_check = [subtopic_filter] if subtopic_filter else list(get_config().subtopics.keys())

        if self.use_keyword_prefilter:
            subtopics_to_check = self._apply_single_item_prefilter(combined_text, subtopics_to_check, result)
            if subtopics_to_check is None:
                return result

        prompt = self._build_classification_prompt(title, content, item_type, subtopics_to_check, source=source)

        try:
            llm_result = self._call_llm_classify(prompt)
            self._map_single_llm_result(llm_result, subtopics_to_check, result)
            self._fill_unevaluated_subtopics(result)
            return result
        except (anthropic.AuthenticationError, anthropic.BadRequestError):
            raise
        except Exception as e:
            logger.error(f"Error classifying item: {e}")
            for st in subtopics_to_check:
                result["subtopics"][st] = self._not_relevant_entry(f"Classification error: {str(e)}")
            return result

    def _apply_single_item_prefilter(
        self,
        combined_text: str,
        subtopics_to_check: List[str],
        result: Dict,
    ) -> Optional[List[str]]:
        """Apply keyword pre-filter for a single item.

        Populates result["keyword_scores"]. Returns the narrowed subtopic
        list, or None when every subtopic was rejected (result["subtopics"]
        is filled with 'not relevant' entries in that case).
        """
        all_candidates = get_all_keyword_candidates(combined_text, self.keyword_threshold)
        result["keyword_scores"] = all_candidates

        relevant_subtopics = [
            st
            for st in subtopics_to_check
            if st in all_candidates or calculate_subtopic_score(combined_text, st) >= self.keyword_threshold
        ]

        if not relevant_subtopics:
            for st in subtopics_to_check:
                result["subtopics"][st] = self._not_relevant_entry("No keyword matches found - likely not relevant")
            return None

        return relevant_subtopics

    def _call_llm_classify(self, prompt: str, max_tokens: int = 2048) -> Dict:
        """Make a classification API call and return the parsed JSON result."""
        message = self.client.messages.create(
            model=get_config().claude_model_fast,
            max_tokens=max_tokens,
            system=_cached_system_block(),
            messages=[{"role": "user", "content": prompt}],
        )
        log_usage(message, "digest_relevance")
        _log_cache_usage(message, "digest_relevance")

        if not message.content:
            raise ValueError(f"Empty API response (stop={message.stop_reason})")

        response_text = self._extract_json_from_response(message.content[0].text)
        return json.loads(response_text)

    def _map_single_llm_result(
        self,
        llm_result: Dict,
        subtopics_to_check: List[str],
        result: Dict,
    ) -> None:
        """Map LLM classification JSON to subtopic result entries."""
        for subtopic_key in subtopics_to_check:
            if subtopic_key in llm_result:
                result["subtopics"][subtopic_key] = self._parse_subtopic_result(llm_result[subtopic_key])
            else:
                result["subtopics"][subtopic_key] = self._not_relevant_entry("Not evaluated")

    @staticmethod
    def _build_single_json_template(subtopics: List[str]) -> str:
        """Build a JSON structure template for single-item classification."""
        json_structure = {}
        for st_key in subtopics:
            json_structure[st_key] = {
                "relevant": "true/false",
                "topics": ["topic_key1", "topic_key2"],
                "confidence": "high/medium/low/uncertain",
                "reasoning": f"Why this is/isn't relevant to {st_key} team's work",
            }
        return json.dumps(json_structure, indent=2)

    def _build_classification_prompt(
        self,
        title: str,
        content: str,
        item_type: str,
        subtopics: List[str],
        source: str = "",
    ) -> str:
        """Build the volatile (per-item) part of the classification prompt.

        Subtopic descriptions and general rules live in the cached system
        prompt (_build_cached_system_prompt); only item content and the
        requested subtopic subset go here.
        """
        json_template = self._build_single_json_template(subtopics)

        return f"""Evaluate this {item_type} against these teams only: {", ".join(subtopics)}

{item_type.capitalize()} to evaluate:

Source: {source}
Title: {title}

Content: {content[:3000]}

Instructions:
1. For EACH subtopic listed above, determine if this {item_type} is relevant to that team's work
2. If relevant, identify which specific topic area(s) it relates to (use the topic keys like "housing", "alignment", etc.)
3. Provide team-specific reasoning explaining WHY it's relevant to that team

Respond in JSON format:
```json
{json_template}
```"""

    @staticmethod
    def _format_subtopic_sections(subtopics: List[str]) -> str:
        """Format subtopic descriptions into prompt sections."""
        sections = []
        for st_key in subtopics:
            st_info = get_subtopic_info(st_key)
            topics = get_subtopic_topics(st_key)
            topics_list = "\n".join([f"      - {t['name']}: {t['description']}" for t in topics.values()])
            sections.append(f"""
### {st_info["name"]} ({st_key})
{st_info["description"]}

Team context: {st_info["team_context"]}

Topic areas:
{topics_list}
""")
        return "\n".join(sections)

    @staticmethod
    def _format_batch_items(items_with_meta: List[Dict]) -> str:
        """Format items into a numbered text block for batch prompts."""
        items_text = []
        for idx, meta in enumerate(items_with_meta):
            items_text.append(f"""--- ITEM {idx} ---
Source: {meta.get("source_name") or meta.get("source", "")}
Title: {meta["title"]}
Type: {meta["item_type"]}
Content: {meta["content"][:2000]}
""")
        return "\n".join(items_text)

    @staticmethod
    def _build_subtopic_json_template(subtopics: List[str]) -> str:
        """Build a per-item JSON structure template for batch classification."""
        item_structure = {}
        for st_key in subtopics:
            item_structure[st_key] = {
                "relevant": "true/false",
                "topics": ["topic_key1"],
                "confidence": "high/medium/low/uncertain",
                "reasoning": "brief reasoning",
            }
        return json.dumps(item_structure, indent=2)

    def _build_batch_classification_prompt(
        self,
        items_with_meta: List[Dict],
        subtopics: List[str],
    ) -> str:
        """Build the volatile (per-batch) part of the classification prompt.

        Subtopic descriptions and general rules live in the cached system
        prompt (_build_cached_system_prompt); only item content and the
        requested subtopic subset go here.
        """
        items_block = self._format_batch_items(items_with_meta)
        per_item = self._build_subtopic_json_template(subtopics)

        return f"""Evaluate EACH item below against these teams only: {", ".join(subtopics)}

{items_block}

Instructions:
1. For EACH item (numbered 0 to {len(items_with_meta) - 1}), evaluate relevance to each subtopic listed above
2. Respond in JSON format as an object keyed by item index (as string):

```json
{{
  "0": {per_item},
  "1": {per_item}
}}
```"""

    @staticmethod
    def _infer_item_type(source: str) -> str:
        """Infer item type from source identifier."""
        if source == "nber":
            return "research paper"
        if source == "substack":
            return "blog post"
        return "article"

    @staticmethod
    def _extract_json_from_response(response_text: str) -> str:
        """Extract JSON string from an LLM response, stripping markdown code fences."""
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            return response_text[json_start:json_end].strip()
        if "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            return response_text[json_start:json_end].strip()
        return response_text

    @staticmethod
    def _not_relevant_entry(reasoning: str) -> Dict:
        """Create a standard 'not relevant' subtopic classification entry."""
        return {
            "relevant": False,
            "topics": [],
            "confidence": "low",
            "reasoning": reasoning,
        }

    @staticmethod
    def _fill_unevaluated_subtopics(result: Dict) -> None:
        """Fill in any unevaluated subtopics with 'not evaluated' entries."""
        for st in get_config().subtopics:
            if st not in result["subtopics"]:
                result["subtopics"][st] = {
                    "relevant": False,
                    "topics": [],
                    "confidence": "low",
                    "reasoning": "Not evaluated (no keyword matches)",
                }

    def _prefilter_items(
        self,
        items: List[Dict],
        subtopics_to_check: List[str],
    ) -> Tuple[List[Tuple], List[Optional[Dict]]]:
        """Run keyword pre-filter on each item.

        Returns:
            (items_for_api, results) where items_for_api contains
            (original_index, item_meta, result_dict) tuples and results
            has None for items that still need API classification.
        """
        items_for_api: List[Tuple] = []
        results: List[Optional[Dict]] = [None] * len(items)

        for i, item in enumerate(items):
            title = item.get("title", "")
            content = item.get("content", "") or item.get("abstract", "")
            authors = item.get("authors", "") or item.get("author", "")
            source_name = item.get("source_name", "") or item.get("source", "")
            combined_text = f"{source_name} {title} {content}"

            result: Dict = {
                "subtopics": {},
                "cg_connection": check_network_connection(combined_text, authors),
                "keyword_scores": {},
            }

            if self.use_keyword_prefilter:
                all_candidates = get_all_keyword_candidates(combined_text, self.keyword_threshold)
                result["keyword_scores"] = all_candidates

                relevant_subtopics = [
                    st
                    for st in subtopics_to_check
                    if st in all_candidates or calculate_subtopic_score(combined_text, st) >= self.keyword_threshold
                ]

                if not relevant_subtopics:
                    for st in subtopics_to_check:
                        result["subtopics"][st] = self._not_relevant_entry(
                            "No keyword matches found - likely not relevant"
                        )
                    self._fill_unevaluated_subtopics(result)
                    results[i] = result
                    continue

                subtopics_for_item = relevant_subtopics
            else:
                subtopics_for_item = subtopics_to_check

            item_type = self._infer_item_type(item.get("source", ""))
            items_for_api.append(
                (
                    i,
                    {
                        "title": title,
                        "content": content,
                        "item_type": item_type,
                        "source_name": item.get("source_name", "") or item.get("source", ""),
                        "subtopics": subtopics_for_item,
                    },
                    result,
                )
            )

        return items_for_api, results

    @staticmethod
    def _parse_subtopic_result(st_data: Dict) -> Dict:
        """Parse a single subtopic classification from LLM output."""
        raw_rel = st_data.get("relevant", False)
        is_relevant = raw_rel if isinstance(raw_rel, bool) else str(raw_rel).lower() == "true"
        return {
            "relevant": is_relevant,
            "topics": st_data.get("topics", []),
            "confidence": st_data.get("confidence", "low"),
            "reasoning": st_data.get("reasoning", ""),
        }

    def _apply_batch_llm_results(
        self,
        llm_results: Dict,
        items_for_api: List[Tuple],
        results: List[Optional[Dict]],
    ) -> None:
        """Map parsed LLM batch results back to individual item results."""
        for batch_idx, (orig_idx, meta, result) in enumerate(items_for_api):
            item_result = llm_results.get(str(batch_idx), {})
            for st_key in meta["subtopics"]:
                if st_key in item_result:
                    result["subtopics"][st_key] = self._parse_subtopic_result(item_result[st_key])
                else:
                    result["subtopics"][st_key] = self._not_relevant_entry("Not evaluated")
            self._fill_unevaluated_subtopics(result)
            results[orig_idx] = result

    def _classify_batch_items(
        self,
        items: List[Dict],
        subtopic_filter: Optional[str] = None,
    ) -> List[Dict]:
        """Classify a batch of items in a single API call.

        Runs keyword pre-filter individually, then groups passing items
        into a single Claude API call (max 5 per call). Falls back to
        individual classification if the batch call fails.
        """
        subtopics_to_check = [subtopic_filter] if subtopic_filter else list(get_config().subtopics.keys())

        items_for_api, results = self._prefilter_items(items, subtopics_to_check)

        if not items_for_api:
            return results

        all_needed_subtopics = sorted({st for _, meta, _ in items_for_api for st in meta["subtopics"]})
        prompt = self._build_batch_classification_prompt(
            [meta for _, meta, _ in items_for_api],
            all_needed_subtopics,
        )

        try:
            message = self.client.messages.create(
                model=get_config().claude_model_fast,
                max_tokens=4096,
                system=_cached_system_block(),
                messages=[{"role": "user", "content": prompt}],
            )
            log_usage(message, "digest_relevance_batch")
            _log_cache_usage(message, "digest_relevance_batch")
            if not message.content:
                raise ValueError(f"Empty API response (stop={message.stop_reason})")

            response_text = self._extract_json_from_response(message.content[0].text)
            llm_results = json.loads(response_text)
            self._apply_batch_llm_results(llm_results, items_for_api, results)

        except (anthropic.AuthenticationError, anthropic.BadRequestError):
            # Non-transient API errors: the individual fallback would fail
            # identically, burning one doomed API call per item.
            raise
        except Exception as e:
            logger.warning(f"Batch classification failed, falling back to individual: {e}")
            for orig_idx, meta, result in items_for_api:
                if results[orig_idx] is not None:
                    continue
                results[orig_idx] = self._classify_item_fallback(items[orig_idx], subtopic_filter=subtopic_filter)

        return results

    def _classify_item_fallback(
        self,
        item: Dict,
        subtopic_filter: Optional[str] = None,
    ) -> Dict:
        """Classify a single item individually as fallback."""
        title = item.get("title", "")
        content = item.get("content", "") or item.get("abstract", "")
        authors = item.get("authors", "") or item.get("author", "")
        item_type = self._infer_item_type(item.get("source", ""))
        return self.classify_item(
            title=title,
            content=content,
            item_type=item_type,
            authors=authors,
            subtopic_filter=subtopic_filter,
        )

    def classify_batch(
        self,
        items: List[Dict],
        batch_size: int = 5,
        subtopic_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Classify multiple items for relevance.

        Items are grouped into batches and classified with a single API call
        per batch. Falls back to individual classification on batch failure.

        Args:
            items: List of items with 'title' and 'content' keys
            batch_size: Number of items per batch API call
            subtopic_filter: If set, only classify for this subtopic

        Returns:
            List of items with added 'classification' key
        """
        classified_items = []
        total_batches = (len(items) + batch_size - 1) // batch_size

        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            batch_results = self._classify_batch_items(chunk, subtopic_filter=subtopic_filter)
            batch_num = start // batch_size + 1
            logger.info(f"Batch {batch_num}/{total_batches} ({start + len(chunk)}/{len(items)} items)")

            for item, classification in zip(chunk, batch_results):
                if classification is None:
                    classification = self._classify_item_fallback(item, subtopic_filter=subtopic_filter)
                item["classification"] = classification
                classified_items.append(item)

        return classified_items

    def _find_populated_categories(self, classified_items: List[Dict], subtopic: str) -> set:
        """Return set of topic category keys that already have relevant items."""
        populated = set()
        for item in classified_items:
            st_cls = item.get("classification", {}).get("subtopics", {}).get(subtopic, {})
            if st_cls.get("relevant", False):
                for t in st_cls.get("topics", []):
                    populated.add(t)
        return populated

    def _try_audit_match(self, item: Dict, empty_categories: List[str], subtopic: str) -> Optional[str]:
        """Check if item matches any empty category's audit keywords.

        If a match is found, promotes the item as an uncertain match and
        returns the matched category key. Returns None if no match.
        """
        title = item.get("title", "")
        content = item.get("content", "") or item.get("abstract", "")
        combined = f"{title} {content}"

        for cat_key in empty_categories:
            if check_audit_keywords(combined, cat_key, subtopic):
                item.setdefault("classification", {}).setdefault("subtopics", {})[subtopic] = {
                    "relevant": True,
                    "topics": [cat_key],
                    "confidence": "uncertain",
                    "reasoning": "Added by category audit pass — topic unclear from abstract/snippet",
                }
                return cat_key
        return None

    def category_audit_pass(
        self,
        classified_items: List[Dict],
        subtopic: str = "abundance",
    ) -> List[Dict]:
        """Audit pass to catch items for empty categories.

        After classification, checks which topic categories have 0 items.
        For empty categories, scans ALL classified items with simplified
        audit keyword sets. Matches are added with confidence "uncertain"
        and marked [Relevance uncertain].

        Works for any subtopic. Returns the modified items list.
        """
        topics = get_subtopic_topics(subtopic)
        audit_kws = get_audit_keywords(subtopic)
        populated = self._find_populated_categories(classified_items, subtopic)

        empty_categories = [k for k in topics if k not in populated and k in audit_kws]
        if not empty_categories:
            return classified_items

        logger.info(f"  Audit pass: {len(empty_categories)} empty categories: {empty_categories}")

        added = 0
        for item in classified_items:
            st_cls = item.get("classification", {}).get("subtopics", {}).get(subtopic, {})
            if st_cls.get("relevant", False):
                continue

            matched = self._try_audit_match(item, empty_categories, subtopic)
            if matched:
                added += 1
                empty_categories = [c for c in empty_categories if c != matched]

            if not empty_categories:
                break

        if added:
            logger.info(f"  Audit pass: added {added} uncertain items")

        return classified_items

    def filter_relevant(
        self,
        items: List[Dict],
        subtopic: str = "abundance",
    ) -> List[Dict]:
        """
        Filter items to only those marked as relevant for a subtopic.

        Args:
            items: List of items with 'classification' key
            subtopic: Which subtopic to filter for

        Returns:
            Filtered list of relevant items
        """
        return [
            item
            for item in items
            if item.get("classification", {}).get("subtopics", {}).get(subtopic, {}).get("relevant", False)
        ]


def main():
    """Test the relevance classifier."""
    try:
        classifier = RelevanceClassifier()
    except ValueError as e:
        logger.error(f"Error: {e}")
        print("Set ANTHROPIC_API_KEY environment variable to test the classifier.")
        return

    # Example item
    test_item = {
        "title": "The Effect of Land Use Regulation on Housing Supply",
        "abstract": "This paper examines how zoning restrictions affect housing construction in major US cities. We find that restrictive zoning reduces housing supply by 30%.",
        "source": "nber",
    }

    print("Testing classification...")
    result = classifier.classify_item(
        title=test_item["title"],
        content=test_item["abstract"],
    )

    print("\nClassification result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
