"""
LLM-based abstract generation for items missing summaries.

Produces 2-3 bullet points suitable for the feed card previews.
The feed template uses `extract_bullets(2)` which splits on newlines
and strips leading `- ` prefixes, so we generate that format directly.

Fallback chain on Claude refusal:
  1. Claude Sonnet (primary)
  2. Extractive summary (immediate, free — pulls first sentences from content)
  3. Gemini Flash retry (async second pass — different content policy)

Used by:
- digest.pipeline               (inline during crawl runs)
"""

import logging
import os
import re
import time

import anthropic

from digest.settings import get_config
from digest.usage import log_usage

logger = logging.getLogger(__name__)

# Truncate input content to stay well within context limits
_MAX_CONTENT_CHARS = 3000

_SYSTEM_PROMPT = (
    "You are a research digest assistant. Given an article's title, source, "
    "and content, produce exactly 2-3 concise bullet points summarizing the "
    "key findings or arguments. Each bullet should be one sentence, max 90 "
    "characters. Lead with substance — never start with 'This article' or "
    "'The author'. Output each bullet on its own line prefixed with '- '."
)

_USER_TEMPLATE = """Title: {title}
Source: {source}

Content:
{content}"""

# Patterns for detecting boilerplate / non-content sentences
_BOILERPLATE_RE = re.compile(
    r"^(subscribe|sign up|share this|click here|read more|follow us|"
    r"newsletter|copyright|all rights|terms of|privacy policy|"
    r"posted by|written by|photo by|image credit|advertisement|"
    r"this (article|post|piece) (was|is)|related (articles|posts)|"
    r"table of contents|skip to)",
    re.IGNORECASE,
)

# Sentence boundary: period/exclamation/question followed by space or newline
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"
)


def needs_abstract(abstract: str | None) -> bool:
    """Return True if the abstract is missing or too short to be useful."""
    if not abstract:
        return True
    return len(abstract.strip()) < 50


def generate_extractive_abstract(content: str, max_bullets: int = 3) -> str | None:
    """Extract first 2-3 meaningful sentences from content as bullet points.

    This is a zero-cost fallback that produces a reasonable summary from
    well-written articles without any LLM call.

    Returns bullet-formatted string or None if content is unsuitable.
    """
    if not content or len(content.strip()) < 100:
        return None

    # Clean up content: collapse whitespace, strip HTML tags
    text = re.sub(r"<[^>]+>", " ", content)
    text = re.sub(r"\s+", " ", text).strip()

    # Split into sentences
    sentences = _SENTENCE_SPLIT_RE.split(text)

    bullets = []
    for sentence in sentences:
        s = sentence.strip()
        # Skip short / boilerplate sentences
        if len(s) < 30 or len(s) > 200:
            continue
        if _BOILERPLATE_RE.search(s):
            continue
        # Ensure sentence ends with punctuation
        if not s[-1] in ".!?":
            s = s.rstrip(",;:") + "."
        # Truncate to ~90 chars at word boundary for bullet display
        if len(s) > 95:
            truncated = s[:90].rsplit(" ", 1)[0]
            s = truncated.rstrip(".,;:") + "..."
        bullets.append(s)
        if len(bullets) >= max_bullets:
            break

    if len(bullets) < 2:
        return None

    return "\n".join(f"- {b}" for b in bullets)


def generate_abstract(
    title: str,
    content: str,
    source: str,
    client: anthropic.Anthropic,
    delay: float = 0.1,
) -> tuple[str | None, bool]:
    """Call Claude to produce a 2-3 bullet summary for an item.

    Returns:
        Tuple of (abstract_text, was_refused).
        - (text, False) on success
        - (None, True) on refusal
        - (None, False) on other failures
    """
    if not content or len(content.strip()) < 50:
        return None, False

    truncated = content[:_MAX_CONTENT_CHARS]
    user_msg = _USER_TEMPLATE.format(
        title=title or "Untitled",
        source=source or "Unknown",
        content=truncated,
    )

    try:
        response = client.messages.create(
            model=get_config().claude_model_fast,
            max_tokens=300,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        log_usage(response, "digest_abstract")
        if not response.content:
            logger.warning("Empty API response for '%s' (stop=%s)", title, response.stop_reason)
            return None, False
        text = response.content[0].text.strip()
    except Exception as exc:
        logger.warning("Abstract generation failed for '%s': %s", title, exc)
        return None, False

    if delay > 0:
        time.sleep(delay)

    # Validate output
    if len(text) < 20:
        logger.warning("Abstract too short for '%s': %r", title, text)
        return None, False
    lower = text.lower()
    if lower.startswith("i cannot") or lower.startswith("i'm sorry"):
        logger.warning("Model refused for '%s'", title)
        return None, True

    return text, False


def generate_abstract_gemini(
    title: str,
    content: str,
    source: str,
    delay: float = 0.1,
) -> str | None:
    """Call Gemini Flash to produce a 2-3 bullet summary.

    Used as a fallback when Claude refuses due to content policy.
    Gemini has a different content policy and is less likely to refuse
    on policy analysis / AI safety content.

    Returns the bullet-point string, or None if generation failed.
    """
    gemini_model = get_config().gemini_model
    if not gemini_model:
        logger.debug("No gemini_model configured, skipping Gemini fallback")
        return None

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.debug("No GEMINI_API_KEY set, skipping Gemini fallback")
        return None

    try:
        from google import genai
    except ImportError:
        logger.debug("google-genai not installed, skipping Gemini fallback")
        return None

    if not content or len(content.strip()) < 50:
        return None

    truncated = content[:_MAX_CONTENT_CHARS]
    prompt = (
        "You are a research digest assistant. Given an article's title, source, "
        "and content, produce 4-6 concise bullet points summarizing the "
        "key findings or arguments. Each bullet should be 1-2 sentences and "
        "capture a distinct insight — aim for 100-150 characters per bullet. "
        "Lead with substance — never start with 'This article' or 'The author'. "
        "Output each bullet on its own line prefixed with '- '.\n\n"
        f"Title: {title or 'Untitled'}\n"
        f"Source: {source or 'Unknown'}\n\n"
        f"Content:\n{truncated}"
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt,
        )
        text = response.text.strip()
    except Exception as exc:
        logger.warning("Gemini abstract failed for '%s': %s", title, exc)
        return None

    if delay > 0:
        time.sleep(delay)

    # Validate
    if len(text) < 20:
        logger.warning("Gemini abstract too short for '%s': %r", title, text)
        return None

    return text


def summarize_items(
    items: list[dict],
    client: anthropic.Anthropic,
    delay: float = 0.1,
) -> int:
    """Generate abstracts in-place for pipeline items that need them.

    Only processes items that have content but need an abstract.
    On Claude refusal, immediately falls back to extractive summary
    so the feed card is never blank.

    Returns the number of items updated.
    """
    candidates = [
        item for item in items
        if needs_abstract(item.get("abstract"))
        and item.get("content")
        and not item.get("_abstract_refused")
    ]

    if not candidates:
        return 0

    updated = 0
    for item in candidates:
        content = item.get("content", "")
        title = item.get("title", "")
        source = item.get("source_name", item.get("source", ""))

        result, was_refused = generate_abstract(
            title=title,
            content=content,
            source=source,
            client=client,
            delay=delay,
        )
        if result:
            item["abstract"] = result
            updated += 1
        elif was_refused:
            # Claude refused — use extractive fallback immediately
            extractive = generate_extractive_abstract(content)
            if extractive:
                item["abstract"] = extractive
                logger.info("Extractive fallback for '%s'", title)
                updated += 1
            # Mark as refused either way so Gemini retry picks it up
            item["_abstract_refused"] = True
        elif content and len(content.strip()) >= 50:
            # Other failure (API error, too short, etc.)
            item["_abstract_refused"] = True

    return updated


def retry_refused_with_gemini(
    items: list[dict],
    delay: float = 0.2,
) -> int:
    """Second pass: retry refused items with Gemini Flash.

    Items that Claude refused get an extractive abstract immediately,
    but this pass tries to upgrade them to a proper LLM summary via
    Gemini which has a different content policy.

    Returns the number of items upgraded.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.debug("No GEMINI_API_KEY, skipping Gemini retry pass")
        return 0

    refused = [
        item for item in items
        if item.get("_abstract_refused")
        and item.get("content")
    ]

    if not refused:
        return 0

    logger.info("Retrying %d refused items with Gemini Flash", len(refused))
    upgraded = 0
    for item in refused:
        result = generate_abstract_gemini(
            title=item.get("title", ""),
            content=item.get("content", ""),
            source=item.get("source_name", item.get("source", "")),
            delay=delay,
        )
        if result:
            item["abstract"] = result
            # Clear the refusal flag since we got a good abstract
            item["_abstract_refused"] = False
            upgraded += 1
            logger.info("Gemini succeeded for '%s'", item.get("title", ""))

    if upgraded:
        logger.info("Gemini upgraded %d/%d refused items", upgraded, len(refused))

    return upgraded
