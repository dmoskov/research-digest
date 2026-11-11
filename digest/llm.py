"""Shared Anthropic client factory for the digest pipeline.

Every pipeline call site must construct its client here so they all get the
same retry/timeout policy. The anthropic SDK natively retries connection
errors, timeouts, 408/409/429 and >=500 (including 529 overloaded) with
exponential backoff, and fails immediately on other 4xx errors such as
AuthenticationError and BadRequestError.

Credentials come from one of two places. By default the SDK reads
``ANTHROPIC_API_KEY``. A deployment whose credential is not a static key —
workload identity federation, say, where a short-lived token is minted per
exchange — supplies ``DigestConfig.anthropic_client_factory`` instead, and
every call site here routes through it.
"""

import logging
import os
from typing import Callable, Optional

import anthropic

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 3


def configured_client_factory() -> Optional[Callable[..., anthropic.Anthropic]]:
    """Return the host application's client factory, or None.

    Safe to call before :func:`digest.settings.configure` — an unconfigured
    pipeline simply has no factory, which is the same answer as a
    configuration that did not set one.
    """
    from digest.settings import get_config, is_configured

    if not is_configured():
        return None
    return get_config().anthropic_client_factory


def anthropic_credentials_available(api_key: Optional[str] = None) -> bool:
    """Whether some Anthropic credential source is configured.

    Used by call sites that want to fail loudly up front rather than on the
    first request. An explicit key or ``ANTHROPIC_API_KEY`` counts, and so
    does a configured client factory — that factory owns its own credential
    resolution, so no environment variable need be set at all.
    """
    import os

    if api_key or os.environ.get("ANTHROPIC_API_KEY"):
        return True
    return configured_client_factory() is not None


def get_anthropic_client(
    api_key: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> anthropic.Anthropic:
    """Build an Anthropic client with the pipeline-standard retry policy.

    Args:
        api_key: API key (defaults to the ANTHROPIC_API_KEY env var). An
            explicit key wins over a configured client factory, matching the
            SDK's own precedence.
        timeout: Per-request timeout in seconds
        max_retries: SDK-level retries for transient errors (429/5xx/timeouts)
    """
    kwargs = {"timeout": timeout, "max_retries": max_retries}
    if api_key:
        kwargs["api_key"] = api_key
        return anthropic.Anthropic(**kwargs)

    factory = configured_client_factory()
    if factory is not None:
        return factory(**kwargs)

    return anthropic.Anthropic(**kwargs)


def response_text(response) -> str:
    """Return the text of the first text block in a Messages response.

    Not ``content[0].text``: a model that thinks puts a ``thinking`` block
    first, and indexing blindly reads ``.text`` off it and raises. Thinking is
    on by default on the newest models, so the naive form is a trap waiting for
    whichever model the deployment configures next.

    Returns "" when the response carries no text block at all — callers already
    guard the empty-content case and treat it as a failed item.
    """
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


def effort_params() -> dict:
    """Keyword arguments carrying ``DigestConfig.claude_effort``, or none.

    Splatted into every ``messages.create()`` call the pipeline makes. Empty
    when no effort is configured, because not every model accepts the field.
    """
    from digest.settings import get_config

    effort = get_config().claude_effort
    return {"output_config": {"effort": effort}} if effort else {}


def refusal_category(response) -> Optional[str]:
    """Why the model declined this request, or None when it did not.

    The newest models' safety classifiers decline with an ordinary HTTP 200 and
    ``stop_reason == "refusal"``. The content is then empty or cut off partway,
    so check this before reading the text. Returns the ``stop_details``
    category ("cyber", "bio", ...), or "unspecified" for a refusal without one.
    """
    if getattr(response, "stop_reason", None) != "refusal":
        return None
    category = getattr(getattr(response, "stop_details", None), "category", None)
    return category if isinstance(category, str) and category else "unspecified"


# ---------------------------------------------------------------------------
# Non-Anthropic fallback
# ---------------------------------------------------------------------------


def gemini_generate(prompt: str, system: Optional[str] = None) -> Optional[str]:
    """Send one prompt to the configured Gemini model and return its text.

    This is the fallback for content Claude declines: a model from another
    vendor has a different content policy, so an item one refuses the other
    will usually read. It is used for abstracts (``digest.summarizer``) and
    for classification (``digest.classifiers``).

    Returns None when there is no fallback to use — ``gemini_model`` unset, no
    ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``, or the ``gemini`` extra not
    installed — and when Gemini answers with nothing, which is how it reports
    a block of its own. An API failure is logged and also returns None: the
    caller is already on its second attempt and has nowhere further to go, so
    it treats every None the same way and leaves the item for the next run.
    """
    from digest.settings import get_config

    model = get_config().gemini_model
    if not model:
        logger.debug("No gemini_model configured, no fallback available")
        return None

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.debug("No GEMINI_API_KEY set, no fallback available")
        return None

    try:
        from google import genai
    except ImportError:
        logger.debug("google-genai not installed, no fallback available")
        return None

    try:
        # Bound to a name on purpose: the SDK closes the HTTP client when the
        # Client object is collected, so `genai.Client(...).models.generate_content(...)`
        # fails mid-request with "client has been closed".
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"system_instruction": system} if system else None,
        )
        text = response.text
    except Exception as exc:
        logger.warning("Gemini fallback (%s) failed: %s", model, exc)
        return None

    if not text or not text.strip():
        feedback = getattr(response, "prompt_feedback", None)
        logger.warning(
            "Gemini fallback (%s) returned no text (block_reason=%s)",
            model,
            getattr(feedback, "block_reason", None),
        )
        return None

    return text
