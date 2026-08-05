"""Shared Anthropic client factory for the digest pipeline.

Every pipeline call site must construct its client here so they all get the
same retry/timeout policy. The anthropic SDK natively retries connection
errors, timeouts, 408/409/429 and >=500 (including 529 overloaded) with
exponential backoff, and fails immediately on other 4xx errors such as
AuthenticationError and BadRequestError.
"""

from typing import Optional

import anthropic

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 3


def get_anthropic_client(
    api_key: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> anthropic.Anthropic:
    """Build an Anthropic client with the pipeline-standard retry policy.

    Args:
        api_key: API key (defaults to the ANTHROPIC_API_KEY env var)
        timeout: Per-request timeout in seconds
        max_retries: SDK-level retries for transient errors (429/5xx/timeouts)
    """
    kwargs = {"timeout": timeout, "max_retries": max_retries}
    if api_key:
        kwargs["api_key"] = api_key
    return anthropic.Anthropic(**kwargs)
