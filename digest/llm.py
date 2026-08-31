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

from typing import Callable, Optional

import anthropic

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
