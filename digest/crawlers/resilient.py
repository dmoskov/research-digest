"""Shared retry/backoff logic and identity constants for all crawlers."""

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared bot identity — used by all crawlers
# ---------------------------------------------------------------------------
# Built per call from DigestConfig, never module constants: this pipeline
# identifies *you* to every site it fetches. Shipping a fixed identity would
# make every deployment crawl the web under one name, so rate-limit reputation,
# blocklisting and abuse complaints would all land on whoever that name belongs
# to. It also matters for correctness: CrossRef and OpenAlex route callers into
# their "polite pool" by the mailto in the User-Agent.
#
# An honest reader/bot User-Agent is deliberate. Browser-like UAs cause TLS
# fingerprint mismatches with anti-bot systems (Python requests uses OpenSSL,
# not BoringSSL) and trigger harder blocks than declaring what you are.


def bot_user_agent() -> str:
    """Configured crawler User-Agent, e.g. ``AcmeDigest/1.0 (+https://acme.org/bot)``."""
    from digest.settings import get_config

    config = get_config()
    suffix = f" (+{config.bot_info_url})" if config.bot_info_url else ""
    return f"{config.bot_name}/1.0{suffix}"


def bot_contact() -> str:
    """Configured contact address, for APIs that route by mailto. May be empty."""
    from digest.settings import get_config

    return get_config().bot_contact


def rss_headers() -> dict:
    """Headers for RSS/Atom feed fetching."""
    return {
        "User-Agent": bot_user_agent(),
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*;q=0.1",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en",
    }


def html_headers() -> dict:
    """Headers for HTML page fetching (scraping)."""
    return {
        "User-Agent": bot_user_agent(),
        "Accept": "text/html, application/xhtml+xml, */*;q=0.1",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en",
    }


def api_headers() -> dict:
    """Headers for JSON API fetching, carrying the mailto when one is configured."""
    contact = bot_contact()
    from digest.settings import get_config

    name = get_config().bot_name
    agent = f"{name}/1.0 (mailto:{contact})" if contact else bot_user_agent()
    return {
        "User-Agent": agent,
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
    }

# Status codes that are worth retrying (transient errors)
_RETRYABLE = {429, 500, 502, 503, 504}


def _resolve_wait(
    attempt: int,
    backoff_base: float,
    retry_after_header: Optional[str] = None,
) -> float:
    """Compute backoff wait, optionally honoring a Retry-After header."""
    wait = backoff_base**attempt
    if retry_after_header:
        try:
            wait = max(wait, float(retry_after_header))
        except ValueError:
            logger.debug("Failed to parse Retry-After header: %s", retry_after_header)
    return wait


def _log_retry_and_sleep(label: str, reason: str, attempt: int, max_retries: int, wait: float) -> None:
    """Log a retry warning and sleep for the backoff period."""
    logger.warning(
        "%s: %s on attempt %d/%d, retrying in %.1fs",
        label,
        reason,
        attempt,
        max_retries,
        wait,
    )
    time.sleep(wait)


def resilient_get(
    session: requests.Session,
    url: str,
    *,
    params: Optional[dict] = None,
    timeout: int = 30,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    source_label: str = "",
) -> requests.Response:
    """GET with exponential backoff on transient failures.

    Retries on connection errors and 429/5xx status codes.
    Raises the final exception if all retries are exhausted.
    """
    label = source_label or url[:60]
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)

            if response.status_code in _RETRYABLE:
                wait = _resolve_wait(
                    attempt,
                    backoff_base,
                    response.headers.get("Retry-After"),
                )
                _log_retry_and_sleep(label, f"HTTP {response.status_code}", attempt, max_retries, wait)
                continue

            response.raise_for_status()
            return response

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_exc = exc
            wait = _resolve_wait(attempt, backoff_base)
            reason = "timeout" if isinstance(exc, requests.exceptions.Timeout) else f"connection error ({exc})"
            _log_retry_and_sleep(label, reason, attempt, max_retries, wait)

        except requests.exceptions.RequestException:
            raise

    if last_exc:
        raise last_exc
    raise requests.exceptions.RetryError(f"{label}: all {max_retries} retries exhausted")


# ---------------------------------------------------------------------------
# Cloudflare bypass via curl_cffi (optional dependency)
# ---------------------------------------------------------------------------


def cloudflare_get(url: str, *, timeout: int = 30, source_label: str = "") -> Optional[requests.Response]:
    """Fetch a URL using curl_cffi to bypass Cloudflare Bot Fight Mode.

    Uses TLS fingerprint impersonation (chrome) to pass Cloudflare's
    JA3/JA4 checks. Falls back to None if curl_cffi is not installed.

    Returns a requests-compatible Response, or None if unavailable.
    """
    try:
        from curl_cffi import requests as cffi_requests
    except ImportError:
        logger.warning(
            "%s: curl_cffi not installed, cannot bypass Cloudflare. Install with: pip install curl_cffi",
            source_label or url[:60],
        )
        return None

    label = source_label or url[:60]
    try:
        resp = cffi_requests.get(url, impersonate="chrome", timeout=timeout)
        if resp.status_code == 200:
            logger.info("%s: Cloudflare bypass succeeded via curl_cffi", label)
        return resp
    except Exception as exc:
        logger.warning("%s: curl_cffi request failed: %s", label, exc)
        return None
