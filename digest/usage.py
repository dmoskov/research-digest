"""Structured logging of Anthropic API usage, for cost tracking.

Every ``messages.create()`` call in the pipeline reports here. The default
implementation emits one JSON line per call on the ``anthropic_usage`` logger,
queryable in CloudWatch Logs Insights:

    filter @message like "anthropic_usage"
    | stats sum(cost_usd) as cost, sum(input_tokens) as in_tk,
            sum(output_tokens) as out_tk, count(*) as calls
      by service
    | sort cost desc

A host application with its own accounting can replace it:

    from digest.usage import set_usage_logger
    set_usage_logger(my_metrics.record_llm_call)

Usage logging is best-effort and never raises — a logging failure must never
break the API call that produced the response.
"""

import json
import logging
from typing import Callable, Optional

logger = logging.getLogger("anthropic_usage")

# Per-million-token (input, output) pricing. Keys are prefixes so both short
# aliases ("claude-opus-4-6") and dated variants ("claude-sonnet-4-20250514")
# match. Unknown models cost 0.0 — they still log tokens, so a missing entry
# shows up as calls with zero cost rather than as missing data.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-4": (1.0, 5.0),
}

_override: Optional[Callable] = None


def set_usage_logger(fn: Optional[Callable]) -> None:
    """Route usage reports to ``fn(response, service)``; ``None`` restores the default."""
    global _override
    _override = fn


def _get_pricing(model: str) -> tuple[float, float]:
    for prefix, pricing in MODEL_PRICING.items():
        if model.startswith(prefix):
            return pricing
    return (0.0, 0.0)


def log_usage(response, service: str) -> None:
    """Report token usage and estimated cost for one API call (best-effort)."""
    if _override is not None:
        try:
            _override(response, service)
        except Exception:  # noqa: BLE001 - never break the caller
            logger.debug("usage logger override failed for service=%s", service, exc_info=True)
        return
    try:
        u = response.usage
        model = getattr(response, "model", "unknown")
        in_tk = getattr(u, "input_tokens", 0) or 0
        out_tk = getattr(u, "output_tokens", 0) or 0
        price_in, price_out = _get_pricing(model)
        cost = in_tk * price_in / 1e6 + out_tk * price_out / 1e6
        logger.info(
            json.dumps(
                {
                    "event": "anthropic_usage",
                    "service": service,
                    "model": model,
                    "input_tokens": in_tk,
                    "output_tokens": out_tk,
                    "cost_usd": round(cost, 6),
                }
            )
        )
    except Exception:  # noqa: BLE001 - usage logging must never break the caller
        logger.debug("log_usage failed for service=%s", service, exc_info=True)
