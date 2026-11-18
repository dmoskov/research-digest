"""Cost reported by digest.usage.log_usage, including prompt-cache tokens."""

import json
import logging

import pytest
from anthropic.types import CacheCreation, Message, Usage

from digest.usage import log_usage, set_usage_logger


@pytest.fixture(autouse=True)
def default_usage_logger():
    set_usage_logger(None)
    yield
    set_usage_logger(None)


def _logged(caplog, model: str, usage: Usage) -> dict:
    message = Message(
        id="msg_test",
        type="message",
        role="assistant",
        model=model,
        content=[],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=usage,
    )
    with caplog.at_level(logging.INFO, logger="anthropic_usage"):
        log_usage(message, "test_service")
    lines = [r.getMessage() for r in caplog.records if r.name == "anthropic_usage"]
    assert len(lines) == 1, lines
    return json.loads(lines[0])


COST_CASES = [
    pytest.param(
        "claude-sonnet-5-5",
        Usage(input_tokens=1000, output_tokens=100),
        0.003,
        id="uncached call: input and output only",
    ),
    pytest.param(
        "claude-sonnet-5-5",
        Usage(
            input_tokens=2000,
            output_tokens=700,
            cache_creation_input_tokens=4000,
            cache_read_input_tokens=0,
            cache_creation=CacheCreation(ephemeral_5m_input_tokens=4000, ephemeral_1h_input_tokens=0),
        ),
        0.021,
        id="5-minute cache write costs 1.25x input",
    ),
    pytest.param(
        "claude-sonnet-5-5",
        Usage(
            input_tokens=2000,
            output_tokens=700,
            cache_creation_input_tokens=4000,
            cache_read_input_tokens=0,
            cache_creation=CacheCreation(ephemeral_5m_input_tokens=1000, ephemeral_1h_input_tokens=3000),
        ),
        0.0255,
        id="1-hour cache write costs 2x input",
    ),
    pytest.param(
        "claude-sonnet-5-5",
        Usage(
            input_tokens=2000,
            output_tokens=700,
            cache_creation_input_tokens=4000,
            cache_read_input_tokens=0,
        ),
        0.021,
        id="cache write without a TTL breakdown is priced as 5-minute",
    ),
    pytest.param(
        "claude-sonnet-5-5",
        Usage(input_tokens=2000, output_tokens=700, cache_creation_input_tokens=0, cache_read_input_tokens=40000),
        0.015,
        id="Sonnet 5.5 cache read costs 0.05x input",
    ),
    pytest.param(
        "claude-opus-5-5",
        Usage(input_tokens=1000, output_tokens=100, cache_creation_input_tokens=0, cache_read_input_tokens=10000),
        0.008,
        id="Opus 5.5 cache read costs 0.05x input",
    ),
    pytest.param(
        "claude-opus-4-6",
        Usage(input_tokens=1000, output_tokens=100, cache_creation_input_tokens=0, cache_read_input_tokens=10000),
        0.0125,
        id="Opus 4.6 cache read costs 0.1x input",
    ),
    pytest.param(
        "claude-fable-5-2",
        Usage(input_tokens=1000, output_tokens=100, cache_creation_input_tokens=0, cache_read_input_tokens=10000),
        0.014,
        id="Fable 5.2 cache read costs 0.025x input",
    ),
    pytest.param(
        "claude-fable-5-1",
        Usage(input_tokens=1000, output_tokens=100, cache_creation_input_tokens=0, cache_read_input_tokens=10000),
        0.0175,
        id="Fable 5.1 cache read costs 0.025x input",
    ),
    pytest.param(
        "claude-fable-5",
        Usage(input_tokens=1000, output_tokens=100, cache_creation_input_tokens=0, cache_read_input_tokens=10000),
        0.025,
        id="Fable 5 cache read costs 0.1x input",
    ),
]


@pytest.mark.parametrize("model, usage, expected_cost", COST_CASES)
def test_cost_includes_cache_tokens(caplog, model, usage, expected_cost):
    assert _logged(caplog, model, usage)["cost_usd"] == pytest.approx(expected_cost, abs=1e-9)


def test_log_line_reports_cache_tokens_separately_from_input(caplog):
    line = _logged(
        caplog,
        "claude-sonnet-5-5",
        Usage(
            input_tokens=2173,
            output_tokens=729,
            cache_creation_input_tokens=3481,
            cache_read_input_tokens=6962,
            cache_creation=CacheCreation(ephemeral_5m_input_tokens=3481, ephemeral_1h_input_tokens=0),
        ),
    )

    assert line["input_tokens"] == 2173
    assert line["output_tokens"] == 729
    assert line["cache_creation_input_tokens"] == 3481
    assert line["cache_read_input_tokens"] == 6962


def test_uncached_call_logs_zero_cache_tokens(caplog):
    line = _logged(caplog, "claude-sonnet-5-5", Usage(input_tokens=1000, output_tokens=100))

    assert line["cache_creation_input_tokens"] == 0
    assert line["cache_read_input_tokens"] == 0
