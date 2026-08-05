"""Central logging setup for the digest pipeline."""

import logging
import os


def setup_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=level or os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
    )
