"""Apply the digest schema migrations.

Migrations are ``.sql`` files in ``digest/schema/``, applied in filename order
and recorded in ``digest_schema_migrations`` so each runs exactly once. Run them
on container start (``research-digest migrate``) — a failing migration should
crash the process rather than let a half-migrated schema serve traffic.

Each file is applied in a single transaction. Migrations must therefore avoid
statements Postgres refuses inside a transaction block (notably
``CREATE INDEX CONCURRENTLY``); use a plain ``CREATE INDEX`` and accept the
lock, or apply such changes out of band.
"""

import logging
import os
from typing import List

from digest.db import get_connection

logger = logging.getLogger(__name__)

SCHEMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema")
TRACKING_TABLE = "digest_schema_migrations"

_CREATE_TRACKING = f"""
CREATE TABLE IF NOT EXISTS {TRACKING_TABLE} (
    filename TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


def available_migrations() -> List[str]:
    """Return migration filenames in apply order."""
    if not os.path.isdir(SCHEMA_DIR):
        raise RuntimeError(f"Migration directory not found: {SCHEMA_DIR}")
    return sorted(f for f in os.listdir(SCHEMA_DIR) if f.endswith(".sql"))


def applied_migrations() -> List[str]:
    """Return filenames already recorded as applied."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_CREATE_TRACKING)
            # TRACKING_TABLE is a module constant, not caller input.
            cur.execute(f"SELECT filename FROM {TRACKING_TABLE} ORDER BY filename")  # noqa: S608
            return [row[0] for row in cur.fetchall()]


def run_migrations(dry_run: bool = False) -> List[str]:
    """Apply every migration not yet recorded. Returns the filenames handled."""
    available = available_migrations()
    already = set(applied_migrations())
    pending = [f for f in available if f not in already]

    if not pending:
        logger.info("Schema is up to date (%d migration(s) applied)", len(already))
        return []

    if dry_run:
        for filename in pending:
            logger.info("pending: %s", filename)
        return pending

    for filename in pending:
        path = os.path.join(SCHEMA_DIR, filename)
        with open(path) as fh:
            sql = fh.read()

        logger.info("Applying %s", filename)
        # One transaction per file: a failure leaves that migration unrecorded
        # and unapplied, so a fixed version can be re-run cleanly.
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    f"INSERT INTO {TRACKING_TABLE} (filename) VALUES (%s)",  # noqa: S608
                    (filename,),
                )
        logger.info("Applied %s", filename)

    return pending
