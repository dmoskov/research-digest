"""PostgreSQL access for the digest pipeline.

By default the pool is built from environment variables:

    DIGEST_DB_HOST      (required)
    DIGEST_DB_PORT      (default 5432)
    DIGEST_DB_NAME      (required)
    DIGEST_DB_USER      (required)
    DIGEST_DB_PASSWORD  (required)
    DIGEST_DB_SECRET_ARN  optional AWS Secrets Manager ARN; when set, host /
                          username / password / dbname / port are read from the
                          secret instead. Requires the `aws` extra.

A host application that already owns its connection pool should inject it
instead of setting these, so there is one pool per process rather than two:

    from digest.db import use_connection_factory
    use_connection_factory(my_app.get_connection)

The factory must return a psycopg2 connection whose ``cursor()`` yields a
``RealDictCursor`` — the storage layer indexes rows by column name.

Failures here raise. A digest run that cannot reach the database must not
degrade to crawling into the void.
"""

import json
import logging
import os
import threading
from contextlib import contextmanager
from typing import Callable, Optional

from psycopg2 import pool
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

_pool: Optional[pool.ThreadedConnectionPool] = None
_pool_lock = threading.Lock()
_connection_factory: Optional[Callable] = None

MIN_CONNECTIONS = 1
MAX_CONNECTIONS = int(os.environ.get("DIGEST_DB_MAX_CONNECTIONS", "10"))


def use_connection_factory(factory: Optional[Callable]) -> None:
    """Route all database access through ``factory`` instead of the built-in pool.

    Pass ``None`` to restore the built-in env-var pool. Any pool this module
    already opened is closed.
    """
    global _connection_factory, _pool
    with _pool_lock:
        _connection_factory = factory
        if _pool is not None:
            _pool.closeall()
            _pool = None


def _secret_credentials() -> dict:
    """Read DB credentials from AWS Secrets Manager, if an ARN is configured."""
    secret_arn = os.environ.get("DIGEST_DB_SECRET_ARN")
    if not secret_arn:
        return {}
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as e:
        raise RuntimeError(
            "DIGEST_DB_SECRET_ARN is set but boto3 is not installed. "
            "Install the aws extra: pip install 'research-digest[aws]'"
        ) from e

    region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    try:
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_arn)
    except ClientError as e:
        raise RuntimeError(
            f"Failed to read database secret '{secret_arn}' from Secrets Manager: {e}"
        ) from e
    secret = json.loads(response["SecretString"])
    missing = [k for k in ("host", "username", "password") if not secret.get(k)]
    if missing:
        raise RuntimeError(
            f"Database secret '{secret_arn}' is missing required fields: {', '.join(missing)}"
        )
    return {
        "host": secret["host"],
        "port": int(secret.get("port", 5432)),
        "dbname": secret.get("dbname") or _require_env("DIGEST_DB_NAME"),
        "user": secret["username"],
        "password": secret["password"],
    }


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Configure the digest database via DIGEST_DB_* "
            "environment variables, or inject a pool with "
            "digest.db.use_connection_factory()."
        )
    return value


def get_db_config() -> dict:
    """Resolve database connection parameters."""
    secret = _secret_credentials()
    if secret:
        return secret
    return {
        "host": _require_env("DIGEST_DB_HOST"),
        "port": int(os.environ.get("DIGEST_DB_PORT", "5432")),
        "dbname": _require_env("DIGEST_DB_NAME"),
        "user": _require_env("DIGEST_DB_USER"),
        "password": _require_env("DIGEST_DB_PASSWORD"),
    }


def _get_pool() -> pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                config = get_db_config()
                logger.info(
                    "Opening digest DB pool to %s:%s/%s",
                    config["host"], config["port"], config["dbname"],
                )
                _pool = pool.ThreadedConnectionPool(
                    MIN_CONNECTIONS, MAX_CONNECTIONS, **config
                )
    return _pool


@contextmanager
def get_connection():
    """Yield a connection that **commits on clean exit** and rolls back on error.

    The commit-on-exit semantics are load-bearing: the storage layer issues
    INSERT/UPDATE inside ``with get_connection()`` blocks and never calls
    ``commit()`` itself. An injected factory must therefore commit on clean exit
    too, or every write silently disappears.
    """
    if _connection_factory is not None:
        with _connection_factory() as conn:
            yield conn
        return

    connection_pool = _get_pool()
    conn = None
    try:
        conn = connection_pool.getconn()
        yield conn
        conn.commit()
    except BaseException:
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if conn is not None:
            connection_pool.putconn(conn)


@contextmanager
def get_cursor(cursor_factory=RealDictCursor):
    """Yield a cursor on an auto-committing connection.

    Rows are ``RealDictRow`` (dict-like) by default — index them by column name.
    Pass ``cursor_factory=None`` for psycopg2's default tuple cursor.
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=cursor_factory) as cur:
            yield cur


def close_pool() -> None:
    """Close the built-in pool, if one is open."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.closeall()
            _pool = None
