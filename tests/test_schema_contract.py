"""The schema must declare every column the storage layer writes.

This exists because 0.1.1 fixed the opposite: `001_initial.sql` was missing two
`items` columns that `digest.storage` inserts, so `migrate` succeeded and then
every item store failed at runtime. No unit test caught it — they all mock the
cursor, so a column that does not exist is indistinguishable from one that does.

These tests need no database. They read the shipped SQL and the storage module's
own INSERT statements and check the two agree, which is the cheap half of the
guard; `test_integration_db.py` is the expensive half.
"""

import re
from pathlib import Path

import pytest

import digest.storage as storage_module
from digest.migrate import SCHEMA_DIR

SCHEMA_FILES = sorted(Path(SCHEMA_DIR).glob("*.sql"))


def declared_columns() -> dict[str, set[str]]:
    """Map table -> declared columns, from CREATE TABLE and ALTER TABLE ADD COLUMN."""
    tables: dict[str, set[str]] = {}
    sql = "\n".join(p.read_text() for p in SCHEMA_FILES)

    # CREATE TABLE [IF NOT EXISTS] name ( ...body... );
    for match in re.finditer(
        r"CREATE TABLE (?:IF NOT EXISTS )?(\w+)\s*\((.*?)\n\);", sql, re.S | re.I
    ):
        table, body = match.group(1), match.group(2)
        columns = set()
        for raw in body.split("\n"):
            line = raw.split("--")[0].strip()
            if not line:
                continue
            first = line.split()[0].strip("(),")
            # Skip table-level constraint clauses, which are not columns.
            if first.upper() in {
                "PRIMARY", "UNIQUE", "FOREIGN", "CONSTRAINT", "CHECK", "EXCLUDE",
            }:
                continue
            if re.fullmatch(r"\w+", first):
                columns.add(first)
        tables[table] = columns

    for table, column in re.findall(
        r"ALTER TABLE (\w+) ADD COLUMN (?:IF NOT EXISTS )?(\w+)", sql, re.I
    ):
        tables.setdefault(table, set()).add(column)

    return tables


def inserted_columns() -> list[tuple[str, list[str]]]:
    """Extract (table, columns) for every INSERT INTO ... (cols) in digest.storage."""
    source = Path(storage_module.__file__).read_text()
    found = []
    for match in re.finditer(r"INSERT INTO (\w+)\s*\(([^)]*)\)", source, re.S):
        table = match.group(1)
        columns = [
            c.strip()
            for c in match.group(2).replace("\n", " ").split(",")
            if c.strip() and re.fullmatch(r"\w+", c.strip())
        ]
        if columns:
            found.append((table, columns))
    return found


class TestSchemaCoversStorage:
    def test_schema_files_are_present(self):
        assert SCHEMA_FILES, f"no .sql files shipped in {SCHEMA_DIR}"

    def test_storage_has_insert_statements_to_check(self):
        """Guards the guard: a parser that silently finds nothing proves nothing."""
        inserts = inserted_columns()
        assert len(inserts) >= 4, f"expected several INSERTs, parsed {len(inserts)}"

    def test_every_inserted_column_is_declared(self):
        tables = declared_columns()
        problems = []
        for table, columns in inserted_columns():
            if table not in tables:
                problems.append(f"INSERT INTO {table}: table not declared in schema")
                continue
            for column in columns:
                if column not in tables[table]:
                    problems.append(f"{table}.{column} is inserted but not declared")
        assert not problems, "schema does not cover the storage layer:\n  " + "\n  ".join(problems)

    def test_the_regression_columns_are_declared(self):
        """The specific omission that broke 0.1.0."""
        items = declared_columns().get("items", set())
        assert "api_refused_at" in items
        assert "api_refusal_type" in items

    @pytest.mark.parametrize(
        "table",
        [
            "items", "classifications", "item_topics", "cg_connections", "sources",
            "feeds", "digest_snapshots", "topic_summaries", "user_read_items",
            "user_feed_preferences",
        ],
    )
    def test_documented_tables_exist(self, table):
        """The README promises these ten tables."""
        assert table in declared_columns()


class TestMigrationHygiene:
    def test_filenames_sort_into_apply_order(self):
        names = [p.name for p in SCHEMA_FILES]
        assert names == sorted(names)
        for name in names:
            assert re.match(r"^\d{3}_", name), f"{name} needs a NNN_ prefix to order"

    def test_no_concurrent_index_creation(self):
        """Each migration runs in one transaction; Postgres rejects CONCURRENTLY there."""
        for path in SCHEMA_FILES:
            assert "CONCURRENTLY" not in path.read_text().upper(), (
                f"{path.name} uses CREATE INDEX CONCURRENTLY, which cannot run "
                "inside the transaction migrate() wraps each file in"
            )

    def test_migrations_after_the_first_are_idempotent(self):
        """A follow-up migration may be applied to databases in several states."""
        for path in SCHEMA_FILES[1:]:
            body = path.read_text().upper()
            for statement, guard in (
                ("ADD COLUMN", "IF NOT EXISTS"),
                ("CREATE TABLE", "IF NOT EXISTS"),
                ("CREATE INDEX", "IF NOT EXISTS"),
            ):
                if statement in body:
                    assert guard in body, f"{path.name}: {statement} without {guard}"
