# Changelog

## 0.1.1 — 2026-08-05

**Fixes an incomplete initial schema in 0.1.0.** `001_initial.sql` omitted
`items.api_refused_at` and `items.api_refusal_type`, which `digest.storage`
writes on every item upsert — so on 0.1.0 `migrate` succeeded and then every
single item store failed with `column "api_refused_at" does not exist`. A crawl
would report sources fetched and zero items stored.

Cause: the schema was consolidated from the originating deployment's base
schema file, which had never absorbed the migration that added those two
columns. Caught by running the pipeline against a real PostgreSQL rather than
trusting the unit tests, none of which touch a database.

- `001_initial.sql` now declares both columns, so fresh installs are correct.
- `002_item_refusal_tracking.sql` adds them idempotently, repairing databases
  created by 0.1.0 (which recorded 001 as applied, so the corrected 001 will
  never re-run for them).

Upgrading from 0.1.0: `research-digest migrate`. No data loss either way — the
failed inserts never committed.

## 0.1.0 — 2026-08-05

First public release, extracted from the pipeline Coefficient Giving has run in
production since 2025.

The extraction made the engine configuration-driven rather than rewriting it, so
the crawlers, enrichment and classification path are the ones that have been in
service. What changed in the process:

- **`DigestConfig`** — taxonomy, `team_context` prompts, organisation name,
  model IDs and source rules are all supplied by the host application. Nothing
  organisation-specific remains in the engine.
- **`digest.db`** — environment-variable connection pool by default, with
  `use_connection_factory()` for applications that already own a pool.
- **`digest.usage`** — per-call token and cost logging, with an injectable sink.
- **`research-digest` CLI** — `crawl`, `classify`, `seed-sources`, `migrate`.
  Phase 2 of the pipeline previously lived outside the package, so an installed
  copy had a crawler and no way to drive the classifier.
- **Schema shipped with the package** — 10 tables and 13 indexes, applied by
  `research-digest migrate` with its own migration tracking.
- **Seed-time source validation** — required fields, `crawler_type` against the
  dispatch registry, duplicate keys; all problems reported at once. This found a
  duplicate source key in the original deployment, where the later definition
  had been silently winning via `ON CONFLICT`.

Two bugs fixed during extraction:

- The topic-summary cache never hit. A `RealDictRow` was unpacked as a tuple,
  binding column *names* instead of values, so the invalidation hash never
  matched and every summary was regenerated on every call — at full model price,
  while logging "cache stale" as though that were expected.
- `DigestConfig.site_base_url` replaces a hardcoded CDN hostname that pointed at
  a decommissioned environment, so generated summary links resolved to the wrong
  host.

### Known wart

The network-connection table is named `cg_connections`, inherited from the first
deployment. Nothing outside `digest.storage` depends on the name; renaming it is
a schema change on your side, so it was left alone rather than shipped as a
migration you did not ask for.
