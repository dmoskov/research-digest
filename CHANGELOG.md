# Changelog

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
