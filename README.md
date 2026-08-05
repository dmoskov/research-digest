# research-digest

[![CI](https://github.com/dmoskov/research-digest/actions/workflows/ci.yml/badge.svg)](https://github.com/dmoskov/research-digest/actions/workflows/ci.yml)

Crawl research sources, classify what you find against your own taxonomy with an
LLM, and serve the result as topic-filtered digest feeds.

The pipeline is subject-matter agnostic. You supply the taxonomy, the prompts
and the source list; the package supplies the crawlers, the classifier, the
deduplication, the storage layer and the schema.

---

## What you get

| Layer | Module | What it does |
|---|---|---|
| Crawlers | `digest.crawlers` | 7 crawler types: RSS, CrossRef, OpenAlex, arXiv Atom, NBER, OSF preprints, generic HTML scraper — with retry, backoff and Cloudflare handling |
| Enrichment | `digest.enrichment` | Fetches full text for items whose feed gave only a stub |
| Summarisation | `digest.summarizer` | Generates card-preview abstracts; falls back to extractive, then to Gemini for content Claude declines |
| Classification | `digest.classifiers` | Keyword prefilter → batched LLM classification against every subtopic at once, with prompt caching |
| Storage | `digest.storage` | Upserts items, classifications, topics; cross-source dedup by content hash |
| Feeds | `digest.topic_summarizer` | Cached per-topic LLM summaries with deep links |
| Schema | `digest.schema` | 10 tables + 13 indexes, applied by `research-digest migrate` |

Two phases, run separately so neither loses the other's work:

```
research-digest crawl    → fetch, enrich, dedupe, store raw items
research-digest classify → score stored items against your taxonomy
```

Phase 1 stores each source's items as soon as it finishes that source, so an
interruption costs you at most the source in flight. Phase 2 picks up whatever
is unclassified, so it can fail, be fixed and be re-run without re-crawling.

---

## Quickstart

```bash
pip install 'research-digest @ git+https://github.com/dmoskov/research-digest@v0.1.1'
```

Extras: `[gemini]` for the non-Anthropic abstract fallback, `[aws]` to read
database credentials from Secrets Manager, `[dev]` for the test tooling.

### 1. Describe your subject matter

```python
# myorg/digest_config.py  —  see examples/digest_config.py for a fuller one
from digest.settings import DigestConfig

CONFIG = DigestConfig(
    org_name="Acme Foundation",
    subtopics={
        "climate": {
            "name": "Climate",
            "description": "Decarbonisation policy and technology",
            "team_context": """We fund work to decarbonise heavy industry and the grid.

SCOPE (apply strictly). Relevant = decarbonisation policy, grid and industrial
energy technology, carbon removal, and the economics of the energy transition.
NOT relevant: general climate *science* with no technology or policy lever —
paleoclimate reconstruction, ecosystem monitoring, climate modelling methods —
and unrelated physics, chemistry or biology from the same journals.""",
        },
    },
    subtopic_topics={
        "climate": {
            "grid": {
                "name": "Grid",
                "description": "Transmission, storage, interconnection",
                "keywords": ["transmission", "interconnection queue", "grid storage"],
            },
            "industrial": {
                "name": "Industrial Heat",
                "description": "Cement, steel, process heat",
                "keywords": ["green steel", "cement decarbonisation", "process heat"],
            },
        },
    },
    site_base_url="https://digests.acme.org",
    # Identify your crawler honestly — this is what publishers see, and it is
    # where rate-limit reputation and any abuse complaints land.
    bot_name="AcmeDigest",
    bot_info_url="https://acme.org/bot",
    bot_contact="tech@acme.org",
)
```

**`team_context` is the single biggest lever on classification quality.** It goes
into the prompt verbatim. A one-line description gets you a classifier that
marks half the journal relevant. State the scope, then state explicitly what is
*out* of scope and why — the negative examples do most of the work. Every
subtopic in the example above follows that shape; copy it.

### 2. Point the CLI at it and create the schema

```bash
export DIGEST_CONFIG=myorg.digest_config:CONFIG
export ANTHROPIC_API_KEY=sk-...
export DIGEST_DB_HOST=... DIGEST_DB_NAME=... DIGEST_DB_USER=... DIGEST_DB_PASSWORD=...

research-digest migrate
```

### 3. Tell it what to crawl

```json
[
  {
    "key": "nber",
    "name": "NBER Working Papers",
    "source_type": "academic_journal",
    "crawler_type": "nber",
    "subtopics": ["climate"],
    "feed_url": "https://back.nber.org/rss/new.xml"
  },
  {
    "key": "volts",
    "name": "Volts",
    "source_type": "blog",
    "crawler_type": "rss",
    "subtopics": ["climate"],
    "feed_url": "https://www.volts.wtf/feed"
  }
]
```

A fuller list is in [`examples/sources.json`](examples/sources.json).

```bash
research-digest seed-sources sources.json --dry-run   # validates, writes nothing
research-digest seed-sources sources.json
```

`--dry-run` validates required fields, checks every `crawler_type` against the
dispatch registry and rejects duplicate keys, reporting all problems at once. A
`crawler_type` typo would otherwise surface as a source that silently never crawls.

### 4. Run it

```bash
research-digest crawl --test-mode          # crawl only, store nothing — start here
research-digest crawl --store-db --use-state --days-back 30
research-digest classify --workers 8
```

---

## Configuration reference

Required: `org_name`, `subtopics`, `subtopic_topics`. Everything else is optional.

| Field | Purpose |
|---|---|
| `topic_groups` | Display grouping of topics within a subtopic |
| `audit_keywords` | Broader keyword sets for the audit pass, which rescans candidates for topics that came back empty. Derived from your first 8 topic keywords if omitted |
| `network_connections` | "Do we already know this author/org/publication?" — `{"authors": {...}, "organizations": {...}, "publications": {...}}`. Omit to disable |
| `source_auto_tags` | `{source_key: [(subtopic, topic_key)]}` — force a topic onto items from a source |
| `secondary_topics` | Topic keys that are catch-alls. An auto-tag rule only fires when the classifier found nothing outside this set, so an org's newsletter gets the catch-all while its real analysis sorts under the real topic |
| `static_sources` | Legacy in-code source registry. Leave empty; seed the `sources` table instead |
| `site_base_url` | Base URL for deep links in generated summaries |
| `bot_name` | **Set this.** Crawler User-Agent, sent to every site fetched — `{bot_name}/1.0`. Rate-limit reputation and abuse complaints follow it |
| `bot_info_url` | Optional page describing your crawler, appended as `(+url)`. Publishers check it before blocking |
| `bot_contact` | Optional email. CrossRef and OpenAlex route callers with a mailto into a faster "polite pool" |
| `claude_model` | Summarisation model |
| `claude_model_fast` | Classification model — runs on every item, so this dominates cost |
| `gemini_model` | Optional fallback for abstracts Claude declines. Needs the `gemini` extra |

Misconfiguration raises at `configure()` time — empty taxonomy, a subtopic with
no topics, a topic set with no subtopic, a subtopic missing `team_context`. The
package never falls back to a default taxonomy, because an empty one classifies
everything as irrelevant and looks exactly like a working pipeline with a quiet week.

---

## Database

`research-digest migrate` applies `digest/schema/*.sql` in filename order and
records each in `digest_schema_migrations`, so it is safe to run on every
container start. One transaction per file — avoid `CREATE INDEX CONCURRENTLY`,
which Postgres refuses inside a transaction.

Ten tables: `items`, `classifications`, `item_topics`, `cg_connections`,
`sources`, `feeds`, `digest_snapshots`, `topic_summaries`, `user_read_items`,
`user_feed_preferences`. Drop the two `user_*` tables if you are not serving a
personalised reading UI.

> `cg_connections` is the network-connection table — the name is inherited from
> the pipeline's first deployment. Rename it if you like; nothing outside
> `digest.storage` depends on the name.

**Already have a connection pool?** Inject it rather than running two:

```python
from digest.db import use_connection_factory
use_connection_factory(my_app.get_connection)
```

The factory must yield a psycopg2 connection whose `cursor()` returns a
`RealDictCursor`, **and must commit on clean exit** — the storage layer never
calls `commit()` itself, so a non-committing factory silently discards every write.

---

## Cost

Classification is the cost centre: it runs on every crawled item, batched 5 per
call, against all subtopics at once. The static part of the prompt (your whole
taxonomy) carries a cache breakpoint, so the second call onwards reads it from
cache rather than paying full input price.

Order of magnitude, at Sonnet-class pricing with ~500 crawled items a week:
low single-digit dollars per week. Opus-class classification is roughly 5×.
Set `claude_model_fast` deliberately.

Watch it: every call emits a JSON line on the `anthropic_usage` logger with
model, tokens and estimated cost. `digest.usage.set_usage_logger()` redirects
that into your own metrics. Zero `cache_read` tokens across a run means
something is invalidating the prompt cache — that is a real cost regression,
and the classifier logs cache counters on every call so you can see it.

---

## Testing

Three layers, because the first two cannot catch what the third does.

```bash
pip install '.[dev]'
pytest                      # 188 unit + contract tests; no database, no network, <1s
```

**Unit** — crawler parsing, scoring, classifier prompt assembly, storage SQL
against a mocked cursor. Runs against a fixture taxonomy in `tests/conftest.py`,
not any real deployment's config, so it stays meaningful once you swap in your
own. Tests asserting on *your* taxonomy's content belong in your suite, not here.

**Schema contract** (`tests/test_schema_contract.py`) — parses the shipped SQL
and the storage layer's own `INSERT` statements and checks they agree. No
database needed. This exists because a mocked cursor cannot tell a column that
exists from one that does not; see the 0.1.1 note in CHANGELOG.md.

**Integration** (`tests/test_integration_db.py`) — executes the real schema and
the real writes against PostgreSQL. Skipped unless `DIGEST_DB_HOST` is set:

```bash
docker run -d --name rd-test -e POSTGRES_PASSWORD=test -e POSTGRES_USER=test \
    -e POSTGRES_DB=digest_test -p 5433:5432 postgres:16

DIGEST_DB_HOST=localhost DIGEST_DB_PORT=5433 DIGEST_DB_NAME=digest_test \
DIGEST_DB_USER=test DIGEST_DB_PASSWORD=test pytest
```

CI runs all three on every push, the integration layer against a Postgres 16
service. What no layer covers: live classification, which would mean spending
tokens on every CI run. The classifier's request assembly and response parsing
are unit-tested against recorded shapes; the API call itself is not.

---

## Deploying

[`examples/Dockerfile`](examples/Dockerfile) and
[`examples/run-pipeline.sh`](examples/run-pipeline.sh) are a working reference:
one image, run on a schedule (cron, ECS scheduled task, Kubernetes CronJob),
migrating then crawling then classifying.

Exit code 1 from `crawl` means some individual sources failed, which is normal;
2 or higher is fatal, and the reference runner distinguishes them. Run
`research-digest migrate` on every start — it is a no-op once applied, and a
failing migration should crash the container rather than let a half-migrated
schema serve traffic.

Rendering the feeds is deliberately out of scope: query `items` joined to
`classifications` and `item_topics` from whatever web stack you already have.

Per-source failures are tracked in `sources.crawl_status`,
`sources.crawl_error` and `sources.consecutive_failures` (reset on success), so
a source that has quietly rotted is visible without reading logs. Crawlers
break constantly — feeds move, publishers add Cloudflare, journals change their
API. Alert on `consecutive_failures`.

---

## Provenance

Extracted from the research-digest pipeline Coefficient Giving has run in
production since 2025 — six subtopics, ~40 topic areas, roughly 150 sources
crawled daily. The extraction removed the organisation-specific parts rather
than rewriting the engine, so the crawlers and the classification path are the
ones that have been in service, not a clean-room reimplementation.

Published under the MIT licence; contributions welcome via issues and pull
requests.
