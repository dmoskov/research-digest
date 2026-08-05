-- research-digest: initial schema
-- PostgreSQL 14+
--
-- Ten tables in three groups:
--   pipeline  — items, classifications, item_topics, cg_connections, sources
--   presentation — feeds, digest_snapshots, topic_summaries
--   per-user  — user_read_items, user_feed_preferences
--
-- The per-user tables key on an email string rather than a user id: the
-- pipeline has no user model of its own and defers identity to the host app.
-- Drop them if you are not serving a personalised reading UI.

-- ---------------------------------------------------------------------------
-- Pipeline
-- ---------------------------------------------------------------------------

-- One row per crawled article/paper.
CREATE TABLE IF NOT EXISTS items (
    id SERIAL PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    content_hash TEXT,             -- 16-char hex hash of normalized title+author, for cross-source dedup
    title TEXT NOT NULL,
    abstract TEXT,
    content TEXT,
    authors TEXT,
    source TEXT NOT NULL,          -- sources.key of the origin
    source_id TEXT,                -- paper number, post ID, etc.
    source_name TEXT,              -- journal or publication name
    published_date DATE,
    crawled_at TIMESTAMPTZ DEFAULT NOW(),
    raw_metadata JSONB DEFAULT '{}'
);

-- Per-subtopic relevance verdict. One row per (item, subtopic).
CREATE TABLE IF NOT EXISTS classifications (
    id SERIAL PRIMARY KEY,
    item_id INTEGER REFERENCES items(id) ON DELETE CASCADE,
    subtopic TEXT NOT NULL,        -- a DigestConfig.subtopics key
    relevant BOOLEAN NOT NULL,
    confidence TEXT,               -- 'high', 'medium', 'low', 'uncertain'
    reasoning TEXT,
    classified_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(item_id, subtopic)
);

-- Topic assignments. An item may match several topics within one subtopic.
CREATE TABLE IF NOT EXISTS item_topics (
    id SERIAL PRIMARY KEY,
    item_id INTEGER REFERENCES items(id) ON DELETE CASCADE,
    subtopic TEXT NOT NULL,
    topic_key TEXT NOT NULL,
    UNIQUE(item_id, subtopic, topic_key)
);

-- Matches against DigestConfig.network_connections ("we know this author/org").
-- Named cg_connections for historical reasons — it is the network-connection
-- table, not anything organisation-specific.
CREATE TABLE IF NOT EXISTS cg_connections (
    id SERIAL PRIMARY KEY,
    item_id INTEGER REFERENCES items(id) ON DELETE CASCADE,
    connection_name TEXT,
    connection_description TEXT,
    UNIQUE(item_id)
);

-- What the pipeline crawls. Seed with `research-digest seed-sources`.
CREATE TABLE IF NOT EXISTS sources (
    id SERIAL PRIMARY KEY,
    key TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    source_type TEXT NOT NULL,        -- 'academic_journal', 'blog', 'preprint', 'policy'
    crawler_type TEXT NOT NULL,       -- 'rss', 'crossref', 'openalex', 'arxiv_atom', 'nber', 'html_scraper', 'osf_preprint'
    subtopics TEXT[] NOT NULL DEFAULT '{}',
    url TEXT,
    feed_url TEXT,
    issn TEXT,
    openalex_source_id TEXT,
    crawl_config JSONB DEFAULT '{}',
    is_enabled BOOLEAN DEFAULT true,
    crawl_status TEXT DEFAULT 'pending',   -- 'ok', 'empty', 'error', 'pending'
    crawl_error TEXT,                      -- last error message; null when ok
    consecutive_failures INTEGER NOT NULL DEFAULT 0,  -- reset on success; drives staleness alerts
    last_crawled_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Presentation
-- ---------------------------------------------------------------------------

-- A feed is a saved filter over classified items — one per audience.
CREATE TABLE IF NOT EXISTS feeds (
    id SERIAL PRIMARY KEY,
    slug TEXT UNIQUE NOT NULL,     -- URL-friendly name
    name TEXT NOT NULL,
    description TEXT,
    filter_config JSONB NOT NULL,  -- {"subtopics": ["..."], "topics": [...], "min_confidence": "medium"}
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true
);

-- Which items a feed contained in a given week, for history.
CREATE TABLE IF NOT EXISTS digest_snapshots (
    id SERIAL PRIMARY KEY,
    feed_id INTEGER REFERENCES feeds(id),
    week_ending DATE NOT NULL,
    item_ids INTEGER[] NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Cached LLM topic summaries, invalidated by item_hash.
CREATE TABLE IF NOT EXISTS topic_summaries (
    id SERIAL PRIMARY KEY,
    feed_id INTEGER REFERENCES feeds(id),
    topic_key TEXT NOT NULL,
    week_ending DATE NOT NULL,
    summary TEXT NOT NULL,
    item_hash TEXT,                -- hash of covered item IDs; a mismatch means regenerate
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(feed_id, topic_key, week_ending)
);

-- ---------------------------------------------------------------------------
-- Per-user state (optional)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS user_read_items (
    id SERIAL PRIMARY KEY,
    user_email TEXT NOT NULL,
    item_id INTEGER REFERENCES items(id) ON DELETE CASCADE,
    read_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_email, item_id)
);

CREATE TABLE IF NOT EXISTS user_feed_preferences (
    id SERIAL PRIMARY KEY,
    user_email TEXT NOT NULL,
    feed_slug TEXT NOT NULL,
    preferences JSONB NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_email, feed_slug)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_items_published ON items(published_date DESC);
CREATE INDEX IF NOT EXISTS idx_items_source ON items(source);
CREATE INDEX IF NOT EXISTS idx_items_content_hash ON items(content_hash) WHERE content_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_items_fulltext ON items USING GIN(
    to_tsvector('english', coalesce(title, '') || ' ' || coalesce(abstract, ''))
);

CREATE INDEX IF NOT EXISTS idx_classifications_subtopic ON classifications(subtopic, relevant);
CREATE INDEX IF NOT EXISTS idx_classifications_item ON classifications(item_id);
-- Partial index covering the hot-path feed JOIN, which only ever wants relevant rows.
CREATE INDEX IF NOT EXISTS idx_classifications_feed_lookup ON classifications(item_id, subtopic) WHERE relevant = true;

CREATE INDEX IF NOT EXISTS idx_item_topics_subtopic ON item_topics(subtopic, topic_key);

CREATE INDEX IF NOT EXISTS idx_sources_crawler_type ON sources(crawler_type) WHERE is_enabled = true;
CREATE INDEX IF NOT EXISTS idx_sources_subtopics ON sources USING GIN(subtopics) WHERE is_enabled = true;

CREATE INDEX IF NOT EXISTS idx_user_read_items_email ON user_read_items(user_email);
CREATE INDEX IF NOT EXISTS idx_user_read_items_item ON user_read_items(item_id);
CREATE INDEX IF NOT EXISTS idx_user_feed_prefs_lookup ON user_feed_preferences(user_email, feed_slug);
