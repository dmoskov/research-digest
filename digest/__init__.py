"""research-digest — crawl, classify and summarise research into topic digests.

The pipeline runs in two phases, deliberately decoupled so neither loses work
if the other is interrupted:

  1. crawl    — fetch every enabled source, enrich, dedupe, store raw items
  2. classify — score stored items against your taxonomy with an LLM

Both are exposed on the ``research-digest`` CLI and importable as
:mod:`digest.pipeline` and :mod:`digest.classify_worker`.

Configure before use::

    from digest.settings import DigestConfig, configure
    configure(DigestConfig(org_name="...", subtopics={...}, subtopic_topics={...}))

Submodules are imported lazily: importing :mod:`digest` alone does not require
a configuration, a database, or an API key.
"""

__version__ = "0.1.0"

from digest.settings import DigestConfig, configure, get_config, is_configured

__all__ = ["DigestConfig", "configure", "get_config", "is_configured", "__version__"]
