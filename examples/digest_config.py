"""A worked DigestConfig. Copy this, replace the subject matter, keep the shape.

Point the CLI at it:

    export DIGEST_CONFIG=examples.digest_config:CONFIG
    research-digest crawl --test-mode
"""

from digest.settings import DigestConfig

CONFIG = DigestConfig(
    org_name="Acme Foundation",
    # One subtopic per team or per feed. team_context goes into the classifier
    # prompt verbatim and is the main lever on precision — note that both of
    # these say what is OUT of scope, not just what is in it.
    subtopics={
        "climate": {
            "name": "Climate",
            "description": "Decarbonisation policy and technology",
            "team_context": """We fund work to decarbonise heavy industry and the electricity grid.

SCOPE (apply strictly). Relevant = decarbonisation policy, grid and industrial energy technology, carbon removal, and the economics of the energy transition. NOT relevant: general climate *science* with no technology or policy lever — paleoclimate reconstruction, ecosystem monitoring, climate-model methods — and unrelated physics, chemistry or biology appearing in the same journals.""",  # noqa: E501
        },
        "housing": {
            "name": "Housing",
            "description": "Housing supply and land use",
            "team_context": """We fund work to increase housing supply in high-cost metros.

SCOPE (apply strictly). Relevant = land-use and zoning policy, permitting and construction cost, housing finance as it bears on supply, and the empirical economics of housing markets. NOT relevant: homelessness service delivery, mortgage-market finance with no supply channel, and architecture or urban-design criticism.""",  # noqa: E501
        },
    },
    # Topic areas within each subtopic. `keywords` drives the cheap prefilter
    # that decides which items are worth an LLM call; `weight` scales its score.
    subtopic_topics={
        "climate": {
            "grid": {
                "name": "Grid",
                "description": "Transmission, storage, interconnection",
                "keywords": [
                    "transmission line",
                    "interconnection queue",
                    "grid storage",
                    "capacity market",
                    "FERC",
                ],
            },
            "industrial": {
                "name": "Industrial Heat",
                "description": "Cement, steel, process heat",
                "keywords": [
                    "green steel",
                    "cement decarbonisation",
                    "process heat",
                    "industrial electrification",
                ],
            },
            "general_climate": {
                "name": "General Climate",
                "description": "Cross-cutting transition work",
                "keywords": ["energy transition", "net zero"],
            },
        },
        "housing": {
            "zoning": {
                "name": "Zoning",
                "description": "Land use and permitting",
                "keywords": ["zoning reform", "upzoning", "permitting time", "land use regulation"],
            },
            "construction_cost": {
                "name": "Construction Cost",
                "description": "What building costs and why",
                "keywords": ["construction cost", "building code", "modular construction"],
            },
        },
    },
    # Catch-all topics. A source_auto_tags rule only fires when the classifier
    # found nothing outside this set, so an org's newsletter picks up the
    # catch-all while its real analysis still sorts under the real topic.
    secondary_topics=frozenset({"general_climate"}),
    source_auto_tags={"acme_blog": [("climate", "general_climate")]},
    # Optional: "do we already know this author or organisation?"
    network_connections={
        "authors": {"jane quimby": "Co-authored our 2024 grid report"},
        "organizations": {"institute for widgets": "Grantee since 2023"},
        "publications": {},
    },
    site_base_url="https://digests.acme.org",
    # Crawler identity, sent to every site fetched. Set it to something that
    # identifies you: this is where rate-limit reputation and abuse complaints
    # land, and the mailto gets you into CrossRef's and OpenAlex's polite pool.
    bot_name="AcmeDigest",
    bot_info_url="https://acme.org/bot",
    bot_contact="tech@acme.org",
)
