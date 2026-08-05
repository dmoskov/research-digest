"""``research-digest`` command line entrypoint.

    research-digest crawl    --store-db --use-state --days-back 30   # phase 1
    research-digest classify --workers 8                             # phase 2
    research-digest seed-sources sources.json
    research-digest migrate

The pipeline needs a :class:`~digest.settings.DigestConfig` before it can run.
Point ``DIGEST_CONFIG`` at the module attribute holding yours::

    export DIGEST_CONFIG=myorg.digest_config:CONFIG

A host application that already imports :mod:`digest` and calls ``configure()``
itself can invoke these subcommands directly (``digest.pipeline.main``,
``digest.classify_worker.main``) and skip the env var.
"""

import argparse
import importlib
import logging
import os
import sys

logger = logging.getLogger(__name__)

CONFIG_ENV_VAR = "DIGEST_CONFIG"


def load_config_from_env() -> None:
    """Import and install the DigestConfig named by ``DIGEST_CONFIG``.

    No-op when the process already configured itself in-band.
    """
    from digest.settings import DigestConfig, configure, is_configured

    if is_configured():
        return

    spec = os.environ.get(CONFIG_ENV_VAR)
    if not spec:
        raise SystemExit(
            f"{CONFIG_ENV_VAR} is not set. Point it at the module attribute holding "
            "your DigestConfig, e.g.\n"
            f"    export {CONFIG_ENV_VAR}=myorg.digest_config:CONFIG"
        )

    module_path, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(
            f"{CONFIG_ENV_VAR}='{spec}' is malformed — expected 'module.path:ATTRIBUTE'"
        )

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise SystemExit(f"{CONFIG_ENV_VAR}: cannot import '{module_path}': {e}") from e

    try:
        config = getattr(module, attr)
    except AttributeError as e:
        raise SystemExit(
            f"{CONFIG_ENV_VAR}: module '{module_path}' has no attribute '{attr}'"
        ) from e

    if not isinstance(config, DigestConfig):
        raise SystemExit(
            f"{CONFIG_ENV_VAR}: {spec} is {type(config).__name__}, expected DigestConfig"
        )
    configure(config)


def _cmd_crawl(argv) -> int:
    load_config_from_env()
    from digest.pipeline import main as crawl_main

    return crawl_main(argv) or 0


def _cmd_classify(argv) -> int:
    load_config_from_env()
    from digest.classify_worker import main as classify_main

    return classify_main(argv) or 0


def _cmd_seed_sources(argv) -> int:
    from digest.logging_config import setup_logging
    from digest.sources import load_sources_from_json, upsert_sources

    setup_logging()
    parser = argparse.ArgumentParser(prog="research-digest seed-sources")
    parser.add_argument("path", help="JSON file containing an array of source objects")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print without writing"
    )
    args = parser.parse_args(argv)

    sources = load_sources_from_json(args.path)
    count = upsert_sources(sources, dry_run=args.dry_run)
    verb = "Would upsert" if args.dry_run else "Upserted"
    print(f"{verb} {count} sources")
    return 0


def _cmd_migrate(argv) -> int:
    from digest.logging_config import setup_logging
    from digest.migrate import run_migrations

    setup_logging()
    parser = argparse.ArgumentParser(prog="research-digest migrate")
    parser.add_argument(
        "--dry-run", action="store_true", help="List pending migrations without applying"
    )
    args = parser.parse_args(argv)

    applied = run_migrations(dry_run=args.dry_run)
    verb = "Pending" if args.dry_run else "Applied"
    print(f"{verb}: {len(applied)} migration(s)" + (f" — {', '.join(applied)}" if applied else ""))
    return 0


COMMANDS = {
    "crawl": (_cmd_crawl, "Phase 1: crawl enabled sources, enrich and store items"),
    "classify": (_cmd_classify, "Phase 2: classify stored items against your taxonomy"),
    "seed-sources": (_cmd_seed_sources, "Upsert the sources registry from a JSON file"),
    "migrate": (_cmd_migrate, "Create or update the digest database schema"),
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        width = max(len(name) for name in COMMANDS)
        lines = [
            "usage: research-digest <command> [options]",
            "",
            "commands:",
            *(f"  {name:<{width}}  {help_text}" for name, (_, help_text) in COMMANDS.items()),
            "",
            f"Set {CONFIG_ENV_VAR}=module.path:ATTRIBUTE to point at your DigestConfig.",
            "Run 'research-digest <command> --help' for command options.",
        ]
        print("\n".join(lines))
        return 0

    command, *rest = argv
    if command not in COMMANDS:
        print(
            f"research-digest: unknown command '{command}'. "
            f"Valid commands: {', '.join(COMMANDS)}",
            file=sys.stderr,
        )
        return 2

    handler, _ = COMMANDS[command]
    return handler(rest)


if __name__ == "__main__":
    sys.exit(main())
