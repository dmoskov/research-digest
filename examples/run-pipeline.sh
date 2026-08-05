#!/bin/bash
# Reference two-phase runner.
#
# Phase 1 crawls and stores; phase 2 classifies what is stored. They are
# separate processes on purpose: each commits incrementally, so an interruption
# in either costs at most the work in flight, and phase 2 can be re-run after a
# fix without re-crawling.
set -euo pipefail

DAYS_BACK="${DAYS_BACK:-30}"
WORKERS="${WORKERS:-8}"

# Applies any pending schema migrations. Safe on every start; crashes the
# container if a migration fails, so a bad deploy never serves traffic.
research-digest migrate

echo "$(date '+%Y-%m-%d %H:%M:%S') === Phase 1: crawl + store ==="

research-digest crawl \
    --store-db --use-state --days-back "$DAYS_BACK" \
    && CRAWL_EXIT=0 || CRAWL_EXIT=$?

# Exit 1 means some individual sources failed, which is normal and expected —
# feeds move and publishers add bot protection. Exit 2+ is fatal.
if [ "$CRAWL_EXIT" -ne 0 ] && [ "$CRAWL_EXIT" -ne 1 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') crawl failed (exit=$CRAWL_EXIT)"
    exit "$CRAWL_EXIT"
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') === Phase 2: classify ($WORKERS workers) ==="

research-digest classify \
    --workers "$WORKERS" --days-back "$DAYS_BACK" \
    && CLASSIFY_EXIT=0 || CLASSIFY_EXIT=$?

echo "$(date '+%Y-%m-%d %H:%M:%S') done (crawl=$CRAWL_EXIT, classify=$CLASSIFY_EXIT)"
exit "$CLASSIFY_EXIT"
