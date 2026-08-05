"""Deduplication helpers for digest items."""

# =============================================================================
# DEDUPLICATION
# =============================================================================

import hashlib
import re
import unicodedata


def compute_content_hash(title: str, authors: str = "") -> str:
    """Compute a stable hash for deduplicating items across sources.

    Normalizes the title (lowercase, strip punctuation/whitespace, remove
    unicode accents) and combines with a normalized first-author last name
    when available.  Two items with the same content_hash are almost
    certainly the same work published on different platforms.

    Returns a 16-char hex string (64-bit, collision-safe for <100k items).
    """
    # Normalize title: lowercase, strip accents, collapse whitespace, drop punctuation
    norm = unicodedata.normalize("NFKD", title.lower())
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    norm = re.sub(r"[^\w\s]", "", norm)   # drop punctuation
    norm = re.sub(r"\s+", " ", norm).strip()

    # Extract first-author last name for disambiguation
    # (two different papers can share a title but rarely share title+author)
    author_key = ""
    if authors:
        first_author = authors.split(",")[0].strip()
        parts = first_author.split()
        if parts:
            author_key = parts[-1].lower()  # last name

    blob = f"{norm}|{author_key}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
