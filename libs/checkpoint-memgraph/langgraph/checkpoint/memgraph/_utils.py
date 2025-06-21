"""Utility helpers for Memgraph checkpoint back‑end."""

from __future__ import annotations

from typing import Any, Dict
from urllib.parse import urlparse, unquote


def parse_bolt_uri(conn: str) -> Dict[str, Any]:
    """
    Parse a ``bolt://user:pass@host:port`` (or ``neo4j://``) style URI into the
    separate pieces required by the Neo4j Python driver.

    Returns a mapping with ``bolt_uri``, ``user`` and ``password`` keys.
    """
    parsed = urlparse(conn)
    if parsed.scheme not in ("bolt", "neo4j"):
        raise ValueError(f"Unsupported scheme in URI: {parsed.scheme!r}")

    user = unquote(parsed.username) if parsed.username else "neo4j"
    password = unquote(parsed.password) if parsed.password else "neo4j"
    bolt_uri = f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 7687}"
    return {"bolt_uri": bolt_uri, "user": user, "password": password}

