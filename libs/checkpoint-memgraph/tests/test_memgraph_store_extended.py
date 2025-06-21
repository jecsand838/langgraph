# libs/checkpoint-memgraph/tests/test_memgraph_store_extended.py
"""
Extended integration tests for the synchronous ``MemgraphStore``.

Coverage
--------
* CRUD helpers (exists, count, list_keys, list_items, etc.)
* Bulk operations (put_many, delete_many)
* Namespace deletion
* Hybrid search:
    - Lexical path when no embedder is configured
    - Listing of all items (query=None)
    - Safe handling of special‑character queries (parameter‑binding check)
These tests assume a **local Memgraph instance** is reachable on the Bolt URI
specified by ``MEMGRAPH_BOLT_URI``.  When the database is not available, the
entire test module is skipped so CI jobs without Memgraph do not fail.
"""
from __future__ import annotations

import os
import uuid
from typing import Iterable, Tuple

import pytest

from langgraph.store.memgraph import MemgraphStore

# --------------------------------------------------------------------------- #
# Connection details (overridable via env var)
# --------------------------------------------------------------------------- #
BOLT_URI = os.getenv(
    "MEMGRAPH_BOLT_URI", "bolt://testuser123:BiggerPassword1233@localhost:7687"
)


def _have_db() -> bool:
    """Quick connectivity probe so CI can gracefully skip when DB unavailable."""
    try:
        st = MemgraphStore.from_conn_string(BOLT_URI)
        st.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _have_db(), reason="Memgraph instance not reachable on localhost"
)

# --------------------------------------------------------------------------- #
# Test fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def store() -> Iterable[MemgraphStore]:
    """
    Provide a configured ``MemgraphStore`` for the entire module.

    Each test receives its own randomly‑generated namespace prefix to avoid any
    cross‑test interference.
    """
    st = MemgraphStore.from_conn_string(BOLT_URI)
    st.setup()
    yield st
    st.close()


def _ns() -> Tuple[str, ...]:
    """Generate a unique namespace for every test."""
    return ("tests", str(uuid.uuid4()))


# --------------------------------------------------------------------------- #
# CRUD & helper tests
# --------------------------------------------------------------------------- #
def test_exists_and_count(store: MemgraphStore) -> None:
    ns = _ns()
    store.put(ns, "k1", {"foo": 1})
    store.put(ns, "k2", {"foo": 2})

    assert store.exists(ns, "k1") is True
    assert store.exists(ns, "bogus") is False
    assert store.count(ns) == 2


def test_list_keys_and_items(store: MemgraphStore) -> None:
    ns = _ns()
    items = {f"key{i}": {"v": i} for i in range(5)}
    for k, v in items.items():
        store.put(ns, k, v)

    assert set(store.list_keys(ns)) == set(items)
    retrieved = {it.key: it.value for it in store.list_items(ns)}
    assert retrieved == items


def test_put_many_and_delete_many(store: MemgraphStore) -> None:
    ns = _ns()
    kvs = [(f"k{i}", {"x": i}) for i in range(3)]
    store.put_many(ns, kvs)

    assert all(store.exists(ns, k) for k, _ in kvs)

    store.delete_many(ns, [k for k, _ in kvs[:2]])
    assert not store.exists(ns, kvs[0][0])
    assert not store.exists(ns, kvs[1][0])
    assert store.exists(ns, kvs[2][0])


def test_delete_namespace(store: MemgraphStore) -> None:
    base = ("alpha", str(uuid.uuid4()))
    ns_child = base + ("child",)

    store.put(base, "a", 1)
    store.put(ns_child, "b", 2)

    # deleting parent should remove both
    store.delete_namespace(base)
    assert store.count(base) == 0
    assert store.count(ns_child) == 0


# --------------------------------------------------------------------------- #
# SEARCH tests (lexical / hybrid)
# --------------------------------------------------------------------------- #
def test_search_lexical_basic(store: MemgraphStore) -> None:
    """Plain text query should hit lexical path when no embedder configured."""
    ns = _ns()
    store.put(ns, "doc1", {"text": "cats are cute"})
    store.put(ns, "doc2", {"text": "dogs are great"})

    hits = store.search(ns, query="cats")
    assert len(hits) == 1
    assert hits[0].key == "doc1"


def test_search_list_all(store: MemgraphStore) -> None:
    """query=None should list everything in the namespace prefix."""
    ns = _ns()
    store.put(ns, "a", 1)
    store.put(ns, "b", 2)

    hits = store.search(ns, query=None)
    assert {h.key for h in hits} == {"a", "b"}


def test_search_special_chars(store: MemgraphStore) -> None:
    """Ensure special characters don't break Cypher (parameterised query)."""
    ns = _ns()
    store.put(ns, "doc", {"text": "O'Reilly"})  # contains quote

    hits = store.search(ns, query="O'Reilly")
    assert len(hits) == 1
    assert hits[0].key == "doc"
