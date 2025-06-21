# libs/checkpoint-memgraph/tests/store/test_memgraph_store_extended.py
"""
Extended integration tests for the synchronous MemgraphStore.

These tests cover the additional helper/utility methods introduced to reach
parity with the Postgres‑backed implementation (exists, count, list_keys,
list_items, bulk‑writes, namespace deletion, etc.).
"""
from __future__ import annotations

import os
import uuid
from typing import Iterable, Tuple

import pytest

from langgraph.store.memgraph import MemgraphStore

# --------------------------------------------------------------------------- #
BOLT_URI = os.getenv(
    "MEMGRAPH_BOLT_URI", "bolt://testuser123:BiggerPassword1233@localhost:7687"
)


def _have_db() -> bool:
    """Quick connectivity probe so CI can gracefully skip when DB unavailable."""
    try:
        store = MemgraphStore.from_conn_string(BOLT_URI)
        store.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _have_db(), reason="Memgraph instance not reachable on localhost"
)


# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def store() -> Iterable[MemgraphStore]:
    """
    Provide an isolated MemgraphStore instance for the entire module.

    A fresh/random namespace prefix is generated in each test to avoid clashes
    when tests are run concurrently across different environments.
    """
    st = MemgraphStore.from_conn_string(BOLT_URI)
    st.setup()
    yield st
    st.close()


# --------------------------------------------------------------------------- #
def _ns() -> Tuple[str, ...]:
    """Generate a unique namespace for every test call."""
    return ("tests", str(uuid.uuid4()))


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

    keys = set(store.list_keys(ns))
    assert keys == set(items)

    retrieved = {it.key: it.value for it in store.list_items(ns)}
    assert retrieved == items


def test_put_many_and_delete_many(store: MemgraphStore) -> None:
    ns = _ns()
    kvs = [(f"k{i}", {"x": i}) for i in range(3)]
    store.put_many(ns, kvs)

    assert all(store.exists(ns, k) for k, _ in kvs)

    store.delete_many(ns, [k for k, _ in kvs[:2]])
    assert store.exists(ns, kvs[0][0]) is False
    assert store.exists(ns, kvs[1][0]) is False
    assert store.exists(ns, kvs[2][0]) is True


def test_delete_namespace(store: MemgraphStore) -> None:
    base = ("alpha", str(uuid.uuid4()))
    ns_child = base + ("child",)

    store.put(base, "a", 1)
    store.put(ns_child, "b", 2)

    # delete parent should remove both
    store.delete_namespace(base)
    assert store.count(base) == 0
    assert store.count(ns_child) == 0


def test_search_lexical(store: MemgraphStore) -> None:
    ns = _ns()
    store.put(ns, "doc1", {"text": "cats are cute"})
    store.put(ns, "doc2", {"text": "dogs are great"})

    hits = store.search(ns, query="cats")
    assert len(hits) == 1
    assert hits[0].key == "doc1"
