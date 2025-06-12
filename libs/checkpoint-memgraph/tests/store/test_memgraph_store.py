import os
import time
from datetime import timedelta

import pytest

from langgraph.store.memgraph import MemgraphStore

BOLT_URI = os.getenv("MEMGRAPH_BOLT_URI", "bolt://testuser123:BiggerPassword1233@localhost:7687")


def _have_db():
    try:
        st = MemgraphStore.from_conn_string(BOLT_URI)
        st.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _have_db(), reason="Memgraph instance not reachable on localhost"
)


@pytest.fixture(scope="module")
def store():
    st = MemgraphStore.from_conn_string(
        BOLT_URI,
        ttl={"default_ttl": 0.05, "refresh_on_read": False, "sweep_interval_minutes": None},
    )
    st.setup()
    yield st
    st.close()


def test_basic_put_get(store):
    ns = ("demo", "user123")
    store.put(ns, "prefs", {"theme": "light"})
    item = store.get(ns, "prefs")
    assert item and item.value["theme"] == "light"


def test_namespace_listing(store):
    ns1 = ("alpha",)
    ns2 = ("alpha", "beta")
    store.put(ns1, "k1", {"x": 1})
    store.put(ns2, "k2", {"x": 2})
    names = set(store.list_namespaces(("alpha",)))
    assert ns1 in names and ns2 in names


def test_ttl_expiry(store):
    ns = ("tmp",)
    store.put(ns, "garbage", {"foo": "bar"})
    time.sleep(4)  # 0.05 min ≈ 3 s
    store.sweep_ttl()
    assert store.get(ns, "garbage") is None
