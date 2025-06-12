# libs/checkpoint-memgraph/tests/store/test_async_memgraph_store.py
"""
Integration tests for the asynchronous AsyncMemgraphStore.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from typing import Iterable, Tuple

import pytest

from langgraph.store.memgraph.aio import AsyncMemgraphStore

# --------------------------------------------------------------------------- #
BOLT_URI = os.getenv(
    "MEMGRAPH_BOLT_URI", "bolt://testuser123:BiggerPassword1233@localhost:7687"
)


def _have_db() -> bool:
    try:
        st = AsyncMemgraphStore.from_conn_string(BOLT_URI)
        asyncio.get_event_loop().run_until_complete(st.close())
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _have_db(), reason="Memgraph instance not reachable on localhost"
)


# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def event_loop() -> Iterable[asyncio.AbstractEventLoop]:  # pytest‑asyncio default
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def astore() -> Iterable[AsyncMemgraphStore]:
    st = AsyncMemgraphStore.from_conn_string(BOLT_URI)
    await st.setup()
    yield st
    await st.close()


def _ns() -> Tuple[str, ...]:
    return ("async-tests", str(uuid.uuid4()))


# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_async_put_get(astore: AsyncMemgraphStore) -> None:
    ns = _ns()
    await astore.aput(ns, "k", {"y": 7})
    item = await astore.aget(ns, "k")
    assert item and item.value == {"y": 7}


@pytest.mark.asyncio
async def test_async_exists_count_delete(astore: AsyncMemgraphStore) -> None:
    ns = _ns()
    await astore.aput(ns, "a", 1)
    await astore.aput(ns, "b", 2)

    assert await astore.aexists(ns, "a") is True
    assert await astore.acount(ns) == 2

    await astore.adelete_namespace(ns)
    assert await astore.acount(ns) == 0
