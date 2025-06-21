"""
Integration tests for the asynchronous AsyncMemgraphStore.
"""
from __future__ import annotations

import os
import uuid
from typing import AsyncGenerator, Tuple

import pytest

from langgraph.store.memgraph.aio import AsyncMemgraphStore
from tests.conftest import DEFAULT_MEMGRAPH_URI


@pytest.fixture
async def astore() -> AsyncGenerator[AsyncMemgraphStore, None]:
    """Function-scoped AsyncMemgraphStore fixture."""
    st = AsyncMemgraphStore.from_conn_string(DEFAULT_MEMGRAPH_URI)
    await st.setup()
    yield st
    await st.close()


def _ns() -> Tuple[str, ...]:
    return "async-tests", str(uuid.uuid4())


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