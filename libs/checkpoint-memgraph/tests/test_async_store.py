from __future__ import annotations

import asyncio
import itertools
import sys
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings

from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchOp,
)
from langgraph.store.memgraph.aio import AsyncMemgraphStore
from tests.conftest import DEFAULT_MEMGRAPH_URI
from tests.embed_test_utils import CharacterEmbeddings

TTL_SECONDS = 6
TTL_MINUTES = TTL_SECONDS / 60


@pytest.fixture(scope="function")
async def store() -> AsyncIterator[AsyncMemgraphStore]:
    if sys.version_info < (3, 10):
        pytest.skip("Async Memgraph tests require Python 3.10+")

    ttl_config = {
        "default_ttl": TTL_MINUTES,
        "refresh_on_read": True,
        "sweep_interval_minutes": TTL_MINUTES / 2,
    }

    async with AsyncMemgraphStore.from_conn_string(
        DEFAULT_MEMGRAPH_URI, ttl=ttl_config
    ) as store:
        await store._clean()
        await store.setup()
        # Test idempotency
        await store.setup()

        await store.start_ttl_sweeper()
        yield store
        await store.stop_ttl_sweeper()
        await store._clean()


async def test_no_running_loop(store: AsyncMemgraphStore) -> None:
    with pytest.raises(asyncio.InvalidStateError):
        store.put(("foo", "bar"), {"val": "baz"})
    with pytest.raises(asyncio.InvalidStateError):
        store.get(("foo", "bar"))
    with pytest.raises(asyncio.InvalidStateError):
        store.delete(("foo", "bar"))
    with pytest.raises(asyncio.InvalidStateError):
        store.search(("foo",))
    with pytest.raises(asyncio.InvalidStateError):
        store.list_namespaces(prefix=("foo",))
    with pytest.raises(asyncio.InvalidStateError):
        store.batch([PutOp(namespace=("foo", "bar"), key="default", value={"val": "baz"})])

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(store.put, ("foo", "bar"), {"val": "baz"})
        result = await asyncio.wrap_future(future)
        assert result is None
        future = executor.submit(store.get, ("foo", "bar"))
        result = await asyncio.wrap_future(future)
        assert result.value == {"val": "baz"}
        result = await asyncio.wrap_future(
            executor.submit(store.list_namespaces, prefix=("foo",))
        )


async def test_large_batches(request: Any, store: AsyncMemgraphStore) -> None:
    N = 100
    M = 10

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for m in range(M):
            for i in range(N):
                futures += [
                    executor.submit(
                        store.put,
                        ("test", "foo", "bar", "baz", str(m % 2)),
                        {"foo": "bar" + str(i)},
                        key=f"key{i}",
                    ),
                    executor.submit(
                        store.get,
                        ("test", "foo", "bar", "baz", str(m % 2)),
                        f"key{i}",
                    ),
                    executor.submit(
                        store.list_namespaces,
                        prefix=None,
                    ),
                    executor.submit(
                        store.search,
                        ("test",),
                    ),
                    executor.submit(
                        store.put,
                        ("test", "foo", "bar", "baz", str(m % 2)),
                        {"foo": "bar" + str(i)},
                        key=f"key{i}",
                    ),
                    executor.submit(
                        store.delete,
                        ("test", "foo", "bar", "baz", str(m % 2)),
                        f"key{i}",
                    ),
                ]

        results = await asyncio.gather(
            *(asyncio.wrap_future(future) for future in futures)
        )
    assert len(results) == M * N * 6


async def test_large_batches_async(store: AsyncMemgraphStore) -> None:
    N = 1000
    M = 10
    coros = []
    for m in range(M):
        for i in range(N):
            coros.append(
                store.aput(
                    ("test", "foo", "bar", "baz", str(m % 2)),
                    value={"foo": "bar" + str(i)},
                    key=f"key{i}",
                )
            )
            coros.append(
                store.aget(
                    ("test", "foo", "bar", "baz", str(m % 2)),
                    f"key{i}",
                )
            )
            coros.append(
                store.alist_namespaces(
                    prefix=None,
                )
            )
            coros.append(
                store.asearch(
                    ("test",),
                )
            )
            coros.append(
                store.aput(
                    ("test", "foo", "bar", "baz", str(m % 2)),
                    value={"foo": "bar" + str(i)},
                    key=f"key{i}",
                )
            )
            coros.append(
                store.adelete(
                    ("test", "foo", "bar", "baz", str(m % 2)),
                    f"key{i}",
                )
            )

    results = await asyncio.gather(*coros)
    assert len(results) == M * N * 6


async def test_abatch_order(store: AsyncMemgraphStore) -> None:
    # Setup test data
    await store.aput(("test", "foo"), {"data": "value1"}, "key1")
    await store.aput(("test", "bar"), {"data": "value2"}, "key2")

    ops = [
        GetOp(namespace=("test", "foo"), key="key1"),
        PutOp(namespace=("test", "bar"), key="key2", value={"data": "value2"}),
        SearchOp(namespace_prefix=("test",)),
        ListNamespacesOp(),
        GetOp(namespace=("test",), key="key3"),
    ]

    results = await store.abatch(ops)
    assert len(results) == 5
    assert isinstance(results[0], Item)
    assert isinstance(results[0].value, dict)
    assert results[0].value == {"data": "value1"}
    assert results[0].key == "key1"
    assert results[1] is None
    assert isinstance(results[2], list)
    assert len(results[2]) == 2
    assert isinstance(results[3], list)
    assert ("test", "foo") in results[3] and ("test", "bar") in results[3]
    assert results[4] is None

    ops_reordered = [
        SearchOp(namespace_prefix=("test",), limit=5, offset=0),
        GetOp(namespace=("test", "bar"), key="key2"),
        ListNamespacesOp(limit=5, offset=0),
        PutOp(namespace=("test",), key="key3", value={"data": "value3"}),
        GetOp(namespace=("test", "foo"), key="key1"),
    ]

    results_reordered = await store.abatch(ops_reordered)
    assert len(results_reordered) == 5
    assert isinstance(results_reordered[0], list)
    assert len(results_reordered[0]) == 2
    assert isinstance(results_reordered[1], Item)
    assert results_reordered[1].value == {"data": "value2"}
    assert results_reordered[1].key == "key2"
    assert isinstance(results_reordered[2], list)
    assert ("test", "foo") in results_reordered[2] and (
        "test",
        "bar",
    ) in results_reordered[2]
    assert results_reordered[3] is None
    assert isinstance(results_reordered[4], Item)
    assert results_reordered[4].value == {"data": "value1"}
    assert results_reordered[4].key == "key1"


async def test_batch_get_ops(store: AsyncMemgraphStore) -> None:
    # Setup test data
    await store.aput(("test",), {"data": "value1"}, "key1")
    await store.aput(("test",), {"data": "value2"}, "key2")

    ops = [
        GetOp(namespace=("test",), key="key1"),
        GetOp(namespace=("test",), key="key2"),
        GetOp(namespace=("test",), key="key3"),
    ]

    results = await store.abatch(ops)

    assert len(results) == 3
    assert results[0] is not None
    assert results[1] is not None
    assert results[2] is None
    assert results[0].key == "key1"
    assert results[1].key == "key2"


async def test_batch_put_ops(store: AsyncMemgraphStore) -> None:
    ops = [
        PutOp(namespace=("test",), key="key1", value={"data": "value1"}),
        PutOp(namespace=("test",), key="key2", value={"data": "value2"}),
        PutOp(namespace=("test",), key="key3", value=None),
    ]

    results = await store.abatch(ops)

    assert len(results) == 3
    assert all(result is None for result in results)

    # Verify the puts worked
    items = await store.asearch(("test",), limit=10)
    assert len(items) == 2


async def test_batch_search_ops(store: AsyncMemgraphStore) -> None:
    # Setup test data
    await store.aput(("test", "foo"), {"data": "value1"}, "key1")
    await store.aput(("test", "bar"), {"data": "value2"}, "key2")

    ops = [
        SearchOp(namespace_prefix=("test",), filter={"data": "value1"}),
        SearchOp(namespace_prefix=("test",), limit=5, offset=0),
    ]

    results = await store.abatch(ops)

    assert len(results) == 2
    assert len(results[0]) == 1
    assert len(results[1]) == 2


async def test_batch_list_namespaces_ops(store: AsyncMemgraphStore) -> None:
    # Setup test data
    await store.aput(("test", "namespace1"), {"data": "value1"}, "key1")
    await store.aput(("test", "namespace2"), {"data": "value2"}, "key2")

    ops = [ListNamespacesOp(limit=10, offset=0)]

    results = await store.abatch(ops)

    assert len(results) == 1
    assert len(results[0]) == 2
    assert ("test", "namespace1") in results[0]
    assert ("test", "namespace2") in results[0]


@asynccontextmanager
async def _create_vector_store(
    distance_type: str,
    fake_embeddings: CharacterEmbeddings,
    text_fields: list[str] | None = None,
) -> AsyncIterator[AsyncMemgraphStore]:
    """Create a store with vector search enabled."""
    if sys.version_info < (3, 10):
        pytest.skip("Async Memgraph tests require Python 3.10+")

    index_config = {
        "dims": fake_embeddings.dims,
        "embed": fake_embeddings,
        "distance_type": distance_type,
        "fields": text_fields,
    }

    async with AsyncMemgraphStore.from_conn_string(
        DEFAULT_MEMGRAPH_URI,
        index=index_config,
    ) as store:
        await store._clean()
        await store.setup()
        yield store
        await store._clean()


@pytest.fixture(
    scope="function",
    params=["l2", "inner_product", "cosine"],
)
async def vector_store(
    request,
    fake_embeddings: CharacterEmbeddings,
) -> AsyncIterator[AsyncMemgraphStore]:
    """Create a store with vector search enabled."""
    distance_type = request.param
    async with _create_vector_store(distance_type, fake_embeddings) as store:
        yield store


async def test_vector_store_initialization(
    vector_store: AsyncMemgraphStore, fake_embeddings: CharacterEmbeddings
) -> None:
    """Test store initialization with embedding config."""
    assert vector_store.index_config is not None
    assert vector_store.index_config["dims"] == fake_embeddings.dims
    if isinstance(vector_store.embeddings, Embeddings):
        assert vector_store.embeddings == fake_embeddings


async def test_vector_insert_with_auto_embedding(
    vector_store: AsyncMemgraphStore,
) -> None:
    """Test inserting items that get auto-embedded."""
    docs = [
        ("doc1", {"text": "short text"}),
        ("doc2", {"text": "longer text document"}),
        ("doc3", {"text": "longest text document here"}),
    ]

    for key, value in docs:
        await vector_store.aput(("test",), value, key)

    results = await vector_store.asearch(("test",), query="long text")
    assert len(results) > 0

    doc_order = [r.key for r in results]
    assert "doc2" in doc_order
    assert "doc3" in doc_order


async def test_vector_update_with_embedding(vector_store: AsyncMemgraphStore) -> None:
    """Test that updating items properly updates their embeddings."""
    await vector_store.aput(("test",), {"text": "zany zebra Xerxes"}, "doc1")
    await vector_store.aput(("test",), {"text": "something about dogs"}, "doc2")
    await vector_store.aput(("test",), {"text": "text about birds"}, "doc3")

    results_initial = await vector_store.asearch(("test",), query="Zany Xerxes")
    assert len(results_initial) > 0
    assert results_initial[0].key == "doc1"
    initial_score = results_initial[0].score

    await vector_store.aput(("test",), {"text": "new text about dogs"}, "doc1")

    results_after = await vector_store.asearch(("test",), query="Zany Xerxes")
    after_score = next((r.score for r in results_after if r.key == "doc1"), 0.0)
    assert after_score < initial_score

    results_new = await vector_store.asearch(("test",), query="new text about dogs")
    for r in results_new:
        if r.key == "doc1":
            assert r.score > after_score

    # Don't index this one
    await vector_store.aput(
        ("test",), {"text": "new text about dogs"}, "doc4", index=False
    )
    results_new = await vector_store.asearch(
        ("test",), query="new text about dogs", limit=3
    )
    assert not any(r.key == "doc4" for r in results_new)


async def test_vector_search_with_filters(vector_store: AsyncMemgraphStore) -> None:
    """Test combining vector search with filters."""
    docs = [
        ("doc1", {"text": "red apple", "color": "red", "score": 4.5}),
        ("doc2", {"text": "red car", "color": "red", "score": 3.0}),
        ("doc3", {"text": "green apple", "color": "green", "score": 4.0}),
        ("doc4", {"text": "blue car", "color": "blue", "score": 3.5}),
    ]

    for key, value in docs:
        await vector_store.aput(("test",), value, key)

    results = await vector_store.asearch(
        ("test",), query="apple", filter={"color": "red"}
    )
    assert len(results) == 1
    assert results[0].key == "doc1"

    results = await vector_store.asearch(
        ("test",), query="car", filter={"color": "red"}
    )
    assert len(results) == 1
    assert results[0].key == "doc2"

    results_gt = await vector_store.asearch(
        ("test",), query="bbbbluuu", filter={"score": {"$gt": 3.2}}
    )
    assert {r.key for r in results_gt} == {"doc1", "doc3", "doc4"}

    results_gte = await vector_store.asearch(
        ("test",),
        query="apple",
        filter={"score": {"$gte": 4.0}, "color": "green"},
    )
    assert len(results_gte) == 1
    assert results_gte[0].key == "doc3"


async def test_vector_search_pagination(vector_store: AsyncMemgraphStore) -> None:
    """Test pagination with vector search."""
    for i in range(5):
        await vector_store.aput(
            ("test",), {"text": f"test document number {i}"}, f"doc{i}"
        )

    results_page1 = await vector_store.asearch(("test",), query="test", limit=2)
    results_page2 = await vector_store.asearch(
        ("test",), query="test", limit=2, offset=2
    )

    assert len(results_page1) == 2
    assert len(results_page2) == 2
    assert results_page1[0].key != results_page2[0].key

    all_results = await vector_store.asearch(("test",), query="test", limit=10)
    assert len(all_results) == 5


async def test_vector_search_edge_cases(vector_store: AsyncMemgraphStore) -> None:
    """Test edge cases in vector search."""
    await vector_store.aput(("test",), {"text": "test document"}, "doc1")

    perfect_match = await vector_store.asearch(("test",), query="text test document")
    perfect_score = perfect_match[0].score

    results = await vector_store.asearch(("test",), query="")
    assert len(results) == 0

    results = await vector_store.asearch(("test",), query=None)
    assert len(results) == 1
    assert results[0].score is None

    long_query = "foo " * 100
    results = await vector_store.asearch(("test",), query=long_query)
    assert len(results) == 1
    assert results[0].score < perfect_score

    special_query = "test!@#$%^&*()"
    results = await vector_store.asearch(("test",), query=special_query)
    assert len(results) == 1
    assert results[0].score < perfect_score


@pytest.mark.parametrize(
    "distance_type",
    ["cosine", "inner_product", "l2"],
)
async def test_embed_with_path(
    request: Any,
    fake_embeddings: CharacterEmbeddings,
    distance_type: str,
) -> None:
    """Test vector search with specific text fields."""
    async with _create_vector_store(
        distance_type,
        fake_embeddings,
        text_fields=["key1", "key3"],
    ) as store:
        doc1 = {
            "key1": "xxx",
            "key2": "yyy",
            "key3": "zzz",
        }
        doc2 = {
            "key0": "uuu",
            "key1": "vvv",
            "key2": "www",
            "key3": "xxx",
        }
        await store.aput(("test",), doc1, "doc1")
        await store.aput(("test",), doc2, "doc2")

        # doc2.key3 and doc1.key1 both would have the highest score
        results = await store.asearch(("test",), query="xxx")
        assert len(results) == 2
        assert results[0].key != results[1].key
        ascore = results[0].score
        bscore = results[1].score
        assert ascore == pytest.approx(bscore, abs=1e-3)

        # Un-indexed - will have low results for both.
        results = await store.asearch(("test",), query="www")
        assert len(results) == 2
        assert results[0].score < ascore
        assert results[1].score < ascore


@pytest.mark.parametrize(
    "distance_type",
    ["cosine", "inner_product", "l2"],
)
async def test_search_sorting(
    request: Any,
    fake_embeddings: CharacterEmbeddings,
    distance_type: str,
) -> None:
    """Test operation-level field configuration for vector search."""
    async with _create_vector_store(
        distance_type,
        fake_embeddings,
        text_fields=["key1"],
    ) as store:
        amatch = {
            "key1": "mmm",
        }

        await store.aput(("test", "M"), amatch, "M")
        N = 10
        for i in range(N):
            await store.aput(("test", "A"), {"key1": "no"}, f"A{i}")
        for i in range(N):
            await store.aput(("test", "Z"), {"key1": "no"}, f"Z{i}")

        results = await store.asearch(("test",), query="mmm", limit=10)
        assert len(results) == 10
        assert len(set(r.key for r in results)) == 10
        assert results[0].key == "M"
        assert results[0].score > results[1].score


async def test_store_ttl(store: AsyncMemgraphStore):
    ns = ("foo",)
    await store.start_ttl_sweeper()
    await store.aput(
        ns,
        key="item1",
        value={"foo": "bar"},
        ttl=TTL_MINUTES,
    )
    await asyncio.sleep(TTL_SECONDS - 2)
    res = await store.aget(ns, key="item1", refresh_ttl=True)
    assert res is not None
    await asyncio.sleep(TTL_SECONDS - 2)
    results = await store.asearch(ns, query="foo", refresh_ttl=True)
    assert len(results) == 1
    await asyncio.sleep(TTL_SECONDS - 2)
    res = await store.aget(ns, key="item1", refresh_ttl=False)
    assert res is not None
    await asyncio.sleep(TTL_SECONDS + 1)
    # TTL sweeper runs every TTL_MINUTES/2, which is TTL_SECONDS/2.
    # Total sleep time is (TTL_S-2)*3 + TTL_S+1 ~= 4*TTL_S - 5.
    # The sweeper should have run.
    results = await store.asearch(ns, query="bar", refresh_ttl=False)
    assert len(results) == 0