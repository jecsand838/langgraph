from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from typing import Any, Callable, cast

import orjson
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession, AsyncTransaction
from typing_extensions import Self

from langgraph.store.base import (
    GetOp,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchOp,
)
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.memgraph.base import (
    BaseMemgraphStore,
    MemgraphIndexConfig,
    TTLConfig,
    _decode_ns_text,
    _ensure_index_config,
    _group_ops,
    _record_to_item,
    _record_to_search_item,
)

logger = logging.getLogger(__name__)


class AsyncMemgraphStore(AsyncBatchedBaseStore, BaseMemgraphStore[AsyncDriver]):
    """Asynchronous Memgraph-backed store with optional vector search."""

    __slots__ = (
        "database",
        "_deserializer",
        "index_config",
        "embeddings",
        "ttl_config",
        "_ttl_sweeper_task",
        "_ttl_stop_event",
    )
    supports_ttl: bool = True

    def __init__(
        self,
        conn: AsyncDriver,
        *,
        database: str = "memgraph",
        deserializer: Callable[[str], dict[str, Any]] | None = None,
        index: MemgraphIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> None:
        """Initialize the AsyncMemgraphStore."""
        super().__init__()
        self.conn = conn
        self.database = database
        self._deserializer = deserializer or (lambda v: orjson.loads(v))
        self.index_config = index
        if self.index_config:
            self.embeddings, self.index_config = _ensure_index_config(
                self.index_config
            )
        else:
            self.embeddings = None
        self.ttl_config = ttl
        self._ttl_sweeper_task: asyncio.Task[None] | None = None
        self._ttl_stop_event = asyncio.Event()

    @classmethod
    @asynccontextmanager
    async def from_uri(
        cls,
        uri: str,
        *,
        auth: tuple[str, str] | None = None,
        database: str = "memgraph",
        index: MemgraphIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> AsyncIterator[Self]:
        """Create a new AsyncMemgraphStore instance from a connection URI."""
        driver = AsyncGraphDatabase.driver(uri, auth=auth)
        try:
            yield cls(driver, database=database, index=index, ttl=ttl)
        finally:
            await driver.close()

    @asynccontextmanager
    async def _asession(self) -> AsyncIterator[AsyncSession]:
        """Get an async session."""
        async with self.conn.session(database=self.database) as session:
            yield session

    @asynccontextmanager
    async def _atransaction(
        self, session: AsyncSession | None = None
    ) -> AsyncIterator[AsyncTransaction]:
        """Get an async transaction."""
        if session:
            async with session.begin_transaction() as tx:
                yield tx
        else:
            async with self._asession() as s:
                async with s.begin_transaction() as tx:
                    yield tx

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of operations asynchronously."""
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        async with self._atransaction() as tx:
            if GetOp in grouped_ops:
                await self._abatch_get_ops(
                    cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]), results, tx
                )
            if SearchOp in grouped_ops:
                await self._abatch_search_ops(
                    cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp]),
                    results,
                    tx,
                )
            if ListNamespacesOp in grouped_ops:
                await self._abatch_list_namespaces_ops(
                    cast(
                        Sequence[tuple[int, ListNamespacesOp]],
                        grouped_ops[ListNamespacesOp],
                    ),
                    results,
                    tx,
                )
            if PutOp in grouped_ops:
                await self._abatch_put_ops(
                    cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp]), tx
                )

        return results

    async def _abatch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
        tx: AsyncTransaction,
    ) -> None:
        """Handle GetOp operations in a batch."""
        for query, params, namespace, items in self._get_batch_GET_ops_queries(get_ops):
            result = await tx.run(query, params)
            key_to_idx = {item["key"]: item["idx"] for item in items}
            async for record in result:
                idx = key_to_idx.get(record["key"])
                if idx is not None:
                    results[idx] = _record_to_item(
                        namespace, record.data(), loader=self._deserializer
                    )

    async def _abatch_put_ops(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
        tx: AsyncTransaction,
    ) -> None:
        """Handle PutOp operations in a batch."""
        queries, embedding_request = self._prepare_batch_PUT_queries(put_ops)

        for query, params in queries:
            await tx.run(query, params)

        if embedding_request:
            if self.embeddings is None:
                raise ValueError(
                    "Embedding configuration is required for vector operations."
                )

            query, txt_params = embedding_request
            unique_texts = sorted({param[-1] for param in txt_params})
            vectors = await self.embeddings.aembed_documents(unique_texts)
            text_to_vector = dict(zip(unique_texts, vectors))

            embedding_batch = [
                {"prefix": ns, "key": k, "embedding": text_to_vector[text]}
                for (ns, k, text) in txt_params
            ]

            await tx.run(query, {"batch": embedding_batch})

    async def _abatch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        tx: AsyncTransaction,
    ) -> None:
        """Handle SearchOp operations in a batch."""
        queries, embedding_requests = self._prepare_batch_search_queries(search_ops)

        op_idx_to_params = {
            op_idx: queries[i][1] for i, (op_idx, _) in enumerate(search_ops)
        }

        if embedding_requests and self.embeddings:
            unique_texts = sorted({text for _, text in embedding_requests})
            embeddings = await self.embeddings.aembed_documents(unique_texts)
            text_to_embedding = dict(zip(unique_texts, embeddings))

            for op_idx, text in embedding_requests:
                if op_idx in op_idx_to_params:
                    op_idx_to_params[op_idx]["embedding"] = text_to_embedding[text]

        for i, (op_idx, _) in enumerate(search_ops):
            query, params = queries[i]
            result = await tx.run(query, params)
            search_items = [
                _record_to_search_item(
                    _decode_ns_text(record["prefix"]),
                    record.data(),
                    loader=self._deserializer,
                )
                async for record in result
            ]
            results[op_idx] = search_items

    async def _abatch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
        tx: AsyncTransaction,
    ) -> None:
        """Handle ListNamespacesOp operations in a batch."""
        queries = self._get_batch_list_namespaces_queries(list_ops)
        for i, (op_idx, _) in enumerate(list_ops):
            query, params = queries[i]
            result = await tx.run(query, params)
            namespaces = [
                _decode_ns_text(row["truncated_prefix"])
                async for row in result
                if row["truncated_prefix"]
            ]
            results[op_idx] = namespaces

    async def setup(self) -> None:
        """Set up the store database asynchronously."""

        async def _get_version(tx: AsyncTransaction, table: str) -> int:
            result = await tx.run(
                """
                MERGE (m:Migration {name: $table})
                ON CREATE SET m.version = -1
                RETURN m.version AS v
                """,
                {"table": table},
            )
            record = await result.single()
            return record["v"] if record else -1

        async def _set_version(tx: AsyncTransaction, table: str, version: int) -> None:
            await tx.run(
                """
                MATCH (m:Migration {name: $table})
                SET m.version = $version
                """,
                {"table": table, "version": version},
            )

        async with self._asession() as session:
            # Main migrations
            async with session.begin_transaction() as tx:
                version = await _get_version(tx, "store_migrations")
            for v, cypher in enumerate(
                self.MIGRATIONS[version + 1 :], start=version + 1
            ):
                try:
                    await session.run(cypher)
                    async with session.begin_transaction() as tx:
                        await _set_version(tx, "store_migrations", v)
                except Exception as e:
                    logger.error(
                        f"Failed to apply migration {v}.\\nCypher={cypher}\\nError={e}"
                    )
                    raise

            # Vector migrations
            if self.index_config:
                async with session.begin_transaction() as tx:
                    version = await _get_version(tx, "vector_migrations")
                for v, migration in enumerate(
                    self.VECTOR_MIGRATIONS[version + 1 :], start=version + 1
                ):
                    if migration.condition and not migration.condition(self):
                        continue
                    cypher = migration.cypher
                    params = {}
                    if migration.params:
                        params = {
                            k: val(self) if callable(val) else val
                            for k, val in migration.params.items()
                        }
                    final_cypher = cypher.format(**params)
                    try:
                        await session.run(final_cypher)
                        async with session.begin_transaction() as tx:
                            await _set_version(tx, "vector_migrations", v)
                    except Exception as e:
                        logger.error(
                            f"Failed to apply vector migration {v}.\\nCypher={final_cypher}\\nError={e}"
                        )
                        raise

    async def sweep_ttl(self) -> int:
        """Delete expired store items based on TTL."""
        async with self._asession() as session:
            result = await session.run(
                """
                MATCH (n:StoreItem)
                WHERE n.expires_at IS NOT NULL AND n.expires_at < localdatetime()
                DETACH DELETE n
                RETURN count(n) as deleted_count
                """
            )
            record = await result.single()
            return record.data()["deleted_count"] if record else 0

    async def start_ttl_sweeper(
        self, sweep_interval_minutes: int | None = None
    ) -> asyncio.Task[None]:
        """Periodically delete expired store items based on TTL."""
        if not self.ttl_config:
            return asyncio.create_task(asyncio.sleep(0))

        if self._ttl_sweeper_task and not self._ttl_sweeper_task.done():
            return self._ttl_sweeper_task

        self._ttl_stop_event.clear()
        interval = float(
            sweep_interval_minutes or self.ttl_config.get("sweep_interval_minutes") or 5
        )
        logger.info(f"Starting store TTL sweeper with interval {interval} minutes")

        async def _sweep_loop() -> None:
            while not self._ttl_stop_event.is_set():
                try:
                    # Wait for the given interval or until the stop event is set
                    await asyncio.wait_for(
                        self._ttl_stop_event.wait(), timeout=interval * 60
                    )
                    # If wait finishes without timeout, it means stop event was set
                    break
                except asyncio.TimeoutError:
                    # This is the normal path, timeout occurred, so we sweep
                    pass
                except asyncio.CancelledError:
                    # Task was cancelled
                    break

                if self._ttl_stop_event.is_set():
                    break

                try:
                    expired_items = await self.sweep_ttl()
                    if expired_items > 0:
                        logger.info(f"Store swept {expired_items} expired items")
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.exception("Store TTL sweep iteration failed", exc_info=exc)

        task = asyncio.create_task(_sweep_loop())
        task.set_name("ttl_sweeper")
        self._ttl_sweeper_task = task
        return task

    async def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        """Stop the TTL sweeper task if it's running."""
        if not self._ttl_sweeper_task or self._ttl_sweeper_task.done():
            return True

        logger.info("Stopping TTL sweeper task")
        self._ttl_stop_event.set()
        try:
            await asyncio.wait_for(self._ttl_sweeper_task, timeout=timeout)
            logger.info("TTL sweeper task stopped gracefully.")
            self._ttl_sweeper_task = None
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out waiting for TTL sweeper task to stop. Cancelling."
            )
            self._ttl_sweeper_task.cancel()
            try:
                # Wait a moment for cancellation to propagate
                await self._ttl_sweeper_task
            except asyncio.CancelledError:
                pass
            self._ttl_sweeper_task = None
            return False
        except asyncio.CancelledError:
            # This can happen if the task is cancelled externally
            logger.info("TTL sweeper task was already cancelled during stop.")
            self._ttl_sweeper_task = None
            return True

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop_ttl_sweeper(timeout=2.0)