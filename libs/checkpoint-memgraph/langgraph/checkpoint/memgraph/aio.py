# langgraph/checkpoint/memgraph/aio.py
"""Asynchronous Memgraph checkpoint saver (initial functional release)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession, Transaction

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata, get_checkpoint_metadata,
)
from langgraph.checkpoint.memgraph import _ainternal
from langgraph.checkpoint.memgraph.base import BaseMemgraphSaver
from langgraph.checkpoint.memgraph.saver import MemgraphSaver  # reuse helpers


class AsyncMemgraphSaver(BaseMemgraphSaver):
    """Async variant mirroring MemgraphSaver (subset of features)."""

    lock: asyncio.Lock

    def __init__(
        self,
        conn: _ainternal.Conn,
        *,
        serde=None,
    ) -> None:
        super().__init__(serde=serde)
        self.conn = conn
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_event_loop()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(cls, bolt_uri: str, **driver_kwargs: Any) -> AsyncIterator["AsyncMemgraphSaver"]:
        driver: AsyncDriver = AsyncGraphDatabase.driver(bolt_uri, **driver_kwargs)  # type: ignore[arg-type]
        try:
            yield cls(driver)
        finally:
            await driver.close()

    # ------------------------------------------------------------------ #
    async def setup(self) -> None:
        async with _ainternal.get_connection(self.conn) as sess:
            async with sess.begin_transaction() as tx:  # type: ignore[attr-defined]
                rec = await tx.run("MATCH (m:Migration) RETURN max(m.v) AS v")
                latest = (await rec.single()).get("v") or -1  # type: ignore[arg-type]
                for v, sql in enumerate(self.MIGRATIONS):
                    if v > latest:
                        await tx.run(sql)
                        await tx.run("CREATE (:Migration {v:$v})", v=v)

    # ------------------------------------------------------------------ #
    @asynccontextmanager
    async def _cursor(self) -> AsyncIterator[Transaction]:
        async with self.lock, _ainternal.get_connection(self.conn) as sess:
            async with sess.begin_transaction() as tx:  # type: ignore[attr-defined]
                yield tx

    # ------------------------------------------------------------------ #
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ):
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parent_id = config["configurable"].get("checkpoint_id")
        blobs = self._dump_blobs(
            thread_id, checkpoint_ns, checkpoint.pop("channel_values"), new_versions
        )

        async with self._cursor() as tx:
            await tx.run(
                """
                MERGE (c:Checkpoint {
                    thread_id:$tid,
                    checkpoint_ns:$ns,
                    checkpoint_id:$cid
                })
                SET c.checkpoint = $checkpoint,
                    c.metadata = $metadata,
                    c.parent_checkpoint_id = $parent_id
                """,
                tid=thread_id,
                ns=checkpoint_ns,
                cid=checkpoint["id"],
                checkpoint=checkpoint,
                metadata=get_checkpoint_metadata(config, metadata),
                parent_id=parent_id,
            )
            for tid, ns, channel, ver, type_tag, blob in blobs:
                await tx.run(
                    """
                    MERGE (b:Blob {
                        thread_id:$tid, checkpoint_ns:$ns,
                        channel:$chan, version:$ver
                    })
                    ON CREATE SET b.type=$type_tag, b.blob=$blob
                    MERGE (c:Checkpoint {thread_id:$tid, checkpoint_ns:$ns, checkpoint_id:$cid})
                    MERGE (c)-[:HAS_BLOB]->(b)
                    """,
                    tid=tid,
                    ns=ns,
                    chan=channel,
                    ver=ver,
                    type_tag=type_tag,
                    blob=blob,
                    cid=checkpoint["id"],
                )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }
