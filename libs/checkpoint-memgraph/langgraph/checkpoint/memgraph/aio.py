
"""Asynchronous Memgraph checkpoint saver (feature‑parity with sync version)."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Sequence

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession
from neo4j.exceptions import ClientError

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.memgraph import _ainternal
from langgraph.checkpoint.memgraph.base import BaseMemgraphSaver
from langgraph.checkpoint.memgraph.saver import _is_dup_ddl  # reuse helper


class AsyncMemgraphSaver(BaseMemgraphSaver):
    """Async variant mirroring MemgraphSaver (subset of features)."""

    lock: asyncio.Lock

    # ------------------------------------------------------------------ #
    # constructor
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        conn: _ainternal.Conn,
        *,
        serde=None,
    ) -> None:
        super().__init__(serde=serde)
        self.conn = conn
        self.lock = asyncio.Lock()
        # keep a reference to the running loop for blob helpers
        self.loop = asyncio.get_event_loop()

    # ------------------------------------------------------------------ #
    # factory
    # ------------------------------------------------------------------ #
    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls, bolt_uri: str, **driver_kwargs: Any
    ) -> AsyncIterator["AsyncMemgraphSaver"]:
        driver: AsyncDriver = AsyncGraphDatabase.driver(  # type: ignore[arg-type]
            bolt_uri, **driver_kwargs
        )
        try:
            yield cls(driver)
        finally:
            await driver.close()

    # ------------------------------------------------------------------ #
    # Schema initialisation
    # ------------------------------------------------------------------ #
    async def setup(self) -> None:
        """
        Apply schema migrations.

        Each DDL statement is executed in its own implicit transaction because
        Memgraph disallows constraint/index creation inside an explicit multi‑
        command transaction.
        """
        async with _ainternal.get_connection(self.conn) as sess:

            async def run_safe(cypher: str) -> None:
                try:
                    await sess.run(cypher)
                except ClientError as exc:  # pragma: no cover
                    if not _is_dup_ddl(exc):
                        raise

            # baseline marker
            await sess.run("MERGE (:Migration {v: -1})")
            rec = await sess.run("MATCH (m:Migration) RETURN max(m.v) AS v")
            latest = (await rec.single())["v"] or -1

            for v, mig in enumerate(self.MIGRATIONS):
                if v <= latest:
                    continue
                await run_safe(mig)
                await sess.run("CREATE (:Migration {v:$v})", v=v)

    # ------------------------------------------------------------------ #
    # internal context manager yielding a TX guarded by an asyncio.Lock
    # ------------------------------------------------------------------ #
    @asynccontextmanager
    async def _cursor(self) -> AsyncIterator[AsyncSession]:
        async with self.lock, _ainternal.get_connection(self.conn) as sess:
            async with sess.begin_transaction() as tx:  # type: ignore[attr-defined]
                yield tx

    # ------------------------------------------------------------------ #
    # Public API
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

    # ------------------------------------------------------------------ #
    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        if not writes:
            return

        tid = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        cid = config["configurable"]["checkpoint_id"]

        rows = self._dump_writes(tid, ns, cid, task_id, task_path, writes)

        async with self._cursor() as tx:
            for (
                tid,
                ns,
                cid,
                t_id,
                t_path,
                idx,
                channel,
                type_tag,
                blob,
            ) in rows:
                await tx.run(
                    """
                    MERGE (w:Write {
                        thread_id:$tid, checkpoint_ns:$ns, checkpoint_id:$cid,
                        task_id:$task_id, idx:$idx
                    })
                    SET w.task_path=$task_path,
                        w.channel=$channel,
                        w.type=$type_tag,
                        w.blob=$blob
                    WITH w
                    MATCH (c:Checkpoint {
                        thread_id:$tid, checkpoint_ns:$ns, checkpoint_id:$cid
                    })
                    MERGE (c)-[:HAS_WRITE]->(w)
                    """,
                    tid=tid,
                    ns=ns,
                    cid=cid,
                    task_id=t_id,
                    idx=idx,
                    task_path=t_path,
                    channel=channel,
                    type_tag=type_tag,
                    blob=blob,
                )

    # ------------------------------------------------------------------ #
    # retrieval
    # ------------------------------------------------------------------ #
    def _build_tuple(
        self,
        chk_node,
        blobs: list[tuple[str, str, bytes]],
        writes: list[tuple[str, str, str, bytes]],
    ) -> CheckpointTuple:
        ckpt = chk_node["checkpoint"]
        ckpt["channel_values"] = self._load_blobs(blobs)
        pend = self._load_writes(writes)
        return CheckpointTuple(
            {
                "configurable": {
                    "thread_id": chk_node["thread_id"],
                    "checkpoint_ns": chk_node["checkpoint_ns"],
                    "checkpoint_id": chk_node["checkpoint_id"],
                }
            },
            ckpt,
            chk_node["metadata"],
            (
                {
                    "configurable": {
                        "thread_id": chk_node["thread_id"],
                        "checkpoint_ns": chk_node["checkpoint_ns"],
                        "checkpoint_id": chk_node.get("parent_checkpoint_id"),
                    }
                }
                if chk_node.get("parent_checkpoint_id")
                else None
            ),
            pend,
        )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        where, params = self._search_where(config, None, None)
        cypher = f"""
        MATCH (c:Checkpoint) {where}
        OPTIONAL MATCH (c)-[:HAS_BLOB]->(b:Blob)
        WITH c, collect([b.channel, b.type, b.blob]) AS blobs
        OPTIONAL MATCH (c)-[:HAS_WRITE]->(w:Write)
        WITH c, blobs, collect([w.task_id, w.channel, w.type, w.blob]) AS writes
        RETURN c AS chk, blobs, writes
        ORDER BY c.checkpoint_id DESC
        LIMIT 1
        """
        async with _ainternal.get_connection(self.conn) as sess:
            rec = await sess.run(cypher, **params)
            row = await rec.single()
            if not row:
                return None
            return self._build_tuple(row["chk"], row["blobs"], row["writes"])

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ):
        where, params = self._search_where(config, filter, before)
        cypher = f"""
        MATCH (c:Checkpoint) {where}
        OPTIONAL MATCH (c)-[:HAS_BLOB]->(b:Blob)
        WITH c, collect([b.channel, b.type, b.blob]) AS blobs
        OPTIONAL MATCH (c)-[:HAS_WRITE]->(w:Write)
        WITH c, blobs, collect([w.task_id, w.channel, w.type, w.blob]) AS writes
        RETURN c AS chk, blobs, writes
        ORDER BY c.checkpoint_id DESC
        """
        if limit:
            cypher += " LIMIT $limit"
            params["limit"] = limit

        async with _ainternal.get_connection(self.conn) as sess:
            res = await sess.run(cypher, **params)
            async for row in res:
                yield self._build_tuple(row["chk"], row["blobs"], row["writes"])

    # ------------------------------------------------------------------ #
    async def adelete_thread(self, thread_id: str) -> None:
        async with self._cursor() as tx:
            await tx.run(
                """
                MATCH (n)
                WHERE (n:Checkpoint OR n:Blob OR n:Write) AND n.thread_id = $tid
                DETACH DELETE n
                """,
                tid=str(thread_id),
            )
