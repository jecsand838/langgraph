"""Asynchronous Memgraph checkpoint saver (feature‑parity with sync version).

Key fixes (2025‑06‑21)
──────────────────────
• Added `_require_checkpoint_id` guard in `aput_writes`; raises a clear
  `ValueError` if a caller forgets to persist a checkpoint first.
• Re‑worked `aput_writes` Cypher so the MERGE of `Write` and creation of the
  relationship to its parent `Checkpoint` happen in a *single* statement,
  eliminating the Memgraph.ExecutionException seen with `WITH … MATCH`.
• Added debug‑level logging statements mirroring the synchronous saver.
• Relies on improved (de)serialization helpers in BaseMemgraphSaver that skip
  NULL placeholder rows, eliminating the “Unknown serialization type: None”
  runtime error.
"""
from __future__ import annotations

import asyncio
import logging
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

logger = logging.getLogger(__name__)


class AsyncMemgraphSaver(BaseMemgraphSaver):
    """Async variant that offers the same public API as `MemgraphSaver`."""

    lock: asyncio.Lock

    # ─────────────────────────────────────────────────────────────────── #
    # Construction / factory helpers
    # ─────────────────────────────────────────────────────────────────── #
    def __init__(self, conn: _ainternal.Conn, *, serde=None) -> None:
        super().__init__(serde=serde)
        self.conn = conn
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_event_loop()

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

    # ─────────────────────────────────────────────────────────────────── #
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────── #
    @staticmethod
    def _require_checkpoint_id(config: RunnableConfig) -> str:
        try:
            return config["configurable"]["checkpoint_id"]
        except KeyError as exc:  # pragma: no cover
            raise ValueError(
                "AsyncMemgraphSaver.aput_writes requires "
                "`config['configurable']['checkpoint_id']` – "
                "call `await saver.aput(...)` first."
            ) from exc

    # ─────────────────────────────────────────────────────────────────── #
    # Schema initialisation
    # ─────────────────────────────────────────────────────────────────── #
    async def setup(self) -> None:
        async with _ainternal.get_connection(self.conn) as sess:

            async def run_safe(cypher: str) -> None:
                try:
                    await sess.run(cypher)
                except ClientError as exc:  # pragma: no cover
                    if not _is_dup_ddl(exc):
                        raise

            await sess.run("MERGE (:Migration {v: -1})")
            rec = await sess.run("MATCH (m:Migration) RETURN max(m.v) AS v")
            latest = (await rec.single())["v"] or -1

            for v, mig in enumerate(self.MIGRATIONS):
                if v <= latest:
                    continue
                await run_safe(mig)
                await sess.run("CREATE (:Migration {v:$v})", v=v)

    # ─────────────────────────────────────────────────────────────────── #
    # Internal cursor helper
    # ─────────────────────────────────────────────────────────────────── #
    @asynccontextmanager
    async def _cursor(self) -> AsyncIterator[AsyncSession]:
        async with self.lock, _ainternal.get_connection(self.conn) as sess:
            async with sess.begin_transaction() as tx:  # type: ignore[attr-defined]
                yield tx

    # ─────────────────────────────────────────────────────────────────── #
    # Public API – checkpoint persistence
    # ─────────────────────────────────────────────────────────────────── #
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
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
                    MERGE (c:Checkpoint {
                        thread_id:$tid, checkpoint_ns:$ns, checkpoint_id:$cid
                    })
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

        logger.debug(
            "AsyncMemgraphSaver: saved checkpoint %s for thread %s",
            checkpoint["id"],
            thread_id,
        )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    # ─────────────────────────────────────────────────────────────────── #
    # Public API – intermediate writes                                 FIX
    # ─────────────────────────────────────────────────────────────────── #
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
        cid = self._require_checkpoint_id(config)

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
                # Single MATCH‑MERGE statement avoids WITH … MATCH (which caused
                # Memgraph.ExecutionException in certain server versions).
                await tx.run(
                    """
                    MATCH (c:Checkpoint {
                        thread_id:$tid, checkpoint_ns:$ns, checkpoint_id:$cid
                    })
                    MERGE (w:Write {
                        thread_id:$tid, checkpoint_ns:$ns, checkpoint_id:$cid,
                        task_id:$task_id, idx:$idx
                    })
                    SET w.task_path=$task_path,
                        w.channel=$channel,
                        w.type=$type_tag,
                        w.blob=$blob
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

        logger.debug(
            "AsyncMemgraphSaver: stored %d pending writes for checkpoint %s (task %s)",
            len(writes),
            cid,
            task_id,
        )

    # ─────────────────────────────────────────────────────────────────── #
    # Retrieval utilities  (unchanged)
    # ─────────────────────────────────────────────────────────────────── #
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

    # ─────────────────────────────────────────────────────────────────── #
    # Deletion
    # ─────────────────────────────────────────────────────────────────── #
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
        logger.info("AsyncMemgraphSaver: deleted all data for thread %s", thread_id)
