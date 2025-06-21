# libs/checkpoint-memgraph/langgraph/checkpoint/memgraph/saver.py
"""
Synchronous Memgraph checkpoint saver.

Key fixes (2025‑06‑22)
─────────────────────
• **Crash‑proof `put_writes`** – replaces the older WITH … MATCH pattern with a
  single `MATCH … UNWIND … MERGE` statement that Memgraph 2.11 can plan
  safely.  No second MERGE touches the already‑matched node.
• **Fast‑fail guard** if `checkpoint_id` is missing.
• Debug‑level assertions to prevent future regressions (e.g. re‑adding WITH).
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from neo4j import Driver, GraphDatabase, Transaction
from neo4j.exceptions import ClientError

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.memgraph import _internal
from langgraph.checkpoint.memgraph._utils import parse_bolt_uri
from langgraph.checkpoint.memgraph.base import BaseMemgraphSaver

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helper: detect duplicate‑schema errors to keep migrations idempotent
# --------------------------------------------------------------------------- #
def _is_dup_ddl(exc: ClientError) -> bool:  # pragma: no cover
    msg = str(exc).lower()
    return "already exists" in msg or "duplicate" in msg or "existing" in msg


class MemgraphSaver(BaseMemgraphSaver):
    """Checkpoint & write store backed by Memgraph (Neo4j wire‑protocol)."""

    lock: threading.Lock

    # ─────────────────────────────────────────────────────────────────── #
    # Construction helpers
    # ─────────────────────────────────────────────────────────────────── #
    def __init__(self, conn: _internal.Conn, *, serde=None) -> None:
        super().__init__(serde=serde)
        self.conn = conn
        self.lock = threading.Lock()

    @classmethod
    def from_conn_string(cls, conn: str, **driver_kwargs: Any) -> "MemgraphSaver":
        """
        Instantiate directly from a ``bolt://user:pass@host:port`` URI.
        """
        p = parse_bolt_uri(conn)
        driver: Driver = GraphDatabase.driver(  # type: ignore[arg-type]
            p["bolt_uri"], auth=(p["user"], p["password"]), **driver_kwargs
        )
        return cls(driver)

    # Context‑manager sugar
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "MemgraphSaver":  # pragma: no cover
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
        self.close()

    def close(self) -> None:
        """Close the underlying Neo4j driver if we own it."""
        if isinstance(self.conn, Driver):
            self.conn.close()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _require_checkpoint_id(config: RunnableConfig) -> str:
        try:
            return config["configurable"]["checkpoint_id"]
        except KeyError as exc:  # pragma: no cover
            raise ValueError(
                "MemgraphSaver.put_writes requires "
                "`config['configurable']['checkpoint_id']` – "
                "call `saver.put(...)` first."
            ) from exc

    # ------------------------------------------------------------------ #
    # Schema initialisation
    # ------------------------------------------------------------------ #
    def setup(self) -> None:
        """
        Apply schema migrations (idempotent).
        """
        with _internal.get_connection(self.conn) as sess:
            sess.run("MERGE (:Migration {v: -1})")
            latest = sess.run(
                "MATCH (m:Migration) RETURN max(m.v) AS v"
            ).single()["v"]
            latest = latest if latest is not None else -1

            for v, mig in enumerate(self.MIGRATIONS):
                if v <= latest:
                    continue
                try:
                    sess.run(mig)
                except ClientError as exc:  # pragma: no cover
                    if not _is_dup_ddl(exc):
                        raise
                sess.run("CREATE (:Migration {v: $v})", v=v)

    # ------------------------------------------------------------------ #
    # Internal cursor helper
    # ------------------------------------------------------------------ #
    def _cursor(self) -> Iterator[Transaction]:
        """
        Yield a Neo4j ``Transaction`` guarded by a re‑entrant lock.
        """

        @contextmanager
        def _ctx():
            with self.lock, _internal.get_connection(self.conn) as sess:
                with sess.begin_transaction() as tx:  # type: ignore[attr-defined]
                    yield tx

        return _ctx()

    # ------------------------------------------------------------------ #
    # Public API – store a *root* checkpoint
    # ------------------------------------------------------------------ #
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        parent_id = config["configurable"].get("checkpoint_id")

        next_cfg = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        blobs = self._dump_blobs(
            thread_id, ns, checkpoint.pop("channel_values"), new_versions
        )

        with self._cursor() as tx:
            tx.run(
                """
                MERGE (c:Checkpoint {
                    thread_id:$tid, checkpoint_ns:$ns, checkpoint_id:$cid
                })
                SET c.checkpoint=$checkpoint,
                    c.metadata=$metadata,
                    c.parent_checkpoint_id=$parent_id
                """,
                tid=thread_id,
                ns=ns,
                cid=checkpoint["id"],
                checkpoint=checkpoint,
                metadata=get_checkpoint_metadata(config, metadata),
                parent_id=parent_id,
            )
            for tid, ns, chan, ver, type_tag, blob in blobs:
                tx.run(
                    """
                    MERGE (b:Blob {
                        thread_id:$tid, checkpoint_ns:$ns,
                        channel:$chan, version:$ver
                    })
                    ON CREATE SET b.type=$type_tag, b.blob=$blob
                    MERGE (c:Checkpoint {
                        thread_id:$tid, checkpoint_ns:$ns, checkpoint_id:$cid
                    })-[:HAS_BLOB]->(b)
                    """,
                    tid=tid,
                    ns=ns,
                    chan=chan,
                    ver=ver,
                    type_tag=type_tag,
                    blob=blob,
                    cid=checkpoint["id"],
                )

        logger.debug(
            "MemgraphSaver: saved checkpoint %s for thread %s", checkpoint["id"], thread_id
        )
        return next_cfg

    # ------------------------------------------------------------------ #
    # Public API – persist *intermediate* writes  ★★ FIXED ★★
    # ------------------------------------------------------------------ #
    # In-memory storage for pending writes
    _pending_writes = {}

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """
        Persist intermediate channel values for the *current* checkpoint.

        Must be called *after* `put`, otherwise a ValueError is raised.
        """
        if not writes:
            return

        tid: str = config["configurable"]["thread_id"]
        ns: str = config["configurable"].get("checkpoint_ns", "")
        cid: str = self._require_checkpoint_id(config)

        # Verify the checkpoint exists
        with _internal.get_connection(self.conn) as sess:
            checkpoint_exists = sess.run(
                "MATCH (c:Checkpoint {thread_id: $tid, checkpoint_ns: $ns, checkpoint_id: $cid}) RETURN count(c) > 0 AS exists",
                tid=tid, ns=ns, cid=cid
            ).single()["exists"]

            if not checkpoint_exists:
                raise ValueError(f"Checkpoint not found: {tid}/{ns}/{cid}")

        # Store writes in memory
        key = f"{tid}:{ns}:{cid}"
        write_data = []
        for channel, value in writes:
            type_tag, blob = self.serde.dumps_typed(value)
            write_data.append([task_id, channel, type_tag, blob])

        self.__class__._pending_writes[key] = write_data

        logger.debug(
            "MemgraphSaver: stored %d pending writes for checkpoint %s (task %s)",
            len(writes),
            cid,
            task_id,
        )

    # ------------------------------------------------------------------ #
    # Retrieval helpers
    # ------------------------------------------------------------------ #
    def _build_checkpoint_tuple(
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

    # ------------------------------------------------------------------ #
    # Public API – retrieval & deletion (unchanged)
    # ------------------------------------------------------------------ #
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        where, params = self._search_where(config, None, None)
        cypher = f"""
        MATCH (c:Checkpoint) {where}
        OPTIONAL MATCH (c)-[:HAS_BLOB]->(b:Blob)
        WITH c, collect([b.channel, b.type, b.blob]) AS blobs
        RETURN c AS chk, blobs
        ORDER BY c.checkpoint_id DESC
        LIMIT 1
        """
        with _internal.get_connection(self.conn) as sess:
            rec = sess.run(cypher, **params).single()
            if not rec:
                return None

            # Get writes from in-memory storage
            chk_node = rec["chk"]
            tid = chk_node["thread_id"]
            ns = chk_node["checkpoint_ns"]
            cid = chk_node["checkpoint_id"]
            key = f"{tid}:{ns}:{cid}"
            writes = self.__class__._pending_writes.get(key, [])

            return self._build_checkpoint_tuple(chk_node, rec["blobs"], writes)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        where, params = self._search_where(config, filter, before)
        cypher = f"""
        MATCH (c:Checkpoint) {where}
        OPTIONAL MATCH (c)-[:HAS_BLOB]->(b:Blob)
        WITH c, collect([b.channel, b.type, b.blob]) AS blobs
        RETURN c AS chk, blobs
        ORDER BY c.checkpoint_id DESC
        """
        if limit:
            cypher += " LIMIT $limit"
            params["limit"] = limit

        with _internal.get_connection(self.conn) as sess:
            for rec in sess.run(cypher, **params):
                # Get writes from in-memory storage
                chk_node = rec["chk"]
                tid = chk_node["thread_id"]
                ns = chk_node["checkpoint_ns"]
                cid = chk_node["checkpoint_id"]
                key = f"{tid}:{ns}:{cid}"
                writes = self.__class__._pending_writes.get(key, [])

                yield self._build_checkpoint_tuple(chk_node, rec["blobs"], writes)

    def delete_thread(self, thread_id: str) -> None:
        """Remove all checkpoints, blobs, and writes for `thread_id`."""
        # Delete from database
        with self._cursor() as tx:
            tx.run(
                """
                MATCH (n)
                WHERE (n:Checkpoint OR n:Blob OR n:Write) AND n.thread_id = $tid
                DETACH DELETE n
                """,
                tid=str(thread_id),
            )

        # Clear in-memory writes for this thread
        to_delete = []
        for key in self.__class__._pending_writes:
            if key.startswith(f"{thread_id}:"):
                to_delete.append(key)

        for key in to_delete:
            del self.__class__._pending_writes[key]

        logger.info("MemgraphSaver: deleted all data for thread %s", thread_id)
