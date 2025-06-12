from __future__ import annotations

import threading
from typing import Any, Iterator, Sequence

from neo4j import Driver, GraphDatabase, Transaction

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memgraph import _internal
from langgraph.checkpoint.memgraph._utils import parse_bolt_uri
from langgraph.checkpoint.memgraph.base import BaseMemgraphSaver


class MemgraphSaver(BaseMemgraphSaver):
    """Checkpointer that stores checkpoints in Memgraph (Neo4j‑compatible)."""

    lock: threading.Lock

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        conn: _internal.Conn,
        *,
        serde=None,
    ) -> None:
        super().__init__(serde=serde)
        self.conn = conn
        self.lock = threading.Lock()

    # ---------- builder ------------------------------------------------ #
    @classmethod
    def from_conn_string(cls, conn: str, **driver_kwargs: Any) -> "MemgraphSaver":
        """
        Build a :class:`MemgraphSaver` from a *single* Bolt/Neo4j connection URI
        (optionally containing inline ``user:pass`` credentials).

        Example accepted URIs::

            bolt://localhost:7687
            bolt://neo4j:secret@db.example.com:7687
            neo4j://scott:tiger@10.0.0.5
        """
        parsed = parse_bolt_uri(conn)
        driver: Driver = GraphDatabase.driver(  # type: ignore[arg-type]
            parsed["bolt_uri"], auth=(parsed["user"], parsed["password"]), **driver_kwargs
        )
        return cls(driver)

    # ---------- context management ------------------------------------- #
    def __enter__(self) -> "MemgraphSaver":  # pragma: no cover
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
        self.close()

    def close(self) -> None:
        """Close the underlying connection (if we own it)."""
        if isinstance(self.conn, Driver):
            self.conn.close()

    # ------------------------------------------------------------------ #
    # Schema initialisation
    # ------------------------------------------------------------------ #
    def setup(self) -> None:
        """Run Cypher migrations (idempotent)."""
        with _internal.get_connection(self.conn) as sess:
            with sess.begin_transaction() as tx:  # type: ignore[attr-defined]
                tx.run("MERGE (:Migration {v: -1})")  # ensure table exists
                latest = tx.run(
                    "MATCH (m:Migration) RETURN max(m.v) AS v"
                ).single()["v"]
                for v, mig in enumerate(self.MIGRATIONS):
                    if v > latest:
                        tx.run(mig)
                        tx.run("CREATE (:Migration {v: $v})", v=v)

    # ------------------------------------------------------------------ #
    # Internal cursor helper
    # ------------------------------------------------------------------ #
    def _cursor(self) -> Iterator[Transaction]:
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            with self.lock, _internal.get_connection(self.conn) as sess:
                with sess.begin_transaction() as tx:  # type: ignore[attr-defined]
                    yield tx

        return _ctx()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parent_id = config["configurable"].get("checkpoint_id")

        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        blobs = self._dump_blobs(
            thread_id, checkpoint_ns, checkpoint.pop("channel_values"), new_versions
        )

        with self._cursor() as tx:
            # Upsert checkpoint node
            tx.run(
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
            # Upsert blob nodes & relationship
            for tid, ns, channel, ver, type_tag, blob in blobs:
                tx.run(
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
        return next_config

    # ------------------------------------------------------------------ #
    # --------- writes support ----------------------------------------- #
    # ------------------------------------------------------------------ #
    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint."""
        if not writes:
            return

        tid = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        cid = config["configurable"]["checkpoint_id"]

        rows = self._dump_writes(
            tid,
            ns,
            cid,
            task_id,
            task_path,
            writes,
        )

        with self._cursor() as tx:
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
                tx.run(
                    """
                    MERGE (w:Write {
                        thread_id:$tid,
                        checkpoint_ns:$ns,
                        checkpoint_id:$cid,
                        task_id:$task_id,
                        idx:$idx
                    })
                    SET w.task_path=$t_path,
                        w.channel=$channel,
                        w.type=$type_tag,
                        w.blob=$blob
                    WITH w
                    MATCH (c:Checkpoint {
                        thread_id:$tid,
                        checkpoint_ns:$ns,
                        checkpoint_id:$cid
                    })
                    MERGE (c)-[:HAS_WRITE]->(w)
                    """,
                    tid=tid,
                    ns=ns,
                    cid=cid,
                    task_id=t_id,
                    t_path=t_path,
                    idx=idx,
                    channel=channel,
                    type_tag=type_tag,
                    blob=blob,
                )

    # ------------------------------------------------------------------ #
    # --------- checkpoint retrieval utilities ------------------------- #
    # ------------------------------------------------------------------ #
    def _build_checkpoint_tuple(
        self,
        chk_node,
        blobs: list[tuple[str, str, bytes]],
        writes: list[tuple[str, str, str, bytes]],
    ) -> CheckpointTuple:
        """Helper to construct CheckpointTuple from raw record pieces."""
        checkpoint_dict = chk_node["checkpoint"]
        checkpoint_dict["channel_values"] = self._load_blobs(blobs)

        pending_writes = self._load_writes(writes)

        return CheckpointTuple(
            {
                "configurable": {
                    "thread_id": chk_node["thread_id"],
                    "checkpoint_ns": chk_node["checkpoint_ns"],
                    "checkpoint_id": chk_node["checkpoint_id"],
                }
            },
            checkpoint_dict,
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
            pending_writes,
        )

    # ------------------------------------------------------------------ #
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Retrieve a single checkpoint tuple."""
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
        with _internal.get_connection(self.conn) as sess:
            rec = sess.run(cypher, **params).single()
            if not rec:
                return None
            chk_node = rec["chk"]
            blobs = rec["blobs"]
            writes = rec["writes"]

        return self._build_checkpoint_tuple(chk_node, blobs, writes)

    # ------------------------------------------------------------------ #
    # --------- list ----------------------------------------------------#
    # ------------------------------------------------------------------ #
    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """
        Yield checkpoints matching the given criteria ordered by newest first.
        """
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

        with _internal.get_connection(self.conn) as sess:
            result = sess.run(cypher, **params)
            for rec in result:
                yield self._build_checkpoint_tuple(
                    rec["chk"],
                    rec["blobs"],
                    rec["writes"],
                )

    # ------------------------------------------------------------------ #
    # --------- delete helpers ----------------------------------------- #
    # ------------------------------------------------------------------ #
    def delete_thread(self, thread_id: str) -> None:
        """Remove all checkpoints / blobs / writes for a given thread."""
        with self._cursor() as tx:
            tx.run(
                """
                MATCH (n)
                WHERE (n:Checkpoint OR n:Blob OR n:Write)
                  AND n.thread_id = $tid
                DETACH DELETE n
                """,
                tid=str(thread_id),
            )
