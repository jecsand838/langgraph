# libs/checkpoint-memgraph/langgraph/checkpoint/memgraph/saver.py
"""Synchronous Memgraph checkpoint saver (production‑ready)."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from neo4j import Driver, GraphDatabase, Transaction
from neo4j.exceptions import ClientError  # ← NEW

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.memgraph import _internal
from langgraph.store.memgraph._utils import parse_bolt_uri  # reuse common helper
from langgraph.checkpoint.memgraph.base import BaseMemgraphSaver


# --------------------------------------------------------------------------- #
# helper: detect duplicate‑schema errors so we can ignore them for idempotency
# --------------------------------------------------------------------------- #
def _is_dup_ddl(exc: ClientError) -> bool:  # pragma: no cover
    msg = str(exc).lower()
    return "already exists" in msg or "duplicate" in msg or "existing" in msg


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

    @classmethod
    def from_conn_string(cls, conn: str, **driver_kwargs: Any) -> "MemgraphSaver":
        """Instantiate directly from a Bolt/Neo4j URI (user/pass supported)."""
        p = parse_bolt_uri(conn)
        driver: Driver = GraphDatabase.driver(  # type: ignore[arg-type]
            p["bolt_uri"], auth=(p["user"], p["password"]), **driver_kwargs
        )
        return cls(driver)

    # Context‑manager hooks (allow ``with MemgraphSaver.from_conn_string()``)
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "MemgraphSaver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Cleanly close the underlying Driver, if we created one."""
        if isinstance(self.conn, Driver):
            self.conn.close()

    # ------------------------------------------------------------------ #
    # Schema initialisation
    # ------------------------------------------------------------------ #
    def setup(self) -> None:
        """
        Apply schema migrations.

        Memgraph forbids constraint/index creation inside an explicit
        multi‑command transaction, so each statement is executed in its own
        *implicit* (auto‑commit) transaction via ``session.run``.
        """
        with _internal.get_connection(self.conn) as sess:
            # baseline marker
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
        return next_config

    # ------------------------------------------------------------------ #
    # Writes (intermediate channel output)
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

        rows = self._dump_writes(tid, ns, cid, task_id, task_path, writes)

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
    # Retrieval helpers
    # ------------------------------------------------------------------ #
    def _build_checkpoint_tuple(
        self,
        chk_node,
        blobs: list[tuple[str, str, bytes]],
        writes: list[tuple[str, str, str, bytes]],
    ) -> CheckpointTuple:
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
            return self._build_checkpoint_tuple(rec["chk"], rec["blobs"], rec["writes"])

    # ------------------------------------------------------------------ #
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
        OPTIONAL MATCH (c)-[:HAS_WRITE]->(w:Write)
        WITH c, blobs, collect([w.task_id, w.channel, w.type, w.blob]) AS writes
        RETURN c AS chk, blobs, writes
        ORDER BY c.checkpoint_id DESC
        """
        if limit:
            cypher += " LIMIT $limit"
            params["limit"] = limit

        with _internal.get_connection(self.conn) as sess:
            for rec in sess.run(cypher, **params):
                yield self._build_checkpoint_tuple(rec["chk"], rec["blobs"], rec["writes"])

    # ------------------------------------------------------------------ #
    def delete_thread(self, thread_id: str) -> None:
        """Remove all checkpoints, blobs and writes for the given thread."""
        with self._cursor() as tx:
            tx.run(
                """
                MATCH (n)
                WHERE (n:Checkpoint OR n:Blob OR n:Write) AND n.thread_id = $tid
                DETACH DELETE n
                """,
                tid=str(thread_id),
            )
