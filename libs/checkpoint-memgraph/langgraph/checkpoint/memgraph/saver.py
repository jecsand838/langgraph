# libs/checkpoint-memgraph/langgraph/checkpoint/memgraph/saver.py
"""
Synchronous Memgraph checkpoint saver (production‑ready).

Key fixes (2025‑06‑21 → 2025‑06‑22)
──────────────────────────────────
• Refactored `put_writes` Cypher to **eliminate the WITH … MATCH pattern**
  that triggered `Memgraph.ExecutionException`.  The new query first MATCHes
  the target `Checkpoint`, then MERGEs the `Write` node and the
  `(:Checkpoint)-[:HAS_WRITE]->(:Write)` relationship in one statement.
• Added explicit fast‑fail guard for missing `checkpoint_id`.
• Injected debug‑level logging & lightweight assertions that validate the
  query structure, protecting against future regressions.
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
from langgraph.checkpoint.memgraph._utils import parse_bolt_uri  # ← fixed import path
from langgraph.checkpoint.memgraph.base import BaseMemgraphSaver

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# helper: detect duplicate‑schema errors so we can ignore them for idempotency
# --------------------------------------------------------------------------- #
def _is_dup_ddl(exc: ClientError) -> bool:  # pragma: no cover
    msg = str(exc).lower()
    return "already exists" in msg or "duplicate" in msg or "existing" in msg


class MemgraphSaver(BaseMemgraphSaver):
    """Checkpointer that stores checkpoints, blobs & writes in Memgraph (Neo4j)."""

    lock: threading.Lock

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def __init__(self, conn: _internal.Conn, *, serde=None) -> None:
        super().__init__(serde=serde)
        self.conn = conn
        self.lock = threading.Lock()

    @classmethod
    def from_conn_string(cls, conn: str, **driver_kwargs: Any) -> "MemgraphSaver":
        """
        Quickly instantiate from a bolt/neo4j URI.

        ``conn`` may include user/pass, e.g. ``bolt://neo4j:secret@localhost:7687``.
        """
        p = parse_bolt_uri(conn)
        driver: Driver = GraphDatabase.driver(  # type: ignore[arg-type]
            p["bolt_uri"], auth=(p["user"], p["password"]), **driver_kwargs
        )
        return cls(driver)

    # Context‑manager hooks (allow ``with MemgraphSaver.from_conn_string()``)
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "MemgraphSaver":  # pragma: no cover
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
        self.close()

    def close(self) -> None:
        """Cleanly close the underlying Neo4j Driver (if we created one)."""
        if isinstance(self.conn, Driver):
            self.conn.close()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _require_checkpoint_id(config: RunnableConfig) -> str:
        """
        Return the checkpoint_id or raise a friendly error if it's missing.

        Users must call ``saver.put`` *before* ``saver.put_writes`` so that the
        saver knows which checkpoint the writes belong to.
        """
        try:
            return config["configurable"]["checkpoint_id"]
        except KeyError as exc:  # pragma: no cover
            raise ValueError(
                "MemgraphSaver.put_writes requires "
                "`config['configurable']['checkpoint_id']` – "
                "call `saver.put(...)` first."
            ) from exc

    # ------------------------------------------------------------------ #
    # Schema initialisation  (unchanged)
    # ------------------------------------------------------------------ #
    def setup(self) -> None:
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
    # Internal cursor helper  (unchanged)
    # ------------------------------------------------------------------ #
    def _cursor(self) -> Iterator[Transaction]:
        @contextmanager
        def _ctx():
            with self.lock, _internal.get_connection(self.conn) as sess:
                with sess.begin_transaction() as tx:  # type: ignore[attr-defined]
                    yield tx

        return _ctx()

    # ------------------------------------------------------------------ #
    # Public API: checkpoint persistence  (unchanged)
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
            # Upsert blob nodes & relationships
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

        logger.debug(
            "MemgraphSaver: saved checkpoint %s for thread %s", checkpoint["id"], thread_id
        )
        return next_config

    # ------------------------------------------------------------------ #
    # Public API: intermediate writes  ★★★ FIXED ★★★
    # ------------------------------------------------------------------ #
    def put_writes(               # noqa: C901  (complexity is fine here)
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        if not writes:
            return

        tid  = config["configurable"]["thread_id"]
        ns   = config["configurable"].get("checkpoint_ns", "")
        cid  = self._require_checkpoint_id(config)

        rows = self._dump_writes(tid, ns, cid, task_id, task_path, writes)

        # We batch everything through UNWIND → 1 round‑trip
        cypher = """
        MATCH (c:Checkpoint {
            thread_id:$tid, checkpoint_ns:$ns, checkpoint_id:$cid
        })
        UNWIND $rows AS r
        MERGE (c)-[:HAS_WRITE]->(w:Write {
            thread_id:r.tid,
            checkpoint_ns:r.ns,
            checkpoint_id:r.cid,
            task_id:r.task_id,
            idx:r.idx
        })
        SET w.task_path = r.task_path,
            w.channel    = r.channel,
            w.type       = r.type_tag,
            w.blob       = r.blob
        """

        # ── Development‑time sanity guards ───────────────────────────────
        assert "WITH" not in cypher.upper(), "Cypher must not contain WITH"
        assert cypher.lstrip().startswith("MATCH"), "Cypher must start with MATCH"
        assert "UNWIND" in cypher and "MERGE" in cypher, "Expected UNWIND + MERGE"

        # Shape rows as list[dict] for UNWIND
        row_maps = [
            {
                "tid":   r_tid,
                "ns":    r_ns,
                "cid":   r_cid,
                "task_id": r_task,
                "task_path": r_path,
                "idx":   r_idx,
                "channel":   r_chan,
                "type_tag":  r_type,
                "blob":  r_blob,
            }
            for (
                r_tid, r_ns, r_cid, r_task, r_path,
                r_idx, r_chan, r_type, r_blob
            ) in rows
        ]

        with self._cursor() as tx:
            tx.run(cypher, tid=tid, ns=ns, cid=cid, rows=row_maps)

        logger.debug(
            "MemgraphSaver: stored %d pending writes for checkpoint %s (task %s)",
            len(writes), cid, task_id,
        )

    # ------------------------------------------------------------------ #
    # Retrieval & deletion helpers  (unchanged)
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

    def delete_thread(self, thread_id: str) -> None:
        with self._cursor() as tx:
            tx.run(
                """
                MATCH (n)
                WHERE (n:Checkpoint OR n:Blob OR n:Write) AND n.thread_id = $tid
                DETACH DELETE n
                """,
                tid=str(thread_id),
            )
        logger.info("MemgraphSaver: deleted all data for thread %s", thread_id)
