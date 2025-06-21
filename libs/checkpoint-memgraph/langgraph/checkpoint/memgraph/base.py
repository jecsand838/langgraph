# libs/checkpoint-memgraph/langgraph/checkpoint/memgraph/base.py
"""
Shared helper routines, Cypher schema migrations, and blob (de)serialization
utilities for both synchronous and asynchronous Memgraph savers.
"""
from __future__ import annotations

import random
from typing import Any, Mapping, Sequence, cast

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

# --------------------------------------------------------------------------- #
# NOTE:  The constraint syntax below follows Memgraph ≥ 2.11 requirements.
# --------------------------------------------------------------------------- #
MIGRATIONS: Sequence[str] = (
    "MERGE (:Migration {v: 0})",  # 0 – baseline marker (no‑op)
    """
    CREATE CONSTRAINT ON (c:Checkpoint)
    ASSERT c.thread_id, c.checkpoint_ns, c.checkpoint_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT ON (b:Blob)
    ASSERT b.thread_id, b.checkpoint_ns, b.channel, b.version IS UNIQUE
    """,
    """
    CREATE CONSTRAINT ON (w:Write)
    ASSERT w.thread_id, w.checkpoint_ns, w.checkpoint_id,
           w.task_id, w.idx IS UNIQUE
    """,
)


class BaseMemgraphSaver(BaseCheckpointSaver[str]):
    """Logic shared by sync & async Memgraph saver implementations."""

    MIGRATIONS: Sequence[str] = MIGRATIONS

    # ------------------------------------------------------------------ #
    # Blob helpers
    # ------------------------------------------------------------------ #
    def _load_blobs(
        self,
        blob_records: list[tuple[str, str, bytes]],
    ) -> dict[str, Any]:
        """
        Convert `(channel, type, blob)` triples to decoded Python objects,
        ignoring any records that are null/empty placeholders.
        """
        if not blob_records:
            return {}

        return {
            channel: self.serde.loads_typed((type_tag, blob))
            for channel, type_tag, blob in blob_records
            # Skip rows produced by OPTIONAL MATCH where all fields are null
            if channel
            and type_tag
            and type_tag != "empty"
        }

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: dict[str, Any],
        versions: ChannelVersions,
    ) -> list[tuple[str, str, str, str, bytes | None]]:
        if not versions:
            return []
        out: list[tuple[str, str, str, str, bytes | None]] = []
        for channel, ver in versions.items():
            if channel in values:
                type_tag, blob = self.serde.dumps_typed(values[channel])
            else:
                type_tag, blob = "empty", None
            out.append((thread_id, checkpoint_ns, channel, cast(str, ver), type_tag, blob))
        return out

    # ------------------------------------------------------------------ #
    def _load_writes(
        self,
        writes: list[tuple[str, str, str, bytes]],
    ) -> list[tuple[str, str, Any]]:
        """Decode pending‑write rows; silently drop null/placeholder rows."""
        return [
            (
                task_id,
                channel,
                self.serde.loads_typed((type_tag, blob)),
            )
            for task_id, channel, type_tag, blob in writes
            if channel and type_tag
        ]

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        task_path: str,
        writes: Sequence[tuple[str, Any]],
    ) -> list[tuple[str, str, str, str, str, int, str, str, bytes]]:
        return [
            (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                task_path,
                WRITES_IDX_MAP.get(channel, idx),
                channel,
                *self.serde.dumps_typed(value),
            )
            for idx, (channel, value) in enumerate(writes)
        ]

    # ------------------------------------------------------------------ #
    def get_next_version(self, current: str | None) -> str:
        if current is None:
            current_int = 0
        else:
            current_int = int(current.split(".")[0])
        next_int = current_int + 1
        return f"{next_int:032}.{random.random():016}"

    # ------------------------------------------------------------------ #
    # Cypher WHERE‑clause helper (unchanged)
    # ------------------------------------------------------------------ #
    def _search_where(
        self,
        config: Mapping[str, Any] | None,
        filter: Mapping[str, Any] | None,
        before: Mapping[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        wheres: list[str] = []
        params: dict[str, Any] = {}

        if config:
            tid = config["configurable"]["thread_id"]
            wheres.append("c.thread_id = $thread_id")
            params["thread_id"] = tid

            ns = config["configurable"].get("checkpoint_ns", "")
            wheres.append("c.checkpoint_ns = $checkpoint_ns")
            params["checkpoint_ns"] = ns

            cid = get_checkpoint_id(config)
            if cid:
                wheres.append("c.checkpoint_id = $checkpoint_id")
                params["checkpoint_id"] = cid

        if before is not None:
            wheres.append("c.checkpoint_id < $before_id")
            params["before_id"] = get_checkpoint_id(before)

        if filter:
            for k, v in filter.items():
                pk = f"meta_{k}"
                wheres.append(f"c.metadata[{repr(k)}] = ${pk}")
                params[pk] = v

        where_clause = "WHERE " + " AND ".join(wheres) if wheres else ""
        return where_clause, params
