from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import TracebackType
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Type,
)

from neo4j import GraphDatabase, Session, Transaction

from langgraph.store.base import (
    BaseStore,  # type: ignore[attr-defined]
    Item,  # type: ignore[attr-defined]
    TTLConfig,  # type: ignore[attr-defined]
)

from ._utils import parse_bolt_uri

__all__ = ["MemgraphStore"]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _VectorIndexConfig:
    dims: int
    metric: str = "cos"
    name: str = "memory_embeddings"
    capacity: int = 1_000_000
    embed: Optional[Any] = None  # object with embed_query / embed_documents


class _MemgraphStoreConnMixin:
    NODE_LABEL: str = "MemoryEntry"

    @staticmethod
    def _cypher_ns_prefix_filter(prefix: Tuple[str, ...]) -> str:
        base_pred = [
            f"size(n.namespace) >= {len(prefix)}",
            "(n.expire_at IS NULL OR n.expire_at >= datetime())",
        ]
        for idx, part in enumerate(prefix):
            base_pred.append(f'n.namespace[{idx}] = "{part}"')
        return " AND ".join(base_pred)

    def _vector_index_cypher(self, cfg: _VectorIndexConfig) -> str:
        return (
            f"""
            CREATE VECTOR INDEX {cfg.name}
            ON :{self.NODE_LABEL}(embedding)
            WITH CONFIG {{
              "dimension": {cfg.dims},
              "metric": "{cfg.metric}",
              "capacity": {cfg.capacity}
            }}
            """
        )

    def _create_schema(self, session: Session, vcfg: _VectorIndexConfig | None) -> None:
        session.run(
            f"""
            CREATE CONSTRAINT entry_unique IF NOT EXISTS
            ON (n:{self.NODE_LABEL})
            ASSERT (n.namespace, n.key) IS UNIQUE
            """
        )
        session.run(
            f"""
            CREATE INDEX entry_expire IF NOT EXISTS
            FOR (n:{self.NODE_LABEL}) ON (n.expire_at)
            """
        )
        if vcfg:
            session.run(self._vector_index_cypher(vcfg))


class MemgraphStore(_MemgraphStoreConnMixin, BaseStore):
    # ------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------ #
    def __init__(
        self,
        bolt_uri: str,
        *,
        user: str = "neo4j",
        password: str = "neo4j",
        index: Mapping[str, Any] | None = None,
        ttl: Mapping[str, Any] | None = None,
        node_label: str | None = None,
        driver_kwargs: dict | None = None,
    ):
        self._driver = GraphDatabase.driver(
            bolt_uri, auth=(user, password), **(driver_kwargs or {})
        )
        if node_label:
            self.NODE_LABEL = str(node_label)

        self._vector_cfg: _VectorIndexConfig | None = None
        if index:
            self._vector_cfg = _VectorIndexConfig(
                dims=index["dims"],
                metric=index.get("metric", "cos"),
                name=index.get("name", "memory_embeddings"),
                capacity=index.get("capacity", 1_000_000),
                embed=index.get("embed"),
            )

        self._ttl_cfg: TTLConfig | None = None
        if ttl:
            self._ttl_cfg = TTLConfig(**ttl)  # type: ignore[arg-type]

        self._setup_done = False
        self._ttl_thread: threading.Thread | None = None
        self._ttl_stop_evt = threading.Event()

    # ------------- builders ---------------- #
    @classmethod
    def from_conn_string(cls, conn: str, **kwargs: Any) -> "MemgraphStore":
        parsed = parse_bolt_uri(conn)
        return cls(parsed["bolt_uri"], user=parsed["user"], password=parsed["password"], **kwargs)  # type: ignore[arg-type]

    # ------------- context ----------------- #
    def __enter__(self) -> "MemgraphStore":
        return self

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        """Synchronous batch operation implementation."""
        results = []
        for op in ops:
            # Basic implementation - you may want to optimize this for true bulk operations
            if hasattr(op, 'operation') and hasattr(op, 'namespace') and hasattr(op, 'key'):
                if op.operation == 'get':
                    result = self.get(op.namespace, op.key)
                elif op.operation == 'put':
                    result = self.put(op.namespace, op.key, op.value)
                elif op.operation == 'delete':
                    self.delete(op.namespace, op.key)
                    result = None
                else:
                    raise ValueError(f"Unknown operation: {op.operation}")
                results.append(result)
            else:
                raise ValueError(f"Invalid operation format: {op}")
        return results

    def abatch(self, ops: Iterable[Any]) -> list[Any]:
        """Async batch operation implementation (sync wrapper)."""
        import asyncio
        return asyncio.run(self._abatch_impl(ops))
    
    async def _abatch_impl(self, ops: Iterable[Any]) -> list[Any]:
        """Helper for async batch implementation."""
        return self.batch(ops)  # Delegate to sync version

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._ttl_thread and self._ttl_thread.is_alive():
            self._ttl_stop_evt.set()
            self._ttl_thread.join(timeout=5)
        self._driver.close()

    # ============================================================ #
    # schema / setup
    # ============================================================ #
    def setup(self) -> None:
        if self._setup_done:
            return
        with self._driver.session() as sess:
            self._create_schema(sess, self._vector_cfg)
        self._setup_done = True
        if self._ttl_cfg and self._ttl_cfg.sweep_interval_minutes:
            self.start_ttl_sweeper()

    initialise = setup

    # ============================================================ #
    # ttl helpers
    # ============================================================ #
    def start_ttl_sweeper(self) -> None:
        if self._ttl_thread and self._ttl_thread.is_alive():
            return
        if not self._ttl_cfg or not self._ttl_cfg.sweep_interval_minutes:
            return

        def _loop() -> None:
            interval = self._ttl_cfg.sweep_interval_minutes * 60
            while not self._ttl_stop_evt.wait(interval):
                try:
                    self.sweep_ttl()
                except Exception:
                    logger.exception("TTL sweep failed")

        self._ttl_thread = threading.Thread(
            target=_loop, name="memgraph-ttl-sweeper", daemon=True
        )
        self._ttl_thread.start()

    def sweep_ttl(self) -> None:
        with self._driver.session() as sess:
            sess.execute_write(
                lambda tx: tx.run(
                    f"""
                    MATCH (n:{self.NODE_LABEL})
                    WHERE n.expire_at IS NOT NULL AND n.expire_at < datetime()
                    DETACH DELETE n
                    """
                )
            )

    # ============================================================ #
    # internal utils
    # ============================================================ #
    def _expiry_dt(self, ttl_minutes: float | None) -> Optional[str]:
        if ttl_minutes is None and self._ttl_cfg:
            ttl_minutes = self._ttl_cfg.default_ttl
        if ttl_minutes is None:
            return None
        return (datetime.now(tz=timezone.utc) + timedelta(minutes=ttl_minutes)).isoformat()

    def _refresh_node_ttl(self, namespace: Tuple[str, ...], key: str) -> None:
        expire_at = self._expiry_dt(None)
        with self._driver.session() as sess:
            sess.run(
                f"""
                MATCH (n:{self.NODE_LABEL})
                WHERE n.namespace = $ns AND n.key = $key
                SET n.expire_at = datetime($exp)
                """,
                ns=list(namespace),
                key=key,
                exp=expire_at,
            )

    # ============================================================ #
    # CRUD and helper methods
    # ============================================================ #
    def put(
        self,
        namespace: Tuple[str, ...],
        key: str,
        value: Any,
        *,
        index: bool = True,
        ttl: float | None = None,
    ) -> Item:
        if not (isinstance(namespace, (tuple, list)) and all(isinstance(p, str) for p in namespace)):
            raise TypeError("namespace must be tuple[str, ...]")

        value_json = json.dumps(value, default=str)
        embed_vec: List[float] | None = None
        if index and self._vector_cfg and self._vector_cfg.embed:
            embed_vec = self._vector_cfg.embed.embed_documents([value_json])[0]  # type: ignore[attr-defined]
            if hasattr(embed_vec, "tolist"):
                embed_vec = embed_vec.tolist()

        expire_at = self._expiry_dt(ttl)

        with self._driver.session() as sess:

            def _tx(tx: Transaction) -> None:
                tx.run(
                    f"""
                    MATCH (n:{self.NODE_LABEL})
                    WHERE n.namespace = $ns AND n.key = $key
                    DETACH DELETE n
                    """,
                    ns=list(namespace),
                    key=key,
                )
                tx.run(
                    f"""
                    CREATE (n:{self.NODE_LABEL} {{
                        namespace: $ns,
                        key: $key,
                        value: $val,
                        expire_at: (
                            CASE WHEN $exp IS NULL THEN NULL ELSE datetime($exp) END
                        ),
                        embedding: $embedding
                    }})
                    """,
                    ns=list(namespace),
                    key=key,
                    val=value_json,
                    exp=expire_at,
                    embedding=embed_vec,
                )

            sess.execute_write(_tx)

        return Item(namespace=namespace, key=key, value=value, expires_at=expire_at)

    def put_many(
        self,
        namespace: Tuple[str, ...],
        items: Iterable[Tuple[str, Any]],
        *,
        index: bool = True,
        ttl: float | None = None,
    ) -> None:
        for key, value in items:
            self.put(namespace, key, value, index=index, ttl=ttl)

    def get(
        self,
        namespace: Tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Optional[Item]:
        with self._driver.session() as sess:
            rec = sess.execute_read(
                lambda tx: tx.run(
                    f"""
                    MATCH (n:{self.NODE_LABEL})
                    WHERE n.namespace = $ns AND n.key = $key
                    RETURN n
                    LIMIT 1
                    """,
                    ns=list(namespace),
                    key=key,
                ).single()
            )
        if not rec:
            return None
        node = rec["n"]
        if node.get("expire_at") and node["expire_at"] < datetime.now(tz=timezone.utc):
            self.delete(namespace, key)
            return None
        if refresh_ttl or (refresh_ttl is None and self._ttl_cfg and self._ttl_cfg.refresh_on_read):
            self._refresh_node_ttl(namespace, key)
            node["expire_at"] = self._expiry_dt(None)
        return Item(
            namespace=tuple(node["namespace"]),
            key=node["key"],
            value=json.loads(node["value"]),
            expires_at=node.get("expire_at"),
        )

    def exists(self, namespace: Tuple[str, ...], key: str) -> bool:
        with self._driver.session() as sess:
            rec = sess.run(
                f"""
                MATCH (n:{self.NODE_LABEL})
                WHERE n.namespace = $ns AND n.key = $key
                RETURN 1
                LIMIT 1
                """,
                ns=list(namespace),
                key=key,
            ).single()
        return rec is not None

    def count(self, namespace_prefix: Tuple[str, ...]) -> int:
        predicate = self._cypher_ns_prefix_filter(namespace_prefix)
        with self._driver.session() as sess:
            rec = sess.run(
                f"""
                MATCH (n:{self.NODE_LABEL})
                WHERE {predicate}
                RETURN count(n) AS cnt
                """
            ).single()
        return int(rec["cnt"]) if rec else 0

    def list_keys(self, namespace: Tuple[str, ...]) -> List[str]:
        pred = self._cypher_ns_prefix_filter(namespace)
        with self._driver.session() as sess:
            res = sess.run(
                f"""
                MATCH (n:{self.NODE_LABEL})
                WHERE {pred}
                RETURN n.key AS k
                """
            )
            return [r["k"] for r in res]

    def list_items(self, namespace: Tuple[str, ...]) -> List[Item]:
        pred = self._cypher_ns_prefix_filter(namespace)
        items: List[Item] = []
        with self._driver.session() as sess:
            res = sess.run(
                f"""
                MATCH (n:{self.NODE_LABEL})
                WHERE {pred}
                RETURN n
                """
            )
            for rec in res:
                n = rec["n"]
                items.append(
                    Item(
                        namespace=tuple(n["namespace"]),
                        key=n["key"],
                        value=json.loads(n["value"]),
                        expires_at=n.get("expire_at"),
                    )
                )
        return items

    def delete(
        self,
        namespace: Tuple[str, ...],
        key: str,
    ) -> None:
        with self._driver.session() as sess:
            sess.execute_write(
                lambda tx: tx.run(
                    f"""
                    MATCH (n:{self.NODE_LABEL})
                    WHERE n.namespace = $ns AND n.key = $key
                    DETACH DELETE n
                    """,
                    ns=list(namespace),
                    key=key,
                )
            )

    def delete_many(self, namespace: Tuple[str, ...], keys: Iterable[str]) -> None:
        with self._driver.session() as sess:
            sess.execute_write(
                lambda tx: tx.run(
                    f"""
                    MATCH (n:{self.NODE_LABEL})
                    WHERE n.namespace = $ns AND n.key IN $keys
                    DETACH DELETE n
                    """,
                    ns=list(namespace),
                    keys=list(keys),
                )
            )

    def delete_namespace(self, namespace_prefix: Tuple[str, ...]) -> None:
        pred = self._cypher_ns_prefix_filter(namespace_prefix)
        with self._driver.session() as sess:
            sess.execute_write(
                lambda tx: tx.run(
                    f"""
                    MATCH (n:{self.NODE_LABEL})
                    WHERE {pred}
                    DETACH DELETE n
                    """
                )
            )

    # search method updated below
    def search(
        self,
        namespace_prefix: Tuple[str, ...],
        *,
        query: str | Sequence[float] | None = None,
        filter: Mapping[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> List[Item]:
        predicate = self._cypher_ns_prefix_filter(namespace_prefix)
        items: List[Item] = []

        # vector branch
        if (
            query is not None
            and self._vector_cfg
            and self._vector_cfg.embed
            and isinstance(query, str)
        ):
            q_vec = self._vector_cfg.embed.embed_query(query)  # type: ignore[attr-defined]
            if hasattr(q_vec, "tolist"):
                q_vec = q_vec.tolist()
            k = limit + offset
            params: Dict[str, Any] = {"vec": q_vec, "k": k}
            cypher = f"""
                CALL vector_search.search("{self._vector_cfg.name}", $k, $vec)
                YIELD node, similarity
                WITH node, similarity
                WHERE {predicate}
            """
            if filter:
                for fk, fv in filter.items():
                    cypher += f" AND node.{fk} = ${fk}"
                    params[fk] = fv
            cypher += """
                RETURN node, similarity
                ORDER BY similarity DESC
                LIMIT $k
            """
            with self._driver.session() as sess:
                result = sess.run(cypher, **params)
                rows = result.data()
                for rec in rows[offset : offset + limit]:
                    n = rec["node"]
                    if refresh_ttl or (
                        refresh_ttl is None
                        and self._ttl_cfg
                        and self._ttl_cfg.refresh_on_read
                    ):
                        self._refresh_node_ttl(tuple(n["namespace"]), n["key"])
                    items.append(
                        Item(
                            namespace=tuple(n["namespace"]),
                            key=n["key"],
                            value=json.loads(n["value"]),
                            score=rec["similarity"],
                            expires_at=n.get("expire_at"),
                        )
                    )
            return items

        # lexical branch
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        cypher = f"""
            MATCH (n:{self.NODE_LABEL})
            WHERE {predicate}
        """
        if query:
            cypher += " AND n.value CONTAINS $query "
            params["query"] = query
        if filter:
            for fk, fv in filter.items():
                cypher += f" AND n.{fk} = ${fk} "
                params[fk] = fv
        cypher += """
            RETURN n
            SKIP $offset
            LIMIT $limit
        """
        with self._driver.session() as sess:
            result = sess.run(cypher, **params)
            for rec in result:
                n = rec["n"]
                if refresh_ttl or (
                    refresh_ttl is None
                    and self._ttl_cfg
                    and self._ttl_cfg.refresh_on_read
                ):
                    self._refresh_node_ttl(tuple(n["namespace"]), n["key"])
                items.append(
                    Item(
                        namespace=tuple(n["namespace"]),
                        key=n["key"],
                        value=json.loads(n["value"]),
                        expires_at=n.get("expire_at"),
                    )
                )
        return items