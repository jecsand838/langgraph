from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace, TracebackType
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
)

from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ClientError

from langgraph.store.base import (  # type: ignore[attr-defined]
    BaseStore,
    Item,
)

from . import _MemgraphStoreConnMixin, _VectorIndexConfig
from ._utils import parse_bolt_uri

__all__ = ["AsyncMemgraphStore"]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# helper: detect duplicate‑schema errors (so we can ignore them for idempotency)
# --------------------------------------------------------------------------- #
def _is_duplicate_ddl(exc: ClientError) -> bool:  # pragma: no cover
    msg = str(exc).lower()
    return "already exists" in msg or "duplicate" in msg or "existing" in msg


# =========================================================================== #
#                             A S Y N C   S T O R E                          #
# =========================================================================== #
class AsyncMemgraphStore(_MemgraphStoreConnMixin, BaseStore):  # type: ignore[misc]
    """
    Asynchronous Memgraph key–value store.

    **Event‑loop safety**

    * A separate Neo4j async driver is kept for **each event‑loop** that
      touches the store.  This fully eliminates “Future attached to a different
      loop” errors that occur when the same store instance is shared across
      multiple loops (e.g. under `pytest‑asyncio`).
    """

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
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
    ) -> None:
        # Connection details
        self._bolt_uri = bolt_uri
        self._user = user
        self._password = password
        self._driver_kwargs = driver_kwargs or {}

        # Neo4j drivers keyed by the owning event‑loop
        self._drivers: dict[asyncio.AbstractEventLoop, Any] = {}

        if node_label:
            self.NODE_LABEL = str(node_label)

        # Optional vector‑index configuration
        self._vector_cfg: _VectorIndexConfig | None = None
        if index:
            self._vector_cfg = _VectorIndexConfig(
                dims=index["dims"],
                metric=index.get("metric", "cos"),
                name=index.get("name", "memory_embeddings"),
                capacity=index.get("capacity", 1_000_000),
                embed=index.get("embed"),
            )

        # TTL configuration
        self._ttl_cfg: SimpleNamespace | None = None
        if ttl:
            _defaults = {
                "default_ttl": None,
                "refresh_on_read": False,
                "sweep_interval_minutes": None,
            }
            _defaults.update(ttl)
            self._ttl_cfg = SimpleNamespace(**_defaults)

        self._setup_done = False
        self._ttl_task: Optional[asyncio.Task[None]] = None

    # ------------------------------------------------------------------ #
    # driver helpers (one driver per event‑loop)
    # ------------------------------------------------------------------ #
    def _create_driver(self, loop: asyncio.AbstractEventLoop):
        """Create a Neo4j async driver bound to *loop*."""
        driver = AsyncGraphDatabase.driver(
            self._bolt_uri, auth=(self._user, self._password), **self._driver_kwargs
        )

        # Patch driver internals so sockets & connectors share this loop
        try:
            pool = driver._pool  # type: ignore[attr-defined]
            pool._loop = loop
            if hasattr(pool, "_connector"):
                pool._connector._loop = loop  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover
            pass  # best‑effort only

        return driver

    def _get_driver(self):
        loop = asyncio.get_running_loop()
        if loop not in self._drivers:
            self._drivers[loop] = self._create_driver(loop)
        return self._drivers[loop]

    # Expose property so existing code (`self._driver`) continues to work
    @property
    def _driver(self):
        return self._get_driver()

    # ------------------------------------------------------------------ #
    # builders
    # ------------------------------------------------------------------ #
    @classmethod
    def from_conn_string(cls, conn: str, **kwargs: Any) -> "AsyncMemgraphStore":
        parsed = parse_bolt_uri(conn)
        return cls(
            parsed["bolt_uri"], user=parsed["user"], password=parsed["password"], **kwargs
        )

    # ------------------------------------------------------------------ #
    # async context
    # ------------------------------------------------------------------ #
    async def __aenter__(self) -> "AsyncMemgraphStore":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        await self.close()

    # ------------------------------------------------------------------ #
    # driver lifecycle
    # ------------------------------------------------------------------ #
    async def close(self) -> None:
        if self._ttl_task:
            self._ttl_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ttl_task

        # Close all cached drivers
        for drv in list(self._drivers.values()):
            try:
                await drv.close()
            except RuntimeError as exc:  # pragma: no cover
                if "event loop is closed" not in str(exc).lower():
                    raise
        self._drivers.clear()

    # ------------------------------------------------------------------ #
    # schema / setup
    # ------------------------------------------------------------------ #
    async def setup(self) -> None:
        """Initialise database schema (idempotent)."""
        if self._setup_done:
            return

        async with self._driver.session() as sess:  # type: ignore[attr-defined]
            async def run_safe(cypher: str) -> None:
                try:
                    result = await sess.run(cypher)
                    await result.consume()
                except ClientError as exc:  # pragma: no cover
                    if not _is_duplicate_ddl(exc):
                        raise

            # uniqueness constraint
            await run_safe(
                f"""
                CREATE CONSTRAINT ON (n:{self.NODE_LABEL})
                ASSERT n.namespace, n.key IS UNIQUE
                """
            )
            # expiry index
            await run_safe(f"CREATE INDEX ON :{self.NODE_LABEL}(expire_at)")
            # vector index (optional)
            if self._vector_cfg:
                await run_safe(self._vector_index_cypher(self._vector_cfg))

        self._setup_done = True
        if self._ttl_cfg and self._ttl_cfg.sweep_interval_minutes:
            self._start_ttl_sweeper()

    initialise = setup  # alias

    # ============================================================ #
    # TTL SWEEPER
    # ============================================================ #
    def _start_ttl_sweeper(self) -> None:
        if self._ttl_task or not self._ttl_cfg or not self._ttl_cfg.sweep_interval_minutes:
            return

        async def _loop() -> None:
            await asyncio.sleep(0)
            interval = self._ttl_cfg.sweep_interval_minutes * 60  # type: ignore[operator]
            while True:
                await asyncio.sleep(interval)
                try:
                    await self.sweep_ttl()
                except Exception:  # pragma: no cover
                    logger.exception("Async TTL sweep failed")

        self._ttl_task = asyncio.create_task(_loop())

    async def sweep_ttl(self) -> None:
        async with self._driver.session() as sess:  # type: ignore[attr-defined]
            async def _tx(tx):
                await (
                    await tx.run(
                        f"""
                        MATCH (n:{self.NODE_LABEL})
                        WHERE n.expire_at IS NOT NULL AND n.expire_at < datetime()
                        DETACH DELETE n
                        """
                    )
                ).consume()

            await sess.execute_write(_tx)

    # ============================================================ #
    # internal helpers
    # ============================================================ #
    def _expiry_dt(self, ttl_minutes: float | None) -> Optional[str]:
        if ttl_minutes is None and self._ttl_cfg:
            ttl_minutes = self._ttl_cfg.default_ttl
        if ttl_minutes is None:
            return None
        return (datetime.now(tz=timezone.utc) + timedelta(minutes=ttl_minutes)).isoformat()

    async def _refresh_node_ttl(self, namespace: Tuple[str, ...], key: str) -> None:
        exp = self._expiry_dt(None)
        async with self._driver.session() as sess:  # type: ignore[attr-defined]
            await (
                await sess.run(
                    f"""
                    MATCH (n:{self.NODE_LABEL})
                    WHERE n.namespace = $ns AND n.key = $key
                    SET n.expire_at = datetime($exp)
                    """,
                    ns=list(namespace),
                    key=key,
                    exp=exp,
                )
            ).consume()

    # ============================================================ #
    # CRUD: PUT
    # ============================================================ #
    async def aput(
        self,
        namespace: Tuple[str, ...],
        key: str,
        value: Any,
        *,
        index: bool = True,
        ttl: float | None = None,
    ) -> Item:
        if not (
            isinstance(namespace, (tuple, list)) and all(isinstance(p, str) for p in namespace)
        ):
            raise TypeError("namespace must be tuple[str, ...]")

        value_json = json.dumps(value, default=str)
        embed_vec: List[float] | None = None
        if index and self._vector_cfg and self._vector_cfg.embed:
            embed_vec = self._vector_cfg.embed.embed_documents([value_json])[0]  # type: ignore[attr-defined]
            if hasattr(embed_vec, "tolist"):
                embed_vec = embed_vec.tolist()

        exp = self._expiry_dt(ttl)

        async with self._driver.session() as sess:  # type: ignore[attr-defined]

            async def _tx(tx):  # type: ignore[valid-type]
                # Delete existing node (if any)
                await (
                    await tx.run(
                        f"""
                        MATCH (n:{self.NODE_LABEL})
                        WHERE n.namespace = $ns AND n.key = $key
                        DETACH DELETE n
                        """,
                        ns=list(namespace),
                        key=key,
                    )
                ).consume()

                # Create new node
                await (
                    await tx.run(
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
                        exp=exp,
                        embedding=embed_vec,
                    )
                ).consume()

            await sess.execute_write(_tx)

        return self._build_item(
            namespace=namespace,
            key=key,
            value=value,
            expires_at=exp,
        )

    async def aput_many(
        self,
        namespace: Tuple[str, ...],
        items: Iterable[Tuple[str, Any]],
        *,
        index: bool = True,
        ttl: float | None = None,
    ) -> None:
        for k, v in items:
            await self.aput(namespace, k, v, index=index, ttl=ttl)

    # ============================================================ #
    # CRUD: GET / EXISTS / COUNT
    # ============================================================ #
    async def aget(
        self,
        namespace: Tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Optional[Item]:
        async with self._driver.session() as sess:  # type: ignore[attr-defined]

            async def _read(tx):
                r = await tx.run(
                    f"""
                    MATCH (n:{self.NODE_LABEL})
                    WHERE n.namespace = $ns AND n.key = $key
                    RETURN n LIMIT 1
                    """,
                    ns=list(namespace),
                    key=key,
                )
                return await r.single()

            rec = await sess.execute_read(_read)

        if not rec:
            return None
        n = rec["n"]
        if n.get("expire_at") and n["expire_at"] < datetime.now(tz=timezone.utc):
            await self.adelete(namespace, key)
            return None
        if refresh_ttl or (
            refresh_ttl is None and self._ttl_cfg and self._ttl_cfg.refresh_on_read
        ):
            await self._refresh_node_ttl(namespace, key)
            n["expire_at"] = self._expiry_dt(None)
        return self._build_item(
            namespace=tuple(n["namespace"]),
            key=n["key"],
            value=json.loads(n["value"]),
            expires_at=n.get("expire_at"),
        )

    async def aexists(self, namespace: Tuple[str, ...], key: str) -> bool:
        async with self._driver.session() as sess:  # type: ignore[attr-defined]
            rec = await sess.run(
                f"""
                MATCH (n:{self.NODE_LABEL})
                WHERE n.namespace = $ns AND n.key = $key
                RETURN 1 LIMIT 1
                """,
                ns=list(namespace),
                key=key,
            )
            return (await rec.single()) is not None

    async def acount(self, namespace_prefix: Tuple[str, ...]) -> int:
        pred = self._cypher_ns_prefix_filter(namespace_prefix)
        async with self._driver.session() as sess:  # type: ignore[attr-defined]
            rec = await sess.run(
                f"""
                MATCH (n:{self.NODE_LABEL})
                WHERE {pred}
                RETURN count(n) AS cnt
                """
            )
            row = await rec.single()
        return int(row["cnt"]) if row else 0

    # ============================================================ #
    # CRUD: LIST KEYS / ITEMS
    # ============================================================ #
    async def alist_keys(self, namespace: Tuple[str, ...]) -> List[str]:
        pred = self._cypher_ns_prefix_filter(namespace)
        async with self._driver.session() as sess:  # type: ignore[attr-defined]
            res = await sess.run(
                f"""
                MATCH (n:{self.NODE_LABEL})
                WHERE {pred}
                RETURN n.key AS k
                """
            )
            return [r["k"] async for r in res]

    async def alist_items(self, namespace: Tuple[str, ...]) -> List[Item]:
        pred = self._cypher_ns_prefix_filter(namespace)
        items: List[Item] = []
        async with self._driver.session() as sess:  # type: ignore[attr-defined]
            res = await sess.run(
                f"""
                MATCH (n:{self.NODE_LABEL})
                WHERE {pred}
                RETURN n
                """
            )
            async for rec in res:
                n = rec["n"]
                items.append(
                    self._build_item(
                        namespace=tuple(n["namespace"]),
                        key=n["key"],
                        value=json.loads(n["value"]),
                        expires_at=n.get("expire_at"),
                    )
                )
        return items

    # ============================================================ #
    # CRUD: DELETE
    # ============================================================ #
    async def adelete(self, namespace: Tuple[str, ...], key: str) -> None:
        async with self._driver.session() as sess:  # type: ignore[attr-defined]
            async def _tx(tx):
                await (
                    await tx.run(
                        f"""
                        MATCH (n:{self.NODE_LABEL})
                        WHERE n.namespace = $ns AND n.key = $key
                        DETACH DELETE n
                        """,
                        ns=list(namespace),
                        key=key,
                    )
                ).consume()

            await sess.execute_write(_tx)

    async def adelete_namespace(self, namespace_prefix: Tuple[str, ...]) -> None:
        pred = self._cypher_ns_prefix_filter(namespace_prefix)
        async with self._driver.session() as sess:  # type: ignore[attr-defined]
            async def _tx(tx):
                await (
                    await tx.run(
                        f"""
                        MATCH (n:{self.NODE_LABEL})
                        WHERE {pred}
                        DETACH DELETE n
                        """
                    )
                ).consume()

            await sess.execute_write(_tx)

    # ============================================================ #
    # SEARCH (vector + lexical)
    # ============================================================ #
    async def asearch(
        self,
        namespace_prefix: Tuple[str, ...],
        *,
        query: str | Sequence[float] | None = None,
        filter: Mapping[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> List[Item]:
        pred = self._cypher_ns_prefix_filter(namespace_prefix)
        items: List[Item] = []

        # ---------- vector branch ----------
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
                WHERE {pred}
            """
            if filter:
                for fk, fv in filter.items():
                    cypher += f" AND node.{fk} = ${fk} "
                    params[fk] = fv
            cypher += """
                RETURN node, similarity
                ORDER BY similarity DESC
                LIMIT $k
            """
            async with self._driver.session() as sess:  # type: ignore[attr-defined]
                res = await sess.run(cypher, **params)
                rows = await res.data()
                for rec in rows[offset : offset + limit]:
                    n = rec["node"]
                    if refresh_ttl or (
                        refresh_ttl is None
                        and self._ttl_cfg
                        and self._ttl_cfg.refresh_on_read
                    ):
                        await self._refresh_node_ttl(tuple(n["namespace"]), n["key"])
                    items.append(
                        self._build_item(
                            namespace=tuple(n["namespace"]),
                            key=n["key"],
                            value=json.loads(n["value"]),
                            score=rec["similarity"],
                            expires_at=n.get("expire_at"),
                        )
                    )
            return items

        # ---------- lexical branch ----------
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        cypher = f"""
            MATCH (n:{self.NODE_LABEL})
            WHERE {pred}
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
        async with self._driver.session() as sess:  # type: ignore[attr-defined]
            res = await sess.run(cypher, **params)
            async for rec in res:
                n = rec["n"]
                if refresh_ttl or (
                    refresh_ttl is None
                    and self._ttl_cfg
                    and self._ttl_cfg.refresh_on_read
                ):
                    await self._refresh_node_ttl(tuple(n["namespace"]), n["key"])
                items.append(
                    self._build_item(
                        namespace=tuple(n["namespace"]),
                        key=n["key"],
                        value=json.loads(n["value"]),
                        expires_at=n.get("expire_at"),
                    )
                )
        return items

    # ============================================================ #
    # batch helper
    # ============================================================ #
    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        results = []
        for op in ops:
            if hasattr(op, "operation") and hasattr(op, "namespace") and hasattr(op, "key"):
                if op.operation == "get":
                    result = await self.aget(op.namespace, op.key)
                elif op.operation == "put":
                    result = await self.aput(op.namespace, op.key, op.value)
                elif op.operation == "delete":
                    await self.adelete(op.namespace, op.key)
                    result = None
                else:
                    raise ValueError(f"Unknown operation: {op.operation}")
                results.append(result)
            else:
                raise ValueError(f"Invalid operation format: {op}")
        return results

    # Sync façade for non‑async callers
    def batch(self, ops: Iterable[Any]) -> list[Any]:
        import asyncio

        return asyncio.run(self.abatch(ops))
