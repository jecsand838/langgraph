"""
Synchronous Memgraph key–value store used by LangGraph memory components.

Key refactor features
--------------------
* Safe Cypher parameter binding – avoids reserved names such as ``query``.
* Modular hybrid search:
    - ``_vector_search`` handles semantic / vector retrieval.
    - ``_lexical_search`` handles keyword / fallback retrieval.
* Automatic degradation: if no embedder or vector index is available,
  text queries fall back to lexical search instead of raising.
* Thorough type checking and clear error messages for unsupported inputs.
"""

from __future__ import annotations

import inspect
import json
import logging
import threading
from dataclasses import dataclass, fields, is_dataclass
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
    Union,
)

from neo4j import GraphDatabase, Session, Transaction
from neo4j.exceptions import ClientError

from langgraph.store.base import (  # type: ignore[attr-defined]
    BaseStore,
    Item,
)

from ._utils import parse_bolt_uri

__all__ = ["MemgraphStore"]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Vector‑index helper
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _VectorIndexConfig:
    dims: int
    metric: str = "cos"
    name: str = "memory_embeddings"
    capacity: int = 1_000_000
    embed: Optional[Any] = None  # object with embed_query / embed_documents


# --------------------------------------------------------------------------- #
# Connection mix‑in shared by sync & async stores
# --------------------------------------------------------------------------- #
class _MemgraphStoreConnMixin:
    """
    Helpers common to both the synchronous and asynchronous Memgraph stores.
    """

    NODE_LABEL: str = "MemoryEntry"

    # ------------------------------------------------------------------ #
    # Robust Item factory (handles multiple LangGraph versions)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_item(
        *,
        namespace: Tuple[str, ...],
        key: str,
        value: Any,
        score: Optional[float] = None,
        expires_at: Optional[str] = None,
    ) -> Item:
        """Instantiate an :class:`langgraph.store.base.Item` compatible with all
        LangGraph releases."""
        # Detect supported/required fields
        if is_dataclass(Item):
            item_field_names = {f.name for f in fields(Item)}
        else:
            hints = getattr(Item, "__annotations__", {})
            item_field_names = set(hints) | set(dir(Item))

        kwargs: Dict[str, Any] = {
            "namespace": namespace,
            "key": key,
            "value": value,
        }
        if score is not None and "score" in item_field_names:
            kwargs["score"] = score

        if expires_at is not None:
            for cand in ("expires_at", "expires", "expiry"):
                if cand in item_field_names:
                    kwargs[cand] = expires_at
                    break

        # Provide defaults for *required* keyword‑only params
        sig = inspect.signature(Item)
        for name, param in sig.parameters.items():
            if (
                param.kind is inspect.Parameter.KEYWORD_ONLY
                and param.default is inspect.Parameter.empty
                and name not in kwargs
            ):
                kwargs[name] = (
                    datetime.now(tz=timezone.utc).isoformat()
                    if name.endswith("_at") or "time" in name
                    else None
                )

        return Item(**kwargs)  # type: ignore[arg-type]

    # ----------------------- helper: tolerant run ----------------------- #
    @staticmethod
    def _run_safe(session: Session, cypher: str) -> None:
        """Execute Cypher but ignore duplicate‑schema errors (idempotency)."""
        try:
            session.run(cypher)
        except ClientError as exc:  # pragma: no cover
            msg = str(exc).lower()
            if any(w in msg for w in ("already exists", "duplicate", "existing")):
                return
            raise

    # -------------------- helper: namespace predicate ------------------- #
    @staticmethod
    def _cypher_ns_prefix_filter(prefix: Tuple[str, ...]) -> str:
        base_pred = [
            f"size(n.namespace) >= {len(prefix)}",
            "(n.expire_at IS NULL OR n.expire_at >= datetime())",
        ]
        for idx, part in enumerate(prefix):
            base_pred.append(f'n.namespace[{idx}] = "{part}"')
        return " AND ".join(base_pred)

    # ------------------- helper: vector‑index statement ----------------- #
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

    # ------------------------ schema initialisation --------------------- #
    def _create_schema(self, session: Session, vcfg: _VectorIndexConfig | None) -> None:
        self._run_safe(
            session,
            f"""
            CREATE CONSTRAINT ON (n:{self.NODE_LABEL})
            ASSERT n.namespace, n.key IS UNIQUE
            """,
        )
        self._run_safe(session, f"CREATE INDEX ON :{self.NODE_LABEL}(expire_at)")
        if vcfg:
            self._run_safe(session, self._vector_index_cypher(vcfg))


# =========================================================================== #
#                              S Y N C   S T O R E                           #
# =========================================================================== #
class MemgraphStore(_MemgraphStoreConnMixin, BaseStore):  # type: ignore[misc]
    """
    Synchronous Memgraph‑backed store with hybrid (vector + lexical) search.

    ``search`` decides on vector vs. lexical path automatically:

    * *Text + embedder*  → semantic vector search.
    * *Text (no embedder)* → lexical ``CONTAINS`` search.
    * *Vector input*     → direct vector similarity search.
    * *None*             → list items in namespace.
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
    ):
        self._driver = GraphDatabase.driver(
            bolt_uri, auth=(user, password), **(driver_kwargs or {})
        )
        if node_label:
            self.NODE_LABEL = str(node_label)

        # Optional vector index configuration
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
        self._ttl_thread: threading.Thread | None = None
        self._ttl_stop_evt = threading.Event()

    # --------------------------- builders -------------------------------- #
    @classmethod
    def from_conn_string(cls, conn: str, **kwargs: Any) -> "MemgraphStore":
        parsed = parse_bolt_uri(conn)
        return cls(
            parsed["bolt_uri"],
            user=parsed["user"],
            password=parsed["password"],
            **kwargs,
        )  # type: ignore[arg-type]

    # --------------------------- context -------------------------------- #
    def __enter__(self) -> "MemgraphStore":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # batch helpers (basic)
    # ------------------------------------------------------------------ #
    def batch(self, ops: Iterable[Any]) -> list[Any]:
        results = []
        for op in ops:
            if hasattr(op, "operation") and hasattr(op, "namespace") and hasattr(op, "key"):
                if op.operation == "get":
                    result = self.get(op.namespace, op.key)
                elif op.operation == "put":
                    result = self.put(op.namespace, op.key, op.value)
                elif op.operation == "delete":
                    self.delete(op.namespace, op.key)
                    result = None
                else:
                    raise ValueError(f"Unknown operation: {op.operation}")
                results.append(result)
            else:
                raise ValueError(f"Invalid operation format: {op}")
        return results

    def abatch(self, ops: Iterable[Any]) -> list[Any]:
        import asyncio

        return asyncio.run(self._abatch_impl(ops))

    async def _abatch_impl(self, ops: Iterable[Any]) -> list[Any]:
        return self.batch(ops)

    # ------------------------------------------------------------------ #
    # driver lifecycle / setup
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        if self._ttl_thread and self._ttl_thread.is_alive():
            self._ttl_stop_evt.set()
            self._ttl_thread.join(timeout=5)
        self._driver.close()

    def setup(self) -> None:
        if self._setup_done:
            return
        with self._driver.session() as sess:
            self._create_schema(sess, self._vector_cfg)
        self._setup_done = True
        if self._ttl_cfg and self._ttl_cfg.sweep_interval_minutes:
            self._start_ttl_sweeper()

    initialise = setup  # alias

    # ------------------------------------------------------------------ #
    # TTL sweeper (optional background thread)
    # ------------------------------------------------------------------ #
    def _start_ttl_sweeper(self) -> None:
        if self._ttl_thread and self._ttl_thread.is_alive():
            return
        if not self._ttl_cfg or not self._ttl_cfg.sweep_interval_minutes:
            return

        def _loop() -> None:
            interval = self._ttl_cfg.sweep_interval_minutes * 60
            while not self._ttl_stop_evt.wait(interval):
                try:
                    self.sweep_ttl()
                except Exception:  # pragma: no cover
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

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _expiry_dt(self, ttl_minutes: float | None) -> Optional[str]:
        if ttl_minutes is None and self._ttl_cfg:
            ttl_minutes = self._ttl_cfg.default_ttl
        if ttl_minutes is None:
            return None
        return (datetime.now(tz=timezone.utc) + timedelta(minutes=ttl_minutes)).isoformat()

    def _refresh_node_ttl(self, namespace: Tuple[str, ...], key: str) -> None:
        exp = self._expiry_dt(None)
        with self._driver.session() as sess:
            sess.run(
                f"""
                MATCH (n:{self.NODE_LABEL})
                WHERE n.namespace = $ns AND n.key = $key
                SET n.expire_at = datetime($exp)
                """,
                ns=list(namespace),
                key=key,
                exp=exp,
            )

    # ------------------------------------------------------------------ #
    # new: list distinct namespaces
    # ------------------------------------------------------------------ #
    def list_namespaces(self, namespace_prefix: Tuple[str, ...]) -> List[Tuple[str, ...]]:
        pred = self._cypher_ns_prefix_filter(namespace_prefix)
        with self._driver.session() as sess:
            res = sess.run(
                f"""
                MATCH (n:{self.NODE_LABEL})
                WHERE {pred}
                RETURN DISTINCT n.namespace AS ns
                """
            )
            return [tuple(r["ns"]) for r in res]

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #
    def put(
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
                    exp=exp,
                    embedding=embed_vec,
                )

            sess.execute_write(_tx)

        return self._build_item(
            namespace=namespace,
            key=key,
            value=value,
            expires_at=exp,
        )

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

    # ------------------------------------------------------------------ #
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
        if refresh_ttl or (
            refresh_ttl is None and self._ttl_cfg and self._ttl_cfg.refresh_on_read
        ):
            self._refresh_node_ttl(namespace, key)
            node["expire_at"] = self._expiry_dt(None)

        return self._build_item(
            namespace=tuple(node["namespace"]),
            key=node["key"],
            value=json.loads(node["value"]),
            expires_at=node.get("expire_at"),
        )

    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
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
                    self._build_item(
                        namespace=tuple(n["namespace"]),
                        key=n["key"],
                        value=json.loads(n["value"]),
                        expires_at=n.get("expire_at"),
                    )
                )
        return items

    # ------------------------------------------------------------------ #
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

    # ============================================================ #
    # SEARCH – public dispatcher
    # ============================================================ #
    def search(
        self,
        namespace_prefix: Tuple[str, ...],
        *,
        query: Union[str, Sequence[float], None] = None,
        filter: Mapping[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> List[Item]:
        """Hybrid lexical / semantic search (see class docstring)."""
        if isinstance(query, str):
            if self._vector_cfg and self._vector_cfg.embed:
                # Semantic path
                return self._vector_search(
                    namespace_prefix,
                    text=query,
                    filter=filter,
                    limit=limit,
                    offset=offset,
                    refresh_ttl=refresh_ttl,
                )
            # Lexical fallback
            return self._lexical_search(
                namespace_prefix,
                term=query,
                filter=filter,
                limit=limit,
                offset=offset,
                refresh_ttl=refresh_ttl,
            )

        if isinstance(query, Sequence) and not isinstance(query, (str, bytes)):
            if not self._vector_cfg:
                raise ValueError("Vector query supplied but no vector index configured.")
            return self._vector_search(
                namespace_prefix,
                vector=list(query),
                filter=filter,
                limit=limit,
                offset=offset,
                refresh_ttl=refresh_ttl,
            )
        # query is None → list
        return self._lexical_search(
            namespace_prefix,
            term=None,
            filter=filter,
            limit=limit,
            offset=offset,
            refresh_ttl=refresh_ttl,
        )

    # ============================================================ #
    # Internal helpers – search variants
    # ============================================================ #
    def _vector_search(
        self,
        namespace_prefix: Tuple[str, ...],
        *,
        text: Optional[str] = None,
        vector: Optional[List[float]] = None,
        filter: Mapping[str, Any] | None,
        limit: int,
        offset: int,
        refresh_ttl: bool | None,
    ) -> List[Item]:
        """Vector similarity search (text is embedded if provided)."""
        if not self._vector_cfg:
            raise RuntimeError("Vector search requested but no vector index configured.")

        if vector is not None and text is not None:
            raise ValueError("Provide either *text* or *vector* (not both).")

        if text is not None:
            if not self._vector_cfg.embed:
                raise RuntimeError("Embedder not configured.")
            vector = self._vector_cfg.embed.embed_query(text)  # type: ignore[attr-defined]
            if hasattr(vector, "tolist"):
                vector = vector.tolist()

        if vector is None:
            raise ValueError("Vector search requires a vector.")

        k = limit + offset
        params: Dict[str, Any] = {"vec": vector, "k": k}
        cypher = (
            f"""
            CALL vector_search.search("{self._vector_cfg.name}", $k, $vec)
            YIELD node, similarity
            WITH node, similarity
            WHERE {self._cypher_ns_prefix_filter(namespace_prefix)}
            """
        )
        if filter:
            for fk, fv in filter.items():
                cypher += f" AND node.{fk} = ${fk} "
                params[fk] = fv
        cypher += """
            RETURN node, similarity
            ORDER BY similarity DESC
            LIMIT $k
        """

        items: List[Item] = []
        with self._driver.session() as sess:
            rows = sess.run(cypher, parameters=params).data()
            for rec in rows[offset : offset + limit]:
                n = rec["node"]
                if refresh_ttl or (
                    refresh_ttl is None
                    and self._ttl_cfg
                    and self._ttl_cfg.refresh_on_read
                ):
                    self._refresh_node_ttl(tuple(n["namespace"]), n["key"])
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

    def _lexical_search(
        self,
        namespace_prefix: Tuple[str, ...],
        *,
        term: Optional[str],
        filter: Mapping[str, Any] | None,
        limit: int,
        offset: int,
        refresh_ttl: bool | None,
    ) -> List[Item]:
        """Simple substring search on the JSON stringified `value`."""
        params: Dict[str, Any] = {"lim": limit, "off": offset}
        cypher = (
            f"""
            MATCH (n:{self.NODE_LABEL})
            WHERE {self._cypher_ns_prefix_filter(namespace_prefix)}
            """
        )
        if term:
            cypher += " AND n.value CONTAINS $term "
            params["term"] = term
        if filter:
            for fk, fv in filter.items():
                cypher += f" AND n.{fk} = ${fk} "
                params[fk] = fv
        cypher += """
            RETURN n
            SKIP $off
            LIMIT $lim
        """

        items: List[Item] = []
        with self._driver.session() as sess:
            for rec in sess.run(cypher, parameters=params):
                n = rec["n"]
                if refresh_ttl or (
                    refresh_ttl is None
                    and self._ttl_cfg
                    and self._ttl_cfg.refresh_on_read
                ):
                    self._refresh_node_ttl(tuple(n["namespace"]), n["key"])
                items.append(
                    self._build_item(
                        namespace=tuple(n["namespace"]),
                        key=n["key"],
                        value=json.loads(n["value"]),
                        expires_at=n.get("expire_at"),
                    )
                )
        return items
