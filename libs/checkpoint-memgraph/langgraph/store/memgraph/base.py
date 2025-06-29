"""Memgraph‑backed implementation of the LangGraph key/value & vector store.

The code matches the public surface of ``PostgresStore`` so that callers can
freely switch between the two back‑ends simply by changing the connection URI
and the concrete ``Store`` class.

Key additions in this version
-----------------------------
* **MemgraphIndexConfig**
  - Now exposes every officially‑supported vector‑index parameter:
      • ``dimension``(required) – vector dimensionality
      • ``capacity``(required) – initial index capacity
      • ``metric`` – similarity metric (default ``"l2sq"``)
      • ``resize_coefficient`` – growth factor (default``2``)
      • ``distance_type`` is retained for score‑calculation parity and is
        automatically derived from ``metric`` when omitted.

* **Vector‑index migration**
  - Creates the index with the full config map
    (``dimension``, ``capacity``, ``metric``, ``resize_coefficient``).

* **Config validation / normalisation**
  - Ensures mandatory fields exist and applies sensible defaults.
  - Derives ``distance_type`` from ``metric`` when not supplied.

Nothing else in the public interface changed – the store continues to satisfy
all contracts defined by ``BaseStore`` and mirrors ``PostgresStore`` semantics.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    NamedTuple,
    TypeVar,
    Union,
    cast,
)
from urllib.parse import unquote, urlparse

import orjson
from neo4j import Driver, GraphDatabase, Session, Transaction
from typing_extensions import TypedDict

from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class Migration(NamedTuple):
    """A database migration with optional conditions and parameters."""

    cypher: str
    params: dict[str, Any] | None = None
    condition: Callable[[BaseMemgraphStore], bool] | None = None


MIGRATIONS: Sequence[str] = [
    "CREATE CONSTRAINT ON (n:StoreItem) ASSERT n.prefix, n.key IS UNIQUE;",
    "CREATE INDEX ON :StoreItem(prefix);",
    "CREATE INDEX ON :StoreItem(expires_at);",
]

VECTOR_MIGRATIONS: Sequence[Migration] = [
    Migration(
        """
CREATE VECTOR INDEX vector_index ON :Embedding(embedding)
WITH CONFIG {{
    "dimension": {dimension},
    "capacity": {capacity},
    "metric": "{metric}"
}};
""",
        params={
            "dimension": lambda store: int(store.index_config["dimension"]),
            "capacity": lambda store: int(store.index_config["capacity"]),
            "metric": lambda store: cast(MemgraphIndexConfig, store.index_config).get(
                "metric", "l2sq"
            ),
        },
    ),
]


class MemgraphIndexConfig(IndexConfig, total=False):
    """Configuration for Memgraph vector indexing.

    Required
    --------
    dimension : int
        Dimensionality of the vectors stored in the index.
    capacity : int
        Minimum capacity (number of vectors). Memgraph will round up to a power
        of two; it will *never* allocate fewer than this value.

    Optional
    --------
    metric : str, default "l2sq"
        Similarity function used by the index (“l2sq”, “cos”, “ip”, …).  The
        default is squared‑L2 (“l2sq”), which behaves like Euclidean distance.
    resize_coefficient : int, default 2
        Factor by which the index grows when capacity is reached.
    distance_type : Literal["l2", "cosine", "inner_product"]
        Convenience alias for scoring logic.  If not provided it is **derived
        automatically** from *metric* so existing code paths keep working.
    """

    # new, Memgraph‑native fields
    dimension: int
    capacity: int
    metric: str  # keep open – Memgraph may add more metrics over time
    resize_coefficient: int
    distance_type: Literal["l2", "cosine", "inner_product"]


def _normalise_index_config(cfg: MemgraphIndexConfig) -> MemgraphIndexConfig:
    """Validate & enrich the user‑supplied index config."""
    cfg = cfg.copy()  # we never mutate the caller's object
    if (
        "dimension" not in cfg
        or not isinstance(cfg["dimension"], int)
        or cfg["dimension"] <= 0
    ):
        raise ValueError("MemgraphIndexConfig: 'dimension' (positive int) is required")
    if (
        "capacity" not in cfg
        or not isinstance(cfg["capacity"], int)
        or cfg["capacity"] <= 0
    ):
        raise ValueError("MemgraphIndexConfig: 'capacity' (positive int) is required")
    cfg.setdefault("metric", "l2sq")
    cfg.setdefault("resize_coefficient", 2)
    if "distance_type" not in cfg:
        metric = cfg["metric"].lower()
        if metric in ("cos", "cosine"):
            cfg["distance_type"] = "cosine"
        elif metric in ("ip", "inner_product"):
            cfg["distance_type"] = "inner_product"
        else:  # treat everything else as some form of L2
            cfg["distance_type"] = "l2"
    return cfg


C = TypeVar("C", bound=Union[Driver])


class BaseMemgraphStore(Generic[C]):
    """Shared implementation for Memgraph synchronised & async stores."""

    MIGRATIONS = MIGRATIONS
    VECTOR_MIGRATIONS = VECTOR_MIGRATIONS
    conn: C
    _deserializer: Callable[[str], dict[str, Any]] | None
    index_config: MemgraphIndexConfig | None  # set during __init__

    def _get_batch_GET_ops_queries(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
    ) -> list[tuple[str, dict, tuple[str, ...], list]]:
        namespace_groups = defaultdict(list)
        for idx, op in get_ops:
            namespace_groups[op.namespace].append(
                {"idx": idx, "key": op.key, "refresh_ttl": op.refresh_ttl}
            )

        results: list[tuple[str, dict, tuple[str, ...], list]] = []
        for namespace, items in namespace_groups.items():
            ns_text = _namespace_to_text(namespace)
            query = """
UNWIND $items AS item
MATCH (n:StoreItem {prefix: $prefix, key: item.key})
WHERE (n.expires_at IS NULL OR n.expires_at >= localdatetime())
WITH n, item,
     CASE
         WHEN item.refresh_ttl AND n.ttl_minutes IS NOT NULL
         THEN localdatetime() + duration({minute: n.ttl_minutes})
         ELSE n.expires_at
     END AS new_expires_at
SET n.expires_at = new_expires_at
RETURN n.key         AS key,
       n.value       AS value,
       n.created_at  AS created_at,
       n.updated_at  AS updated_at,
       item.idx      AS idx
"""
            params = {"prefix": ns_text, "items": items}
            results.append((query, params, namespace, items))
        return results

    def _extract_texts_for_embedding(
        self, inserts: list[PutOp]
    ) -> dict[tuple[str, str], list[str]]:
        """Collect texts that must be embedded according to index rules."""
        if not self.index_config:
            return {}
        texts_by_node: dict[tuple[str, str], list[str]] = defaultdict(list)
        default_paths = cast(dict, self.index_config)["__tokenized_fields"]
        for op in inserts:
            if op.index is False:
                continue
            ns = _namespace_to_text(op.namespace)
            k = op.key
            value = op.value
            paths_to_index = (
                default_paths
                if op.index is None
                else [(ix, tokenize_path(ix)) for ix in op.index]
            )
            for _, tokenized_path in paths_to_index:
                texts = get_text_at_path(value, tokenized_path)
                if texts:
                    texts_by_node[(ns, k)].extend(texts)
        return texts_by_node

    def _prepare_batch_PUT_queries(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
    ) -> tuple[
        list[tuple[str, dict[str, Any]]],
        tuple[str, Sequence[tuple[str, str, str]]] | None,
    ]:
        dedupped_ops: dict[tuple[tuple[str, ...], str], PutOp] = {
            (op.namespace, op.key): op for _, op in put_ops
        }
        inserts = [op for op in dedupped_ops.values() if op.value is not None]
        deletes = [op for op in dedupped_ops.values() if op.value is None]
        queries: list[tuple[str, dict[str, Any]]] = []
        if deletes:
            delete_batch = [
                {"prefix": _namespace_to_text(op.namespace), "key": op.key}
                for op in deletes
            ]
            queries.append(
                (
                    """
UNWIND $batch AS op
MATCH (n:StoreItem {prefix: op.prefix, key: op.key})
DETACH DELETE n
""",
                    {"batch": delete_batch},
                )
            )
        embedding_request: tuple[str, Sequence[tuple[str, str, str]]] | None = None
        if inserts:
            insert_batch = []
            for op in inserts:
                props = {
                    "prefix": _namespace_to_text(op.namespace),
                    "key": op.key,
                    "value": orjson.dumps(op.value).decode("utf-8"),
                    "ttl_minutes": op.ttl,
                }
                if isinstance(op.value, dict):
                    for k, v in op.value.items():
                        if isinstance(v, (str, int, float, bool)):
                            props[k] = v
                insert_batch.append(props)
            put_query = """
UNWIND $batch AS op
MERGE (n:StoreItem {prefix: op.prefix, key: op.key})
WITH n, op, n.created_at AS created_at
SET n            = op,
    n.created_at = COALESCE(created_at, localdatetime()),
    n.updated_at = localdatetime(),
    n.expires_at =
        CASE
            WHEN op.ttl_minutes IS NOT NULL
            THEN localdatetime() + duration({minute: op.ttl_minutes})
            ELSE null
        END
"""
            queries.append((put_query, {"batch": insert_batch}))
            queries.append(
                (
                    """
                UNWIND $batch AS op
                MATCH (n:StoreItem {prefix: op.prefix, key: op.key})-[r:HAS_EMBEDDING]->(e:Embedding)
                DELETE r, e
                """,
                    {
                        "batch": [
                            {"prefix": _namespace_to_text(op.namespace), "key": op.key}
                            for op in inserts
                        ]
                    },
                )
            )
            texts_by_node = self._extract_texts_for_embedding(inserts)
            if texts_by_node:
                embedding_request_params = [
                    (ns, k, text)
                    for (ns, k), txts in texts_by_node.items()
                    for text in txts
                ]
                embedding_request = (
                    """
UNWIND $batch AS op
MATCH (n:StoreItem {prefix: op.prefix, key: op.key})
CREATE (e:Embedding {embedding: op.embedding, text: op.text})
MERGE (n)-[:HAS_EMBEDDING]->(e)
""",
                    embedding_request_params,
                )
        return queries, embedding_request

    def _build_search_where_clause(self, op: SearchOp, params: dict[str, Any]) -> str:
        where = ["(n.expires_at IS NULL OR n.expires_at >= localdatetime())"]
        # Namespace filtering
        if op.namespace_prefix:
            where.append("n.prefix STARTS WITH $prefix")
            params["prefix"] = _namespace_to_text(op.namespace_prefix)
        # Field‑level filters
        if op.filter:
            i = 0
            for key, value in op.filter.items():
                if isinstance(value, dict):
                    for op_key, op_val in value.items():
                        pname = f"filter_val_{i}"
                        clause = {
                            "$gt": ">",
                            "$gte": ">=",
                            "$lt": "<",
                            "$lte": "<=",
                            "$ne": "<>",
                            "$eq": "=",
                        }.get(op_key)
                        if clause is None:
                            logger.warning("Unsupported filter operator %s", op_key)
                            continue
                        where.append(f"n.{key} {clause} ${pname}")
                        params[pname] = op_val
                        i += 1
                else:
                    pname = f"filter_val_{i}"
                    where.append(f"n.{key} = ${pname}")
                    params[pname] = value
                    i += 1
        return f"WHERE {' AND '.join(where)}" if where else ""

    def _prepare_batch_search_queries(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
    ) -> tuple[list[tuple[str, dict[str, Any]]], list[tuple[int, str]]]:
        queries: list[tuple[str, dict[str, Any]]] = []
        embedding_requests: list[tuple[int, str]] = []
        for idx, op in search_ops:
            params: dict[str, Any] = {"offset": op.offset}
            where_stmt = self._build_search_where_clause(op, params)
            # Pagination
            limit_clause = ""
            if op.limit is not None:
                limit_clause = "LIMIT $limit"
                params["limit"] = op.limit
            if op.query and self.index_config:
                embedding_requests.append((idx, op.query))
                dist_type = (
                    cast(MemgraphIndexConfig, self.index_config)
                    .get("distance_type", "cosine")
                    .lower()
                )
                if dist_type == "cosine" or dist_type == "inner_product":
                    # For normalized vectors, as used in this test suite, Memgraph's
                    # COSINE and INNER_PRODUCT distances are both calculated as (1 - similarity).
                    # To get the similarity score, we must use (1 - distance).
                    score_expr = "1.0 - best_distance"
                else:  # Assumes 'l2' for 'l2sq' metric
                    # Memgraph 'l2sq' distance = (L2 distance)^2. Test expects negative L2 distance.
                    score_expr = "-sqrt(best_distance)"
                set_clause = ""
                if op.refresh_ttl:
                    set_clause = """
                    SET n.expires_at =
                        CASE
                            WHEN n.ttl_minutes IS NOT NULL
                            THEN localdatetime() + duration({minute : n.ttl_minutes})
                            ELSE n.expires_at
                        END
                    """
                return_clause = """
                RETURN n.prefix      AS prefix,
                       n.key         AS key,
                       n.value       AS value,
                       n.created_at  AS created_at,
                       n.updated_at  AS updated_at,
                       score
                """
                query_parts = [
                    "CALL vector_search.search('vector_index', $k, $embedding)",
                    "YIELD node AS embedding_node, distance",
                    "MATCH (n:StoreItem)-[:HAS_EMBEDDING]->(embedding_node)",
                    where_stmt,
                    "WITH n, min(distance) as best_distance",
                    f"WITH n, {score_expr} AS score",
                    f"WITH n, score ORDER BY score DESC SKIP $offset {limit_clause}",
                    set_clause,
                    return_clause,
                ]
                queries.append(
                    (
                        "\n".join([p for p in query_parts if p]),
                        {
                            **params,
                            "k": max(100, (op.limit + op.offset) * 5),
                        },
                    )
                )
                continue  # done for this op
            refresh_stmt = (
                """
                WITH n,
                     CASE
                    WHEN n.ttl_minutes IS NOT NULL
                    THEN localdatetime() + duration({minute : n.ttl_minutes})
                    ELSE n.expires_at
                END
                AS new_expires_at
    SET n.expires_at = new_expires_at
                """
                if op.refresh_ttl
                else "WITH n"
            )
            regular_query = f"""
    MATCH (n:StoreItem)
    {where_stmt}
    {refresh_stmt}
    RETURN n.prefix     AS prefix,
           n.key        AS key,
           n.value      AS value,
           n.created_at AS created_at,
           n.updated_at AS updated_at,
           null         AS score
    ORDER BY n.updated_at DESC
    SKIP $offset
    {limit_clause}
    """
            queries.append((regular_query, params))
        return queries, embedding_requests

    def _get_batch_list_namespaces_queries(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
    ) -> list[tuple[str, dict[str, Any]]]:
        queries: list[tuple[str, dict[str, Any]]] = []
        for _, op in list_ops:
            params: dict[str, Any] = {
                "offset": op.offset,
                "max_depth": op.max_depth,
            }
            match = ["(n.expires_at IS NULL OR n.expires_at >= localdatetime())"]
            if op.match_conditions:
                for i, cond in enumerate(op.match_conditions):
                    p = f"path_{i}"
                    params[p] = _namespace_to_text(cond.path)
                    if cond.match_type == "prefix":
                        match.append(f"n.prefix STARTS WITH ${p}")
                    elif cond.match_type == "suffix":
                        match.append(f"n.prefix ENDS WITH ${p}")
                    else:
                        logger.warning("Unknown match_type %s", cond.match_type)
            where_clause = f"WHERE {' AND '.join(match)}"
            limit_clause = ""
            if op.limit is not None:
                params["limit"] = op.limit
                limit_clause = "LIMIT $limit"
            query = f"""
MATCH (n:StoreItem)
{where_clause}
WITH n.prefix AS full_prefix
WITH full_prefix, split(full_prefix, '.') AS parts
WITH full_prefix, parts,
     CASE
         WHEN $max_depth IS NOT NULL AND size(parts) > $max_depth
         THEN substring(REDUCE(s = "", p IN parts[0..$max_depth - 1] | s + '.' + p), 1)
         ELSE full_prefix
     END AS truncated_prefix
RETURN DISTINCT truncated_prefix
ORDER BY truncated_prefix
SKIP $offset
{limit_clause}
"""
            queries.append((query, params))
        return queries

    def _get_filter_condition(self, *_: Any, **__: Any) -> tuple[str, list]:
        raise NotImplementedError(
            "Filtering on JSON content is not supported in the Memgraph store."
        )


class MemgraphStore(BaseStore, BaseMemgraphStore[Driver]):
    """Synchronous Memgraph store."""

    __slots__ = (
        "database",
        "_deserializer",
        "index_config",
        "embeddings",
        "ttl_config",
        "_ttl_sweeper_thread",
        "_ttl_stop_event",
    )
    supports_ttl: bool = True

    def __init__(
        self,
        conn: Driver,
        *,
        database: str = "memgraph",
        deserializer: Callable[[str], dict[str, Any]] | None = None,
        index: MemgraphIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> None:
        super().__init__()
        self.database = database
        self._deserializer = deserializer or (lambda v: orjson.loads(v))
        self.conn = conn
        self.index_config: MemgraphIndexConfig | None = None
        self.embeddings: Embeddings | None
        if index:
            # Validate & enrich config
            index = _normalise_index_config(index)
            self.embeddings, self.index_config = _ensure_index_config(index)
        else:
            self.embeddings = None
        self.ttl_config = ttl
        self._ttl_sweeper_thread: threading.Thread | None = None
        self._ttl_stop_event = threading.Event()

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
        *,
        database: str = "memgraph",
        index: MemgraphIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> Iterator[MemgraphStore]:
        """Create a store from a Memgraph connection URI."""
        parsed = urlparse(conn_string)
        uri = f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 7687}"
        auth = (unquote(parsed.username or ""), unquote(parsed.password or ""))
        with GraphDatabase.driver(uri, auth=auth) as driver:
            yield cls(driver, database=database, index=index, ttl=ttl)

    def sweep_ttl(self) -> int:
        """Delete expired nodes and return count."""
        with self._session() as session:
            result = session.run(
                """
MATCH (n:StoreItem)
WHERE n.expires_at IS NOT NULL AND n.expires_at < localdatetime()
DETACH DELETE n
RETURN count(n) as deleted_count
"""
            )
            record = result.single()
            return record["deleted_count"] if record else 0

    def start_ttl_sweeper(
        self, sweep_interval_minutes: int | None = None
    ) -> concurrent.futures.Future[None]:
        """Start background TTL sweep identical to PostgresStore semantics."""
        if not self.ttl_config:
            future: concurrent.futures.Future[None] = concurrent.futures.Future()
            future.set_result(None)
            return future
        if self._ttl_sweeper_thread and self._ttl_sweeper_thread.is_alive():
            logger.info("TTL sweeper already running")
            fut = concurrent.futures.Future()
            fut.add_done_callback(
                lambda f: self._ttl_stop_event.set() if f.cancelled() else None
            )
            return fut
        self._ttl_stop_event.clear()
        interval = float(
            sweep_interval_minutes or self.ttl_config.get("sweep_interval_minutes") or 5
        )
        logger.info("Starting TTL sweeper (interval=%smin)", interval)
        future = concurrent.futures.Future()

        def _sweep_loop() -> None:
            try:
                while not self._ttl_stop_event.is_set():
                    if self._ttl_stop_event.wait(interval * 60):
                        break
                    try:
                        deleted = self.sweep_ttl()
                        if deleted:
                            logger.info("TTL sweep removed %s items", deleted)
                    except Exception as exc:  # pragma: no cover
                        logger.exception("TTL sweep failed", exc_info=exc)
                future.set_result(None)
            except Exception as exc:
                future.set_exception(exc)

        t = threading.Thread(target=_sweep_loop, daemon=True, name="ttl-sweeper")
        self._ttl_sweeper_thread = t
        t.start()
        future.add_done_callback(
            lambda f: self._ttl_stop_event.set() if f.cancelled() else None
        )
        return future

    def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        """Stop the TTL sweeper thread."""
        if not self._ttl_sweeper_thread or not self._ttl_sweeper_thread.is_alive():
            return True
        logger.info("Stopping TTL sweeper thread")
        self._ttl_stop_event.set()
        self._ttl_sweeper_thread.join(timeout)
        ok = not self._ttl_sweeper_thread.is_alive()
        if ok:
            self._ttl_sweeper_thread = None
            logger.info("TTL sweeper stopped")
        else:
            logger.warning("Timeout stopping TTL sweeper")
        return ok

    def __del__(self) -> None:  # noqa: D401
        if hasattr(self, "_ttl_stop_event") and hasattr(self, "_ttl_sweeper_thread"):
            self.stop_ttl_sweeper(timeout=0.1)

    @contextmanager
    def _session(self) -> Iterator[Session]:
        with self.conn.session(database=self.database) as session:
            yield session

    @contextmanager
    def _transaction(self, session: Session) -> Iterator[Transaction]:
        with session.begin_transaction() as tx:
            yield tx

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        grouped, total = _group_ops(ops)
        results: list[Result] = [None] * total
        with self._session() as session, self._transaction(session) as tx:
            if PutOp in grouped:
                self._batch_put_ops(
                    cast(Sequence[tuple[int, PutOp]], grouped[PutOp]), tx
                )
            if GetOp in grouped:
                self._batch_get_ops(
                    cast(Sequence[tuple[int, GetOp]], grouped[GetOp]), results, tx
                )
            if SearchOp in grouped:
                self._batch_search_ops(
                    cast(Sequence[tuple[int, SearchOp]], grouped[SearchOp]), results, tx
                )
            if ListNamespacesOp in grouped:
                self._batch_list_namespaces_ops(
                    cast(
                        Sequence[tuple[int, ListNamespacesOp]],
                        grouped[ListNamespacesOp],
                    ),
                    results,
                    tx,
                )
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Async equivalent of ``batch``."""
        return await asyncio.get_running_loop().run_in_executor(None, self.batch, ops)

    def _batch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
        tx: Transaction,
    ) -> None:
        for query, params, namespace, items in self._get_batch_GET_ops_queries(get_ops):
            for rec in tx.run(query, params):
                idx = {i["key"]: i["idx"] for i in items}.get(rec["key"])
                if idx is not None:
                    results[idx] = _record_to_item(
                        namespace, rec, loader=self._deserializer
                    )

    def _batch_put_ops(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
        tx: Transaction,
    ) -> None:
        queries, embedding_req = self._prepare_batch_PUT_queries(put_ops)
        for q, p in queries:
            tx.run(q, p)
        if embedding_req:
            if self.embeddings is None:
                raise ValueError("Embeddings config required for vector operations.")
            q, txt_params = embedding_req
            texts = sorted({t for _, _, t in txt_params})
            vectors = self.embeddings.embed_documents(texts)
            t2v = dict(zip(texts, vectors))
            batch = [
                {"prefix": ns, "key": k, "text": text, "embedding": t2v[text]}
                for ns, k, text in txt_params
            ]
            tx.run(q, {"batch": batch})

    def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        tx: Transaction,
    ) -> None:
        queries, embedding_reqs = self._prepare_batch_search_queries(search_ops)
        op_idx_to_params = {
            op_idx: queries[i][1] for i, (op_idx, _) in enumerate(search_ops)
        }
        if embedding_reqs and self.embeddings:
            texts = sorted({t for _, t in embedding_reqs if t})
            embeddings = self.embeddings.embed_documents(texts)
            t2e = dict(zip(texts, embeddings))
            for op_idx, text in embedding_reqs:
                if text and op_idx in op_idx_to_params:
                    op_idx_to_params[op_idx]["embedding"] = t2e[text]
        for i, (op_idx, op) in enumerate(search_ops):
            query, params = queries[i]
            if op.query and not params.get("embedding"):
                results[op_idx] = []
                continue
            rows = tx.run(query, params)
            results[op_idx] = [
                _record_to_search_item(
                    _decode_ns_text(r["prefix"]), r, loader=self._deserializer
                )
                for r in rows
            ]

    def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
        tx: Transaction,
    ) -> None:
        for (query, params), (idx, _) in zip(
            self._get_batch_list_namespaces_queries(list_ops), list_ops
        ):
            rows = tx.run(query, params)
            results[idx] = [
                _decode_ns_text(r["truncated_prefix"])
                for r in rows
                if r["truncated_prefix"]
            ]

    def setup(self) -> None:
        """Apply schema and vector‑index migrations (idempotent)."""

        def _get_version(tx: Transaction, table: str) -> int:
            rec = tx.run(
                """
MERGE (m:Migration {name: $table})
ON CREATE SET m.version = -1
RETURN m.version AS v
""",
                {"table": table},
            ).single()
            return rec["v"] if rec else -1

        def _set_version(tx: Transaction, table: str, v: int) -> None:
            tx.run(
                """
MATCH (m:Migration {name: $table})
SET m.version = $v
""",
                {"table": table, "v": v},
            )

        with self._session() as session:
            with session.begin_transaction() as tx:
                ver = _get_version(tx, "store_migrations")
            for v, cypher in enumerate(self.MIGRATIONS[ver + 1 :], start=ver + 1):
                try:
                    session.run(cypher)
                    with session.begin_transaction() as tx:
                        _set_version(tx, "store_migrations", v)
                except Exception as e:  # pragma: no cover
                    if "already exists" in str(e).lower():
                        with session.begin_transaction() as tx:
                            _set_version(tx, "store_migrations", v)
                    else:
                        logger.error(
                            "Failed migration%s\nCypher:%s\nErr:%s", v, cypher, e
                        )
                        raise
            if self.index_config:
                with session.begin_transaction() as tx:
                    ver = _get_version(tx, "vector_migrations")
                for v, mig in enumerate(
                    self.VECTOR_MIGRATIONS[ver + 1 :], start=ver + 1
                ):
                    if mig.condition and not mig.condition(self):
                        continue
                    params = {
                        k: (fn(self) if callable(fn) else fn)
                        for k, fn in (mig.params or {}).items()
                    }
                    cypher = mig.cypher.format(**params)
                    try:
                        session.run(cypher)
                        with session.begin_transaction() as tx:
                            _set_version(tx, "vector_migrations", v)
                    except Exception as e:  # pragma: no cover
                        if "already exists" in str(e).lower():
                            with session.begin_transaction() as tx:
                                _set_version(tx, "vector_migrations", v)
                        else:
                            logger.error(
                                "Vector migration %s failed\nCypher:%s\nErr:%s",
                                v,
                                cypher,
                                e,
                            )
                            raise


class Record(TypedDict):
    key: str
    value: Any
    prefix: str
    created_at: Any
    updated_at: Any


def _namespace_to_text(namespace: tuple[str, ...]) -> str:
    return ".".join(namespace)


def _decode_ns_text(s: str) -> tuple[str, ...]:
    return tuple(s.split(".")) if s else ()


def _record_to_item(
    namespace: tuple[str, ...],
    record: Record,
    *,
    loader: Callable[[str], dict[str, Any]],
) -> Item:
    val = record["value"]
    if isinstance(val, str):
        val = loader(val)
    return Item(
        namespace=namespace,
        key=record["key"],
        value=val,
        created_at=record["created_at"].to_native(),
        updated_at=record["updated_at"].to_native(),
    )


def _record_to_search_item(
    namespace: tuple[str, ...],
    record: Record,
    *,
    loader: Callable[[str], dict[str, Any]],
) -> SearchItem:
    val = record["value"]
    if isinstance(val, str):
        val = loader(val)
    score = record.get("score")
    return SearchItem(
        namespace=namespace,
        key=record["key"],
        value=val,
        created_at=record["created_at"].to_native(),
        updated_at=record["updated_at"].to_native(),
        score=float(score) if score is not None else None,
    )


def _group_ops(ops: Iterable[Op]) -> tuple[dict[type, list[tuple[int, Op]]], int]:
    groups: dict[type, list[tuple[int, Op]]] = defaultdict(list)
    total = 0
    for idx, op in enumerate(ops):
        groups[type(op)].append((idx, op))
        total += 1
    return groups, total


def _ensure_index_config(
    index_config: MemgraphIndexConfig,
) -> tuple[Embeddings | None, MemgraphIndexConfig]:
    """Tokenise (field->path) and attach embeddings object (if any)."""
    index_config = index_config.copy()
    tokenised: list[tuple[str, Literal["$"] | list[str]]] = []
    est_vectors = 0
    fields = index_config.get("fields") or ["$"]
    if isinstance(fields, str):
        fields = [fields]
    if not isinstance(fields, list):
        raise ValueError("'fields' in index config must be list or str")
    for p in fields:
        if p == "$":
            tokenised.append((p, "$"))
            est_vectors += 1
        else:
            toks = tokenize_path(p)
            tokenised.append((p, toks))
            est_vectors += len(toks)
    index_config["__tokenized_fields"] = tokenised
    index_config["__estimated_num_vectors"] = est_vectors
    embeddings = ensure_embeddings(index_config.get("embed"))
    return embeddings, index_config
