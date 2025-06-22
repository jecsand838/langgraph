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

if TYPE_CHECKING:
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
CREATE VECTOR INDEX vector_index ON :StoreItem(embedding) WITH CONFIG {{
    "dimension": {dims},
    "similarityFunction": "{similarity_function}",
    "capacity": {capacity}
}};
""",
        params={
            "dims": lambda store: store.index_config["dims"],
            "similarity_function": lambda store: {
                "l2": "EUCLIDEAN",
                "cosine": "COSINE",
                "inner_product": "COSINE",
            }.get(
                cast(MemgraphIndexConfig, store.index_config)
                .get("distance_type", "cosine")
                .lower(),
                "COSINE",
            ),
            "capacity": lambda store: cast(MemgraphIndexConfig, store.index_config)
            .get("ann_index_config", {})
            .get("capacity", 1000),
        },
    ),
]


C = TypeVar("C", bound=Union[Driver])


class MemgraphIndexConfig(IndexConfig, total=False):
    """Configuration for vector embeddings in Memgraph store."""

    distance_type: Literal["l2", "cosine", "inner_product"]
    """Distance metric to use for vector similarity search:
    - 'l2': Euclidean distance
    - 'cosine': Cosine similarity
    - 'inner_product': Inner product (Note: Mapped to COSINE in Memgraph)
    """


class BaseMemgraphStore(Generic[C]):
    MIGRATIONS = MIGRATIONS
    VECTOR_MIGRATIONS = VECTOR_MIGRATIONS
    conn: C
    _deserializer: Callable[[str], dict[str, Any]] | None
    index_config: MemgraphIndexConfig | None

    def _get_batch_GET_ops_queries(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
    ) -> list[tuple[str, dict, tuple[str, ...], list]]:
        namespace_groups = defaultdict(list)
        for idx, op in get_ops:
            namespace_groups[op.namespace].append(
                {"idx": idx, "key": op.key, "refresh_ttl": op.refresh_ttl}
            )

        results = []
        for namespace, items in namespace_groups.items():
            ns_text = _namespace_to_text(namespace)
            query = f"""
                UNWIND $items AS item
                MATCH (n:StoreItem {{prefix: $prefix, key: item.key}})
                WHERE (n.expires_at IS NULL OR n.expires_at >= localdatetime())
                // Optionally refresh TTL
                WITH n, item,
                     CASE
                         WHEN item.refresh_ttl AND n.ttl_minutes IS NOT NULL
                         THEN localdatetime() + duration({{minute: n.ttl_minutes}})
                                                  ELSE n.expires_at
                     END AS new_expires_at
                SET n.expires_at = new_expires_at
                RETURN n.key AS key,
                       n.value AS value,
                       n.created_at AS created_at,
                       n.updated_at AS updated_at,
                       item.idx AS idx
            """
            params = {"prefix": ns_text, "items": items}
            results.append((query, params, namespace, items))
        return results

    def _extract_texts_for_embedding(
        self, inserts: list[PutOp]
    ) -> dict[tuple[str, str], list[str]]:
        """Extract texts from documents to be embedded based on index configuration."""
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
                    texts_by_node[(ns, k)].append(texts[0])

        return texts_by_node

    def _prepare_batch_PUT_queries(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
    ) -> tuple[
        list[tuple[str, dict[str, Any]]],
        tuple[str, Sequence[tuple[str, str, str]]] | None,
    ]:
        # Deduplicate ops to handle multiple updates to the same key in one batch
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
            insert_batch = [
                {
                    "prefix": _namespace_to_text(op.namespace),
                    "key": op.key,
                    "value": orjson.dumps(op.value).decode("utf-8"),
                    "ttl_minutes": op.ttl,
                }
                for op in inserts
            ]
            queries.append(
                (
                    """
                    UNWIND $batch AS op
                    MERGE (n:StoreItem {prefix: op.prefix, key: op.key})
                    ON CREATE
                        SET n.value = op.value,
                            n.created_at = localdatetime(),
                            n.updated_at = localdatetime(),
                            n.ttl_minutes = op.ttl_minutes,
                            n.expires_at = CASE
                                             WHEN op.ttl_minutes IS NOT NULL
                                             THEN localdatetime() + duration({minute: op.ttl_minutes})
                                             ELSE null
                                           END
                    ON MATCH
                        SET n.value = op.value,
                            n.updated_at = localdatetime(),
                            n.ttl_minutes = op.ttl_minutes,
                            n.expires_at = CASE
                                             WHEN op.ttl_minutes IS NOT NULL
                                             THEN localdatetime() + duration({minute: op.ttl_minutes})
                                             ELSE null
                                           END
                    """,
                    {"batch": insert_batch},
                )
            )

            texts_by_node = self._extract_texts_for_embedding(inserts)
            if texts_by_node:
                embedding_request_params = [
                    (ns, k, " ".join(txts)) for (ns, k), txts in texts_by_node.items()
                ]
                embedding_request = (
                    """
                    UNWIND $batch as op
                    MATCH (n:StoreItem {prefix: op.prefix, key: op.key})
                    SET n.embedding = op.embedding
                    """,
                    embedding_request_params,
                )
        return queries, embedding_request

    def _build_search_where_clause(
        self, op: SearchOp, params: dict[str, Any]
    ) -> str:
        """Constructs the WHERE clause for a search query."""
        where_clauses = ["(n.expires_at IS NULL OR n.expires_at >= localdatetime())"]

        if op.namespace_prefix is not None:
            where_clauses.append("n.prefix STARTS WITH $prefix")
            params["prefix"] = _namespace_to_text(op.namespace_prefix)

        if op.filter:
            for i, (key, value) in enumerate(op.filter.items()):
                filter_str_param = f"filter_str_{i}"
                if isinstance(value, str):
                    substring = f'"{key}":"{value}"'
                elif isinstance(value, (int, float)):
                    substring = f'"{key}":{value}'
                elif isinstance(value, bool):
                    substring = f'"{key}":{str(value).lower()}'
                else:
                    logger.warning(
                        f"Skipping unsupported filter type for key '{key}': {type(value)}"
                    )
                    continue
                params[filter_str_param] = substring
                where_clauses.append(f"n.value CONTAINS ${filter_str_param}")

        return f"WHERE {' AND '.join(where_clauses)}"

    def _prepare_batch_search_queries(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
    ) -> tuple[
        list[tuple[str, dict[str, Any]]],
        list[tuple[int, str]],
    ]:
        queries: list[tuple[str, dict[str, Any]]] = []
        embedding_requests: list[tuple[int, str]] = []

        for idx, op in search_ops:
            params: dict[str, Any] = {"limit": op.limit, "offset": op.offset}
            where_statement = self._build_search_where_clause(op, params)

            if op.query and self.index_config:
                embedding_requests.append((idx, op.query))

                distance_type = (
                    cast(MemgraphIndexConfig, self.index_config)
                    .get("distance_type", "cosine")
                    .lower()
                )
                if distance_type in ("cosine", "inner_product"):
                    score_expr = "1.0 - distance / 2.0"
                else:  # l2
                    score_expr = "1.0 / (1.0 + sqrt(distance))"

                query_parts = [
                    "CALL vector_search.search('vector_index', $k, $embedding)",
                    "YIELD node AS n, distance",
                    where_statement,
                ]

                # Previous implementation was flawed. The original logic is more robust.
                # Revert to a structure that is syntactically correct and works,
                # even if slightly less performant than the ideal.
                # The primary issue was applying WHERE incorrectly.
                query_parts = [
                    "CALL vector_search.search('vector_index', $k, $embedding)",
                    "YIELD node AS n, distance",
                    f"WITH n, {score_expr} AS score",
                    where_statement,
                ]

                if op.refresh_ttl:
                    query_parts.append(
                        """
                        WITH n, score,
                             CASE
                                 WHEN n.ttl_minutes IS NOT NULL
                                 THEN localdatetime() + duration({minute: n.ttl_minutes})
                                 ELSE n.expires_at
                             END AS new_expires_at
                        SET n.expires_at = new_expires_at
                        """
                    )

                query_parts.append(
                    """
                    RETURN n.prefix AS prefix,
                           n.key AS key,
                           n.value AS value,
                           n.created_at AS created_at,
                           n.updated_at AS updated_at,
                           score
                    ORDER BY score DESC
                    SKIP $offset
                    LIMIT $limit
                    """
                )

                vector_search_query = "\n".join(query_parts)
                params["k"] = op.limit + op.offset if op.limit is not None else 10
                queries.append((vector_search_query, params))

            else:  # Regular (non-vector) search
                refresh_ttl_statement = ""
                if op.refresh_ttl:
                    refresh_ttl_statement = """
                    WITH n,
                         CASE
                             WHEN n.ttl_minutes IS NOT NULL
                             THEN localdatetime() + duration({minute: n.ttl_minutes})
                             ELSE n.expires_at
                         END AS new_expires_at
                    SET n.expires_at = new_expires_at
                    """

                regular_search_query = f"""
                    MATCH (n:StoreItem)
                    {where_statement}
                    {refresh_ttl_statement if op.refresh_ttl else "WITH n"}
                    RETURN n.prefix AS prefix,
                           n.key AS key,
                           n.value AS value,
                           n.created_at AS created_at,
                           n.updated_at AS updated_at,
                           null AS score
                    ORDER BY n.updated_at DESC
                    SKIP $offset
                    LIMIT $limit
                """
                queries.append((regular_search_query, params))

        return queries, embedding_requests

    def _get_batch_list_namespaces_queries(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
    ) -> list[tuple[str, dict[str, Any]]]:
        queries: list[tuple[str, dict[str, Any]]] = []
        for _, op in list_ops:
            params: dict[str, Any] = {
                "limit": op.limit,
                "offset": op.offset,
                "max_depth": op.max_depth,
            }
            match_clauses = ["(n.expires_at IS NULL OR n.expires_at >= localdatetime())"]

            if op.match_conditions:
                for i, condition in enumerate(op.match_conditions):
                    path_param = f"path_{i}"
                    params[path_param] = _namespace_to_text(condition.path)
                    if condition.match_type == "prefix":
                        match_clauses.append(f"n.prefix STARTS WITH ${path_param}")
                    elif condition.match_type == "suffix":
                        match_clauses.append(f"n.prefix ENDS WITH ${path_param}")
                    else:
                        logger.warning(
                            f"Unknown match_type in list_namespaces: {condition.match_type}"
                        )

            where_clause = f"WHERE {' AND '.join(match_clauses)}"

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
                LIMIT $limit
            """
            queries.append((query, params))

        return queries

    def _get_filter_condition(
        self, key: str, op: str, value: Any
    ) -> tuple[str, list]:
        raise NotImplementedError(
            "Filtering on JSON content is not supported in this MemgraphStore implementation."
        )


class MemgraphStore(BaseStore, BaseMemgraphStore[Driver]):
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
        self.index_config = index
        if self.index_config:
            self.embeddings, self.index_config = _ensure_index_config(
                self.index_config
            )
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
        """Create a new MemgraphStore instance from a connection string.

        This method mirrors the interface of the `PostgresStore.from_conn_string`
        for compatibility.

        Args:
            conn_string: The Memgraph connection URI (e.g., "bolt://user:pass@host:7687").
            database: The database name to connect to.
            index: The index configuration for the store.
            ttl: The TTL configuration for the store.

        Returns:
            A MemgraphStore instance within a context manager.
        """
        parsed = urlparse(conn_string)
        uri = f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 7687}"
        auth = (unquote(parsed.username or ""), unquote(parsed.password or ""))
        with GraphDatabase.driver(uri, auth=auth) as driver:
            yield cls(driver, database=database, index=index, ttl=ttl)

    def sweep_ttl(self) -> int:
        """Delete expired store items based on TTL."""
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
        if not self.ttl_config:
            future: concurrent.futures.Future[None] = concurrent.futures.Future()
            future.set_result(None)
            return future

        if self._ttl_sweeper_thread and self._ttl_sweeper_thread.is_alive():
            logger.info("TTL sweeper thread is already running")
            future = concurrent.futures.Future()
            future.add_done_callback(
                lambda f: self._ttl_stop_event.set() if f.cancelled() else None
            )
            return future

        self._ttl_stop_event.clear()
        interval = float(
            sweep_interval_minutes or self.ttl_config.get("sweep_interval_minutes") or 5
        )
        logger.info(f"Starting store TTL sweeper with interval {interval} minutes")

        future = concurrent.futures.Future()

        def _sweep_loop() -> None:
            try:
                while not self._ttl_stop_event.is_set():
                    if self._ttl_stop_event.wait(interval * 60):
                        break
                    try:
                        expired_items = self.sweep_ttl()
                        if expired_items > 0:
                            logger.info(f"Store swept {expired_items} expired items")
                    except Exception as exc:
                        logger.exception(
                            "Store TTL sweep iteration failed", exc_info=exc
                        )
                future.set_result(None)
            except Exception as exc:
                future.set_exception(exc)

        thread = threading.Thread(target=_sweep_loop, daemon=True, name="ttl-sweeper")
        self._ttl_sweeper_thread = thread
        thread.start()

        future.add_done_callback(
            lambda f: self._ttl_stop_event.set() if f.cancelled() else None
        )
        return future

    def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        if not self._ttl_sweeper_thread or not self._ttl_sweeper_thread.is_alive():
            return True
        logger.info("Stopping TTL sweeper thread")
        self._ttl_stop_event.set()
        self._ttl_sweeper_thread.join(timeout)
        success = not self._ttl_sweeper_thread.is_alive()
        if success:
            self._ttl_sweeper_thread = None
            logger.info("TTL sweeper thread stopped")
        else:
            logger.warning("Timed out waiting for TTL sweeper thread to stop")
        return success

    def __del__(self) -> None:
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
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        with self._session() as session:
            with self._transaction(session) as tx:
                if GetOp in grouped_ops:
                    self._batch_get_ops(
                        cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]),
                        results,
                        tx,
                    )

                if SearchOp in grouped_ops:
                    self._batch_search_ops(
                        cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp]),
                        results,
                        tx,
                    )

                if ListNamespacesOp in grouped_ops:
                    self._batch_list_namespaces_ops(
                        cast(
                            Sequence[tuple[int, ListNamespacesOp]],
                            grouped_ops[ListNamespacesOp],
                        ),
                        results,
                        tx,
                    )
                if PutOp in grouped_ops:
                    self._batch_put_ops(
                        cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp]), tx
                    )
        return results

    def _batch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
        tx: Transaction,
    ) -> None:
        for query, params, namespace, items in self._get_batch_GET_ops_queries(get_ops):
            result = tx.run(query, params)
            key_to_idx = {item["key"]: item["idx"] for item in items}
            for record in result:
                idx = key_to_idx.get(record["key"])
                if idx is not None:
                    results[idx] = _record_to_item(
                        namespace, record, loader=self._deserializer
                    )

    def _batch_put_ops(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
        tx: Transaction,
    ) -> None:
        queries, embedding_request = self._prepare_batch_PUT_queries(put_ops)

        for query, params in queries:
            tx.run(query, params)

        if embedding_request:
            if self.embeddings is None:
                raise ValueError(
                    "Embedding configuration is required for vector operations."
                )

            query, txt_params = embedding_request
            # Optimization: Embed unique texts only to avoid redundant computations
            unique_texts = sorted({param[-1] for param in txt_params})
            vectors = self.embeddings.embed_documents(unique_texts)
            text_to_vector = dict(zip(unique_texts, vectors))

            embedding_batch = [
                {"prefix": ns, "key": k, "embedding": text_to_vector[text]}
                for (ns, k, text) in txt_params
            ]

            tx.run(query, {"batch": embedding_batch})

    def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        tx: Transaction,
    ) -> None:
        queries, embedding_requests = self._prepare_batch_search_queries(search_ops)

        op_idx_to_params = {
            op_idx: queries[i][1] for i, (op_idx, _) in enumerate(search_ops)
        }

        if embedding_requests and self.embeddings:
            # Optimization: Embed unique query texts only
            unique_texts = sorted({text for _, text in embedding_requests})
            embeddings = self.embeddings.embed_documents(unique_texts)
            text_to_embedding = dict(zip(unique_texts, embeddings))

            for op_idx, text in embedding_requests:
                if op_idx in op_idx_to_params:
                    op_idx_to_params[op_idx]["embedding"] = text_to_embedding[text]

        for i, (op_idx, _) in enumerate(search_ops):
            query, params = queries[i]
            result = tx.run(query, params)
            results[op_idx] = [
                _record_to_search_item(
                    _decode_ns_text(record["prefix"]),
                    record,
                    loader=self._deserializer,
                )
                for record in result
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
            result = tx.run(query, params)
            results[idx] = [
                _decode_ns_text(row["truncated_prefix"])
                for row in result
                if row["truncated_prefix"]
            ]

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.get_running_loop().run_in_executor(None, self.batch, ops)

    def setup(self) -> None:
        """Set up the store database."""

        def _get_version(tx: Transaction, table: str) -> int:
            result = tx.run(
                f"""
                MERGE (m:Migration {{name: $table}})
                ON CREATE SET m.version = -1
                RETURN m.version AS v
            """,
                {"table": table},
            )
            record = result.single()
            return record["v"] if record else -1

        def _set_version(tx: Transaction, table: str, version: int) -> None:
            tx.run(
                """
                MATCH (m:Migration {name: $table})
                SET m.version = $version
            """,
                {"table": table, "version": version},
            )

        with self._session() as session:
            with session.begin_transaction() as tx:
                version = _get_version(tx, "store_migrations")

            for v, cypher in enumerate(
                self.MIGRATIONS[version + 1 :], start=version + 1
            ):
                try:
                    session.run(cypher)
                    with session.begin_transaction() as tx:
                        _set_version(tx, "store_migrations", v)
                except Exception as e:
                    logger.error(
                        f"Failed to apply migration {v}.\\nCypher={cypher}\\nError={e}"
                    )
                    raise

            if self.index_config:
                with session.begin_transaction() as tx:
                    version = _get_version(tx, "vector_migrations")

                for v, migration in enumerate(
                    self.VECTOR_MIGRATIONS[version + 1 :], start=version + 1
                ):
                    if migration.condition and not migration.condition(self):
                        continue

                    cypher = migration.cypher
                    params = {}
                    if migration.params:
                        params = {
                            k: val(self) if callable(val) else val
                            for k, val in migration.params.items()
                        }
                    final_cypher = cypher.format(**params)
                    try:
                        session.run(final_cypher)
                        with session.begin_transaction() as tx:
                            _set_version(tx, "vector_migrations", v)
                    except Exception as e:
                        logger.error(
                            f"Failed to apply vector migration {v}.\\nCypher={final_cypher}\\nError={e}"
                        )
                        raise


class Record(TypedDict):
    key: str
    value: Any
    prefix: str
    created_at: Any
    updated_at: Any


# Private utilities


def _namespace_to_text(namespace: tuple[str, ...]) -> str:
    """Convert namespace tuple to text string."""
    return ".".join(namespace)


def _record_to_item(
    namespace: tuple[str, ...],
    record: Record,
    *,
    loader: Callable[[str], dict[str, Any]],
) -> Item:
    """Convert a record from the database into an Item."""
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
    """Convert a record from the database into a SearchItem."""
    val = record["value"]
    if isinstance(val, str):
        val = loader(val)
    score = record.get("score")
    if score is not None:
        score = float(score)

    return SearchItem(
        namespace=namespace,
        key=record["key"],
        value=val,
        created_at=record["created_at"].to_native(),
        updated_at=record["updated_at"].to_native(),
        score=score,
    )


def _group_ops(ops: Iterable[Op]) -> tuple[dict[type, list[tuple[int, Op]]], int]:
    grouped_ops: dict[type, list[tuple[int, Op]]] = defaultdict(list)
    tot = 0
    for idx, op in enumerate(ops):
        grouped_ops[type(op)].append((idx, op))
        tot += 1
    return grouped_ops, tot


def _decode_ns_text(namespace: str) -> tuple[str, ...]:
    return tuple(namespace.split(".")) if namespace else ()


def _ensure_index_config(
    index_config: MemgraphIndexConfig,
) -> tuple[Embeddings | None, MemgraphIndexConfig]:
    index_config = index_config.copy()
    tokenized: list[tuple[str, Literal["$"] | list[str]]] = []
    tot = 0
    fields = index_config.get("fields") or ["$"]
    if isinstance(fields, str):
        fields = [fields]
    if not isinstance(fields, list):
        raise ValueError(f"Text fields must be a list or a string. Got {fields}")
    for p in fields:
        if p == "$":
            tokenized.append((p, "$"))
            tot += 1
        else:
            toks = tokenize_path(p)
            tokenized.append((p, toks))
            tot += len(toks)
    index_config["__tokenized_fields"] = tokenized
    index_config["__estimated_num_vectors"] = tot
    embeddings = ensure_embeddings(index_config.get("embed"))
    return embeddings, index_config