# libs/checkpoint-memgraph/langgraph/store/memgraph/aio.py
"""
Asynchronous Memgraph key–value store façade.

Why a façade?
~~~~~~~~~~~~~
The official Neo4j/Memgraph async driver internally pins itself to the
event‑loop that exists at *import‑time*.  Test‑suites that spawn their own
loops (e.g. pytest‑asyncio) will then observe the dreaded:

    RuntimeError: Task ... got Future <...> attached to a different loop

To guarantee reliability in every environment, this module implements
:class:`AsyncMemgraphStore` by **delegating all work** to the already‑robust
synchronous :class:`~langgraph.store.memgraph.MemgraphStore` and running each
call in a worker‑thread via :func:`asyncio.to_thread`.  Users still get an
async/`await` API, while the driver never crosses event‑loop boundaries.
"""

from __future__ import annotations

import asyncio
from types import TracebackType
from typing import Any, Iterable, List, Mapping, Optional, Tuple, Type

from langgraph.store.base import (  # type: ignore[attr-defined]
    BaseStore,
    Item,
)

from . import MemgraphStore
from ._utils import parse_bolt_uri

__all__ = ["AsyncMemgraphStore"]


class AsyncMemgraphStore(BaseStore):  # type: ignore[misc]
    """
    Lightweight asynchronous wrapper around the synchronous
    :class:`~langgraph.store.memgraph.MemgraphStore`.
    """

    # ------------------------------------------------------------------ #
    # Construction helpers
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
        # Create *one* synchronous backend instance.
        self._sync = MemgraphStore(
            bolt_uri,
            user=user,
            password=password,
            index=index,
            ttl=ttl,
            node_label=node_label,
            driver_kwargs=driver_kwargs,
        )

    # Convenience builder that matches the sync‑store signature
    @classmethod
    def from_conn_string(cls, conn: str, **kwargs: Any) -> "AsyncMemgraphStore":
        parsed = parse_bolt_uri(conn)
        return cls(
            parsed["bolt_uri"],
            user=parsed["user"],
            password=parsed["password"],
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Async context‑manager
    # ------------------------------------------------------------------ #
    async def __aenter__(self) -> "AsyncMemgraphStore":  # noqa: D401 (imperative)
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        await self.close()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def setup(self) -> None:
        await asyncio.to_thread(self._sync.setup)

    initialise = setup  # public alias

    async def close(self) -> None:
        await asyncio.to_thread(self._sync.close)

    # ------------------------------------------------------------------ #
    # CRUD methods (async wrappers)
    # ------------------------------------------------------------------ #
    async def aput(
        self,
        namespace: Tuple[str, ...],
        key: str,
        value: Any,
        *,
        index: bool = True,
        ttl: float | None = None,
    ) -> Item:
        return await asyncio.to_thread(
            self._sync.put,
            namespace,
            key,
            value,
            index=index,
            ttl=ttl,
        )

    async def aget(
        self,
        namespace: Tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Optional[Item]:
        return await asyncio.to_thread(
            self._sync.get,
            namespace,
            key,
            refresh_ttl=refresh_ttl,
        )

    async def aexists(self, namespace: Tuple[str, ...], key: str) -> bool:
        return await asyncio.to_thread(self._sync.exists, namespace, key)

    async def acount(self, namespace_prefix: Tuple[str, ...]) -> int:
        return await asyncio.to_thread(self._sync.count, namespace_prefix)

    async def adelete(self, namespace: Tuple[str, ...], key: str) -> None:
        await asyncio.to_thread(self._sync.delete, namespace, key)

    async def adelete_namespace(self, namespace_prefix: Tuple[str, ...]) -> None:
        await asyncio.to_thread(self._sync.delete_namespace, namespace_prefix)

    # ------------------------------------------------------------------ #
    # Batch helper
    # ------------------------------------------------------------------ #
    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        return await asyncio.to_thread(self._sync.batch, ops)

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        """Blocking convenience method mirroring the sync backend."""
        return asyncio.run(self.abatch(ops))

    # ------------------------------------------------------------------ #
    # Additional parity helpers
    # ------------------------------------------------------------------ #
    async def alist_keys(self, namespace: Tuple[str, ...]) -> List[str]:
        return await asyncio.to_thread(self._sync.list_keys, namespace)

    async def alist_items(self, namespace: Tuple[str, ...]) -> List[Item]:
        return await asyncio.to_thread(self._sync.list_items, namespace)

    async def asearch(
        self,
        namespace_prefix: Tuple[str, ...],
        *,
        query: str | List[float] | None = None,
        filter: Mapping[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> List[Item]:
        return await asyncio.to_thread(
            self._sync.search,
            namespace_prefix,
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
            refresh_ttl=refresh_ttl,
        )
