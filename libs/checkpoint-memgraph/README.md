# LangGraph Checkpoint & Store — Memgraph Edition

Production‑ready **checkpoint** and **long‑term memory store** implementations
for [LangGraph](https://github.com/langchain-ai/langgraph) that leverage
[Memgraph](https://memgraph.com/) through the Bolt protocol
(via the official *neo4j‑python* driver).

<p align="center">
  <a href="https://pypi.org/project/langgraph-checkpoint-memgraph/">
    <img src="https://img.shields.io/pypi/v/langgraph-checkpoint-memgraph.svg" />
  </a>
  <a href="https://github.com/langchain-ai/langgraph/actions/workflows/test.yml">
    <img src="https://github.com/langchain-ai/langgraph/actions/workflows/test.yml/badge.svg" />
  </a>
</p>

---

- **Sync & Async** savers (`MemgraphSaver`, `AsyncMemgraphSaver`)
- **Sync & Async** stores (`MemgraphStore`, `AsyncMemgraphStore`)
- Vector‑similarity (HNSW) search with *optional* embeddings
- Hierarchical namespaces (`("users", "42")`, `("docs", "faq")`, …)
- Built‑in TTL expiry with background sweeper
- Fully typed (PEP 561) & covered by an extensive test‑suite

---

## Installation

```bash
pip install langgraph-checkpoint-memgraph
````

> **Heads‑up**
> The package pulls in **`neo4j>=5.14`** automatically.
> No native drivers or Memgraph‑specific wheels are required.

## Quick Start — Checkpoint Saver

> \[!IMPORTANT]
> Run `.setup()` **once** per database to create indexes / constraints.

```python title="basic_checkpoint.py"
from langgraph.checkpoint.memgraph import MemgraphSaver

WRITE_CFG = {"configurable": {"thread_id": "my-thread", "checkpoint_ns": ""}}
READ_CFG  = {"configurable": {"thread_id": "my-thread"}}

BOLT_URI = "bolt://memgraph_user:secret@localhost:7687"

checkpoint = {
    "v": 1,
    "ts": "2024-08-01T12:00:00.000000+00:00",
    "id": "chk‑1",
    "channel_values": {"greeting": "hello"},
    "channel_versions": {"greeting": 1},
}

with MemgraphSaver.from_conn_string(BOLT_URI) as saver:
    saver.setup()                       # <‑‑ create schema on first run
    saver.put(WRITE_CFG, checkpoint, {}, {})  # store checkpoint
    latest = saver.get_tuple(READ_CFG)        # retrieve latest
    print(latest.checkpoint["channel_values"]["greeting"])   # -> hello
```

### Async flavour

```python title="basic_checkpoint_async.py"
import asyncio
from langgraph.checkpoint.memgraph.aio import AsyncMemgraphSaver

async def main() -> None:
    BOLT_URI = "bolt://memgraph_user:secret@localhost:7687"
    cfg = {"configurable": {"thread_id": "async", "checkpoint_ns": ""}}

    async with AsyncMemgraphSaver.from_conn_string(BOLT_URI) as saver:
        await saver.setup()

        checkpoint = {
            "v": 1,
            "ts": "2024‑08‑01T12:00:00Z",
            "id": "chk‑async‑1",
            "channel_values": {"foo": "bar"},
            "channel_versions": {"foo": 1},
        }

        await saver.aput(cfg, checkpoint, {}, {})
        latest = await saver.aget_tuple(cfg)
        print(latest.checkpoint["channel_values"]["foo"])  # -> bar

asyncio.run(main())
```

### Advanced: Bring‑your‑own Neo4j driver

If you need explicit control over the connection (connection pooling, SSL
settings, …) you can create the driver yourself and pass it in:

```python
from neo4j import GraphDatabase
from langgraph.checkpoint.memgraph import MemgraphSaver

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("memgraph_user", "secret"),
    max_connection_lifetime=180,        # custom parameter
)
try:
    saver = MemgraphSaver(driver)
    saver.setup()
    # ...
finally:
    driver.close()
```

*(Async works the same with `AsyncGraphDatabase.driver`.)*

---

## Quick Start — Long‑term Memory **Store**

```python title="basic_store.py"
from langgraph.store.memgraph import MemgraphStore

store = MemgraphStore.from_conn_string(
    "bolt://memgraph_user:secret@localhost:7687",
    ttl={                       # enable TTL — 30 mins default
        "default_ttl": 30,
        "refresh_on_read": True,
        "sweep_interval_minutes": 5,    # background sweeper
    },
    index={                     # enable HNSW vector search
        "dims": 768,
        "metric": "cos",
        "embed": my_embedding_model,    # any object with embed_*()
    },
)
store.setup()

# --------------------------------------------
# Plain CRUD
# --------------------------------------------
ns = ("users", "42")

store.put(ns, key="profile", value={"bio": "AI enthusiast"})
profile = store.get(ns, "profile")      # -> Item(...)

# --------------------------------------------
# Vector / lexical search
# --------------------------------------------
hits = store.search(("users",), query="AI", limit=5)
for h in hits:
    print(h.namespace, h.key, h.score)
```

### Async store

```python
from langgraph.store.memgraph.aio import AsyncMemgraphStore

async with AsyncMemgraphStore.from_conn_string("bolt://user:pass@localhost:7687") as store:
    await store.setup()
    await store.aput(("docs",), key="intro", value="Welcome to LangGraph!")
    results = await store.asearch(("docs",), query="Welcome")
    print(results[0].value)
```

---

## Running Memgraph locally

Spin up Memgraph in seconds with Docker:

```bash
docker run -it --rm -p 7687:7687 -e MEMGRAPH="--log-level=ERROR" memgraph/memgraph-platform
```

The default user/password is `memgraph`/`memgraph` (set your own for prod!).

---

## Testing

The repository ships a comprehensive test‑suite that boots Memgraph via Docker
Compose.  Run all tests against a matrix of Memgraph versions:

```bash
# Requires Docker & GNU Make
make test
```

Watch mode (auto re‑run on change):

```bash
make test_watch
```

---

## Reference

| Class                    | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| **`MemgraphSaver`**      | Blocking `CheckpointSaver` using the Neo4j driver            |
| **`AsyncMemgraphSaver`** | `asyncio` counterpart                                        |
| **`MemgraphStore`**      | High‑level key/value store with optional vector search & TTL |
| **`AsyncMemgraphStore`** | Async store                                                  |

> Full API reference is available in the
> [LangGraph documentation](https://langchain-ai.github.io/langgraph/).

---
