# LangGraph Memgraph Checkpoint

Implementation of a LangGraph CheckpointSaver that uses [Memgraph](https://memgraph.com/).

## Dependencies

The package pulls in `neo4j>=5.14` automatically. No native drivers or Memgraph-specific wheels are required.

## Setup

You will need to have a Memgraph instance running. The easiest way to do this is with Docker:

```bash
docker run -it --rm -p 7687:7687 -p 7444:7444 memgraph/memgraph-mage
```

The default user/password is `memgraph`/`memgraph`.

## Usage

> [!IMPORTANT]
> When using Memgraph checkpointers for the first time, make sure to call the `.setup()` method on them to create the required indexes and constraints.

```python
from langgraph.checkpoint.memgraph import MemgraphSaver

write_config = {"configurable": {"thread_id": "my-thread-id"}}
read_config = {"configurable": {"thread_id": "my-thread-id"}}

DB_URI = "bolt://memgraph:memgraph@localhost:7687"

with MemgraphSaver.from_conn_string(DB_URI) as checkpointer:
    # call .setup() the first time you're using the checkpointer
    checkpointer.setup()

    checkpoint = {
        "v": 1,
        "ts": "2024-08-01T12:00:00.000000+00:00",
        "id": "some_checkpoint_id",
        "channel_values": {"messages": ["Hello, world!"]},
        "channel_versions": {"messages": 1},
        "versions_seen": {},
    }

    # store checkpoint
    checkpointer.put(write_config, checkpoint, {}, {})

    # load checkpoint
    loaded_checkpoint_tuple = checkpointer.get_tuple(read_config)
    print(loaded_checkpoint_tuple.checkpoint)

    # list checkpoints
    checkpoints = list(checkpointer.list(read_config))
    print(f"Found {len(checkpoints)} checkpoints.")
```

### Async

```python
import asyncio
from langgraph.checkpoint.memgraph.aio import AsyncMemgraphSaver

write_config = {"configurable": {"thread_id": "my-async-thread-id"}}
read_config = {"configurable": {"thread_id": "my-async-thread-id"}}

DB_URI = "bolt://memgraph:memgraph@localhost:7687"

async def main():
    async with AsyncMemgraphSaver.from_conn_string(DB_URI) as checkpointer:
        # call .setup() the first time you're using the checkpointer
        await checkpointer.setup()

        checkpoint = {
            "v": 1,
            "ts": "2024-08-01T12:00:00.000000+00:00",
            "id": "some_async_checkpoint_id",
            "channel_values": {"messages": ["Hello, async world!"]},
            "channel_versions": {"messages": 1},
            "versions_seen": {},
        }

        # store checkpoint
        await checkpointer.aput(write_config, checkpoint, {}, {})

        # load checkpoint
        loaded_checkpoint_tuple = await checkpointer.aget_tuple(read_config)
        print(loaded_checkpoint_tuple.checkpoint)

        # list checkpoints
        checkpoints = [c async for c in checkpointer.alist(read_config)]
        print(f"Found {len(checkpoints)} async checkpoints.")

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced: Bring-your-own Neo4j driver

If you need explicit control over the connection (e.g., connection pooling, SSL settings), you can create the driver yourself and pass it in:

```python
from neo4j import GraphDatabase
from langgraph.checkpoint.memgraph import MemgraphSaver

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("memgraph", "memgraph"),
    max_connection_lifetime=180,  # custom parameter
)
try:
    checkpointer = MemgraphSaver(driver)
    checkpointer.setup()
    # ... use checkpointer as needed
finally:
    driver.close()
```

The async version works similarly with `AsyncGraphDatabase.driver`.