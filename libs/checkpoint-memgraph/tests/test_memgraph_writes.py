import os
import uuid
from datetime import datetime, timezone

import pytest
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.memgraph import MemgraphSaver

BOLT_URI = os.getenv(
    "MEMGRAPH_BOLT_URI",
    "bolt://testuser123:BiggerPassword1233@localhost:7687",
)


def _have_db() -> bool:
    """Return True if the Memgraph database is reachable on the configured URI."""
    print(f"Attempting to connect with URI: {BOLT_URI}")
    try:
        saver = MemgraphSaver.from_conn_string(BOLT_URI)
        saver.close()
        print("Connection to Memgraph was successful.")
        return True
    except Exception as exc:  # pragma: no cover
        print(f"Failed to connect to Memgraph: {exc}")
        return False


pytestmark = pytest.mark.skipif(
    not _have_db(), reason="Memgraph instance not reachable on localhost"
)


@pytest.fixture(scope="module")
def saver() -> MemgraphSaver:
    cp = MemgraphSaver.from_conn_string(BOLT_URI)
    cp.setup()
    yield cp
    cp.close()


# --------------------------------------------------------------------------- #
def _base_checkpoint() -> dict:
    """Return a baseline checkpoint dict accepted by MemgraphSaver."""
    return {
        "id": str(uuid.uuid4()),
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "v": 0,
        "channel_values": {},
        "channel_versions": {},
    }


# --------------------------------------------------------------------------- #
def test_put_writes_and_retrieve(saver: MemgraphSaver) -> None:
    """Ensure writes are persisted and can be retrieved intact."""
    tid = "thr-" + str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": tid}}

    # ── Store root checkpoint and capture the *returned* cfg with checkpoint_id ──
    cfg = saver.put(cfg, _base_checkpoint(), {}, {})

    # ── Store pending writes linked to that checkpoint ──
    writes = [("out", {"msg": "hello"}), ("log", 123)]
    saver.put_writes(cfg, writes, task_id="task-1", task_path="foo.bar")

    # ── Retrieve and validate ──
    ctuple = saver.get_tuple(cfg)
    assert isinstance(ctuple, CheckpointTuple)
    pending = ctuple.pending_writes
    assert len(pending) == 2

    channels = {ch for _, ch, _ in pending}
    assert channels == {"out", "log"}
    values = {ch: val for _, ch, val in pending}
    assert values["out"]["msg"] == "hello"
    assert values["log"] == 123


def test_delete_thread_removes_data(saver: MemgraphSaver) -> None:
    """Deleting a thread should remove *all* checkpoints, blobs and writes."""
    tid = "thr-" + str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": tid}}

    # Create a checkpoint (returns cfg incl. checkpoint_id)
    cfg = saver.put(cfg, _base_checkpoint(), {}, {})

    # Sanity check that data exists
    assert saver.get_tuple(cfg) is not None

    # Delete everything for the thread
    saver.delete_thread(tid)

    # Verify nothing remains
    assert saver.get_tuple(cfg) is None
    assert list(saver.list(cfg)) == []
