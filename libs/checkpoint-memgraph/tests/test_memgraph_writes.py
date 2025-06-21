import uuid
from datetime import datetime, timezone

import pytest
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.memgraph import MemgraphSaver
from tests.conftest import DEFAULT_MEMGRAPH_URI


def _have_db() -> bool:
    """Return True if the Memgraph database is reachable on the configured URI."""
    print(f"Attempting to connect with URI: {DEFAULT_MEMGRAPH_URI}")
    try:
        saver = MemgraphSaver.from_conn_string(DEFAULT_MEMGRAPH_URI)
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
    cp = MemgraphSaver.from_conn_string(DEFAULT_MEMGRAPH_URI)
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

def test_put_writes_and_retrieve(saver: MemgraphSaver) -> None:
    """Ensure writes are persisted and can be retrieved intact."""
    tid = "thr-" + str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": tid}}
    cfg = saver.put(cfg, _base_checkpoint(), {}, {})
    writes = [("out", {"msg": "hello"}), ("log", 123)]
    saver.put_writes(cfg, writes, task_id="task-1", task_path="foo.bar")
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
    tid = "thr-" + str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": tid}}
    cfg = saver.put(cfg, _base_checkpoint(), {}, {})
    assert saver.get_tuple(cfg) is not None
    saver.delete_thread(tid)
    assert saver.get_tuple(cfg) is None
    assert list(saver.list(cfg)) == []
