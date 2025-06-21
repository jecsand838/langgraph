import uuid
from datetime import datetime, timezone

import pytest
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.memgraph import MemgraphSaver
from tests.conftest import DEFAULT_MEMGRAPH_URI


def _have_db() -> bool:
    try:
        saver = MemgraphSaver.from_conn_string(DEFAULT_MEMGRAPH_URI)
        saver.close()
        return True
    except Exception:
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

def _empty_ckpt(v: int, ckpt_id: str | None = None) -> dict:
    """Helper to build a minimal checkpoint payload accepted by MemgraphSaver."""
    return {
        "id": ckpt_id or str(uuid.uuid4()),
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "v": v,
        "channel_values": {},
        "channel_versions": {},
    }

def test_put_and_get(saver: MemgraphSaver) -> None:
    cfg = {"configurable": {"thread_id": "thr-" + str(uuid.uuid4())}}
    ckpt = _empty_ckpt(1)
    saver.put(cfg, ckpt, {}, {})
    latest = saver.get_tuple(cfg)
    assert isinstance(latest, CheckpointTuple)
    assert latest.checkpoint["v"] == 1


def test_list_returns_all(saver: MemgraphSaver) -> None:
    tid = "thr-" + str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": tid}}
    for i in range(3):
        saver.put(cfg, _empty_ckpt(i, ckpt_id=str(i)), {}, {})
    cks = list(saver.list(cfg))
    assert len(cks) == 3
    assert cks[0].checkpoint["id"] == "2"
    assert {c.checkpoint["v"] for c in cks} == {0, 1, 2}


def test_list_with_metadata_filter(saver: MemgraphSaver) -> None:
    tid = "thr-" + str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": tid}}
    saver.put(cfg, _empty_ckpt(0), {"tag": "alpha"}, {})
    saver.put(cfg, _empty_ckpt(1), {"tag": "beta"}, {})
    alpha_only = list(saver.list(cfg, filter={"tag": "alpha"}))
    assert len(alpha_only) == 1
    assert alpha_only[0].metadata.get("tag") == "alpha"
