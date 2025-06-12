# langgraph/checkpoint/memgraph/__init__.py
"""
Memgraph checkpoint & store plug‑in for LangGraph.

This top‑level package only wires public symbols so that imports succeed.  The
full implementation lives in sibling modules (`saver.py`, `aio.py`, `store/*`).
"""
from __future__ import annotations

from importlib import import_module

# Lazy import to avoid the Neo4j driver overhead when users only need type hints
_mem_saver = import_module("langgraph.checkpoint.memgraph.saver")
_mem_aio = import_module("langgraph.checkpoint.memgraph.aio")

MemgraphSaver = _mem_saver.MemgraphSaver
AsyncMemgraphSaver = _mem_aio.AsyncMemgraphSaver

__all__ = ["MemgraphSaver", "AsyncMemgraphSaver"]
