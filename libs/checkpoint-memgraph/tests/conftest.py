# libs/checkpoint-memgraph/tests/conftest.py
"""
PyTest configuration for checkpoint‑memgraph.

Ensures that the library root (two levels up) is on the import path so that
`import langgraph.checkpoint.memgraph` resolves when the package has not been
installed into the active environment.
"""
from __future__ import annotations

import pathlib
import site
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    site.addsitedir(str(ROOT_DIR))
