"""Run-scoped observer used by the resilience kernel to collect per-step
events (performance, failures, infeasibility incidents, …) for later
flushing to CSV / npz artefacts.

Concurrency
-----------
For backwards compatibility the module exposes ``gather``/``data``/``clear``
free functions. They forward to the *current* :class:`Observer`, which is
selected per thread via a thread-local stack. The default observer at the
bottom of the stack is shared (legacy global behavior); each
``with Observer() as obs:`` block pushes a fresh observer on top so any
``gather`` call made inside the block writes only to that block's state.
This makes ``start_resilience_simulation`` reentrant within a single
Python process (e.g. threaded MC drivers): two runs no longer share or
overwrite each other's gathered data.

Usage
-----
Library-internal hooks should keep calling ``observer.gather(...)`` —
they automatically target whichever observer the surrounding code
opened. Callers that need isolation wrap a region in ``with Observer():``
and read its ``data()`` snapshot at the end.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List


_local = threading.local()


class Observer:
    """A keyed multi-list of gathered events for one run/region.

    Use as a context manager to make this observer the active target for
    module-level ``gather/data/clear`` calls inside the ``with`` block.
    Nested contexts push onto a thread-local stack and restore on exit.
    """

    def __init__(self) -> None:
        self._data: Dict[str, List[Any]] = {}

    # ---- core API ----------------------------------------------------------
    def gather(self, key: str, value: Any) -> None:
        bucket = self._data.get(key)
        if bucket is None:
            self._data[key] = [value]
        else:
            bucket.append(value)

    def data(self) -> Dict[str, List[Any]]:
        return self._data

    def clear(self) -> None:
        self._data.clear()

    # ---- context manager ---------------------------------------------------
    def __enter__(self) -> "Observer":
        stack = getattr(_local, "stack", None)
        if stack is None:
            stack = []
            _local.stack = stack
        stack.append(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        stack = getattr(_local, "stack", None)
        if stack and stack[-1] is self:
            stack.pop()


# ---- default (shared) observer — legacy global behavior -------------------
# Free functions write here when no ``with Observer()`` context is active.
_default = Observer()


def _current() -> Observer:
    stack = getattr(_local, "stack", None)
    if stack:
        return stack[-1]
    return _default


# ---- module-level legacy API — delegates to the current observer ----------
def gather(key: str, value: Any) -> None:
    _current().gather(key, value)


def data() -> Dict[str, List[Any]]:
    return _current().data()


def clear() -> None:
    _current().clear()
