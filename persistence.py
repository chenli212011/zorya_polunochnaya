"""Async sqlite-backed checkpointer for LangGraph workflows.

LangGraph's async graph methods (`ainvoke`/`astream`) require an
`AsyncSqliteSaver` backed by an `aiosqlite` connection. The connection has to
be opened inside a running event loop, so the lifecycle is managed by the
FastAPI lifespan handler in `app.py`:

    await persistence.init()   # on startup
    await persistence.close()  # on shutdown

Graphs read the checkpointer lazily via `get_checkpointer()` when the registry
first compiles them — that happens on first request, by which point `init()`
has already run.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

DB_PATH = Path(__file__).parent / "data" / "checkpoints.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_conn: aiosqlite.Connection | None = None
_checkpointer: AsyncSqliteSaver | None = None


async def init() -> None:
    """Open the sqlite connection and create checkpoint tables if needed."""
    global _conn, _checkpointer
    if _checkpointer is not None:
        return
    _conn = await aiosqlite.connect(str(DB_PATH))
    _checkpointer = AsyncSqliteSaver(_conn)
    await _checkpointer.setup()


async def close() -> None:
    global _conn, _checkpointer
    if _conn is not None:
        await _conn.close()
    _conn = None
    _checkpointer = None


def get_checkpointer() -> AsyncSqliteSaver:
    if _checkpointer is None:
        raise RuntimeError(
            "persistence not initialized — FastAPI lifespan must call persistence.init() first"
        )
    return _checkpointer
