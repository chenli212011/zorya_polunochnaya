"""HTTP routes for browsing and deleting checkpointed conversation threads."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import persistence

router = APIRouter(prefix="/api/threads", tags=["threads"])


def _role(msg) -> str:
    if isinstance(msg, HumanMessage):
        return "user"
    if isinstance(msg, AIMessage):
        return "assistant"
    if isinstance(msg, SystemMessage):
        return "system"
    if isinstance(msg, ToolMessage):
        return "tool"
    return "unknown"


def _content(msg) -> str:
    c = getattr(msg, "content", "")
    if isinstance(c, str):
        return c
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block) for block in c
    )


def _title_from_messages(messages: list) -> str:
    for m in messages:
        if isinstance(m, HumanMessage):
            text = _content(m).strip().splitlines()[0] if _content(m) else ""
            return text[:80]
    return ""


@router.get("")
async def list_threads() -> dict:
    seen: dict[str, dict] = {}
    saver = persistence.get_checkpointer()
    # alist() yields newest-first; the first occurrence per thread_id is the
    # latest checkpoint for that thread.
    async for ck in saver.alist(None):
        tid = ck.config["configurable"].get("thread_id")
        if not tid or tid in seen:
            continue
        messages = ck.checkpoint.get("channel_values", {}).get("messages", [])
        seen[tid] = {
            "thread_id": tid,
            "title": _title_from_messages(messages) or "(empty)",
            "message_count": len(messages),
            "ts": ck.checkpoint.get("ts"),
        }
    threads = list(seen.values())
    threads.sort(key=lambda t: t["ts"] or "", reverse=True)
    return {"threads": threads}


@router.get("/{thread_id}")
async def get_thread(thread_id: str) -> dict:
    saver = persistence.get_checkpointer()
    tup = await saver.aget_tuple({"configurable": {"thread_id": thread_id}})
    if not tup:
        raise HTTPException(status_code=404, detail=f"unknown thread: {thread_id}")
    messages = tup.checkpoint.get("channel_values", {}).get("messages", [])
    return {
        "thread_id": thread_id,
        "messages": [
            {"role": _role(m), "content": _content(m)}
            for m in messages
            if _role(m) in ("user", "assistant")
        ],
        "ts": tup.checkpoint.get("ts"),
    }


@router.delete("/{thread_id}")
async def delete_thread(thread_id: str) -> dict:
    saver = persistence.get_checkpointer()
    await saver.adelete_thread(thread_id)
    return {"deleted": thread_id}
