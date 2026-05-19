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


def _split_content(msg) -> tuple[str, list[str]]:
    """Return (text, image_data_urls) extracted from a message's content blocks."""
    c = getattr(msg, "content", "")
    if isinstance(c, str):
        return c, []
    text_parts: list[str] = []
    images: list[str] = []
    for block in c:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            continue
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "image_url":
            url_field = block.get("image_url", {})
            url = url_field.get("url") if isinstance(url_field, dict) else str(url_field)
            if url:
                images.append(url)
        elif btype == "image":
            source = block.get("source", {}) or {}
            if source.get("type") == "base64":
                images.append(
                    f"data:{source.get('media_type', 'image/jpeg')};base64,{source.get('data', '')}"
                )
            elif source.get("type") == "url" and source.get("url"):
                images.append(source["url"])
    return "".join(text_parts), images


def _content(msg) -> str:
    return _split_content(msg)[0]


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
    out: list[dict] = []
    for m in messages:
        role = _role(m)
        if role not in ("user", "assistant"):
            continue
        text, images = _split_content(m)
        out.append({"role": role, "content": text, "images": images})
    return {
        "thread_id": thread_id,
        "messages": out,
        "ts": tup.checkpoint.get("ts"),
    }


@router.delete("/{thread_id}")
async def delete_thread(thread_id: str) -> dict:
    saver = persistence.get_checkpointer()
    await saver.adelete_thread(thread_id)
    return {"deleted": thread_id}
