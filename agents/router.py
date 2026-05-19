"""FastAPI router exposing registered LangGraph agents as HTTP endpoints."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field, model_validator

from agents import registry

logger = logging.getLogger("zorya.agents")

router = APIRouter(prefix="/api/agents", tags=["agents"])


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class AgentRunRequest(BaseModel):
    input: str | None = Field(
        default=None, description="Single user message (use with thread_id for stateful chats).",
    )
    images: list[str] | None = Field(
        default=None,
        description="Data-URL images attached to this turn (e.g. 'data:image/jpeg;base64,...').",
    )
    messages: list[ChatMessage] | None = Field(
        default=None, description="Full conversation history (stateless mode).",
    )
    model: str | None = Field(
        default=None,
        description="Override the agent's default model (passed via configurable).",
    )
    provider: Literal["openai", "anthropic"] | None = Field(
        default=None,
        description="LLM provider to use for this turn (passed via configurable).",
    )
    thread_id: str | None = Field(
        default=None,
        description="Persistent thread ID. Required when the agent has a checkpointer attached.",
    )

    @model_validator(mode="after")
    def _validate(self):
        if not self.input and not self.messages and not self.images:
            raise ValueError("provide 'input', 'images', or 'messages'")
        return self

    def _multimodal_human(self) -> HumanMessage:
        blocks: list[dict] = []
        if self.input:
            blocks.append({"type": "text", "text": self.input})
        for url in self.images or []:
            blocks.append({"type": "image_url", "image_url": {"url": url}})
        return HumanMessage(content=blocks)

    def to_graph_input(self) -> dict:
        # When a thread_id is set, only the new user message is sent — the
        # checkpointer replays prior state and MessagesState's reducer appends.
        if self.images:
            return {"messages": [self._multimodal_human()]}
        if self.thread_id and self.input:
            return {"messages": [("user", self.input)]}
        if self.messages:
            return {"messages": [(m.role, m.content) for m in self.messages]}
        return {"messages": [("user", self.input or "")]}

    def to_runnable_config(self, require_thread: bool = False) -> dict:
        configurable: dict = {}
        if self.thread_id:
            configurable["thread_id"] = self.thread_id
        elif require_thread:
            raise HTTPException(
                status_code=400,
                detail="this agent has a checkpointer attached and requires a thread_id",
            )
        if self.model:
            configurable["model"] = self.model
        if self.provider:
            configurable["provider"] = self.provider
        return {"configurable": configurable}


def _final_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            if isinstance(msg.content, str):
                return msg.content
            return "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in msg.content
            )
    return ""


def _load(name: str):
    try:
        return registry.build(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown agent: {name}")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("")
async def list_agents() -> dict[str, Any]:
    return {
        "agents": [
            {"name": s.name, "description": s.description}
            for s in registry.list_specs()
        ]
    }


@router.post("/{name}/run")
async def run_agent(name: str, body: AgentRunRequest) -> dict[str, Any]:
    graph = _load(name)
    require_thread = getattr(graph, "checkpointer", None) is not None
    config = body.to_runnable_config(require_thread=require_thread)
    logger.info("agent.run name=%s model=%s thread_id=%s", name, body.model, body.thread_id)
    result = await graph.ainvoke(body.to_graph_input(), config=config)
    return {"output": _final_text(result["messages"]), "thread_id": body.thread_id}


@router.post("/{name}/stream")
async def stream_agent(
    name: str,
    body: AgentRunRequest,
    mode: Literal["updates", "messages"] = Query(
        "updates",
        description="`updates` emits per-node state changes; `messages` emits LLM tokens.",
    ),
) -> StreamingResponse:
    graph = _load(name)
    require_thread = getattr(graph, "checkpointer", None) is not None
    run_config = body.to_runnable_config(require_thread=require_thread)
    logger.info(
        "agent.stream name=%s mode=%s model=%s thread_id=%s",
        name, mode, body.model, body.thread_id,
    )
    graph_input = body.to_graph_input()

    async def event_stream():
        try:
            if mode == "messages":
                async for chunk, _meta in graph.astream(
                    graph_input, stream_mode="messages", config=run_config,
                ):
                    text = chunk.content if isinstance(chunk.content, str) else "".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in chunk.content
                    )
                    if text:
                        yield f"data: {json.dumps({'type': 'token', 'content': text})}\n\n"
            else:
                async for chunk in graph.astream(
                    graph_input, stream_mode="updates", config=run_config,
                ):
                    for node, update in chunk.items():
                        msgs = update.get("messages", []) if isinstance(update, dict) else []
                        payload = {
                            "type": "update",
                            "node": node,
                            "messages": [
                                {
                                    "type": m.__class__.__name__,
                                    "content": (
                                        m.content if isinstance(m.content, str) else str(m.content)
                                    ),
                                }
                                for m in msgs
                            ],
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
        except Exception as exc:
            logger.exception("agent stream failed")
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
