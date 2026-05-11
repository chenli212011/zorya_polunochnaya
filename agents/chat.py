"""Minimal chat agent: single LLM node, no tools.

The OpenAI model is selected per-request via LangGraph's `configurable` config:

    graph.ainvoke(state, config={"configurable": {"model": "gpt-4o-mini"}})

Falls back to `settings.default_openai_model` when no model is configured.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph

from agents.registry import register
from config import settings
from persistence import get_checkpointer

SYSTEM_PROMPT = "You are a helpful, concise assistant."


@lru_cache(maxsize=16)
def _get_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        temperature=0.7,
    )


@register("chat", description="Plain chat agent (OpenAI, model selectable per request)")
def build_chat_agent():
    if not settings.has_openai:
        raise RuntimeError("OPENAI_API_KEY is not set")

    async def chat_node(state: MessagesState, config: RunnableConfig) -> dict:
        model = (config.get("configurable") or {}).get("model") or settings.default_openai_model
        llm = _get_llm(model)
        response = await llm.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.set_finish_point("chat")
    return graph.compile(checkpointer=get_checkpointer())
