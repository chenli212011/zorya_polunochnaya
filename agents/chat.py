"""Chat agent: single LLM node, no tools.

Provider and model are selected per-request via LangGraph's `configurable` config:

    graph.ainvoke(
        state,
        config={"configurable": {"provider": "anthropic", "model": "claude-sonnet-4-5"}},
    )

`provider` is `"openai"` (default) or `"anthropic"`. The latter is used for
image-bearing turns so vision-capable Claude models can handle them.
Multimodal content is passed through unchanged — LangChain's provider adapters
translate `image_url` content blocks to each backend's native shape.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph

from agents.registry import register
from config import settings
from persistence import get_checkpointer

SYSTEM_PROMPT = "You are a helpful, concise assistant."


@lru_cache(maxsize=16)
def _get_openai_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        temperature=0.7,
    )


@lru_cache(maxsize=16)
def _get_anthropic_llm(model: str) -> ChatAnthropic:
    return ChatAnthropic(
        model=model,
        api_key=settings.anthropic_api_key,
        temperature=0.7,
    )


@register("chat", description="Plain chat agent (OpenAI text / Anthropic vision, selectable per request)")
def build_chat_agent():
    if not (settings.has_openai or settings.has_anthropic):
        raise RuntimeError("Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set")

    async def chat_node(state: MessagesState, config: RunnableConfig) -> dict:
        cfg = (config.get("configurable") or {})
        provider = (cfg.get("provider") or "openai").lower()

        if provider == "anthropic":
            if not settings.has_anthropic:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            model = cfg.get("model") or settings.default_anthropic_model
            llm = _get_anthropic_llm(model)
        else:
            if not settings.has_openai:
                raise RuntimeError("OPENAI_API_KEY is not set")
            model = cfg.get("model") or settings.default_openai_model
            llm = _get_openai_llm(model)

        response = await llm.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.set_finish_point("chat")
    return graph.compile(checkpointer=get_checkpointer())
