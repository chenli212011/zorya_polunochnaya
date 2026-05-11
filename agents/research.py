"""Example research agent: Anthropic Claude + Tavily web search.

Demonstrates the canonical ReAct pattern wired through LangGraph's prebuilt
`create_react_agent`. Use this as a template when adding new agents.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent

from agents.registry import register
from config import settings

SYSTEM_PROMPT = (
    "You are a careful research assistant. Use the web search tool when you "
    "need current facts, citations, or anything beyond your training. "
    "Always cite sources by URL when you use search results."
)


@register("research", description="Web-search-enabled research agent (Anthropic + Tavily)")
def build_research_agent():
    if not settings.has_anthropic:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    if not settings.has_tavily:
        raise RuntimeError("TAVILY_API_KEY is not set")

    llm = ChatAnthropic(
        model=settings.default_anthropic_model,
        api_key=settings.anthropic_api_key,
        temperature=0,
    )
    tools = [
        TavilySearch(
            max_results=5,
            tavily_api_key=settings.tavily_api_key,
        ),
    ]
    return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
