"""Agentic workflows for the Zorya app.

Each submodule defines a LangGraph workflow and registers it with the registry
via the @register("name") decorator. Importing this package eagerly imports
all known agents so the registry is populated.
"""

from agents import chat  # noqa: F401  (registers "chat")
from agents import research  # noqa: F401  (registers "research")
from agents.registry import build, list_agents, list_specs

__all__ = ["build", "list_agents", "list_specs"]
