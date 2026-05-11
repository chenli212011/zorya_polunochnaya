"""Lightweight registry for LangGraph workflows.

Each agent is a zero-arg builder that returns a compiled LangGraph graph.
Builders are invoked lazily on first use and the result is cached so we don't
re-instantiate models per request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

GraphBuilder = Callable[[], Any]


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    builder: GraphBuilder


_REGISTRY: dict[str, AgentSpec] = {}
_CACHE: dict[str, Any] = {}


def register(name: str, description: str = "") -> Callable[[GraphBuilder], GraphBuilder]:
    def decorator(builder: GraphBuilder) -> GraphBuilder:
        if name in _REGISTRY:
            raise ValueError(f"agent already registered: {name}")
        _REGISTRY[name] = AgentSpec(name=name, description=description, builder=builder)
        return builder

    return decorator


def list_agents() -> list[str]:
    return sorted(_REGISTRY.keys())


def list_specs() -> list[AgentSpec]:
    return [_REGISTRY[n] for n in list_agents()]


def build(name: str) -> Any:
    if name not in _REGISTRY:
        raise KeyError(name)
    if name not in _CACHE:
        _CACHE[name] = _REGISTRY[name].builder()
    return _CACHE[name]
