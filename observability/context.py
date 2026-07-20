"""Context propagation for AgentLens.

Uses contextvars so that the active trace, agent, and task are visible to
the client wrapper without threading them through every function call.
contextvars (unlike plain globals) are safe under threads and asyncio,
so concurrent requests in a FastAPI app do not bleed into each other's
traces.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

_current_trace_id: ContextVar[Optional[str]] = ContextVar("agentlens_trace", default=None)
_current_agent: ContextVar[Optional[str]] = ContextVar("agentlens_agent", default=None)
_current_task: ContextVar[Optional[str]] = ContextVar("agentlens_task", default=None)


def get_trace_id() -> Optional[str]:
    return _current_trace_id.get()


def get_agent() -> Optional[str]:
    return _current_agent.get()


def get_task() -> Optional[str]:
    return _current_task.get()


def set_trace_id(trace_id: Optional[str]):
    return _current_trace_id.set(trace_id)


def set_agent(agent: Optional[str]):
    return _current_agent.set(agent)


def set_task(task: Optional[str]):
    return _current_task.set(task)


def reset_trace_id(token) -> None:
    _current_trace_id.reset(token)


def reset_agent(token) -> None:
    _current_agent.reset(token)


def reset_task(token) -> None:
    _current_task.reset(token)
