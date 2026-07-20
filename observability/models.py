"""Data models for AgentLens traces and spans.

A Trace represents one end-to-end task (e.g. one user query through a
multi-agent pipeline). A Span represents one LLM call made during that
trace, tagged with the agent that made it.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Span:
    """A single instrumented LLM call."""

    trace_id: str
    agent: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float
    span_id: str = field(default_factory=lambda: _new_id("span"))
    task: Optional[str] = None
    prompt_hash: Optional[str] = None
    prompt_preview: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Trace:
    """One end-to-end task, containing many spans."""

    app: str
    trace_id: str = field(default_factory=lambda: _new_id("trace"))
    name: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    # Outcome fields: set via Lens.record_outcome() to enable
    # cost-per-outcome analysis.
    outcome_success: Optional[bool] = None
    outcome_meta: Optional[dict[str, Any]] = None
