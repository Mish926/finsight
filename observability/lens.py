"""The main AgentLens entry point.

Minimal integration into an existing pipeline:

    from agentlens import Lens

    lens = Lens(app="finsight", db_path="agentlens.db")
    client = lens.wrap(Groq(api_key=...))     # drop-in replacement

    with lens.trace(name=user_question) as trace_id:
        with lens.agent("planner"):
            client.chat.completions.create(...)   # auto-attributed
        with lens.agent("retriever", task="doc_search"):
            ...
    lens.record_outcome(trace_id, success=verified_ok)

Calls made with no active trace are attached to an auto-created orphan
trace, so partial instrumentation still records data instead of failing.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from . import context
from .models import Span, Trace
from .storage import Storage
from .wrapper import WrappedClient


class Lens:
    def __init__(self, app: str = "default", db_path: str = "agentlens.db"):
        self.app = app
        self.storage = Storage(db_path)

    # -- instrumentation ---------------------------------------------------

    def wrap(self, client: Any) -> WrappedClient:
        """Wrap an OpenAI-compatible client (OpenAI SDK, Groq SDK, or
        anything exposing client.chat.completions.create with a .usage
        block on the response)."""
        return WrappedClient(client, on_span=self._record_span, ensure_trace=self._ensure_trace)

    @contextmanager
    def trace(self, name: Optional[str] = None) -> Iterator[str]:
        """Open a trace covering one end-to-end task. Yields the trace_id
        so it can be used later with record_outcome()."""
        trace = Trace(app=self.app, name=name)
        self.storage.insert_trace(trace)
        token = context.set_trace_id(trace.trace_id)
        try:
            yield trace.trace_id
        finally:
            context.reset_trace_id(token)
            self.storage.end_trace(trace.trace_id, time.time())

    @contextmanager
    def agent(self, name: str, task: Optional[str] = None) -> Iterator[None]:
        """Attribute all LLM calls inside this block to the named agent."""
        agent_token = context.set_agent(name)
        task_token = context.set_task(task) if task is not None else None
        try:
            yield
        finally:
            context.reset_agent(agent_token)
            if task_token is not None:
                context.reset_task(task_token)

    def record_outcome(
        self, trace_id: str, success: bool, meta: Optional[dict[str, Any]] = None
    ) -> None:
        """Attach a success/failure outcome to a trace. This is what turns
        cost-per-call into cost-per-outcome: e.g. wire FinSight's answer
        verifier result in here."""
        self.storage.record_outcome(trace_id, success, meta)

    # -- internals ---------------------------------------------------------

    def _ensure_trace(self) -> str:
        trace_id = context.get_trace_id()
        if trace_id is None:
            orphan = Trace(app=self.app, name="(orphan)")
            self.storage.insert_trace(orphan)
            context.set_trace_id(orphan.trace_id)
            trace_id = orphan.trace_id
        return trace_id

    def _record_span(self, span: Span) -> None:
        self.storage.insert_span(span)
