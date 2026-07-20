"""Framework-agnostic LLM client instrumentation.

Instead of hooking a specific framework (LangChain callbacks, etc.),
AgentLens wraps the client object itself. Groq's SDK is intentionally
OpenAI-compatible -- both expose client.chat.completions.create(...) and
return a response with a .usage block -- so one wrapper covers both, and
any pipeline that calls the client directly (like FinSight does) is
instrumentable with zero changes to pipeline code.

The wrapper is a transparent attribute proxy: any attribute other than
the intercepted call path passes straight through, so wrapped clients
keep working for everything else they do.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable

from . import context
from .models import Span
from .pricing import estimate_cost

_PREVIEW_CHARS = 200


def _hash_messages(messages: Any) -> tuple[str, str]:
    """Stable hash + short preview of a messages array, for redundancy
    detection. Only message content is hashed, not sampling params."""
    try:
        canonical = json.dumps(messages, sort_keys=True, default=str)
    except (TypeError, ValueError):
        canonical = str(messages)
    digest = hashlib.sha256(canonical.encode("utf-8", errors="replace")).hexdigest()[:16]
    # Preview: last user-ish content, most useful for eyeballing a report.
    preview = ""
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict):
            preview = str(last.get("content", ""))[:_PREVIEW_CHARS]
        else:
            preview = str(last)[:_PREVIEW_CHARS]
    return digest, preview


class _Proxy:
    """Transparent attribute proxy."""

    def __init__(self, target: Any):
        object.__setattr__(self, "_target", target)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_target"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_target"), name, value)


class _CompletionsProxy(_Proxy):
    def __init__(self, target: Any, on_span: Callable[[Span], None], ensure_trace: Callable[[], str]):
        super().__init__(target)
        object.__setattr__(self, "_on_span", on_span)
        object.__setattr__(self, "_ensure_trace", ensure_trace)

    def create(self, *args: Any, **kwargs: Any) -> Any:
        target = object.__getattribute__(self, "_target")
        on_span = object.__getattribute__(self, "_on_span")
        ensure_trace = object.__getattribute__(self, "_ensure_trace")

        trace_id = ensure_trace()
        agent = context.get_agent() or "unattributed"
        task = context.get_task()
        model = kwargs.get("model", "unknown")
        prompt_hash, preview = _hash_messages(kwargs.get("messages"))

        start = time.perf_counter()
        started_at = time.time()
        error: str | None = None
        response = None
        try:
            response = target.create(*args, **kwargs)
            return response
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            prompt_tokens = completion_tokens = 0
            usage = getattr(response, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            resolved_model = getattr(response, "model", None) or model
            span = Span(
                trace_id=trace_id,
                agent=agent,
                task=task,
                model=resolved_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                cost_usd=estimate_cost(resolved_model, prompt_tokens, completion_tokens),
                prompt_hash=prompt_hash,
                prompt_preview=preview,
                started_at=started_at,
                error=error,
            )
            on_span(span)


class _ChatProxy(_Proxy):
    def __init__(self, target: Any, on_span: Callable, ensure_trace: Callable):
        super().__init__(target)
        object.__setattr__(
            self, "_completions", _CompletionsProxy(target.completions, on_span, ensure_trace)
        )

    @property
    def completions(self) -> _CompletionsProxy:
        return object.__getattribute__(self, "_completions")


class WrappedClient(_Proxy):
    """A wrapped OpenAI-compatible client. Everything passes through
    untouched except chat.completions.create, which is instrumented."""

    def __init__(self, target: Any, on_span: Callable, ensure_trace: Callable):
        super().__init__(target)
        object.__setattr__(self, "_chat", _ChatProxy(target.chat, on_span, ensure_trace))

    @property
    def chat(self) -> _ChatProxy:
        return object.__getattribute__(self, "_chat")
