"""Lightweight per-case telemetry.

A context-local collector that tools, structured LLM calls, and the LangChain
callback handler all append to. The eval runner resets it per case and persists
the events — this is where tool-call validity rates, retry counts, latencies,
and token usage come from.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


@dataclass
class TelemetryEvent:
    kind: str  # llm | structured_llm | tool
    name: str
    seconds: float
    ok: bool = True
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Telemetry:
    case_id: str = "interactive"
    events: list[TelemetryEvent] = field(default_factory=list)

    def record(self, kind: str, name: str, seconds: float, ok: bool = True, **meta: Any) -> None:
        self.events.append(TelemetryEvent(kind, name, round(seconds, 3), ok, meta))

    def summary(self) -> dict[str, Any]:
        llm = [e for e in self.events if e.kind in ("llm", "structured_llm")]
        tools = [e for e in self.events if e.kind == "tool"]
        return {
            "case_id": self.case_id,
            "llm_calls": len(llm),
            "llm_seconds": round(sum(e.seconds for e in llm), 1),
            "prompt_tokens": sum(e.meta.get("input_tokens", 0) for e in llm),
            "completion_tokens": sum(e.meta.get("output_tokens", 0) for e in llm),
            "tool_calls": len(tools),
            "tool_errors": sum(1 for e in tools if not e.ok),
            "structured_retries": sum(
                max(0, e.meta.get("attempts", 1) - 1) for e in self.events if e.kind == "structured_llm"
            ),
            "structured_fallbacks": sum(
                1 for e in self.events if e.kind == "structured_llm" and e.meta.get("fallback_used")
            ),
        }


# Module-level (not a ContextVar): LangGraph may run nodes/tools in worker
# threads where ContextVar values don't propagate. Cases run sequentially.
_state: dict[str, Telemetry | None] = {"current": None}
_default = Telemetry()


def current() -> Telemetry:
    return _state["current"] or _default


@contextmanager
def case_scope(case_id: str):
    """Collect telemetry for one case; yields the collector."""
    t = Telemetry(case_id=case_id)
    previous = _state["current"]
    _state["current"] = t
    try:
        yield t
    finally:
        _state["current"] = previous


@contextmanager
def timed(kind: str, name: str, **meta: Any):
    """Time a block and record it; marks ok=False if the block raises."""
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        current().record(kind, name, time.perf_counter() - t0, ok=False, **meta)
        raise
    current().record(kind, name, time.perf_counter() - t0, ok=True, **meta)


class LLMTimingHandler(BaseCallbackHandler):
    """Records every chat-model call made inside a graph invocation (the ReAct loop)."""

    def __init__(self) -> None:
        self._starts: dict[Any, float] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        self._starts[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        t0 = self._starts.pop(run_id, None)
        usage: dict[str, Any] = {}
        try:
            msg = response.generations[0][0].message
            usage = msg.usage_metadata or {}
        except (AttributeError, IndexError):
            pass
        current().record(
            "llm", "chat", time.perf_counter() - t0 if t0 else 0.0,
            input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
        )

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        t0 = self._starts.pop(run_id, None)
        current().record("llm", "chat", time.perf_counter() - t0 if t0 else 0.0, ok=False)
