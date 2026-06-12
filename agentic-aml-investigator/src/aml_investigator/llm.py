"""The reliability core: every structured LLM output goes through one hardened path.

Measured on this machine (RTX 4060 Laptop, 8 GB VRAM, Ollama 0.30.6):

- ``method="function_calling"`` (tool-call based extraction) is the fastest and
  most reliable path on granite4.1:8b (~4 s) and qwen3.5:9b (~35 s).
- ``method="json_schema"`` (Ollama grammar-constrained decoding) always yields
  parseable JSON but is slower, and on thinking models only works when thinking
  is left ENABLED — setting ``think=false`` silently disables ``format=``
  constrained decoding on this Ollama version.

So: try function_calling, escalate to json_schema, then fall back to a
deterministic default. No LLM output is ever consumed unvalidated.
"""

import time
from dataclasses import dataclass

from langchain_core.exceptions import OutputParserException
from langchain_ollama import ChatOllama
from loguru import logger
from pydantic import BaseModel

from aml_investigator import telemetry
from aml_investigator.settings import settings


def get_chat_model(model: str | None = None, **overrides) -> ChatOllama:
    """ChatOllama with project defaults (deterministic, fixed context budget)."""
    params: dict = {
        "model": model or settings.agent_model,
        "base_url": settings.ollama_base_url,
        "temperature": settings.temperature,
        "num_ctx": settings.num_ctx,
        # timeout reaches the underlying ollama httpx client; without it a hung
        # generation blocks the whole pipeline indefinitely (observed on long
        # qwen json_schema calls).
        "client_kwargs": {"timeout": settings.request_timeout},
    }
    params.update(overrides)
    return ChatOllama(**params)


@dataclass
class StructuredCallOutcome:
    """Validated result plus how hard we had to work to get it."""

    value: BaseModel
    attempts: int
    method_used: str  # function_calling | json_schema | fallback
    seconds: float


def structured_llm_call(
    schema: type[BaseModel],
    system: str,
    user: str,
    *,
    model: str | None = None,
    fallback: BaseModel | None = None,
    name: str = "structured_call",
) -> StructuredCallOutcome:
    """Call the LLM and return a validated instance of ``schema``, or ``fallback``.

    Escalation ladder: function_calling -> json_schema -> fallback. Each retry
    appends the previous failure to the prompt so the model can correct itself.

    Raises:
        RuntimeError: if every method fails and no ``fallback`` was provided.
    """
    llm = get_chat_model(model)
    messages = [("system", system), ("user", user)]
    t0 = time.perf_counter()
    last_error: Exception | None = None

    for attempt, method in enumerate(("function_calling", "json_schema"), start=1):
        try:
            structured = llm.with_structured_output(schema, method=method)
            result = structured.invoke(messages)
            if result is None:  # function_calling can yield None when no tool call is emitted
                raise OutputParserException("model returned no structured payload")
            seconds = time.perf_counter() - t0
            telemetry.current().record(
                "structured_llm", name, seconds, attempts=attempt, method=method, model=llm.model
            )
            return StructuredCallOutcome(result, attempt, method, round(seconds, 2))
        except Exception as e:
            # Broad on purpose: parse/validation errors AND transport faults (timeouts,
            # connection drops). Every failure mode escalates then falls back, so a
            # flaky local model degrades quality but never blocks or crashes the graph.
            last_error = e
            logger.warning(f"{name}: failed ({type(e).__name__}: {str(e)[:120]}), escalating")
            messages = [
                ("system", system),
                ("user", f"{user}\n\nYour previous answer was invalid: {str(e)[:400]}\n"
                         "Answer again, strictly matching the required schema."),
            ]

    seconds = time.perf_counter() - t0
    telemetry.current().record(
        "structured_llm", name, seconds, ok=fallback is not None,
        attempts=2, method="fallback", fallback_used=True, model=llm.model,
    )
    if fallback is not None:
        logger.error(f"{name}: all structured methods failed, using deterministic fallback")
        return StructuredCallOutcome(fallback, 3, "fallback", round(seconds, 2))
    raise RuntimeError(f"{name}: structured call failed with no fallback: {last_error}")
