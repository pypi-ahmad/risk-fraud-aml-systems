"""structured_llm_call escalation ladder, with the LLM layer mocked."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.exceptions import OutputParserException

from aml_investigator.llm import structured_llm_call
from aml_investigator.schemas import TriageDecision

GOOD = TriageDecision(priority="high", checks=["sanctions_check"], rationale="test")
FALLBACK = TriageDecision(priority="medium", checks=["profile_account"], rationale="fallback")


def _mock_llm(side_effects):
    """A ChatOllama stand-in whose with_structured_output returns queued results."""
    llm = MagicMock()
    llm.model = "mock"
    queue = list(side_effects)

    def with_structured_output(schema, method):
        runnable = MagicMock()
        effect = queue.pop(0)
        if isinstance(effect, Exception):
            runnable.invoke.side_effect = effect
        else:
            runnable.invoke.return_value = effect
        return runnable

    llm.with_structured_output.side_effect = with_structured_output
    return llm


@patch("aml_investigator.llm.get_chat_model")
def test_first_method_succeeds(get_model):
    get_model.return_value = _mock_llm([GOOD])
    outcome = structured_llm_call(TriageDecision, "s", "u", fallback=FALLBACK)
    assert outcome.value == GOOD
    assert outcome.method_used == "function_calling"
    assert outcome.attempts == 1


@patch("aml_investigator.llm.get_chat_model")
def test_escalates_to_json_schema(get_model):
    get_model.return_value = _mock_llm([OutputParserException("no tool call"), GOOD])
    outcome = structured_llm_call(TriageDecision, "s", "u", fallback=FALLBACK)
    assert outcome.value == GOOD
    assert outcome.method_used == "json_schema"
    assert outcome.attempts == 2


@patch("aml_investigator.llm.get_chat_model")
def test_falls_back_to_default(get_model):
    get_model.return_value = _mock_llm(
        [OutputParserException("bad"), OutputParserException("bad again")]
    )
    outcome = structured_llm_call(TriageDecision, "s", "u", fallback=FALLBACK)
    assert outcome.value == FALLBACK
    assert outcome.method_used == "fallback"


@patch("aml_investigator.llm.get_chat_model")
def test_none_result_treated_as_failure(get_model):
    get_model.return_value = _mock_llm([None, GOOD])
    outcome = structured_llm_call(TriageDecision, "s", "u", fallback=FALLBACK)
    assert outcome.value == GOOD
    assert outcome.method_used == "json_schema"


@patch("aml_investigator.llm.get_chat_model")
def test_raises_without_fallback(get_model):
    get_model.return_value = _mock_llm(
        [OutputParserException("bad"), OutputParserException("bad")]
    )
    with pytest.raises(RuntimeError):
        structured_llm_call(TriageDecision, "s", "u")
