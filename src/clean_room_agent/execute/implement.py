"""Execute function for implement passes (code generation)."""

from __future__ import annotations

import logging

from clean_room_agent.execute.dataclasses import (
    PartPlan,
    PlanStep,
    StepResult,
    ValidationResult,
)
from clean_room_agent.execute.parsers import parse_implement_response
from clean_room_agent.execute.prompts import build_implement_prompt
from clean_room_agent.llm.client import LoggedLLMClient
from clean_room_agent.retrieval.dataclasses import ContextPackage

logger = logging.getLogger(__name__)


def execute_implement(
    context: ContextPackage,
    step: PlanStep,
    llm: LoggedLLMClient,
    *,
    plan: PartPlan | None = None,
    cumulative_diff: str | None = None,
    failure_context: ValidationResult | None = None,
) -> StepResult:
    """Execute an implement pass: build prompt, call LLM, parse edits.

    Raises ValueError on parse failure — caller handles logging to raw DB.
    model_config is derived from llm.config.
    """
    system, user = build_implement_prompt(
        context, step,
        model_config=llm.config,
        plan=plan,
        cumulative_diff=cumulative_diff,
        failure_context=failure_context,
    )

    response = llm.complete(user, system=system)

    edits = parse_implement_response(response.text)
    return StepResult(
        success=True,
        edits=edits,
        raw_response=response.text,
    )


def execute_test_implement(
    context: ContextPackage,
    step: PlanStep,
    llm: LoggedLLMClient,
    *,
    test_plan: PartPlan | None = None,
    cumulative_diff: str | None = None,
    failure_context: ValidationResult | None = None,
) -> StepResult:
    """Execute a test implement pass: build prompt, call LLM, parse edits.

    Raises ValueError on parse failure — caller handles logging to raw DB.
    model_config is derived from llm.config.
    """
    system, user = build_implement_prompt(
        context, step,
        pass_type="test_implement",
        model_config=llm.config,
        plan=test_plan,
        cumulative_diff=cumulative_diff,
        failure_context=failure_context,
    )

    response = llm.complete(user, system=system)

    edits = parse_implement_response(response.text)
    return StepResult(
        success=True,
        edits=edits,
        raw_response=response.text,
    )
