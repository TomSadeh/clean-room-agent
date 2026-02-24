"""Execute function for plan passes (meta_plan, part_plan, adjustment)."""

from __future__ import annotations

import logging

from clean_room_agent.execute.dataclasses import (
    MetaPlan,
    PartPlan,
    PlanAdjustment,
    StepResult,
    ValidationResult,
)
from clean_room_agent.execute.parsers import parse_plan_response, validate_plan
from clean_room_agent.execute.prompts import build_plan_prompt
from clean_room_agent.llm.client import LoggedLLMClient
from clean_room_agent.retrieval.dataclasses import ContextPackage

logger = logging.getLogger(__name__)


def execute_plan(
    context: ContextPackage,
    task_description: str,
    llm: LoggedLLMClient,
    *,
    pass_type: str,
    cumulative_diff: str | None = None,
    prior_results: list[StepResult] | None = None,
    test_results: list[ValidationResult] | None = None,
) -> MetaPlan | PartPlan | PlanAdjustment:
    """Execute a plan pass: build prompt, call LLM, parse and validate response.

    Hard error on budget overflow, parse failure, or validation failure.
    Caller decides how to handle. model_config is derived from llm.config.
    """
    system, user = build_plan_prompt(
        context, task_description,
        pass_type=pass_type,
        model_config=llm.config,
        cumulative_diff=cumulative_diff,
        prior_results=prior_results,
        test_results=test_results,
    )

    response = llm.complete(user, system=system)

    result = parse_plan_response(response.text, pass_type)

    # Validate structure for meta_plan, part_plan, and adjustment revised_steps
    if isinstance(result, (MetaPlan, PartPlan)):
        warnings = validate_plan(result)
        if warnings:
            raise ValueError(
                f"Plan validation failed for {pass_type}: {'; '.join(warnings)}"
            )
    elif isinstance(result, PlanAdjustment) and result.revised_steps:
        # Wrap in a synthetic PartPlan for validation (cycle/dup detection)
        synthetic = PartPlan(
            part_id="_adjustment",
            task_summary="adjustment validation",
            steps=result.revised_steps,
            rationale="synthetic",
        )
        warnings = validate_plan(synthetic)
        if warnings:
            raise ValueError(
                f"Adjustment validation failed: {'; '.join(warnings)}"
            )

    return result
