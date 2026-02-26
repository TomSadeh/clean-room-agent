"""Prompt construction for plan and implement execute stages."""

from __future__ import annotations

from clean_room_agent.execute.dataclasses import (
    PartPlan,
    PlanStep,
    StepResult,
    ValidationResult,
)
from clean_room_agent.llm.client import ModelConfig
from clean_room_agent.retrieval.dataclasses import ContextPackage
from clean_room_agent.token_estimation import CHARS_PER_TOKEN_CONSERVATIVE


# -- System prompts (data-driven lookup) --

SYSTEM_PROMPTS: dict[str, str] = {
    "meta_plan": (
        "You are Jane, a task decomposition planner. Given a codebase context and task description, "
        "decompose the task into independent parts that can be implemented sequentially.\n\n"
        "Output a JSON object with exactly these fields:\n"
        "- task_summary: string — concise summary of the task\n"
        "- parts: array of objects, each with:\n"
        "  - id: string — unique identifier (e.g. \"p1\", \"p2\")\n"
        "  - description: string — what this part accomplishes\n"
        "  - affected_files: array of file paths this part will modify\n"
        "  - depends_on: array of part IDs this part depends on (empty if independent)\n"
        "- rationale: string — why this decomposition was chosen\n\n"
        "Rules:\n"
        "- Do not include code in plans — only describe what changes to make\n"
        "- Explicit dependency edges are mandatory — if part B requires part A's changes, list A in depends_on\n"
        "- Each part should be independently testable\n"
        "- Output only valid JSON, no markdown fencing or extra text"
    ),
    "part_plan": (
        "You are Jane, a detailed step planner. Given a codebase context and a specific part description, "
        "break the part into small implementation steps.\n\n"
        "Output a JSON object with exactly these fields:\n"
        "- part_id: string — the ID of the part being planned\n"
        "- task_summary: string — concise summary of this part's goal\n"
        "- steps: array of objects, each with:\n"
        "  - id: string — unique identifier (e.g. \"s1\", \"s2\")\n"
        "  - description: string — what this step accomplishes\n"
        "  - target_files: array of file paths this step will modify\n"
        "  - target_symbols: array of function/class names to modify\n"
        "  - depends_on: array of step IDs this step depends on\n"
        "- rationale: string — why this step breakdown was chosen\n\n"
        "Rules:\n"
        "- Steps must be small enough for reliable single-pass code generation\n"
        "- Each step should modify at most 2-3 files\n"
        "- Do not include code — only describe the changes\n"
        "- Output only valid JSON"
    ),
    "test_plan": (
        "You are Jane, a test planner. Given a codebase context and code changes, "
        "plan test coverage for all changed and new functions/methods.\n\n"
        "Output a JSON object with exactly these fields:\n"
        "- part_id: string — the ID of the part being tested (e.g. \"p1_tests\")\n"
        "- task_summary: string — concise summary of what tests will cover\n"
        "- steps: array of objects, each with:\n"
        "  - id: string — unique identifier (e.g. \"t1\", \"t2\")\n"
        "  - description: string — what behavior to verify, what assertions to make\n"
        "  - target_files: array of test file paths to create or modify\n"
        "  - target_symbols: array of function/method names under test\n"
        "  - depends_on: array of step IDs this step depends on\n"
        "- rationale: string — why this test breakdown was chosen\n\n"
        "Rules:\n"
        "- Cover all changed and new functions/methods, not just happy paths\n"
        "- Include edge cases and error paths\n"
        "- Each step should test one logical behavior group\n"
        "- Follow the project's existing test file naming conventions\n"
        "- Do not include code — only describe what to test\n"
        "- Output only valid JSON"
    ),
    "adjustment": (
        "You are Jane, a plan reviewer. Given test results and prior changes, revise the remaining "
        "implementation steps.\n\n"
        "Output a JSON object with exactly these fields:\n"
        "- revised_steps: array of step objects (same format as part plan steps)\n"
        "- rationale: string — why these adjustments were made\n"
        "- changes_made: array of strings — what was changed from the original plan\n\n"
        "Rules:\n"
        "- Cannot undo completed steps — only revise remaining steps\n"
        "- If all remaining steps are still valid, return them unchanged with rationale explaining why\n"
        "- Output only valid JSON"
    ),
    "implement": (
        "You are Jane, a code editor. Given a codebase context and a specific step to implement, "
        "produce search/replace edits.\n\n"
        "Output one or more edit blocks in this exact format:\n"
        "<edit file=\"path/to/file.py\">\n"
        "<search>\nexact text to find\n</search>\n"
        "<replacement>\nnew text to replace it with\n</replacement>\n"
        "</edit>\n\n"
        "Rules:\n"
        "- Search text must match the file content EXACTLY (including whitespace and indentation)\n"
        "- Make minimal changes — only modify what the step requires\n"
        "- All edits must be in one response\n"
        "- For deletions, use an empty <replacement></replacement>\n"
        "- For new code insertion, use a search string that matches the insertion point context"
    ),
    "test_implement": (
        "You are Jane, a test code editor. Given a codebase context and a test step to implement, "
        "produce search/replace edits that create or modify test code.\n\n"
        "Output one or more edit blocks in this exact format:\n"
        "<edit file=\"path/to/test_file.py\">\n"
        "<search>\nexact text to find\n</search>\n"
        "<replacement>\nnew text to replace it with\n</replacement>\n"
        "</edit>\n\n"
        "Rules:\n"
        "- Search text must match the file content EXACTLY (including whitespace and indentation)\n"
        "- Follow the project's existing test framework and conventions\n"
        "- Import functions under test correctly\n"
        "- Each test function should test one behavior\n"
        "- All edits must be in one response\n"
        "- For new test files, empty <search></search> is not allowed; "
        "use a search string that matches the insertion point or create file content\n"
        "- For new code insertion, use a search string that matches the insertion point context"
    ),
    "documentation": (
        "You are Jane, a documentation specialist. Given a source file and its task context, "
        "improve docstrings and inline comments without changing any code logic.\n\n"
        "Output one or more edit blocks in this exact format:\n"
        "<edit file=\"path/to/file.py\">\n"
        "<search>\nexact text to find\n</search>\n"
        "<replacement>\nnew text to replace it with\n</replacement>\n"
        "</edit>\n\n"
        "Rules:\n"
        "- ONLY modify docstrings and # comments — never change code logic, signatures, or imports\n"
        "- Search text must match the file content EXACTLY (including whitespace and indentation)\n"
        "- Match the existing docstring format in the file (Google, NumPy, or Sphinx style); "
        "default to Google style if no convention is established\n"
        "- Do not over-comment obvious code — focus on non-obvious logic, edge cases, and intent\n"
        "- Add module-level docstrings if missing\n"
        "- Add or improve function/class docstrings to describe purpose, args, returns, and raises\n"
        "- All edits must be in one response\n"
        "- If the file already has good documentation, output no edit blocks"
    ),
}

_PLAN_PASS_TYPES = frozenset({"meta_plan", "part_plan", "test_plan", "adjustment"})

_IMPLEMENT_STEP_HEADERS: dict[str, str] = {
    "implement": "Step to Implement",
    "test_implement": "Test Step to Implement",
}


def _estimate_prompt_tokens(system: str, user: str) -> int:
    """Conservative token estimate for a (system, user) prompt pair."""
    return (len(system) + len(user)) // CHARS_PER_TOKEN_CONSERVATIVE


def _validate_prompt_budget(system: str, user: str, model_config: ModelConfig) -> None:
    """R3: Validate prompt fits within model context window."""
    estimated = _estimate_prompt_tokens(system, user)
    available = model_config.context_window - model_config.max_tokens
    if estimated > available:
        raise ValueError(
            f"Prompt too large for model context window: ~{estimated} tokens estimated, "
            f"but only {available} tokens available "
            f"(context_window={model_config.context_window}, "
            f"max_tokens={model_config.max_tokens}). "
            f"Reduce context or split the task."
        )


def build_plan_prompt(
    context: ContextPackage,
    task_description: str,
    *,
    pass_type: str,
    model_config: ModelConfig,
    cumulative_diff: str | None = None,
    prior_results: list[StepResult] | None = None,
    test_results: list[ValidationResult] | None = None,
) -> tuple[str, str]:
    """Build system and user prompts for a plan pass.

    Returns:
        (system_prompt, user_prompt)

    Raises:
        ValueError: If pass_type is unknown or prompt exceeds budget.
    """
    if pass_type not in _PLAN_PASS_TYPES:
        raise ValueError(f"Unknown plan pass_type: {pass_type!r}")
    system = SYSTEM_PROMPTS[pass_type]

    # Build user prompt
    # Note: ContextPackage.to_prompt_text() already includes # Task section.
    # Only add task_description if it differs from context.task.raw_task (e.g.
    # part descriptions that differ from the top-level task).
    parts = [context.to_prompt_text()]
    if task_description != context.task.raw_task:
        parts.append(f"\n# Current Objective\n{task_description}\n")

    if cumulative_diff:
        parts.append(f"\n<prior_changes>\n{cumulative_diff}\n</prior_changes>\n")

    if prior_results:
        completed_section = "\n<completed_steps>\n"
        for i, sr in enumerate(prior_results):
            status = "success" if sr.success else "failed"
            completed_section += f"Step {i+1}: {status}"
            if sr.error_info:
                completed_section += f" — {sr.error_info}"
            completed_section += "\n"
        completed_section += "</completed_steps>\n"
        parts.append(completed_section)

    if test_results:
        test_section = "\n<test_results>\n"
        for i, tr in enumerate(test_results):
            status = "passed" if tr.success else "failed"
            test_section += f"Validation {i+1}: {status}\n"
            if tr.test_output:
                test_section += f"Test output: {tr.test_output}\n"
            if tr.failing_tests:
                test_section += f"Failing tests: {', '.join(tr.failing_tests)}\n"
        test_section += "</test_results>\n"
        parts.append(test_section)

    user = "".join(parts)

    # R3: Budget validation
    _validate_prompt_budget(system, user, model_config)

    return system, user


def _assemble_implement_user_prompt(
    context: ContextPackage,
    step: PlanStep,
    *,
    step_header: str = "Step to Implement",
    plan: PartPlan | None = None,
    cumulative_diff: str | None = None,
    failure_context: ValidationResult | None = None,
) -> str:
    """Assemble the user prompt for an implement-style pass.

    Shared by build_implement_prompt and build_test_implement_prompt.
    """
    parts = [context.to_prompt_text()]
    parts.append(f"\n# {step_header}\nID: {step.id}\n{step.description}\n")

    if step.target_files:
        parts.append(f"Target files: {', '.join(step.target_files)}\n")
    if step.target_symbols:
        parts.append(f"Target symbols: {', '.join(step.target_symbols)}\n")

    if plan:
        constraints = f"\n<plan_constraints>\nPart: {plan.part_id}\nGoal: {plan.task_summary}\n"
        constraints += "Steps in this part:\n"
        for s in plan.steps:
            marker = " (current)" if s.id == step.id else ""
            constraints += f"  - {s.id}: {s.description}{marker}\n"
        constraints += "</plan_constraints>\n"
        parts.append(constraints)

    if cumulative_diff:
        parts.append(f"\n<prior_changes>\n{cumulative_diff}\n</prior_changes>\n")

    if failure_context:
        fail_section = "\n<test_failures>\n"
        if failure_context.test_output:
            fail_section += f"Test output:\n{failure_context.test_output}\n"
        if failure_context.failing_tests:
            fail_section += f"Failing tests: {', '.join(failure_context.failing_tests)}\n"
        if failure_context.lint_output:
            fail_section += f"Lint output:\n{failure_context.lint_output}\n"
        if failure_context.type_check_output:
            fail_section += f"Type check output:\n{failure_context.type_check_output}\n"
        fail_section += "</test_failures>\n"
        parts.append(fail_section)

    return "".join(parts)


def build_implement_prompt(
    context: ContextPackage,
    step: PlanStep,
    *,
    pass_type: str = "implement",
    model_config: ModelConfig,
    plan: PartPlan | None = None,
    cumulative_diff: str | None = None,
    failure_context: ValidationResult | None = None,
) -> tuple[str, str]:
    """Build system and user prompts for an implement or test-implement pass.

    Args:
        pass_type: "implement" or "test_implement".

    Returns:
        (system_prompt, user_prompt)

    Raises:
        ValueError: If pass_type is unknown or prompt exceeds budget.
    """
    if pass_type not in _IMPLEMENT_STEP_HEADERS:
        raise ValueError(f"Unknown implement pass_type: {pass_type!r}")

    system = SYSTEM_PROMPTS[pass_type]
    user = _assemble_implement_user_prompt(
        context, step,
        step_header=_IMPLEMENT_STEP_HEADERS[pass_type],
        plan=plan,
        cumulative_diff=cumulative_diff,
        failure_context=failure_context,
    )

    # R3: Budget validation
    _validate_prompt_budget(system, user, model_config)

    return system, user


def build_documentation_prompt(
    file_content: str,
    file_path: str,
    task_description: str,
    part_description: str,
    model_config: ModelConfig,
    environment_brief: str | None = None,
) -> tuple[str, str]:
    """Build system and user prompts for a documentation pass on a single file.

    Returns:
        (system_prompt, user_prompt)

    Raises:
        ValueError: If prompt exceeds budget.
    """
    system = SYSTEM_PROMPTS["documentation"]

    parts = []
    if environment_brief:
        parts.append(f"{environment_brief}\n")

    parts.append(f"# File: {file_path}\n")
    parts.append(f'<code lang="{_lang_from_path(file_path)}">\n{file_content}\n</code>\n')
    parts.append(f"\n# Task Context\nTask: {task_description}\nPart: {part_description}\n")

    user = "".join(parts)

    # R3: Budget validation
    _validate_prompt_budget(system, user, model_config)

    return system, user


def _lang_from_path(file_path: str) -> str:
    """Infer language identifier from file extension."""
    if file_path.endswith(".py"):
        return "python"
    if file_path.endswith((".ts", ".tsx")):
        return "typescript"
    if file_path.endswith((".js", ".jsx")):
        return "javascript"
    return "text"
