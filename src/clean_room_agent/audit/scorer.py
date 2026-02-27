"""Automated metrics computation for retrieval audit."""

from __future__ import annotations

import fnmatch
import logging

from clean_room_agent.audit.dataclasses import AuditScores, ReferenceTask
from clean_room_agent.retrieval.dataclasses import ContextPackage

logger = logging.getLogger(__name__)


def score_retrieval(
    ref_task: ReferenceTask,
    package: ContextPackage,
    parse_success_rate: float = 1.0,
) -> AuditScores:
    """Compute automated metrics for a retrieval run against a reference task.

    Args:
        ref_task: The reference task with known-correct context requirements.
        package: The ContextPackage produced by the retrieval pipeline.
        parse_success_rate: Fraction of LLM calls that produced parseable output
                          (computed externally from raw DB logs).

    Returns:
        AuditScores with all automated metrics populated.
    """
    context_paths = {fc.path for fc in package.files}

    # Must-contain recall
    must_present = []
    must_missing = []
    for required_path in ref_task.must_contain_files:
        if _path_in_context(required_path, context_paths):
            must_present.append(required_path)
        else:
            must_missing.append(required_path)

    must_recall = (
        len(must_present) / len(ref_task.must_contain_files)
        if ref_task.must_contain_files
        else 1.0
    )

    # Should-contain recall
    should_present = []
    should_missing = []
    for desired_path in ref_task.should_contain_files:
        if _path_in_context(desired_path, context_paths):
            should_present.append(desired_path)
        else:
            should_missing.append(desired_path)

    should_recall = (
        len(should_present) / len(ref_task.should_contain_files)
        if ref_task.should_contain_files
        else 1.0
    )

    # Exclusion accuracy
    exclusion_violations = []
    for excluded_pattern in ref_task.must_not_contain:
        for ctx_path in context_paths:
            if _matches_exclusion(ctx_path, excluded_pattern):
                exclusion_violations.append(ctx_path)

    exclusion_total = len(ref_task.must_not_contain)
    if exclusion_total > 0:
        # Count how many exclusion patterns were violated (at least one match)
        violated_patterns = sum(
            1 for pattern in ref_task.must_not_contain
            if any(_matches_exclusion(p, pattern) for p in context_paths)
        )
        exclusion_accuracy = 1.0 - (violated_patterns / exclusion_total)
    else:
        exclusion_accuracy = 1.0

    # Budget utilization
    if package.budget is not None:
        budget_available = package.budget.effective_budget
        budget_utilization = (
            package.total_token_estimate / budget_available
            if budget_available > 0
            else 0.0
        )
    else:
        budget_utilization = 0.0

    lo, hi = ref_task.budget_range
    budget_pct = budget_utilization * 100
    budget_in_range = lo <= budget_pct <= hi

    return AuditScores(
        must_contain_recall=must_recall,
        should_contain_recall=should_recall,
        exclusion_accuracy=exclusion_accuracy,
        budget_utilization=budget_utilization,
        budget_in_range=budget_in_range,
        parse_success_rate=parse_success_rate,
        must_contain_present=must_present,
        must_contain_missing=must_missing,
        should_contain_present=should_present,
        should_contain_missing=should_missing,
        exclusion_violations=exclusion_violations,
    )


def _path_in_context(required_path: str, context_paths: set[str]) -> bool:
    """Check if a required path is present in context.

    Supports both exact match and glob patterns. A required path like
    'src/foo/bar.py' matches exactly. A pattern like 'src/foo/*.py'
    matches any file in that directory.
    """
    # Exact match first
    if required_path in context_paths:
        return True

    # Glob match (for patterns in must_contain_files)
    if any(fnmatch.fnmatch(ctx_path, required_path) for ctx_path in context_paths):
        return True

    return False


def _matches_exclusion(path: str, pattern: str) -> bool:
    """Check if a context path matches an exclusion pattern.

    Supports glob patterns: 'src/foo/*' matches anything under src/foo/.
    """
    return fnmatch.fnmatch(path, pattern)
