"""Audit protocol data model: reference tasks, scores, findings, and results."""

from __future__ import annotations

from dataclasses import dataclass, field


VALID_TASK_TYPES = ("bug_fix", "feature", "refactor", "test", "docs", "performance")
VALID_SEVERITIES = ("high", "medium", "low")
VALID_ROOT_CAUSES = ("prompt", "data", "architecture", "model")


@dataclass
class ReferenceTask:
    """A reference task with known-correct context requirements.

    Ground truth is specified as 'what the context must contain' — not
    'what each stage should produce.'
    """
    id: str
    description: str
    task_type: str

    # Three-tier file requirements
    must_contain_files: list[str] = field(default_factory=list)
    should_contain_files: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)

    # Content-level requirements
    must_contain_information: list[str] = field(default_factory=list)

    # Budget utilization range [min%, max%]
    budget_range: tuple[int, int] = (20, 80)

    # Routing notes (non-prescriptive, for discussion only)
    routing_reasoning: str = ""

    def __post_init__(self):
        if not self.id:
            raise ValueError("Reference task id must be non-empty")
        if not self.description:
            raise ValueError("Reference task description must be non-empty")
        if self.task_type not in VALID_TASK_TYPES:
            raise ValueError(
                f"task_type must be one of {VALID_TASK_TYPES}, got {self.task_type!r}"
            )
        if not self.must_contain_files:
            raise ValueError(
                f"Reference task {self.id} must have at least one must_contain_files entry"
            )
        lo, hi = self.budget_range
        if not (0 <= lo <= hi <= 100):
            raise ValueError(
                f"budget_range must be [lo, hi] with 0 <= lo <= hi <= 100, "
                f"got [{lo}, {hi}]"
            )


@dataclass
class AuditScores:
    """Automated metric scores for a single reference task run."""
    must_contain_recall: float  # 0.0 - 1.0
    should_contain_recall: float  # 0.0 - 1.0
    exclusion_accuracy: float  # 0.0 - 1.0
    budget_utilization: float  # fraction of budget used
    budget_in_range: bool  # within expected budget_range?
    parse_success_rate: float  # 0.0 - 1.0

    @property
    def task_score(self) -> float:
        """Composite score: min(must_contain_recall, exclusion_accuracy)."""
        return min(self.must_contain_recall, self.exclusion_accuracy)

    # Detail breakdowns for diagnostics
    must_contain_present: list[str] = field(default_factory=list)
    must_contain_missing: list[str] = field(default_factory=list)
    should_contain_present: list[str] = field(default_factory=list)
    should_contain_missing: list[str] = field(default_factory=list)
    exclusion_violations: list[str] = field(default_factory=list)


@dataclass
class Finding:
    """A specific finding from an audit run."""
    description: str
    severity: str  # high, medium, low
    root_cause: str  # prompt, data, architecture, model
    detail: str
    action: str

    def __post_init__(self):
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(
                f"severity must be one of {VALID_SEVERITIES}, got {self.severity!r}"
            )
        if self.root_cause not in VALID_ROOT_CAUSES:
            raise ValueError(
                f"root_cause must be one of {VALID_ROOT_CAUSES}, got {self.root_cause!r}"
            )


@dataclass
class AuditResult:
    """Complete result of auditing one reference task."""
    audit_id: str
    task_id: str  # reference task id
    date: str
    pipeline_version: str
    model: str

    scores: AuditScores
    findings: list[Finding] = field(default_factory=list)
    summary: str = ""
    blocking: bool = False  # cannot proceed to next pass?

    # Pipeline metadata captured during run
    stages_selected: list[str] = field(default_factory=list)
    routing_reasoning: str = ""
    total_latency_ms: int = 0
    error: str = ""  # non-empty if pipeline failed entirely


@dataclass
class AuditSuiteResult:
    """Aggregate result of running the full audit suite."""
    results: list[AuditResult] = field(default_factory=list)
    pipeline_version: str = ""
    model: str = ""

    @property
    def mean_task_score(self) -> float:
        scores = [r.scores.task_score for r in self.results if not r.error]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def mean_must_recall(self) -> float:
        vals = [r.scores.must_contain_recall for r in self.results if not r.error]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def mean_exclusion_accuracy(self) -> float:
        vals = [r.scores.exclusion_accuracy for r in self.results if not r.error]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def worst_task(self) -> AuditResult | None:
        valid = [r for r in self.results if not r.error]
        if not valid:
            return None
        return min(valid, key=lambda r: r.scores.task_score)

    @property
    def failed_tasks(self) -> list[AuditResult]:
        return [r for r in self.results if r.error]

    @property
    def blocking_findings(self) -> list[AuditResult]:
        return [r for r in self.results if r.blocking]
